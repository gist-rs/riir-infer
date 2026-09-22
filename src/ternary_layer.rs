//! Ternary-weight transformer structs (Plan 333 T2.2).
//!
//! Mirrors [`crate::llama_layer`] — the architecture is the same LLaMA/Qwen
//! shape (`RMSNorm` → GQA + `RoPE` → `SwiGLU`, separate `lm_head`). The single
//! difference is the **weight container**: the 7 projection matrices are
//! ternary `{-1, 0, +1}` with a per-128-weight f16 scale
//! ([`TernaryGroupWeights`], katgpt-rs Issue 578) instead of dense `Vec<f32>`.
//!
//! Named `ternary`, not `bitnet` (2026-08-10): the format is `Q2_0_g128`, not
//! `BitNet`'s `i2_s`, and the model this targets is `qwen35`, not the `BitNet`
//! b1.58 family. The whole substrate — `TernaryGroupWeights`,
//! `simd_ternary_group_matvec`, `ternary_group_scale` — is named `ternary`.
//!
//! ## What stays dense
//!
//! Embeddings (`wte`), the LM head, and the `RMSNorm` gammas are f32 here — the
//! usual ternary-LLM convention, where only the projections are ternary.
//!
//! ⚠️ **Ternary-Bonsai-27B does NOT follow that convention.** Reading the real
//! GGUF header (2026-08-10) shows `token_embd.weight` and `output.weight` are
//! *also* type 42 (`Q2_0`), `[5120 × 248320]` each — 498 of the file's 851
//! tensors are `Q2_0`, and only the norms are F32. Loading that model into this
//! struct therefore means dequantizing the embedding table and LM head to f32
//! (~5.1 GB at 248320 × 5120), which is a real memory cost, not free.
//!
//! ## This is the ternary `BitLinear` substrate, not the Bonsai runner
//!
//! This struct is LLaMA-shaped. Ternary-Bonsai-27B is `general.architecture =
//! "qwen35"` — a DeltaNet/attention **hybrid** (48 SSM layers + 16 full-
//! attention layers at `full_attention_interval = 4`, gated attention, QK-norm,
//! mrope). Running the published model needs the ternary port of
//! [`crate::deltanet`], not this type. See Issue 593.
//!
//! ## Loading
//!
//! `Q2_0` GGUF tensors → [`crate::quant::q2_0::repack_q2_0_to_ternary_group`]
//! (Plan 333 T3.1c, feature `q2_0_ternary_bridge`). That bridge is a *repack*,
//! not an expansion — both formats are 34 bytes per 128 weights. Every Bonsai
//! projection has `cols ∈ {5120, 6144, 17408}`, all multiples of 128, so the
//! bridge's alignment precondition holds for the real file.

use katgpt_core::TernaryGroupWeights;

/// Per-layer ternary transformer weights.
///
/// Shapes mirror [`crate::llama_layer::LlamaLayerWeights`]; each
/// `TernaryGroupWeights` carries its own `rows`/`cols`, so the forward pass
/// slices activations to `w.cols` / `w.rows` rather than re-deriving them from
/// `Config`.
pub struct TernaryLayerWeights {
    // Attention projections (ternary)
    pub attn_wq: TernaryGroupWeights, // [n_head * head_dim, n_embd]
    pub attn_wk: TernaryGroupWeights, // [n_kv_head * head_dim, n_embd]
    pub attn_wv: TernaryGroupWeights, // [n_kv_head * head_dim, n_embd]
    pub attn_wo: TernaryGroupWeights, // [n_embd, n_head * head_dim]
    // SwiGLU MLP (ternary)
    pub gate_proj: TernaryGroupWeights, // [mlp_hidden, n_embd]
    pub up_proj: TernaryGroupWeights,   // [mlp_hidden, n_embd]
    pub down_proj: TernaryGroupWeights, // [n_embd, mlp_hidden]
    // RMSNorm gammas stay dense f32 (no offset, LLaMA/Qwen convention)
    pub input_norm: Vec<f32>,     // [n_embd]
    pub post_attn_norm: Vec<f32>, // [n_embd] — alias: ffn_norm
}

impl TernaryLayerWeights {
    /// The 7 ternary projections in forward-pass order.
    pub fn projections(&self) -> [&TernaryGroupWeights; 7] {
        [
            &self.attn_wq,
            &self.attn_wk,
            &self.attn_wv,
            &self.attn_wo,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ]
    }

    /// G1 corruption check: every projection satisfies the bit-plane
    /// representation invariant (`pos_bits & neg_bits == 0` — a weight is never
    /// both `+1` and `-1`).
    ///
    /// A loader that mis-parses `Q2_0_g128` typically violates this, so it is
    /// the cheapest post-load sanity gate available.
    pub fn invariants_hold(&self) -> bool {
        self.projections().iter().all(|w| w.invariant_holds())
    }
}

/// All ternary transformer weights.
///
/// Separate `lm_head` (not tied to `wte`). No `wpe` — positions come from `RoPE`.
pub struct TernaryTransformerWeights {
    pub wte: Vec<f32>,                    // [vocab_size, n_embd]
    pub lm_head: Vec<f32>,                // [vocab_size, n_embd] (separate from wte)
    pub final_norm: Vec<f32>,             // [n_embd] (final RMSNorm)
    pub layers: Vec<TernaryLayerWeights>, // [n_layer]
}

impl TernaryTransformerWeights {
    /// G1 corruption check across every layer. See
    /// [`TernaryLayerWeights::invariants_hold`].
    pub fn invariants_hold(&self) -> bool {
        self.layers.iter().all(|l| l.invariants_hold())
    }

    /// Total ternary parameter count (`rows * cols` summed over all
    /// projections). Excludes the dense embedding / LM head / norms.
    pub fn ternary_param_count(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.projections())
            .map(|w| w.rows * w.cols)
            .sum()
    }

    /// Bytes occupied by the ternary projections: two bit-planes plus one f16
    /// scale per 128-weight group — 2.125 bits/weight at `cols % 128 == 0`.
    pub fn ternary_bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.projections())
            .map(|w| {
                let planes = 2 * w.rows * w.blocks64 * size_of::<u64>();
                let scales = w.rows * w.groups_per_row * size_of::<half::f16>();
                planes + scales
            })
            .sum()
    }
}
