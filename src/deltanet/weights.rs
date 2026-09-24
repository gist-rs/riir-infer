//! `DeltaNet` weight types and loader for hybrid DeltaNet/Attention inference (Plan 182).
//!
//! Supports models like Qwen 3.5 that mix `DeltaNet` (linear recurrent) layers
//! with standard attention layers. Each layer type has different weights:
//!
//! **Full attention layers**: standard Q/K/V/O projections + `SwiGLU` MLP
//! **Linear attention (`DeltaNet`) layers**: fused QKV/a/b/z projections + `out_proj` +
//!   conv1d + `A_log/dt_bias/norm` + `SwiGLU` MLP
//!
//! # Weight Layout
//!
//! Global weights:
//! - `wte`: embedding table [`vocab_size`, `n_embd`]
//! - `final_norm`: final `RMSNorm` `[n_embd]`
//! - `lm_head`: tied to `wte` (Qwen 3.5 uses `tie_word_embeddings: true`)
//!
//! Per-layer weights differ by type (see `DeltaNetLayerWeights`).

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::safetensors_loader::bf16_to_f32;
use crate::types::{Config, DeltaNetLayerType, matmul};

// ---------------------------------------------------------------------------
// `Proj` — shape-carrying projection matrix (Issue 594 Stage 1)
// ---------------------------------------------------------------------------

/// Shape-carrying projection matrix for the hybrid `DeltaNet` forward.
///
/// Carries the matrix shape `[rows, cols]` alongside the data, eliminating the
/// loose `rows, cols` arguments at every `matmul` call site in `forward.rs` +
/// `tree_forward/`. One `matvec(x, y)` call replaces
/// `matmul(y, &w.data, x, w.rows, w.cols)` — the shape is structural, so a
/// wrong-shape projection fails a `debug_assert` instead of silently reading
/// out-of-bounds.
///
/// **Stage 1 (this):** the dense arm only. The dense `DeltaNetLayerWeights`
/// projection fields become `Proj`; all live values are `Proj::Dense`.
///
/// **Stage 2 (future):** the ternary arm wraps `TernaryGroupWeights`, and the
/// parallel `DeltaNetTernaryLayerWeights` struct converges onto `Proj` too,
/// unifying the dense + ternary forwards behind one `matvec` dispatch.
///
/// Empty projections (`rows == 0`) represent the unused layer-type's fields
/// (e.g. `attn_wq` on a `DeltaNet` layer). `matvec` on an empty projection is a
/// no-op — the forward never calls it, but the shape stays consistent.
#[derive(Clone, Debug)]
pub enum Proj {
    /// Dense f32 projection `[rows, cols]` (row-major).
    Dense {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    /// Bit-plane packed ternary projection (Issue 594 Stage 2 target).
    ///
    /// Not constructed by the dense loader; the ternary path
    /// (`DeltaNetTernaryLayerWeights`) still uses `TernaryGroupWeights`
    /// directly until Stage 2 unifies the two structs.
    #[cfg(feature = "deltanet_ternary_inference")]
    Ternary(katgpt_core::TernaryGroupWeights),
}

impl Proj {
    /// Construct a dense projection from flat row-major data + shape.
    #[inline]
    pub fn dense(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            rows * cols,
            "Proj::dense: data.len() {} != rows*cols {}*{}",
            data.len(),
            rows,
            cols,
        );
        Self::Dense { data, rows, cols }
    }

    /// Construct an empty projection (the unused layer-type's field).
    #[inline]
    pub fn empty() -> Self {
        Self::Dense {
            data: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }

    /// Matrix-vector multiply: `y = self @ x`.
    ///
    /// `x.len()` must equal `cols`, `y.len()` must equal `rows`. Empty
    /// projections (`rows == 0`) are a no-op.
    #[inline(always)]
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Self::Dense { data, rows, cols } => {
                if *rows == 0 {
                    debug_assert!(data.is_empty(), "rows==0 but data not empty");
                    return;
                }
                debug_assert_eq!(x.len(), *cols, "Proj::matvec input dim mismatch");
                debug_assert_eq!(y.len(), *rows, "Proj::matvec output dim mismatch");
                matmul(y, data, x, *rows, *cols);
            }
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(t) => {
                if t.rows == 0 {
                    return;
                }
                debug_assert_eq!(x.len(), t.cols, "Proj::Ternary matvec input dim mismatch");
                debug_assert_eq!(y.len(), t.rows, "Proj::Ternary matvec output dim mismatch");
                katgpt_core::simd::simd_ternary_group_matvec(t, x, y);
            }
        }
    }

    /// Batched matrix-matrix multiply (Issue 597): for each position `p` in
    /// `0..batch`, `y[p*rows..(p+1)*rows] = self @ x[p*cols..(p+1)*cols]`.
    ///
    /// `x` is `[batch, cols]` row-major; `y` is `[batch, rows]` row-major.
    /// The result is **bit-identical** to `batch` separate `matvec` calls —
    /// each output element is the same dot product, just computed in a
    /// weight-reuse loop order that cuts weight memory traffic by `batch`.
    ///
    /// Empty projections (`rows == 0`) are a no-op.
    #[inline]
    pub fn matmat(&self, x: &[f32], y: &mut [f32], batch: usize) {
        match self {
            Self::Dense { data, rows, cols } => {
                if *rows == 0 || batch == 0 {
                    return;
                }
                debug_assert_eq!(x.len(), batch * cols, "Proj::matmat input shape mismatch");
                debug_assert_eq!(y.len(), batch * rows, "Proj::matmat output shape mismatch");
                katgpt_core::simd::simd_matmul_rows_batched(y, data, x, *rows, *cols, batch);
            }
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(t) => {
                if t.rows == 0 || batch == 0 {
                    return;
                }
                debug_assert_eq!(
                    x.len(),
                    batch * t.cols,
                    "Proj::Ternary matmat input mismatch"
                );
                debug_assert_eq!(
                    y.len(),
                    batch * t.rows,
                    "Proj::Ternary matmat output mismatch"
                );
                katgpt_core::simd::simd_ternary_group_matmul_batch(t, x, batch, y);
            }
        }
    }

    /// Total element count (`rows * cols`). Used by `verify_shapes`.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Dense { data, .. } => data.len(),
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(t) => t.rows * t.cols,
        }
    }

    /// `true` if the projection has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row count (output dimension).
    #[inline]
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense { rows, .. } => *rows,
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(t) => t.rows,
        }
    }

    /// Column count (input dimension).
    #[inline]
    pub fn cols(&self) -> usize {
        match self {
            Self::Dense { cols, .. } => *cols,
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(t) => t.cols,
        }
    }

    /// Immutable access to the dense data slice (Dense arm only).
    ///
    /// Panics on the Ternary arm — ternary data is bit-plane packed and
    /// has no f32 view. Used by upload paths (e.g. wgpu buffer construction)
    /// that need to read the dense weights as a flat `&[f32]`.
    #[inline]
    pub fn dense_data(&self) -> &[f32] {
        match self {
            Self::Dense { data, .. } => data,
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(_) => panic!("dense_data on Proj::Ternary"),
        }
    }

    /// Mutable access to the dense data slice (Dense arm only).
    ///
    /// Panics on the Ternary arm — ternary data is bit-plane packed and
    /// cannot be mutated as f32. Used by tests/benches that fill weights
    /// with seeded random values.
    #[inline]
    pub fn dense_data_mut(&mut self) -> &mut [f32] {
        match self {
            Self::Dense { data, .. } => data,
            #[cfg(feature = "deltanet_ternary_inference")]
            Self::Ternary(_) => panic!("dense_data_mut on Proj::Ternary"),
        }
    }
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

/// Per-layer weights for a hybrid DeltaNet/Attention model.
///
/// Linear attention (`DeltaNet`) and full attention layers use different subsets:
///
/// **Linear attention layers** use:
/// `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`,
/// `conv1d_weight`, `a_log`, `dt_bias`, `linear_norm`,
/// `gate_proj`, `up_proj`, `down_proj`, `input_norm`, `post_attn_norm`
///
/// **Full attention layers** use:
/// `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, `attn_q_norm`, `attn_k_norm`,
/// `gate_proj`, `up_proj`, `down_proj`, `input_norm`, `post_attn_norm`
///
/// Fields unused by a layer type are empty `Vec`.
pub struct DeltaNetLayerWeights {
    // --- Full attention projections (only for full attention layers) ---
    /// Gated Q projection `[2*q_dim, n_embd]` (Issue 594).
    ///
    /// Qwen3.5 gated attention concatenates q and gate in a single projection:
    /// the output is `[q(hd), gate(hd)]` per head, interleaved. The forward
    /// splits this into `q_buf` and `gate_buf`, applies QK-norm + `RoPE` to q,
    /// runs softmax attention, then multiplies the output by `sigmoid(gate)`
    /// before the output projection.
    pub attn_wq: Proj, // [2*q_dim, n_embd] = [2*n_head*head_dim, n_embd]
    pub attn_wk: Proj, // [kv_dim, n_embd]
    pub attn_wv: Proj, // [kv_dim, n_embd]
    pub attn_wo: Proj, // [n_embd, q_dim]
    /// QK-norm `RMSNorm` gamma for Q (per-head, [`head_dim`]). Qwen3.5 applies
    /// `RMSNorm` to each Q head before `RoPE`.
    pub attn_q_norm: Vec<f32>, // [head_dim]
    /// QK-norm `RMSNorm` gamma for K (per-head, [`head_dim`]).
    pub attn_k_norm: Vec<f32>, // [head_dim]

    // --- Linear attention (DeltaNet) projections ---
    pub in_proj_qkv: Proj, // [(linear_num_key_heads * linear_key_head_dim + 2 * linear_num_value_heads * linear_key_head_dim), n_embd]
    pub in_proj_a: Proj,   // [linear_num_key_heads, n_embd]
    pub in_proj_b: Proj,   // [linear_num_key_heads, n_embd]
    pub in_proj_z: Proj,   // [linear_num_value_heads * linear_key_head_dim, n_embd]
    pub out_proj: Proj,    // [n_embd, linear_num_value_heads * linear_key_head_dim]

    // --- DeltaNet-specific: conv1d + recurrence params ---
    pub conv1d_weight: Vec<f32>, // [linear_num_key_heads * 2 + linear_num_value_heads, conv_kernel_dim]
    pub a_log: Vec<f32>,         // [linear_num_value_heads] (GGUF `ssm_a`)
    pub dt_bias: Vec<f32>,       // [linear_num_key_heads * 2 + linear_num_value_heads]
    pub linear_norm: Vec<f32>,   // [linear_key_head_dim] — per-head gamma, shared across heads

    // --- SwiGLU MLP (both layer types) ---
    pub gate_proj: Proj, // [mlp_hidden, n_embd]
    pub up_proj: Proj,   // [mlp_hidden, n_embd]
    pub down_proj: Proj, // [n_embd, mlp_hidden]

    // --- RMSNorm (both layer types) ---
    pub input_norm: Vec<f32>,     // [n_embd]
    pub post_attn_norm: Vec<f32>, // [n_embd]
}

/// All weights for a hybrid DeltaNet/Attention model.
///
/// Qwen 3.5 uses tied embeddings (`tie_word_embeddings: true`),
/// so `lm_head` is a clone of `wte`.
pub struct QwenDeltaNetWeights {
    pub wte: Vec<f32>,        // [vocab_size, n_embd]
    pub final_norm: Vec<f32>, // [n_embd]
    /// Tied to `wte` for Qwen 3.5. Kept as `Proj` (Issue 594 Stage 1).
    pub lm_head: Proj, // [vocab_size, n_embd]
    pub layers: Vec<DeltaNetLayerWeights>, // [n_layer]
    /// Per-layer type map (from Config, cached for dispatch).
    pub layer_types: Vec<DeltaNetLayerType>,
}

// ---------------------------------------------------------------------------
// Helper: parse layer_types from config.json
// ---------------------------------------------------------------------------

/// Parse `text_config.layer_types` from config.json.
///
/// Maps `"linear_attention"` → `DeltaNet`, `"full_attention"` → `Attention`.
fn parse_layer_types(config_json: &serde_json::Value) -> Result<Vec<DeltaNetLayerType>> {
    let text_config = config_json
        .get("text_config")
        .context("missing 'text_config' in config.json")?;

    let arr = text_config
        .get("layer_types")
        .and_then(|v| v.as_array())
        .context("missing 'text_config.layer_types' in config.json")?;

    arr.iter()
        .enumerate()
        .map(|(i, v)| {
            let s = v
                .as_str()
                .with_context(|| format!("layer_types[{i}] is not a string"))?;
            match s {
                "linear_attention" => Ok(DeltaNetLayerType::DeltaNet),
                "full_attention" => Ok(DeltaNetLayerType::Attention),
                _ => bail!("unknown layer_types[{i}]: '{s}'"),
            }
        })
        .collect()
}

/// Parse `hidden_size`, `intermediate_size` etc. from config.json `text_config`.
fn parse_model_dims(config_json: &serde_json::Value) -> Result<ModelDims> {
    let tc = config_json
        .get("text_config")
        .context("missing 'text_config' in config.json")?;

    Ok(ModelDims {
        hidden_size: tc
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .context("missing text_config.hidden_size")? as usize,
        intermediate_size: tc
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .context("missing text_config.intermediate_size")? as usize,
        num_attention_heads: tc
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .context("missing text_config.num_attention_heads")?
            as usize,
        num_hidden_layers: tc
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .context("missing text_config.num_hidden_layers")? as usize,
        num_key_value_heads: tc
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .context("missing text_config.num_key_value_heads")?
            as usize,
        head_dim: tc
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .context("missing text_config.head_dim")? as usize,
        linear_key_head_dim: tc
            .get("linear_key_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize,
        linear_num_key_heads: tc
            .get("linear_num_key_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize,
        linear_num_value_heads: tc
            .get("linear_num_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize,
        linear_conv_kernel_dim: tc
            .get("linear_conv_kernel_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize,
        // Issue 594: partial RoPE dimension count. HuggingFace config stores it
        // as `rope_dimension_count` (Qwen3.5 = 64 of head_dim 256). GGUF stores
        // it as `rope.dimension_count`. 0 = not present → full rotation.
        rope_dimension_count: tc
            .get("rope_dimension_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        vocab_size: tc
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(248_320) as usize,
        rms_norm_eps: tc
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6),
        tie_word_embeddings: tc
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
    })
}

struct ModelDims {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    linear_key_head_dim: usize,
    #[allow(dead_code)] // parsed from config.json, may be used by future forward pass
    linear_num_key_heads: usize,
    #[allow(dead_code)] // parsed from config.json, may be used by future forward pass
    linear_num_value_heads: usize,
    linear_conv_kernel_dim: usize,
    /// Partial `RoPE` dimension count (Issue 594). 0 = full rotation (`head_dim`).
    rope_dimension_count: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    tie_word_embeddings: bool,
}

// ---------------------------------------------------------------------------
// Config construction from parsed dims
// ---------------------------------------------------------------------------

// Issue 724: `needless_update` is allowed because the struct spread at the
// literal's end is load-bearing under feature skew — katgpt_core::Config's
// fields are gated on katgpt-types features while the literal's cfg gates
// track ENGINE features, so a graph enabling katgpt-core's side (e.g.
// katgpt-rs defaults) without engine's fills the remainder from Default
// instead of failing E0063. Under full engine features the spread is
// redundant — clippy cannot see the feature resolution.
#[allow(unexpected_cfgs, clippy::needless_update)]
fn build_config(dims: &ModelDims, layer_types: Vec<DeltaNetLayerType>) -> Config {
    Config {
        vocab_size: dims.vocab_size,
        block_size: 32768,
        n_embd: dims.hidden_size,
        n_head: dims.num_attention_heads,
        head_dim: dims.head_dim,
        mlp_hidden: dims.intermediate_size,
        n_layer: dims.num_hidden_layers,
        n_kv_head: dims.num_key_value_heads,
        bos_token: 151_643,
        draft_lookahead: 0,
        tree_budget: 0,
        parallel_threshold: 8192,
        lora_rank: 0,
        lora_alpha: 1.0,
        lora_dropout: 0.0,
        lora_targets: Vec::new(),
        screening_threshold: 0.0,
        sparse_threshold: 0.0,
        early_exit_patience: 0,
        early_exit_gap: 0.0,
        mtp_activation_threshold: 0,
        mtp_cluster_vocab_threshold: dims.vocab_size,
        mtp_shared_kv_prompt_threshold: 32768,
        mtp_cluster_size: 1024,
        mtp_min_output_tokens: 16,
        mtp_cluster_topk: 1,
        mask_token: 0,
        sp_kv_window: 128,
        sp_kv_predictor_hidden: 0,
        width_rollouts: 1,
        d2f_block_size: 16,
        mls_layers: 0,
        rms_norm_eps: dims.rms_norm_eps,
        sp_kv_predictor_lr_mult: 5.0,
        temperature: 0.8,
        hla_decay: 1.0,
        rope_theta: 10000.0,
        attn_logit_softcapping: 0.0,
        final_logit_softcapping: 0.0,
        sp_kv_threshold: 0.5,
        early_stop_threshold: 0.0,
        parallax_gate_scale: 0.0,
        emotion_desperation_threshold: 0.5,
        hla_mode: katgpt_core::types::HlaMode::Standard,
        model_arch: katgpt_core::types::ModelArchitecture::QwenDeltaNet,
        attention_mode: katgpt_core::types::AttentionMode::Causal,
        convergence_selector: katgpt_core::types::ConvergenceSelector::default(),
        loop_mode: katgpt_core::types::LoopMode::None,
        hybrid_pattern: katgpt_core::types::HybridPattern::Uniform,
        // Issue 035 fields — default to 0 = "derive from loop_mode" (which is
        // `None` here, so effective loop count is 1). Matches every other
        // Config constructor in katgpt-core/src/types.rs.
        loop_min: 0,
        loop_max: 0,
        weight_dtype: katgpt_core::types::WeightDtype::BF16,
        hla_normalize: false,
        rms_norm_offset: false,
        tied_embeddings: dims.tie_word_embeddings,
        use_rope: true,
        post_norm: false,
        gated_attn: false,
        parallax_zero_init: true,
        #[cfg(feature = "hydra_budget")]
        hydra_profiles: Vec::new(),
        layer_types,
        deltanet_conv_kernel_size: dims.linear_conv_kernel_dim,
        deltanet_state_dim: dims.linear_num_key_heads
            * dims.linear_key_head_dim
            * dims.linear_key_head_dim,
        deltanet_linear_head_dim: dims.linear_key_head_dim,
        deltanet_linear_n_heads: dims.linear_num_key_heads,
        deltanet_linear_n_value_heads: dims.linear_num_value_heads,
        // Issue 594: partial RoPE dimension count. Read from config.json's
        // `rope_dimension_count` if present; 0 = full rotation (backward compat).
        rope_dimension_count: dims.rope_dimension_count,
        #[cfg(feature = "rim_slots")]
        rim_block_count: 0,
        #[cfg(feature = "rim_slots")]
        rim_tokens_per_block: 2,
        #[cfg(feature = "rim_slots")]
        rim_buffer_token: 0,
        #[cfg(feature = "wall_attention")]
        wall_config: None,
        #[cfg(feature = "collapse_aware_thinking")]
        collapse_budget: katgpt_core::types::ThinkingBudget::default(),
        #[cfg(feature = "belief_drafter")]
        belief_drafter_path: None,
        #[cfg(feature = "belief_drafter")]
        belief_drafter_entropy_threshold: 2.0,
        #[cfg(feature = "gemma4_inference")]
        gemma4_layer_types: Vec::new(),
        #[cfg(feature = "gemma4_inference")]
        sliding_window: 0,
        #[cfg(feature = "gemma4_inference")]
        global_head_dim: 0,
        #[cfg(feature = "gemma4_inference")]
        n_global_kv_head: 0,
        #[cfg(feature = "gemma4_inference")]
        partial_rotary_factor: 1.0,
        #[cfg(feature = "gemma4_inference")]
        rope_theta_full: 0.0,
        // Issue 724: tolerate feature skew between engine and katgpt-core
        // (see the allow on `build_config` for the full rationale).
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Safetensors helpers (local copies to avoid pub dependency issues)
// ---------------------------------------------------------------------------

/// Parsed tensor metadata from safetensors header.
struct TensorMeta {
    #[allow(dead_code)]
    dtype: String,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

/// Parse safetensors file header, returning `(header_json_size, tensor_map)`.
fn parse_safetensors_header(data: &[u8]) -> Result<(usize, BTreeMap<String, TensorMeta>)> {
    if data.len() < 8 {
        bail!("safetensors file too small: {} bytes", data.len());
    }

    let header_len = u64::from_le_bytes(
        data[0..8]
            .try_into()
            .context("failed to read header length")?,
    ) as usize;

    let header_end = 8 + header_len;
    if data.len() < header_end {
        bail!(
            "safetensors header truncated: need {} bytes, have {}",
            header_end,
            data.len()
        );
    }

    let header: serde_json::Value = serde_json::from_slice(&data[8..header_end])
        .context("failed to parse safetensors header JSON")?;

    let obj = header
        .as_object()
        .context("safetensors header is not a JSON object")?;

    let mut tensor_map = BTreeMap::new();
    for (name, value) in obj {
        if name == "__metadata__" {
            continue;
        }

        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .with_context(|| format!("missing data_offsets for tensor '{name}'"))?;

        if offsets.len() != 2 {
            bail!(
                "expected 2 data_offsets for tensor '{name}', got {}",
                offsets.len()
            );
        }

        let data_start = offsets[0]
            .as_u64()
            .with_context(|| format!("invalid data_offsets[0] for '{name}'"))?
            as usize;
        let data_end = offsets[1]
            .as_u64()
            .with_context(|| format!("invalid data_offsets[1] for '{name}'"))?
            as usize;

        tensor_map.insert(
            name.clone(),
            TensorMeta {
                dtype,
                shape,
                data_start,
                data_end,
            },
        );
    }

    Ok((header_len, tensor_map))
}

/// Extract and dequantize a BF16 tensor from mmap data.
fn extract_tensor(
    mmap: &[u8],
    header_len: usize,
    meta: &TensorMeta,
    name: &str,
) -> Result<Vec<f32>> {
    let data_offset = 8 + header_len;
    let raw_start = data_offset + meta.data_start;
    let raw_end = data_offset + meta.data_end;
    let raw_data = &mmap[raw_start..raw_end];

    let elements: usize = meta.shape.iter().product();
    let expected_bytes = elements * 2; // BF16 = 2 bytes
    if raw_data.len() != expected_bytes {
        bail!(
            "BF16 tensor '{}' size mismatch: expected {} bytes, got {}",
            name,
            expected_bytes,
            raw_data.len()
        );
    }

    let f32_data: Vec<f32> = raw_data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
        .collect();

    Ok(f32_data)
}

/// Remove a tensor from the map, failing if missing.
fn require_tensor(tensors: &mut BTreeMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    tensors
        .remove(name)
        .with_context(|| format!("missing required tensor '{name}'"))
}

/// Load requested tensors from a single safetensors shard file.
fn load_tensors_from_shard(
    shard_path: &Path,
    needed_names: &[String],
) -> Result<BTreeMap<String, Vec<f32>>> {
    let file = std::fs::File::open(shard_path)
        .with_context(|| format!("failed to open shard {}", shard_path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap shard {}", shard_path.display()))?;

    let (header_len, tensor_map) = parse_safetensors_header(&mmap)?;

    let mut result = BTreeMap::new();
    for name in needed_names {
        let meta = tensor_map.get(name).with_context(|| {
            format!(
                "tensor '{}' not found in shard {}",
                name,
                shard_path.display()
            )
        })?;
        let data = extract_tensor(&mmap, header_len, meta, name)?;
        result.insert(name.clone(), data);
    }
    Ok(result)
}

/// Parse the shard index JSON to find which shard each needed tensor is in.
fn parse_shard_index(index_path: &Path, needed: &[String]) -> Result<BTreeMap<String, String>> {
    let index_json = std::fs::read_to_string(index_path)
        .with_context(|| format!("failed to read {}", index_path.display()))?;
    let index: serde_json::Value =
        serde_json::from_str(&index_json).context("failed to parse safetensors index JSON")?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("missing weight_map in safetensors index")?;

    let needed_set: HashSet<&String> = needed.iter().collect();
    let mut result = BTreeMap::new();

    for (tensor_name, shard_val) in weight_map {
        if !needed_set.contains(tensor_name) {
            continue;
        }
        let shard_file = shard_val
            .as_str()
            .with_context(|| format!("invalid shard path for tensor '{tensor_name}'"))?;
        result.insert(tensor_name.clone(), shard_file.to_string());
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Collect needed tensor names based on per-layer type
// ---------------------------------------------------------------------------

/// Build the list of tensor names to load, based on per-layer type.
fn collect_needed_tensor_names(
    n_layer: usize,
    layer_types: &[DeltaNetLayerType],
    tie_word_embeddings: bool,
) -> Vec<String> {
    // Max tensors per layer: 9 (linear) + 3 (mlp) + 2 (norms) = 14
    let mut names = Vec::with_capacity(3 + n_layer * 14);

    // Global weights
    names.push("model.language_model.embed_tokens.weight".to_string());
    names.push("model.language_model.norm.weight".to_string());
    // lm_head only needed when not tied (Qwen 3.5 ties it)
    if !tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }

    for (i, lt) in layer_types.iter().enumerate() {
        let is_linear = *lt == DeltaNetLayerType::DeltaNet;

        if is_linear {
            // Linear attention (DeltaNet) weights
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.in_proj_a.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.in_proj_b.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.in_proj_z.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.out_proj.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.conv1d.weight"
            ));
            names.push(format!("model.language_model.layers.{i}.linear_attn.A_log"));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.dt_bias"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.linear_attn.norm.weight"
            ));
        } else {
            // Full attention weights
            names.push(format!(
                "model.language_model.layers.{i}.self_attn.q_proj.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.self_attn.k_proj.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.self_attn.v_proj.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.self_attn.o_proj.weight"
            ));
        }

        // MLP + norms (all layers)
        names.push(format!(
            "model.language_model.layers.{i}.input_layernorm.weight"
        ));
        names.push(format!(
            "model.language_model.layers.{i}.post_attention_layernorm.weight"
        ));
        names.push(format!(
            "model.language_model.layers.{i}.mlp.gate_proj.weight"
        ));
        names.push(format!(
            "model.language_model.layers.{i}.mlp.up_proj.weight"
        ));
        names.push(format!(
            "model.language_model.layers.{i}.mlp.down_proj.weight"
        ));
    }

    names.shrink_to_fit();
    names
}

// ---------------------------------------------------------------------------
// Public loader
// ---------------------------------------------------------------------------

/// Load Qwen `DeltaNet` hybrid model weights from a `HuggingFace` cache directory.
///
/// Parses `config.json` for model dimensions and per-layer type map,
/// then loads weights from safetensors shard(s). Returns the parsed
/// `Config` and assembled `QwenDeltaNetWeights`.
pub fn load_qwen_deltanet_weights(model_dir: &Path) -> Result<(Config, QwenDeltaNetWeights)> {
    // 1. Parse config.json
    let config_path = model_dir.join("config.json");
    let config_json_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_json: serde_json::Value =
        serde_json::from_str(&config_json_str).context("failed to parse config.json")?;

    let dims = parse_model_dims(&config_json)?;
    let layer_types = parse_layer_types(&config_json)?;
    let config = build_config(&dims, layer_types.clone());

    // 2. Build list of needed tensor names
    let needed =
        collect_needed_tensor_names(config.n_layer, &layer_types, dims.tie_word_embeddings);

    // 3. Determine shard mapping
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    let shard_map = if index_path.exists() {
        parse_shard_index(&index_path, &needed)?
    } else if single_path.exists() {
        needed
            .iter()
            .map(|name| (name.clone(), "model.safetensors".to_string()))
            .collect()
    } else {
        // Try Qwen naming: model.safetensors-00001-of-00001.safetensors
        // Look for any .safetensors file in the directory
        let mut found: BTreeMap<String, String> = BTreeMap::new();
        let entries = std::fs::read_dir(model_dir)
            .with_context(|| format!("failed to read dir {}", model_dir.display()))?;
        let mut shard_files: Vec<String> = Vec::new();
        for entry in entries {
            let entry = entry?;
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            if fname_str.starts_with("model.safetensors") && fname_str.ends_with(".safetensors") {
                shard_files.push(fname_str.to_string());
            }
        }
        if shard_files.is_empty() {
            bail!(
                "no safetensors found in {}: expected index, single file, or numbered shards",
                model_dir.display()
            );
        }
        // Single shard case: all tensors in one file
        if shard_files.len() == 1 {
            for name in &needed {
                found.insert(name.clone(), shard_files[0].clone());
            }
        } else {
            // Multi-shard without index: try to parse each shard header
            for name in &needed {
                for shard in &shard_files {
                    let shard_path = model_dir.join(shard);
                    let Ok(file) = std::fs::File::open(&shard_path) else {
                        continue;
                    };
                    let Ok(mmap) = (unsafe { Mmap::map(&file) }) else {
                        continue;
                    };
                    if let Ok((_, tensor_map)) = parse_safetensors_header(&mmap)
                        && tensor_map.contains_key(name)
                    {
                        found.insert(name.clone(), shard.clone());
                        break;
                    }
                }
            }
        }
        found
    };

    // 4. Group by shard file for efficient loading (one mmap per shard)
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor_name, shard_file) in &shard_map {
        grouped
            .entry(shard_file.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    // 5. Load all tensors from shards
    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (shard_file, tensor_names) in &grouped {
        let shard_path = model_dir.join(shard_file);
        let shard_tensors = load_tensors_from_shard(&shard_path, tensor_names)?;
        tensors.extend(shard_tensors);
    }

    // 6. Assemble global weights
    let wte = require_tensor(&mut tensors, "model.language_model.embed_tokens.weight")?;
    let final_norm = require_tensor(&mut tensors, "model.language_model.norm.weight")?;
    // Tied embeddings: lm_head = wte
    let lm_head = if dims.tie_word_embeddings {
        Proj::dense(wte.clone(), config.vocab_size, config.n_embd)
    } else {
        Proj::dense(
            require_tensor(&mut tensors, "lm_head.weight")?,
            config.vocab_size,
            config.n_embd,
        )
    };

    // Shape dims for `Proj::dense` construction (mirror `zeros`).
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let kvd = config.n_kv_head * config.head_dim;
    let mlp = config.mlp_hidden;
    let lhd = config.deltanet_linear_head_dim;
    let lkh = config.deltanet_linear_n_heads;
    let lvh = config.deltanet_linear_n_value_heads;
    let l_qkv_out = lkh * lhd + 2 * lvh * lhd;
    let l_a_out = lkh;
    let l_z_out = lvh * lhd;
    let l_out_in = lvh * lhd;

    // 7. Assemble per-layer weights
    let mut layers = Vec::with_capacity(config.n_layer);
    for (i, lt) in layer_types.iter().enumerate() {
        let is_linear = *lt == DeltaNetLayerType::DeltaNet;

        let layer = if is_linear {
            DeltaNetLayerWeights {
                // Full attention: empty
                attn_wq: Proj::empty(),
                attn_wk: Proj::empty(),
                attn_wv: Proj::empty(),
                attn_wo: Proj::empty(),
                attn_q_norm: Vec::new(),
                attn_k_norm: Vec::new(),
                // Linear attention
                in_proj_qkv: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"),
                    )?,
                    l_qkv_out,
                    n,
                ),
                in_proj_a: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.linear_attn.in_proj_a.weight"),
                    )?,
                    l_a_out,
                    n,
                ),
                in_proj_b: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.linear_attn.in_proj_b.weight"),
                    )?,
                    l_a_out,
                    n,
                ),
                in_proj_z: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.linear_attn.in_proj_z.weight"),
                    )?,
                    l_z_out,
                    n,
                ),
                out_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.linear_attn.out_proj.weight"),
                    )?,
                    n,
                    l_out_in,
                ),
                conv1d_weight: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.linear_attn.conv1d.weight"),
                )?,
                a_log: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.linear_attn.A_log"),
                )?,
                dt_bias: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.linear_attn.dt_bias"),
                )?,
                linear_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.linear_attn.norm.weight"),
                )?,
                // MLP + norms
                gate_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.mlp.gate_proj.weight"),
                    )?,
                    mlp,
                    n,
                ),
                up_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.mlp.up_proj.weight"),
                    )?,
                    mlp,
                    n,
                ),
                down_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.mlp.down_proj.weight"),
                    )?,
                    n,
                    mlp,
                ),
                input_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.input_layernorm.weight"),
                )?,
                post_attn_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.post_attention_layernorm.weight"),
                )?,
            }
        } else {
            DeltaNetLayerWeights {
                // Full attention
                attn_wq: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.self_attn.q_proj.weight"),
                    )?,
                    2 * q_dim,
                    n,
                ),
                attn_wk: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.self_attn.k_proj.weight"),
                    )?,
                    kvd,
                    n,
                ),
                attn_wv: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.self_attn.v_proj.weight"),
                    )?,
                    kvd,
                    n,
                ),
                attn_wo: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.self_attn.o_proj.weight"),
                    )?,
                    n,
                    q_dim,
                ),
                attn_q_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.self_attn.q_norm.weight"),
                )?,
                attn_k_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.self_attn.k_norm.weight"),
                )?,
                // Linear attention: empty
                in_proj_qkv: Proj::empty(),
                in_proj_a: Proj::empty(),
                in_proj_b: Proj::empty(),
                in_proj_z: Proj::empty(),
                out_proj: Proj::empty(),
                conv1d_weight: Vec::new(),
                a_log: Vec::new(),
                dt_bias: Vec::new(),
                linear_norm: Vec::new(),
                // MLP + norms
                gate_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.mlp.gate_proj.weight"),
                    )?,
                    mlp,
                    n,
                ),
                up_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.mlp.up_proj.weight"),
                    )?,
                    mlp,
                    n,
                ),
                down_proj: Proj::dense(
                    require_tensor(
                        &mut tensors,
                        &format!("model.language_model.layers.{i}.mlp.down_proj.weight"),
                    )?,
                    n,
                    mlp,
                ),
                input_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.input_layernorm.weight"),
                )?,
                post_attn_norm: require_tensor(
                    &mut tensors,
                    &format!("model.language_model.layers.{i}.post_attention_layernorm.weight"),
                )?,
            }
        };
        layers.push(layer);
    }

    let weights = QwenDeltaNetWeights {
        wte,
        final_norm,
        lm_head,
        layers,
        layer_types,
    };

    Ok((config, weights))
}

// ---------------------------------------------------------------------------
// zeros() and verify_shapes()
// ---------------------------------------------------------------------------

impl QwenDeltaNetWeights {
    /// Create zero-initialized weights for testing.
    ///
    /// **Do NOT use for inference** — weights are all zeros.
    /// This exists for GOAT proof tests that verify weight shapes match config.
    pub fn zeros(config: &Config) -> Self {
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;

        let layer_types = if config.layer_types.is_empty() {
            vec![DeltaNetLayerType::Attention; config.n_layer]
        } else {
            config.layer_types.clone()
        };

        // Linear attention dimensions (from actual Qwen 3.5 model).
        // These are stored in the config's deltanet_* fields.
        let linear_key_head_dim = config.deltanet_linear_head_dim;
        let linear_num_key_heads = config.deltanet_linear_n_heads;
        let linear_num_value_heads = config.deltanet_linear_n_value_heads;
        let conv_ks = config.deltanet_conv_kernel_size;

        // Linear attention intermediate dims
        let linear_qkv_out = linear_num_key_heads * linear_key_head_dim
            + 2 * linear_num_value_heads * linear_key_head_dim;
        // Issue 594: `ssm_alpha` / `ssm_beta` emit one scalar per **value**
        // head, not per key head — the real GGUF has
        // `blk.N.ssm_{alpha,beta}.weight = [5120 x 48]` with
        // `num_v_heads = 48` and `num_k_heads = 16`, and the forward reads
        // `a_raw[h]` / `b_raw[h]` for `h in 0..n_v_heads`. The ternary side
        // (`ternary_weights.rs`) already had this right.
        let linear_a_out = linear_num_value_heads;
        let linear_b_out = linear_num_value_heads;
        let linear_z_out = linear_num_value_heads * linear_key_head_dim;
        let linear_out_in = linear_num_value_heads * linear_key_head_dim;
        // conv1d: operates on concatenated [q, k, v] channels after QKV projection.
        // conv_dim = q_dim + k_dim + v_dim = n_k_heads*head_dim + n_k_heads*head_dim + n_v_heads*head_dim
        let conv_dim = linear_num_key_heads * linear_key_head_dim * 2
            + linear_num_value_heads * linear_key_head_dim;

        let layers: Vec<DeltaNetLayerWeights> = layer_types
            .iter()
            .map(|&lt| {
                let is_linear = lt == DeltaNetLayerType::DeltaNet;
                DeltaNetLayerWeights {
                    // Full attention projections
                    attn_wq: if is_linear {
                        Proj::empty()
                    } else {
                        // Issue 594: gated attention — attn_wq is [2*q_dim, n_embd]
                        // (q + gate concatenated per head). The forward computes
                        // attn_wq @ x → [2*q_dim], then splits into q and gate.
                        Proj::dense(vec![0.0f32; 2 * q_dim * n], 2 * q_dim, n)
                    },
                    attn_wk: if is_linear {
                        Proj::empty()
                    } else {
                        Proj::dense(vec![0.0f32; kvd * n], kvd, n)
                    },
                    attn_wv: if is_linear {
                        Proj::empty()
                    } else {
                        Proj::dense(vec![0.0f32; kvd * n], kvd, n)
                    },
                    // Issue 594: `attn_wo` consumes the FULL Q width (`q_dim =
                    // n_head * head_dim`), not the KV width — the gate is
                    // applied to the attention output before this projection,
                    // so the width is `q_dim` even under GQA. The real
                    // safetensors/GGUF loaders already build it as
                    // `[n_embd, q_dim]`; this fixture said `kvd`, so it only
                    // agreed with the loaders when `n_head == n_kv_head`.
                    attn_wo: if is_linear {
                        Proj::empty()
                    } else {
                        Proj::dense(vec![0.0f32; n * q_dim], n, q_dim)
                    },
                    // QK-norm: ones (identity gamma) for attention layers, empty for linear
                    attn_q_norm: if is_linear {
                        Vec::new()
                    } else {
                        vec![1.0f32; config.head_dim]
                    },
                    attn_k_norm: if is_linear {
                        Vec::new()
                    } else {
                        vec![1.0f32; config.head_dim]
                    },

                    // Linear attention projections
                    in_proj_qkv: if is_linear {
                        Proj::dense(vec![0.0f32; linear_qkv_out * n], linear_qkv_out, n)
                    } else {
                        Proj::empty()
                    },
                    in_proj_a: if is_linear {
                        Proj::dense(vec![0.0f32; linear_a_out * n], linear_a_out, n)
                    } else {
                        Proj::empty()
                    },
                    in_proj_b: if is_linear {
                        Proj::dense(vec![0.0f32; linear_b_out * n], linear_b_out, n)
                    } else {
                        Proj::empty()
                    },
                    in_proj_z: if is_linear {
                        Proj::dense(vec![0.0f32; linear_z_out * n], linear_z_out, n)
                    } else {
                        Proj::empty()
                    },
                    out_proj: if is_linear {
                        Proj::dense(vec![0.0f32; n * linear_out_in], n, linear_out_in)
                    } else {
                        Proj::empty()
                    },

                    // DeltaNet-specific params
                    conv1d_weight: if is_linear {
                        vec![0.0f32; conv_dim * conv_ks]
                    } else {
                        Vec::new()
                    },
                    // Issue 594: `ssm_a` is `[num_v_heads]` (48), not
                    // `[num_k_heads]` — indexed as `a_log[h]` for
                    // `h in 0..n_v_heads` by the gate loop.
                    a_log: if is_linear {
                        vec![0.0f32; linear_num_value_heads]
                    } else {
                        Vec::new()
                    },
                    dt_bias: if is_linear {
                        vec![0.0f32; linear_num_value_heads] // per-value-head bias for decay gate
                    } else {
                        Vec::new()
                    },
                    // PER-HEAD RMSNorm gamma (Issue 594) — `[head_dim]`,
                    // shared across heads, matching the real `ssm_norm`.
                    linear_norm: if is_linear {
                        vec![0.0f32; linear_key_head_dim]
                    } else {
                        Vec::new()
                    },

                    // MLP (both types)
                    gate_proj: Proj::dense(vec![0.0f32; mlp * n], mlp, n),
                    up_proj: Proj::dense(vec![0.0f32; mlp * n], mlp, n),
                    down_proj: Proj::dense(vec![0.0f32; n * mlp], n, mlp),

                    // Norms (both types)
                    input_norm: vec![0.0f32; n],
                    post_attn_norm: vec![0.0f32; n],
                }
            })
            .collect();

        let wte = vec![0.0f32; config.vocab_size * n];
        let lm_head = Proj::dense(wte.clone(), config.vocab_size, n); // Tied embeddings

        Self {
            wte,
            final_norm: vec![0.0f32; n],
            lm_head,
            layers,
            layer_types,
        }
    }

    /// Verify weight shapes match config dimensions.
    ///
    /// GOAT proof helper: checks every weight tensor has the expected size.
    /// Returns Ok(()) if all shapes match, Err with diagnostic message otherwise.
    pub fn verify_shapes(&self, config: &Config) -> Result<(), String> {
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;

        let linear_key_head_dim = config.deltanet_linear_head_dim;
        let linear_num_key_heads = config.deltanet_linear_n_heads;
        let linear_num_value_heads = config.deltanet_linear_n_value_heads;
        let conv_ks = config.deltanet_conv_kernel_size;

        let linear_qkv_out = linear_num_key_heads * linear_key_head_dim
            + 2 * linear_num_value_heads * linear_key_head_dim;
        // Issue 594: `ssm_alpha` / `ssm_beta` emit one scalar per **value**
        // head, not per key head — the real GGUF has
        // `blk.N.ssm_{alpha,beta}.weight = [5120 x 48]` with
        // `num_v_heads = 48` and `num_k_heads = 16`, and the forward reads
        // `a_raw[h]` / `b_raw[h]` for `h in 0..n_v_heads`. The ternary side
        // (`ternary_weights.rs`) already had this right.
        let linear_a_out = linear_num_value_heads;
        let linear_b_out = linear_num_value_heads;
        let linear_z_out = linear_num_value_heads * linear_key_head_dim;
        let linear_out_in = linear_num_value_heads * linear_key_head_dim;
        let conv_dim = linear_num_key_heads * linear_key_head_dim * 2
            + linear_num_value_heads * linear_key_head_dim;

        if self.wte.len() != config.vocab_size * n {
            return Err(format!(
                "wte: expected {} elements, got {}",
                config.vocab_size * n,
                self.wte.len()
            ));
        }
        if self.final_norm.len() != n {
            return Err(format!(
                "final_norm: expected {n} elements, got {}",
                self.final_norm.len()
            ));
        }
        if self.lm_head.len() != config.vocab_size * n {
            return Err(format!(
                "lm_head: expected {} elements, got {}",
                config.vocab_size * n,
                self.lm_head.len()
            ));
        }
        if self.layers.len() != config.n_layer {
            return Err(format!(
                "layers: expected {} layers, got {}",
                config.n_layer,
                self.layers.len()
            ));
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let is_linear = self.layer_types[i] == DeltaNetLayerType::DeltaNet;
            let prefix = format!("layer {i}");

            if is_linear {
                // Full attention fields should be empty
                if !layer.attn_wq.is_empty() {
                    return Err(format!(
                        "{prefix} attn_wq: linear layer should be empty, got {} elements",
                        layer.attn_wq.len()
                    ));
                }
                if !layer.attn_wk.is_empty() {
                    return Err(format!(
                        "{prefix} attn_wk: linear layer should be empty, got {} elements",
                        layer.attn_wk.len()
                    ));
                }
                if !layer.attn_wv.is_empty() {
                    return Err(format!(
                        "{prefix} attn_wv: linear layer should be empty, got {} elements",
                        layer.attn_wv.len()
                    ));
                }
                if !layer.attn_wo.is_empty() {
                    return Err(format!(
                        "{prefix} attn_wo: linear layer should be empty, got {} elements",
                        layer.attn_wo.len()
                    ));
                }
                if !layer.attn_q_norm.is_empty() {
                    return Err(format!(
                        "{prefix} attn_q_norm: linear layer should be empty, got {} elements",
                        layer.attn_q_norm.len()
                    ));
                }
                if !layer.attn_k_norm.is_empty() {
                    return Err(format!(
                        "{prefix} attn_k_norm: linear layer should be empty, got {} elements",
                        layer.attn_k_norm.len()
                    ));
                }

                // Linear attention fields should have expected sizes
                if layer.in_proj_qkv.len() != linear_qkv_out * n {
                    return Err(format!(
                        "{prefix} in_proj_qkv: expected {} elements, got {}",
                        linear_qkv_out * n,
                        layer.in_proj_qkv.len()
                    ));
                }
                if layer.in_proj_a.len() != linear_a_out * n {
                    return Err(format!(
                        "{prefix} in_proj_a: expected {} elements, got {}",
                        linear_a_out * n,
                        layer.in_proj_a.len()
                    ));
                }
                if layer.in_proj_b.len() != linear_b_out * n {
                    return Err(format!(
                        "{prefix} in_proj_b: expected {} elements, got {}",
                        linear_b_out * n,
                        layer.in_proj_b.len()
                    ));
                }
                if layer.in_proj_z.len() != linear_z_out * n {
                    return Err(format!(
                        "{prefix} in_proj_z: expected {} elements, got {}",
                        linear_z_out * n,
                        layer.in_proj_z.len()
                    ));
                }
                if layer.out_proj.len() != n * linear_out_in {
                    return Err(format!(
                        "{prefix} out_proj: expected {} elements, got {}",
                        n * linear_out_in,
                        layer.out_proj.len()
                    ));
                }
                if layer.conv1d_weight.len() != conv_dim * conv_ks {
                    return Err(format!(
                        "{prefix} conv1d_weight: expected {} elements, got {}",
                        conv_dim * conv_ks,
                        layer.conv1d_weight.len()
                    ));
                }
                if layer.a_log.len() != linear_num_value_heads {
                    return Err(format!(
                        "{prefix} a_log: expected {} elements, got {}",
                        linear_num_value_heads,
                        layer.a_log.len()
                    ));
                }
                if layer.dt_bias.len() != linear_num_value_heads {
                    return Err(format!(
                        "{prefix} dt_bias: expected {} elements, got {}",
                        linear_num_value_heads,
                        layer.dt_bias.len()
                    ));
                }
                // PER-HEAD gamma (Issue 594): `ssm_norm` is `[head_dim]`,
                // shared across heads — NOT `n_v_heads * head_dim`. The old
                // expectation would have REJECTED a correctly-loaded real
                // model from either the GGUF or the safetensors loader.
                if layer.linear_norm.len() != linear_key_head_dim {
                    return Err(format!(
                        "{prefix} linear_norm: expected {} elements (per-head gamma), got {}",
                        linear_key_head_dim,
                        layer.linear_norm.len()
                    ));
                }
            } else {
                // Full attention fields should have expected sizes
                if layer.attn_wq.len() != 2 * q_dim * n {
                    return Err(format!(
                        "{prefix} attn_wq: expected {} elements (gated: 2*q_dim*n), got {}",
                        2 * q_dim * n,
                        layer.attn_wq.len()
                    ));
                }
                if layer.attn_wk.len() != kvd * n {
                    return Err(format!(
                        "{prefix} attn_wk: expected {} elements, got {}",
                        kvd * n,
                        layer.attn_wk.len()
                    ));
                }
                if layer.attn_wv.len() != kvd * n {
                    return Err(format!(
                        "{prefix} attn_wv: expected {} elements, got {}",
                        kvd * n,
                        layer.attn_wv.len()
                    ));
                }
                // Issue 594: `q_dim`, not `kvd` — see the `zeros()` comment.
                // As written before, this validator would have REJECTED a
                // correctly-loaded real model (the same failure mode as the
                // `ssm_norm` validator bug fixed earlier in this issue).
                if layer.attn_wo.len() != n * q_dim {
                    return Err(format!(
                        "{prefix} attn_wo: expected {} elements, got {}",
                        n * q_dim,
                        layer.attn_wo.len()
                    ));
                }
                let hd = config.head_dim;
                if layer.attn_q_norm.len() != hd {
                    return Err(format!(
                        "{prefix} attn_q_norm: expected {hd} elements, got {}",
                        layer.attn_q_norm.len()
                    ));
                }
                if layer.attn_k_norm.len() != hd {
                    return Err(format!(
                        "{prefix} attn_k_norm: expected {hd} elements, got {}",
                        layer.attn_k_norm.len()
                    ));
                }

                // Linear attention fields should be empty
                if !layer.in_proj_qkv.is_empty() {
                    return Err(format!(
                        "{prefix} in_proj_qkv: attention layer should be empty, got {} elements",
                        layer.in_proj_qkv.len()
                    ));
                }
                if !layer.in_proj_a.is_empty() {
                    return Err(format!(
                        "{prefix} in_proj_a: attention layer should be empty, got {} elements",
                        layer.in_proj_a.len()
                    ));
                }
                if !layer.in_proj_b.is_empty() {
                    return Err(format!(
                        "{prefix} in_proj_b: attention layer should be empty, got {} elements",
                        layer.in_proj_b.len()
                    ));
                }
                if !layer.in_proj_z.is_empty() {
                    return Err(format!(
                        "{prefix} in_proj_z: attention layer should be empty, got {} elements",
                        layer.in_proj_z.len()
                    ));
                }
                if !layer.out_proj.is_empty() {
                    return Err(format!(
                        "{prefix} out_proj: attention layer should be empty, got {} elements",
                        layer.out_proj.len()
                    ));
                }
                if !layer.conv1d_weight.is_empty() {
                    return Err(format!(
                        "{prefix} conv1d_weight: attention layer should be empty, got {} elements",
                        layer.conv1d_weight.len()
                    ));
                }
                if !layer.a_log.is_empty() {
                    return Err(format!(
                        "{prefix} a_log: attention layer should be empty, got {} elements",
                        layer.a_log.len()
                    ));
                }
                if !layer.dt_bias.is_empty() {
                    return Err(format!(
                        "{prefix} dt_bias: attention layer should be empty, got {} elements",
                        layer.dt_bias.len()
                    ));
                }
                if !layer.linear_norm.is_empty() {
                    return Err(format!(
                        "{prefix} linear_norm: attention layer should be empty, got {} elements",
                        layer.linear_norm.len()
                    ));
                }
            }

            // Common fields (both types)
            if layer.gate_proj.len() != mlp * n {
                return Err(format!(
                    "{prefix} gate_proj: expected {} elements, got {}",
                    mlp * n,
                    layer.gate_proj.len()
                ));
            }
            if layer.up_proj.len() != mlp * n {
                return Err(format!(
                    "{prefix} up_proj: expected {} elements, got {}",
                    mlp * n,
                    layer.up_proj.len()
                ));
            }
            if layer.down_proj.len() != n * mlp {
                return Err(format!(
                    "{prefix} down_proj: expected {} elements, got {}",
                    n * mlp,
                    layer.down_proj.len()
                ));
            }
            if layer.input_norm.len() != n {
                return Err(format!(
                    "{prefix} input_norm: expected {n} elements, got {}",
                    layer.input_norm.len()
                ));
            }
            if layer.post_attn_norm.len() != n {
                return Err(format!(
                    "{prefix} post_attn_norm: expected {n} elements, got {}",
                    layer.post_attn_norm.len()
                ));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;

    /// GOAT proof T4: verify weight shapes match config for all-Attention layout.
    #[test]
    fn test_weights_all_attention() {
        let config = Config::qwen_deltanet(24, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        weights.verify_shapes(&config).expect("shapes should match");
    }

    /// GOAT proof T4: verify weight shapes for hybrid layout (Qwen 3.5-0.8B actual).
    #[test]
    fn test_weights_hybrid_layers() {
        use crate::types::DeltaNetLayerType::*;
        // Actual Qwen 3.5-0.8B layer layout:
        // L-L-L-F-L-L-L-F-L-L-L-F-L-L-L-F-L-L-L-F-L-L-L-F
        let layer_types = vec![
            DeltaNet, DeltaNet, DeltaNet, Attention, // 0-3
            DeltaNet, DeltaNet, DeltaNet, Attention, // 4-7
            DeltaNet, DeltaNet, DeltaNet, Attention, // 8-11
            DeltaNet, DeltaNet, DeltaNet, Attention, // 12-15
            DeltaNet, DeltaNet, DeltaNet, Attention, // 16-19
            DeltaNet, DeltaNet, DeltaNet, Attention, // 20-23
        ];
        let config = Config::qwen_deltanet(24, layer_types);
        let weights = QwenDeltaNetWeights::zeros(&config);
        weights.verify_shapes(&config).expect("shapes should match");
    }

    /// GOAT proof T4: verify all-DeltaNet layout.
    #[test]
    fn test_weights_all_deltanet() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet; 12];
        let config = Config::qwen_deltanet(12, layer_types);
        let weights = QwenDeltaNetWeights::zeros(&config);
        weights.verify_shapes(&config).expect("shapes should match");
    }

    /// GOAT proof T4: mismatched layer count should fail validation.
    #[test]
    fn test_weights_wrong_layer_count() {
        let layer_types = vec![DeltaNetLayerType::DeltaNet; 4];
        let config = Config::qwen_deltanet(8, layer_types.clone());
        assert!(
            config.validate().is_err(),
            "should fail: layer_types.len() != n_layer"
        );
    }

    /// GOAT proof T4: verify config validate catches bad `state_dim`.
    #[test]
    fn test_config_bad_state_dim() {
        let mut config = Config::qwen_deltanet(8, vec![]);
        config.deltanet_state_dim = 0; // must be > 0
        assert!(config.validate().is_err());
    }

    /// Verify tied embeddings: wte and `lm_head` have same length in `zeros()`.
    #[test]
    fn test_tied_embeddings() {
        let config = Config::qwen_deltanet(8, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        assert_eq!(weights.wte.len(), weights.lm_head.len());
        assert_eq!(weights.wte.len(), config.vocab_size * config.n_embd);
    }
}
