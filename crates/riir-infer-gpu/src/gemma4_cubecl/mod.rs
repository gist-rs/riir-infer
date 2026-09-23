//! CubeCL integration for the Gemma-4 GPU forward pass (Phase 1: forward-only).
//!
//! Hybrid CPU/CubeCL decode forward pass for Gemma-4-12B inference, adapted
//! from the Gemma-2 CubeCL path (`crate::gemma2_cubecl`):
//! - **CubeCL (GPU)**: GEMV (Q/K/V, Wo, gate/up/down, lm_head) + flash attention
//! - **CPU fallback**: RMSNorm, partial RoPE, QK-Norm, V-RMSNorm, GeGLU,
//!   residual add, layer output scale
//!
//! # Gemma-4 architectural deltas vs Gemma-2
//!
//! | Feature | Gemma-2 | Gemma-4 |
//! |---------|---------|---------|
//! | Attention layers | All full | Alternating Sliding (5) + Full (1) |
//! | KV dims per layer | Uniform | Sliding: `n_kv_head*head_dim`, Full: `n_global_kv_head*global_head_dim` |
//! | Sliding window | None | `config.sliding_window` (1024) on Sliding layers |
//! | RoPE | Full rotation, single theta | Partial (Full layers rotate first 128 of 512), two thetas |
//! | QK-Norm | None | Per-head RMSNorm on Q + K before RoPE |
//! | V-RMSNorm | None | RMSNorm (no gamma) on V before cache store |
//! | Attention scale | `1/sqrt(head_dim)` | `1.0` (no pre-scaling) |
//! | Layer output scale | None | Per-layer scalar after residual (Issue 397) |
//! | Final logit softcap | 30.0 | 30.0 (same) |
//!
//! # Sync Points (4 per layer — same shape as Gemma-2)
//!
//! | Sync | GPU Kernels | CPU Ops After Sync |
//! |------|------------|-------------------|
//! | 1 | Q + K + V GEMVs | QK-Norm, partial RoPE, V-RMSNorm, store K/V in cache |
//! | 2 | Attention + Wo GEMV | RMSNorm + residual add + layer output scale |
//! | 3 | Gate + Up GEMVs | GeGLU |
//! | 4 | Down GEMV | RMSNorm + residual add |
//!
//! # Phase 1 scope (this module)
//!
//! - Forward-only (no backward/training — Phase 2).
//! - f32 weights only (f16 / Q4_K weight formats deferred to Phase 3).
//! - CPU-side attention window slicing (the GPU attention kernel itself is
//!   unchanged from Gemma-2; the sliding-window mask is handled by slicing
//!   the KV cache window before upload, matching the CPU reference).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

use crate::attention_cubecl::{AttentionCubeCL, AttentionParams};
use riir_infer_core::transformer::gemma4::Gemma4TransformerWeights;
use riir_infer_core::types::{Config, Gemma4LayerType};

// ── Submodules ─────────────────────────────────────────────────────

pub mod kv_cache;
pub mod weight_buffers;
pub mod dispatch;

pub use kv_cache::Gemma4CpuKVCache;
pub use weight_buffers::{
    Gemma4CubeCLWeightBuffers, Gemma4NormGammas, per_layer_attn_params, per_layer_cache_dims,
};

// ── CPU fallback operations ────────────────────────────────────────
// These match the riir-engine CPU reference exactly. They run at GPU sync
// points in the hybrid forward path.

/// RMSNorm with learnable gamma: `x[i] = x[i] * rsqrt(mean(x²) + eps) * gamma[i]`.
///
/// Operates on the first `dim` elements of `data`. The gamma vector must have
/// at least `dim` elements. Gemma-4 gammas have the +1 offset pre-applied
/// during GGUF weight loading.
pub fn rmsnorm_gamma(data: &mut [f32], gamma: &[f32], dim: usize, eps: f32) {
    let sum_sq: f32 = data[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        data[i] = data[i] * inv_rms * gamma[i];
    }
}

/// RMSNorm WITHOUT gamma (Issue 397 / llama.cpp convention): applied to V
/// before storing in the KV cache. Normalizes V to unit RMS.
pub fn rmsnorm_no_gamma(data: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = data[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for v in data[..dim].iter_mut() {
        *v *= inv_rms;
    }
}

/// Apply partial rotary RoPE to a Q or K buffer in-place.
///
/// Delegates to `riir_infer_core::transformer::gemma4::apply_partial_rope`, which
/// implements the rotate-half convention: only the first `n_rot` dims of each
/// head are rotated (paired as `(i, i + n_rot/2)`); the remaining
/// `head_dim - n_rot` dims pass through unchanged.
///
/// - Sliding layers: `n_rot == head_dim` (full rotation).
/// - Full layers: `n_rot == global_head_dim * partial_rotary_factor` (e.g. 128 of 512).
pub fn apply_partial_rope_gpu(
    buf: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_rot: usize,
    freq_table: &[f32],
) {
    riir_infer_core::transformer::gemma4::apply_partial_rope(buf, pos, head_dim, n_rot, freq_table);
}

/// Tanh GELU approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GeGLU activation: `out[i] = GELU(gate[i]) * up[i]`.
pub fn geglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = gelu_tanh(g) * u;
    }
}

/// Logit softcapping: `out[i] = cap * fast_tanh(x[i] / cap)`.
///
/// Uses the Padé [2/2] `fast_tanh` — the SAME approximation the CPU
/// reference's softcap uses (`forward_gemma4_impl`, e7d28c1bc) — keeping
/// the GPU/CPU implementations symmetric (Issue 957, the gemma2 twin).
pub fn softcap(data: &mut [f32], cap: f32) {
    for x in data.iter_mut() {
        *x = cap * katgpt_core::simd::fast_tanh(*x / cap);
    }
}

// ── Main GPU forward struct ────────────────────────────────────────

/// CubeCL-accelerated Gemma-4 forward pass (hybrid CPU/GPU decode).
///
/// Holds the CubeCL compute client, uploaded GEMV weight handles, CPU norm
/// gamma copies, and the per-layer-dim KV cache. Construct once via [`new`],
/// then call [`forward`] for each decode step.
///
/// **Phase 1 (this struct):** forward-only, f32 weights, hybrid CPU/GPU.
/// The GPU does GEMV + attention; the CPU does RMSNorm, partial RoPE, QK-Norm,
/// V-RMSNorm, GeGLU, and residual adds at 4 sync points per layer.
pub struct GpuGemma4CubeCL {
    /// CubeCL compute client (shares the same wgpu Device/Queue as GpuContext).
    pub client: ComputeClient<ActiveRuntime>,
    /// Model configuration (Gemma-4-12B constants).
    pub config: Config,
    /// CubeCL f32 GEMV weight handles (uploaded once, cloned per launch).
    pub weights: Gemma4CubeCLWeightBuffers,
    /// CPU RMSNorm gamma vectors (for CPU-side RMSNorm fallback).
    pub norm_gammas: Gemma4NormGammas,
    /// Per-layer-dim CPU KV cache (Sliding = ring buffer, Full = unbounded).
    pub kv_cache: Gemma4CpuKVCache,
    /// CPU embedding weights (for embedding lookup).
    pub wte_cpu: Vec<f32>,
    /// GEMV autotune cache (plane vs tiled, per (m, n) dimension pair).
    pub gemv_autotune: crate::gemv_autotune::GemvAutotune,
    /// Pre-computed per-layer attention parameters (q_dim, kv_dim, head_dim, etc.).
    pub layer_params: Vec<weight_buffers::Gemma4LayerAttnParams>,
    /// RoPE frequency tables: index 0 = sliding (theta=10_000, head_dim),
    /// index 1 = full (theta=1_000_000, global_head_dim).
    pub rope_freq_tables: [Vec<f32>; 2],
    /// Scratch buffer for GeGLU output (`mlp_hidden`), reused across layers.
    pub mlp_hidden_scratch: std::cell::RefCell<Vec<f32>>,
    /// Monotonic weight-replacement counter (Issue 687 H2). Incremented on
    /// every [`Self::replace_weights`]. [`Gemma4BackwardHandles`] records the
    /// version it was built at; the backward asserts the versions match so a
    /// silent stale-transposed-weights backward (e.g. a LoRA re-merge between
    /// handle build and backward) fails loudly instead.
    pub weights_version: u64,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuGemma4CubeCL {
    /// Initialize the CubeCL-accelerated Gemma-4 forward pass.
    ///
    /// Uploads all GEMV weights to CubeCL's GPU buffer pool, clones norm gammas
    /// to CPU, and constructs the per-layer-dim KV cache from the config's
    /// `gemma4_layer_types` pattern.
    ///
    /// # Panics
    ///
    /// Panics if `config.gemma4_layer_types.len() != config.n_layer` (the
    /// caller must build the layer-type pattern before construction).
    pub fn new(
        client: ComputeClient<ActiveRuntime>,
        weights: &Gemma4TransformerWeights,
        config: &Config,
    ) -> Self {
        assert_eq!(
            config.gemma4_layer_types.len(),
            config.n_layer,
            "gemma4_layer_types length must equal n_layer"
        );

        let cubecl_weights = Gemma4CubeCLWeightBuffers::from_weights(&client, weights, config);
        let norm_gammas = Gemma4NormGammas::from_weights(weights);
        let (kv_stride, sliding_capacity, _head_dims) = per_layer_cache_dims(config);
        let kv_cache = Gemma4CpuKVCache::new(config.n_layer, kv_stride, sliding_capacity);
        let layer_params = per_layer_attn_params(config);

        // Build the two RoPE frequency tables (sliding + full).
        // Each table has `head_dim / 2` entries: freq[d] = 1 / theta^(2d / head_dim).
        let rope_freq_sliding = build_rope_freq_table(config.rope_theta, config.head_dim);
        let rope_freq_full =
            build_rope_freq_table(config.rope_theta_full, config.global_head_dim);

        Self {
            client,
            config: config.clone(),
            weights: cubecl_weights,
            norm_gammas,
            kv_cache,
            wte_cpu: weights.wte.clone(),
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            layer_params,
            rope_freq_tables: [rope_freq_sliding, rope_freq_full],
            mlp_hidden_scratch: std::cell::RefCell::new(vec![0.0f32; config.mlp_hidden]),
            weights_version: 0,
        }
    }

    /// Replace the GPU weight handles + CPU norm gammas without recreating
    /// the autotune cache or CubeCL client.
    ///
    /// Used by the LoRA training loop: each step merges the LoRA delta into
    /// the working weights, then calls this to refresh the GPU handles. This
    /// avoids the ~seconds of autotune re-benchmarking + JIT recompilation
    /// that happens on every `new()` call.
    ///
    /// The CubeCL client, KV cache structure, layer params, RoPE tables, and
    /// autotune cache are all preserved.
    pub fn replace_weights(&mut self, weights: &Gemma4TransformerWeights) {
        self.weights = Gemma4CubeCLWeightBuffers::from_weights(&self.client, weights, &self.config);
        self.norm_gammas = Gemma4NormGammas::from_weights(weights);
        self.wte_cpu = weights.wte.clone();
        // Any Gemma4BackwardHandles built from the previous weights are now
        // stale — bump the version so the backward's staleness assert fires
        // (Issue 687 H2).
        self.weights_version += 1;
        // Reset KV cache (same as new()).
        for layer in 0..self.config.n_layer {
            self.kv_cache.keys[layer].clear();
            self.kv_cache.values[layer].clear();
            self.kv_cache.n_positions[layer] = 0;
        }
    }

    /// Forward pass: hybrid CPU/CubeCL decode for one token position.
    ///
    /// Returns the logits vector of length `vocab_size` with final logit
    /// softcapping applied (Gemma-4: cap=30.0).
    ///
    /// # Algorithm
    ///
    /// 1. CPU embedding lookup (single row, scaled by `sqrt(n_embd)`).
    /// 2. Per-layer hybrid CPU/CubeCL dispatch (4 sync points each), with
    ///    per-layer-type (Sliding vs Full) dimension dispatch.
    /// 3. CPU final RMSNorm.
    /// 4. CubeCL lm_head GEMV (tied embedding).
    /// 5. CPU final logit softcapping.
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let vocab = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;

        // 1. Embedding lookup (CPU): single row, scaled by sqrt(n_embd).
        let sqrt_n = (n as f32).sqrt();
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * sqrt_n;
        }

        // 2. Per-layer hybrid dispatch.
        for layer_idx in 0..self.config.n_layer {
            hidden = self.forward_layer(hidden, layer_idx, pos);
        }

        // 3. Final RMSNorm (CPU).
        rmsnorm_gamma(&mut hidden, &self.norm_gammas.final_norm, n, eps);

        // 4. lm_head GEMV (tied wte): logits = wte @ hidden.
        let mut logits = self.dispatch_gemv(&self.weights.wte, &hidden, vocab, n);

        // 5. Final logit softcapping.
        if self.config.final_logit_softcapping > 0.0 {
            softcap(&mut logits, self.config.final_logit_softcapping);
        }

        logits
    }

    /// Process a single transformer layer (hybrid CPU/CubeCL, 4 sync points).
    ///
    /// Dispatches per layer type (Sliding vs Full):
    /// - Sliding: `head_dim`, `n_kv_head`, sliding-window attention.
    /// - Full: `global_head_dim`, `n_global_kv_head`, full attention, partial RoPE.
    fn forward_layer(&mut self, mut hidden: Vec<f32>, layer_idx: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let norms = &self.norm_gammas.layers[layer_idx];
        let lp = &self.layer_params[layer_idx];
        let layer_type = self.config.gemma4_layer_types[layer_idx];

        let q_dim = lp.q_dim;
        let kv_dim = lp.kv_dim;
        let head_dim = lp.head_dim;
        let n_head = lp.n_head;
        let n_kv_head = lp.n_kv_head;
        let n_rot = lp.rope_rot_dim;
        let rope_freq = &self.rope_freq_tables[lp.rope_table_idx];

        // Save residual.
        let residual = hidden.clone();

        // ── Pass A: QKV + QK-Norm + partial RoPE + V-norm ─────────────

        // CPU: input RMSNorm.
        rmsnorm_gamma(&mut hidden, &norms.input_norm, n, eps);

        // GPU: QKV GEMVs (batched, 1 sync) → Sync 1.
        let (mut q, mut k, mut v) =
            self.dispatch_qkv(&self.weights.layers[layer_idx], &hidden, q_dim, kv_dim, n);

        // CPU: QK-Norm (NEW vs Gemma-2). Per-head RMSNorm over head_dim.
        // attn_q_norm / attn_k_norm length == head_dim (shared across heads).
        for h in 0..n_head {
            let off = h * head_dim;
            rmsnorm_gamma(&mut q[off..off + head_dim], &norms.attn_q_norm, head_dim, eps);
        }
        for h in 0..n_kv_head {
            let off = h * head_dim;
            rmsnorm_gamma(&mut k[off..off + head_dim], &norms.attn_k_norm, head_dim, eps);
        }

        // CPU: partial RoPE on Q and K (V is not rotated).
        apply_partial_rope_gpu(&mut q, pos, head_dim, n_rot, rope_freq);
        apply_partial_rope_gpu(&mut k, pos, head_dim, n_rot, rope_freq);

        // CPU: V RMSNorm (no gamma) before cache store (Issue 397 / llama.cpp).
        rmsnorm_no_gamma(&mut v, kv_dim, eps);

        // CPU: store K, V in per-layer KV cache.
        self.kv_cache.store(layer_idx, pos, &k, &v);

        // ── Pass B: Attention + Wo + residual ────────────────────────

        // Compute the attention window for this layer type.
        let (t_start, n_pos) = match layer_type {
            Gemma4LayerType::Sliding => {
                let sw = self.config.sliding_window;
                let start = pos.saturating_sub(sw.saturating_sub(1));
                (start, pos - start + 1)
            }
            Gemma4LayerType::Full => (0, pos + 1),
        };

        // GPU: Attention + Wo (2 launches, 1 sync) → Sync 2.
        let wo_out = self.dispatch_attention_wo(
            &self.weights.layers[layer_idx],
            &q,
            layer_idx,
            t_start,
            n_pos,
            head_dim,
            n_head,
            n_kv_head,
            n,
            q_dim,
        );

        // CPU: post-attention RMSNorm + residual add (Gemma-2-style post-norm).
        let mut hidden = wo_out;
        rmsnorm_gamma(&mut hidden, &norms.post_attn_norm, n, eps);
        for (h, r) in hidden.iter_mut().zip(residual.iter()) {
            *h += r;
        }

        // Save residual 2.
        let residual2 = hidden.clone();

        // ── Pass C: MLP ──────────────────────────────────────────────

        // CPU: pre-MLP RMSNorm.
        rmsnorm_gamma(&mut hidden, &norms.pre_mlp_norm, n, eps);

        // GPU: Gate + Up GEMVs (batched, 1 sync) → Sync 3.
        let (gate, up) = self.dispatch_gate_up(&self.weights.layers[layer_idx], &hidden, mlp, n);

        // CPU: GeGLU — reuse the persistent scratch buffer.
        let mut mlp_hidden = self.mlp_hidden_scratch.borrow_mut();
        debug_assert_eq!(mlp_hidden.len(), mlp);
        geglu(&gate, &up, &mut mlp_hidden);

        // GPU: Down GEMV (1 sync) → Sync 4.
        let mut down_out = self.dispatch_gemv(
            &self.weights.layers[layer_idx].down_proj,
            &mlp_hidden,
            n,
            mlp,
        );
        drop(mlp_hidden);

        // CPU: post-MLP RMSNorm + residual add.
        rmsnorm_gamma(&mut down_out, &norms.post_mlp_norm, n, eps);
        for (h, r) in down_out.iter_mut().zip(residual2.iter()) {
            *h += r;
        }

        // CPU: layer output scale (Issue 397). Applied after the second
        // residual to prevent activation explosion. Default 1.0 (no-op).
        if norms.layer_output_scale != 1.0 {
            for h in down_out.iter_mut() {
                *h *= norms.layer_output_scale;
            }
        }

        down_out
    }
}

/// Build a RoPE frequency table: `freq[d] = 1 / theta^(2d / head_dim)` for
/// `d in 0..head_dim/2`.
///
/// This matches the `RopeFreqTable` substrate in riir-engine. We rebuild it
/// here (rather than depending on `RopeFreqTable` directly) to keep the GPU
/// module's dependency surface minimal — the table is just a `Vec<f32>`.
fn build_rope_freq_table(theta: f32, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|d| 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32))
        .collect()
}
