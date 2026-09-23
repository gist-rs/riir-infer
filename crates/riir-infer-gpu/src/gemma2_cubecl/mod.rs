//! CubeCL integration for Gemma 2 GPU forward pass (Plan 106 T2.6).
//!
//! Hybrid CPU/CubeCL decode forward pass for Gemma 2 inference:
//! - **CubeCL (GPU)**: GEMV (Q/K/V, Wo, gate/up/down, lm_head) + flash attention
//! - **CPU fallback**: RMSNorm, RoPE, GeGLU, residual add
//!
//! # Sync Points (4 per layer)
//!
//! Each layer has 4 GPU↔CPU synchronization points where data transfers
//! between CubeCL-managed GPU buffers and CPU:
//!
//! | Sync | GPU Kernels | CPU Ops After Sync |
//! |------|------------|-------------------|
//! | 1 | Q + K + V GEMVs | RoPE(Q), store K/V in cache |
//! | 2 | Attention + Wo GEMV (batched) | RMSNorm + residual add |
//! | 3 | Gate + Up GEMVs | GeGLU |
//! | 4 | Down GEMV | RMSNorm + residual add |
//!
//! On Apple Silicon unified memory, each ~KB transfer adds <0.05ms.
//! Total sync overhead per layer: ~0.2ms. For 26 layers: ~5ms.
//! This is acceptable for T2.6 wiring milestone (~7% of 75ms budget).
//!
//! # Buffer Lifecycle
//!
//! ```text
//! Init:
//!   CPU weights ──create_from_slice──► CubeCL Handles (persist for lifetime)
//!   CPU norm gammas ──clone──► Vec<f32> (persist, used for CPU RMSNorm)
//!
//! Per forward pass:
//!   CPU hidden ──create_from_slice──► CubeCL Handle
//!   CubeCL kernel launch (handle-to-handle, no sync)
//!   CubeCL Handle ──read_one──► CPU Vec<f32>
//! ```
//!
//! Weight Handles are cloned per kernel launch (cheap ref-count increment).
//! The underlying GPU buffer persists until all clones are dropped.
//!
//! # Future Optimization (Track 3)
//!
//! - CubeCL RMSNorm/RoPE/GeGLU/add kernels → eliminate all sync points
//! - GPU-resident KV cache → eliminate per-attention upload
//! - Attention + Wo fusion (CODA epilogue) → reduce sync point 2
//! - Gate + Up + GeGLU + Down fusion → reduce sync points 3+4

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

use crate::attention_cubecl::{AttentionCubeCL, AttentionParams};
#[cfg(feature = "q8_kv_cache")]
use crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_f16_cubecl::{F16Handle, GemvF16CubeCL};
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_geglu_cubecl::GemvGegluCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_geglu_f16_cubecl::GemvGegluF16CubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_geglu_q4k_cubecl::GemvGegluQ4KCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_q4k_cubecl::{GemvQ4KCubeCL, Q4KHandle};

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_qkv_f16_cubecl::GemvQkvF16CubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_qkv_q4k_cubecl::GemvQkvQ4KCubeCL;
use crate::norms_cubecl::{ResidualAddCubeCL, RmsNormCubeCL};
use crate::rope_geglu_cubecl::{GegluCubeCL, RopeCubeCL, RopeFromCombinedCubeCL};
#[cfg(feature = "gpu_decode_fusion")]
use crate::sampling_cubecl::ArgmaxCubeCL;
#[cfg(feature = "wall_attention")]
use crate::wall_cubecl::{WallRescaleCubeCL, WallRescaleFromCombinedCubeCL};
use bytemuck::Zeroable;
use riir_infer_core::gemma_layer::GemmaTransformerWeights;
use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};
#[cfg(feature = "gemma_lora")]
use riir_infer_core::transformer::gemma2_lora::GemmaLayerLora;
use riir_infer_core::types::Config;
#[cfg(feature = "wall_attention")]
use riir_infer_core::wall_config::WallConfig;

// ── CPU fallback operations ────────────────────────────────────────
// Used until equivalent CubeCL kernels are implemented (Track 3).

/// RMSNorm with learnable gamma: `x[i] = x[i] * rsqrt(mean(x²) + eps) * gamma[i]`.
///
/// Gemma 2 gamma has +1 offset pre-applied during weight loading.
/// Operates in-place on `data`.
#[cfg(feature = "cubecl_runtime")]
pub fn rmsnorm_gamma(data: &mut [f32], gamma: &[f32], dim: usize, eps: f32) {
    // Pass 1: sum of squares
    let sum_sq: f32 = data[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

    // Pass 2: normalize and apply gamma
    for i in 0..dim {
        data[i] = data[i] * inv_rms * gamma[i];
    }
}

/// Rotary Position Embedding (interleaved, applied in-place).
///
/// For each head, for each pair `(x[d], x[d + head_dim/2])`:
/// ```text
/// freq   = 1 / theta^(2d / head_dim)
/// angle  = pos * freq
/// x'[d]        = x[d] * cos(angle) - x[d + half] * sin(angle)
/// x'[d + half] = x[d] * sin(angle) + x[d + half] * cos(angle)
/// ```
///
/// **Rotate-half** ([`RopePairing::RotateHalf`]), matching HuggingFace's
/// `rotate_half` and `riir_infer_core::rope::apply_rope_with_freq` — the CPU
/// reference this GPU path is validated against, and the convention the
/// Gemma-2 GGUF weights were trained under.
///
/// Issue 435: this helper previously paired adjacent components `(2d, 2d+1)`
/// while the CPU reference paired half-split ones. The rotation stayed a valid
/// rotation, so nothing errored — attention was just computed against scrambled
/// head components, costing ~8× on training CE (6.33 vs 0.79 at step 1).
#[cfg(feature = "cubecl_runtime")]
pub fn apply_rope(
    data: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_heads: usize,
    theta: f32,
) {
    let half_dim = head_dim / 2;
    for head in 0..n_heads {
        let base = head * head_dim;
        for d in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = base + d;
            let idx1 = base + d + half_dim;
            let x0 = data[idx0];
            let x1 = data[idx1];

            data[idx0] = x0 * cos_val - x1 * sin_val;
            data[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/// Tanh GELU approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
#[cfg(feature = "cubecl_runtime")]
fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GeGLU activation: `out[i] = GELU(gate[i]) * up[i]`.
///
/// Note: `gelu_tanh(g)` already includes the `g` factor
/// (`GELU(x) = 0.5 * x * (1 + tanh(...))`), so we do NOT multiply by `g` again.
#[cfg(feature = "cubecl_runtime")]
pub fn geglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = gelu_tanh(g) * u;
    }
}

/// Apply a single LoRA delta to `output` in-place: `output += (α/r)·B@(A@input)`.
///
/// Thin wrapper around `katgpt_core::types::lora_apply`, sized to the adapter's
/// rank. Called at each GPU sync point during the hybrid forward pass
/// (Plan 410 Phase 3A.1). The LoRA rank is small (typically ≤16), so the CPU
/// compute is negligible relative to the base GPU GEMV.
///
/// `scratch` is reused across all 7 insertion points per layer — grown to
/// `max_rank` once, never re-allocated in the hot path.
#[cfg(feature = "gemma_lora")]
#[inline]
pub fn apply_lora_delta(
    output: &mut [f32],
    input: &[f32],
    adapter: Option<&katgpt_core::types::LoraAdapter>,
    scratch: &mut Vec<f32>,
) {
    if let Some(adp) = adapter {
        let r = adp.rank;
        if scratch.len() < r {
            scratch.resize(r, 0.0);
        }
        katgpt_core::types::lora_apply(output, adp, input, &mut scratch[..r]);
    }
}

/// Logit softcapping: `out[i] = cap * fast_tanh(x[i] / cap)`.
///
/// Uses the Padé [2/2] `fast_tanh` — the SAME approximation the CPU
/// reference's softcap uses (e7d28c1bc, 2026-07-24). Before Issue 957 this
/// site used std `f32::tanh`, so GPU logits diverged from the CPU reference
/// by up to 30·0.025 ≈ 0.75 at the Padé worst point (|x|≈2) — a pure
/// implementation asymmetry, not GPU precision (the hybrid path's hidden
/// state agrees to <0.001 per layer).
#[cfg(feature = "cubecl_runtime")]
pub fn softcap(data: &mut [f32], cap: f32) {
    for x in data.iter_mut() {
        *x = cap * katgpt_core::simd::fast_tanh(*x / cap);
    }
}

// ── Speculative Decoding Result ───────────────────────────────────

/// Result of speculative decoding with early-exit draft (Plan 171 T34).
///
/// Contains the generated tokens and acceptance statistics for
/// performance analysis.
#[cfg(feature = "gpu_decode_fusion")]
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Generated token IDs (excluding prompt tokens).
    pub tokens: Vec<usize>,
    /// Total draft tokens generated across all speculation rounds.
    pub total_draft_tokens: usize,
    /// Total draft tokens that matched the full model's prediction.
    pub total_accepted: usize,
    /// Number of speculation rounds executed.
    pub speculation_rounds: usize,
}

#[cfg(feature = "gpu_decode_fusion")]
impl SpeculativeResult {
    /// Acceptance rate: fraction of draft tokens that were accepted.
    ///
    /// Returns 0.0 if no draft tokens were generated (edge case).
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_draft_tokens == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_draft_tokens as f64
        }
    }

    /// Average draft tokens accepted per speculation round.
    pub fn avg_accepted_per_round(&self) -> f64 {
        if self.speculation_rounds == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.speculation_rounds as f64
        }
    }
}

// ── KV cache types (extracted to submodule) ────────────────────────

pub mod kv_cache;
pub use kv_cache::*;

// ── F16 weight buffers (extracted to submodule) ───────────────────

pub mod weight_buffers;
pub use weight_buffers::*;

// ── Dispatch helpers (extracted to submodule, Issue 003) ──────────

pub mod dispatch;

// ── GPU forward / generate methods (extracted to submodule, Issue 003) ──

pub mod gpu_forward;

/// Per-layer CubeCL weight handles for GEMV operations.
///
/// Each handle wraps a GPU buffer uploaded once during init.
/// Handles are cloned per kernel launch (cheap ref-count increment).
#[cfg(feature = "cubecl_runtime")]
pub struct CubeCLLayerWeights {
    pub attn_wq: Handle,
    pub attn_wk: Handle,
    pub attn_wv: Handle,
    pub attn_wo: Handle,
    pub gate_proj: Handle,
    pub up_proj: Handle,
    pub down_proj: Handle,
}

/// **Plan 409 Phase 3 (2026-07-09):** Delta routing state for the GPU hybrid path.
///
/// Replicates the CPU `ForwardContext::block_deltas` + scratch buffers so the
/// GPU path can apply the same per-block delta routing that the CPU forward does.
/// Without this, the GPU output diverges from CPU at every 4th layer (block boundary).
#[cfg(feature = "delta_routing")]
pub struct DeltaRoutingState {
    /// Accumulated deltas per block: `[n_blocks][n_embd]`, zero-init.
    block_deltas: Vec<Vec<f32>>,
    /// Delta routing query weights: `[n_layer][n_embd]`, zero-init for pretrained models.
    query_weights: Vec<Vec<f32>>,
    /// Delta routing norm weights: `[n_layer][n_embd]`, one-init (identity RMSNorm).
    norm_weights: Vec<Vec<f32>>,
    /// Scratch buffer for softmax logits: `[n_blocks]`.
    logits_buf: Vec<f32>,
    /// Scratch buffer for query·norm product: `[n_embd]`.
    qn_buf: Vec<f32>,
    /// Embedding dimension.
    n_embd: usize,
}

#[cfg(feature = "delta_routing")]
impl DeltaRoutingState {
    const BLOCK_SIZE: usize = 4;

    fn new(weights: &GemmaTransformerWeights, config: &Config) -> Self {
        let n_blocks = config.n_layer.div_ceil(Self::BLOCK_SIZE);
        Self {
            block_deltas: vec![vec![0.0; config.n_embd]; n_blocks],
            query_weights: weights.delta_routing_query.clone(),
            norm_weights: weights.delta_routing_norm.clone(),
            logits_buf: vec![0.0; n_blocks],
            qn_buf: vec![0.0; config.n_embd],
            n_embd: config.n_embd,
        }
    }

    /// Apply the delta routing step after a layer's residual add.
    ///
    /// `hidden` is the post-residual-add hidden state (will be modified at
    /// block boundaries). `pre_layer_residual` is the residual saved BEFORE
    /// this layer's RMSNorm (i.e., the previous layer's output).
    pub fn apply_step(
        &mut self,
        hidden: &mut [f32],
        pre_layer_residual: &[f32],
        layer_idx: usize,
    ) {
        let block_idx = layer_idx / Self::BLOCK_SIZE;
        let pos_in_block = layer_idx % Self::BLOCK_SIZE;

        // Accumulate delta: current x minus pre-layer residual
        if block_idx < self.block_deltas.len() {
            for d in 0..self.n_embd {
                self.block_deltas[block_idx][d] += hidden[d] - pre_layer_residual[d];
            }
        }

        // At block boundary: route accumulated deltas
        if pos_in_block == Self::BLOCK_SIZE - 1 && block_idx < self.block_deltas.len() {
            self.depth_route(hidden, 0..=block_idx, layer_idx);
            self.block_deltas[block_idx].fill(0.0);
        }
    }

    /// Depth routing: compute softmax-weighted sum of block deltas, add to residual.
    /// Replicates `riir_infer_core::transformer::depth_route` exactly.
    fn depth_route(
        &mut self,
        residual: &mut [f32],
        source_range: std::ops::RangeInclusive<usize>,
        layer_idx: usize,
    ) {
        let start = *source_range.start();
        let end = (*source_range.end()).min(self.block_deltas.len().saturating_sub(1));
        if end < start {
            return;
        }
        let n_sources = end - start + 1;

        let query = &self.query_weights[layer_idx];
        let norm = &self.norm_weights[layer_idx];

        // Pre-compute query·norm product
        for d in 0..self.n_embd {
            self.qn_buf[d] = query[d] * norm[d];
        }

        // Compute logits (RMSNorm + dot product with query·norm)
        let eps = 1e-5f32;
        let mut max_logit = f32::NEG_INFINITY;
        for i in 0..n_sources {
            let src_idx = start + i;
            let src = &self.block_deltas[src_idx];
            let sum_sq: f32 = src.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / self.n_embd as f32 + eps).sqrt();
            let mut logit = 0.0f32;
            for (&src_d, qn) in src.iter().zip(self.qn_buf.iter()).take(self.n_embd) {
                logit += qn * src_d * inv_rms;
            }
            self.logits_buf[i] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for i in 0..n_sources {
            let exp_val = (self.logits_buf[i] - max_logit).exp();
            self.logits_buf[i] = exp_val;
            sum_exp += exp_val;
        }
        let inv_sum = 1.0 / sum_exp;

        // Weighted sum of sources, added to residual
        for i in 0..n_sources {
            let src_idx = start + i;
            let w = self.logits_buf[i] * inv_sum;
            let src = &self.block_deltas[src_idx];
            for d in 0..self.n_embd {
                residual[d] += w * src[d];
            }
        }
    }
}

/// **Plan 409 Phase 3:** Per-layer intermediate trace for drift isolation.
///
/// Captured by [`GpuGemmaCubeCL::forward_layer_debug`]. Each field is the
/// output of one stage in the layer pipeline, cloned to CPU for comparison
/// against the CPU reference.
#[cfg(feature = "cubecl_runtime")]
#[derive(Debug, Clone)]
pub struct LayerDebugTrace {
    /// Hidden after input RMSNorm (input to QKV GEMV).
    pub normed_input: Vec<f32>,
    /// Q after QKV GEMV (pre-RoPE).
    pub q: Vec<f32>,
    /// K after QKV GEMV (pre-RoPE).
    pub k: Vec<f32>,
    /// V after QKV GEMV.
    pub v: Vec<f32>,
    /// Wo output (after attention + Wo GEMV). This is the full attention block output.
    pub wo_out: Vec<f32>,
    /// Hidden after post-attn RMSNorm + residual add.
    pub residual1: Vec<f32>,
    /// Hidden after pre-MLP RMSNorm (input to gate/up GEMV).
    pub pre_mlp_normed: Vec<f32>,
    /// Gate projection output.
    pub gate: Vec<f32>,
    /// Up projection output.
    pub up: Vec<f32>,
    /// GeGLU activation output (input to down GEMV).
    pub geglu: Vec<f32>,
    /// Down projection output.
    pub down_out: Vec<f32>,
    /// Final layer output (after post-MLP RMSNorm + residual add).
    pub hidden_out: Vec<f32>,
}

/// All CubeCL GEMV weight handles for Gemma 2 inference.
#[cfg(feature = "cubecl_runtime")]
pub struct CubeCLWeightBuffers {
    pub layers: Vec<CubeCLLayerWeights>,
    pub wte: Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl CubeCLWeightBuffers {
    /// Upload all GEMV weights to CubeCL's buffer pool.
    ///
    /// Creates persistent GPU buffers via `create_from_slice`.
    /// Handles are cloned per launch; underlying buffers persist until
    /// all clones are dropped (i.e., until `CubeCLWeightBuffers` is dropped).
    pub fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
    ) -> Self {
        let wte = client.create_from_slice(f32::as_bytes(&weights.wte));

        let layers = weights
            .layers
            .iter()
            .map(|l| CubeCLLayerWeights {
                attn_wq: client.create_from_slice(f32::as_bytes(&l.attn_wq)),
                attn_wk: client.create_from_slice(f32::as_bytes(&l.attn_wk)),
                attn_wv: client.create_from_slice(f32::as_bytes(&l.attn_wv)),
                attn_wo: client.create_from_slice(f32::as_bytes(&l.attn_wo)),
                gate_proj: client.create_from_slice(f32::as_bytes(&l.gate_proj)),
                up_proj: client.create_from_slice(f32::as_bytes(&l.up_proj)),
                down_proj: client.create_from_slice(f32::as_bytes(&l.down_proj)),
            })
            .collect();

        Self { layers, wte }
    }
}

// ── CubeCL Q4_K weight handles ─────────────────────────────────────

/// Per-layer CubeCL Q4_K weight handles for dequant+GEMV operations.
///
/// Each projection is stored as a [`Q4KHandle`] containing packed Q4_K blocks
/// (as u32 array) and pre-decoded d/dmin f32 values. The CubeCL kernel
/// dequantizes on-the-fly during the dot product.
#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // qkv_combined/gate_up_combined are Plan 171 fused-path scaffolding (Issue 429 clippy).
pub struct CubeCLQ4KLayerWeights {
    pub attn_wq: Q4KHandle,
    pub attn_wk: Q4KHandle,
    pub attn_wv: Q4KHandle,
    /// Combined [Wq|Wk|Wv] Q4_K weights for fused triple QKV GEMV (Plan 171 T28).
    pub qkv_combined: Option<crate::gemv_qkv_q4k_cubecl::Q4KQKVHandle>,
    pub attn_wo: Q4KHandle,
    pub gate_proj: Q4KHandle,
    pub up_proj: Q4KHandle,
    /// Combined [W_gate|W_up] Q4_K weights for fused GeGLU (Plan 171 T29).
    pub gate_up_combined: Option<crate::gemv_geglu_q4k_cubecl::Q4KGegluHandle>,
    pub down_proj: Q4KHandle,
}

/// All CubeCL Q4_K weight handles for Gemma 2 inference.
///
/// Stores quantized weights as [`Q4KHandle`] per projection.
/// CPU-side embedding weights (`wte_cpu`) live in [`GpuGemmaCubeCL`].
#[cfg(feature = "cubecl_runtime")]
pub struct CubeCLQ4KWeightBuffers {
    pub layers: Vec<CubeCLQ4KLayerWeights>,
    pub wte: Q4KHandle,
}

#[cfg(feature = "cubecl_runtime")]
impl CubeCLQ4KWeightBuffers {
    /// Quantize and upload all GEMV weights to CubeCL GPU buffers as Q4_K.
    ///
    /// Each f32 projection is quantized row-by-row to Q4_K blocks (4.5 bpw),
    /// then uploaded via [`Q4KHandle::from_blocks`]. This reduces GPU memory
    /// by ~7× compared to f32 weights.
    ///
    /// # Memory
    ///
    /// - GPU: ~0.4 GB for Q4_K weights (vs ~2.7 GB for f32)
    /// - CPU: ~10 MB for wte_cpu (f32 embedding, for lookup only)
    pub fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let wte = Self::upload_projection(client, &weights.wte, config.vocab_size, config.n_embd);

        let layers = weights
            .layers
            .iter()
            .map(|l| {
                let q_dim = config.n_head * config.head_dim;
                let kv_dim = config.n_kv_head * config.head_dim;
                let n = config.n_embd;
                let mlp = config.mlp_hidden;

                let attn_wq = Self::upload_projection(client, &l.attn_wq, q_dim, n);
                let attn_wk = Self::upload_projection(client, &l.attn_wk, kv_dim, n);
                let attn_wv = Self::upload_projection(client, &l.attn_wv, kv_dim, n);
                let gate_proj = Self::upload_projection(client, &l.gate_proj, mlp, n);
                let up_proj = Self::upload_projection(client, &l.up_proj, mlp, n);

                // Build fused combined handles (Plan 171 T28/T29)
                let qkv_combined = Some(crate::gemv_qkv_q4k_cubecl::Q4KQKVHandle::from_separate(
                    client, &attn_wq, &attn_wk, &attn_wv,
                ));
                let gate_up_combined =
                    Some(crate::gemv_geglu_q4k_cubecl::Q4KGegluHandle::from_separate(
                        client, &gate_proj, &up_proj,
                    ));

                CubeCLQ4KLayerWeights {
                    attn_wq,
                    attn_wk,
                    attn_wv,
                    qkv_combined,
                    attn_wo: Self::upload_projection(client, &l.attn_wo, n, q_dim),
                    gate_proj,
                    up_proj,
                    gate_up_combined,
                    down_proj: Self::upload_projection(client, &l.down_proj, n, mlp),
                }
            })
            .collect();

        Self { layers, wte }
    }

    /// Quantize and upload a single projection matrix to Q4_K GPU buffers.
    ///
    /// Pads input dimension to multiple of [`QK_K`] (256) if needed.
    fn upload_projection(
        client: &ComputeClient<ActiveRuntime>,
        data: &[f32],
        m: usize,
        n: usize,
    ) -> Q4KHandle {
        assert_eq!(data.len(), m * n, "Projection size mismatch");
        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;
        let total_blocks = m * blocks_per_row;
        let mut blocks = Vec::with_capacity(total_blocks);

        // Pad row to multiple of QK_K if needed
        let mut padded_row = vec![0.0f32; padded_n];
        for row in 0..m {
            let src = &data[row * n..(row + 1) * n];
            padded_row[..n].copy_from_slice(src);
            if padded_n > n {
                padded_row[n..].fill(0.0);
            }

            let start = blocks.len();
            blocks.resize(start + blocks_per_row, BlockQ4K::zeroed());
            quantize_row_q4_k(&padded_row, &mut blocks[start..]);
        }

        Q4KHandle::from_blocks(client, &blocks, m, padded_n)
    }
}

// ── Weight format enum ─────────────────────────────────────────────

/// Weight storage format for CubeCL GEMV dispatch.
///
/// Selects between f32 (full precision) and Q4_K (quantized, ~7× less memory).
/// The forward pass dispatches the appropriate CubeCL kernel based on this format.
#[cfg(feature = "cubecl_runtime")]
pub enum CubeCLWeightFormat {
    /// Full f32 weights — uses `gemv_plane_f32` / `gemv_tile_f32` kernels.
    F32(CubeCLWeightBuffers),
    /// F16 weights — uses `gemv_plane_f16_f32` / `gemv_tile_f16_f32` kernels
    /// with f16→f32 cast during dot product. ~2× less memory bandwidth.
    F16(CubeCLF16WeightBuffers),
    /// Q4_K quantized weights — uses `gemv_q4k_plane` / `gemv_q4k_tiled` kernels
    /// with fused inline dequantization during the dot product.
    Q4K(CubeCLQ4KWeightBuffers),
}

// ── CPU norm gamma storage ─────────────────────────────────────────

/// Per-layer RMSNorm gamma vectors (kept on CPU for CPU-side RMSNorm).
#[cfg(feature = "cubecl_runtime")]
pub struct LayerNormGammas {
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub pre_mlp_norm: Vec<f32>,
    pub post_mlp_norm: Vec<f32>,
}

/// All RMSNorm gamma vectors for Gemma 2 inference.
#[cfg(feature = "cubecl_runtime")]
pub struct NormGammas {
    pub layers: Vec<LayerNormGammas>,
    pub final_norm: Vec<f32>,
}

#[cfg(feature = "cubecl_runtime")]
impl NormGammas {
    pub fn from_weights(weights: &GemmaTransformerWeights) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|l| LayerNormGammas {
                input_norm: l.input_norm.clone(),
                post_attn_norm: l.post_attn_norm.clone(),
                pre_mlp_norm: l.pre_mlp_norm.clone(),
                post_mlp_norm: l.post_mlp_norm.clone(),
            })
            .collect();

        Self {
            layers,
            final_norm: weights.final_norm.clone(),
        }
    }
}

/// Per-layer GPU handles for RMSNorm gamma vectors (T2.12).
///
/// Each gamma vector is uploaded to CubeCL's buffer pool once at construction.
/// Handles are cloned per kernel launch (cheap ref-count increment).
#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Norm-gamma handles are GPU-side RMSNorm scaffolding (Issue 429 clippy).
pub struct GpuLayerNormGammaHandles {
    pub input_norm: Handle,
    pub post_attn_norm: Handle,
    pub pre_mlp_norm: Handle,
    pub post_mlp_norm: Handle,
}

/// All GPU-resident RMSNorm gamma handles for Gemma 2 inference (T2.12).
///
/// Uploaded once at construction from CPU weight data.
/// Used by [`GpuGemmaCubeCL::dispatch_rmsnorm_gpu`] for GPU-side RMSNorm.
#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // layers field unused until batched norm wired (Issue 429 clippy).
pub struct GpuNormGammaHandles {
    pub layers: Vec<GpuLayerNormGammaHandles>,
    pub final_norm: Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuNormGammaHandles {
    /// Upload all RMSNorm gamma vectors to CubeCL GPU buffers.
    pub fn upload(
        client: &ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
    ) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|l| GpuLayerNormGammaHandles {
                input_norm: client.create_from_slice(f32::as_bytes(&l.input_norm)),
                post_attn_norm: client.create_from_slice(f32::as_bytes(&l.post_attn_norm)),
                pre_mlp_norm: client.create_from_slice(f32::as_bytes(&l.pre_mlp_norm)),
                post_mlp_norm: client.create_from_slice(f32::as_bytes(&l.post_mlp_norm)),
            })
            .collect();

        let final_norm = client.create_from_slice(f32::as_bytes(&weights.final_norm));

        Self { layers, final_norm }
    }
}

// ── Main integration struct ────────────────────────────────────────

/// CubeCL-accelerated Gemma 2 decode forward pass.
///
/// Manages CubeCL weight handles, CPU KV cache, and CPU norm gammas.
/// Provides a `forward(token, pos)` method that produces logits using
/// a hybrid CPU/CubeCL dispatch strategy.
///
/// # Lifecycle
///
/// 1. Create via `GpuGemmaCubeCL::new(client, weights, config)`.
/// 2. Call `forward(token, pos)` for each token in the sequence.
/// 3. CPU KV cache grows with each call (no reset needed between tokens).
///
/// # Example
///
/// ```rust,ignore
/// let ctx = GpuContext::new()?;
/// let client = ctx.cubecl_client();
/// let mut cubecl_fwd = GpuGemmaCubeCL::new(client, &weights, &config);
///
/// let logits = cubecl_fwd.forward(token_id, position);
/// let next_token = sample_token(&logits, &mut rng, 0.0);
/// ```
// ── Wall Attention GPU state (Plan 193 T3) ────────────────────────────
//
// GPU-side state for Wall Attention decode. Maintains running prefix sum
// across decode steps, uploaded to GPU each step for zero-copy rescaling.
// Feature-gated — zero cost when wall_attention is disabled.
#[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
struct WallGpuState {
    /// Wall configuration (gate_bias, gate_max, etc.).
    config: WallConfig,
    /// Running prefix sum: prefix[d] = cumulative sum of log_gates[0..=pos][d].
    /// Length: head_dim. Updated on CPU each decode step, uploaded to GPU.
    prefix_cpu: Vec<f32>,
    /// Gate projection weights W_g: [gate_proj_dim, d_model].
    /// Loaded once from checkpoint.
    w_g: Vec<f32>,
    /// d_model dimension.
    d_model: usize,
}

#[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
impl WallGpuState {
    /// Create new Wall GPU state.
    fn new(config: WallConfig, d_model: usize, head_dim: usize) -> Self {
        let prefix_cpu = vec![0.0f32; head_dim];
        Self {
            config,
            prefix_cpu,
            w_g: Vec::new(),
            d_model,
        }
    }

    /// Create with pre-loaded gate weights.
    fn with_weights(mut self, w_g: Vec<f32>) -> Self {
        self.w_g = w_g;
        self
    }

    /// Get the concatenated prefix buffer [prefix_q | prefix_k] for GPU upload.
    ///
    /// For single-token decode, both Q and K at the current position use the
    /// same prefix sum. The K prefix for cached positions was baked into the
    /// KV cache at store time (same as RoPE'd K).
    fn prefix_qk_buffer(&self) -> Vec<f32> {
        // [prefix_q(head_dim) | prefix_k(head_dim)] — same values for decode
        let mut buf = Vec::with_capacity(self.prefix_cpu.len() * 2);
        buf.extend_from_slice(&self.prefix_cpu);
        buf.extend_from_slice(&self.prefix_cpu);
        buf
    }

    /// Advance decode by one step using CPU hidden data.
    ///
    /// Called from `forward_gpu` with the embedding-lookup hidden state.
    /// This is the CPU path — the gate GEMV (gate_proj_dim × d_model)
    /// is small enough that CPU computation is competitive with GPU upload
    /// overhead (head_dim = 256 floats = 1 KB prefix update).
    fn step_cpu(&mut self, hidden: &[f32]) {
        use riir_infer_core::wall::{wall_gate_project, wall_prefix_decode};
        let head_dim = self.prefix_cpu.len();
        if self.w_g.is_empty() {
            return;
        }
        let gate = wall_gate_project(
            hidden,
            &self.w_g,
            self.config.gate_bias,
            self.config.gate_max,
            self.config.gate_proj_dim,
            self.d_model,
        );
        let log_gate: Vec<f32> = gate.iter().map(|g| g.ln()).collect();
        let log_gate_head = if log_gate.len() >= head_dim {
            &log_gate[..head_dim]
        } else {
            &log_gate
        };
        self.prefix_cpu = wall_prefix_decode(log_gate_head, &self.prefix_cpu);
    }

    /// Reset for new sequence.
    fn reset(&mut self) {
        self.prefix_cpu.fill(0.0);
    }

    /// Whether Wall is active (has weights loaded).
    fn is_active(&self) -> bool {
        !self.w_g.is_empty()
    }
}

#[cfg(feature = "cubecl_runtime")]
pub struct GpuGemmaCubeCL {
    /// CubeCL compute client sharing the same wgpu Device/Queue as GpuContext.
    pub client: ComputeClient<ActiveRuntime>,
    /// Model configuration (Gemma 2 2B constants).
    pub config: Config,
    /// CubeCL GEMV weight handles (uploaded once, cloned per launch).
    /// F32 or Q4_K format selected at construction time.
    pub weights: CubeCLWeightFormat,
    /// CPU RMSNorm gamma vectors (used for CPU-side RMSNorm fallback).
    pub norm_gammas: NormGammas,
    /// GPU RMSNorm gamma handles (uploaded once, used for GPU-side RMSNorm).
    pub gpu_norm_gammas: GpuNormGammaHandles,
    /// CPU KV cache — f32 variant (grows with each position).
    /// Used by CPU-hybrid `forward()` path. GPU-resident path uses `gpu_kv_cache`.
    pub kv_cache: CpuKVCache,
    /// GPU-resident KV cache — pre-allocated combined `[keys || values]` buffers.
    /// Used by `forward_gpu()` path. Eliminates the per-layer K/V sync.
    #[allow(dead_code)] // Used by forward_gpu path (Issue 429 clippy).
    pub gpu_kv_cache: Option<GpuKVCache>,
    /// CPU KV cache — Q8_0 quantized variant (3.5× memory reduction).
    /// When `Some`, attention dispatch uses inline Q8_0 dequantization
    /// via `AttentionQ8KVCubeCL` instead of f32 `AttentionCubeCL`.
    #[cfg(feature = "q8_kv_cache")]
    kv_cache_q8: Option<CpuKVCacheQ8>,
    /// CPU embedding weights (for embedding lookup).
    pub wte_cpu: Vec<f32>,
    /// Attention parameters (Gemma 2 2B constants).
    attn_params: AttentionParams,
    /// GEMV autotune cache — benchmarks plane vs tiled on first use per (m, n).
    /// Caches the fastest variant for each unique dimension pair.
    pub gemv_autotune: crate::gemv_autotune::GemvAutotune,
    /// Wall Attention state: running prefix sum buffer and config.
    /// When Some, Wall rescaling replaces RoPE in forward_layer_gpu.
    /// Feature-gated — zero cost when wall_attention is disabled.
    #[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
    wall_state: Option<WallGpuState>,
    /// Cached RoPE cos/sin table + GPU handle, keyed on `pos`.
    ///
    /// Within a single forward pass `pos` is constant across layers and Q/K,
    /// so the table is computed once and reused `n_layer * 2` times instead of
    /// recomputing `head_dim / 2` `powf` calls + a GPU buffer upload per call.
    /// Wrapped in `RefCell` because `dispatch_rope_gpu` takes `&self`.
    #[cfg(feature = "cubecl_runtime")]
    #[allow(dead_code)]
    // Cached by dispatch_rope_gpu; dead under cubecl_runtime-only (Issue 429 clippy).
    rope_cos_sin_cache: std::cell::RefCell<crate::rope_geglu_cubecl::RopeCosSinCache>,
    /// Delta routing state (Plan 097 / Plan 409 Phase 3 fix, 2026-07-09).
    ///
    /// The CPU forward (`forward_gemma2_layers`) applies delta routing at every
    /// 4th layer (block boundary). The GPU hybrid path was MISSING this step,
    /// causing the entire residual stream to diverge from layer 3 onward
    /// (Plan 409 root cause — NOT an f32 GEMV precision issue as previously
    /// hypothesized). This state replicates the CPU's `block_deltas` + scratch
    /// buffers so the GPU path can apply the same routing.
    #[cfg(feature = "delta_routing")]
    pub delta_routing_state: DeltaRoutingState,
    /// Per-layer LoRA adapters (Plan 410 Phase 3A.1). Empty = no LoRA active
    /// (zero overhead — the `forward_layer` insertion-point checks short-circuit
    /// when `lora_layers.is_empty()`). When non-empty, low-rank weight deltas are
    /// applied on CPU at each of the 4 GPU sync points, mirroring the CPU
    /// `forward_gemma2_with_lora` semantics.
    #[cfg(feature = "gemma_lora")]
    pub lora_layers: Vec<GemmaLayerLora>,
    /// Scratch buffer for LoRA `A @ input` intermediate, reused across all 7
    /// insertion points per layer. Grown to `max_rank` on first use, never
    /// re-allocated in the hot path.
    #[cfg(feature = "gemma_lora")]
    pub lora_scratch: Vec<f32>,
    /// Scratch buffer for the per-layer GeGLU output (`mlp_hidden`).
    /// Reused across all `n_layer` layers per forward pass — grown once to
    /// `config.mlp_hidden` and cleared (not re-allocated) per layer.
    pub mlp_hidden_scratch: std::cell::RefCell<Vec<f32>>,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuGemmaCubeCL {
    /// Initialize CubeCL-accelerated Gemma 2 forward pass.
    ///
    /// Uploads all GEMV weights to CubeCL's GPU buffer pool and clones
    /// norm gammas to CPU. The client should share the same wgpu Device/Queue
    /// as the GpuContext (via `init_device`).
    ///
    /// # Memory
    ///
    /// - GPU: ~2.7 GB for f32 GEMV weights (CubeCL pool, persists)
    /// - CPU: ~440 KB for norm gammas + KV cache grows with positions
    pub fn new(
        client: ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let cubecl_weights = CubeCLWeightBuffers::from_weights(&client, weights);
        let norm_gammas = NormGammas::from_weights(weights);
        let gpu_norm_gammas = GpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;
        let kv_cache = CpuKVCache::new(config.n_layer, kv_stride);
        let gpu_kv_cache = Some(GpuKVCache::new(
            &client,
            config.n_layer,
            kv_stride,
            config.block_size,
        ));

        let attn_params = AttentionParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0, // Set per attention dispatch.
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        };

        Self {
            client,
            config: config.clone(),
            weights: CubeCLWeightFormat::F32(cubecl_weights),
            norm_gammas,
            gpu_norm_gammas,
            kv_cache,
            gpu_kv_cache,
            #[cfg(feature = "q8_kv_cache")]
            kv_cache_q8: None,
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            #[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
            wall_state: None,
            #[cfg(feature = "cubecl_runtime")]
            rope_cos_sin_cache: std::cell::RefCell::new(
                crate::rope_geglu_cubecl::RopeCosSinCache::new(),
            ),
            #[cfg(feature = "delta_routing")]
            delta_routing_state: DeltaRoutingState::new(weights, config),
            #[cfg(feature = "gemma_lora")]
            lora_layers: Vec::new(),
            #[cfg(feature = "gemma_lora")]
            lora_scratch: Vec::new(),
            mlp_hidden_scratch: std::cell::RefCell::new(vec![0.0f32; config.mlp_hidden]),
        }
    }

    /// Initialize CubeCL-accelerated Gemma 2 forward pass with Q8_0 KV cache.
    ///
    /// Quantizes all GEMV weights to Q4_K format (4.5 bpw) on upload, reducing
    /// GPU memory by ~7× compared to f32. The CubeCL kernel dequantizes on-the-fly
    /// during the dot product — no intermediate f32 weight materialization.
    ///
    /// # Memory
    ///
    /// - GPU: ~0.4 GB for Q4_K weights (vs ~2.7 GB for f32)
    /// - CPU: ~10 MB for wte_cpu (f32 embedding lookup) + KV cache
    pub fn new_q4k(
        client: ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let q4k_weights = CubeCLQ4KWeightBuffers::from_weights(&client, weights, config);
        let norm_gammas = NormGammas::from_weights(weights);
        let gpu_norm_gammas = GpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;
        let kv_cache = CpuKVCache::new(config.n_layer, kv_stride);
        let gpu_kv_cache = Some(GpuKVCache::new(
            &client,
            config.n_layer,
            kv_stride,
            config.block_size,
        ));

        let attn_params = AttentionParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0,
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        };

        Self {
            client,
            config: config.clone(),
            weights: CubeCLWeightFormat::Q4K(q4k_weights),
            norm_gammas,
            gpu_norm_gammas,
            kv_cache,
            gpu_kv_cache,
            #[cfg(feature = "q8_kv_cache")]
            kv_cache_q8: None,
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            #[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
            wall_state: None,
            #[cfg(feature = "cubecl_runtime")]
            rope_cos_sin_cache: std::cell::RefCell::new(
                crate::rope_geglu_cubecl::RopeCosSinCache::new(),
            ),
            #[cfg(feature = "delta_routing")]
            delta_routing_state: DeltaRoutingState::new(weights, config),
            #[cfg(feature = "gemma_lora")]
            lora_layers: Vec::new(),
            #[cfg(feature = "gemma_lora")]
            lora_scratch: Vec::new(),
            mlp_hidden_scratch: std::cell::RefCell::new(vec![0.0f32; config.mlp_hidden]),
        }
    }

    /// Initialize CubeCL Gemma 2 forward pass with f16 weights (Plan 106 T2.10).
    ///
    /// Converts all GEMV weights from f32 to f16 on CPU, then uploads to GPU.
    /// The CubeCL kernel casts f16→f32 on-the-fly during the dot product,
    /// halving weight memory bandwidth while maintaining f32 accumulation precision.
    ///
    /// # Memory
    ///
    /// - GPU: ~1.35 GB for f16 weights (vs ~2.7 GB for f32)
    /// - CPU: ~10 MB for wte_cpu (f32 embedding lookup) + KV cache
    ///
    /// # Precision
    ///
    /// f16 has ~3 decimal digits of precision. Per-element quantization error
    /// is ~0.01–0.05 for typical LLM weight magnitudes (±10). Accumulated over
    /// a full row (N=2048 for Gemma 2 2B), total error stays within ~0.5–2.0,
    /// which is acceptable for inference and comparable to Q4_K quantization.
    pub fn new_f16(
        client: ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let f16_weights = CubeCLF16WeightBuffers::from_weights(&client, weights, config);
        let norm_gammas = NormGammas::from_weights(weights);
        let gpu_norm_gammas = GpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;
        let kv_cache = CpuKVCache::new(config.n_layer, kv_stride);
        let gpu_kv_cache = Some(GpuKVCache::new(
            &client,
            config.n_layer,
            kv_stride,
            config.block_size,
        ));

        let attn_params = AttentionParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0,
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        };

        Self {
            client,
            config: config.clone(),
            weights: CubeCLWeightFormat::F16(f16_weights),
            norm_gammas,
            gpu_norm_gammas,
            kv_cache,
            gpu_kv_cache,
            #[cfg(feature = "q8_kv_cache")]
            kv_cache_q8: None,
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            #[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
            wall_state: None,
            #[cfg(feature = "cubecl_runtime")]
            rope_cos_sin_cache: std::cell::RefCell::new(
                crate::rope_geglu_cubecl::RopeCosSinCache::new(),
            ),
            #[cfg(feature = "delta_routing")]
            delta_routing_state: DeltaRoutingState::new(weights, config),
            #[cfg(feature = "gemma_lora")]
            lora_layers: Vec::new(),
            #[cfg(feature = "gemma_lora")]
            lora_scratch: Vec::new(),
            mlp_hidden_scratch: std::cell::RefCell::new(vec![0.0f32; config.mlp_hidden]),
        }
    }

    /// Initialize CubeCL Gemma 2 forward pass with f32 weights + Q8_0 KV cache.
    ///
    /// Same as [`Self::new()`] but uses Q8_0 quantized KV cache for ~3.5× memory reduction.
    /// Attention dispatch uses inline Q8_0 dequantization via `AttentionQ8KVCubeCL`.
    ///
    /// # Memory
    ///
    /// - GPU: ~2.7 GB for f32 weights (same as `new`)
    /// - CPU KV: ~123 MB at pos=2048 (vs ~438 MB for f32 cache)
    #[cfg(feature = "q8_kv_cache")]
    pub fn new_with_q8kv(
        client: ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let cubecl_weights = CubeCLWeightBuffers::from_weights(&client, weights);
        let norm_gammas = NormGammas::from_weights(weights);
        let gpu_norm_gammas = GpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;
        let kv_cache = CpuKVCache::new(config.n_layer, kv_stride);
        let kv_cache_q8 = CpuKVCacheQ8::new(config.n_layer, config.n_kv_head, config.head_dim);
        let gpu_kv_cache = Some(GpuKVCache::new(
            &client,
            config.n_layer,
            kv_stride,
            config.block_size,
        ));

        let attn_params = AttentionParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0,
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        };

        Self {
            client,
            config: config.clone(),
            weights: CubeCLWeightFormat::F32(cubecl_weights),
            norm_gammas,
            gpu_norm_gammas,
            kv_cache,
            gpu_kv_cache,
            kv_cache_q8: Some(kv_cache_q8),
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            #[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
            wall_state: None,
            #[cfg(feature = "cubecl_runtime")]
            rope_cos_sin_cache: std::cell::RefCell::new(
                crate::rope_geglu_cubecl::RopeCosSinCache::new(),
            ),
            #[cfg(feature = "delta_routing")]
            delta_routing_state: DeltaRoutingState::new(weights, config),
            #[cfg(feature = "gemma_lora")]
            lora_layers: Vec::new(),
            #[cfg(feature = "gemma_lora")]
            lora_scratch: Vec::new(),
            mlp_hidden_scratch: std::cell::RefCell::new(vec![0.0f32; config.mlp_hidden]),
        }
    }

    /// Initialize CubeCL-accelerated Gemma 2 forward pass with Q4_K weights + Q8_0 KV cache.
    ///
    /// # Memory
    ///
    /// - GPU: ~0.4 GB for Q4_K weights
    /// - CPU KV: ~123 MB at pos=2048 (vs ~438 MB for f32 cache)
    #[cfg(feature = "q8_kv_cache")]
    pub fn new_q4k_with_q8kv(
        client: ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let q4k_weights = CubeCLQ4KWeightBuffers::from_weights(&client, weights, config);
        let norm_gammas = NormGammas::from_weights(weights);
        let gpu_norm_gammas = GpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;
        let kv_cache = CpuKVCache::new(config.n_layer, kv_stride);
        let kv_cache_q8 = CpuKVCacheQ8::new(config.n_layer, config.n_kv_head, config.head_dim);
        let gpu_kv_cache = Some(GpuKVCache::new(
            &client,
            config.n_layer,
            kv_stride,
            config.block_size,
        ));

        let attn_params = AttentionParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0,
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        };

        Self {
            client,
            config: config.clone(),
            weights: CubeCLWeightFormat::Q4K(q4k_weights),
            norm_gammas,
            gpu_norm_gammas,
            kv_cache,
            gpu_kv_cache,
            kv_cache_q8: Some(kv_cache_q8),
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: crate::gemv_autotune::GemvAutotune::new(),
            #[cfg(all(feature = "cubecl_runtime", feature = "wall_attention"))]
            wall_state: None,
            #[cfg(feature = "cubecl_runtime")]
            rope_cos_sin_cache: std::cell::RefCell::new(
                crate::rope_geglu_cubecl::RopeCosSinCache::new(),
            ),
            #[cfg(feature = "delta_routing")]
            delta_routing_state: DeltaRoutingState::new(weights, config),
            #[cfg(feature = "gemma_lora")]
            lora_layers: Vec::new(),
            #[cfg(feature = "gemma_lora")]
            lora_scratch: Vec::new(),
            mlp_hidden_scratch: std::cell::RefCell::new(vec![0.0f32; config.mlp_hidden]),
        }
    }

    // ── LoRA API (Plan 410 Phase 3A.1) ─────────────────────────────────

    /// Load per-layer LoRA adapters for the GPU forward path.
    ///
    /// When loaded, the hybrid forward (`forward`) applies low-rank weight
    /// deltas at the 7 matmul insertion points per layer (Q/K/V/O + gate/up/down),
    /// mirroring the CPU `forward_gemma2_with_lora` semantics. The LoRA deltas
    /// are applied on CPU at each GPU sync point — no new GPU kernels are
    /// introduced. The scratch buffer is pre-sized to `max_rank` across all
    /// adapters to avoid hot-path allocation.
    ///
    /// Pass an empty `Vec` (or call [`clear_lora`](Self::clear_lora)) to disable.
    #[cfg(feature = "gemma_lora")]
    pub fn set_lora(&mut self, layers: Vec<GemmaLayerLora>) {
        let max_rank = layers
            .iter()
            .flat_map(|l| {
                [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]
                    .into_iter()
                    .filter_map(|a| a.as_ref().map(|adp| adp.rank))
            })
            .max()
            .unwrap_or(0);
        self.lora_scratch = vec![0.0; max_rank];
        self.lora_layers = layers;
    }

    /// Unload all LoRA adapters. After this call, `forward` is bit-identical
    /// to a no-LoRA forward (the insertion-point checks short-circuit on
    /// `lora_layers.is_empty()`).
    #[cfg(feature = "gemma_lora")]
    pub fn clear_lora(&mut self) {
        self.lora_layers.clear();
    }

    /// Returns `true` if any LoRA adapter is loaded.
    #[cfg(feature = "gemma_lora")]
    #[inline]
    pub fn has_lora(&self) -> bool {
        !self.lora_layers.is_empty()
    }

    // ── Wall Attention API (Plan 193 T3) ───────────────────────────────

    /// Enable Wall Attention for this decode session.
    ///
    /// When enabled, Wall rescaling replaces RoPE in `forward_layer_gpu`.
    /// The Wall gate weights `w_g` must be provided (loaded from checkpoint
    /// or initialized with Xavier init).
    ///
    /// # Arguments
    ///
    /// * `config` — Wall configuration (gate_bias, gate_max, etc.)
    /// * `w_g` — Gate projection weights [gate_proj_dim, d_model]
    ///
    /// # Panics
    ///
    /// Panics if `w_g.len() != gate_proj_dim * n_embd`.
    #[cfg(feature = "wall_attention")]
    pub fn enable_wall(&mut self, config: WallConfig, w_g: Vec<f32>) {
        let gate_proj_dim = config.gate_proj_dim;
        assert_eq!(
            w_g.len(),
            gate_proj_dim * self.config.n_embd,
            "w_g must have gate_proj_dim * n_embd elements ({} * {} = {}, got {})",
            gate_proj_dim,
            self.config.n_embd,
            gate_proj_dim * self.config.n_embd,
            w_g.len()
        );
        let state =
            WallGpuState::new(config, self.config.n_embd, self.config.head_dim).with_weights(w_g);
        self.wall_state = Some(state);
    }

    /// Disable Wall Attention and revert to RoPE.
    #[cfg(feature = "wall_attention")]
    pub fn disable_wall(&mut self) {
        self.wall_state = None;
    }

    /// Reset Wall prefix state for a new sequence.
    ///
    /// Must be called when starting a new sequence to clear the running
    /// prefix sum. Otherwise, prefix accumulation from the previous
    /// sequence would carry over.
    #[cfg(feature = "wall_attention")]
    pub fn reset_wall(&mut self) {
        if let Some(ref mut wall) = self.wall_state {
            wall.reset();
        }
    }

    /// Advance the Wall Attention prefix one decode step if Wall is active.
    ///
    /// Ownership rule (Issue 957): the entry point that owns the embedding
    /// lookup steps the wall exactly once per token — `forward`,
    /// `forward_trace_layers`, `forward_layer0_only`, `forward_hidden`, and
    /// the fully-GPU arms of `forward_gpu` /
    /// `forward_gpu_logits_handle_max_layer`. Delegation paths (the
    /// `delta_routing` early-returns into `forward()`) skip their own step
    /// and let the callee do it — stepping twice would advance the prefix
    /// twice per token.
    #[cfg(feature = "wall_attention")]
    fn wall_step_if_active(&mut self, hidden: &[f32]) {
        if let Some(ref mut wall) = self.wall_state
            && wall.is_active()
        {
            wall.step_cpu(hidden);
        }
    }

    /// Forward pass: hybrid CPU/CubeCL decode for one token position.
    /// Run forward pass for one token at one position.
    ///
    /// Returns logits vector of length `vocab_size` with final logit
    /// softcapping applied (Gemma 2: cap=30.0).
    ///
    /// # Algorithm
    ///
    /// 1. CPU embedding lookup (single row, scaled by sqrt(n_embd))
    /// 2. Per-layer hybrid CPU/CubeCL dispatch (4 sync points each)
    /// 3. CPU final RMSNorm
    /// 4. CubeCL lm_head GEMV (tied embedding)
    /// 5. CPU final logit softcapping
    pub fn forward(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let vocab = self.config.vocab_size;
        let embed_scale = (n as f32).sqrt();

        // 1. Embedding lookup on CPU (single row from wte, scaled by sqrt(n_embd))
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * embed_scale;
        }

        // 1b. Wall Attention gate step (Plan 193 T3). `forward()` owns the
        // step for direct callers AND for the `forward_gpu()` delta_routing
        // delegation, which skips its own step (Issue 957 single-step rule).
        #[cfg(feature = "wall_attention")]
        self.wall_step_if_active(&hidden);

        // 2. Process all layers
        for layer_idx in 0..self.config.n_layer {
            hidden = self.forward_layer(hidden, layer_idx, pos);
            if layer_idx == 0 && pos <= 1 && std::env::var("RIIR_GPU_TRACE").as_deref() == Ok("1") {
                print_vec_stats(&format!("hidden after L0 pos={pos}"), &hidden, 4);
            }
        }

        // 3. Final RMSNorm
        let eps = self.config.rms_norm_eps as f32;
        rmsnorm_gamma(&mut hidden, &self.norm_gammas.final_norm, n, eps);

        // 4. LM head (tied wte): logits = wte @ hidden
        //
        // Issue 429 T4: when `lm_head_cpu` is enabled, run the lm_head GEMV on
        // CPU via simd_matmul_rows_parallel using wte_cpu (always f32). The
        // hidden state is already CPU-resident in this hybrid path, so zero
        // extra data movement. The 256000-row lm_head has worse worst-case
        // f32 precision on SPIR-V/Vulkan (max_abs_diff 0.28->26 compounding);
        // CPU gives bit-identical-to-reference logits.
        #[cfg(feature = "lm_head_cpu")]
        let logits = {
            let mut out = vec![0.0f32; vocab];
            katgpt_core::simd::simd_matmul_rows_parallel(
                &mut out,
                &self.wte_cpu,
                &hidden,
                vocab,
                n,
            );
            out
        };
        #[cfg(not(feature = "lm_head_cpu"))]
        let logits = match &self.weights {
            CubeCLWeightFormat::F32(w) => self.dispatch_gemv(&w.wte, &hidden, vocab, n),
            CubeCLWeightFormat::F16(w) => self.dispatch_gemv_f16(&w.wte, &hidden),
            CubeCLWeightFormat::Q4K(w) => self.dispatch_gemv_q4k(&w.wte, &hidden),
        };

        // 5. Final logit softcapping: logits = cap * tanh(logits / cap)
        let mut logits = logits;
        if self.config.final_logit_softcapping > 0.0 {
            softcap(&mut logits, self.config.final_logit_softcapping);
        }

        logits
    }

    /// Run a full CubeCL forward pass, capturing the hidden state after each
    /// layer (the production hybrid path: CPU norms/RoPE/GeGLU/residual + GPU
    /// GEMV + GPU attention).
    ///
    /// **Plan 409 Phase 3 (2026-07-09):** This exists because the prior
    /// `forward_layer_by_layer` diagnostic on `GpuGemmaForwardPass` uses the
    /// WGSL path (`dispatch_gemma2_layer`), NOT the CubeCL path that production
    /// `forward()` uses. The Plan 409 Phase 2 root-cause isolation was therefore
    /// pointed at the wrong implementation. This method lets the drift test
    /// compare CPU against the ACTUAL production CubeCL path.
    ///
    /// Returns `(after_embed, after_layer[], logits)`.
    pub fn forward_trace_layers(
        &mut self,
        token: usize,
        pos: usize,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
        let n = self.config.n_embd;
        let vocab = self.config.vocab_size;
        let embed_scale = (n as f32).sqrt();

        // 1. Embedding lookup on CPU
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * embed_scale;
        }
        let after_embed = hidden.clone();

        // Wall Attention gate step (Issue 957: this entry owns the embedding
        // lookup, so it owns the step — see wall_step_if_active).
        #[cfg(feature = "wall_attention")]
        self.wall_step_if_active(&hidden);

        // 2. Process all layers, capturing hidden after each
        let mut after_layer = Vec::with_capacity(self.config.n_layer);
        for layer_idx in 0..self.config.n_layer {
            hidden = self.forward_layer(hidden, layer_idx, pos);
            after_layer.push(hidden.clone());
        }

        // 3. Final RMSNorm
        let eps = self.config.rms_norm_eps as f32;
        rmsnorm_gamma(&mut hidden, &self.norm_gammas.final_norm, n, eps);

        // 4. LM head (tied wte): logits = wte @ hidden
        // Issue 429 T4: CPU offload variant (see forward() for rationale).
        #[cfg(feature = "lm_head_cpu")]
        let logits = {
            let mut out = vec![0.0f32; vocab];
            katgpt_core::simd::simd_matmul_rows_parallel(
                &mut out,
                &self.wte_cpu,
                &hidden,
                vocab,
                n,
            );
            out
        };
        #[cfg(not(feature = "lm_head_cpu"))]
        let logits = match &self.weights {
            CubeCLWeightFormat::F32(w) => self.dispatch_gemv(&w.wte, &hidden, vocab, n),
            CubeCLWeightFormat::F16(w) => self.dispatch_gemv_f16(&w.wte, &hidden),
            CubeCLWeightFormat::Q4K(w) => self.dispatch_gemv_q4k(&w.wte, &hidden),
        };

        // 5. Final logit softcapping
        let mut logits = logits;
        if self.config.final_logit_softcapping > 0.0 {
            softcap(&mut logits, self.config.final_logit_softcapping);
        }

        (after_embed, after_layer, logits)
    }

    /// **Plan 409 Phase 3 (2026-07-09):** Run one layer with full intermediate
    /// capture, for isolating which GEMV introduces drift.
    ///
    /// Takes the hidden state BEFORE this layer (residual stream input) and
    /// returns ALL intermediates: normed_input, Q, K, V, attn_out, wo_out,
    /// residual1 (after attn+norm+add), pre_mlp_normed, gate, up, geglu, down,
    /// and the final hidden output.
    ///
    /// This is a debug-only method — it clones every intermediate to CPU.
    #[allow(clippy::type_complexity)]
    pub fn forward_layer_debug(
        &mut self,
        hidden_in: Vec<f32>,
        layer_idx: usize,
        pos: usize,
    ) -> LayerDebugTrace {
        let n = self.config.n_embd;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kv_dim = self.config.n_kv_head * self.config.head_dim;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let norms = &self.norm_gammas.layers[layer_idx];

        let residual = hidden_in.clone();

        // RMSNorm
        let mut hidden = hidden_in;
        rmsnorm_gamma(&mut hidden, &norms.input_norm, n, eps);
        let normed_input = hidden.clone();

        // QKV GEMV
        let (mut q, mut k, v) = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_qkv(&w.layers[layer_idx], &hidden, q_dim, kv_dim, n)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_qkv_f16(&w.layers[layer_idx], &hidden, q_dim, kv_dim, n)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_qkv_q4k(&w.layers[layer_idx], &hidden, q_dim, kv_dim, n)
            }
        };

        // RoPE on Q and K (at pos=0, RoPE is identity, but we apply it for parity)
        apply_rope(
            &mut q,
            pos,
            self.config.head_dim,
            self.config.n_head,
            self.config.rope_theta,
        );
        apply_rope(
            &mut k,
            pos,
            self.config.head_dim,
            self.config.n_kv_head,
            self.config.rope_theta,
        );

        // Store K, V in KV cache (REQUIRED before attention reads from cache)
        self.kv_cache.store(layer_idx, pos, &k, &v);

        // For intermediate capture, also grab attn_out and wo_out separately.
        // We re-dispatch attention+wo in trace mode (with sync after attention).
        let wo_out = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_attention_wo_trace(&w.layers[layer_idx], &q, layer_idx, pos, n)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_attention_wo_f16(&w.layers[layer_idx], &q, layer_idx, pos, n)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_attention_wo_q4k(&w.layers[layer_idx], &q, layer_idx, pos, n)
            }
        };

        // Post-attn RMSNorm + residual add
        let mut hidden = wo_out.clone();
        rmsnorm_gamma(&mut hidden, &norms.post_attn_norm, n, eps);
        for (h, r) in hidden.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        let residual1 = hidden.clone();

        // Pre-MLP RMSNorm
        rmsnorm_gamma(&mut hidden, &norms.pre_mlp_norm, n, eps);
        let pre_mlp_normed = hidden.clone();

        // Gate + Up GEMV
        let (gate, up) = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_gate_up(&w.layers[layer_idx], &hidden, mlp, n)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gate_up_f16(&w.layers[layer_idx], &hidden, mlp, n)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_gate_up_q4k(&w.layers[layer_idx], &hidden, mlp, n)
            }
        };

        // GeGLU
        let mut geglu_out = vec![0.0f32; mlp];
        geglu(&gate, &up, &mut geglu_out);

        // Down GEMV
        let down_out = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_gemv(&w.layers[layer_idx].down_proj, &geglu_out, n, mlp)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gemv_f16(&w.layers[layer_idx].down_proj, &geglu_out)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_gemv_q4k(&w.layers[layer_idx].down_proj, &geglu_out)
            }
        };

        // Post-MLP RMSNorm + residual add
        let mut hidden = down_out.clone();
        rmsnorm_gamma(&mut hidden, &norms.post_mlp_norm, n, eps);
        for (h, r) in hidden.iter_mut().zip(residual1.iter()) {
            *h += r;
        }

        LayerDebugTrace {
            normed_input,
            q,
            k,
            v,
            wo_out,
            residual1,
            pre_mlp_normed,
            gate,
            up,
            geglu: geglu_out,
            down_out,
            hidden_out: hidden,
        }
    }

    /// Run embedding + Layer 0 only, returning hidden state after Layer 0.
    ///
    /// Used for isolation testing: compare CubeCL Layer 0 output against
    /// CPU reference (`forward_gemma2_trace` → `after_layer[0]`).
    pub fn forward_layer0_only(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let embed_scale = (n as f32).sqrt();

        // 1. Embedding lookup on CPU (same as forward())
        let mut hidden = vec![0.0f32; n];
        let tok_off = token * n;
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.wte_cpu[tok_off + i] * embed_scale;
        }

        // Wall Attention gate step (Issue 957: this entry owns the embedding
        // lookup, so it owns the step — see wall_step_if_active).
        #[cfg(feature = "wall_attention")]
        self.wall_step_if_active(&hidden);

        // 2. Layer 0 only
        self.forward_layer(hidden, 0, pos)
    }

    /// Single Gemma 2 transformer layer (hybrid CPU/CubeCL dispatch).
    ///
    /// # Compute Pass Structure (4 sync points)
    ///
    /// ```text
    /// [CPU] Save residual = hidden
    /// [CPU] RMSNorm(hidden, input_norm)
    /// [GPU] GEMV Q + K + V → q, k, v           ← Sync 1
    /// [CPU] RoPE(q)
    /// [CPU] Store k, v in CPU KV cache
    /// [GPU] Attention + Wo → wo_out              ← Sync 2 (batched, no intermediate sync)
    /// [CPU] RMSNorm(wo_out, post_attn_norm)
    /// [CPU] hidden = wo_out + residual
    /// [CPU] residual2 = hidden
    /// [CPU] RMSNorm(hidden, pre_mlp_norm)
    /// [GPU] GEMV gate + up → gate, up            ← Sync 3
    /// [CPU] GeGLU(gate, up) → mlp_hidden
    /// [GPU] GEMV down → down_out                 ← Sync 4
    /// [CPU] RMSNorm(down_out, post_mlp_norm)
    /// [CPU] hidden = down_out + residual2
    /// ```
    fn forward_layer(&mut self, mut hidden: Vec<f32>, layer_idx: usize, pos: usize) -> Vec<f32> {
        let n = self.config.n_embd;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kv_dim = self.config.n_kv_head * self.config.head_dim;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let norms = &self.norm_gammas.layers[layer_idx];

        // Diagnostic: trace intermediate values for layer 0, pos 0 and 1.
        // Gated behind RIIR_GPU_TRACE env var — the hardcoded `pos <= 1` trace
        // was spamming every new-sequence forward (riir-agents embedding
        // extraction, Plan 523). Production + benchmarks run with trace OFF.
        let trace =
            layer_idx == 0 && pos <= 1 && std::env::var("RIIR_GPU_TRACE").as_deref() == Ok("1");
        if trace {
            println!("\n── CubeCL forward_layer L0 P{pos} trace ──");
            print_vec_stats("input hidden", &hidden, 4);
        }

        // Save residual for post-norm add
        let residual = hidden.clone();

        // ── Pass A: QKV + RoPE ─────────────────────────────────────

        // CPU: RMSNorm
        rmsnorm_gamma(&mut hidden, &norms.input_norm, n, eps);
        if trace {
            print_vec_stats("after input_rmsnorm", &hidden, 4);
        }

        // GPU: QKV GEMVs (batched, 1 sync for all 3) → Sync 1
        // `mut` is required when `gemma_lora` is ON (LoRA deltas reassign q/k/v
        // at line ~2415). When `gemma_lora` is OFF, the reassignment is cfg'd out
        // and clippy flags `mut` as unused — suppress that here.
        #[cfg_attr(not(feature = "gemma_lora"), allow(unused_mut))]
        let (mut q, mut k, mut v) = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_qkv(&w.layers[layer_idx], &hidden, q_dim, kv_dim, n)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_qkv_f16(&w.layers[layer_idx], &hidden, q_dim, kv_dim, n)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_qkv_q4k(&w.layers[layer_idx], &hidden, q_dim, kv_dim, n)
            }
        };

        // Plan 410 Phase 3A.1: Apply Q/K/V LoRA deltas (CPU, at sync point 1).
        // The input is the RMSNormed hidden state (pre-QKV), which is still in `hidden`.
        // `v` is made mutable specifically for this — the base dispatch returns it
        // immutable because RoPE only touches q/k.
        #[cfg(feature = "gemma_lora")]
        if !self.lora_layers.is_empty() {
            let layer_lora = &self.lora_layers[layer_idx];
            let x_in = &hidden[..n];
            apply_lora_delta(&mut q, x_in, layer_lora.q.as_ref(), &mut self.lora_scratch);
            apply_lora_delta(&mut k, x_in, layer_lora.k.as_ref(), &mut self.lora_scratch);
            apply_lora_delta(&mut v, x_in, layer_lora.v.as_ref(), &mut self.lora_scratch);
        }

        if trace {
            print_vec_stats("Q after gemv", &q, 4);
            print_vec_stats("K after gemv", &k, 4);
            print_vec_stats("V after gemv", &v, 4);
        }

        // CPU: positional encoding — Wall rescale replaces RoPE when Wall is
        // active (mirrors forward_layer_gpu's Pass-A branch).
        //
        // Issue 957 fix: this hybrid path is where forward_gpu() lands whenever
        // delta_routing is ON (riir-gpu DEFAULT), so the previous unconditional
        // RoPE here made enable_wall() a silent no-op on the default feature
        // set — the prefix was stepped but never consumed, and logits came out
        // bit-identical to RoPE. The K prefix is baked into the KV cache at
        // store time below (same as the GPU path).
        #[cfg(feature = "wall_attention")]
        if let Some(wall) = self.wall_state.as_ref().filter(|s| s.is_active()) {
            use riir_infer_core::wall::wall_rescale_qk_inplace;
            // Single-token decode: prefix_q == prefix_k (WallGpuState).
            let prefix = &wall.prefix_cpu;
            wall_rescale_qk_inplace(
                &mut q,
                &mut k,
                prefix,
                prefix,
                self.config.head_dim,
            );
        } else {
            apply_rope(
                &mut q,
                pos,
                self.config.head_dim,
                self.config.n_head,
                self.config.rope_theta,
            );
            apply_rope(
                &mut k,
                pos,
                self.config.head_dim,
                self.config.n_kv_head,
                self.config.rope_theta,
            );
        }
        #[cfg(not(feature = "wall_attention"))]
        {
            apply_rope(
                &mut q,
                pos,
                self.config.head_dim,
                self.config.n_head,
                self.config.rope_theta,
            );
            apply_rope(
                &mut k,
                pos,
                self.config.head_dim,
                self.config.n_kv_head,
                self.config.rope_theta,
            );
        }
        if trace {
            print_vec_stats("Q after pos-enc (rope|wall)", &q, 4);
            print_vec_stats("K after pos-enc (rope|wall)", &k, 4);
        }

        // CPU: Store K, V in KV cache (Q8_0 if enabled, f32 otherwise)
        #[cfg(feature = "q8_kv_cache")]
        if let Some(ref mut q8_cache) = self.kv_cache_q8 {
            q8_cache.store(layer_idx, pos, &k, &v);
        } else {
            self.kv_cache.store(layer_idx, pos, &k, &v);
        }
        #[cfg(not(feature = "q8_kv_cache"))]
        self.kv_cache.store(layer_idx, pos, &k, &v);

        if trace {
            let n_pos = self.kv_cache.n_positions(layer_idx);
            let kv_len = (pos + 1) * self.kv_cache.kv_stride;
            let k_sum: f32 = self.kv_cache.keys[layer_idx]
                [..kv_len.min(self.kv_cache.keys[layer_idx].len())]
                .iter()
                .map(|x| x.abs())
                .sum();
            println!(
                "  kv_cache: n_pos={n_pos} pos={pos} kv_stride={} k_abs_sum={k_sum:.4}",
                self.kv_cache.kv_stride
            );
        }

        // ── Pass B: Attention + Wo + residual ──────────────────────

        // GPU: Attention + Wo → wo_out → Sync 2
        //
        // When LoRA is active (Plan 410 Phase 3A.1), use the split dispatch that
        // returns attn_out separately — the O-LoRA delta needs attn_out as input.
        // This adds one extra GPU→CPU sync (attn_out) but is only active when LoRA
        // is loaded. F32 only; F16/Q4K + LoRA is unsupported (training uses F32).
        #[cfg(feature = "gemma_lora")]
        let (attn_out_for_lora, mut wo_out) = if !self.lora_layers.is_empty() {
            match &self.weights {
                CubeCLWeightFormat::F32(w) => {
                    let (attn_out, wo_out) = self.dispatch_attention_wo_split(
                        &w.layers[layer_idx],
                        &q,
                        layer_idx,
                        pos,
                        n,
                    );
                    (Some(attn_out), wo_out)
                }
                _ => panic!("gemma_lora with non-F32 weights is unsupported (training uses F32)"),
            }
        } else {
            (
                None,
                self.dispatch_attention_wo_by_format(&q, layer_idx, pos, n),
            )
        };
        #[cfg(not(feature = "gemma_lora"))]
        let wo_out = self.dispatch_attention_wo_by_format(&q, layer_idx, pos, n);

        // Plan 410 Phase 3A.1: Apply O-LoRA delta (CPU, at sync point 2).
        // The input is attn_out (pre-Wo projection). Only active when LoRA is loaded.
        #[cfg(feature = "gemma_lora")]
        if let Some(ref attn_out) = attn_out_for_lora {
            let layer_lora = &self.lora_layers[layer_idx];
            apply_lora_delta(
                &mut wo_out,
                attn_out,
                layer_lora.o.as_ref(),
                &mut self.lora_scratch,
            );
        }
        if trace {
            print_vec_stats("wo_out", &wo_out, 4);
        }
        // CPU: RMSNorm + add residual (post-norm: norm then add)
        let mut hidden = wo_out;
        rmsnorm_gamma(&mut hidden, &norms.post_attn_norm, n, eps);
        if trace {
            print_vec_stats("after post_attn_rmsnorm", &hidden, 4);
        }
        for (h, r) in hidden.iter_mut().zip(residual.iter()) {
            *h += r;
        }
        if trace {
            print_vec_stats("after residual_add1", &hidden, 4);
        }

        // Save residual2
        let residual2 = hidden.clone();

        // ── Pass C: MLP ────────────────────────────────────────────

        // CPU: RMSNorm
        rmsnorm_gamma(&mut hidden, &norms.pre_mlp_norm, n, eps);
        if trace {
            print_vec_stats("after pre_mlp_rmsnorm", &hidden, 4);
        }

        // GPU: Gate + Up GEMVs (batched, 1 sync) → Sync 3
        // `mut` is required when `gemma_lora` is ON (LoRA deltas reassign gate/up
        // at line ~2555). When `gemma_lora` is OFF, the reassignment is cfg'd out
        // and clippy flags `mut` as unused — suppress that here.
        #[cfg_attr(not(feature = "gemma_lora"), allow(unused_mut))]
        let (mut gate, mut up) = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_gate_up(&w.layers[layer_idx], &hidden, mlp, n)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gate_up_f16(&w.layers[layer_idx], &hidden, mlp, n)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_gate_up_q4k(&w.layers[layer_idx], &hidden, mlp, n)
            }
        };

        // Plan 410 Phase 3A.1: Apply gate/up LoRA deltas (CPU, at sync point 3).
        // The input is the pre-MLP RMSNormed hidden state, still in `hidden`.
        #[cfg(feature = "gemma_lora")]
        if !self.lora_layers.is_empty() {
            let layer_lora = &self.lora_layers[layer_idx];
            let x_in = &hidden[..n];
            apply_lora_delta(
                &mut gate,
                x_in,
                layer_lora.gate.as_ref(),
                &mut self.lora_scratch,
            );
            apply_lora_delta(
                &mut up,
                x_in,
                layer_lora.up.as_ref(),
                &mut self.lora_scratch,
            );
        }

        if trace {
            print_vec_stats("gate after gemv", &gate, 4);
            print_vec_stats("up after gemv", &up, 4);
        }

        // CPU: GeGLU — reuse the persistent `mlp_hidden_scratch` instead of
        // allocating a fresh `vec![0.0f32; mlp]` per layer per forward call.
        // `geglu` fully overwrites the slice, so no fill/clear is needed.
        let mut mlp_hidden = self.mlp_hidden_scratch.borrow_mut();
        debug_assert_eq!(mlp_hidden.len(), mlp);
        geglu(&gate, &up, &mut mlp_hidden);
        if trace {
            print_vec_stats("mlp_hidden after geglu", &mlp_hidden, 4);
        }

        // GPU: Down GEMV (1 sync) → Sync 4
        // `mut` is required when `gemma_lora` is ON (LoRA delta reassigns down_out
        // at line ~2590). When `gemma_lora` is OFF, the reassignment is cfg'd out
        // and clippy flags `mut` as unused — suppress that here.
        #[cfg_attr(not(feature = "gemma_lora"), allow(unused_mut))]
        let mut down_out = match &self.weights {
            CubeCLWeightFormat::F32(w) => {
                self.dispatch_gemv(&w.layers[layer_idx].down_proj, &mlp_hidden, n, mlp)
            }
            CubeCLWeightFormat::F16(w) => {
                self.dispatch_gemv_f16(&w.layers[layer_idx].down_proj, &mlp_hidden)
            }
            CubeCLWeightFormat::Q4K(w) => {
                self.dispatch_gemv_q4k(&w.layers[layer_idx].down_proj, &mlp_hidden)
            }
        };

        // Plan 410 Phase 3A.1: Apply down LoRA delta (CPU, at sync point 4).
        // The input is the GeGLU hidden state (`mlp_hidden`).
        //
        // Keep `mlp_hidden` alive across this block (it's the LoRA input),
        // then release the borrow before returning.
        #[cfg(feature = "gemma_lora")]
        if !self.lora_layers.is_empty() {
            let layer_lora = &self.lora_layers[layer_idx];
            apply_lora_delta(
                &mut down_out,
                &mlp_hidden,
                layer_lora.down.as_ref(),
                &mut self.lora_scratch,
            );
        }
        drop(mlp_hidden);

        if trace {
            print_vec_stats("down_out after gemv", &down_out, 4);
        }
        // CPU: RMSNorm + add residual2
        let mut hidden = down_out;
        rmsnorm_gamma(&mut hidden, &norms.post_mlp_norm, n, eps);
        for (h, r) in hidden.iter_mut().zip(residual2.iter()) {
            *h += r;
        }
        if trace {
            print_vec_stats("hidden after layer0", &hidden, 4);
        }

        // Plan 409 Phase 3 fix (2026-07-09): Apply delta routing at block boundaries.
        // The CPU forward (`forward_gemma2_layers`) applies this after every 4th layer.
        // Without it, the GPU output diverges from CPU starting at layer 3 (BLOCK_SIZE-1).
        #[cfg(feature = "delta_routing")]
        self.delta_routing_state
            .apply_step(&mut hidden, &residual, layer_idx);

        hidden
    }
}

/// Print diagnostic stats for a vector: first N elements, min/max/mean/norm.
#[allow(dead_code)]
fn print_vec_stats(label: &str, v: &[f32], show_first: usize) {
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let first: Vec<f32> = v.iter().take(show_first).copied().collect();
    let n_nan = v.iter().filter(|x| x.is_nan()).count();
    let n_zero = v.iter().filter(|x| **x == 0.0).count();
    println!(
        "  {label:>30} len={:5} [{:+.4}, {:+.4}] mean={:+.4} norm={:.4} nan={n_nan} zero={n_zero} first={first:?}",
        v.len(),
        min,
        max,
        mean,
        norm,
    );
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests;
