//! CubeCL f16 weight buffer types for Gemma 2 GEMV operations.
//!
//! Extracted from `gemma2_cubecl/mod.rs` to keep the main file under the
//! 2048-line guideline. Contains:
//! - `CubeCLF16LayerWeights` — per-layer f16 weight handles
//! - `CubeCLF16WeightBuffers` — all-layer f16 weight buffer collection

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_f16_cubecl::F16Handle;
use riir_infer_core::gemma_layer::GemmaTransformerWeights;
use riir_infer_core::types::Config;

// ── CubeCL weight handles ──────────────────────────────────────────

/// Per-layer CubeCL f16 weight handles for GEMV operations (Plan 106 T2.10).
///
/// Each handle wraps a GPU buffer with f16 data (2 bytes per element).
/// The CubeCL kernel casts f16→f32 on-the-fly during the dot product,
/// halving weight memory bandwidth vs f32.
#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // qkv_combined is fused QKV GEMV scaffolding (Issue 429 clippy).
pub struct CubeCLF16LayerWeights {
    pub attn_wq: F16Handle,
    pub attn_wk: F16Handle,
    pub attn_wv: F16Handle,
    /// Combined [Wq|Wk|Wv] f16 weights for fused triple QKV GEMV.
    /// Layout: [Wq(q_dim × n) | Wk(kv_dim × n) | Wv(kv_dim × n)].
    /// `m` = q_dim + 2 * kv_dim, `n` = n_embd.
    pub qkv_combined: F16Handle,
    pub attn_wo: F16Handle,
    pub gate_proj: F16Handle,
    pub up_proj: F16Handle,
    pub down_proj: F16Handle,
}

/// All CubeCL f16 weight handles for Gemma 2 inference (Plan 106 T2.10).
///
/// Stores weights as `half::f16` (2 bytes per element) for ~2× bandwidth
/// reduction during GEMV weight reads. Accumulation remains f32 precision.
///
/// # Memory
///
/// - GPU: ~1.35 GB for f16 weights (vs ~2.7 GB for f32)
/// - CPU: ~10 MB for wte_cpu (f32 embedding, for lookup only)
#[cfg(feature = "cubecl_runtime")]
pub struct CubeCLF16WeightBuffers {
    pub layers: Vec<CubeCLF16LayerWeights>,
    /// Tied embedding weights as f16 for lm_head GEMV.
    pub wte: F16Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl CubeCLF16WeightBuffers {
    /// Convert and upload all GEMV weights to CubeCL GPU buffers as f16.
    ///
    /// Each f32 projection is converted to f16 on CPU, then uploaded via
    /// [`F16Handle::from_f32`]. This reduces GPU memory by ~2× compared
    /// to f32 weights while maintaining f32 accumulation precision.
    pub fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        let wte = F16Handle::from_f32(client, &weights.wte, config.vocab_size, config.n_embd);

        let layers = weights
            .layers
            .iter()
            .map(|l| {
                let q_dim = config.n_head * config.head_dim;
                let kv_dim = config.n_kv_head * config.head_dim;
                let n = config.n_embd;
                let mlp = config.mlp_hidden;

                CubeCLF16LayerWeights {
                    attn_wq: F16Handle::from_f32(client, &l.attn_wq, q_dim, n),
                    attn_wk: F16Handle::from_f32(client, &l.attn_wk, kv_dim, n),
                    attn_wv: F16Handle::from_f32(client, &l.attn_wv, kv_dim, n),
                    // Combined QKV: [Wq | Wk | Wv] for fused triple GEMV
                    qkv_combined: {
                        let mut combined = Vec::with_capacity((q_dim + 2 * kv_dim) * n);
                        combined.extend_from_slice(&l.attn_wq);
                        combined.extend_from_slice(&l.attn_wk);
                        combined.extend_from_slice(&l.attn_wv);
                        F16Handle::from_f32(client, &combined, q_dim + 2 * kv_dim, n)
                    },
                    attn_wo: F16Handle::from_f32(client, &l.attn_wo, n, q_dim),
                    gate_proj: F16Handle::from_f32(client, &l.gate_proj, mlp, n),
                    up_proj: F16Handle::from_f32(client, &l.up_proj, mlp, n),
                    down_proj: F16Handle::from_f32(client, &l.down_proj, n, mlp),
                }
            })
            .collect();

        Self { layers, wte }
    }
}
