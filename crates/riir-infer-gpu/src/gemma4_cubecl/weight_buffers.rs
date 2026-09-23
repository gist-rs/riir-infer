//! CubeCL f32 weight buffers for Gemma-4 GEMV operations.
//!
//! Mirrors `gemma2_cubecl::weight_buffers` but with per-layer-type sizing:
//! Sliding layers use `head_dim`-based Q/K dims, Full layers use
//! `global_head_dim`-based Q/K dims. MLP (gate/up/down) sizes are uniform
//! across all layers.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

use riir_infer_core::transformer::gemma4::{
    Gemma4TransformerWeights, head_dim_for, kv_dim_for, q_dim_for,
};
use riir_infer_core::types::{Config, Gemma4LayerType};

/// Per-layer CubeCL f32 weight handles for Gemma-4 GEMV operations.
///
/// Each handle wraps a GPU buffer with f32 data. The Q/K/V/O projection sizes
/// vary per layer type (see `Gemma4LayerType`): sliding layers use
/// `head_dim`-based dims, full-attention layers use `global_head_dim`-based
/// dims. MLP (gate/up/down) sizes are uniform.
pub struct Gemma4CubeCLLayerWeights {
    pub attn_wq: Handle,
    pub attn_wk: Handle,
    pub attn_wv: Handle,
    pub attn_wo: Handle,
    pub gate_proj: Handle,
    pub up_proj: Handle,
    pub down_proj: Handle,
}

/// All CubeCL f32 weight handles for Gemma-4 inference.
///
/// Weights are uploaded once at construction and persist for the lifetime of
/// the [`GpuGemma4CubeCL`] instance. Each clone of a handle is a cheap
/// refcount increment; the underlying GPU buffer persists until all clones drop.
pub struct Gemma4CubeCLWeightBuffers {
    pub layers: Vec<Gemma4CubeCLLayerWeights>,
    /// Tied embedding weights (`[vocab_size, n_embd]`) for lm_head GEMV.
    pub wte: Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl Gemma4CubeCLWeightBuffers {
    /// Upload all GEMV weights to CubeCL GPU buffers (f32).
    ///
    /// Per-layer Q/K/V/O sizes are derived from the layer type stored in
    /// [`Gemma4LayerWeights::layer_type`].
    pub fn from_weights(
        client: &ComputeClient<ActiveRuntime>,
        weights: &Gemma4TransformerWeights,
        config: &Config,
    ) -> Self {
        let wte =
            client.create_from_slice(f32::as_bytes(&weights.wte));

        let layers = weights
            .layers
            .iter()
            .map(|l| {
                let layer_type = l.layer_type;
                let n = config.n_embd;
                let mlp = config.mlp_hidden;
                let q_dim = q_dim_for(config, layer_type);
                let kv_dim = kv_dim_for(config, layer_type);

                Gemma4CubeCLLayerWeights {
                    attn_wq: upload_mat(client, &l.attn_wq, q_dim * n),
                    attn_wk: upload_mat(client, &l.attn_wk, kv_dim * n),
                    attn_wv: upload_mat(client, &l.attn_wv, kv_dim * n),
                    attn_wo: upload_mat(client, &l.attn_wo, n * q_dim),
                    gate_proj: upload_mat(client, &l.gate_proj, mlp * n),
                    up_proj: upload_mat(client, &l.up_proj, mlp * n),
                    down_proj: upload_mat(client, &l.down_proj, n * mlp),
                }
            })
            .collect();

        Self { layers, wte }
    }
}

#[cfg(feature = "cubecl_runtime")]
fn upload_mat(client: &ComputeClient<ActiveRuntime>, data: &[f32], expected_len: usize) -> Handle {
    debug_assert_eq!(
        data.len(),
        expected_len,
        "weight length mismatch (expected {expected_len}, got {})",
        data.len()
    );
    client.create_from_slice(f32::as_bytes(data))
}

// ── CPU-side norm gamma vectors ────────────────────────────────────

/// Per-layer RMSNorm gamma vectors (CPU copies, used for CPU-side RMSNorm).
///
/// Gemma-4 has 4 RMSNorms per layer (input/post_attn/pre_mlp/post_mlp) +
/// Q/K-norm gammas (NEW vs Gemma-2). The +1 offset is pre-applied during
/// GGUF weight loading (the gamma vectors stored here are the raw loaded
/// values, which already include the offset).
pub struct Gemma4LayerNormGammas {
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub pre_mlp_norm: Vec<f32>,
    pub post_mlp_norm: Vec<f32>,
    /// Q-norm gamma (length = head_dim for this layer's type).
    pub attn_q_norm: Vec<f32>,
    /// K-norm gamma (length = head_dim for this layer's type).
    pub attn_k_norm: Vec<f32>,
    /// Per-layer output scale (Issue 397). Default 1.0 when absent.
    pub layer_output_scale: f32,
}

pub struct Gemma4NormGammas {
    pub layers: Vec<Gemma4LayerNormGammas>,
    pub final_norm: Vec<f32>,
}

impl Gemma4NormGammas {
    pub fn from_weights(weights: &Gemma4TransformerWeights) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|l| Gemma4LayerNormGammas {
                input_norm: l.input_norm.clone(),
                post_attn_norm: l.post_attn_norm.clone(),
                pre_mlp_norm: l.pre_mlp_norm.clone(),
                post_mlp_norm: l.post_mlp_norm.clone(),
                attn_q_norm: l.attn_q_norm.clone(),
                attn_k_norm: l.attn_k_norm.clone(),
                layer_output_scale: l.layer_output_scale,
            })
            .collect();
        Self {
            layers,
            final_norm: weights.final_norm.clone(),
        }
    }
}

// ── Per-layer shape derivation (re-exported helpers) ───────────────

/// Derive the per-layer KV stride, sliding-window capacity, and head_dim
/// for the given config. Used to size the [`Gemma4CpuKVCache`] at construction.
pub fn per_layer_cache_dims(
    config: &Config,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut kv_stride = Vec::with_capacity(config.n_layer);
    let mut sliding_capacity = Vec::with_capacity(config.n_layer);
    let mut head_dims = Vec::with_capacity(config.n_layer);
    for &layer_type in &config.gemma4_layer_types {
        let kvd = kv_dim_for(config, layer_type);
        let hd = head_dim_for(config, layer_type);
        kv_stride.push(kvd);
        head_dims.push(hd);
        // Sliding layers use a ring buffer of size `sliding_window`.
        // Full-attention layers are unbounded (capacity 0 sentinel).
        match layer_type {
            Gemma4LayerType::Sliding => sliding_capacity.push(config.sliding_window),
            Gemma4LayerType::Full => sliding_capacity.push(0),
        }
    }
    (kv_stride, sliding_capacity, head_dims)
}

/// Per-layer derived attention parameters (computed once at construction).
pub struct Gemma4LayerAttnParams {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub n_kv_head: usize,
    pub n_head: usize,
    /// RoPE rotation dimension (partial for Full layers).
    pub rope_rot_dim: usize,
    /// RoPE frequency table index (0 = sliding, 1 = full).
    pub rope_table_idx: usize,
}

/// Pre-compute per-layer attention parameters from the config.
pub fn per_layer_attn_params(config: &Config) -> Vec<Gemma4LayerAttnParams> {
    use riir_infer_core::transformer::gemma4::{n_kv_head_for, rope_rot_dim_for};
    config
        .gemma4_layer_types
        .iter()
        .map(|&layer_type| Gemma4LayerAttnParams {
            q_dim: q_dim_for(config, layer_type),
            kv_dim: kv_dim_for(config, layer_type),
            head_dim: head_dim_for(config, layer_type),
            n_kv_head: n_kv_head_for(config, layer_type),
            n_head: config.n_head,
            rope_rot_dim: rope_rot_dim_for(config, layer_type),
            rope_table_idx: match layer_type {
                Gemma4LayerType::Sliding => 0,
                Gemma4LayerType::Full => 1,
            },
        })
        .collect()
}
