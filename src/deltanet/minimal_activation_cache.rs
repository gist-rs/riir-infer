//! Minimal activation cache for the 5-field recomputation backward
//! (Issue 641 T3 — moved from `riir-gpu` to break the dependency cycle;
//! `riir-gpu` re-exports these types for back-compat).
//!
//! # Why this lives in `riir-engine`
//!
//! The GPU forward (`riir-gpu`) populates this cache, and the CPU recomputation
//! backward (`riir-train-engine::deltanet::model_backward_recompute`) consumes it.
//! Because `riir-gpu` depends on `riir-engine` (not the reverse), the shared
//! activation contract MUST live in the engine. `riir-gpu` provides
//! `pub use riir_engine::deltanet::minimal_activation_cache::{...}` for
//! back-compat with its existing forward methods.
//!
//! # Memory budget (Bonsai-27B: `n_embd=5120`, `n_v_heads=64`, `head_dim=128`)
//!
//! Per token per `DeltaNet` layer:
//!   `x_in`:         5120 × 4 B = 20 KiB
//!   `norm_x`:       5120 × 4 B = 20 KiB
//!   `qkv_expanded`: 3 × 64 × 128 × 4 B = 96 KiB
//!   beta:         64 × 4 B = 256 B
//!   decay:        64 × 4 B = 256 B
//!   Total: ~136 KiB
//!
//! Per token per attention layer:
//!   `x_in` + `norm_x` = 40 KiB
//!
//! For 64 tokens × 64 layers (48 `DeltaNet` + 16 attention): ~560 MiB.
//! vs the 13-field CPU cache: ~870 MiB. Saves ~310 MiB.

/// DeltaNet-specific minimal activations. Stored inside [`MinimalLayerActivations`]
/// for `DeltaNet` layers; `None` for attention layers.
#[derive(Clone, Debug)]
pub struct DeltanetMinimalActs {
    /// Expanded + L2-normalized Q/K/V `[3 * n_v_heads * head_dim]`.
    /// Layout: `[Q(n_v × hd) | K(n_v × hd) | V(n_v × hd)]`.
    pub qkv_expanded: Vec<f32>,

    /// Update rate β = sigmoid(b) `[n_v_heads]`.
    pub beta: Vec<f32>,

    /// Decay gate g = `exp(−a_log` · `softplus(a_raw` + `dt_bias`)) `[n_v_heads]`.
    pub decay: Vec<f32>,
}

impl DeltanetMinimalActs {
    /// Split `qkv_expanded` into per-head q, k, v slices.
    ///
    /// Returns `(q_head, k_head, v_head)` each of length `head_dim`, for the
    /// given head index `h` in `0..n_v_heads`.
    pub fn head_qkv(&self, h: usize, n_v_heads: usize, head_dim: usize) -> (&[f32], &[f32], &[f32]) {
        let hd = head_dim;
        let per_head = n_v_heads * hd;
        let q_start = h * hd;
        let k_start = per_head + h * hd;
        let v_start = 2 * per_head + h * hd;
        (
            &self.qkv_expanded[q_start..q_start + hd],
            &self.qkv_expanded[k_start..k_start + hd],
            &self.qkv_expanded[v_start..v_start + hd],
        )
    }
}

/// Minimal per-layer per-token activations saved by the GPU forward.
///
/// 5 fields for `DeltaNet` layers (`x_in`, `norm_x`, `qkv_expanded`, `beta`,
/// `decay`); 2 fields (`x_in`, `norm_x`) for attention layers.
///
/// The missing fields needed by the full backward are recomputed from these
/// minimal inputs + frozen weights during the backward pass (Issue 641 T3).
#[derive(Clone, Debug)]
pub struct MinimalLayerActivations {
    /// Pre-input-RMSNorm hidden state `[n_embd]`. Saved for ALL layers.
    /// Needed by the `RMSNorm` backward: `grad_x_in = rmsnorm_backward(grad_norm_x, x_in, ...)`.
    pub x_in: Vec<f32>,

    /// Post-input-RMSNorm `[n_embd]`. Saved for ALL layers.
    /// The input to the QKV/z/a/b projections (`DeltaNet`) or Q/K/V projections (attention).
    pub norm_x: Vec<f32>,

    /// DeltaNet-only fields. `None` for attention layers.
    pub deltanet: Option<DeltanetMinimalActs>,
}

impl MinimalLayerActivations {
    /// Construct an attention-layer entry (`x_in` + `norm_x` only).
    pub fn attention(x_in: Vec<f32>, norm_x: Vec<f32>) -> Self {
        Self {
            x_in,
            norm_x,
            deltanet: None,
        }
    }

    /// Construct a DeltaNet-layer entry (all 5 fields).
    pub fn deltanet(
        x_in: Vec<f32>,
        norm_x: Vec<f32>,
        qkv_expanded: Vec<f32>,
        beta: Vec<f32>,
        decay: Vec<f32>,
    ) -> Self {
        Self {
            x_in,
            norm_x,
            deltanet: Some(DeltanetMinimalActs {
                qkv_expanded,
                beta,
                decay,
            }),
        }
    }

    /// Returns `true` if this is a `DeltaNet` layer (has deltanet acts).
    pub fn is_deltanet(&self) -> bool {
        self.deltanet.is_some()
    }
}

/// Collector for a full training sequence's minimal activations.
///
/// Indexed by `[token][raw_layer_index]` — ALL layers (`DeltaNet` + Attention),
/// unlike the legacy `TrainingActivationCollector` which only stores `DeltaNet`
/// layers.
///
/// Populated incrementally by the GPU forward's training-mode method
/// (`forward_token_training` on cudarc, `forward_token_training_minimal` on
/// `CubeCL`) — one `Vec<MinimalLayerActivations>` entry per target token.
///
/// # Dimensions (stored for the backward pass)
///
/// - `n_deltanet_layers`: number of `DeltaNet` layers (for memory estimation)
/// - `n_v_heads`: value heads per `DeltaNet` layer
/// - `head_dim`: dimension per head
/// - `n_embd`: model embedding dimension
#[derive(Clone, Debug, Default)]
pub struct MinimalActivationCache {
    /// `[t][raw_layer_idx]` activations. ALL layers, not just `DeltaNet`.
    pub tokens: Vec<Vec<MinimalLayerActivations>>,

    /// Pre-final-RMSNorm hidden state `[n_embd]` — the input to the final
    /// `RMSNorm` before `lm_head`. Saved once per forward (not per token).
    pub x_pre_finalnorm: Vec<f32>,

    /// Number of `DeltaNet` layers (for memory estimation + backward routing).
    pub n_deltanet_layers: usize,
    pub n_v_heads: usize,
    pub head_dim: usize,
    pub n_embd: usize,
}

impl MinimalActivationCache {
    /// Create a new minimal activation cache with known dimensions.
    ///
    /// `n_deltanet_layers` is the count of `DeltaNet` layers (used for memory
    /// estimation). Per-token layer arrays are pre-allocated via `begin_token`.
    pub fn new(n_deltanet_layers: usize, n_v_heads: usize, head_dim: usize, n_embd: usize) -> Self {
        Self {
            tokens: Vec::new(),
            x_pre_finalnorm: Vec::new(),
            n_deltanet_layers,
            n_v_heads,
            head_dim,
            n_embd,
        }
    }

    /// Prepare the cache for a new token position. `n_layer` is the TOTAL
    /// number of layers (`DeltaNet` + Attention) — pre-allocates the per-token
    /// layer array.
    pub fn begin_token(&mut self, n_layer: usize) {
        self.tokens.push(Vec::with_capacity(n_layer));
    }

    /// Push one layer's activations for the current (last) token.
    pub fn push_layer_activation(&mut self, acts: MinimalLayerActivations) {
        self.tokens
            .last_mut()
            .expect("begin_token not called")
            .push(acts);
    }

    /// Get the activations for token `t`, raw layer index `layer`.
    pub fn get(&self, t: usize, layer: usize) -> &MinimalLayerActivations {
        &self.tokens[t][layer]
    }

    /// Number of token positions collected.
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Total memory used by the collected activations, in bytes.
    ///
    /// Includes: per-layer `x_in` + `norm_x` (ALL layers), `DeltaNet` extras
    /// (`qkv_expanded` + beta + decay), and the final `x_pre_finalnorm`.
    pub fn mem_usage_bytes(&self) -> usize {
        let per_float = std::mem::size_of::<f32>();
        let n_tokens = self.tokens.len();
        if n_tokens == 0 {
            return self.x_pre_finalnorm.len() * per_float;
        }
        let n_total_layers = self.tokens[0].len();
        let n_attn = n_total_layers.saturating_sub(self.n_deltanet_layers);
        // ALL layers: x_in + norm_x = 2 * n_embd floats
        let base_floats = 2 * self.n_embd;
        // DeltaNet extra: qkv_expanded + beta + decay
        let deltanet_extra = 3 * self.n_v_heads * self.head_dim + 2 * self.n_v_heads;
        n_tokens
            * (n_attn * base_floats + self.n_deltanet_layers * (base_floats + deltanet_extra))
            * per_float
            + self.x_pre_finalnorm.len() * per_float
    }

    /// Clear all collected activations (keep dimensions).
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.x_pre_finalnorm.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_cache_push_and_get_deltanet() {
        let mut cache = MinimalActivationCache::new(1, 2, 16, 64);
        cache.begin_token(1);
        cache.push_layer_activation(MinimalLayerActivations::deltanet(
            vec![0.1; 64],
            vec![0.2; 64],
            vec![0.3; 3 * 2 * 16],
            vec![0.4; 2],
            vec![0.5; 2],
        ));
        assert_eq!(cache.n_tokens(), 1);
        let acts = cache.get(0, 0);
        assert!(acts.is_deltanet());
        assert_eq!(acts.x_in.len(), 64);
        assert_eq!(acts.norm_x.len(), 64);
        let dn = acts.deltanet.as_ref().unwrap();
        assert_eq!(dn.qkv_expanded.len(), 3 * 2 * 16);
        assert_eq!(dn.beta.len(), 2);
        assert_eq!(dn.decay.len(), 2);
    }

    #[test]
    fn test_minimal_cache_mixed_layers() {
        let mut cache = MinimalActivationCache::new(1, 2, 16, 64);
        cache.begin_token(2);
        cache.push_layer_activation(MinimalLayerActivations::deltanet(
            vec![0.1; 64],
            vec![0.2; 64],
            vec![0.3; 96],
            vec![0.4; 2],
            vec![0.5; 2],
        ));
        cache.push_layer_activation(MinimalLayerActivations::attention(
            vec![0.6; 64],
            vec![0.7; 64],
        ));
        assert_eq!(cache.n_tokens(), 1);
        assert!(cache.get(0, 0).is_deltanet());
        assert!(!cache.get(0, 1).is_deltanet());
    }

    #[test]
    fn test_minimal_cache_head_qkv_split() {
        let dn = DeltanetMinimalActs {
            qkv_expanded: (0..3 * 2 * 16).map(|i| i as f32).collect(),
            beta: vec![0.0; 2],
            decay: vec![0.0; 2],
        };
        let (q, k, v) = dn.head_qkv(1, 2, 16);
        assert_eq!(q.len(), 16);
        assert_eq!(k.len(), 16);
        assert_eq!(v.len(), 16);
        // head 1: q starts at 1*16=16, k at 2*16+1*16=48, v at 4*16+1*16=80
        assert_eq!(q[0], 16.0);
        assert_eq!(k[0], 48.0);
        assert_eq!(v[0], 80.0);
    }

    #[test]
    fn test_minimal_cache_mem_usage() {
        let mut cache = MinimalActivationCache::new(1, 2, 16, 64);
        cache.begin_token(1);
        cache.push_layer_activation(MinimalLayerActivations::deltanet(
            vec![0.0; 64],
            vec![0.0; 64],
            vec![0.0; 96],
            vec![0.0; 2],
            vec![0.0; 2],
        ));
        cache.x_pre_finalnorm = vec![0.0; 64];
        // Per token per DeltaNet layer: (64+64) + (96+2+2) = 228 floats
        // + x_pre_finalnorm: 64 floats
        // Total: 292 floats × 4 bytes = 1168 bytes
        assert_eq!(cache.mem_usage_bytes(), 292 * 4);
    }

    #[test]
    fn test_minimal_cache_clear() {
        let mut cache = MinimalActivationCache::new(1, 2, 16, 64);
        cache.begin_token(1);
        cache.push_layer_activation(MinimalLayerActivations::deltanet(
            vec![0.1; 64],
            vec![0.2; 64],
            vec![0.3; 96],
            vec![0.4; 2],
            vec![0.5; 2],
        ));
        cache.x_pre_finalnorm = vec![0.0; 64];
        assert_eq!(cache.n_tokens(), 1);
        assert!(!cache.x_pre_finalnorm.is_empty());
        cache.clear();
        assert_eq!(cache.n_tokens(), 0);
        assert!(cache.x_pre_finalnorm.is_empty());
        // Dimensions preserved
        assert_eq!(cache.n_deltanet_layers, 1);
    }
}
