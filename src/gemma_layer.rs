//! Gemma 2 model weight structs for real model inference (Plan 087).
//!
//! Gemma 2 uses `GeGLU` MLP (3 weights: gate/up/down vs current 2),
//! `RMSNorm` with offset (gamma stored as gamma-1), `RoPE` (no wpe),
//! tied embeddings (`lm_head` = wte.T), and post-norm.

/// Per-layer Gemma 2 transformer weights.
pub struct GemmaLayerWeights {
    // Attention projections
    pub attn_wq: Vec<f32>, // [n_embd, n_embd]
    pub attn_wk: Vec<f32>, // [kv_dim, n_embd]
    pub attn_wv: Vec<f32>, // [kv_dim, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_embd]
    // GeGLU MLP (3 weights instead of 2)
    pub gate_proj: Vec<f32>, // [mlp_hidden, n_embd]
    pub up_proj: Vec<f32>,   // [mlp_hidden, n_embd]
    pub down_proj: Vec<f32>, // [n_embd, mlp_hidden]
    // RMSNorm gammas (stored as gamma-1, add +1 during load)
    pub input_norm: Vec<f32>,     // [n_embd]
    pub post_attn_norm: Vec<f32>, // [n_embd] (post-norm, Gemma 2 specific)
    pub pre_mlp_norm: Vec<f32>,   // [n_embd]
    pub post_mlp_norm: Vec<f32>,  // [n_embd] (post-norm, Gemma 2 specific)
}

/// All Gemma 2 transformer weights.
/// No wpe (uses `RoPE`), no separate `lm_head` (tied to wte).
pub struct GemmaTransformerWeights {
    pub wte: Vec<f32>,                  // [vocab_size, n_embd]
    pub final_norm: Vec<f32>,           // [n_embd] (final RMSNorm)
    pub layers: Vec<GemmaLayerWeights>, // [n_layer]
    // Delta routing weights (Plan 097: Delta Block cross-layer routing)
    #[cfg(feature = "delta_routing")]
    pub delta_routing_query: Vec<Vec<f32>>, // [n_layer][n_embd] zero-init (trained during fine-tuning)
    #[cfg(feature = "delta_routing")]
    pub delta_routing_norm: Vec<Vec<f32>>, // [n_layer][n_embd] one-init (identity RMSNorm)
}

// ── f16 Weight Variants (Plan 095) ───────────────────────────

/// Per-layer Gemma 2 transformer weights stored as f16.
///
/// Halves memory bandwidth for weight reads during inference.
/// `RMSNorm` gammas remain f32 (tiny: 4 × 2304 = 36 KB per layer).
/// Input/output activations are always f32 — only weights are f16.
pub struct GemmaLayerWeightsF16 {
    // Attention projections (f16)
    pub attn_wq: Vec<half::f16>, // [n_embd, n_embd]
    pub attn_wk: Vec<half::f16>, // [kv_dim, n_embd]
    pub attn_wv: Vec<half::f16>, // [kv_dim, n_embd]
    pub attn_wo: Vec<half::f16>, // [n_embd, n_embd]
    // GeGLU MLP (f16)
    pub gate_proj: Vec<half::f16>, // [mlp_hidden, n_embd]
    pub up_proj: Vec<half::f16>,   // [mlp_hidden, n_embd]
    pub down_proj: Vec<half::f16>, // [n_embd, mlp_hidden]
    // RMSNorm gammas (kept as f32 — negligible size, avoids conversion overhead)
    pub input_norm: Vec<f32>,     // [n_embd]
    pub post_attn_norm: Vec<f32>, // [n_embd]
    pub pre_mlp_norm: Vec<f32>,   // [n_embd]
    pub post_mlp_norm: Vec<f32>,  // [n_embd]
}

/// All Gemma 2 transformer weights with f16 storage.
///
/// Tied `lm_head` (wte) stored as f16 — halves the 2.36 GB vocab projection
/// to 1.18 GB, which alone saves ~7 ms/token at 17.6 GB/s effective bandwidth.
pub struct GemmaTransformerWeightsF16 {
    pub wte: Vec<half::f16>,               // [vocab_size, n_embd]
    pub final_norm: Vec<f32>,              // [n_embd] (final RMSNorm, kept f32)
    pub layers: Vec<GemmaLayerWeightsF16>, // [n_layer]
    // Delta routing weights (Plan 097: Delta Block cross-layer routing)
    // Kept as f32 since they're trained separately — negligible size vs projections
    #[cfg(feature = "delta_routing")]
    pub delta_routing_query: Vec<Vec<f32>>, // [n_layer][n_embd] zero-init
    #[cfg(feature = "delta_routing")]
    pub delta_routing_norm: Vec<Vec<f32>>, // [n_layer][n_embd] one-init
}
