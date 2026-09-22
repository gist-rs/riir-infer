//! LLaMA-family model weight structs for real model inference.
//!
//! Covers `LLaMA`, Mistral, `MiniCPM`, and other LLaMA-architecture variants.
//! Uses `SwiGLU` MLP (gate/up/down), `RMSNorm` (no offset), `RoPE`, separate `lm_head`.

/// Per-layer `LLaMA` transformer weights.
pub struct LlamaLayerWeights {
    // Attention projections
    pub attn_wq: Vec<f32>, // [n_head * head_dim, n_embd]
    pub attn_wk: Vec<f32>, // [n_kv_head * head_dim, n_embd]
    pub attn_wv: Vec<f32>, // [n_kv_head * head_dim, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_head * head_dim]
    // SwiGLU MLP (3 weights)
    pub gate_proj: Vec<f32>, // [mlp_hidden, n_embd]
    pub up_proj: Vec<f32>,   // [mlp_hidden, n_embd]
    pub down_proj: Vec<f32>, // [n_embd, mlp_hidden]
    // RMSNorm gammas (no offset, unlike Gemma 2)
    pub input_norm: Vec<f32>,     // [n_embd]
    pub post_attn_norm: Vec<f32>, // [n_embd] — alias: ffn_norm in LLaMA naming
}

/// All LLaMA-family transformer weights.
/// Separate `lm_head` (not tied to wte). No wpe (uses `RoPE`).
pub struct LlamaTransformerWeights {
    pub wte: Vec<f32>,                  // [vocab_size, n_embd]
    pub lm_head: Vec<f32>,              // [vocab_size, n_embd] (separate from wte)
    pub final_norm: Vec<f32>,           // [n_embd] (final RMSNorm)
    pub layers: Vec<LlamaLayerWeights>, // [n_layer]
}

impl LlamaTransformerWeights {
    /// Zero-copy view of the llama matrices through the generic weights
    /// surface (Issue 558 / riir-train Plan 410 Stage-0): gate→`mlp_w1`,
    /// down→`mlp_w2`. Consumes `self` (the matrices MOVE, never copy).
    ///
    /// The llama-only pieces — the per-layer norm gammas, the up projections,
    /// and the final norm — have no slot the generic plain forward would
    /// apply, so they are dropped here; the GPU lane attaches them separately
    /// (`riir-gpu` `GpuLlamaExtras`). `wpe` is a 1-element placeholder: llama
    /// has no learned-position table (`RoPE`), and the generic plain forwards
    /// (wpe-additive, identity-norm, dense-ReLU) are NOT valid on this
    /// surface — they assert against it (`assert_not_llama_lane`).
    ///
    /// The cfg arms reference THIS crate's features, which forward to the
    /// `katgpt-transformer` features that gate the `LayerWeights` fields —
    /// so each arm matches the struct layout by construction (a literal in a
    /// consumer crate cannot know that feature set).
    pub fn into_generic(self, config: &crate::types::Config) -> crate::transformer::TransformerWeights {
        // `from_parts` (katgpt-transformer) owns BOTH literals' optional-field
        // cfg arms — consumer-side literals cannot know the unified feature
        // set (the delta_routing miss at infer-core's default was the second
        // measured casualty after the LayerWeights one).
        crate::transformer::TransformerWeights::from_parts(
            self.wte,
            // Placeholder: llama has no learned-position table (RoPE) — the
            // generic plain forwards are not valid on this surface.
            vec![0.0; 1],
            self.lm_head,
            self.layers
                .into_iter()
                .map(|l| {
                    crate::transformer::LayerWeights::from_parts(
                        l.attn_wq,
                        l.attn_wk,
                        l.attn_wv,
                        l.attn_wo,
                        l.gate_proj,
                        l.down_proj,
                    )
                })
                .collect(),
            config.n_embd,
            config.n_layer,
        )
    }
}
