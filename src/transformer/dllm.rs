//! Discrete Diffusion Language Model (dLLM) forward variants.
//!
//! Two D2F (Discrete Diffusion Forcing) forward passes for training and
//! inference under the `dllm` feature flag:
//!
//! - `forward_bidirectional` — teacher mode: all positions attend to all
//!   positions (full bidirectional). Returns logits for ALL positions.
//! - `forward_block_causal` — student mode: prompt positions attend to all
//!   prompt positions; generation positions attend bidirectionally within
//!   their block and causally across blocks.
//!
//! Both write per-position logits into a caller-provided `all_logits`
//! buffer of shape `[seq_len * vocab_size]` and return a slice of the
//! last position's logits for backward compatibility.

use super::*;

/// Bidirectional forward pass for all positions (D2F teacher mode).
///
/// Based on `forward_prefill()` pattern — all positions attend to all positions.
/// Unlike prefill, this:
/// - Returns logits for ALL positions (not just last)
/// - Has no prompt/cache separation
///
/// Writes per-position logits into `all_logits` [`seq_len` × `vocab_size`].
/// Returns a slice of the last position's logits for backward compat.
#[allow(clippy::too_many_arguments)]
pub fn forward_bidirectional<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    all_logits: &'a mut [f32],
) -> &'a mut [f32] {
    let seq_len = tokens.len();
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    assert!(
        seq_len > 0,
        "bidirectional forward requires at least one token"
    );
    assert!(
        seq_len <= config.block_size,
        "seq_len {seq_len} exceeds block_size {}",
        config.block_size
    );
    assert!(
        all_logits.len() >= seq_len * config.vocab_size,
        "all_logits buffer too small"
    );

    // Initialize hidden states for multi-layer
    if config.n_layer > 1 {
        for (p, &token) in tokens.iter().enumerate() {
            let tok_off = token * n;
            let pos_off = p * n;
            unsafe {
                load_embed_add(
                    &mut prefill.hidden[p * n..],
                    &weights.wte,
                    tok_off,
                    &weights.wpe,
                    pos_off,
                    n,
                );
            }
        }
    }

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Phase A: Compute K/V for ALL positions -> store in cache
        for (p, &token) in tokens.iter().enumerate() {
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                unsafe {
                    load_embed_add(&mut ctx.x, &weights.wte, tok_off, &weights.wpe, pos_off, n);
                }
            }

            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            let pos_off = p * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(pos_off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(pos_off),
                    kvd,
                );
            }
        }

        // Phase B: Bidirectional attention for ALL positions (t_n = seq_len)
        for (p, &token) in tokens.iter().enumerate() {
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                for i in 0..n {
                    unsafe {
                        *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                            + *weights.wpe.get_unchecked(pos_off + i);
                    }
                }
            }

            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Bidirectional: t_n = seq_len (full range)
            let scale = 1.0 / (hd as f32).sqrt();
            ctx.attn_out[..n].fill(0.0);
            for h in 0..config.n_head {
                let kv_group = h * n_kv / config.n_head;
                unsafe {
                    attention_head(
                        &ctx.q,
                        &layer_cache.key,
                        &layer_cache.value,
                        &mut ctx.attn_out,
                        &mut ctx.scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        seq_len, // <- BIDIRECTIONAL: full sequence range
                        scale,
                    );
                }
            }

            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // MLP
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                crate::types::matmul(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, config.mlp_hidden, n);
                crate::types::matmul(&mut ctx.up, &layer_weights.mlp_w_up, &ctx.x, config.mlp_hidden, n);
                crate::types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
            }
            #[cfg(not(feature = "gated_mlp"))]
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
            }
            #[cfg(feature = "sparse_mlp")]
            {
                let alive = crate::types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                );
                if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                    crate::types::matmul(
                        &mut ctx.x,
                        &layer_weights.mlp_w2,
                        &ctx.hidden,
                        n,
                        config.mlp_hidden,
                    );
                }
            }
            #[cfg(not(feature = "sparse_mlp"))]
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
                }
            }

            // Always store hidden state for LM Head computation
            prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
        }
    }

    // LM Head for ALL positions
    for p in 0..seq_len {
        ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

        let logits_offset = p * config.vocab_size;
        crate::types::matmul(
            &mut all_logits[logits_offset..logits_offset + config.vocab_size],
            &weights.lm_head,
            &ctx.x,
            config.vocab_size,
            n,
        );
    }

    // Return last position's logits slice for backward compat
    let last_off = (seq_len - 1) * config.vocab_size;
    &mut all_logits[last_off..last_off + config.vocab_size]
}

/// Block-causal forward pass for all positions (D2F student mode).
///
/// Implements the 3-rule D2F attention:
/// - Prompt positions attend to all prompt positions (bidirectional)
/// - Generation positions attend bidirectionally within their block
/// - Generation positions attend causally across blocks
///
/// Writes per-position logits into `all_logits` [`seq_len` × `vocab_size`].
/// Returns a slice of the last position's logits for backward compat.
#[allow(clippy::too_many_arguments)]
pub fn forward_block_causal<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    prompt_len: usize,
    block_size: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    all_logits: &'a mut [f32],
) -> &'a mut [f32] {
    let seq_len = tokens.len();
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    assert!(
        seq_len > 0,
        "block-causal forward requires at least one token"
    );
    assert!(
        seq_len <= config.block_size,
        "seq_len {seq_len} exceeds block_size {}",
        config.block_size
    );
    assert!(
        all_logits.len() >= seq_len * config.vocab_size,
        "all_logits buffer too small"
    );

    // Initialize hidden states for multi-layer
    if config.n_layer > 1 {
        for (p, &token) in tokens.iter().enumerate() {
            let tok_off = token * n;
            let pos_off = p * n;
            unsafe {
                load_embed_add(
                    &mut prefill.hidden[p * n..],
                    &weights.wte,
                    tok_off,
                    &weights.wpe,
                    pos_off,
                    n,
                );
            }
        }
    }

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Phase A: Compute K/V for ALL positions -> store in cache
        for (p, &token) in tokens.iter().enumerate() {
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                unsafe {
                    load_embed_add(&mut ctx.x, &weights.wte, tok_off, &weights.wpe, pos_off, n);
                }
            }

            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            let pos_off = p * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(pos_off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(pos_off),
                    kvd,
                );
            }
        }

        // Phase B: Block-causal attention for ALL positions
        for (p, &token) in tokens.iter().enumerate() {
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                for i in 0..n {
                    unsafe {
                        *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                            + *weights.wpe.get_unchecked(pos_off + i);
                    }
                }
            }

            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Block-causal: t_n depends on position
            let t_n = block_causal_t_n(p, prompt_len, block_size, seq_len);
            let scale = 1.0 / (hd as f32).sqrt();
            ctx.attn_out[..n].fill(0.0);
            for h in 0..config.n_head {
                let kv_group = h * n_kv / config.n_head;
                unsafe {
                    attention_head(
                        &ctx.q,
                        &layer_cache.key,
                        &layer_cache.value,
                        &mut ctx.attn_out,
                        &mut ctx.scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                    );
                }
            }

            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // MLP
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                crate::types::matmul(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, config.mlp_hidden, n);
                crate::types::matmul(&mut ctx.up, &layer_weights.mlp_w_up, &ctx.x, config.mlp_hidden, n);
                crate::types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
            }
            #[cfg(not(feature = "gated_mlp"))]
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
            }
            #[cfg(feature = "sparse_mlp")]
            {
                let alive = crate::types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                );
                if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                    crate::types::matmul(
                        &mut ctx.x,
                        &layer_weights.mlp_w2,
                        &ctx.hidden,
                        n,
                        config.mlp_hidden,
                    );
                }
            }
            #[cfg(not(feature = "sparse_mlp"))]
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
                }
            }

            // Always store hidden state for LM Head computation
            prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
        }
    }

    // LM Head for ALL positions
    for p in 0..seq_len {
        ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

        let logits_offset = p * config.vocab_size;
        crate::types::matmul(
            &mut all_logits[logits_offset..logits_offset + config.vocab_size],
            &weights.lm_head,
            &ctx.x,
            config.vocab_size,
            n,
        );
    }

    // Return last position's logits slice for backward compat
    let last_off = (seq_len - 1) * config.vocab_size;
    &mut all_logits[last_off..last_off + config.vocab_size]
}

/// Set-causal forward pass (Research 376 Phase 0 T0.3, 2026-07-04).
///
/// Generalizes [`forward_block_causal`] to arbitrary position-set orderings
/// per Arriola & Kuleshov, Set Diffusion (arXiv:2607.01775). This is the
/// private-engine CPU reference for the WGSL kernel at
/// `riir-gpu/src/kernels/attention_score_set_causal.wgsl`; the public
/// katgpt-rs counterpart is `forward_set_causal_positions` (T0.2).
///
/// # Attention rule
///
/// For each query position `q` with `gen_step_q = position_order[q]`, attends
/// to all key positions `t` where `position_order[t] <= gen_step_q` — i.e.,
/// positions revealed in the **same generation set** OR **earlier sets**.
/// This realizes the paper's `M_SD` + `M_OSC` + `M_SC` mask composition as one rule.
///
/// # Convention (matches the WGSL kernel)
///
/// `position_order[p]` = the generation step at which position `p` is revealed.
///
/// # Common instantiations
///
/// | Method | `position_order` | Effect |
/// |--------|-----------------|--------|
/// | Block-causal (D2F) | `[0,0,0,0, 1,1,1,1, ...]` (p / B) | Prefix mask |
/// | AR (singleton sets) | `[0, 1, 2, 3, ...]` (p) | Lower-triangular mask |
/// | MDLM (uniform) | `[0, 0, 0, ...]` (all same step) | Fully bidirectional |
/// | SW-SetDLM | sampled from `PositionOffsetSchedule` | Arbitrary sets |
///
/// Writes per-position logits into `all_logits` [`seq_len` × `vocab_size`].
#[cfg(feature = "set_diffusion")]
#[allow(clippy::too_many_arguments)]
pub fn forward_set_causal<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    position_order: &[u32],
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    all_logits: &'a mut [f32],
) -> &'a mut [f32] {
    let seq_len = tokens.len();
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    assert!(
        seq_len > 0,
        "set-causal forward requires at least one token"
    );
    assert!(
        seq_len <= config.block_size,
        "seq_len {seq_len} exceeds block_size {}",
        config.block_size
    );
    assert_eq!(
        position_order.len(),
        seq_len,
        "position_order must have same length as tokens ({}), got {}",
        seq_len,
        position_order.len()
    );
    assert!(
        all_logits.len() >= seq_len * config.vocab_size,
        "all_logits buffer too small"
    );

    // Initialize hidden states for multi-layer
    if config.n_layer > 1 {
        for (p, &token) in tokens.iter().enumerate() {
            let tok_off = token * n;
            let pos_off = p * n;
            unsafe {
                load_embed_add(
                    &mut prefill.hidden[p * n..],
                    &weights.wte,
                    tok_off,
                    &weights.wpe,
                    pos_off,
                    n,
                );
            }
        }
    }

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Phase A: Compute K/V for ALL positions -> store in cache (identical
        // to forward_block_causal — KV projections are mask-independent).
        for (p, &token) in tokens.iter().enumerate() {
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                unsafe {
                    load_embed_add(&mut ctx.x, &weights.wte, tok_off, &weights.wpe, pos_off, n);
                }
            }

            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            let pos_off = p * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(pos_off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(pos_off),
                    kvd,
                );
            }
        }

        // Phase B: Set-causal attention for ALL positions.
        // For each query position p, attend to all t where
        // position_order[t] <= position_order[p] (same-set + preceding sets).
        for (p, &token) in tokens.iter().enumerate() {
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                for i in 0..n {
                    unsafe {
                        *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                            + *weights.wpe.get_unchecked(pos_off + i);
                    }
                }
            }

            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            let query_gen_step = position_order[p];
            let scale = 1.0 / (hd as f32).sqrt();
            ctx.attn_out[..n].fill(0.0);
            for h in 0..config.n_head {
                let kv_group = h * n_kv / config.n_head;
                unsafe {
                    attention_head_set_causal(
                        &ctx.q,
                        &layer_cache.key,
                        &layer_cache.value,
                        &mut ctx.attn_out,
                        &mut ctx.scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        seq_len,
                        scale,
                        position_order,
                        query_gen_step,
                    );
                }
            }

            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // MLP
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                crate::types::matmul(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, config.mlp_hidden, n);
                crate::types::matmul(&mut ctx.up, &layer_weights.mlp_w_up, &ctx.x, config.mlp_hidden, n);
                crate::types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
            }
            #[cfg(not(feature = "gated_mlp"))]
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
            }
            #[cfg(feature = "sparse_mlp")]
            {
                let alive = crate::types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                );
                if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                    crate::types::matmul(
                        &mut ctx.x,
                        &layer_weights.mlp_w2,
                        &ctx.hidden,
                        n,
                        config.mlp_hidden,
                    );
                }
            }
            #[cfg(not(feature = "sparse_mlp"))]
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
                }
            }

            // Always store hidden state for LM Head computation
            prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
        }
    }

    // LM Head for ALL positions
    for p in 0..seq_len {
        ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

        let logits_offset = p * config.vocab_size;
        crate::types::matmul(
            &mut all_logits[logits_offset..logits_offset + config.vocab_size],
            &weights.lm_head,
            &ctx.x,
            config.vocab_size,
            n,
        );
    }

    // Return last position's logits slice for backward compat
    let last_off = (seq_len - 1) * config.vocab_size;
    &mut all_logits[last_off..last_off + config.vocab_size]
}
