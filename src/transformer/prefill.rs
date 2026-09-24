//! Bidirectional prefill pipeline.
//!
//! Two-phase prompt processing for prompt-prefill-then-decode generation:
//!
//! - `forward_prefill` — Phase A computes K/V for all prompt positions and
//!   stores in cache; Phase B does bidirectional attention over all prompt
//!   K/V. Returns logits for the last prompt position (used to sample the
//!   first generation token). KV cache is populated as a side effect,
//!   shared with subsequent decode calls.
//! - `generate_with_prefill` — end-to-end pipeline that switches from
//!   reader `LoRA` (during prefill) to writer `LoRA` (during decode).
//! - `generate_with_prefill_and_domain_latent` — domain-latent-conditioned
//!   variant (feature = "`domain_latent`").

use super::*;

/// Bidirectional prefill: process prompt tokens with full mutual attention.
///
/// For each transformer layer:
///   Phase A: Compute K/V for all prompt positions -> store in KV cache
///   Phase B: For each position, attend to ALL prompt K/V (bidirectional)
///
/// Returns logits for the last prompt position (used to sample first gen token).
/// KV cache is populated as a side effect, shared with subsequent decode calls.
///
/// Zero-copy: no allocations. Reuses `ForwardContext` buffers per-position,
/// `PrefillContext::hidden` for multi-layer inter-layer state.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let prompt_len = tokens.len().min(prefill.max_prompt_len);
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    assert!(prompt_len > 0, "prefill requires at least one token");
    assert!(
        prompt_len <= config.block_size,
        "prompt_len {prompt_len} exceeds block_size {}",
        config.block_size
    );

    // Pre-compute embeddings for ALL positions (Issue 045: avoids recomputation in Phase A+B)
    for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
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

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // -- Phase A: Compute K/V + Q for ALL positions -> store in cache --
        // Issue 375: fused Phase A/B — Q projection moved here from Phase B,
        // and the attention residual (xr = rmsnorm(hidden)) is saved for Phase B
        // reuse. This eliminates 1 rmsnorm + 1 Q matmul per position in Phase B.
        for (p, _) in tokens.iter().enumerate().take(prompt_len) {
            // Load hidden state (always from pre-computed embeddings, Issue 045)
            ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

            // Pre-attention norm (matches forward_base exactly: double rmsnorm)
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            // K/V projections
            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
            #[cfg(feature = "domain_latent")]
            if layer_idx == config.n_layer / 2
                && let Some(dl) = domain_latent
            {
                for i in 0..kvd {
                    unsafe {
                        *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                        *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    }
                }
            }

            // Store K/V in cache
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

            // Q projection (fused: pre-computed here, reused in Phase B)
            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Save Q and attention residual (xr) for Phase B reuse
            let q_off = p * n;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.q.as_ptr(),
                    prefill.queries.as_mut_ptr().add(q_off),
                    n,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.xr.as_ptr(),
                    prefill.residuals.as_mut_ptr().add(q_off),
                    n,
                );
            }
        }

        // -- Phase B: Bidirectional attention for ALL positions --
        // Loads pre-computed Q and xr from fused Phase A, skipping redundant
        // hidden state load + rmsnorm + Q matmul per position.
        for (p, _) in tokens.iter().enumerate().take(prompt_len) {
            // Load pre-computed Q from fused Phase A
            let q_off = p * n;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    prefill.queries.as_ptr().add(q_off),
                    ctx.q.as_mut_ptr(),
                    n,
                );
                // Load pre-computed attention residual (xr = rmsnorm(hidden))
                std::ptr::copy_nonoverlapping(
                    prefill.residuals.as_ptr().add(q_off),
                    ctx.xr.as_mut_ptr(),
                    n,
                );
            }

            // Bidirectional attention (Issue 042: parallel for long prompts)
            let scale = 1.0 / (hd as f32).sqrt();
            ctx.attn_out[..n].fill(0.0);
            unsafe {
                attention_heads_parallel(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.head_scores,
                    config.n_head,
                    n_kv,
                    kvd,
                    hd,
                    prompt_len,
                    scale,
                    0.0,
                    config.block_size,
                );
            }

            // Output projection + residual
            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // MLP: residual -> RMSNorm -> MLP -> residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                crate::types::matmul(
                    &mut ctx.hidden,
                    &layer_weights.mlp_w1,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                crate::types::matmul(
                    &mut ctx.up,
                    &layer_weights.mlp_w_up,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
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
            // MLP w2 (with sparse support)
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

            // Store hidden state for next layer (multi-layer only)
            if config.n_layer > 1 {
                prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
            }
        }
    }

    // Snapshot hidden state (last position)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    crate::types::matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Full generation pipeline: bidirectional prefill -> causal decode.
/// Switches from reader `LoRA` to writer `LoRA` at the prefill->decode boundary.
/// Zero-copy: all buffers pre-allocated, no allocations in request path.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    // 1. Bidirectional prefill with reader LoRA
    let logits = {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
            )
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
                domain_latent,
            )
        }
    };

    // 2. Sample first generation token from prefill output
    // softmax_scaled fuses temperature division + softmax, saving one pass vs manual divide
    crate::types::softmax_scaled(logits, 1.0 / config.temperature);
    // Reuse pre-allocated CDF buffer -- avoids vocab_size allocation per token.
    // Re-borrow ctx.logits directly to avoid conflicting with cdf_buf borrow.
    let mut token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);

    // Pre-allocate for the worst-case generation count to avoid reallocation
    // in the decode loop. Starts with the first sampled token.
    let mut generated = Vec::with_capacity(max_gen_tokens);
    generated.push(token);

    // 3. Causal decode with writer LoRA
    for pos in prompt_tokens.len().. {
        if pos >= config.block_size || generated.len() >= max_gen_tokens {
            break;
        }

        let logits = {
            #[cfg(not(feature = "domain_latent"))]
            {
                super::forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                super::forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                    domain_latent,
                )
            }
        };
        // softmax_scaled fuses temperature division + softmax, saving one pass vs manual divide
        crate::types::softmax_scaled(logits, 1.0 / config.temperature);

        // Reuse pre-allocated CDF buffer -- avoids vocab_size allocation per token.
        // Re-borrow ctx.logits directly to avoid conflicting with cdf_buf borrow.
        token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        generated.push(token);

        if token == config.bos_token {
            break;
        }
    }

    generated.shrink_to_fit();
    generated
}

/// Generate with prefill and optional domain latent (Plan 038).
/// Convenience wrapper for callers that need domain conditioning during generation.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill_and_domain_latent(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    generate_with_prefill(
        ctx,
        prefill,
        weights,
        cache,
        config,
        rng,
        prompt_tokens,
        max_gen_tokens,
        lora_pair,
        domain_latent,
    )
}
