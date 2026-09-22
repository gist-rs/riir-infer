//! LLaMA-family forward pass.
//!
//! Supports `LLaMA`, Mistral, `MiniCPM`, and other LLaMA-architecture models.
//! Differences from Gemma 2:
//! - No embedding scaling (no `sqrt(n_embd`))
//! - No post-attention / post-MLP norms
//! - `SwiGLU` MLP (`SiLU` activation) instead of `GeGLU`
//! - Separate `lm_head` (not tied to wte)
//! - No logit softcapping

use super::*;
use crate::llama_layer::LlamaTransformerWeights;

/// LLaMA-family forward pass (decode, single token).
pub fn forward_llama<'a>(
    ctx: &'a mut ForwardContext,
    weights: &LlamaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] (no scaling, unlike Gemma 2)
    let tok_off = token * n;
    unsafe {
        load_embed(&mut ctx.x, &weights.wte, tok_off, n);
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // b. Pre-attention RMSNorm (no offset)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV projections — parallel (Issue 053: serial for small models)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }

        // d. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e. Store K,V in per-layer cache
        let pos_off = pos * kvd;
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

        // f. Multi-head attention with GQA (no softcapping for LLaMA)
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..q_dim].fill(0.0);
        let t_n = pos + 1;
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
                t_n,
                scale,
                0.0, // no attention logit softcapping
                config.block_size,
            );
        }

        // g. Output projection
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // h. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // i. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);

        // j. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // k. SwiGLU (Issue 053: threshold-gated parallelism)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::swiglu(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // l. Down projection
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // m. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: reuse existing infrastructure if enabled
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(ctx, layer_idx, n, None);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // 4. Separate lm_head (not tied)
    types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Generate tokens using `LLaMA` weights with temperature sampling.
pub fn generate_llama(
    weights: &LlamaTransformerWeights,
    config: &Config,
    rng: &mut Rng,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    // Pre-allocate for prompt + worst-case generated tokens so the autoregressive
    // decode loop never reallocates.
    let mut tokens: Vec<usize> = Vec::with_capacity(prompt_tokens.len() + max_tokens);
    tokens.extend_from_slice(prompt_tokens);

    // Prefill: process all prompt tokens
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        forward_llama(&mut ctx, weights, &mut cache, token, pos, config);
    }

    // Generate: autoregressive decode
    for _ in 0..max_tokens {
        // Temperature scaling + softmax on last forward pass logits
        softmax_scaled(&mut ctx.logits, 1.0 / config.temperature);

        // Reuse pre-allocated CDF buffer -- avoids vocab_size allocation per token
        let next = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        if next == 1 {
            break;
        } // EOS
        tokens.push(next);

        // Forward pass for next position
        let pos = tokens.len() - 1;
        forward_llama(&mut ctx, weights, &mut cache, tokens[pos], pos, config);
    }

    tokens.shrink_to_fit();
    tokens
}

/// LLaMA-family forward pass that captures attention pattern features at specified layers.
///
/// This is a **cold-path diagnostic function** (Issue 380 Path A) — it duplicates
/// the layer loop from [`forward_llama`] to compute attention entropy at the
/// last token position for each layer in `capture_layers`. The entropy is
/// computed from `softmax(Q_h · K^T)` over all positions `[0..t_n]` for each
/// head `h`, after `RoPE` is applied to Q/K and after K is stored in the cache.
///
/// **Do NOT use in hot paths** — the entropy computation adds `O(n_head * t_n)`
/// work per captured layer, plus a `t_n`-sized allocation. Use [`forward_llama`]
/// for production decode.
///
/// The rest of the forward pass is identical to [`forward_llama`]: no embedding
/// scaling, `SwiGLU` MLP, RMSNorm-with-gamma pre-norms (no post-norms), separate
/// `lm_head`, and no logit softcapping.
///
/// Returns `(&mut ctx.logits, Vec<AttnLayerFeatures>)`. The features vector
/// has one entry per layer in `capture_layers` (in ascending layer order, since
/// the layer loop is sequential). Layers not in `capture_layers` are skipped.
///
/// # Contract
///
/// - `capture_layers` entries must be `< config.n_layer`.
/// - `token < config.vocab_size`.
/// - `pos` is the 0-based position of `token` in the sequence; `t_n = pos + 1`.
/// - The KV cache for positions `[0..pos]` must already be populated.
#[allow(clippy::too_many_arguments)]
pub fn forward_llama_attn_capture<'a>(
    ctx: &'a mut ForwardContext,
    weights: &LlamaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    capture_layers: &[usize],
) -> (&'a mut [f32], Vec<AttnLayerFeatures>) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] (no scaling, unlike Gemma 2)
    let tok_off = token * n;
    unsafe {
        load_embed(&mut ctx.x, &weights.wte, tok_off, n);
    }

    // Loop-invariant attention scale and token count.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;
    let ln_tn = (t_n as f32).ln().max(1e-12);

    let mut features: Vec<AttnLayerFeatures> = Vec::with_capacity(capture_layers.len());

    // 2. Layer loop (inlined from forward_llama — no hooks, no LoRA).
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // b. Pre-attention RMSNorm (no offset)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV projections — parallel (Issue 053: serial for small models)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }

        // d. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e. Store K,V in per-layer cache
        let pos_off = pos * kvd;
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

        // ── Attention capture (Issue 380 Path A) ──
        // After RoPE+cache-store, before the fused attention call: compute
        // per-head softmax(Q·K^T) at the last token position to extract
        // entropy / max-weight / self-weight. LLaMA has no attention logit
        // softcapping, so the score is just dot * scale.
        if capture_layers.contains(&layer_idx) {
            let mut entropy_sum = 0.0_f32;
            let mut max_weight_sum = 0.0_f32;
            let mut self_weight_sum = 0.0_f32;

            let scores_buf = &mut ctx.head_scores[..t_n];

            for h in 0..config.n_head {
                let q_off = h * hd;
                let kv_group = h * n_kv / config.n_head;
                let kv_group_off = kv_group * hd;

                // Compute Q·K scores for all positions [0..t_n].
                let mut max_score = f32::NEG_INFINITY;
                for (t, slot) in scores_buf.iter_mut().enumerate().take(t_n) {
                    let k_off = t * kvd + kv_group_off;
                    let dot = crate::simd::simd_dot_f32(
                        &ctx.q[q_off..q_off + hd],
                        &layer_cache.key[k_off..k_off + hd],
                        hd,
                    );
                    let score = dot * scale; // LLaMA: no softcapping
                    *slot = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Softmax (stable: subtract max).
                let mut sum = 0.0_f32;
                for slot in scores_buf.iter_mut().take(t_n) {
                    let e = (*slot - max_score).exp();
                    *slot = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;

                // Entropy + max weight + self weight (last position).
                let mut entropy = 0.0_f32;
                let mut max_w = 0.0_f32;
                for &e in scores_buf.iter().take(t_n) {
                    let w = e * inv_sum;
                    if w > 1e-12 {
                        entropy -= w * w.ln();
                    }
                    if w > max_w {
                        max_w = w;
                    }
                }
                let self_w = scores_buf[t_n - 1] * inv_sum;

                entropy_sum += entropy / ln_tn;
                max_weight_sum += max_w;
                self_weight_sum += self_w;
            }

            let n_head_f = config.n_head as f32;
            features.push(AttnLayerFeatures {
                layer_idx,
                entropy_normalized: entropy_sum / n_head_f,
                mean_max_weight: max_weight_sum / n_head_f,
                mean_self_weight: self_weight_sum / n_head_f,
            });
        }

        // f. Multi-head attention with GQA (no softcapping for LLaMA)
        ctx.attn_out[..q_dim].fill(0.0);
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
                t_n,
                scale,
                0.0, // no attention logit softcapping
                config.block_size,
            );
        }

        // g. Output projection
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // h. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // i. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);

        // j. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // k. SwiGLU (Issue 053: threshold-gated parallelism)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::swiglu(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // l. Down projection
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // m. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: reuse existing infrastructure if enabled
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(ctx, layer_idx, n, None);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // 4. Separate lm_head (not tied)
    types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    (&mut ctx.logits, features)
}
