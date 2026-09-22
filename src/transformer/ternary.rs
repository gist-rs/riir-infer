//! Ternary-weight forward pass (Plan 333 T2.2).
//!
//! Structurally identical to [`super::llama::forward_llama`] — `RMSNorm` → GQA
//! with `RoPE` → `SwiGLU` → residual, separate `lm_head`, no softcapping. The only
//! difference is that the 7 projection matmuls call
//! [`simd_ternary_group_matvec`] on [`TernaryGroupWeights`] instead of the
//! dense `matmul` on `Vec<f32>`.
//!
//! **No new SIMD kernel lives here.** The group-scale ternary kernel shipped in
//! katgpt-rs Issue 578 (NEON + scalar); this file is the thin wrapper the plan
//! calls for. The one local helper, [`bitlinear`], exists only to slice the
//! caller-owned scratch buffers to the exact lengths the kernel asserts.
//!
//! ## Bonsai is `RMSNorm`, not `LayerNorm`
//!
//! Ternary-Bonsai-27B is Qwen3.6 lineage. `BitNet` b1.58's paper text says
//! "`LayerNorm` before the `BitLinear`"; this model uses `RMSNorm` with a learned
//! gamma and no offset, same as `LLaMA`. `rmsnorm_with_gamma_eps` is correct here.
//!
//! ## Zero-alloc (G4)
//!
//! Every buffer is owned by [`ForwardContext`]; the kernel writes into
//! `&mut [f32]` the caller supplies. The steady-state token loop allocates
//! nothing — asserted by `test_forward_ternary_zero_alloc_steady_state`.

use super::*;
use crate::ternary_layer::TernaryTransformerWeights;
use katgpt_core::{TernaryGroupWeights, simd_ternary_group_matvec_parallel};

/// A `BitLinear` projection: `y[..w.rows] = W × x[..w.cols]`.
///
/// The kernel asserts exact slice lengths (`x.len() == w.cols`,
/// `y.len() == w.rows`), while `ForwardContext` buffers are sized to the
/// worst case across layers. Slicing here is the whole job — the arithmetic
/// lives in katgpt-rs.
///
/// Uses the **row-parallel** kernel (Issue 594 pre-flight, 2026-08-10). On real
/// Ternary-Bonsai-27B shapes it is **7.21×** the serial kernel — 0.25 → 1.80
/// tok/s — and it is bit-identical to serial (rows are independent, so the
/// split changes nothing) and allocation-free, so neither G1 nor G4 is
/// weakened. Below 256 rows it delegates to the serial kernel, which is why the
/// small test fixtures here still exercise the serial path.
#[inline(always)]
fn bitlinear(y: &mut [f32], w: &TernaryGroupWeights, x: &[f32]) {
    simd_ternary_group_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
}

/// Ternary-weight forward pass (decode, single token).
///
/// Writes logits into `ctx.logits` and returns `&mut` to it. Zero-alloc.
pub fn forward_ternary<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TernaryTransformerWeights,
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

    // 1. Embedding: x = wte[token] (dense — the embedding table is not ternary)
    let tok_off = token * n;
    unsafe {
        load_embed(&mut ctx.x, &weights.wte, tok_off, n);
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // b. Pre-attention RMSNorm (learned gamma, no offset)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV BitLinear projections — parallel above the rayon threshold
        //    (below it, the ~5µs task overhead dominates the matvec).
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| bitlinear(q_buf, wq, x_in));
                s.spawn(move |_| bitlinear(k_buf, wk, x_in));
                s.spawn(move |_| bitlinear(v_buf, wv, x_in));
            });
        } else {
            bitlinear(q_buf, wq, x_in);
            bitlinear(k_buf, wk, x_in);
            bitlinear(v_buf, wv, x_in);
        }

        // d. RoPE on Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e. Store K,V in the per-layer cache
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

        // f. Multi-head attention with GQA (no logit softcapping)
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
                0.0,
                config.block_size,
            );
        }

        // g. Output BitLinear + residual (no post-norm)
        bitlinear(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // h. Save residual2 → pre-MLP RMSNorm
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // i. SwiGLU: SiLU(gate·x) ⊙ (up·x)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| bitlinear(gate_buf, wg, x_in));
                s.spawn(move |_| bitlinear(up_buf, wu, x_in));
            });
        } else {
            bitlinear(gate_buf, wg, x_in);
            bitlinear(up_buf, wu, x_in);
        }
        types::swiglu(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // j. Down BitLinear + residual (no post-norm)
        bitlinear(&mut ctx.x, &layer_weights.down_proj, &ctx.hidden);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // 4. Separate lm_head (dense — not ternary, not tied to wte)
    types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Generate tokens using ternary weights with temperature sampling.
///
/// Mirrors [`super::llama::generate_llama`]: prefill the prompt, then decode.
pub fn generate_ternary(
    weights: &TernaryTransformerWeights,
    config: &Config,
    rng: &mut Rng,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    // Pre-allocate for prompt + worst-case generated tokens so the decode loop
    // never reallocates.
    let mut tokens: Vec<usize> = Vec::with_capacity(prompt_tokens.len() + max_tokens);
    tokens.extend_from_slice(prompt_tokens);

    // Prefill: process all prompt tokens; only the last logits are consumed.
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        forward_ternary(&mut ctx, weights, &mut cache, token, pos, config);
    }

    // Generate: autoregressive decode
    for _ in 0..max_tokens {
        softmax_scaled(&mut ctx.logits, 1.0 / config.temperature);

        // Reuse the pre-allocated CDF buffer — no vocab_size alloc per token.
        let next = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        if next == 1 {
            break;
        } // EOS
        tokens.push(next);

        let pos = tokens.len() - 1;
        forward_ternary(&mut ctx, weights, &mut cache, tokens[pos], pos, config);
    }

    tokens.shrink_to_fit();
    tokens
}

#[cfg(test)]
mod tests;
