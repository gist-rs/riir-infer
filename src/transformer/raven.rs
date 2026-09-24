//! Raven Routing Slot Memory (RSM) — fixed-size KV cache replacement.
//!
//! Distilled from "Raven: High-Recall Sequence Modeling with Sparse Memory
//! Routing". Replaces the growing `[block_size, kv_dim]` cache with a fixed
//! `[num_slots, kv_dim]` memory updated via sparse Top-K routing. Unselected
//! slots are completely frozen. Per-token compute: `O(num_slots`) — constant
//! regardless of sequence length.

use super::*;

// Issue 019 F2 / Plan 406 Phase 2 T2.3: `RavenKVCache` is now re-exported
// from katgpt-transformer (canonical superset — adds `readout_scores`/
// `readout_output` pre-alloc buffers + `r_t()` accessor; visibility on
// `router_scored`/`router_r_t` widened from `pub(crate)` to `pub`, which is
// strictly more permissive). The local forward path (`forward_raven`) and
// router/readout/update kernels stay here — they are runtime-specific.

/// Sparse router: computes Top-K routing vector from raw logits (zero-alloc variant).
///
/// Implements: `r_t = Normalize(TopK(Sigmoid(raw_logits)))`
/// Unselected slots get 0.0 -> completely frozen during update.
///
/// Uses pre-allocated buffers to avoid heap allocations on the hot path.
///
/// `scored` is resized in-place (clear + extend from raw logits) to avoid per-push reallocation.
/// `r_t` is zero-filled then partially populated for Top-K slots only.
pub fn raven_compute_router_into(
    raw_logits: &[f32],
    top_k: usize,
    scored: &mut Vec<(usize, f32)>,
    r_t: &mut Vec<f32>,
) {
    let num_slots = raw_logits.len();
    let top_k = top_k.min(num_slots);

    // Reuse pre-allocated buffers: resize once then fill by index
    // (avoids per-push capacity/length checks in the hot loop)
    scored.resize(num_slots, (0, 0.0f32));
    for (i, &x) in raw_logits.iter().enumerate() {
        // fast_sigmoid is #[inline(always)] — no per-element call overhead.
        scored[i] = (i, katgpt_core::simd::fast_sigmoid(x));
    }

    // Partial sort: find Top-K by descending score (O(n) average)
    if top_k < num_slots {
        scored.select_nth_unstable_by(num_slots - top_k, |a, b| {
            katgpt_core::float_order::asc(a.1, b.1)
        });
    }

    // Zero-fill r_t then populate only Top-K slots
    r_t.clear();
    r_t.resize(num_slots, 0.0);
    let mut sum = 0.0f32;

    // Keep only Top-K (the last top_k elements after partial sort are the largest)
    for (idx, score) in scored.iter().rev().take(top_k) {
        r_t[*idx] = *score;
        sum += *score;
    }

    // Normalize only the Top-K slots (skip zero entries)
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for (idx, score) in scored.iter().rev().take(top_k) {
            r_t[*idx] = *score * inv_sum;
        }
    }
}

/// Backward-compatible wrapper that allocates fresh buffers.
pub fn raven_compute_router(raw_logits: &[f32], top_k: usize) -> Vec<f32> {
    // Pre-allocate scratch buffers to input size (avoids growth reallocs).
    let n = raw_logits.len();
    let mut scored = Vec::with_capacity(n);
    let mut r_t = Vec::with_capacity(n);
    raven_compute_router_into(raw_logits, top_k, &mut scored, &mut r_t);
    r_t
}

/// Gated memory update: Raven Equation 18.
///
/// For each slot:
///   `decay = exp(forget_rate * r_t[slot])`
///   `H_new = decay * H_old + (1 - decay) * new_content`
///
/// When `r_t[slot] == 0`: `decay = exp(0) = 1.0` -> `H_new = H_old` (FROZEN)
/// When `r_t[slot] > 0`: `decay < 1.0` -> old content decays, new writes in
#[allow(clippy::too_many_arguments)]
pub fn raven_update(
    keys: &mut [f32],
    values: &mut [f32],
    new_key: &[f32],
    new_value: &[f32],
    r_t: &[f32],
    forget_rate: f32,
    num_slots: usize,
    kv_dim: usize,
) {
    for (slot, &route) in r_t.iter().enumerate().take(num_slots) {
        if route == 0.0 {
            continue; // Frozen slot -- skip entirely
        }
        let decay = (forget_rate * route).exp();
        let write = 1.0 - decay;
        let offset = slot * kv_dim;

        // Process 4 elements at a time to help LLVM auto-vectorize.
        // mul_add emits FMA: decay*H + write*new = H.mul_add(decay, write*new)
        // → 1 FMA + 1 mul per element vs 2 mul + 1 add (33% fewer FP ops).
        let chunks = kv_dim / 4;
        for c in 0..chunks {
            let d = c * 4;
            keys[offset + d] = keys[offset + d].mul_add(decay, write * new_key[d]);
            keys[offset + d + 1] = keys[offset + d + 1].mul_add(decay, write * new_key[d + 1]);
            keys[offset + d + 2] = keys[offset + d + 2].mul_add(decay, write * new_key[d + 2]);
            keys[offset + d + 3] = keys[offset + d + 3].mul_add(decay, write * new_key[d + 3]);
            values[offset + d] = values[offset + d].mul_add(decay, write * new_value[d]);
            values[offset + d + 1] =
                values[offset + d + 1].mul_add(decay, write * new_value[d + 1]);
            values[offset + d + 2] =
                values[offset + d + 2].mul_add(decay, write * new_value[d + 2]);
            values[offset + d + 3] =
                values[offset + d + 3].mul_add(decay, write * new_value[d + 3]);
        }
        // Handle remaining elements
        for d in (chunks * 4)..kv_dim {
            keys[offset + d] = keys[offset + d].mul_add(decay, write * new_key[d]);
            values[offset + d] = values[offset + d].mul_add(decay, write * new_value[d]);
        }
    }
}

/// Readout: attention over fixed slot memory.
/// `O(num_slots * kv_dim)` -- constant regardless of sequence length.
///
/// **Allocating convenience wrapper.** For hot paths, prefer
/// [`raven_readout_into`] which reuses pre-allocated buffers.
pub fn raven_readout(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut scores_buf = vec![0.0f32; num_slots];
    let mut output = vec![0.0f32; kv_dim];
    raven_readout_into(
        query,
        keys,
        values,
        num_slots,
        kv_dim,
        &mut scores_buf,
        &mut output,
    );
    output
}

/// Zero-alloc variant of `raven_readout` that writes into pre-allocated buffers.
///
/// Uses SIMD dot product for Q.K^T scoring and pre-computes `inv_sum` to
/// eliminate division inside the value accumulation loop.
pub fn raven_readout_into(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
    scores_buf: &mut [f32],
    output: &mut [f32],
) {
    // Q . K^T into scores_buf using SIMD dot product
    for (i, k_chunk) in keys.chunks(kv_dim).take(num_slots).enumerate() {
        scores_buf[i] = crate::simd::simd_dot_f32(query, k_chunk, kv_dim);
    }

    let max_score = scores_buf[..num_slots]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(scores - max) in-place, accumulate sum
    let mut sum_exp = 0.0f32;
    for s in &mut scores_buf[..num_slots] {
        let e = (*s - max_score).exp();
        *s = e;
        sum_exp += e;
    }
    let inv_sum = 1.0 / sum_exp;
    output[..kv_dim].fill(0.0);
    for (i, v_chunk) in values.chunks(kv_dim).take(num_slots).enumerate() {
        let weight = scores_buf[i] * inv_sum; // scores_buf now holds exp(scores - max)
        for (out, v) in output[..kv_dim].iter_mut().zip(v_chunk) {
            *out += weight * v;
        }
    }
}

/// Forward pass using `RavenKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` except attention:
/// - Generates router logits from K projection (dummy: use K directly)
/// - Calls `raven_update()` instead of writing to flat KV array
/// - Calls `raven_readout()` instead of scanning all past positions
/// - Everything else (`RMSNorm`, MLP, residual, LM head) stays identical
pub fn forward_raven<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut RavenKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    unsafe {
        load_embed_add(
            &mut ctx.x,
            &weights.wte,
            tok_off,
            &weights.wpe,
            pos_off_emb,
            n,
        );
    }

    // 2. Layer loop
    for layer_weights in &weights.layers {
        // Pre-attention: RMSNorm -> save residual -> RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Raven: generate router logits from K (dummy projection)
        // For PoC: use first num_slots elements of K repeated as logits.
        // In production, this would be a learned linear projection: W_route * x_t
        // Reuse pre-allocated query buffer for router logits (zero-alloc)
        ctx.raven_query_buf.resize(cache.num_slots, 0.0);
        for (i, slot) in ctx.raven_query_buf.iter_mut().enumerate() {
            *slot = ctx.k[i % kvd];
        }

        // Raven: compute sparse routing vector (zero-alloc via pre-allocated buffers)
        raven_compute_router_into(
            &ctx.raven_query_buf,
            cache.top_k,
            &mut cache.router_scored,
            &mut cache.router_r_t,
        );

        // Copy router_r_t to stack to avoid self-borrow (Issue 023)
        // num_slots is bounded (<=64), so stack allocation is fine
        let mut r_t_stack = [0.0f32; 64];
        let r_t_len = cache.router_r_t.len().min(64);
        r_t_stack[..r_t_len].copy_from_slice(&cache.router_r_t[..r_t_len]);

        // Raven: gated update (only selected slots are modified)
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &ctx.k,
            &ctx.v,
            &r_t_stack[..r_t_len],
            cache.forget_rate,
            cache.num_slots,
            kvd,
        );

        // Raven: readout via attention over fixed slots (O(num_slots) not O(pos))
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);

        for h in 0..config.n_head {
            let q_off = h * hd;
            // Each head reads from the slot memory using its query slice
            let head_query = &ctx.q[q_off..q_off + hd];
            // Pad/reshape query to kv_dim for slot attention (reuse pre-allocated buffer)
            // Issue 022: avoid redundant resize, just fill the kvd slice
            ctx.raven_query_buf[..kvd].fill(0.0);
            let kv_group = h * n_kv / config.n_head;
            for (d, &hq) in head_query.iter().enumerate() {
                ctx.raven_query_buf[kv_group * hd + d] = hq * scale;
            }

            // Issue 020: zero-alloc readout
            raven_readout_into(
                &ctx.raven_query_buf,
                &cache.keys,
                &cache.values,
                cache.num_slots,
                kvd,
                &mut ctx.raven_scores_buf,
                &mut ctx.raven_output_buf,
            );
            let slot_values = &ctx.raven_output_buf[..kvd];

            // Extract this head's attention output
            for d in 0..hd {
                unsafe {
                    *ctx.attn_out.get_unchecked_mut(q_off + d) = slot_values[kv_group * hd + d];
                }
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual -> RMSNorm -> MLP -> residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.up,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}
