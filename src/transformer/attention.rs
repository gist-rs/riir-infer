//! Attention primitives — fused QKV head kernels + loss helpers.
//!
//! Contains the per-head attention kernels (`attention_head*`) used by the
//! decode/prefill paths, the parallel multi-head dispatcher
//! (`attention_heads_parallel`), and the D2F loss/boundary helpers
//! (`masked_cross_entropy`, `block_causal_t_n`).
//!
//! All kernels are zero-allocation: callers pass pre-allocated score
//! buffers. Softmax is fused into the value accumulation pass to avoid a
//! separate normalization round-trip.

use rayon::prelude::*;

/// Fused attention head with GQA support: score -> softmax -> weighted value sum.
/// Avoids separate `softmax()` call and write-back of normalized scores.
///
/// GQA: each Q head (`q_head_offset / hd`) maps to a KV group (`kv_group_offset / hd`).
/// When `n_kv_head == n_head`, `kv_group_offset == q_head_offset` and `kv_dim == n_embd`
/// -> identical to standard MHA (backward compatible).
///
/// # Safety
///
/// Caller must ensure all indices are in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn attention_head(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
) {
    // Pass 1: compute Q.K scores and find max for numerical stability
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = crate::simd::simd_dot_f32(
            &q[q_head_offset..q_head_offset + hd],
            &key_cache[k_off..k_off + hd],
            hd,
        );
        let score = dot * scale;
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    // Pass 2: exp(scores - max) and accumulate sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation (t-outer loop for cache locality)
    // Iterating t-outer, d-inner gives sequential access to both value_cache
    // and attn_out, avoiding kv_dim-strided reads on the inner loop.
    let inv_sum = 1.0 / sum;
    for t in 0..t_n {
        let w = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
        let v_off = t * kv_dim + kv_group_offset;
        for d in 0..hd {
            unsafe {
                *attn_out.get_unchecked_mut(q_head_offset + d) +=
                    w * *value_cache.get_unchecked(v_off + d);
            }
        }
    }
}

/// Set-causal attention head (Research 376 Phase 0 T0.3, 2026-07-04).
///
/// Generalizes [`attention_head`] from a prefix mask `[0..t_n]` to an arbitrary
/// **set-causal** eligibility rule: position `t` is eligible iff
/// `position_order[t] <= query_gen_step`. This is the CPU mirror of the WGSL
/// kernel at `riir-gpu/src/kernels/attention_score_set_causal.wgsl` and the
/// private-engine counterpart of `katgpt-rs::forward_set_causal_positions`.
///
/// Source: Arriola & Kuleshov, Set Diffusion (arXiv:2607.01775). The eligibility
/// rule realizes the paper's `M_SD` (set-diagonal) + `M_OSC` (offset set-causal)
/// + `M_SC` (set-causal) masks as a single predicate.
///
/// Uses scalar `f32::exp` per eligible position (NOT SIMD Cephes polynomial)
/// because the SIMD exp's range-reduction saturates on the `-inf` sentinel
/// used for ineligible positions and yields NaN. Computing `exp(score - max)`
/// only on eligible positions and explicitly zeroing the rest avoids that trap.
///
/// # Safety
///
/// Caller must ensure all indices are in bounds and that
/// `position_order.len() >= seq_len`, `scores_buf.len() >= seq_len`.
#[cfg(feature = "set_diffusion")]
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn attention_head_set_causal(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    seq_len: usize,
    scale: f32,
    position_order: &[u32],
    query_gen_step: u32,
) {
    // Pass 1: compute Q.K scores for ELIGIBLE positions only, find max.
    // Self-attention (t == query position) is always eligible by construction
    // (position_order[query_pos] == query_gen_step), guaranteeing a finite max.
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..seq_len {
        if unsafe { *position_order.get_unchecked(t) } <= query_gen_step {
            let k_off = t * kv_dim + kv_group_offset;
            let dot = crate::simd::simd_dot_f32(
                &q[q_head_offset..q_head_offset + hd],
                &key_cache[k_off..k_off + hd],
                hd,
            );
            let score = dot * scale;
            unsafe {
                *scores_buf.get_unchecked_mut(t) = score;
            }
            if score > max_score {
                max_score = score;
            }
        }
    }

    // Pass 2: exp(score - max) for eligible, zero ineligible, accumulate sum.
    let mut sum = 0.0f32;
    for t in 0..seq_len {
        if unsafe { *position_order.get_unchecked(t) } <= query_gen_step {
            let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
            unsafe {
                *scores_buf.get_unchecked_mut(t) = exp_val;
            }
            sum += exp_val;
        }
    }

    // Pass 3: normalize eligible weights + weighted value accumulation.
    // Ineligible positions contribute zero (their scores_buf slot is stale but
    // never read because the same eligibility gate applies).
    let inv_sum = 1.0 / sum;
    for t in 0..seq_len {
        if unsafe { *position_order.get_unchecked(t) } <= query_gen_step {
            let w = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
            let v_off = t * kv_dim + kv_group_offset;
            for d in 0..hd {
                unsafe {
                    *attn_out.get_unchecked_mut(q_head_offset + d) +=
                        w * *value_cache.get_unchecked(v_off + d);
                }
            }
        }
    }
}

/// Attention head with tanh logit softcapping (Gemma 2).
/// Identical to `attention_head` except scores are softcapped:
///   score = softcap * tanh(score / softcap)
/// This prevents attention logits from growing too large.
///
/// Widened `pub` (was `pub(crate)`, Proposal 041 T1.1) so the riir-engine
/// `causal_validation` module can call it per-head to capture attention
/// weights — the parallel dispatcher reuses the score buffer in sequential
/// mode, so capture must run heads one at a time.
///
/// # Safety
///
/// Caller must ensure all indices are in bounds.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn attention_head_softcap(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    softcap: f32,
) {
    // Pass 1: compute Q.K scores, apply softcapping, find max.
    // Fold the two loop-invariants (scale, 1/softcap) into one factor so the
    // per-t score is a single multiply instead of multiply + divide.
    let scale_over_softcap = scale / softcap;
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = crate::simd::simd_dot_f32(
            &q[q_head_offset..q_head_offset + hd],
            &key_cache[k_off..k_off + hd],
            hd,
        );
        let score = softcap * crate::simd::fast_tanh(dot * scale_over_softcap);
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    // Pass 2: exp(scores - max) and accumulate sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation (t-outer loop for cache locality)
    let inv_sum = 1.0 / sum;
    for t in 0..t_n {
        let w = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
        let v_off = t * kv_dim + kv_group_offset;
        for d in 0..hd {
            unsafe {
                *attn_out.get_unchecked_mut(q_head_offset + d) +=
                    w * *value_cache.get_unchecked(v_off + d);
            }
        }
    }
}

/// Per-head attention with softcapping, operating on pre-split slices (Plan 096).
///
/// Same as [`attention_head_softcap`] but takes per-head slices directly,
/// enabling `par_chunks_mut`-based parallel execution with non-overlapping `&mut [f32]`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn attention_head_on_slices(
    q_head: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out_head: &mut [f32],
    scores_buf: &mut [f32],
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    softcap: f32,
) {
    // Pass 1: compute Q.K scores, apply softcapping, find max.
    // Fold the two loop-invariants (scale, 1/softcap) into one factor so the
    // per-t score is a single multiply instead of multiply + divide.
    let scale_over_softcap = scale / softcap;
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = crate::simd::simd_dot_f32(&q_head[..hd], &key_cache[k_off..k_off + hd], hd);
        let score = softcap * crate::simd::fast_tanh(dot * scale_over_softcap);
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    // Pass 2: exp(scores - max) and accumulate sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation (t-outer loop for cache locality)
    let inv_sum = 1.0 / sum;
    for t in 0..t_n {
        let w = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
        let v_off = t * kv_dim + kv_group_offset;
        for d in 0..hd {
            unsafe {
                *attn_out_head.get_unchecked_mut(d) += w * *value_cache.get_unchecked(v_off + d);
            }
        }
    }
}

/// Non-softcap variant of [`attention_head_on_slices`].
///
/// Used when `softcap == 0.0` to avoid division by zero in the softcap formula.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn attention_head_on_slices_no_softcap(
    q_head: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out_head: &mut [f32],
    scores_buf: &mut [f32],
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
) {
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = crate::simd::simd_dot_f32(&q_head[..hd], &key_cache[k_off..k_off + hd], hd);
        let score = dot * scale;
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    let inv_sum = 1.0 / sum;
    for t in 0..t_n {
        let w = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
        let v_off = t * kv_dim + kv_group_offset;
        for d in 0..hd {
            unsafe {
                *attn_out_head.get_unchecked_mut(d) += w * *value_cache.get_unchecked(v_off + d);
            }
        }
    }
}

/// Multi-head attention with parallel head execution via rayon (Plan 096).
///
/// Each head `h` gets its own score region: `head_scores[h * block_size..(h+1) * block_size]`.
/// Output writes are non-overlapping: `attn_out[h * head_dim..(h+1) * head_dim]`.
/// K/V cache reads are shared (read-only) across heads with the same GQA group.
///
/// Falls back to sequential for short sequences where thread overhead exceeds benefit.
/// Issue 741 T10 Phase A: widened `pub` — consumed cross-crate by the
/// relocated riir-train-engine training modules (`gemma2_train` / `gemma4_train`).
/// D4 drift-row reversible.
///
/// # Safety
///
/// All slice accesses use unchecked indexing. The caller must uphold:
///
/// - `q.len() >= n_head * head_dim`
/// - `key_cache.len() >= t_n * kv_dim` and `value_cache.len() >= t_n * kv_dim`,
///   with `n_kv_head * head_dim <= kv_dim` (the per-step GQA read window
///   `kv_group * head_dim .. + head_dim` must fit inside one `kv_dim` row)
/// - `attn_out.len() >= n_head * head_dim`, and `attn_out` must not alias
///   `q`, `key_cache`, or `value_cache` (written via unchecked mutable access)
/// - `head_scores.len() >= n_head * block_size` for the rayon path (each head
///   gets its own `block_size` score region; a shorter slice silently skips
///   the excess heads). The sequential fallback only needs `>= block_size`.
/// - `t_n <= block_size` (per-head scores are indexed `0..t_n` within one block)
///
/// `attn_out` is **accumulated into** (`+=`), not assigned — zero it first if
/// pure attention output is required.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_heads_parallel(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    head_scores: &mut [f32],
    n_head: usize,
    n_kv_head: usize,
    kv_dim: usize,
    head_dim: usize,
    t_n: usize,
    scale: f32,
    softcap: f32,
    block_size: usize,
) {
    /// Minimum sequence length before parallelizing attention heads.
    /// Below this, sequential execution avoids thread spawning overhead.
    /// At `t_n<512` each head does <131K ops (~8us) -- rayon spawn cost exceeds savings.
    /// Effectively disables parallel heads for typical decode lengths (<100 tokens).
    const PARALLEL_HEADS_MIN_SEQ: usize = 512;

    if t_n < PARALLEL_HEADS_MIN_SEQ || n_head <= 1 {
        // Sequential fallback: reuse head_scores[0..block_size] for all heads
        for h in 0..n_head {
            let kv_group = h * n_kv_head / n_head;
            unsafe {
                if softcap > 0.0 {
                    attention_head_softcap(
                        q,
                        key_cache,
                        value_cache,
                        attn_out,
                        &mut head_scores[..block_size],
                        h * head_dim,
                        kv_group * head_dim,
                        kv_dim,
                        head_dim,
                        t_n,
                        scale,
                        softcap,
                    );
                } else {
                    attention_head(
                        q,
                        key_cache,
                        value_cache,
                        attn_out,
                        &mut head_scores[..block_size],
                        h * head_dim,
                        kv_group * head_dim,
                        kv_dim,
                        head_dim,
                        t_n,
                        scale,
                    );
                }
            }
        }
    } else {
        // Parallel: rayon par_chunks_mut gives each head exclusive &mut [f32].
        // SAFETY: par_chunks_mut guarantees non-overlapping mutable slices per head.
        let q_dim = n_head * head_dim;
        if softcap > 0.0 {
            attn_out[..q_dim]
                .par_chunks_mut(head_dim)
                .zip(head_scores.par_chunks_mut(block_size))
                .enumerate()
                .for_each(|(h, (attn_h, scores_h))| {
                    let kv_group_offset = (h * n_kv_head / n_head) * head_dim;
                    unsafe {
                        attention_head_on_slices(
                            &q[h * head_dim..],
                            key_cache,
                            value_cache,
                            attn_h,
                            scores_h,
                            kv_group_offset,
                            kv_dim,
                            head_dim,
                            t_n,
                            scale,
                            softcap,
                        );
                    }
                });
        } else {
            attn_out[..q_dim]
                .par_chunks_mut(head_dim)
                .zip(head_scores.par_chunks_mut(block_size))
                .enumerate()
                .for_each(|(h, (attn_h, scores_h))| {
                    let kv_group_offset = (h * n_kv_head / n_head) * head_dim;
                    unsafe {
                        attention_head_on_slices_no_softcap(
                            &q[h * head_dim..],
                            key_cache,
                            value_cache,
                            attn_h,
                            scores_h,
                            kv_group_offset,
                            kv_dim,
                            head_dim,
                            t_n,
                            scale,
                        );
                    }
                });
        }
    }
}

// ── D2F: Discrete Diffusion Language Model (Plan 068) ───────────────

/// Compute cross-entropy loss on masked positions only, importance-weighted by `1/p_mask`.
///
/// Matches reference: loss computation in D2F-train/utils/loss.py:31-66
/// loss = CE(`student_logits[masked]`, `targets[masked]`) / `p_mask[masked]`
/// Returns mean loss over masked positions, or 0.0 if no positions are masked.
#[cfg(feature = "dllm")]
pub fn masked_cross_entropy(
    logits: &[f32],
    targets: &[usize],
    mask_indicators: &[bool],
    p_masks: &[f32],
    vocab_size: usize,
) -> f32 {
    let seq_len = targets.len();
    debug_assert_eq!(logits.len(), seq_len * vocab_size);
    debug_assert_eq!(mask_indicators.len(), seq_len);
    debug_assert_eq!(p_masks.len(), seq_len);

    let mut total_loss = 0.0_f32;
    let mut count = 0usize;

    for i in 0..seq_len {
        if !mask_indicators[i] {
            continue;
        }

        let offset = i * vocab_size;
        let tgt = targets[i];

        // Find max for numerical stability
        let mut max_logit = f32::NEG_INFINITY;
        for v in 0..vocab_size {
            unsafe {
                let val = *logits.get_unchecked(offset + v);
                if val > max_logit {
                    max_logit = val;
                }
            }
        }

        // Compute log-softmax via log-sum-exp
        let mut sum_exp = 0.0_f32;
        for v in 0..vocab_size {
            unsafe {
                sum_exp += (*logits.get_unchecked(offset + v) - max_logit).exp();
            }
        }
        let log_sum_exp = sum_exp.ln();

        // CE for target token: -(logit_tgt - max - log_sum_exp)
        let tgt_logit = unsafe { *logits.get_unchecked(offset + tgt) };
        let ce_loss = -(tgt_logit - max_logit - log_sum_exp);

        // Importance weighting: divide by p_mask (higher mask ratio -> lower weight)
        let p_mask = p_masks[i].max(1e-6); // avoid division by zero
        total_loss += ce_loss / p_mask;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        total_loss / count as f32
    }
}

/// Block-causal attention boundary for position `pos`.
///
/// Implements the 3-rule D2F attention mask:
/// - Rule 1: Prompt positions attend to all prompt positions
/// - Rule 2: Within-block bidirectional
/// - Rule 3: Between-block causal (block b sees blocks 0..=b)
///
/// Returns `t_n` -- the number of positions this query attends to.
pub fn block_causal_t_n(pos: usize, prompt_len: usize, block_size: usize, seq_len: usize) -> usize {
    if pos < prompt_len {
        // Rule 1: prompt positions attend to all prompt positions
        prompt_len
    } else {
        // Rule 2+3: within-block bidirectional, across-block causal
        let block_idx = (pos - prompt_len) / block_size;
        let block_end = prompt_len + (block_idx + 1) * block_size;
        block_end.min(seq_len)
    }
}
