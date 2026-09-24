//! Row-logit-floor attention — the opt-in consumer of katgpt-core
//! `row_logit_floor` (riir-infer Issue 011 T1; katgpt-rs Issue 882 P2).
//!
//! Same three passes as [`super::attention::attention_heads_parallel`]
//! (scores → softmax → value sum), with pass 2 replaced by
//! [`floored_coded_exp_inplace`]: every non-sink logit is floored to
//! `m_r − w`, coded in `b` bits over `[m_r − w, m_r]`, and exponentiated
//! through a `2^b`-entry table. Sinks (`pos < n_sink`) keep their exact
//! `f32` logits; the causal row here has no `−∞` entries.
//!
//! The width is the row-independent guarantee `w = ln(n_ctx / ε)`
//! ([`min_width_for_tv`]), recomputed per row from its own context length,
//! so the floor term of the envelope never exceeds `ε` on any row.
//!
//! Every head returns its per-row [`RowLogitFloorStats`] so a caller can
//! report the closed-form envelope beside the model-level quality number.

use katgpt_core::row_logit_floor::{LogitCodec, floored_coded_exp_inplace, min_width_for_tv};
use rayon::prelude::*;

use super::attention::{PARALLEL_HEADS_MIN_SEQ, attention_heads_parallel};

/// What the floored attention path does to each score row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowLogitFloorPolicy {
    /// Leading positions exempt from the floor and the code (the
    /// `kv_sink_window` convention; `0` disables the exemption — the
    /// Issue 011 T4 negative arm).
    pub n_sink: usize,
    /// Code bits, `2..=8`.
    pub bits: u8,
    /// Floor TV budget `ε` per row (the width is `ln(n_ctx / ε)`).
    pub tv: f32,
    /// `None`: per-row width `ln(n_ctx / ε)` from the row's own context.
    /// `Some(n)`: one FIXED width `ln(n / ε)` for every row — the budget
    /// then holds for any row with `n_ctx ≤ n`, and a short row pays the
    /// long-context code step (the Issue 011 T3 proxy: the 64K width on
    /// real rows).
    pub width_ctx: Option<usize>,
}

/// Per-row envelope accounting, summed over rows (heads × layers × tokens).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RowLogitFloorStats {
    /// Rows that took the floored path.
    pub rows: u64,
    /// Non-sink keys over those rows.
    pub ctx_keys: u64,
    /// Non-sink keys raised to the floor.
    pub floored: u64,
    /// Σ floor-TV bound (`A / (1 + A)`).
    pub floor_tv_sum: f64,
    /// Σ code relative bound (`e^{2h} − 1`).
    pub code_rel_sum: f64,
    /// Σ row width (nats).
    pub width_sum: f64,
}

impl RowLogitFloorStats {
    /// Accumulate another tally.
    #[inline]
    pub fn add(&mut self, o: &Self) {
        self.rows += o.rows;
        self.ctx_keys += o.ctx_keys;
        self.floored += o.floored;
        self.floor_tv_sum += o.floor_tv_sum;
        self.code_rel_sum += o.code_rel_sum;
        self.width_sum += o.width_sum;
    }

    /// Mean per-row total-TV bound (floor + code/2, the triangle bound).
    pub fn mean_envelope_tv(&self) -> f64 {
        match self.rows {
            0 => 0.0,
            r => (self.floor_tv_sum + 0.5 * self.code_rel_sum) / r as f64,
        }
    }
}

/// Multi-head attention with the row-logit floor on every row.
///
/// Same contract as [`attention_heads_parallel`] (see its `# Safety`), plus
/// `policy`. Rows with no context key (`t_n <= n_sink`) take the plain path
/// and contribute no stats.
///
/// # Safety
///
/// Identical to [`attention_heads_parallel`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_heads_floored(
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
    policy: RowLogitFloorPolicy,
) -> RowLogitFloorStats {
    if t_n <= policy.n_sink {
        unsafe {
            attention_heads_parallel(
                q,
                key_cache,
                value_cache,
                attn_out,
                head_scores,
                n_head,
                n_kv_head,
                kv_dim,
                head_dim,
                t_n,
                scale,
                softcap,
                block_size,
            );
        }
        return RowLogitFloorStats::default();
    }
    let row = FloorRow {
        codec: LogitCodec::new(policy.bits),
        n_sink: policy.n_sink,
        width: min_width_for_tv(policy.width_ctx.unwrap_or(t_n - policy.n_sink), policy.tv),
        kv_dim,
        hd: head_dim,
        t_n,
        scale,
        softcap,
    };
    let q_dim = n_head * head_dim;
    let group = |h: usize| (h * n_kv_head / n_head) * head_dim;
    match t_n < PARALLEL_HEADS_MIN_SEQ || n_head <= 1 {
        true => {
            let scores = &mut head_scores[..block_size];
            let mut acc = RowLogitFloorStats::default();
            for (h, out_h) in attn_out[..q_dim].chunks_mut(head_dim).enumerate() {
                let st = unsafe {
                    row.head(
                        &q[h * head_dim..],
                        key_cache,
                        value_cache,
                        out_h,
                        scores,
                        group(h),
                    )
                };
                acc.add(&st);
            }
            acc
        }
        false => attn_out[..q_dim]
            .par_chunks_mut(head_dim)
            .zip(head_scores.par_chunks_mut(block_size))
            .enumerate()
            .map(|(h, (out_h, scores_h))| unsafe {
                row.head(
                    &q[h * head_dim..],
                    key_cache,
                    value_cache,
                    out_h,
                    scores_h,
                    group(h),
                )
            })
            .reduce(RowLogitFloorStats::default, |mut a, b| {
                a.add(&b);
                a
            }),
    }
}

/// The per-row constants shared by every head of one call.
#[derive(Clone, Copy)]
struct FloorRow {
    codec: LogitCodec,
    n_sink: usize,
    width: f32,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    softcap: f32,
}

impl FloorRow {
    /// One head: scores (softcapped when `softcap > 0`) → floored coded
    /// softmax → `out_head += Σ p_t · v_t`.
    ///
    /// # Safety
    ///
    /// `scores.len() >= t_n`, `out_head.len() >= hd`, and every
    /// `t * kv_dim + kv_group_offset + hd` (`t < t_n`) inside both caches.
    #[inline(always)]
    unsafe fn head(
        &self,
        q_head: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        out_head: &mut [f32],
        scores: &mut [f32],
        kv_group_offset: usize,
    ) -> RowLogitFloorStats {
        let (hd, kv_dim, t_n) = (self.hd, self.kv_dim, self.t_n);
        let capped = self.softcap > 0.0;
        let scale_over_softcap = match capped {
            true => self.scale / self.softcap,
            false => 0.0,
        };
        for t in 0..t_n {
            let k_off = t * kv_dim + kv_group_offset;
            let dot = crate::simd::simd_dot_f32(&q_head[..hd], &key_cache[k_off..k_off + hd], hd);
            let score = match capped {
                true => self.softcap * crate::simd::fast_tanh(dot * scale_over_softcap),
                false => dot * self.scale,
            };
            unsafe {
                *scores.get_unchecked_mut(t) = score;
            }
        }

        let mut lut = [0.0f32; 256];
        let (rf, z) = floored_coded_exp_inplace(
            &mut scores[..t_n],
            self.n_sink,
            self.width,
            &self.codec,
            &mut lut,
        );

        let inv_sum = 1.0 / z;
        for t in 0..t_n {
            let w = unsafe { *scores.get_unchecked(t) * inv_sum };
            let v_off = t * kv_dim + kv_group_offset;
            for d in 0..hd {
                unsafe {
                    *out_head.get_unchecked_mut(d) += w * *value_cache.get_unchecked(v_off + d);
                }
            }
        }

        let env = self.codec.envelope(self.width, rf.n_floored);
        RowLogitFloorStats {
            rows: 1,
            ctx_keys: (t_n - self.n_sink.min(t_n)) as u64,
            floored: rf.n_floored as u64,
            floor_tv_sum: f64::from(env.floor_tv),
            code_rel_sum: f64::from(env.code_rel),
            width_sum: f64::from(self.width),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny GQA fixture: the floored path at 8 bits must track the plain
    /// path within the closed-form envelope (attention output is a convex
    /// combination of V rows, so `|Δout| ≤ 2·TV·max|v|`).
    #[test]
    fn floored_tracks_plain_within_envelope() {
        let (n_head, n_kv, hd, t_n, block) = (4usize, 2usize, 16usize, 96usize, 128usize);
        let kv_dim = n_kv * hd;
        let lcg = |i: usize| (((i * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0) * 2.0 - 1.0;
        let q: Vec<f32> = (0..n_head * hd).map(|i| 3.0 * lcg(i + 7)).collect();
        let k: Vec<f32> = (0..t_n * kv_dim).map(|i| lcg(i + 101)).collect();
        let v: Vec<f32> = (0..t_n * kv_dim).map(|i| lcg(i + 9_001)).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        for softcap in [0.0f32, 50.0] {
            let mut plain = vec![0.0f32; n_head * hd];
            let mut s1 = vec![0.0f32; n_head * block];
            unsafe {
                attention_heads_parallel(
                    &q, &k, &v, &mut plain, &mut s1, n_head, n_kv, kv_dim, hd, t_n, scale, softcap,
                    block,
                );
            }
            let policy = RowLogitFloorPolicy {
                n_sink: 4,
                bits: 8,
                tv: 1e-3,
                width_ctx: None,
            };
            let mut floored = vec![0.0f32; n_head * hd];
            let mut s2 = vec![0.0f32; n_head * block];
            let st = unsafe {
                attention_heads_floored(
                    &q,
                    &k,
                    &v,
                    &mut floored,
                    &mut s2,
                    n_head,
                    n_kv,
                    kv_dim,
                    hd,
                    t_n,
                    scale,
                    softcap,
                    block,
                    policy,
                )
            };
            assert_eq!(st.rows, n_head as u64);
            assert_eq!(st.ctx_keys, (n_head * (t_n - 4)) as u64);
            let bound = 2.0 * (st.floor_tv_sum + 0.5 * st.code_rel_sum) / st.rows as f64;
            let max_dev = plain
                .iter()
                .zip(&floored)
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0.0, f64::max);
            assert!(
                max_dev <= bound + 1e-6,
                "softcap {softcap}: dev {max_dev} > {bound}"
            );
        }
    }

    #[test]
    fn no_context_rows_take_the_plain_path() {
        let (hd, block) = (8usize, 8usize);
        let q = vec![0.5f32; hd];
        let k = vec![0.25f32; 3 * hd];
        let v: Vec<f32> = (0..3 * hd).map(|i| i as f32).collect();
        let mut a = vec![0.0f32; hd];
        let mut b = vec![0.0f32; hd];
        let mut s = vec![0.0f32; block];
        let policy = RowLogitFloorPolicy {
            n_sink: 4,
            bits: 6,
            tv: 1e-3,
            width_ctx: None,
        };
        unsafe {
            attention_heads_parallel(&q, &k, &v, &mut a, &mut s, 1, 1, hd, hd, 3, 1.0, 0.0, block);
            let st = attention_heads_floored(
                &q, &k, &v, &mut b, &mut s, 1, 1, hd, hd, 3, 1.0, 0.0, block, policy,
            );
            assert_eq!(st, RowLogitFloorStats::default());
        }
        assert_eq!(a, b);
    }
}
