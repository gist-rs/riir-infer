//! One decode row of multi-head attention, with the two opt-in
//! instrument/policy hooks every CPU forward shares:
//!
//! - katgpt-rs Issue 882 P4 `attention_to_answer`: the m_Y probe reads this
//!   row's distribution (the floored one when a floor is set), armed rows only;
//! - riir-infer Issue 011 T1 `row_logit_floor`: with `ctx.logit_floor` set the
//!   softmax is the sink-exempt floored + coded one, `None` is the plain path.
//!
//! One copy so the gemma-2 f16 and LLaMA-family forwards cannot drift apart
//! on either hook. With neither feature on this is exactly
//! [`attention_heads_parallel`].

use rayon::prelude::*;

use super::{ForwardContext, attention_heads_parallel, attention::PARALLEL_HEADS_MIN_SEQ};

/// The per-row attention geometry (constant across heads).
#[derive(Clone, Copy)]
pub(crate) struct AttnShape {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub t_n: usize,
    pub scale: f32,
    /// `> 0` ⇒ tanh softcap (gemma-2); `0` ⇒ none (LLaMA).
    pub softcap: f32,
    pub block_size: usize,
}

/// Zero `ctx.attn_out[..n_head·head_dim]` and accumulate this row's attention
/// into it.
///
/// # Safety
///
/// As [`attention_heads_parallel`]: every `t·kv_dim + group + head_dim`
/// (`t < t_n`) inside both caches, `ctx.head_scores` sized for `block_size`
/// per head.
#[inline]
// `layer_idx` is read by the m_Y probe only.
#[cfg_attr(not(feature = "attention_to_answer"), allow(unused_variables))]
pub(crate) unsafe fn attend_row(
    ctx: &mut ForwardContext,
    key: &[f32],
    value: &[f32],
    layer_idx: usize,
    s: AttnShape,
) {
    #[cfg(feature = "attention_to_answer")]
    if let Some(probe) = ctx.attn_probe.as_mut().filter(|p| p.armed) {
        #[cfg(feature = "row_logit_floor")]
        let floor = ctx.logit_floor;
        #[cfg(not(feature = "row_logit_floor"))]
        let floor = None;
        probe.observe_layer(
            layer_idx,
            &ctx.q,
            key,
            super::attention_probe::ProbeShape {
                n_head: s.n_head,
                n_kv_head: s.n_kv_head,
                kv_dim: s.kv_dim,
                head_dim: s.head_dim,
                t_n: s.t_n,
                scale: s.scale,
                softcap: s.softcap,
            },
            floor,
        );
    }

    ctx.attn_out[..s.n_head * s.head_dim].fill(0.0);

    #[cfg(feature = "row_logit_floor")]
    if let Some(policy) = ctx.logit_floor {
        let st = unsafe {
            super::attention_floor::attention_heads_floored(
                &ctx.q,
                key,
                value,
                &mut ctx.attn_out,
                &mut ctx.head_scores,
                s.n_head,
                s.n_kv_head,
                s.kv_dim,
                s.head_dim,
                s.t_n,
                s.scale,
                s.softcap,
                s.block_size,
                policy,
            )
        };
        ctx.logit_floor_stats.add(&st);
        return;
    }

    unsafe {
        attention_heads_parallel(
            &ctx.q,
            key,
            value,
            &mut ctx.attn_out,
            &mut ctx.head_scores,
            s.n_head,
            s.n_kv_head,
            s.kv_dim,
            s.head_dim,
            s.t_n,
            s.scale,
            s.softcap,
            s.block_size,
        );
    }
}

// ── Issue 013 T4 — the fused deferred-restore read path ─────────────────
//
// The P3 reconstruction lane's SECOND lever (the first was the eager scratch
// serving of `V = G(−θp)·K̂ + λ·E_l[s]`): by linearity the V aggregation is
// `Σ_t w_t·v̂_t + λ·Σ_s W_s·E_l[s]` with `v̂_t = G(−θt)·K̂_t`, so the rotation
// fuses into THIS kernel's pass 3 (the K row is already hot there), and the
// table part regroups into per-(head, row) weight sums `W_s = Σ_{t: s_t=s} w_t`
// applied ONCE per distinct row in an epilogue — the row traffic drops from
// per-position to per-distinct-row and the per-row λE add disappears.
//
// Numerics contract (pre-registered, riir-infer Issue 013 T4):
// - λ = 0 or an all-miss context: the epilogue is skipped and pass 3 is the
//   plain pass 3 over rows that equal `v̂_t` bitwise — BITWISE identical to
//   the eager path (whose λ=0 scratch rows are `v̂_t` by
//   `add_scaled_row_inplace`'s λ==0 early return).
// - λ > 0 with tracked rows: the regrouping reassociates the same product
//   set — error ≤ 2·t_n·ε_f32·λ·max|E| (worst-case pairwise), measured far
//   below the rotation-rounding class the G1 gates already pin.
// - pos 0 copies the raw head slice (the forward's pos-0 identity — the
//   −0.0 spelling must survive, matching `HalfSplitRopeInverse`'s early
//   return).

/// The shared (read-only) fused-reconstruction view — plain slices, no
/// katgpt-core type, so this module compiles ungated. `pub` because it names
/// the [`crate::transformer::ValueStoreHook::fused_recon`] return; the
/// surface is crate-internal vocabulary (the forward + the T4 lane are the
/// only consumers).
#[derive(Clone, Copy)]
pub struct ReconShared<'a> {
    /// Per-position `(sin, cos)` table, `[pos][i][2]` — built with the exact
    /// `(pos as f32 * freq[i]).sin_cos()` expression the action computes, so
    /// the fused rotation is bitwise the on-the-fly one.
    pub cs: &'a [f32],
    /// `head_dim / 2` (the freq-table width).
    pub half: usize,
    /// The cached post-RoPE keys, rows `0..t_n` of `t_n × kv_dim`.
    pub k_cache: &'a [f32],
    pub kv_dim: usize,
    /// Per-position tracked-row index, `u32::MAX` = miss (len ≥ t_n).
    pub row_of_pos: &'a [u32],
    pub lambda: f32,
    /// THIS layer's tracked rows as one slab (`rows × width`).
    pub e_rows: &'a [f32],
    pub table_rows: usize,
    pub table_width: usize,
}

/// The mutable half of the fused view — per-head weight sums and the
/// touched-row lists the epilogue consumes, owned by the serving hook and
/// cleared by the epilogue (the all-zero contract between layer steps).
/// `pub` for the trait-return-type privacy rule (see [`ReconShared`]).
pub struct FusedRecon<'a> {
    pub shared: ReconShared<'a>,
    /// `n_head × table_rows`, all zero between layer steps.
    pub weights: &'a mut [f32],
    /// Per-head touched-row lists (cleared between layer steps).
    pub used: &'a mut [Vec<u32>],
}

/// One head of the fused softcap attention: passes 1–2 are
/// [`attention_head_on_slices`]'s verbatim (the scores come from the same K
/// rows either way); pass 3 reconstructs `v̂_t` from the hot K row instead of
/// reading a served V row, and scatters `w` into `w_row[row]` when the row
/// is tracked and λ ≠ 0. `used` collects first-touches so the epilogue (and
/// the clear) stay sparse.
///
/// # Safety
/// Same indexing contract as [`attention_head_on_slices`]: every
/// `t·kv_dim + kv_group_offset + head_dim` (`t < t_n`) inside `shared.k_cache`,
/// `cs` holding `t_n · half · 2` floats, `row_of_pos` len ≥ t_n,
/// `scores_buf` len ≥ t_n, `w_row` len ≥ table_rows with `table_rows` the
/// `shared.e_rows` row count.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn attention_head_recon_slices(
    q_head: &[f32],
    shared: &ReconShared<'_>,
    attn_out_head: &mut [f32],
    scores_buf: &mut [f32],
    kv_group_offset: usize,
    head_dim: usize,
    t_n: usize,
    scale: f32,
    softcap: f32,
    w_row: &mut [f32],
    used: &mut Vec<u32>,
) {
    let hd = head_dim;
    let half = shared.half;
    debug_assert_eq!(half * 2, hd, "fused recon: head_dim must be 2·half");

    // Pass 1: identical to attention_head_on_slices (the K read either way).
    let scale_over_softcap = scale / softcap;
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * shared.kv_dim + kv_group_offset;
        let dot = crate::simd::simd_dot_f32(&q_head[..hd], &shared.k_cache[k_off..k_off + hd], hd);
        let score = softcap * crate::simd::fast_tanh(dot * scale_over_softcap);
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    // Pass 2: identical.
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: v̂_t = G(−θt)·K̂_t head slice, accumulated; W scatter deferred.
    // The expression is `HalfSplitRopeInverse::apply_inverse_at`'s exact
    // form (multiplication order and negation spelling included) so the λ=0
    // path is bitwise the eager path.
    let inv_sum = 1.0 / sum;
    let scatter = shared.lambda != 0.0 && shared.table_rows > 0;
    for t in 0..t_n {
        let w = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
        let k_off = t * shared.kv_dim + kv_group_offset;
        if scatter {
            let row = unsafe { *shared.row_of_pos.get_unchecked(t) };
            if row != u32::MAX {
                let r = row as usize;
                if w_row[r] == 0.0 {
                    used.push(row);
                }
                w_row[r] += w;
            }
        }
        if t == 0 {
            // The forward's pos-0 identity — bitwise copy (the −0.0 class).
            for d in 0..hd {
                unsafe {
                    *attn_out_head.get_unchecked_mut(d) +=
                        w * *shared.k_cache.get_unchecked(k_off + d);
                }
            }
            continue;
        }
        let cs_off = t * half * 2;
        for i in 0..half {
            let sin_a = unsafe { *shared.cs.get_unchecked(cs_off + 2 * i) };
            let cos_a = unsafe { *shared.cs.get_unchecked(cs_off + 2 * i + 1) };
            let (x0, x1) = unsafe { (
                *shared.k_cache.get_unchecked(k_off + i),
                *shared.k_cache.get_unchecked(k_off + half + i),
            ) };
            let v0 = x0 * cos_a + x1 * sin_a;
            let v1 = -x0 * sin_a + x1 * cos_a;
            unsafe {
                *attn_out_head.get_unchecked_mut(i) += w * v0;
                *attn_out_head.get_unchecked_mut(half + i) += w * v1;
            }
        }
    }
}

/// The fused heads: parallel above [`PARALLEL_HEADS_MIN_SEQ`], sequential
/// below — the [`attention_heads_parallel`] split, recon variant.
///
/// # Safety
/// [`attention_heads_parallel`]'s contract, with the recon slices in place
/// of the value cache: `weights.len() == n_head · table_rows`, `used.len()
/// == n_head`, and every per-head region inside bounds.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attention_heads_recon(
    q: &[f32],
    fr: &mut FusedRecon<'_>,
    attn_out: &mut [f32],
    head_scores: &mut [f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    t_n: usize,
    scale: f32,
    softcap: f32,
    block_size: usize,
) {
    debug_assert!(softcap > 0.0, "the fused recon lane is softcap-shaped");
    let table_rows = fr.shared.table_rows;
    debug_assert_eq!(fr.weights.len(), n_head * table_rows);
    debug_assert_eq!(fr.used.len(), n_head);
    if t_n < PARALLEL_HEADS_MIN_SEQ || n_head <= 1 {
        // Sequential fallback: reuse head_scores[0..block_size] for all heads
        // (the plain split's own shape; disjoint per-head field borrows).
        for h in 0..n_head {
            let kv_group_offset = (h * n_kv_head / n_head) * head_dim;
            let w_row = &mut fr.weights[h * table_rows..(h + 1) * table_rows];
            let used_h = &mut fr.used[h];
            // SAFETY: per-head regions are disjoint; scores[0..block_size]
            // is exclusive to this head in the sequential regime.
            unsafe {
                attention_head_recon_slices(
                    &q[h * head_dim..],
                    &fr.shared,
                    &mut attn_out[h * head_dim..h * head_dim + head_dim],
                    &mut head_scores[..block_size],
                    kv_group_offset,
                    head_dim,
                    t_n,
                    scale,
                    softcap,
                    w_row,
                    used_h,
                );
            }
        }
    } else {
        // Parallel: the plain split's par_chunks_mut shape, with the
        // per-head weight row + used list zipped in. SAFETY: disjoint
        // per-head mutable regions; `shared` is read-only.
        let FusedRecon {
            shared,
            weights,
            used,
        } = fr;
        let shared = &*shared;
        attn_out[..n_head * head_dim]
            .par_chunks_mut(head_dim)
            .zip(head_scores.par_chunks_mut(block_size))
            .zip(weights.par_chunks_mut(table_rows))
            .zip(used.par_iter_mut())
            .enumerate()
            .for_each(|(h, (((attn_h, scores_h), w_row), used_h))| {
                let kv_group_offset = (h * n_kv_head / n_head) * head_dim;
                unsafe {
                    attention_head_recon_slices(
                        &q[h * head_dim..],
                        shared,
                        attn_h,
                        scores_h,
                        kv_group_offset,
                        head_dim,
                        t_n,
                        scale,
                        softcap,
                        w_row,
                        used_h,
                    );
                }
            });
    }
}

/// The deferred-restore epilogue: `attn_out += λ · Σ_s W_{h,s} · E_l[s]`
/// over each head's touched rows, then clears `weights`/`used` (the
/// all-zero contract between layer steps). λ = 0 never reaches here (the
/// kernel skips the scatter); a zero weight-sum row is skipped and its
/// entry still cleared.
pub(crate) fn fused_epilogue(
    attn_out: &mut [f32],
    shared: &ReconShared<'_>,
    weights: &mut [f32],
    used: &mut [Vec<u32>],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) {
    if shared.lambda == 0.0 || shared.table_rows == 0 {
        // Still clear: a caller may have scattered into a zero-table view.
        for u in used.iter_mut() {
            u.clear();
        }
        return;
    }
    let table_rows = shared.table_rows;
    for h in 0..n_head {
        let group = (h * n_kv_head / n_head) * head_dim;
        let w_row = &mut weights[h * table_rows..(h + 1) * table_rows];
        for &row in used[h].iter() {
            let r = row as usize;
            let ws = w_row[r];
            if ws != 0.0 {
                let e_off = r * shared.table_width + group;
                let lam_ws = shared.lambda * ws;
                for d in 0..head_dim {
                    attn_out[h * head_dim + d] += lam_ws * shared.e_rows[e_off + d];
                }
            }
            w_row[r] = 0.0;
        }
        used[h].clear();
    }
}

/// The fused deferred-restore variant of [`attend_row`]: the hook-served
/// reconstruction rides THIS kernel (pass 3 fuses the rotation; the table
/// part lands in the epilogue). The call site guarantees `softcap > 0` and
/// no `row_logit_floor` policy — both fall back to the eager paths. The
/// m_Y probe (when the feature is on and armed) observes the same q/key
/// distribution it always does — it is V-path independent.
///
/// # Safety
/// [`attend_row`]'s contract plus the recon bounds: `fr.shared.row_of_pos`
/// len ≥ `s.t_n`, `cs` holding `s.t_n · half · 2`, and `head_dim == 2·half`.
pub(crate) unsafe fn attend_row_fused(
    ctx: &mut ForwardContext,
    mut fr: FusedRecon<'_>,
    // `layer_idx` is read by the m_Y probe only.
    #[cfg_attr(not(feature = "attention_to_answer"), allow(unused_variables))]
    layer_idx: usize,
    s: AttnShape,
) {
    #[cfg(feature = "attention_to_answer")]
    if let Some(probe) = ctx.attn_probe.as_mut().filter(|p| p.armed) {
        #[cfg(feature = "row_logit_floor")]
        let floor = ctx.logit_floor;
        #[cfg(not(feature = "row_logit_floor"))]
        let floor = None;
        debug_assert!(floor.is_none(), "the fused lane requires the floor off");
        probe.observe_layer(
            layer_idx,
            &ctx.q,
            fr.shared.k_cache,
            super::attention_probe::ProbeShape {
                n_head: s.n_head,
                n_kv_head: s.n_kv_head,
                kv_dim: s.kv_dim,
                head_dim: s.head_dim,
                t_n: s.t_n,
                scale: s.scale,
                softcap: s.softcap,
            },
            floor,
        );
    }

    #[cfg(feature = "row_logit_floor")]
    debug_assert!(
        ctx.logit_floor.is_none(),
        "the fused lane requires the floor off (the call site falls back)"
    );

    ctx.attn_out[..s.n_head * s.head_dim].fill(0.0);

    // SAFETY: the caller upholds attend_row's bounds + the recon slice
    // bounds documented on attention_heads_recon.
    unsafe {
        attention_heads_recon(
            &ctx.q,
            &mut fr,
            &mut ctx.attn_out,
            &mut ctx.head_scores,
            s.n_head,
            s.n_kv_head,
            s.head_dim,
            s.t_n,
            s.scale,
            s.softcap,
            s.block_size,
        );
    }
    // Disjoint fields: the shared view is read-only, the weights/used halves
    // are consumed (and cleared) by the epilogue.
    let FusedRecon {
        shared,
        weights,
        used,
    } = &mut fr;
    fused_epilogue(
        &mut ctx.attn_out,
        shared,
        weights,
        used,
        s.n_head,
        s.n_kv_head,
        s.head_dim,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;

    /// GQA 4/2, `head_dim` 8, room for the parallel-heads split (t ≥ 512).
    fn setup(t_n: usize) -> (Config, ForwardContext, Vec<f32>, Vec<f32>, AttnShape) {
        let mut c = Config::micro();
        (c.n_head, c.n_kv_head, c.head_dim, c.n_embd, c.block_size) = (4, 2, 8, 32, 640);
        let mut ctx = ForwardContext::new(&c);
        let kvd = c.n_kv_head * c.head_dim;
        let mut s = 0x2545_F491u32;
        let mut r = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        for q in &mut ctx.q[..c.n_head * c.head_dim] {
            *q = 3.0 * r();
        }
        let key: Vec<f32> = (0..t_n * kvd).map(|_| r()).collect();
        let value: Vec<f32> = (0..t_n * kvd).map(|_| r()).collect();
        let shape = AttnShape {
            n_head: c.n_head,
            n_kv_head: c.n_kv_head,
            kv_dim: kvd,
            head_dim: c.head_dim,
            t_n,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            softcap: 0.0,
            block_size: c.block_size,
        };
        (c, ctx, key, value, shape)
    }

    /// The extraction is a pure refactor: with no policy set, `attend_row`
    /// is BIT-identical to [`attention_heads_parallel`] (both head splits,
    /// with and without softcap) — the gemma-2 f16 and LLaMA forwards route
    /// through it.
    #[test]
    fn no_policy_is_bit_identical_to_the_plain_kernel() {
        for (t_n, softcap) in [(37, 0.0), (37, 50.0), (600, 0.0), (600, 50.0)] {
            let (c, mut ctx, key, value, mut s) = setup(t_n);
            s.softcap = softcap;
            ctx.attn_out.fill(7.0); // must be zeroed by the helper
            unsafe { attend_row(&mut ctx, &key, &value, 0, s) };
            let got = ctx.attn_out[..c.n_head * c.head_dim].to_vec();

            let mut want = vec![0.0f32; c.n_head * c.head_dim];
            let mut scores = vec![0.0f32; c.n_head * c.block_size];
            unsafe {
                attention_heads_parallel(
                    &ctx.q,
                    &key,
                    &value,
                    &mut want,
                    &mut scores,
                    s.n_head,
                    s.n_kv_head,
                    s.kv_dim,
                    s.head_dim,
                    t_n,
                    s.scale,
                    softcap,
                    s.block_size,
                );
            }
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "t_n={t_n} softcap={softcap}");
        }
    }

    /// With a policy set, the helper IS the floored dispatcher (bit-identical
    /// output) and accumulates its stats into the context.
    #[cfg(feature = "row_logit_floor")]
    #[test]
    fn policy_routes_to_the_floored_kernel() {
        use crate::transformer::attention_floor::{RowLogitFloorPolicy, attention_heads_floored};
        let policy = RowLogitFloorPolicy {
            n_sink: 4,
            bits: 6,
            tv: 1e-3,
            width_ctx: None,
        };
        for t_n in [37, 600] {
            let (c, mut ctx, key, value, s) = setup(t_n);
            ctx.logit_floor = Some(policy);
            unsafe { attend_row(&mut ctx, &key, &value, 0, s) };
            let got = ctx.attn_out[..c.n_head * c.head_dim].to_vec();
            assert_eq!(ctx.logit_floor_stats.rows, c.n_head as u64, "t_n={t_n}");

            let mut want = vec![0.0f32; c.n_head * c.head_dim];
            let mut scores = vec![0.0f32; c.n_head * c.block_size];
            unsafe {
                attention_heads_floored(
                    &ctx.q,
                    &key,
                    &value,
                    &mut want,
                    &mut scores,
                    s.n_head,
                    s.n_kv_head,
                    s.kv_dim,
                    s.head_dim,
                    t_n,
                    s.scale,
                    s.softcap,
                    s.block_size,
                    policy,
                );
            }
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "t_n={t_n}");
        }
    }
}

// ── Issue 013 T4 — the fused deferred-restore kernel gates ──────────────
//
// Table-agnostic by construction (ReconShared carries plain slices), so
// these run UNGATED — the fused path's numerics contract is pinned here, not
// behind the vk_p3_tg lane.
#[cfg(test)]
mod fused_recon_tests {
    use super::*;
    use crate::rope::RopeFreqTable;
    use crate::types::Config;

    const HD: usize = 8;
    const N_HEAD: usize = 4;
    const N_KV: usize = 2;
    const KVD: usize = N_KV * HD;
    const HALF: usize = HD / 2;
    const TABLE_ROWS: usize = 3;

    struct Fixture {
        ctx: ForwardContext,
        shape: AttnShape,
        q: Vec<f32>,
        k: Vec<f32>,
        cs: Vec<f32>,
        e_rows: Vec<f32>,
        row_of_pos: Vec<u32>,
    }

    /// GQA 4/2 at hd 8 (the plain tests' shape), softcap-shaped (the fused
    /// lane's precondition), random q/K with a planted −0.0 at pos 0, the cs
    /// table built with the exact sin_cos expression, and a 3-row tracked
    /// pattern (some positions miss).
    fn fixture(t_n: usize, block_size: usize) -> Fixture {
        let mut c = Config::micro();
        (c.n_head, c.n_kv_head, c.head_dim, c.n_embd, c.block_size) =
            (N_HEAD, N_KV, HD, 32, block_size);
        let ctx = ForwardContext::new(&c);
        let mut s = 0x2545_F491u32;
        let mut r = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let mut q: Vec<f32> = (0..N_HEAD * HD).map(|_| 3.0 * r()).collect();
        let mut k: Vec<f32> = (0..t_n * KVD).map(|_| r()).collect();
        k[2] = -0.0; // the pos-0 −0.0 class, planted
        q[1] = -0.0;
        let freq = RopeFreqTable::new(10_000.0, HD);
        let freq = freq.as_slice();
        let mut cs = Vec::with_capacity(t_n * HALF * 2);
        for p in 0..t_n {
            let n = p as f32;
            for &f in freq {
                let (sin_a, cos_a) = (n * f).sin_cos();
                cs.push(sin_a);
                cs.push(cos_a);
            }
        }
        let e_rows: Vec<f32> = (0..TABLE_ROWS * KVD).map(|i| (i as f32) * 0.125 - 0.75).collect();
        // Deterministic tracked pattern: 0, miss, 1, miss, miss, 2, repeat.
        let row_of_pos: Vec<u32> = (0..t_n)
            .map(|t| match t % 6 {
                0 => 0,
                2 => 1,
                5 => 2,
                _ => u32::MAX,
            })
            .collect();
        let shape = AttnShape {
            n_head: N_HEAD,
            n_kv_head: N_KV,
            kv_dim: KVD,
            head_dim: HD,
            t_n,
            scale: 1.0 / (HD as f32).sqrt(),
            softcap: 50.0,
            block_size,
        };
        Fixture {
            ctx,
            shape,
            q,
            k,
            cs,
            e_rows,
            row_of_pos,
        }
    }

    /// The fused path reads `ctx.q` (the forward's real source) — load the
    /// fixture's q there so both arms see the same queries.
    fn prime(f: &mut Fixture) {
        f.ctx.q[..f.q.len()].copy_from_slice(&f.q);
    }

    /// The EAGER reference: reconstruct rows exactly as the T3 scratch does
    /// (pos-0 bitwise copy; the inverse-rotation expression verbatim; the
    /// λE add per tracked row), then the plain kernel over them.
    fn reference_scratch(f: &Fixture, lambda: f32) -> Vec<f32> {
        let mut scratch = vec![0.0f32; f.k.len()];
        for t in 0..f.shape.t_n {
            let k_off = t * KVD;
            let out = &mut scratch[k_off..k_off + KVD];
            if t == 0 {
                out.copy_from_slice(&f.k[k_off..k_off + KVD]);
            } else {
                let cs_off = t * HALF * 2;
                for h in 0..N_KV {
                    let base = k_off + h * HD;
                    let ob = h * HD;
                    for i in 0..HALF {
                        let sin_a = f.cs[cs_off + 2 * i];
                        let cos_a = f.cs[cs_off + 2 * i + 1];
                        let x0 = f.k[base + i];
                        let x1 = f.k[base + HALF + i];
                        out[ob + i] = x0 * cos_a + x1 * sin_a;
                        out[ob + HALF + i] = -x0 * sin_a + x1 * cos_a;
                    }
                }
            }
            if lambda != 0.0 {
                let row = f.row_of_pos[t];
                if row != u32::MAX {
                    let r = row as usize;
                    let e = &f.e_rows[r * KVD..(r + 1) * KVD];
                    for (o, &e_v) in out.iter_mut().zip(e) {
                        *o += lambda * e_v;
                    }
                }
            }
        }
        scratch
    }

    fn run_plain(f: &mut Fixture, lambda: f32) -> Vec<f32> {
        let scratch = reference_scratch(f, lambda);
        f.ctx.attn_out[..N_HEAD * HD].fill(0.0);
        unsafe {
            attention_heads_parallel(
                &f.q,
                &f.k,
                &scratch,
                &mut f.ctx.attn_out,
                &mut f.ctx.head_scores,
                N_HEAD,
                N_KV,
                KVD,
                HD,
                f.shape.t_n,
                f.shape.scale,
                f.shape.softcap,
                f.shape.block_size,
            );
        }
        f.ctx.attn_out[..N_HEAD * HD].to_vec()
    }

    fn run_fused(f: &mut Fixture, lambda: f32) -> Vec<f32> {
        let shared = ReconShared {
            cs: &f.cs,
            half: HALF,
            k_cache: &f.k,
            kv_dim: KVD,
            row_of_pos: &f.row_of_pos,
            lambda,
            e_rows: &f.e_rows,
            table_rows: TABLE_ROWS,
            table_width: KVD,
        };
        let mut weights = vec![0.0f32; N_HEAD * TABLE_ROWS];
        let mut used: Vec<Vec<u32>> = vec![Vec::new(); N_HEAD];
        {
            let fr = FusedRecon {
                shared,
                weights: &mut weights,
                used: &mut used,
            };
            unsafe { attend_row_fused(&mut f.ctx, fr, 0, f.shape) };
        }
        // The epilogue's all-zero contract, asserted at the seam the forward
        // relies on between layer steps.
        assert!(weights.iter().all(|&w| w == 0.0), "weights must clear");
        assert!(used.iter().all(Vec::is_empty), "used lists must clear");
        f.ctx.attn_out[..N_HEAD * HD].to_vec()
    }

    fn all_miss(f: &mut Fixture) {
        f.row_of_pos.iter_mut().for_each(|r| *r = u32::MAX);
    }

    /// λ=0 and all-miss-λ=1 are BITWISE the eager path (the scratch rows are
    /// `v̂` bitwise on both sides — `add_scaled_row_inplace`'s λ==0 early
    /// return and the miss path's absence of any add) — sequential regime.
    #[test]
    fn fused_sequential_lambda0_and_all_miss_are_bitwise() {
        for lambda in [0.0f32, 1.0] {
            let mut f = fixture(37, 640);
            if lambda != 0.0 {
                all_miss(&mut f);
            }
            prime(&mut f);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&run_plain(&mut f, lambda)),
                bits(&run_fused(&mut f, lambda)),
                "λ={lambda}: the fused path must be bitwise the eager path"
            );
        }
    }

    /// Tracked λ>0 is the regrouped association — same product set, sum
    /// regrouped. The stated bound: ≤ 2·t_n·ε·λ·max|E| (worst-case pairwise
    /// reassociation); assert an order of magnitude inside it.
    #[test]
    fn fused_sequential_tracked_lambda_matches_within_the_regrouping_bound() {
        for lambda in [0.5f32, 1.0] {
            let mut f = fixture(37, 640);
            prime(&mut f);
            let plain = run_plain(&mut f, lambda);
            let fused = run_fused(&mut f, lambda);
            let max_e = f.e_rows.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let bound = 2.0 * 37.0 * f32::EPSILON * lambda * max_e;
            for (a, b) in plain.iter().zip(&fused) {
                assert!(
                    (a - b).abs() <= bound,
                    "λ={lambda}: |{a} − {b}| over the regrouping bound {bound}"
                );
            }
        }
    }

    /// The parallel-heads regime (t_n ≥ 512) — the same laws at the rayon
    /// split, where the per-head weight rows and used lists must be disjoint.
    #[test]
    fn fused_parallel_regime_holds_both_laws() {
        for lambda in [0.0f32, 1.0] {
            let mut f = fixture(600, 640);
            if lambda != 0.0 {
                all_miss(&mut f);
            }
            prime(&mut f);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&run_plain(&mut f, lambda)),
                bits(&run_fused(&mut f, lambda)),
                "parallel λ={lambda}: bitwise"
            );
        }
        let mut f = fixture(600, 640);
        prime(&mut f);
        let plain = run_plain(&mut f, 1.0);
        let fused = run_fused(&mut f, 1.0);
        let max_e = f.e_rows.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let bound = 2.0 * 600.0 * f32::EPSILON * max_e;
        for (a, b) in plain.iter().zip(&fused) {
            assert!((a - b).abs() <= bound, "parallel tracked: {a} vs {b}");
        }
    }

    /// The epilogue is the regrouped table part: `attn += λ·Σ_s W_s·E[s]`
    /// over touched rows, then weights/used clear. Hand-computed.
    #[test]
    fn epilogue_regroups_by_row_and_clears() {
        let f = fixture(1, 64);
        let shared = ReconShared {
            cs: &f.cs,
            half: HALF,
            k_cache: &f.k,
            kv_dim: KVD,
            row_of_pos: &f.row_of_pos,
            lambda: 0.5,
            e_rows: &f.e_rows,
            table_rows: TABLE_ROWS,
            table_width: KVD,
        };
        let mut attn_out = vec![0.0f32; N_HEAD * HD];
        let mut weights = vec![0.0f32; N_HEAD * TABLE_ROWS];
        // Head 0: rows 0 (w 0.5) + 1 (w 0.25); head 1: row 2 (w −0.0 — the
        // zero-weight row is skipped but still cleared); heads 2/3 untouched.
        weights[0] = 0.5;
        weights[1] = 0.25;
        weights[TABLE_ROWS + 2] = 0.0;
        let mut used: Vec<Vec<u32>> = vec![vec![0, 1], vec![2], Vec::new(), Vec::new()];
        fused_epilogue(
            &mut attn_out,
            &shared,
            &mut weights,
            &mut used,
            N_HEAD,
            N_KV,
            HD,
        );
        // E row 0 = (i·0.125 − 0.75) for i in 0..KVD — the head-0 slice.
        for (d, got) in attn_out[..HD].iter().enumerate() {
            let want = 0.5 * (0.5 * f.e_rows[d]) + 0.5 * (0.25 * f.e_rows[KVD + d]);
            assert!((got - want).abs() < 1e-6, "d={d}");
        }
        for got in &attn_out[(N_HEAD - 1) * HD..N_HEAD * HD] {
            assert_eq!(*got, 0.0, "head 3 untouched");
        }
        assert!(weights.iter().all(|&w| w == 0.0), "cleared");
        assert!(used.iter().all(Vec::is_empty), "cleared");
    }
}
