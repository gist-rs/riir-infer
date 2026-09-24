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

use super::{ForwardContext, attention_heads_parallel};

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
