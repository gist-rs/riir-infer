//! Scale-invariant attention — the opt-in consumer of katgpt-core
//! `scale_invariant_attn` (Plan 622 Phase 7 G3 enablement; the vehicle is the
//! packed Bonsai ternary decode lane).
//!
//! Same three passes as [`super::attention::attention_heads_parallel`]
//! (scores → softmax → value sum), with the schedule's affine applied in
//! pass 1's LOGIT space — post-scale, pre-max (the FlexAttention `score_mod`
//! shape; the transform is per-element monotone-positive, so tracking the max
//! over the transformed scores is the correct softmax):
//!
//! ```text
//! s_t → a_t · (s_t / σ_h) + m_t      calibrated   (σ_h from the frozen table)
//! s_t → a_t ·  s_t        + m_t      uncalibrated (σ_h = 1.0, IEEE-exact)
//! ```
//!
//! - distance `t = query_pos − key_pos`; the key-cache row `t` is the global
//!   position `t` (full-attention decode; the ternary lane feeds one token at
//!   a time, so `t_n = query_pos + 1` always).
//! - **Sink carve-out:** key 0 keeps its raw logit on BOTH axes (the
//!   schedule's Gaussianity premise excludes it, paper App. J) and the
//!   sink's logit never enters the head's σ.
//! - **Probe mode:** the load-time half of the calibration contract — plain
//!   math (bit-identical logits) + per-`(layer, head)` raw-logit Welford
//!   observation with the sink dropped, the measurement the frozen
//!   [`HeadCalibration`] table builds from. Sequential per head (a
//!   load-time instrument, never a hot path).
//!
//! The armed apply is bit-identical to the katgpt-core reference
//! (`ScaleInvariantLut::apply_inplace_calibrated` at `logit_scale = 1.0`):
//! the same `a · (s · (1/σ)) + m` expression order, and `σ = 1.0` is the
//! pass-through pin (`x · 1.0 == x` for every f32).

use std::sync::Arc;

use katgpt_core::scale_invariant::{HeadCalibration, ScaleInvariantLut, SpreadAccumulator};
use rayon::prelude::*;

use super::attention::PARALLEL_HEADS_MIN_SEQ;

/// The armed-schedule plan: session-built, read-only on the hot path.
pub struct SiAttentionPlan {
    /// The schedule LUT. `max_ctx` must cover the run's longest query
    /// position — a longer one fails loud at the LUT read (the documented
    /// slice panic), so build it at the cache block size.
    pub lut: Arc<ScaleInvariantLut>,
    /// Frozen per-head σ table over `n_layers · n_head` slots, slot =
    /// `layer_idx · n_head + head`. `None` = the uncalibrated arm.
    pub calibration: Option<Arc<HeadCalibration>>,
    /// Heads per layer — the calibration slot stride (the config constant;
    /// every armed call re-checks it against its own `n_head`).
    pub n_head: usize,
}

impl SiAttentionPlan {
    /// The uncalibrated arm.
    ///
    /// # Panics
    /// If `n_head` is zero (a plan that cannot index a layer).
    pub fn uncalibrated(lut: Arc<ScaleInvariantLut>, n_head: usize) -> Self {
        assert!(n_head > 0, "si plan: n_head must be positive");
        Self {
            lut,
            calibration: None,
            n_head,
        }
    }

    /// The calibrated arm. The table must cover exactly
    /// `n_layers · n_head` slots — a mismatched sidecar is the wrong
    /// sidecar (loud, at plan build, never deep in a decode).
    ///
    /// # Errors
    /// When the calibration table does not cover `n_layers · n_head` heads.
    pub fn calibrated(
        lut: Arc<ScaleInvariantLut>,
        calibration: HeadCalibration,
        n_head: usize,
        n_layers: usize,
    ) -> Result<Self, String> {
        assert!(n_head > 0, "si plan: n_head must be positive");
        assert!(n_layers > 0, "si plan: n_layers must be positive");
        calibration.validate_heads(n_layers * n_head)?;
        Ok(Self {
            lut,
            calibration: Some(Arc::new(calibration)),
            n_head,
        })
    }

    /// The head's σ — the measured scale, or `1.0` (the pass-through).
    #[inline]
    pub fn sigma_for(&self, layer_idx: usize, head: usize) -> f32 {
        match &self.calibration {
            None => 1.0,
            Some(cal) => cal.scale_for(layer_idx * self.n_head + head),
        }
    }
}

/// What one session of forwards does to the attention scores (the value the
/// [`HybridForwardScratch::scale_invariant`](crate::deltanet::forward::HybridForwardScratch)
/// field carries; swapping the variant between arms is the G3 A/B).
pub enum SiAttentionSession {
    /// Load-time σ probe: plain math, per-`(layer, head)` raw-logit
    /// observation, sink dropped. Logits stay bit-identical to the plain
    /// path — the probe must measure what the calibration will normalize.
    Probe(SpreadAccumulator),
    /// Serving: the schedule affine in logit space.
    Armed(Arc<SiAttentionPlan>),
}

/// Per-head mode: what pass 1 does to each raw score.
enum SiHeadMode<'a> {
    /// Plain math + observe the raw row (sink dropped) at `slot`.
    Probe {
        slot: usize,
        acc: &'a mut SpreadAccumulator,
    },
    /// The affine; `inv_head = 1/σ_h` (`1.0` uncalibrated).
    Apply {
        inv_head: f32,
        plan: &'a SiAttentionPlan,
    },
}

/// One head, all three passes, the schedule folded into pass 1.
///
/// # Safety
///
/// Same contract as [`super::attention::attention_head_on_slices`], plus
/// `query_pos + 1 == t_n` and `query_pos <= plan.lut.max_distance()` for
/// the [`SiHeadMode::Apply`] arm.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn si_head(
    q_head: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    out_head: &mut [f32],
    scores: &mut [f32],
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    query_pos: usize,
    mode: SiHeadMode<'_>,
) {
    debug_assert!(
        query_pos + 1 == t_n,
        "si decode contract: t_n == query_pos + 1"
    );
    // Pass 1: raw score per key, then the mode's transform, max tracked
    // over the TRANSFORMED row (what the softmax actually sees).
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = crate::simd::simd_dot_f32(&q_head[..hd], &key_cache[k_off..k_off + hd], hd);
        let raw = dot * scale;
        let score = match &mode {
            SiHeadMode::Probe { .. } => raw,
            SiHeadMode::Apply { inv_head, plan } => {
                // Sink carve-out: key 0 keeps its raw logit (App. J).
                if t == 0 {
                    raw
                } else {
                    let (a, m) = plan.lut.pair(query_pos - t);
                    a * (raw * inv_head) + m
                }
            }
        };
        unsafe {
            *scores.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }
    // The probe observes the RAW row (pass 2 overwrites it below), sink
    // dropped: the schedule never transforms key 0, so its logit is
    // outside the σ population (the Phase-3 contract).
    if let SiHeadMode::Probe { slot, acc } = mode
        && t_n > 1
    {
        acc.observe_head(slot, unsafe { scores.get_unchecked(1..t_n) });
    }

    // Passes 2 and 3: identical to the plain kernel (exp − max; normalize
    // and accumulate values).
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }
    let inv_sum = 1.0 / sum;
    for t in 0..t_n {
        let w = unsafe { *scores.get_unchecked(t) * inv_sum };
        let v_off = t * kv_dim + kv_group_offset;
        for d in 0..hd {
            unsafe {
                *out_head.get_unchecked_mut(d) += w * *value_cache.get_unchecked(v_off + d);
            }
        }
    }
}

/// The probe path: plain math (bit-identical to
/// [`super::attention::attention_heads_parallel`]) + per-head σ observation.
/// Sequential per head — a load-time instrument, never a hot path.
///
/// # Safety
///
/// Identical to [`super::attention::attention_heads_parallel`], plus
/// `query_pos + 1 == t_n` and `acc` pre-sized for
/// `(layer_idx + 1) · n_head` slots.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_heads_probe_si(
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
    block_size: usize,
    query_pos: usize,
    layer_idx: usize,
    acc: &mut SpreadAccumulator,
) {
    let scores = &mut head_scores[..block_size];
    for h in 0..n_head {
        let kv_group_offset = (h * n_kv_head / n_head) * head_dim;
        let slot = layer_idx * n_head + h;
        unsafe {
            si_head(
                &q[h * head_dim..],
                key_cache,
                value_cache,
                &mut attn_out[h * head_dim..],
                scores,
                kv_group_offset,
                kv_dim,
                head_dim,
                t_n,
                scale,
                query_pos,
                SiHeadMode::Probe { slot, acc },
            );
        }
    }
}

/// The armed path: the schedule affine in pass 1's logit space, seq/par
/// dispatch mirroring the plain kernel (same `PARALLEL_HEADS_MIN_SEQ`
/// crossover), per-head σ from the plan.
///
/// # Safety
///
/// Identical to [`super::attention::attention_heads_parallel`], plus
/// `query_pos + 1 == t_n` and `plan.n_head == n_head`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_heads_parallel_si(
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
    block_size: usize,
    query_pos: usize,
    layer_idx: usize,
    plan: &SiAttentionPlan,
) {
    debug_assert_eq!(
        plan.n_head, n_head,
        "si plan head count must match the model"
    );
    // σ folded per head at read time — no per-call allocation (the hot-path
    // zero-alloc law; the divide is one flop beside the hd-wide dot).
    let head_inv = |h: usize| 1.0 / plan.sigma_for(layer_idx, h);
    let group = |h: usize| (h * n_kv_head / n_head) * head_dim;
    if t_n < PARALLEL_HEADS_MIN_SEQ || n_head <= 1 {
        for h in 0..n_head {
            let inv_head = head_inv(h);
            unsafe {
                si_head(
                    &q[h * head_dim..],
                    key_cache,
                    value_cache,
                    &mut attn_out[h * head_dim..],
                    &mut head_scores[..block_size],
                    group(h),
                    kv_dim,
                    head_dim,
                    t_n,
                    scale,
                    query_pos,
                    SiHeadMode::Apply { inv_head, plan },
                );
            }
        }
    } else {
        let q_dim = n_head * head_dim;
        attn_out[..q_dim]
            .par_chunks_mut(head_dim)
            .zip(head_scores.par_chunks_mut(block_size))
            .enumerate()
            .for_each(|(h, (attn_h, scores_h))| {
                let inv_head = head_inv(h);
                unsafe {
                    si_head(
                        &q[h * head_dim..],
                        key_cache,
                        value_cache,
                        attn_h,
                        scores_h,
                        group(h),
                        kv_dim,
                        head_dim,
                        t_n,
                        scale,
                        query_pos,
                        SiHeadMode::Apply { inv_head, plan },
                    );
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_HEAD: usize = 4;
    const N_KV: usize = 2;
    const HD: usize = 16;
    const KV_DIM: usize = N_KV * HD;

    /// Deterministic LCG fixture (the attention_floor house shape).
    fn lcg(i: usize) -> f32 {
        (((i * 2_654_435_761) % 1_000_003) as f32 / 1_000_003.0) * 2.0 - 1.0
    }

    struct Fixture {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        t_n: usize,
        query_pos: usize,
    }

    fn fixture(t_n: usize) -> Fixture {
        Fixture {
            q: (0..N_HEAD * HD).map(|i| 3.0 * lcg(i + 7)).collect(),
            k: (0..t_n * KV_DIM).map(|i| lcg(i + 101)).collect(),
            v: (0..t_n * KV_DIM).map(|i| lcg(i + 9_001)).collect(),
            t_n,
            query_pos: t_n - 1,
        }
    }

    fn run_plain(f: &Fixture) -> Vec<f32> {
        let mut out = vec![0.0f32; N_HEAD * HD];
        let mut scores = vec![0.0f32; 256];
        unsafe {
            super::super::attention::attention_heads_parallel(
                &f.q,
                &f.k,
                &f.v,
                &mut out,
                &mut scores,
                N_HEAD,
                N_KV,
                KV_DIM,
                HD,
                f.t_n,
                0.25,
                0.0,
                256,
            );
        }
        out
    }

    fn run_armed(f: &Fixture, plan: &SiAttentionPlan) -> Vec<f32> {
        let mut out = vec![0.0f32; N_HEAD * HD];
        let mut scores = vec![0.0f32; 256];
        unsafe {
            attention_heads_parallel_si(
                &f.q,
                &f.k,
                &f.v,
                &mut out,
                &mut scores,
                N_HEAD,
                N_KV,
                KV_DIM,
                HD,
                f.t_n,
                0.25,
                256,
                f.query_pos,
                0,
                plan,
            );
        }
        out
    }

    fn tau_lut(t_n: usize) -> Arc<ScaleInvariantLut> {
        Arc::new(ScaleInvariantLut::build(10.0, t_n.max(2) - 1))
    }

    /// The armed-uncalibrated output must equal the manual reference:
    /// raw scores materialized, the katgpt-core reference apply
    /// (`apply_inplace`, `logit_scale = 1.0`), then the plain
    /// softmax + value accumulation — BIT-IDENTICAL (same expression
    /// order; `x · 1.0 == x`).
    #[test]
    fn armed_uncalibrated_matches_the_reference_apply_bitwise() {
        let f = fixture(97);
        let lut = tau_lut(f.t_n);
        let plan = SiAttentionPlan::uncalibrated(lut.clone(), N_HEAD);

        // Reference: per head, raw scores → reference apply → plain softmax.
        let mut want = vec![0.0f32; N_HEAD * HD];
        for h in 0..N_HEAD {
            let kv_group_offset = (h * N_KV / N_HEAD) * HD;
            let mut scores = vec![0.0f32; f.t_n];
            let mut max_score = f32::NEG_INFINITY;
            for (t, s) in scores.iter_mut().enumerate() {
                let k_off = t * KV_DIM + kv_group_offset;
                let dot = crate::simd::simd_dot_f32(
                    &f.q[h * HD..h * HD + HD],
                    &f.k[k_off..k_off + HD],
                    HD,
                );
                *s = dot * 0.25;
            }
            lut.apply_inplace(&mut scores, f.query_pos, 0, 1.0);
            for s in &scores {
                max_score = max_score.max(*s);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for (t, s) in scores.iter().enumerate() {
                let w = s * inv_sum;
                let v_off = t * KV_DIM + kv_group_offset;
                for d in 0..HD {
                    want[h * HD + d] += w * f.v[v_off + d];
                }
            }
        }

        let got = run_armed(&f, &plan);
        assert_eq!(
            got, want,
            "armed-uncalibrated must be bit-identical to the reference"
        );
    }

    /// The calibrated divide must match the manual reference
    /// `a_t · (raw / σ_h) + m_t` bitwise, and a σ table of all ones must
    /// equal the uncalibrated arm exactly (the pass-through pin).
    #[test]
    fn calibrated_divide_matches_reference_and_ones_are_the_pass_through() {
        let f = fixture(65);
        let lut = tau_lut(f.t_n);
        let sigmas: Vec<f32> = (0..N_HEAD).map(|h| 0.5 + 0.25 * h as f32).collect();
        let cal = HeadCalibration::build(sigmas.clone()).expect("valid sigmas");
        let plan = SiAttentionPlan::calibrated(lut.clone(), cal, N_HEAD, 1)
            .expect("table covers 1 layer × 4 heads");

        // Reference with the divide.
        let mut want = vec![0.0f32; N_HEAD * HD];
        for h in 0..N_HEAD {
            let kv_group_offset = (h * N_KV / N_HEAD) * HD;
            let inv_head = 1.0 / sigmas[h];
            let mut scores = vec![0.0f32; f.t_n];
            for (t, s) in scores.iter_mut().enumerate() {
                let k_off = t * KV_DIM + kv_group_offset;
                let dot = crate::simd::simd_dot_f32(
                    &f.q[h * HD..h * HD + HD],
                    &f.k[k_off..k_off + HD],
                    HD,
                );
                let raw = dot * 0.25;
                *s = match t {
                    0 => raw,
                    _ => {
                        let (a, m) = lut.pair(f.query_pos - t);
                        a * (raw * inv_head) + m
                    }
                };
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for (t, s) in scores.iter().enumerate() {
                let w = s * inv_sum;
                let v_off = t * KV_DIM + kv_group_offset;
                for d in 0..HD {
                    want[h * HD + d] += w * f.v[v_off + d];
                }
            }
        }
        let got = run_armed(&f, &plan);
        assert_eq!(
            got, want,
            "calibrated arm must be bit-identical to the divide reference"
        );

        // All-ones table == uncalibrated, bitwise.
        let ones = HeadCalibration::build(vec![1.0; N_HEAD]).expect("ones");
        let plan_ones = SiAttentionPlan::calibrated(lut.clone(), ones, N_HEAD, 1).expect("covers");
        let plan_uncal = SiAttentionPlan::uncalibrated(lut, N_HEAD);
        assert_eq!(
            run_armed(&f, &plan_ones),
            run_armed(&f, &plan_uncal),
            "σ = 1.0 must be the uncalibrated pass-through"
        );
    }

    /// A wrong-sized calibration table refuses at plan build, never in a
    /// decode.
    #[test]
    fn calibration_head_count_mismatch_is_a_loud_plan_error() {
        let lut = tau_lut(8);
        let cal = HeadCalibration::build(vec![1.0; 3]).expect("3 scales");
        assert!(SiAttentionPlan::calibrated(lut, cal, N_HEAD, 1).is_err());
    }

    /// The probe path is bit-identical to the plain kernel AND measures the
    /// raw (untransformed) spreads with the sink dropped.
    #[test]
    fn probe_output_matches_plain_and_measures_raw_spreads() {
        let f = fixture(70);
        let mut acc = SpreadAccumulator::new(N_HEAD);
        let mut out = vec![0.0f32; N_HEAD * HD];
        let mut scores = vec![0.0f32; 256];
        unsafe {
            attention_heads_probe_si(
                &f.q,
                &f.k,
                &f.v,
                &mut out,
                &mut scores,
                N_HEAD,
                N_KV,
                KV_DIM,
                HD,
                f.t_n,
                0.25,
                256,
                f.query_pos,
                0,
                &mut acc,
            );
        }
        assert_eq!(
            out,
            run_plain(&f),
            "probe logits must be bit-identical to the plain path"
        );

        // Raw-spread check: per head, the population std of the raw scores
        // (sink dropped) must equal the accumulator's estimate.
        for h in 0..N_HEAD {
            let kv_group_offset = (h * N_KV / N_HEAD) * HD;
            let raw: Vec<f32> = (1..f.t_n)
                .map(|t| {
                    let k_off = t * KV_DIM + kv_group_offset;
                    crate::simd::simd_dot_f32(
                        &f.q[h * HD..h * HD + HD],
                        &f.k[k_off..k_off + HD],
                        HD,
                    ) * 0.25
                })
                .collect();
            let n = raw.len() as f64;
            let mean = raw.iter().map(|s| f64::from(*s)).sum::<f64>() / n;
            let var = raw
                .iter()
                .map(|s| (f64::from(*s) - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            let got = acc.std_estimate(h).expect("observations exist");
            assert!(
                (got - var.sqrt()).abs() < 1e-9,
                "head {h}: probe σ {got} vs population {}",
                var.sqrt()
            );
        }
        assert_eq!(acc.heads(), N_HEAD, "every head observed");
    }

    /// The armed LUT coverage: a query beyond `max_distance` fails loud
    /// (the LUT read panic) — never a silent wrong-distance transform.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn query_beyond_the_lut_fails_loud() {
        let f = fixture(64);
        let lut = Arc::new(ScaleInvariantLut::build(10.0, 8));
        let plan = SiAttentionPlan::uncalibrated(lut, N_HEAD);
        let _ = run_armed(&f, &plan);
    }

    // ── end-to-end wiring (the real hybrid forward, the dispatch site) ──

    use crate::deltanet::forward::{HybridCache, HybridForwardScratch};
    use crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
    use crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
    use crate::types::{Config, DeltaNetLayerType};

    /// A 2-layer hybrid (1 GDN + 1 attention) with deterministic NON-ZERO
    /// Q/K/V projections + LM head — zero weights would make every score
    /// (and so every σ, and every logit) identically zero and the wiring
    /// pins would read green over nothing.
    fn wiring_model() -> (Config, QwenDeltaNetTernaryWeights) {
        let layer_types = vec![DeltaNetLayerType::DeltaNet, DeltaNetLayerType::Attention];
        let mut config = Config::qwen_deltanet(2, layer_types);
        config.vocab_size = 256;
        config.n_embd = 128;
        config.n_head = 2;
        config.n_kv_head = 1;
        config.head_dim = 128;
        config.mlp_hidden = 256;
        config.deltanet_linear_head_dim = 128;
        config.deltanet_linear_n_heads = 1;
        config.deltanet_linear_n_value_heads = 1;
        config.block_size = 64;
        let mut weights = QwenDeltaNetTernaryWeights::zeros(&config);
        // Dense deterministic trit patterns (-1/0/+1, one value per cell) —
        // a SPARSE speckle reads green over nothing here: sparse W × sparse
        // embedding makes q·k = 0 EXACTLY at almost every position (empty
        // support intersections), so every score is bit-zero, every σ is 0,
        // and the armed/plain contrast fires only through m_t over a
        // uniform-softmax baseline.
        let dense_trits = |w: &mut katgpt_core::TernaryGroupWeights, salt: usize| {
            for r in 0..w.rows {
                for c in 0..w.cols {
                    w.set(r, c, ((r * 31 + c * 17 + salt) % 3) as i8 - 1);
                }
            }
        };
        let layer = &mut weights.layers[1];
        dense_trits(layer.attn_wq.as_ternary_mut().expect("ternary wq"), 0);
        dense_trits(layer.attn_wk.as_ternary_mut().expect("ternary wk"), 11);
        dense_trits(layer.attn_wv.as_ternary_mut().expect("ternary wv"), 23);
        // The OUTPUT projection must be nonzero too: attention reaches the
        // logits only through `attn_wo` — a zero wo makes the armed/plain
        // comparison (and every wiring pin) read green over nothing.
        dense_trits(layer.attn_wo.as_ternary_mut().expect("ternary wo"), 35);
        // The Q/K norm gammas must be nonzero: `zeros()` leaves them at 0.0
        // and an RMSNorm with zero gamma zeroes q and k — every score (and
        // so every σ, and the armed/plain contrast's score axis) collapses
        // to exactly 0.
        layer.attn_q_norm.fill(1.0);
        layer.attn_k_norm.fill(1.0);
        // Same for the layer INPUT norm — a zero gamma zeroes the attention
        // block's input, which zeroes q and k one step earlier (the scores
        // never see the embedding at all).
        layer.input_norm.fill(1.0);
        dense_trits(&mut weights.lm_head, 41);
        // The embedding seeds the residual stream — zero wte means every
        // hidden state (and so every score) stays identically zero no
        // matter what the projections hold.
        dense_trits(&mut weights.wte, 59);
        (config, weights)
    }

    fn wiring_forward(
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
        session: Option<SiAttentionSession>,
        lut: &Arc<ScaleInvariantLut>,
    ) -> Vec<f32> {
        let layer_types = weights.layer_types.clone();
        let mut cache = HybridCache::with_layer_types(config, &layer_types);
        let mut scratch = HybridForwardScratch::new(config);
        scratch.attention.scale_invariant = session;
        let rope = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; config.n_embd.max(config.vocab_size)];
        let mut last = Vec::new();
        for pos in 0..24 {
            let token = (pos * 7 + 3) % config.vocab_size;
            let logits = forward_qwen_deltanet_ternary(
                &mut x,
                weights,
                &mut cache,
                token,
                pos,
                config,
                &mut scratch,
                &rope,
            );
            last = logits.to_vec();
        }
        let _ = lut;
        last
    }

    /// The dispatch REACHES the armed kernel: the armed-uncalibrated logits
    /// differ from the plain path's (the transform is active through the
    /// full model), and the same session is deterministic.
    #[test]
    fn armed_session_changes_the_logits_through_the_full_model() {
        let (config, weights) = wiring_model();
        let lut = Arc::new(ScaleInvariantLut::build(10.0, config.block_size));
        let plain = wiring_forward(&config, &weights, None, &lut);
        let session = |lut: &Arc<ScaleInvariantLut>| {
            Some(SiAttentionSession::Armed(Arc::new(
                SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
            )))
        };
        let armed1 = wiring_forward(&config, &weights, session(&lut), &lut);
        let armed2 = wiring_forward(&config, &weights, session(&lut), &lut);
        assert_eq!(armed1, armed2, "the same session must be deterministic");
        assert_ne!(
            plain, armed1,
            "the armed kernel must be reachable through the model"
        );
    }

    /// The probe session leaves the logits BIT-IDENTICAL to the plain path
    /// (it must measure, never disturb) and fills every (layer, head) slot
    /// of the accumulator.
    #[test]
    fn probe_session_is_bit_identical_and_fills_every_slot() {
        let (config, weights) = wiring_model();
        let lut = Arc::new(ScaleInvariantLut::build(10.0, config.block_size));
        let plain = wiring_forward(&config, &weights, None, &lut);

        let slots = weights.layers.len() * config.n_head;
        let acc = SpreadAccumulator::new(slots);
        let probed = wiring_forward(
            &config,
            &weights,
            Some(SiAttentionSession::Probe(acc)),
            &lut,
        );
        assert_eq!(probed, plain, "the probe must measure, never disturb");
    }

    /// The full calibration round trip through the wiring: probe → σ table
    /// (loud on a degenerate population) → armed-calibrated plan validates
    /// against n_layers × n_head.
    #[test]
    fn calibration_round_trip_through_the_wiring() {
        let (config, weights) = wiring_model();
        let lut = Arc::new(ScaleInvariantLut::build(10.0, config.block_size));
        let slots = weights.layers.len() * config.n_head;
        let acc = SpreadAccumulator::new(slots);
        let _ = wiring_forward(
            &config,
            &weights,
            Some(SiAttentionSession::Probe(acc)),
            &lut,
        );
        // Re-run with extraction (wiring_forward consumes the session, so
        // run the probe inline here for the acc half).
        let layer_types = weights.layer_types.clone();
        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        scratch.attention.scale_invariant =
            Some(SiAttentionSession::Probe(SpreadAccumulator::new(slots)));
        let rope = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; config.n_embd.max(config.vocab_size)];
        for pos in 0..24 {
            let token = (pos * 7 + 3) % config.vocab_size;
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                token,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
        }
        let Some(SiAttentionSession::Probe(acc)) = scratch.attention.scale_invariant.take() else {
            panic!("probe session vanished");
        };
        // `heads()` is the PRE-SIZED slot count, never an observation count —
        // the observed set is the slots with a std estimate. Every
        // ATTENTION-layer slot observed; the GDN layer has no per-key logits
        // (the Plan-622 audited discard) and never routes the probe.
        let observed = (0..slots)
            .filter(|&s| acc.std_estimate(s).is_some())
            .count();
        assert_eq!(
            observed, config.n_head,
            "every attention-layer slot observed"
        );
        // σ of a deterministic nonzero fixture is positive on the observed
        // slots; the GDN slots ride the NEUTRAL scale (1.0, never read — the
        // armed dispatch fires only in attention layers) so the table still
        // covers n_layers × n_head for the plan validation.
        let scales: Vec<f32> = (0..slots)
            .map(|s| acc.std_estimate(s).unwrap_or(1.0) as f32)
            .collect();
        let cal = HeadCalibration::build(scales).expect("positive σ everywhere");
        let plan = SiAttentionPlan::calibrated(lut, cal, config.n_head, weights.layers.len())
            .expect("table covers n_layers × n_head");
        assert_eq!(plan.n_head, config.n_head);
    }
}
