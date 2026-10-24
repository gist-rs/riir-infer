//! Plan 622 T3.4 — the FastKV-class tap's G1 gate battery.
//!
//! The FastKV-class selector (token scored by attention received from a
//! window of the LAST queries, max-pooled, top-K) rides the SAME layer-L tap
//! seam as the T3.3 mass arm — `tap_window = Some(w)` picks the pooling.
//! Pinned here, on the synthetic [DeltaNet, Attention, Attention] hybrid
//! (`common/plan622_synth.rs`, L = 1 a full-attention mixer):
//!
//! 1. **Determinism** — an identical run reproduces the tap bit-for-bit.
//! 2. **Window monotonicity** — `fastkv(w=1) ≤ fastkv(w=T)` pointwise (a
//!    max over a smaller window cannot exceed the max over a superset of
//!    the same rows; every key is covered by both windows' last query).
//! 3. **max ≤ sum** — `fastkv(w=T) ≤ mass` pointwise (max ≤ Σ over the
//!    same per-query head-sum rows).
//! 4. **Last-key window invariance** — key T-1 is covered by exactly one
//!    query (the last one), so its fastkv score equals the mass score for
//!    EVERY window.
//! 5. **w = 1 conservation** — a single pooled query distributes exactly
//!    1.0 per head over its causal keys: `Σ_k fastkv[k] = n_head`.
//! 6. **Clamp** — window 0 clamps to 1 (bitwise ≡ w = 1); window > T clamps
//!    to T (bitwise ≡ w = T).
//! 7. **Selection runs** — the armed tap's scores select (K < prunable
//!    actually prunes) and the loud refusals hold (a GDN layer at L refuses
//!    the tap with `tap_window` armed too).
//!
//! The arm is REPORTED, NON-GATING in the plan's harness (T3.4's GOAT gate
//! is derived ≥ random-K); these pins are the correctness floor under the
//! reported column.

#![cfg(all(
    feature = "tokengate",
    feature = "deltanet_inference",
    not(target_arch = "wasm32")
))]

#[path = "common/plan622_synth.rs"]
mod synth;

use riir_infer_core::deltanet::forward::PrunedPrefillPlan;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, prefill_qwen_deltanet_pruned_into,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokengate::scorer_attention_mass;

const T: usize = 24;
const N_SINK: usize = 4;

struct Rig {
    cfg: riir_infer_core::types::Config,
    weights: riir_infer_core::deltanet::weights::QwenDeltaNetWeights,
    rope: RopeFreqTable,
    tokens: Vec<usize>,
}

fn rig(seed: u64) -> Rig {
    let cfg = synth::hybrid_config();
    let weights = synth::synth_hybrid_weights(&cfg, seed);
    let rope = RopeFreqTable::new(cfg.rope_theta, cfg.head_dim);
    let mut sm = synth::SplitMix(seed);
    let tokens: Vec<usize> = (0..T)
        .map(|_| (sm.next_f32().abs() * cfg.vocab_size as f32) as usize % cfg.vocab_size)
        .collect();
    Rig {
        cfg,
        weights,
        rope,
        tokens,
    }
}

fn fresh_run(r: &Rig) -> (HybridCache, HybridForwardScratch, PrefillContext, Vec<f32>) {
    let cache = HybridCache::with_layer_types(&r.cfg, &r.weights.layer_types);
    let scratch = HybridForwardScratch::new(&r.cfg);
    let ctx = PrefillContext::new(&r.cfg, T);
    let logits = vec![0.0f32; r.cfg.vocab_size];
    (cache, scratch, ctx, logits)
}

/// One armed-tap run at L = 1 (the full-attention mixer): returns the scores
/// the scorer consumed (a copy of the tap) and the pruned plan.
fn tap_run(r: &Rig, window: Option<usize>, keep: usize) -> (Vec<f32>, PrunedPrefillPlan) {
    let (mut cache, mut scratch, mut ctx, mut logits) = fresh_run(r);
    let mut plan = PrunedPrefillPlan::new(1, N_SINK, keep);
    let mut tap = vec![0.0f32; T];
    let mut scorer = |_hidden: &[f32], m: Option<&[f32]>, scores: &mut [f32]| {
        scorer_attention_mass(_hidden, m, scores)
    };
    prefill_qwen_deltanet_pruned_into(
        &r.weights,
        &r.cfg,
        &mut cache,
        &r.tokens,
        &mut scratch,
        &r.rope,
        &mut ctx,
        &mut logits,
        None,
        &mut plan,
        Some(&mut tap),
        window,
        &mut scorer,
    );
    let scores = plan.scores.clone();
    (scores, plan)
}

#[test]
fn fastkv_tap_is_deterministic_distinguishing_and_selects() {
    let r = rig(0x6220_0046);
    let keep = T - N_SINK - 4;
    let (scores, plan) = tap_run(&r, Some(8), keep);

    assert!(scores.iter().all(|s| s.is_finite()), "finite fastkv scores");
    assert!(
        scores.iter().all(|&s| s >= 0.0),
        "attention weights are non-negative"
    );
    assert!(
        scores.iter().any(|&s| s != scores[0]),
        "the fastkv tap distinguishes tokens"
    );
    assert!(plan.kept.len() < T, "pruning actually happened");

    // Determinism: an identical run reproduces the scores bit-for-bit.
    let (scores2, _) = tap_run(&r, Some(8), keep);
    synth::assert_bits_eq(&scores, &scores2, "fastkv run-to-run bits");
}

#[test]
fn fastkv_window_monotone_max_le_sum_and_last_key_invariance() {
    let r = rig(0x6220_0047);
    let keep = T - N_SINK - 4;
    let (w1, _) = tap_run(&r, Some(1), keep);
    let (wall, _) = tap_run(&r, Some(T), keep);
    let (mass, _) = tap_run(&r, None, keep);

    // Window monotonicity: w=1's max is over a subset of w=T's rows (the
    // per-query head-sums are computed identically in both runs, so the
    // inequality is exact, not tolerance-class).
    for (a, b) in w1.iter().zip(wall.iter()) {
        assert!(a <= b, "fastkv(w=1) {a} exceeds fastkv(w=T) {b}");
    }
    // max ≤ sum: the mass tap sums the same rows the fastkv tap maxes.
    for (a, b) in wall.iter().zip(mass.iter()) {
        assert!(a <= b, "fastkv(w=T) {a} exceeds the mass {b}");
    }
    // Last-key window invariance: key T-1 is covered by exactly one query
    // (the last), so its score is the same for every window — including the
    // mass's (its single contribution is both the max and the sum).
    assert_eq!(w1[T - 1].to_bits(), wall[T - 1].to_bits());
    assert_eq!(wall[T - 1].to_bits(), mass[T - 1].to_bits());
}

#[test]
fn fastkv_w1_conservation_and_clamps() {
    let r = rig(0x6220_0048);
    let keep = T - N_SINK - 4;

    // w = 1: exactly one pooled query distributing 1.0 per head over its
    // causal keys → Σ_k scores[k] = n_head.
    let (w1, _) = tap_run(&r, Some(1), keep);
    let total: f32 = w1.iter().sum();
    let want = r.cfg.n_head as f32;
    assert!(
        (total - want).abs() < 1e-3 * want,
        "w=1 conservation {total} vs {want}"
    );

    // Window 0 clamps to 1 (bitwise identical); window > T clamps to T.
    let (w0, _) = tap_run(&r, Some(0), keep);
    synth::assert_bits_eq(&w0, &w1, "window 0 clamps to 1");
    let (whuge, _) = tap_run(&r, Some(10_000), keep);
    let (wt, _) = tap_run(&r, Some(T), keep);
    synth::assert_bits_eq(&whuge, &wt, "window > T clamps to T");
}

#[test]
fn fastkv_refusals_stay_loud_with_the_window_armed() {
    let r = rig(0x6220_0049);

    // A GDN layer at L refuses the tap at ANY window kind — a GDN mixer has
    // no attention to pool.
    let (mut cache, mut scratch, mut ctx, mut logits) = fresh_run(&r);
    let mut plan = PrunedPrefillPlan::new(0, N_SINK, 4);
    let mut tap = vec![0.0f32; T];
    let mut scorer = |_h: &[f32], _m: Option<&[f32]>, s: &mut [f32]| s.fill(1.0);
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prefill_qwen_deltanet_pruned_into(
            &r.weights,
            &r.cfg,
            &mut cache,
            &r.tokens,
            &mut scratch,
            &r.rope,
            &mut ctx,
            &mut logits,
            None,
            &mut plan,
            Some(&mut tap),
            Some(8),
            &mut scorer,
        );
    }));
    assert!(out.is_err(), "GDN layer at L must refuse the fastkv tap");
}
