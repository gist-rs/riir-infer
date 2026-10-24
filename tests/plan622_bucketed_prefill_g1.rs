//! Plan 622 T2.2 — the pad-bias bucket shape's G1 gate battery.
//!
//! Pins, on the synthetic [DeltaNet, Attention, Attention] hybrid (the
//! shared `common/plan622_synth.rs` kit), the laws the bucketed pruned
//! prefill guarantees:
//!
//! 1. **Bucketed ≡ unpadded, byte-for-byte** — at every bucket width
//!    (64 and 128 here) and at both prune placements and at K = full
//!    included, the bucketed run's logits, EVERY layer's GDN recurrent +
//!    conv state, and the KV rows `0..real_len` are bit-identical to the
//!    unpadded run. The pads are zero-hidden rows: the GDN recurrence
//!    stops at `real_len` (state provably never observes a pad — the
//!    Issue-1107 law's intent, satisfied by construction for this
//!    single-sequence shape), the attention phases are row-local, and the
//!    final logits read the last KEPT row (`real_len - 1`, never a pad).
//! 2. **Pad slots are dead keys** — the KV slots the pads write
//!    (`real_len..padded_len`) hold exact-zero K/V rows, so even an UNCUT
//!    read (a caller ignoring the cut-back contract) only ever sees finite
//!    zeros there.
//! 3. **The −1e9 pad-bias law** (the fixed-shape arm's premise, T2.4) — a
//!    query attending the FULL padded window through
//!    `attention_heads_parallel_bias` with the pad-bias table is
//!    bit-identical to the same query causally bounded at `real_len` with
//!    no bias: `exp(-1e9 - max) == 0.0` exactly, so the pad keys add `+0.0`
//!    terms and every real-key sum keeps its order and its bits.
//! 4. **Bookkeeping** — `plan.real_len == kept.len()`, `plan.padded_len`
//!    is the next multiple of the width, and the bucketed run is
//!    run-to-run deterministic.

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
use riir_infer_core::tokengate::{PAD_KEY_BIAS, pad_bias_rows_into};
use riir_infer_core::transformer::{KeyBiasLayout, attention_heads_parallel_bias};

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
    // Deterministic token ids in [0, vocab).
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

/// Fresh run state with an explicit prefill capacity (the bucketed shape
/// runs at `padded_len` rows — the context must be sized for the PADDING).
fn fresh_run(r: &Rig, cap: usize) -> (HybridCache, HybridForwardScratch, PrefillContext, Vec<f32>) {
    let cache = HybridCache::with_layer_types(&r.cfg, &r.weights.layer_types);
    let scratch = HybridForwardScratch::new(&r.cfg);
    let ctx = PrefillContext::new(&r.cfg, cap);
    let logits = vec![0.0f32; r.cfg.vocab_size];
    (cache, scratch, ctx, logits)
}

/// The case scorer: all-ones, with an optional poisoned victim slot (drops
/// exactly that one mid-prompt token when `keep_prunable` covers the rest).
fn make_scorer(victim: Option<usize>) -> impl FnMut(&[f32], Option<&[f32]>, &mut [f32]) {
    move |_hidden: &[f32], _mass: Option<&[f32]>, scores: &mut [f32]| {
        scores.fill(1.0);
        if let Some(v) = victim {
            scores[v] = -1e9;
        }
    }
}

fn run_pruned(
    r: &Rig,
    cap: usize,
    prune_at: usize,
    keep_prunable: usize,
    bucket: Option<usize>,
    victim: Option<usize>,
) -> (HybridCache, PrunedPrefillPlan, Vec<f32>) {
    let (mut cache, mut scratch, mut ctx, mut logits) = fresh_run(r, cap);
    let mut plan = PrunedPrefillPlan::new(prune_at, N_SINK, keep_prunable);
    plan.bucket = bucket;
    let mut scorer = make_scorer(victim);
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
        None,
        None,
        &mut scorer,
    );
    (cache, plan, logits)
}

#[test]
fn bucketed_matches_unpadded_byte_for_byte_at_every_width() {
    let r = rig(0x6220_0101);
    let n_kv = r.cfg.n_kv_head * r.cfg.head_dim;
    // (keep_prunable, victim): K = full keeps all 24; the poisoned case keeps
    // scaffold(4) + 5 prunable + last = 10 real rows → 54 pad rows at w=64.
    let cases = [(usize::MAX, None), (5, Some(12))];

    for prune_at in [0usize, 1] {
        for (keep, victim) in cases {
            // The unpadded incumbent (T2.1 shape).
            let (cache_u, plan_u, logits_u) = run_pruned(&r, T, prune_at, keep, None, victim);
            let real_len = plan_u.kept.len();

            for width in [64usize, 128] {
                let padded = real_len.div_ceil(width) * width;
                let (cache_b, plan_b, logits_b) =
                    run_pruned(&r, padded, prune_at, keep, Some(width), victim);

                assert_eq!(plan_b.real_len, real_len, "real_len bookkeeping");
                assert_eq!(plan_b.padded_len, padded, "padded_len bookkeeping");
                assert_eq!(plan_b.kept, plan_u.kept, "same selection");
                synth::assert_bits_eq(
                    &logits_b,
                    &logits_u,
                    &format!("logits (w={width}, L={prune_at}, k={keep})"),
                );

                // EVERY layer's GDN state: pads never enter the recurrence —
                // the state is the unpadded run's, byte for byte, at ANY width.
                for layer in 0..r.weights.layers.len() {
                    synth::assert_bits_eq(
                        &cache_b.deltanet_state.recurrent_states[layer],
                        &cache_u.deltanet_state.recurrent_states[layer],
                        &format!("GDN recurrent layer {layer} (w={width}, L={prune_at}, k={keep})"),
                    );
                    synth::assert_bits_eq(
                        &cache_b.deltanet_state.conv_states[layer],
                        &cache_u.deltanet_state.conv_states[layer],
                        &format!("GDN conv layer {layer} (w={width}, L={prune_at}, k={keep})"),
                    );
                }

                // Attention KV: real rows identical. Pad slots (real_len..)
                // hold the DISCARDED rows' own evolution — a pad query's
                // attention over real values feeds o_proj back into its own
                // hidden row, so slots are nonzero from the second attention
                // layer above L. The law is FINITE (never NaN/inf — nothing
                // downstream can poison) + UNREACHABLE (real queries are
                // causally bounded below real_len; law 3 pins the bias arm).
                for layer in 1..r.weights.layers.len() {
                    let key = &cache_b.kv_cache.layers[layer].key;
                    let val = &cache_b.kv_cache.layers[layer].value;
                    synth::assert_bits_eq(
                        &key[..real_len * n_kv],
                        &cache_u.kv_cache.layers[layer].key[..real_len * n_kv],
                        &format!("KV key rows (layer {layer}, w={width}, L={prune_at}, k={keep})"),
                    );
                    synth::assert_bits_eq(
                        &val[..real_len * n_kv],
                        &cache_u.kv_cache.layers[layer].value[..real_len * n_kv],
                        &format!(
                            "KV value rows (layer {layer}, w={width}, L={prune_at}, k={keep})"
                        ),
                    );
                    for slot in real_len..padded {
                        assert!(
                            key[slot * n_kv..(slot + 1) * n_kv]
                                .iter()
                                .all(|v| v.is_finite()),
                            "pad key slot {slot} must be finite (layer {layer}, w={width})"
                        );
                        assert!(
                            val[slot * n_kv..(slot + 1) * n_kv]
                                .iter()
                                .all(|v| v.is_finite()),
                            "pad value slot {slot} must be finite (layer {layer}, w={width})"
                        );
                    }
                }

                // Determinism: an identical bucketed run reproduces every bit.
                let (cache_b2, _, logits_b2) =
                    run_pruned(&r, padded, prune_at, keep, Some(width), victim);
                synth::assert_bits_eq(
                    &logits_b2,
                    &logits_b,
                    &format!("bucketed determinism (w={width}, L={prune_at}, k={keep})"),
                );
                synth::assert_bits_eq(
                    &cache_b2.deltanet_state.recurrent_states[r.weights.layers.len() - 1],
                    &cache_b.deltanet_state.recurrent_states[r.weights.layers.len() - 1],
                    &format!("bucketed state determinism (w={width})"),
                );
            }
        }
    }
}

#[test]
fn pad_bias_table_makes_full_window_equal_the_causal_window_bit_for_bit() {
    // The T2.4 fixed-shape premise at kernel level: one query row, a cache
    // of real rows + zero pad rows, the pad-bias table over the full padded
    // window — bit-identical to the causal bound at real_len with no bias.
    // Both t_n values sit far below PARALLEL_HEADS_MIN_SEQ (512), so both
    // calls take the SAME (sequential) kernel arm — like-for-like.
    let cfg = synth::hybrid_config();
    let n_head = cfg.n_head;
    let n_kv_head = cfg.n_kv_head;
    let kvd = cfg.n_kv_head * cfg.head_dim;
    let hd = cfg.head_dim;
    let real_len = 24usize;
    let padded_len = 64usize;
    let mut sm = synth::SplitMix(0x6220_0102);

    let mut key = vec![0.0f32; padded_len * kvd];
    let mut val = vec![0.0f32; padded_len * kvd];
    sm.stream_into(&mut key[..real_len * kvd]);
    sm.stream_into(&mut val[..real_len * kvd]);
    // Slots real_len.. are already zeros — the pad rows the bucketed prefill
    // writes (law 2's premise, reproduced here directly).
    let q = sm.stream_vec(n_head * hd);
    let mut out_a = vec![0.0f32; n_head * hd];
    let mut out_b = vec![0.0f32; n_head * hd];
    let scores = vec![0.0f32; n_head * cfg.block_size];

    // (a) causal window at real_len, no bias (the incumbent dispatch).
    let mut scores_a = scores.clone();
    unsafe {
        attention_heads_parallel_bias(
            &q,
            &key,
            &val,
            &mut out_a,
            &mut scores_a,
            n_head,
            n_kv_head,
            kvd,
            hd,
            real_len,
            1.0 / (hd as f32).sqrt(),
            0.0,
            cfg.block_size,
            None,
        );
    }

    // (b) full padded window + the pad-bias table (the fixed-shape arm).
    let mut table = vec![0.0f32; n_head * padded_len];
    pad_bias_rows_into(&mut table, n_head, real_len, padded_len);
    // Table shape: slots < real_len inert, pads dead.
    assert_eq!(table[0], 0.0);
    assert_eq!(table[real_len], PAD_KEY_BIAS);
    assert_eq!(
        table[(n_head - 1) * padded_len + padded_len - 1],
        PAD_KEY_BIAS
    );
    let mut scores_b = scores.clone();
    unsafe {
        attention_heads_parallel_bias(
            &q,
            &key,
            &val,
            &mut out_b,
            &mut scores_b,
            n_head,
            n_kv_head,
            kvd,
            hd,
            padded_len,
            1.0 / (hd as f32).sqrt(),
            0.0,
            cfg.block_size,
            Some((
                &table,
                KeyBiasLayout {
                    head_stride: padded_len,
                },
            )),
        );
    }

    synth::assert_bits_eq(&out_b, &out_a, "full-window + pad bias ≡ causal window");
}

#[test]
fn pad_bias_rows_into_writes_the_dead_zone_and_refuses_short_buffers() {
    let mut table = vec![7.5f32; 3 * 64];
    pad_bias_rows_into(&mut table, 3, 10, 64);
    for h in 0..3 {
        let row = &table[h * 64..(h + 1) * 64];
        assert!(row[..10].iter().all(|v| *v == 0.0), "kept zone inert");
        assert!(row[10..].iter().all(|v| *v == PAD_KEY_BIAS), "pads dead");
    }
    // Exact width (zero pads) stays all-inert.
    let mut exact = vec![1.0f32; 2 * 64];
    pad_bias_rows_into(&mut exact, 2, 64, 64);
    assert!(exact.iter().all(|v| *v == 0.0));

    let mut short = vec![0.0f32; 3 * 64 - 1];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pad_bias_rows_into(&mut short, 3, 10, 64);
    }));
    assert!(result.is_err(), "short buffer must refuse loud");
}
