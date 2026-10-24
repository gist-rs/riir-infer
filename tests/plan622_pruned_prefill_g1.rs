//! Plan 622 T2.1 — the pruned-prefill CPU reference's G1 gate battery.
//!
//! Pins, on the synthetic [DeltaNet, Attention, Attention] hybrid (the
//! T1.3 walk's fixture kit, shared via `common/plan622_synth.rs`):
//!
//! 1. **K = full is a dispatch identity** — the pruned path with every
//!    token kept produces logits, GDN states, and attention KV rows
//!    BIT-IDENTICAL to `prefill_qwen_deltanet_into`, at BOTH prune
//!    placements (L = the DeltaNet mixer, L = an Attention mixer — the
//!    placement rule says the gate may sit at any mixer type).
//! 2. **State split** — pruning at L leaves every layer `<= L` GDN state
//!    byte-identical to the full run at ANY K (the compaction happens
//!    strictly after L), while layers `> L` move.
//! 3. **Movement** — dropping a real mid-prompt token changes the logits
//!    (the lane is not a silent no-op) and stays deterministic run-to-run.
//! 4. **Selection** — the kept set is exactly scaffold ∪ top-K by score,
//!    scaffold always survives (the `n_sink` + last-token law).
//!
//! The decode-continuation seam is deliberately NOT pinned here: after a
//! pruned prefill the per-layer cache fill diverges (T rows in layers
//! `<= L`, K' rows above), which is the serving quartet's T2.3/T2.4
//! surface — see `prefill_qwen_deltanet_pruned_into`'s doc.

#![cfg(all(
    feature = "tokengate",
    feature = "deltanet_inference",
    not(target_arch = "wasm32")
))]

#[path = "common/plan622_synth.rs"]
mod synth;

use riir_infer_core::deltanet::forward::PrunedPrefillPlan;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, prefill_qwen_deltanet_into,
    prefill_qwen_deltanet_pruned_into,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokengate::select_kept_into;

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

fn fresh_run(r: &Rig) -> (HybridCache, HybridForwardScratch, PrefillContext, Vec<f32>) {
    let cache = HybridCache::with_layer_types(&r.cfg, &r.weights.layer_types);
    let scratch = HybridForwardScratch::new(&r.cfg);
    let ctx = PrefillContext::new(&r.cfg, T);
    let logits = vec![0.0f32; r.cfg.vocab_size];
    (cache, scratch, ctx, logits)
}

/// All-ones scorer: no token is distinguished, selection keeps everything
/// when k covers the prunable count.
fn all_ones(_hidden: &[f32], scores: &mut [f32]) {
    scores.fill(1.0);
}

/// Scores the token at `victim` lowest, everything else 1.0 — drops exactly
/// one mid-prompt token (never a scaffold position: the victim sits above
/// `N_SINK` and below `T - 1`).
fn poison(victim: usize) -> impl FnMut(&[f32], &mut [f32]) {
    move |_hidden: &[f32], scores: &mut [f32]| {
        scores.fill(1.0);
        scores[victim] = -1e9;
    }
}

#[test]
fn pruned_kfull_is_bit_identical_to_the_full_prefill_at_both_mixer_kinds() {
    let r = rig(0x6220_0001);
    // L = 0 (the DeltaNet mixer) and L = 1 (an Attention mixer): the
    // placement rule says the gate reads any mixer type's output.
    for prune_at in [0usize, 1] {
        // Full prefill (the incumbent path, no bias).
        let (mut cache_f, mut scratch_f, mut ctx_f, mut logits_f) = fresh_run(&r);
        prefill_qwen_deltanet_into(
            &r.weights,
            &r.cfg,
            &mut cache_f,
            &r.tokens,
            &mut scratch_f,
            &r.rope,
            &mut ctx_f,
            &mut logits_f,
            None,
        );

        // Pruned path at K = full (k >= prunable count → everything kept).
        let (mut cache_p, mut scratch_p, mut ctx_p, mut logits_p) = fresh_run(&r);
        let mut plan = PrunedPrefillPlan::new(prune_at, N_SINK, usize::MAX);
        prefill_qwen_deltanet_pruned_into(
            &r.weights,
            &r.cfg,
            &mut cache_p,
            &r.tokens,
            &mut scratch_p,
            &r.rope,
            &mut ctx_p,
            &mut logits_p,
            None,
            &mut plan,
            &mut all_ones,
        );

        synth::assert_bits_eq(&logits_p, &logits_f, "logits (K=full identity)");
        assert_eq!(plan.kept, (0..T).collect::<Vec<usize>>(), "kept = all");
        synth::assert_bits_eq(
            &cache_p.deltanet_state.recurrent_states[0],
            &cache_f.deltanet_state.recurrent_states[0],
            "GDN recurrent state (layer 0)",
        );
        synth::assert_bits_eq(
            &cache_p.deltanet_state.conv_states[0],
            &cache_f.deltanet_state.conv_states[0],
            "GDN conv state (layer 0)",
        );
        // Attention KV rows: every layer wrote the same slots with the same
        // roped K (the K=full gather is a pure row move).
        for layer in 1..r.weights.layers.len() {
            synth::assert_bits_eq(
                &cache_p.kv_cache.layers[layer].key[..T * r.cfg.n_kv_head * r.cfg.head_dim],
                &cache_f.kv_cache.layers[layer].key[..T * r.cfg.n_kv_head * r.cfg.head_dim],
                &format!("KV key layer {layer} (K=full identity)"),
            );
        }
    }
}

#[test]
fn pruned_keeps_prefix_state_byte_identical_and_moves_the_rest() {
    let r = rig(0x6220_0002);
    let victim = 12; // strictly above N_SINK=4, strictly below T-1=23
    let prune_at = 1usize; // an attention mixer at L; layer 0 (GDN) stays full

    let (mut cache_f, mut scratch_f, mut ctx_f, mut logits_f) = fresh_run(&r);
    prefill_qwen_deltanet_into(
        &r.weights,
        &r.cfg,
        &mut cache_f,
        &r.tokens,
        &mut scratch_f,
        &r.rope,
        &mut ctx_f,
        &mut logits_f,
        None,
    );

    let (mut cache_p, mut scratch_p, mut ctx_p, mut logits_p) = fresh_run(&r);
    let mut plan = PrunedPrefillPlan::new(prune_at, N_SINK, T - N_SINK - 2); // drop exactly 1
    prefill_qwen_deltanet_pruned_into(
        &r.weights,
        &r.cfg,
        &mut cache_p,
        &r.tokens,
        &mut scratch_p,
        &r.rope,
        &mut ctx_p,
        &mut logits_p,
        None,
        &mut plan,
        &mut poison(victim),
    );

    // Selection law: scaffold + last + every prunable except the victim.
    let expected: Vec<usize> = (0..T).filter(|&p| p != victim).collect();
    assert_eq!(plan.kept, expected, "kept = all but the victim");
    assert_eq!(plan.scores[victim], -1e9, "the scorer ran at layer L");

    // State split: layer 0 (<= L) GDN state byte-identical to the full run —
    // pruning cannot touch what L already consumed.
    synth::assert_bits_eq(
        &cache_p.deltanet_state.recurrent_states[0],
        &cache_f.deltanet_state.recurrent_states[0],
        "GDN recurrent state layer 0 (split law: prefix state identical)",
    );

    // Movement: the logits differ (the lane is not a silent no-op) and the
    // layer-2 attention KV rows hold only T-1 live rows worth of writes
    // (slot 0 of layer 2's cache carries the FIRST kept token — its roped K
    // is identical to the full run's slot 0 because kept[0] = 0).
    assert_ne!(
        logits_p, logits_f,
        "dropping a real token must move the logits"
    );
    let kvd = r.cfg.n_kv_head * r.cfg.head_dim;
    synth::assert_bits_eq(
        &cache_p.kv_cache.layers[2].key[..kvd],
        &cache_f.kv_cache.layers[2].key[..kvd],
        "layer-2 KV slot 0 (kept[0] = 0 roped at the same original position)",
    );

    // Determinism: a second identical run reproduces every bit.
    let (mut cache_p2, mut scratch_p2, mut ctx_p2, mut logits_p2) = fresh_run(&r);
    let mut plan2 = PrunedPrefillPlan::new(prune_at, N_SINK, T - N_SINK - 2);
    prefill_qwen_deltanet_pruned_into(
        &r.weights,
        &r.cfg,
        &mut cache_p2,
        &r.tokens,
        &mut scratch_p2,
        &r.rope,
        &mut ctx_p2,
        &mut logits_p2,
        None,
        &mut plan2,
        &mut poison(victim),
    );
    synth::assert_bits_eq(&logits_p2, &logits_p, "run-to-run determinism");
}

#[test]
fn selection_laws_at_the_prefill_surface() {
    // Scaffold survives even when the scorer poisons it; k = 0 keeps only
    // scaffold + last; the kept set feeds the compaction (kept.len() rows
    // survive, first kept row = kept[0]).
    let r = rig(0x6220_0003);
    let (mut cache_p, mut scratch_p, mut ctx_p, mut logits_p) = fresh_run(&r);
    let mut plan = PrunedPrefillPlan::new(1, N_SINK, 0); // scaffold-only
    let mut scaffold_poisoner = |_hidden: &[f32], scores: &mut [f32]| {
        scores.fill(-1e9);
        // "Best" prunable score sits at 5; scaffold still wins structurally.
        scores[5] = 1.0;
    };
    prefill_qwen_deltanet_pruned_into(
        &r.weights,
        &r.cfg,
        &mut cache_p,
        &r.tokens,
        &mut scratch_p,
        &r.rope,
        &mut ctx_p,
        &mut logits_p,
        None,
        &mut plan,
        &mut scaffold_poisoner,
    );
    let mut want = Vec::new();
    select_kept_into(&plan.scores, N_SINK, 0, &mut want);
    assert_eq!(plan.kept, want);
    assert_eq!(plan.kept.len(), N_SINK + 1, "sinks + last, nothing else");
    assert_eq!(plan.kept.last(), Some(&(T - 1)), "the last token survives");
    assert!(!plan.kept.contains(&5), "k=0 keeps no prunable token");
}
