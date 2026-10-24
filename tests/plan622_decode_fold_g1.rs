//! Plan 622 T2.3 — the decode-time constant fold's G1 gate battery.
//!
//! Pins, on the synthetic [DeltaNet, Attention, Attention] hybrid (the
//! shared `common/plan622_synth.rs` kit):
//!
//! 1. **K = full is a dispatch identity, decode side too** — a K=full
//!    pruned prefill continued by [`forward_qwen_deltanet_pruned`] (gate
//!    `None`) produces logits, every layer's GDN state, and the KV rows
//!    BIT-IDENTICAL to the incumbent `prefill_qwen_deltanet_into` +
//!    `forward_qwen_deltanet` loop, at both prune placements.
//! 2. **Prefix-band invariance under decode** — with tokens pruned, the
//!    `<= L` band's GDN states and attention KV rows still match the FULL
//!    decode run byte-for-byte after every decode step (the band saw the
//!    full prompt and identical appends; pruning cannot reach it), while
//!    the logits move (the pruned band's cache is a subset — the lane is
//!    not a silent no-op).
//! 3. **The fold's identity + movement** — a gate whose bias rows are all
//!    zero and whose scales are all exactly 1.0 is byte-identical to gate
//!    `None` (the armed-but-inert identity: the ≡1 skip + zero mask);
//!    arming a real (+12 bias on slot 0, scale 2.0 on head 0) moves the
//!    logits and stays run-to-run deterministic.
//! 4. **Appended-key law** — flipping an APPENDED slot's bias row away
//!    from zero changes the decode logits (the mask covers the appended
//!    region; staying zero is the CONSTRUCTOR contract, Phase 3's to
//!    honor — this cell proves the consumer reads it).
//! 5. **T2.2 composition** — a bucketed pruned prefill continued by the
//!    pruned decode is byte-identical to the unpadded pruned prefill
//!    continued the same way (the cut-back contract: decode overwrites the
//!    pad slots and never reads past them).

#![cfg(all(
    feature = "tokengate",
    feature = "deltanet_inference",
    not(target_arch = "wasm32")
))]

#[path = "common/plan622_synth.rs"]
mod synth;

use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, PrunedPrefillPlan, forward_qwen_deltanet,
    forward_qwen_deltanet_pruned, prefill_qwen_deltanet_into, prefill_qwen_deltanet_pruned_into,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokengate::{PrunedDecodeGate, PrunedDecodeLayout};

const T: usize = 24;
const N_SINK: usize = 4;
const DECODE_TOKENS: usize = 3;

struct Rig {
    cfg: riir_infer_core::types::Config,
    weights: riir_infer_core::deltanet::weights::QwenDeltaNetWeights,
    rope: RopeFreqTable,
    tokens: Vec<usize>,
    decode: Vec<usize>,
}

fn rig(seed: u64) -> Rig {
    let cfg = synth::hybrid_config();
    let weights = synth::synth_hybrid_weights(&cfg, seed);
    let rope = RopeFreqTable::new(cfg.rope_theta, cfg.head_dim);
    let mut sm = synth::SplitMix(seed);
    let tokens: Vec<usize> = (0..T)
        .map(|_| (sm.next_f32().abs() * cfg.vocab_size as f32) as usize % cfg.vocab_size)
        .collect();
    let decode: Vec<usize> = (0..DECODE_TOKENS)
        .map(|_| (sm.next_f32().abs() * cfg.vocab_size as f32) as usize % cfg.vocab_size)
        .collect();
    Rig {
        cfg,
        weights,
        rope,
        tokens,
        decode,
    }
}

/// Fresh run state; `cap` sizes the prefill context (bucketed runs pad).
fn fresh_run(r: &Rig, cap: usize) -> (HybridCache, HybridForwardScratch, PrefillContext) {
    let cache = HybridCache::with_layer_types(&r.cfg, &r.weights.layer_types);
    let scratch = HybridForwardScratch::new(&r.cfg);
    let ctx = PrefillContext::new(&r.cfg, cap);
    (cache, scratch, ctx)
}

/// The incumbent continuation: full prefill + `forward_qwen_deltanet` steps.
fn run_incumbent(r: &Rig) -> (HybridCache, Vec<Vec<f32>>) {
    let (mut cache, mut scratch, mut ctx) = fresh_run(r, T);
    let mut logits = vec![0.0f32; r.cfg.vocab_size];
    prefill_qwen_deltanet_into(
        &r.weights,
        &r.cfg,
        &mut cache,
        &r.tokens,
        &mut scratch,
        &r.rope,
        &mut ctx,
        &mut logits,
        None,
    );
    let mut steps = vec![logits.clone()];
    let mut x = vec![0.0f32; r.cfg.vocab_size.max(r.cfg.n_embd)];
    for (step, &tok) in r.decode.iter().enumerate() {
        let out = forward_qwen_deltanet(
            &mut x,
            &r.weights,
            &mut cache,
            tok,
            T + step,
            &r.cfg,
            &mut scratch,
            &r.rope,
        );
        steps.push(out.to_vec());
    }
    (cache, steps)
}

/// The pruned continuation: pruned prefill + `forward_qwen_deltanet_pruned`.
#[allow(clippy::too_many_arguments)]
fn run_pruned(
    r: &Rig,
    prune_at: usize,
    keep_prunable: usize,
    bucket: Option<usize>,
    gate: Option<&PrunedDecodeGate<'_>>,
) -> (HybridCache, PrunedPrefillPlan, Vec<Vec<f32>>) {
    let cap = match bucket {
        Some(w) => {
            let kept = if keep_prunable == usize::MAX {
                T
            } else {
                N_SINK + keep_prunable + 1
            };
            kept.div_ceil(w) * w
        }
        None => T,
    };
    let (mut cache, mut scratch, mut ctx) = fresh_run(r, cap);
    let mut logits = vec![0.0f32; r.cfg.vocab_size];
    let mut plan = PrunedPrefillPlan::new(prune_at, N_SINK, keep_prunable);
    plan.bucket = bucket;
    // Poison slot 12 when the budget actually prunes (same victim as T2.1's
    // movement cell); all-ones when K = full.
    let mut scorer = move |_hidden: &[f32], scores: &mut [f32]| {
        scores.fill(1.0);
        if keep_prunable != usize::MAX {
            scores[12] = -1e9;
        }
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
        &mut scorer,
    );
    let mut steps = vec![logits.clone()];
    let layout = PrunedDecodeLayout::from_plan(T, &plan);
    let mut x = vec![0.0f32; r.cfg.vocab_size.max(r.cfg.n_embd)];
    for (step, &tok) in r.decode.iter().enumerate() {
        let out = forward_qwen_deltanet_pruned(
            &mut x,
            &r.weights,
            &mut cache,
            tok,
            &layout,
            step,
            gate,
            &r.cfg,
            &mut scratch,
            &r.rope,
        );
        steps.push(out.to_vec());
    }
    (cache, plan, steps)
}

#[test]
fn pruned_kfull_decode_is_bit_identical_to_the_incumbent_decode() {
    let r = rig(0x6220_0201);
    let (cache_i, steps_i) = run_incumbent(&r);
    for prune_at in [0usize, 1] {
        let (cache_p, plan, steps_p) = run_pruned(&r, prune_at, usize::MAX, None, None);
        assert_eq!(plan.real_len, T, "K = full keeps everything");
        for (s, (a, b)) in steps_i.iter().zip(steps_p.iter()).enumerate() {
            synth::assert_bits_eq(b, a, &format!("decode step {s} logits (L={prune_at})"));
        }
        // Full state equality after the decode steps, both bands.
        for layer in 0..r.weights.layers.len() {
            synth::assert_bits_eq(
                &cache_p.deltanet_state.recurrent_states[layer],
                &cache_i.deltanet_state.recurrent_states[layer],
                &format!("GDN recurrent layer {layer} (K=full decode, L={prune_at})"),
            );
            synth::assert_bits_eq(
                &cache_p.deltanet_state.conv_states[layer],
                &cache_i.deltanet_state.conv_states[layer],
                &format!("GDN conv layer {layer} (K=full decode, L={prune_at})"),
            );
        }
        let n_kv = r.cfg.n_kv_head * r.cfg.head_dim;
        let rows = (T + DECODE_TOKENS) * n_kv;
        for layer in 1..r.weights.layers.len() {
            synth::assert_bits_eq(
                &cache_p.kv_cache.layers[layer].key[..rows],
                &cache_i.kv_cache.layers[layer].key[..rows],
                &format!("KV key layer {layer} (K=full decode, L={prune_at})"),
            );
        }
    }
}

#[test]
fn pruned_decode_keeps_the_prefix_band_identical_and_moves_the_logits() {
    let r = rig(0x6220_0202);
    let (cache_i, steps_i) = run_incumbent(&r);
    let prune_at = 1usize;
    let (cache_p, plan, steps_p) = run_pruned(&r, prune_at, T - N_SINK - 2, None, None);
    assert_eq!(plan.real_len, T - 1, "dropped exactly the victim");

    // The logits move (the pruned band is a strict subset).
    assert_ne!(
        steps_p.last().unwrap(),
        steps_i.last().unwrap(),
        "pruned decode must move the final logits"
    );

    // Prefix-band invariance under decode: layer 0 (GDN, <= L) state and
    // layer 1 (attention, <= L) KV rows match the full run exactly, after
    // every decode step's writes.
    synth::assert_bits_eq(
        &cache_p.deltanet_state.recurrent_states[0],
        &cache_i.deltanet_state.recurrent_states[0],
        "GDN layer 0 state (prefix band, post-decode)",
    );
    synth::assert_bits_eq(
        &cache_p.deltanet_state.conv_states[0],
        &cache_i.deltanet_state.conv_states[0],
        "GDN layer 0 conv (prefix band, post-decode)",
    );
    let n_kv = r.cfg.n_kv_head * r.cfg.head_dim;
    let rows = (T + DECODE_TOKENS) * n_kv;
    synth::assert_bits_eq(
        &cache_p.kv_cache.layers[1].key[..rows],
        &cache_i.kv_cache.layers[1].key[..rows],
        "KV layer 1 rows (prefix band, post-decode)",
    );
    synth::assert_bits_eq(
        &cache_p.kv_cache.layers[1].value[..rows],
        &cache_i.kv_cache.layers[1].value[..rows],
        "KV layer 1 values (prefix band, post-decode)",
    );

    // Determinism: a second identical run reproduces every step.
    let (_, _, steps_p2) = run_pruned(&r, prune_at, T - N_SINK - 2, None, None);
    for (s, (a, b)) in steps_p.iter().zip(steps_p2.iter()).enumerate() {
        synth::assert_bits_eq(b, a, &format!("pruned decode determinism, step {s}"));
    }
}

#[test]
fn armed_inert_gate_is_byte_identical_and_armed_gate_moves_and_is_deterministic() {
    let r = rig(0x6220_0203);
    let prune_at = 0usize;
    let keep = 5usize;
    let kept_rows = N_SINK + keep + 1; // 10
    let head_stride = kept_rows + DECODE_TOKENS; // room for the appended keys
    let n_head = r.cfg.n_head;
    let layers = r.weights.layers.len();

    let (_, _, steps_none) = run_pruned(&r, prune_at, keep, None, None);

    // (a) Armed-but-inert: zero bias + all-1 scales ≡ gate None, byte for
    // byte (the ≡1 skip + zero mask are the incumbent).
    let zero_bias = vec![0.0f32; layers * n_head * head_stride];
    let ones_scale = vec![1.0f32; layers * n_head];
    let inert = PrunedDecodeGate {
        bias: &zero_bias,
        head_stride,
        scale: Some(&ones_scale),
    };
    let (_, _, steps_inert) = run_pruned(&r, prune_at, keep, None, Some(&inert));
    for (s, (a, b)) in steps_none.iter().zip(steps_inert.iter()).enumerate() {
        synth::assert_bits_eq(b, a, &format!("inert gate ≡ None, step {s}"));
    }

    // (b) Armed: +12 on slot 0 and scale 2.0 on head 0 of the LAST layer —
    // the logits move, and a rerun reproduces every bit.
    let mut bias = vec![0.0f32; layers * n_head * head_stride];
    let mut scale = vec![1.0f32; layers * n_head];
    let last = layers - 1;
    bias[last * n_head * head_stride] = 12.0;
    scale[last * n_head] = 2.0;
    let armed = PrunedDecodeGate {
        bias: &bias,
        head_stride,
        scale: Some(&scale),
    };
    let (_, _, steps_armed) = run_pruned(&r, prune_at, keep, None, Some(&armed));
    assert_ne!(
        steps_armed.last().unwrap(),
        steps_none.last().unwrap(),
        "an armed gate must move the final logits"
    );
    let (_, _, steps_armed2) = run_pruned(&r, prune_at, keep, None, Some(&armed));
    for (s, (a, b)) in steps_armed.iter().zip(steps_armed2.iter()).enumerate() {
        synth::assert_bits_eq(b, a, &format!("armed gate determinism, step {s}"));
    }

    // (c) Scale-only arm (bias stays zero): moves too — the s_eff fold is
    // its own surface, not a passenger of the mask.
    let mut scale_only = vec![1.0f32; layers * n_head];
    scale_only[last * n_head] = 2.0;
    let gate_scale_only = PrunedDecodeGate {
        bias: &zero_bias,
        head_stride,
        scale: Some(&scale_only),
    };
    let (_, _, steps_scale) = run_pruned(&r, prune_at, keep, None, Some(&gate_scale_only));
    assert_ne!(
        steps_scale.last().unwrap(),
        steps_none.last().unwrap(),
        "a scale-only gate must move the final logits"
    );

    // (d) Appended-key law: flipping APPENDED slot kept_rows' bias (the
    // first decode token's slot) changes decode step 0's logits — the mask
    // covers the appended region; staying zero is the constructor contract.
    let mut appended = bias.clone();
    appended[last * n_head * head_stride + kept_rows] = 5.0;
    let gate_appended = PrunedDecodeGate {
        bias: &appended,
        head_stride,
        scale: Some(&scale),
    };
    let (_, _, steps_appended) = run_pruned(&r, prune_at, keep, None, Some(&gate_appended));
    assert_ne!(
        steps_appended[1], steps_armed[1],
        "appended slot 0's bias must be read at decode step 0"
    );
}

#[test]
fn bucketed_prefill_continues_identically_through_the_pruned_decode() {
    // T2.2 × T2.3 composition: the bucketed prefill's pads are cut back by
    // the decode contract (decode overwrites them, never reads past), so
    // bucketed + pruned decode ≡ unpadded + pruned decode, byte for byte.
    let r = rig(0x6220_0204);
    let prune_at = 0usize;
    let keep = 5usize;
    let (cache_u, plan_u, steps_u) = run_pruned(&r, prune_at, keep, None, None);
    for width in [64usize, 128] {
        let (cache_b, plan_b, steps_b) = run_pruned(&r, prune_at, keep, Some(width), None);
        assert_eq!(plan_b.real_len, plan_u.real_len);
        assert_eq!(plan_b.padded_len, plan_u.real_len.div_ceil(width) * width);
        for (s, (a, b)) in steps_u.iter().zip(steps_b.iter()).enumerate() {
            synth::assert_bits_eq(b, a, &format!("bucketed decode step {s} (w={width})"));
        }
        // GDN state equality after the decode steps (pads never entered).
        for layer in 0..r.weights.layers.len() {
            synth::assert_bits_eq(
                &cache_b.deltanet_state.recurrent_states[layer],
                &cache_u.deltanet_state.recurrent_states[layer],
                &format!("GDN recurrent layer {layer} (bucketed decode, w={width})"),
            );
        }
    }
}
