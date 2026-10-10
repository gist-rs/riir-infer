#![cfg(feature = "issue929_bonsai2_probe")]
//! katgpt-rs Issue 929 (Research 614, DiscoLoop) — Bonsai-2 loop-probe
//! machinery gates: synthetic tiny hybrid weights, NO model file (the
//! real-checkpoint run is the bin's job; these pin the WIRING).
//!
//! 1. `k1_loop_matches_stock_forward_bit_identical` — the load-bearing pin:
//!    at K=1 the looped answer-position path (embed → one stack pass → probe
//!    readout) must produce logits BIT-IDENTICAL to the stock
//!    `forward_qwen_deltanet_ternary` on the same token — the probe's glue
//!    (final norm → folded-head rotate → ternary head matvec) is the stock
//!    decode readout in the same order, never a second implementation. This
//!    is the α=0-equivalence class for the runner: a wiring drift here
//!    indicts the instrument, not the checkpoint (the Issue-022 T3.1 law).
//! 2. `k4_records_four_deterministic_rows` — K loops yield K probe rows with
//!    finite fields, and two fresh-cache runs agree exactly (the
//!    Bench-930 fresh-runtime discipline, pinned at the machinery level).
//! 3. `fixture_consumption_shape` — the katgpt-core §2 fixture meets this
//!    runner's consumption assumptions (both classes present, ASCII prompts
//!    — the BpeTokenizer's domain, bridge iff two-hop).

use katgpt_core::loop_alignment_probe::{FixtureSpec, generate_two_hop_fixture};
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch};
use riir_infer_core::deltanet::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::loop_probe::{
    LoopProbeScratch, embed_answer_token, looped_answer_position_probe,
};
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::{Config, DeltaNetLayerType};

/// Small hybrid config — dims are multiples of 128 (the ternary group size),
/// mirroring `ternary_forward`'s own test config. `n_layer` is derived from
/// the layer-type vec (the zeros constructor sizes both from the config;
/// a mismatch reds as an index panic, so never pin them apart).
fn small_config(layer_types: Vec<DeltaNetLayerType>) -> Config {
    let n_layer = layer_types.len();
    let mut config = Config::qwen_deltanet(n_layer, layer_types);
    config.vocab_size = 256;
    config.n_embd = 128;
    config.n_head = 1;
    config.n_kv_head = 1;
    config.head_dim = 128;
    config.mlp_hidden = 256;
    config.deltanet_linear_head_dim = 128;
    config.deltanet_linear_n_heads = 1;
    config.deltanet_linear_n_value_heads = 1;
    config
}

/// Zero weights with DISTINCT nonzero structure where the readout can see
/// it: per-token embedding rows (ternary + group scales), a non-unit
/// final_norm, and distinct lm_head rows — so logits differ per token and a
/// glue drift moves bits (all-zero logits would hide every wiring bug).
fn readable_weights(config: &Config) -> QwenDeltaNetTernaryWeights {
    let mut weights = QwenDeltaNetTernaryWeights::zeros(config);
    let n = config.n_embd;
    let vocab = config.vocab_size;
    for tok in 0..vocab {
        for c in 0..n {
            // Period-3 ternary pattern, phase-shifted per token.
            let v = match (c + tok) % 3 {
                0 => 1i8,
                1 => -1i8,
                _ => 0i8,
            };
            if v != 0 {
                weights.wte.set(tok, c, v);
                weights.lm_head.set(tok, c, v);
            }
        }
        for g in 0..weights.wte.groups_per_row {
            let scale = 0.5 + 0.01 * ((tok + g) % 7) as f32;
            weights.wte.set_scale(tok, g, scale);
            weights.lm_head.set_scale(tok, g, scale);
        }
    }
    for (i, v) in weights.final_norm.iter_mut().enumerate() {
        *v = 0.8 + 0.01 * (i % 11) as f32;
    }
    weights
}

fn fresh(
    config: &Config,
    weights: &QwenDeltaNetTernaryWeights,
) -> (HybridCache, HybridForwardScratch) {
    let cache = HybridCache::with_layer_types(config, &weights.layer_types);
    let scratch = HybridForwardScratch::new(config);
    (cache, scratch)
}

#[test]
fn k1_loop_matches_stock_forward_bit_identical() {
    use DeltaNetLayerType::*;
    let layer_types = vec![DeltaNet, Attention];
    let config = small_config(layer_types.clone());
    let weights = readable_weights(&config);
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let n = config.n_embd;
    let buf_len = config.vocab_size.max(n);

    // Prefill one token stock (both arms identical prefix).
    let tok0 = 7usize;
    let tok1 = 42usize;
    let mut x = vec![0.0f32; buf_len];
    let (mut cache, mut scratch) = fresh(&config, &weights);
    let stock = forward_qwen_deltanet_ternary(
        &mut x,
        &weights,
        &mut cache,
        tok0,
        0,
        &config,
        &mut scratch,
        &rope_freq,
    );
    let stock_prefix: Vec<u32> = stock.iter().map(|v| v.to_bits()).collect();

    // Arm A: stock forward at the answer position.
    let (mut cache_a, mut scratch_a) = fresh(&config, &weights);
    let mut xa = vec![0.0f32; buf_len];
    let _ = forward_qwen_deltanet_ternary(
        &mut xa,
        &weights,
        &mut cache_a,
        tok0,
        0,
        &config,
        &mut scratch_a,
        &rope_freq,
    );
    let logits_a = forward_qwen_deltanet_ternary(
        &mut xa,
        &weights,
        &mut cache_a,
        tok1,
        1,
        &config,
        &mut scratch_a,
        &rope_freq,
    );
    let bits_a: Vec<u32> = logits_a.iter().map(|v| v.to_bits()).collect();

    // Arm B: the looped machinery at K=1 (embed once → one stack pass →
    // probe readout). Same fresh-cache prefix first.
    let (mut cache_b, mut scratch_b) = fresh(&config, &weights);
    let mut xb = vec![0.0f32; buf_len];
    let _ = forward_qwen_deltanet_ternary(
        &mut xb,
        &weights,
        &mut cache_b,
        tok0,
        0,
        &config,
        &mut scratch_b,
        &rope_freq,
    );
    embed_answer_token(&mut xb, &weights, tok1);
    let mut probe_scratch = LoopProbeScratch::new(&config);
    let mut rows = Vec::new();
    let first = looped_answer_position_probe(
        &mut xb,
        &weights,
        &mut cache_b,
        &mut scratch_b,
        &rope_freq,
        &config,
        1,
        1,
        0,
        &mut probe_scratch,
        &mut rows,
    );
    let bits_b: Vec<u32> = probe_scratch.logits.iter().map(|v| v.to_bits()).collect();

    // The prefix run must be identical too (sanity: same weights, same
    // fresh caches ⇒ same deterministic forward).
    let (mut cache_p, mut scratch_p) = fresh(&config, &weights);
    let mut xp = vec![0.0f32; buf_len];
    let logits_p = forward_qwen_deltanet_ternary(
        &mut xp,
        &weights,
        &mut cache_p,
        tok0,
        0,
        &config,
        &mut scratch_p,
        &rope_freq,
    );
    let bits_p: Vec<u32> = logits_p.iter().map(|v| v.to_bits()).collect();
    assert_eq!(stock_prefix, bits_p, "stock prefix must be deterministic");

    assert_eq!(
        bits_a, bits_b,
        "K=1 looped readout must be BIT-IDENTICAL to the stock forward's logits"
    );
    assert_eq!(first, argmax_bits(&bits_b));
    assert_eq!(rows.len(), 1);
}

#[test]
fn k4_records_four_deterministic_rows() {
    use DeltaNetLayerType::*;
    let layer_types = vec![DeltaNet, Attention, DeltaNet];
    let config = small_config(layer_types.clone());
    let weights = readable_weights(&config);
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let n = config.n_embd;
    let buf_len = config.vocab_size.max(n);

    let run = |tokens: &[usize],
               k: usize|
     -> (
        Vec<riir_infer_core::deltanet::loop_probe::LoopProbeRow>,
        usize,
    ) {
        let (mut cache, mut scratch) = fresh(&config, &weights);
        let mut x = vec![0.0f32; buf_len];
        for (pos, &t) in tokens[..tokens.len() - 1].iter().enumerate() {
            let _ = forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                t,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
            );
        }
        embed_answer_token(&mut x, &weights, *tokens.last().unwrap());
        let mut probe_scratch = LoopProbeScratch::new(&config);
        let mut rows = Vec::new();
        let first = looped_answer_position_probe(
            &mut x,
            &weights,
            &mut cache,
            &mut scratch,
            &rope_freq,
            &config,
            tokens.len() - 1,
            k,
            5,
            &mut probe_scratch,
            &mut rows,
        );
        (rows, first)
    };

    let tokens = [3usize, 11, 29, 101];
    let (rows_a, first_a) = run(&tokens, 4);
    let (rows_b, first_b) = run(&tokens, 4);
    assert_eq!(rows_a.len(), 4, "K loops must yield K probe rows");
    assert_eq!(first_a, first_b, "fresh-cache runs must agree (argmax)");
    assert_eq!(rows_a, rows_b, "fresh-cache runs must agree (rows)");
    for (k, r) in rows_a.iter().enumerate() {
        assert!(r.margin.is_finite(), "loop {k} margin finite");
        assert!(r.cos_alignment.is_finite(), "loop {k} cos finite");
        assert!(
            (-1.0..=1.0).contains(&r.cos_alignment),
            "loop {k} cos in range"
        );
        assert!(
            (-1.0..=1.0).contains(&r.cos_bridge) && r.cos_bridge.is_finite(),
            "loop {k} cos_bridge in range (bridge=5 read)"
        );
    }
}

#[test]
fn fixture_consumption_shape() {
    let spec = FixtureSpec {
        facts_per_prompt: 6,
        two_hop_queries: 3,
        one_hop_queries: 2,
        ..FixtureSpec::default()
    };
    let fixture = generate_two_hop_fixture(spec);
    assert!(!fixture.items.is_empty(), "fixture must be non-empty");
    let two_hop = fixture.items.iter().filter(|i| i.is_two_hop).count();
    let one_hop = fixture.items.iter().filter(|i| !i.is_two_hop).count();
    assert!(
        two_hop > 0,
        "the hard class must be present (AUROC needs both)"
    );
    assert!(one_hop > 0, "the control class must be present");
    for item in &fixture.items {
        assert!(
            item.prompt.is_ascii(),
            "prompts must be ASCII (the BpeTokenizer domain)"
        );
        assert!(!item.answer.is_empty(), "answers must be non-empty");
        assert_eq!(
            item.bridge.is_empty(),
            !item.is_two_hop,
            "bridge iff two-hop"
        );
    }
}

/// Argmax over bit patterns (strict `>`, lowest index on ties) — matches
/// `loop_probe::argmax_of` semantics without re-borrowing f32 ordering.
fn argmax_bits(bits: &[u32]) -> usize {
    // All logits here are finite non-NaN (built from finite math), so the
    // f32 total order via bits only holds for non-negative values — compare
    // through f32 instead.
    let logits: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
    riir_infer_core::deltanet::loop_probe::argmax_of(&logits)
}
