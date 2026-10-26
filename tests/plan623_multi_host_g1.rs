#![cfg(all(feature = "multi_host", feature = "deltanet_ternary_inference"))]
//! Plan 623 Phase 1 — the multi-host G1 battery.
//!
//! **G1 composition gate:** stage(0..k) ∘ stage(k..n) == the 1-box full
//! forward, BIT-IDENTICAL, every k, both archs:
//!
//! 1. Ternary (qwen35-deltanet): the synthetic Bonsai-2 contract
//!    (`common/synth_bonsai2.rs`, 4 layers — 3 DeltaNet + 1 Attention),
//!    split at every k in 1..4, a continuous decode loop per arm (prefill
//!    + steps), logits compared by bits.
//! 2. Gemma 2: directly-constructed f32 weights (softcapped, RoPE, GQA),
//!    split at every k in 1..4, same capture shape.
//! 3. State-segment round trip (T1.3): encode the cache range to the wire,
//!    restore into a FRESH cache mid-run, continue — logits bit-identical.
//! 4. ActivationFrame round trip: f32 exact by bits; bf16 exact THROUGH
//!    ITS OWN ROUNDING (encode → decode == encode(encode) idempotence),
//!    header fields survive; decode reuses the caller buffer (no realloc
//!    after warm-up).
//! 5. Handshake negatives: mismatched arch / geometry / artifact / dtype /
//!    non-tiling ranges refuse.

#[path = "common/synth_bonsai2.rs"]
mod synth;

use synth::*;

use katgpt_types::rng::Rng;
use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_range,
};
use riir_infer_core::deltanet::{HybridCache, HybridForwardScratch};
use riir_infer_core::multi_host::{
    ActivationFrame, ActivationFrameHeader, FrameDtype, HybridStateSegment, LayerRange,
    StageHandshake,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::transformer::{ForwardContext, forward_gemma2, forward_gemma2_range};
use riir_infer_core::types::{Config, DeltaNetLayerType};

const PROMPT: [usize; 5] = [7, 13, 42, 100, 5];
const DECODE_STEPS: usize = 4;

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i)
}

/// The synthetic Bonsai-2 contract (unfolded — rotation `None`), one file
/// (the issue-028 fixture shape, local to this battery).
fn synth_single_bytes() -> Vec<u8> {
    build_gguf(&synth_metadata(false), &synth_tensors_bonsai2())
}

fn assert_bits_equal(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length diverged");
    for (i, (u, v)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            u.to_bits(),
            v.to_bits(),
            "{label}: logit {i} diverged ({} vs {})",
            u,
            v
        );
    }
}

/// The 1-box continuous loop — last prefill logits + every decode step's
/// logits (the reference arm).
fn ternary_full_run(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let layer_types = weights.layer_types.clone();
    let mut cache = HybridCache::with_layer_types(config, &layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let rope = RopeFreqTable::new(
        config.rope_theta,
        riir_infer_core::deltanet::effective_rotary_dim(config),
    );
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    let mut prefill_last = Vec::new();
    for (pos, tok) in PROMPT.iter().enumerate() {
        let l = forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            *tok,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        prefill_last = l[..config.vocab_size].to_vec();
    }
    let mut decode_logits = Vec::new();
    let mut next = argmax(&prefill_last);
    for i in 0..DECODE_STEPS {
        let pos = PROMPT.len() + i;
        let l = forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            next,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        let l = l[..config.vocab_size].to_vec();
        next = argmax(&l);
        decode_logits.push(l);
    }
    (prefill_last, decode_logits)
}

/// The two-stage composed loop: stage A runs layers [0..k) with cache_a,
/// hands the hidden vector to stage B which runs [k..n) with cache_b.
/// `split_wire` (Some) makes the handoff go THROUGH the ActivationFrame
/// f32 encode/decode — the wire posture, not a buffer borrow.
fn ternary_split_run(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    k: usize,
    split_wire: bool,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let layer_types = weights.layer_types.clone();
    let mut cache_a = HybridCache::with_layer_types(config, &layer_types);
    let mut cache_b = HybridCache::with_layer_types(config, &layer_types);
    let mut scratch_a = HybridForwardScratch::new(config);
    let mut scratch_b = HybridForwardScratch::new(config);
    let rope = RopeFreqTable::new(
        config.rope_theta,
        riir_infer_core::deltanet::effective_rotary_dim(config),
    );
    let n = config.n_embd;
    let mut x_a = vec![0.0f32; config.vocab_size.max(n)];
    let mut x_b = vec![0.0f32; config.vocab_size.max(n)];
    let range_a = LayerRange::new(0, k as u32);
    let range_b = LayerRange::new(k as u32, config.n_layer as u32);
    let mut wire: Vec<u8> = Vec::new();
    let mut wire_decode: Vec<f32> = Vec::new();

    let run_stage_pair = |x_a: &mut Vec<f32>,
                          x_b: &mut Vec<f32>,
                          token: usize,
                          pos: usize,
                          cache_a: &mut HybridCache,
                          cache_b: &mut HybridCache,
                          scratch_a: &mut HybridForwardScratch,
                          scratch_b: &mut HybridForwardScratch,
                          wire: &mut Vec<u8>,
                          wire_decode: &mut Vec<f32>| {
        let hidden = forward_qwen_deltanet_ternary_range(
            x_a, weights, cache_a, token, pos, config, scratch_a, &rope, range_a,
        );
        if split_wire {
            wire.clear();
            ActivationFrame::encode_into(0, pos as u32, 1, &hidden[..n], FrameDtype::F32, wire);
            let header =
                ActivationFrame::decode_into(wire, wire_decode).expect("wire frame decode");
            assert_eq!(header.pos, pos as u32, "frame pos survives the wire");
            x_b[..n].copy_from_slice(&wire_decode[..n]);
        } else {
            x_b[..n].copy_from_slice(&hidden[..n]);
        }
        let logits = forward_qwen_deltanet_ternary_range(
            x_b, weights, cache_b, token, pos, config, scratch_b, &rope, range_b,
        );
        logits[..config.vocab_size].to_vec()
    };

    let mut prefill_last = Vec::new();
    for (pos, tok) in PROMPT.iter().enumerate() {
        prefill_last = run_stage_pair(
            &mut x_a,
            &mut x_b,
            *tok,
            pos,
            &mut cache_a,
            &mut cache_b,
            &mut scratch_a,
            &mut scratch_b,
            &mut wire,
            &mut wire_decode,
        );
    }
    let mut decode_logits = Vec::new();
    let mut next = argmax(&prefill_last);
    for i in 0..DECODE_STEPS {
        let pos = PROMPT.len() + i;
        let l = run_stage_pair(
            &mut x_a,
            &mut x_b,
            next,
            pos,
            &mut cache_a,
            &mut cache_b,
            &mut scratch_a,
            &mut scratch_b,
            &mut wire,
            &mut wire_decode,
        );
        next = argmax(&l);
        decode_logits.push(l);
    }
    (prefill_last, decode_logits)
}

#[test]
fn ternary_stage_split_is_bit_identical_at_every_k() {
    let path = write_tmp("riir_623_single", &synth_single_bytes());
    let (config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&path)
            .expect("load synthetic single file");
    let _ = std::fs::remove_file(&path);

    let (full_pre, full_dec) = ternary_full_run(&config, &weights);
    for k in 1..config.n_layer {
        let (pre, dec) = ternary_split_run(&config, &weights, k, false);
        assert_bits_equal(&format!("prefill split@{k}"), &full_pre, &pre);
        for (step, (f, s)) in full_dec.iter().zip(&dec).enumerate() {
            assert_bits_equal(&format!("decode step {step} split@{k}"), f, s);
        }
    }
}

/// The same gate with the stage handoff THROUGH the ActivationFrame f32
/// wire encode/decode — the composition must survive the wire byte-exact.
#[test]
fn ternary_stage_split_through_wire_frame_is_bit_identical() {
    let path = write_tmp("riir_623_wire", &synth_single_bytes());
    let (config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&path)
            .expect("load synthetic single file");
    let _ = std::fs::remove_file(&path);

    let (full_pre, full_dec) = ternary_full_run(&config, &weights);
    let (pre, dec) = ternary_split_run(&config, &weights, config.n_layer / 2, true);
    assert_bits_equal("prefill wire-split", &full_pre, &pre);
    for (step, (f, s)) in full_dec.iter().zip(&dec).enumerate() {
        assert_bits_equal(&format!("decode step {step} wire-split"), f, s);
    }
}

/// T1.3 — the state-segment round trip: after prefill, encode stage A's
/// range state to the wire, restore into a FRESH cache, continue decoding
/// — logits bit-identical to the uninterrupted run. This is the
/// out-of-process handoff simulation (the Issue-028 gap closed).
#[test]
fn ternary_state_segment_round_trip_is_bit_identical() {
    let path = write_tmp("riir_623_seg", &synth_single_bytes());
    let (config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&path)
            .expect("load synthetic single file");
    let _ = std::fs::remove_file(&path);
    let layer_types = weights.layer_types.clone();
    let n = config.n_embd;
    let k = config.n_layer / 2;

    let rope = RopeFreqTable::new(
        config.rope_theta,
        riir_infer_core::deltanet::effective_rotary_dim(&config),
    );
    let mk_caches = || {
        (
            HybridCache::with_layer_types(&config, &layer_types),
            HybridCache::with_layer_types(&config, &layer_types),
            HybridForwardScratch::new(&config),
            HybridForwardScratch::new(&config),
        )
    };
    let range_a = LayerRange::new(0, k as u32);
    let range_b = LayerRange::new(k as u32, config.n_layer as u32);

    // Arm R (reference): plain split, continuous caches.
    let (mut ca, mut cb, mut sa, mut sb) = mk_caches();
    let mut x_a = vec![0.0f32; config.vocab_size.max(n)];
    let mut x_b = vec![0.0f32; config.vocab_size.max(n)];
    let mut prefill_last = Vec::new();
    for (pos, tok) in PROMPT.iter().enumerate() {
        let hidden = forward_qwen_deltanet_ternary_range(
            &mut x_a, &weights, &mut ca, *tok, pos, &config, &mut sa, &rope, range_a,
        );
        x_b[..n].copy_from_slice(&hidden[..n]);
        prefill_last = forward_qwen_deltanet_ternary_range(
            &mut x_b, &weights, &mut cb, *tok, pos, &config, &mut sb, &rope, range_b,
        )[..config.vocab_size]
            .to_vec();
    }
    let mut ref_logits = Vec::new();
    let mut next = argmax(&prefill_last);
    for i in 0..DECODE_STEPS {
        let pos = PROMPT.len() + i;
        let hidden = forward_qwen_deltanet_ternary_range(
            &mut x_a, &weights, &mut ca, next, pos, &config, &mut sa, &rope, range_a,
        );
        x_b[..n].copy_from_slice(&hidden[..n]);
        let l = forward_qwen_deltanet_ternary_range(
            &mut x_b, &weights, &mut cb, next, pos, &config, &mut sb, &rope, range_b,
        )[..config.vocab_size]
            .to_vec();
        next = argmax(&l);
        ref_logits.push(l);
    }

    // Arm M (migrated): IDENTICAL until the prefill→decode boundary, where
    // stage A's range state goes through encode → FRESH cache → restore.
    let (mut ca, mut cb, mut sa, mut sb) = mk_caches();
    let mut x_a = vec![0.0f32; config.vocab_size.max(n)];
    let mut x_b = vec![0.0f32; config.vocab_size.max(n)];
    let mut prefill_last = Vec::new();
    for (pos, tok) in PROMPT.iter().enumerate() {
        let hidden = forward_qwen_deltanet_ternary_range(
            &mut x_a, &weights, &mut ca, *tok, pos, &config, &mut sa, &rope, range_a,
        );
        x_b[..n].copy_from_slice(&hidden[..n]);
        prefill_last = forward_qwen_deltanet_ternary_range(
            &mut x_b, &weights, &mut cb, *tok, pos, &config, &mut sb, &rope, range_b,
        )[..config.vocab_size]
            .to_vec();
    }
    let mut seg: Vec<u8> = Vec::new();
    HybridStateSegment::encode_into(&ca, &layer_types, range_a, &mut seg)
        .expect("encode stage A segment");
    let mut ca2 = HybridCache::with_layer_types(&config, &layer_types);
    HybridStateSegment::restore_into(&seg, &mut ca2, &layer_types, range_a)
        .expect("restore stage A segment");
    let mut mig_logits = Vec::new();
    let mut next = argmax(&prefill_last);
    for i in 0..DECODE_STEPS {
        let pos = PROMPT.len() + i;
        let hidden = forward_qwen_deltanet_ternary_range(
            &mut x_a, &weights, &mut ca2, next, pos, &config, &mut sa, &rope, range_a,
        );
        x_b[..n].copy_from_slice(&hidden[..n]);
        let l = forward_qwen_deltanet_ternary_range(
            &mut x_b, &weights, &mut cb, next, pos, &config, &mut sb, &rope, range_b,
        )[..config.vocab_size]
            .to_vec();
        next = argmax(&l);
        mig_logits.push(l);
    }
    assert_eq!(ref_logits.len(), mig_logits.len());
    for (step, (r, m)) in ref_logits.iter().zip(&mig_logits).enumerate() {
        assert_bits_equal(&format!("migrated decode step {step}"), r, m);
    }
}

// ── Gemma 2 arm ────────────────────────────────────────────────────────────

fn gemma_test_config() -> Config {
    let mut c = Config::micro();
    c.n_layer = 4;
    c.model_arch = katgpt_types::ModelArchitecture::Gemma2;
    c.use_rope = true;
    c.rope_theta = 10000.0;
    c.attn_logit_softcapping = 50.0;
    c.final_logit_softcapping = 30.0;
    c
}

/// Deterministic weights in [-0.12, 0.12]-ish band — no RNG dependency, the
/// same values every run and every arm.
fn det_weight(i: usize, salt: usize) -> f32 {
    (((i * 7 + salt * 13) % 25) as f32 - 12.0) * 0.01
}

fn gemma_test_weights(config: &Config) -> riir_infer_core::gemma_layer::GemmaTransformerWeights {
    use riir_infer_core::gemma_layer::{GemmaLayerWeights, GemmaTransformerWeights};
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let kvd = riir_infer_core::types::kv_dim(config);
    let mlp = config.mlp_hidden;
    let mut layers = Vec::with_capacity(config.n_layer);
    for li in 0..config.n_layer {
        layers.push(GemmaLayerWeights {
            attn_wq: (0..q_dim * n).map(|i| det_weight(i, li + 1)).collect(),
            attn_wk: (0..kvd * n).map(|i| det_weight(i, li + 2)).collect(),
            attn_wv: (0..kvd * n).map(|i| det_weight(i, li + 3)).collect(),
            attn_wo: (0..n * n).map(|i| det_weight(i, li + 4)).collect(),
            gate_proj: (0..mlp * n).map(|i| det_weight(i, li + 5)).collect(),
            up_proj: (0..mlp * n).map(|i| det_weight(i, li + 6)).collect(),
            down_proj: (0..n * mlp).map(|i| det_weight(i, li + 7)).collect(),
            input_norm: (0..n).map(|i| 0.5 + det_weight(i, li + 8).abs()).collect(),
            post_attn_norm: (0..n).map(|i| 0.5 + det_weight(i, li + 9).abs()).collect(),
            pre_mlp_norm: (0..n).map(|i| 0.5 + det_weight(i, li + 10).abs()).collect(),
            post_mlp_norm: (0..n).map(|i| 0.5 + det_weight(i, li + 11).abs()).collect(),
        });
    }
    GemmaTransformerWeights {
        wte: (0..config.vocab_size * n)
            .map(|i| det_weight(i, 99))
            .collect(),
        final_norm: (0..n).map(|i| 0.5 + det_weight(i, 98).abs()).collect(),
        layers,
    }
}

#[test]
fn gemma2_stage_split_is_bit_identical_at_every_k() {
    let config = gemma_test_config();
    let weights = gemma_test_weights(&config);
    let n = config.n_embd;

    // Full run (the 1-box reference): forward_gemma2 = embed + ALL layers
    // + head (the production entry).
    let mut ctx_full = ForwardContext::new(&config);
    let mut cache_full = riir_infer_core::transformer::MultiLayerKVCache::new(&config);
    forward_gemma2(
        &mut ctx_full,
        &weights,
        &mut cache_full,
        PROMPT[0],
        0,
        &config,
    );
    let full_first = ctx_full.logits[..config.vocab_size].to_vec();

    let run_split = |k: usize| -> Vec<f32> {
        let mut ctx_a = ForwardContext::new(&config);
        let mut ctx_b = ForwardContext::new(&config);
        let mut cache_a = riir_infer_core::transformer::MultiLayerKVCache::new(&config);
        let mut cache_b = riir_infer_core::transformer::MultiLayerKVCache::new(&config);

        // Stage A first token: embed (forward_gemma2's step 1) + layers [0..k).
        let tok_off = PROMPT[0] * n;
        let embed_scale = (n as f32).sqrt();
        for i in 0..n {
            ctx_a.x[i] = weights.wte[tok_off + i] * embed_scale;
        }
        forward_gemma2_range(
            &mut ctx_a,
            &weights,
            &mut cache_a,
            0,
            &config,
            LayerRange::new(0, k as u32),
        );
        // Handoff + stage B [k..n).
        ctx_b.x[..n].copy_from_slice(&ctx_a.x[..n]);
        forward_gemma2_range(
            &mut ctx_b,
            &weights,
            &mut cache_b,
            0,
            &config,
            LayerRange::new(k as u32, config.n_layer as u32),
        );
        ctx_b.logits[..config.vocab_size].to_vec()
    };

    for k in 1..config.n_layer {
        let split_first = run_split(k);
        assert_bits_equal(
            &format!("gemma2 first-token split@{k}"),
            &full_first,
            &split_first,
        );
    }
}

// ── Wire-type units ────────────────────────────────────────────────────────

#[test]
fn activation_frame_f32_round_trip_is_exact_and_buffer_reuse_is_alloc_free() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.37 - 41.0).collect();
    let mut wire = Vec::new();
    ActivationFrame::encode_into(3, 17, 1, &values, FrameDtype::F32, &mut wire);
    assert_eq!(
        wire.len(),
        ActivationFrameHeader::WIRE_PREFIX + values.len() * 4
    );

    let mut out: Vec<f32> = Vec::new();
    let header = ActivationFrame::decode_into(&wire, &mut out).expect("decode");
    assert_eq!(header.seq_id, 3);
    assert_eq!(header.pos, 17);
    assert_eq!(header.n_tok, 1);
    assert_eq!(header.dtype, FrameDtype::F32 as u8);
    assert_eq!(out.len(), values.len());
    for (i, (u, v)) in out.iter().zip(&values).enumerate() {
        assert_eq!(u.to_bits(), v.to_bits(), "f32 wire round trip bit {i}");
    }

    // Steady-state decode: a second decode into the SAME buffer must not
    // grow its allocation (resize reuses capacity).
    let cap_before = out.capacity();
    let _ = ActivationFrame::decode_into(&wire, &mut out).expect("redecode");
    assert_eq!(out.capacity(), cap_before, "decode_into reallocated");
}

#[test]
fn activation_frame_bf16_is_exact_through_its_own_rounding() {
    // The bf16 lane's contract: decode(encode(v)) == decode(encode(v)) —
    // idempotent through RNE — NOT bit-identity with v (the lossy law).
    let values: Vec<f32> = [0.0, 1.0, -2.5, 1e-8, 1234.5678, -0.000_007].to_vec();
    let mut wire = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, &values, FrameDtype::Bf16, &mut wire);
    assert_eq!(
        wire.len(),
        ActivationFrameHeader::WIRE_PREFIX + values.len() * 2
    );

    let mut once = Vec::new();
    ActivationFrame::decode_into(&wire, &mut once).expect("bf16 decode");
    let mut wire2 = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, &once, FrameDtype::Bf16, &mut wire2);
    let mut twice = Vec::new();
    ActivationFrame::decode_into(&wire2, &mut twice).expect("bf16 redecode");
    for (i, (a, b)) in once.iter().zip(&twice).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bf16 lane not idempotent at {i}");
    }
    // And a value exactly representable in bf16 survives exactly.
    let exact = vec![0.5f32, -128.0, 1.0 / 1024.0];
    let mut w = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, &exact, FrameDtype::Bf16, &mut w);
    let mut out = Vec::new();
    ActivationFrame::decode_into(&w, &mut out).expect("exact bf16 decode");
    for (i, (u, v)) in out.iter().zip(&exact).enumerate() {
        assert_eq!(u.to_bits(), v.to_bits(), "bf16-exact value {i}");
    }
}

#[test]
fn activation_frame_refuses_garbage() {
    // Bad magic.
    let mut wire = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, &[1.0], FrameDtype::F32, &mut wire);
    wire[0] = b'X';
    assert!(ActivationFrame::decode_into(&wire, &mut Vec::new()).is_err());

    // Truncated payload (header claims more than present).
    let mut wire = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, &[1.0, 2.0, 3.0, 4.0], FrameDtype::F32, &mut wire);
    wire.truncate(wire.len() - 5);
    assert!(ActivationFrame::decode_into(&wire, &mut Vec::new()).is_err());

    // Reserved byte non-zero (newer writer) refuses fail-closed.
    let mut wire = Vec::new();
    ActivationFrame::encode_into(0, 0, 1, &[1.0], FrameDtype::F32, &mut wire);
    // Header layout: magic[0..4] seq_id[4..8] pos[8..12] n_tok[12..14]
    // dtype[14] reserved[15] payload_len[16..20]. Flip the reserved byte.
    wire[15] = 1;
    assert!(ActivationFrame::decode_into(&wire, &mut Vec::new()).is_err());
}

#[test]
fn handshake_refuses_every_mismatch_class() {
    let mk = |arch: &str, geom: u8, art: u8, lo: u32, hi: u32, dtype: FrameDtype| StageHandshake {
        arch: arch.to_string(),
        geometry_fingerprint: [geom; 32],
        artifact_blake3: [art; 32],
        range: LayerRange::new(lo, hi),
        dtype,
    };
    let n = 4u32;
    let a = mk("qwen35", 1, 1, 0, 2, FrameDtype::F32);
    let b_ok = mk("qwen35", 1, 1, 2, 4, FrameDtype::F32);
    // verify_pair is called downstream.verify_pair(upstream): b verifies a.
    assert!(b_ok.verify_pair(&a, n).is_ok());

    let wrong_arch = mk("gemma2", 1, 1, 2, 4, FrameDtype::F32);
    assert!(wrong_arch.verify_pair(&a, n).is_err());
    let wrong_geom = mk("qwen35", 2, 1, 2, 4, FrameDtype::F32);
    assert!(wrong_geom.verify_pair(&a, n).is_err());
    let wrong_art = mk("qwen35", 1, 2, 2, 4, FrameDtype::F32);
    assert!(wrong_art.verify_pair(&a, n).is_err());
    let wrong_dtype = mk("qwen35", 1, 1, 2, 4, FrameDtype::Bf16);
    assert!(wrong_dtype.verify_pair(&a, n).is_err());
    let gap = mk("qwen35", 1, 1, 3, 4, FrameDtype::F32);
    assert!(gap.verify_pair(&a, n).is_err());
    let out_of_bounds = mk("qwen35", 1, 1, 2, 9, FrameDtype::F32);
    assert!(out_of_bounds.verify_pair(&a, n).is_err());
}

#[test]
fn layer_range_validates() {
    assert!(LayerRange::full(4).validate(4).is_ok());
    assert!(LayerRange::new(0, 4).validate(4).is_ok());
    assert!(LayerRange::new(2, 2).validate(4).is_err());
    assert!(LayerRange::new(3, 2).validate(4).is_err());
    assert!(LayerRange::new(0, 5).validate(4).is_err());
    assert!(LayerRange::new(1, 3).validate(4).is_ok());
    assert!(!LayerRange::new(1, 3).is_empty());
    assert!(LayerRange::new(3, 3).is_empty());
}

// Keep the synth kit's unused-but-pub surface referenced so the include
// stays honest (the kit over-serves by design).
#[test]
fn layer_types_fixture_shape() {
    let types = synth_layer_types();
    assert_eq!(types.len(), N_LAYER as usize);
    assert_eq!(types[3], DeltaNetLayerType::Attention);
}

fn synth_layer_types() -> Vec<DeltaNetLayerType> {
    (0..N_LAYER)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                DeltaNetLayerType::Attention
            } else {
                DeltaNetLayerType::DeltaNet
            }
        })
        .collect()
}

// Silence the unused-import lint for the Rng path (the fixture kit covers
// determinism without RNG; the import documents the alternative).
#[allow(unused)]
fn _rng_documentation() -> Rng {
    Rng::new(0)
}
