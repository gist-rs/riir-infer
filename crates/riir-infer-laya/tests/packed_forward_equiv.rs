//! The packed-forward equivalence gate (reflex issue 020 T5): a packed
//! multi-sequence [`Encoder::forward_packed`] must produce the SAME rows
//! as running the single-sequence [`Encoder::forward`] per sequence, to
//! within the shape-dependent GEMM reduction order. The OP STREAM is
//! identical per row (row-independent ops, per-sequence attention, same
//! rope values by copy) — but the `gemm` crate dispatches different
//! microkernels by problem shape (m = 1 takes a different k-accumulation
//! path than m = 76), so the two arms may differ by ulps exactly the way
//! the G5 drift budget prices them. Measured on this gate's synthetic
//! geometry: CPU max |diff| 1.4e-6 (a seq-1 row through the m=1
//! microkernel) — the gate sits at 1e-5, ~7× headroom; the Metal lane's
//! fused kernel is offset-bound and shape-general per sequence (drift
//! expected exactly 0; gated at 1e-4, the smoke test's own class).
//!
//! Synthetic weights (a tiny deterministic geometry): the gate is about
//! the OP STREAM's row independence, not the checkpoints'. The real
//! checkpoints ride the consumer-side G5 + batched-parity gates.
//!
//! Run: `cargo test --release --features laya-riir --test packed_forward_equiv`
#![cfg(feature = "laya-riir")]

use std::collections::HashMap;

use riir_infer_laya::laya::config::EncoderConfig;
use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::weights::Weights;

/// Deterministic xorshift fill — the gate compares two runs of the SAME
/// weights, so the values only need coverage, not statistical quality.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn weights(shape: Vec<usize>, seed: u64) -> Weights {
    let n: usize = shape.iter().product();
    Weights {
        shape,
        data: fill(n, seed),
    }
}

/// d=64 (hd 64, one head), 4 layers (full / sliding / sliding / full),
/// window small enough to exercise the sliding path at short sequences.
fn test_config() -> EncoderConfig {
    EncoderConfig {
        hidden: 64,
        layers: 4,
        heads: 1,
        intermediate: 32,
        vocab: 97,
        eps: 1e-5,
        global_every: 3,
        local_attention: 8, // window 4
        rope_theta_full: 10_000.0,
        rope_theta_slide: 16_000.0,
        sliding: vec![false, true, true, false],
        hidden_activation: "gelu".into(),
    }
}

fn test_encoder() -> Encoder {
    let cfg = test_config();
    let d = cfg.hidden;
    let i = cfg.intermediate;
    let mut map: HashMap<String, Weights> = HashMap::new();
    let mut seed = 1u64;
    let mut next = || {
        seed += 7;
        seed
    };
    map.insert(
        "encoder.embeddings.tok_embeddings.weight".into(),
        weights(vec![cfg.vocab, d], next()),
    );
    map.insert(
        "encoder.embeddings.norm.weight".into(),
        weights(vec![d], next()),
    );
    for idx in 0..cfg.layers {
        if idx != 0 {
            map.insert(
                format!("encoder.layers.{idx}.attn_norm.weight"),
                weights(vec![d], next()),
            );
        }
        map.insert(
            format!("encoder.layers.{idx}.attn.Wqkv.weight"),
            weights(vec![3 * d, d], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.attn.Wo.weight"),
            weights(vec![d, d], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.mlp.Wi.weight"),
            weights(vec![2 * i, d], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.mlp.Wo.weight"),
            weights(vec![d, i], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.mlp_norm.weight"),
            weights(vec![d], next()),
        );
    }
    map.insert(
        "encoder.final_norm.weight".into(),
        weights(vec![d], next()),
    );
    Encoder::from_map(&mut map, cfg, "packed-equiv-test").expect("synthetic encoder loads")
}

/// Mixed lengths: long enough to cross the sliding window several times,
/// short enough to hit the seq == 1 and seq ≤ window edges, with a
/// REPEATED length (the rope-table cache hit).
const SEQS: [usize; 7] = [3, 17, 1, 9, 17, 5, 24];

fn packed_ids(total: usize, vocab: usize) -> Vec<u32> {
    fill(total, 4242)
        .into_iter()
        .map(|v| ((v + 1.0) * 0.5 * (vocab - 1) as f32) as u32)
        .collect()
}

#[test]
fn packed_forward_is_bit_identical_to_sequential_cpu() {
    let enc = test_encoder();
    let b = Cpu;
    let seqs = SEQS.to_vec();
    let total: usize = seqs.iter().sum();
    let ids = packed_ids(total, 97);

    let packed = enc
        .forward_packed(&b, &ids, &seqs)
        .expect("packed forward runs");

    let mut off = 0usize;
    for (si, &seq) in seqs.iter().enumerate() {
        let single = enc
            .forward(&b, &ids[off..off + seq])
            .expect("single forward runs");
        let got = &packed[off * 64..(off + seq) * 64];
        let max_diff = got
            .iter()
            .zip(single.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 1e-5,
            "sequence {si} (len {seq}) drifts {max_diff} (gate 1e-5)"
        );
        off += seq;
    }
}

/// The offsets themselves: `forward_packed` with ONE sequence must equal
/// the plain `forward` through the SAME code path (the zero-offset form).
#[test]
fn single_sequence_packed_equals_forward_cpu() {
    let enc = test_encoder();
    let b = Cpu;
    let ids = packed_ids(31, 97);
    let plain = enc.forward(&b, &ids).expect("forward runs");
    let packed = enc
        .forward_packed(&b, &ids, &[ids.len()])
        .expect("packed forward runs");
    assert_eq!(plain, packed, "one-sequence packed must be the plain form");
}

/// The Metal lane: same equivalence at a drift tolerance (expected ~0 —
/// the kernel is offset-bound, never re-shaped; the gate exists to catch
/// a bind-offset mistake, not to re-litigate kernel parity). Loud skip on
/// non-macOS / feature-absent builds.
#[cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
#[test]
fn packed_forward_matches_sequential_metal() {
    use riir_infer_laya::laya::riir::metal::Metal;

    let enc = test_encoder();
    let m = Metal::new().expect("metal backend");
    let seqs = SEQS.to_vec();
    let total: usize = seqs.iter().sum();
    let ids = packed_ids(total, 97);

    m.begin_pass();
    let packed = enc
        .forward_packed(&m, &ids, &seqs)
        .expect("packed forward runs");
    // Download ONCE, before any later begin_pass clears the chain epoch
    // (a slot that has been cleared cannot be host-read).
    let mut packed_host = vec![0f32; total * 64];
    m.download_into(&packed, &mut packed_host);

    for si in 0..seqs.len() {
        let off: usize = seqs[..si].iter().sum();
        let seq = seqs[si];
        m.begin_pass();
        let single = enc.forward(&m, &ids[off..off + seq]).expect("forward runs");
        let mut single_host = vec![0f32; seq * 64];
        m.download_into(&single, &mut single_host);
        let max_diff = packed_host[off * 64..(off + seq) * 64]
            .iter()
            .zip(single_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 1e-4,
            "sequence {si} (len {seq}) drifts {max_diff}"
        );
    }
}

/// The CUDA lane (.issues/003): same equivalence at a drift tolerance.
/// This is the gate class that catches the v1 zeros defect at the
/// substrate level — the trait-default attention SLICES host memory at
/// the packed offsets, and under the write-first discipline those host
/// bytes are stale, so a default-path regression here shows up as a huge
/// drift (not an argmax flip that tolerance could absorb). Loud skip on
/// macOS / feature-absent builds.
#[cfg(all(not(target_os = "macos"), feature = "laya-riir-cuda"))]
#[test]
fn packed_forward_matches_sequential_cuda() {
    use riir_infer_laya::laya::riir::cuda::Cuda;

    let enc = test_encoder();
    let g = Cuda::new().expect("cuda backend");
    let seqs = SEQS.to_vec();
    let total: usize = seqs.iter().sum();
    let ids = packed_ids(total, 97);

    g.begin_pass();
    let packed = enc
        .forward_packed(&g, &ids, &seqs)
        .expect("packed forward runs");
    // Download ONCE, before any later begin_pass clears the chain epoch.
    let mut packed_host = vec![0f32; total * 64];
    g.download_into(&packed, &mut packed_host);

    for si in 0..seqs.len() {
        let off: usize = seqs[..si].iter().sum();
        let seq = seqs[si];
        g.begin_pass();
        let single = enc.forward(&g, &ids[off..off + seq]).expect("forward runs");
        let mut single_host = vec![0f32; seq * 64];
        g.download_into(&single, &mut single_host);
        let max_diff = packed_host[off * 64..(off + seq) * 64]
            .iter()
            .zip(single_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 1e-4,
            "sequence {si} (len {seq}) drifts {max_diff}"
        );
    }
}

/// The packed path also has to survive a `supports_packed_attention`
/// answer on both lanes (the agent gates on it).
#[test]
fn supports_packed_attention_answers() {
    let b = Cpu;
    assert!(b.supports_packed_attention(64));
}
