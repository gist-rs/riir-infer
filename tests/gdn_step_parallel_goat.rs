//! Gated delta-rule decode step — head-parallel / row-fused variants vs the
//! serial four-pass kernel (`deltanet::forward::gated_deltanet_step_inplace`),
//! at the Bonsai-2 GDN shape: 48 V-heads × 128×128 state (3 MiB per layer,
//! 48 GDN layers per token).
//!
//! Every head (and every state row) is independent, and each variant runs the
//! same per-element operations in the same order, so G1 is BIT-IDENTITY.
//!
//! ```bash
//! cargo test --release --features deltanet_ternary_inference --test gdn_step_parallel_goat -- --include-ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "deltanet_ternary_inference")]

#[path = "../../katgpt-rs/tests/common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;

use riir_infer_core::deltanet::forward::{gated_deltanet_step_inplace, gated_deltanet_step_par_inplace};

const H: usize = 48;
const D: usize = 128;

fn fill(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

struct Step {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
}

fn step(seed: u64) -> Step {
    Step {
        q: fill(seed, H * D, 0.1),
        k: fill(seed + 1, H * D, 0.1),
        v: fill(seed + 2, H * D, 1.0),
        beta: fill(seed + 3, H, 0.5).into_iter().map(|b| 0.5 + b).collect(),
        decay: fill(seed + 4, H, 0.05).into_iter().map(|g| 0.95 + g).collect(),
    }
}

fn serial(st: &mut [f32], s: &Step, out: &mut [f32], kv: &mut [f32], dl: &mut [f32]) {
    gated_deltanet_step_inplace(&s.q, &s.k, &s.v, st, &s.beta, &s.decay, H, D, D, out, kv, dl);
}

fn par(st: &mut [f32], s: &Step, out: &mut [f32]) {
    gated_deltanet_step_par_inplace(&s.q, &s.k, &s.v, st, &s.beta, &s.decay, H, D, D, out);
}

#[test]
fn g1_head_parallel_step_is_bit_identical_over_a_token_stream() {
    let mut st_a = fill(9, H * D * D, 0.01);
    let mut st_b = st_a.clone();
    let (mut out_a, mut out_b) = (vec![0.0f32; H * D], vec![0.0f32; H * D]);
    let (mut kv, mut dl) = (vec![0.0f32; D], vec![0.0f32; D]);
    // 32 consecutive tokens: the recurrent state carries any drift forward.
    for t in 0..32u64 {
        let s = step(100 + t * 7);
        serial(&mut st_a, &s, &mut out_a, &mut kv, &mut dl);
        par(&mut st_b, &s, &mut out_b);
        assert!(out_a.iter().zip(&out_b).all(|(x, y)| x.to_bits() == y.to_bits()), "output differs at token {t}");
    }
    assert!(st_a.iter().zip(&st_b).all(|(x, y)| x.to_bits() == y.to_bits()), "state differs after 32 tokens");
}

#[test]
#[ignore = "wall-clock A/B; run alone with --release"]
fn g2_head_parallel_step_vs_serial_ab() {
    let steps: Vec<Step> = (0..8).map(|k| step(1000 + k * 11)).collect();
    let mut st_a = fill(9, H * D * D, 0.01);
    let mut st_b = st_a.clone();
    let (mut out_a, mut out_b) = (vec![0.0f32; H * D], vec![0.0f32; H * D]);
    let (mut kv, mut dl) = (vec![0.0f32; D], vec![0.0f32; D]);
    let sink = std::cell::Cell::new(0.0f32);
    let r = ab_timing::ab_median_ratio(
        31,
        200,
        50,
        |i| {
            serial(&mut st_a, black_box(&steps[i % steps.len()]), &mut out_a, &mut kv, &mut dl);
            sink.set(sink.get() + out_a[i % out_a.len()]);
        },
        |i| {
            par(&mut st_b, black_box(&steps[i % steps.len()]), &mut out_b);
            sink.set(sink.get() + out_b[i % out_b.len()]);
        },
    );
    r.report("gdn step 48x128x128 par/serial");
    let saved_ms = (r.a_ns_per_iter() - r.b_ns_per_iter()) * 48.0 / 1e6;
    println!("   projected per-token saving (48 GDN layers): {saved_ms:.2} ms");
    black_box(sink.get());
}
