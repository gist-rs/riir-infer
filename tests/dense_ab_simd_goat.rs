//! Dense a/b SIMD GOAT gate (PROMOTED 2026-10-13, the scalar arm removed) — the Dense (BF16→f32) a/b gate projections
//! through `katgpt_core::simd::simd_dot_f32` vs the serial scalar reduction.
//!
//! Bonsai 2 loads `ssm_alpha`/`ssm_beta` as Dense 48×5120 and the decode runs
//! both per GDN layer (48 layers) per token: 2 × 48 × 5120 × 48 ≈ 23.6M MACs
//! that the scalar arm executes as a latency-bound single-accumulator chain.
//!
//! Promotion record (M3 Max, AC, powermode 2): kernel 176.2 → 14.5 µs per
//! projection (interleaved median ratio 0.083); real Bonsai-2-27B PQ2_0,
//! `twt_perf_tg_pp` pp 128 / tg 64, scalar vs SIMD builds: 65/65 generated
//! tokens identical, tg 1.77–1.83 → 1.85–1.86 tok/s, pp 1.89–1.97 → 2.00–2.03
//! tok/s (the SIMD build ran at the HIGHER load avg, 7.7 vs 5.5).
//!
//! - G1 (default run): per-row error vs the scalar reference bounded by
//!   `1e-5 · Σ|w·x|` (the summation order differs, so not bit-identical).
//!   The real-model half (greedy-token identity) runs through the Bonsai-2
//!   decode bins with and without the feature.
//! - G2 (`#[ignore]`, `--release`): interleaved median-of-ratios, the shared
//!   katgpt-rs Issue-723 harness, input churn + sink consumed on both arms.
//!
//! ```bash
//! cargo test --release --features deltanet_ternary_inference --test dense_ab_simd_goat -- --include-ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "deltanet_ternary_inference")]

#[path = "../../katgpt-rs/tests/common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;

use riir_infer_core::deltanet::ternary_weights::GateProjWeights;

const ROWS: usize = 48;
const COLS: usize = 5120;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The pre-promotion Dense arm, verbatim (the reference; no longer shipped).
fn scalar_matvec(y: &mut [f32], data: &[f32], x: &[f32], rows: usize, cols: usize) {
    for (r, yr) in y[..rows].iter_mut().enumerate() {
        let row = &data[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for (&wv, &xv) in row.iter().zip(x[..cols].iter()) {
            acc += wv * xv;
        }
        *yr = acc;
    }
}

#[test]
fn g1_simd_dense_matches_the_scalar_arm_within_reduction_tolerance() {
    let data = fill(7, ROWS * COLS);
    let w = GateProjWeights::Dense(data.clone(), ROWS, COLS);
    for seed in 0..8u64 {
        let x = fill(100 + seed, COLS);
        let (mut got, mut want) = (vec![0.0f32; ROWS], vec![0.0f32; ROWS]);
        w.matvec_into(&mut got, &x);
        scalar_matvec(&mut want, &data, &x, ROWS, COLS);
        for r in 0..ROWS {
            let row = &data[r * COLS..(r + 1) * COLS];
            let mag: f32 = row.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let err = (got[r] - want[r]).abs();
            assert!(
                err <= 1e-5 * mag,
                "seed {seed} row {r}: simd {} scalar {} err {err} mag {mag}",
                got[r],
                want[r]
            );
        }
    }
}

#[test]
#[ignore = "wall-clock A/B; run alone with --release"]
fn g2_simd_dense_vs_scalar_ab() {
    let data = fill(7, ROWS * COLS);
    let w = GateProjWeights::Dense(data.clone(), ROWS, COLS);
    let xs: Vec<Vec<f32>> = (0..16).map(|k| fill(200 + k, COLS)).collect();
    let (mut ya, mut yb) = (vec![0.0f32; ROWS], vec![0.0f32; ROWS]);
    let sink = std::cell::Cell::new(0.0f32);
    let r = ab_timing::ab_median_ratio(
        31,
        200,
        50,
        |i| {
            scalar_matvec(
                &mut ya,
                black_box(&data),
                black_box(&xs[i % xs.len()]),
                ROWS,
                COLS,
            );
            sink.set(sink.get() + ya[i % ROWS]);
        },
        |i| {
            black_box(&w).matvec_into(&mut yb, black_box(&xs[i % xs.len()]));
            sink.set(sink.get() + yb[i % ROWS]);
        },
    );
    r.report("dense a/b 48x5120 simd/scalar");
    let per_token_saved_ms = (r.a_ns_per_iter() - r.b_ns_per_iter()) * 2.0 * 48.0 / 1e6;
    println!(
        "   projected per-token saving (2 projections x 48 GDN layers): {per_token_saved_ms:.2} ms"
    );
    black_box(sink.get());
    assert!(
        r.median < 1.0,
        "the SIMD arm must beat the scalar arm: median ratio {:.4}",
        r.median
    );
}
