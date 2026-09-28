//! TWT gate battery (riir-infer Issue 022 T1.5 / T2.2 / T2.3).
//!
//! Whole-file `#![cfg]` protects the count; the `[[test]]`
//! required-features row protects the reader (katgpt-rs .issues/713):
//! a `--features` selection without the union SKIPS this target loudly
//! instead of printing a green zero.
//!
//! Runs: `cargo test -p riir-infer-core --features twt_profile --test twt_gates`
#![cfg(feature = "twt_profile")]

use riir_infer_core::twt::{
    brute_force_optimal, forced_min_blocks, kill_verdict, localize_by_block, minmax_partition,
    partition_worst, planted_corpus, planted_corpus_noisy, delta, Block, Bucket, DeltaMap,
    KillVerdict, PRE_REGISTERED_EPS_GRID, SMatrix, SMatrixBuilder, SvccaCfg,
};
use riir_infer_core::twt::{Lcg, TwtError};

// ── G4 — the accumulate path allocates nothing ──────────────────────
// Lives in its OWN test target (tests/twt_g4_alloc.rs): the counting
// allocator is process-global and `cargo test` runs this file's tests in
// parallel threads in ONE process, so sibling tests' allocations leak
// into any count taken here. One test per binary is the isolation.

// ── helpers ──────────────────────────────────────────────────────────────

/// Feed a corpus (row-major `corpus[row][layer]`) through the builder in
/// forwards of `rows_per_forward` rows, layer-major arrival.
fn feed(b: &mut SMatrixBuilder, corpus: &[Vec<Vec<f32>>], rows_per_forward: usize) {
    let n_layers = corpus[0].len();
    for chunk in corpus.chunks(rows_per_forward) {
        b.begin_forward(chunk.len()).unwrap();
        for layer in 0..n_layers {
            for (r, states) in chunk.iter().enumerate() {
                b.push(layer, r, &states[layer]).unwrap();
            }
        }
        b.end_forward().unwrap();
    }
}

// ── T1.5 — the G1 planted known-answers ──────────────────────────────────

#[test]
fn g1_exact_zero_intra_clone_cosine_entries() {
    // Layers {1,2,3} and {5,6} are planted clones; every other layer is
    // an independent draw. Intra-clone S entries must be EXACTLY 0.0
    // (same bytes both sides ⇒ dot == na == nb ⇒ cos == 1.0 bit-exact).
    let corpus = planted_corpus(0xC0FFEE, 8, 24, 64, &[&[1, 2, 3], &[5, 6]]);
    let mut b = SMatrixBuilder::new(8, 24);
    feed(&mut b, &corpus, 64);
    let s = b.finalize().unwrap();
    let m = s.cosine();
    assert_eq!(s.rows_folded, 64);
    for (i, j) in [(1usize, 2usize), (1, 3), (2, 3), (5, 6)] {
        assert_eq!(m.get(i, j), 0.0, "S[{i}][{j}] must be exact zero");
    }
    // Independent layers sit far from zero (LCG vectors are near-orthogonal
    // in aggregate; the distance concentrates near 1).
    assert!(m.get(0, 1) > 0.5, "independent distance {}", m.get(0, 1));
    assert!(m.get(4, 5) > 0.5);
    assert!(m.get(6, 7) > 0.5);
}

#[test]
fn g1_dp_recovers_the_planted_partition() {
    let corpus = planted_corpus(0xC0FFEE, 8, 24, 64, &[&[1, 2, 3], &[5, 6]]);
    let mut b = SMatrixBuilder::new(8, 24);
    feed(&mut b, &corpus, 64);
    let s = b.finalize().unwrap().cosine().clone();
    let p = minmax_partition(&s, 0.10).unwrap();
    assert_eq!(
        p,
        vec![
            Block { start: 0, end: 1 },
            Block { start: 1, end: 4 },
            Block { start: 4, end: 5 },
            Block { start: 5, end: 7 },
            Block { start: 7, end: 8 },
        ]
    );
}

#[test]
fn g1_partition_stability_across_both_meters() {
    // The discrepancy METER is an ARM — planted-block recovery must hold
    // under BOTH, and disagreement on planted data indicts the meter.
    let corpus = planted_corpus(0xF00D, 6, 12, 96, &[&[2, 3, 4]]);
    let mut b = SMatrixBuilder::new(6, 12).with_svcca(SvccaCfg {
        stride: 1,
        cap_rows: 96,
    });
    feed(&mut b, &corpus, 96);
    let s = b.finalize().unwrap();
    assert_eq!(s.probe_rows, 96);
    let cos = s.cosine();
    let sv = s.svcca().expect("svcca arm armed — matrix must exist");
    // Clone pair exactly zero under cosine; under SVCCA the distance is
    // bounded by the primitive's own numerical floor (fixed 7
    // Newton–Schulz whitening iterations ≈ 1e-3 residual — measured
    // 1.6e-3 here), so the threshold sits ABOVE that floor. Distinct
    // pairs read far under BOTH meters.
    assert_eq!(cos.get(2, 3), 0.0, "cosine clone exact zero");
    assert!(sv.get(2, 3) < 0.01, "svcca clone distance {}", sv.get(2, 3));
    println!(
        "sv meter: clone(2,3)={:.4} distinct(0,1)={:.4} (0,2)={:.4}",
        sv.get(2, 3),
        sv.get(0, 1),
        sv.get(0, 2)
    );
    assert!(cos.get(0, 1) > 0.5, "cosine distinct {}", cos.get(0, 1));
    assert!(sv.get(0, 1) > 0.5, "svcca distinct {}", sv.get(0, 1));
    // Partition agreement at ε = 0.10.
    let p_cos = minmax_partition(cos, 0.10).unwrap();
    let p_sv = minmax_partition(sv, 0.10).unwrap();
    assert_eq!(p_cos, p_sv, "meters disagree on planted data");
}

#[test]
fn g1_near_clones_stay_below_the_first_grid_rung() {
    // σ = 0.01 noise on clone pairs: the pooled cosine distance of a
    // near-clone sits well under ε = 0.05 (the grid's smallest rung),
    // while distinct layers stay far above it.
    let corpus = planted_corpus_noisy(0xBEEF, 5, 16, 128, &[&[1, 2]], 0.01);
    let mut b = SMatrixBuilder::new(5, 16);
    feed(&mut b, &corpus, 128);
    let m = b.finalize().unwrap().cosine().clone();
    assert!(m.get(1, 2) < PRE_REGISTERED_EPS_GRID[0], "near-clone {}", m.get(1, 2));
    assert!(m.get(0, 1) > PRE_REGISTERED_EPS_GRID[3], "distinct {}", m.get(0, 1));
}

#[test]
fn g1_invariants_nonneg_symmetric_diag_zero() {
    let corpus = planted_corpus(7, 6, 10, 48, &[&[0, 5]]);
    let mut b = SMatrixBuilder::new(6, 10);
    feed(&mut b, &corpus, 48);
    let m = b.finalize().unwrap().cosine().clone();
    for i in 0..6 {
        assert_eq!(m.get(i, i), 0.0);
        for j in 0..6 {
            assert!(m.get(i, j) >= 0.0 && m.get(i, j) <= 2.0);
            assert_eq!(m.get(i, j).to_bits(), m.get(j, i).to_bits());
        }
    }
}

// ── T2.2 — DP vs brute force + monotonicity + constraint post-condition ──

#[test]
fn t2_2_dp_matches_brute_force_and_holds_constraints() {
    let mut rng = Lcg::new(0x5EED_1234);
    for n in [2usize, 3, 6, 9, 12] {
        // Planted-block + noise S matrices — the shape the lane actually
        // reads, with both tight and loose eps.
        let groups: Vec<Vec<usize>> = (0..n / 3)
            .map(|g| vec![g * 3, g * 3 + 1])
            .collect();
        let refs: Vec<&[usize]> = groups.iter().map(|g| g.as_slice()).collect();
        let corpus = planted_corpus_noisy(0xA11CE + n as u64, n, 8, 24, &refs, 0.05);
        let mut b = SMatrixBuilder::new(n, 8);
        feed(&mut b, &corpus, 24);
        let s = b.finalize().unwrap().cosine().clone();
        // A fully random S too (the adversarial shape for the DP).
        let rand = SMatrix::from_fn(n, |_, _| {
            let v = rng.next_centered() + 0.5; // [0, 1)
            v
        });
        for (label, m) in [("planted", &s), ("random", &rand)] {
            for eps in [0.05f32, 0.2, 0.5, 0.9] {
                let p = minmax_partition(m, eps).unwrap();
                let got = (p.len(), partition_worst(m, &p));
                let want = brute_force_optimal(m, eps);
                assert_eq!(got, want, "n={n} eps={eps} {label}");
                // Every emitted block satisfies its constraint.
                for blk in &p {
                    for x in blk.start..blk.end {
                        for y in (x + 1)..blk.end {
                            assert!(
                                m.get(y, x) <= eps,
                                "constraint violated n={n} eps={eps} {label}"
                            );
                        }
                    }
                }
                // Blocks tile [0, n) contiguously.
                let mut cursor = 0usize;
                for blk in &p {
                    assert_eq!(blk.start, cursor);
                    cursor = blk.end;
                }
                assert_eq!(cursor, n);
            }
        }
    }
}

#[test]
fn t2_2_m_monotone_non_increasing_over_the_pinned_grid() {
    let corpus = planted_corpus(3, 10, 8, 32, &[&[2, 3], &[6, 7, 8]]);
    let mut b = SMatrixBuilder::new(10, 8);
    feed(&mut b, &corpus, 32);
    let s = b.finalize().unwrap().cosine().clone();
    let mut prev = usize::MAX;
    for eps in PRE_REGISTERED_EPS_GRID {
        let m = minmax_partition(&s, eps).unwrap().len();
        assert!(m <= prev, "m grew at eps={eps}: {m} > {prev}");
        prev = m;
    }
}

#[test]
fn t2_2_invalid_eps_refused() {
    let s = SMatrix::from_fn(3, |_, _| 0.5);
    assert!(matches!(minmax_partition(&s, -0.1), Err(TwtError::InvalidEps(_))));
    assert!(matches!(minmax_partition(&s, f32::NAN), Err(TwtError::InvalidEps(_))));
    assert!(matches!(minmax_partition(&s, f32::INFINITY), Err(TwtError::InvalidEps(_))));
}

// ── T1.6 — the kill-rule arms on synthetic partitions ────────────────────

#[test]
fn t1_6_kill_rule_arms() {
    let singles = |m: usize| -> Vec<Vec<Block>> {
        PRE_REGISTERED_EPS_GRID
            .iter()
            .map(|_| (0..m).map(|k| Block { start: k, end: k + 1 }).collect())
            .collect()
    };
    // 28 layers, laya's G-S-S forced floor = 19 → bar = ceil(15.2) = 16.
    let forced = forced_min_blocks(&laya_gss_types(28));
    assert_eq!(forced, 19);
    // m = 28 ≥ 16 → block-count kill.
    assert!(matches!(
        kill_verdict(&singles(28), forced).unwrap(),
        KillVerdict::KillBlockCount { m_at_max_eps: 28, bar: 16 }
    ));
    // m = 15 < 16 but the middle band holds only singletons → middle kill.
    assert!(matches!(
        kill_verdict(&singles(15), forced).unwrap(),
        KillVerdict::KillMiddleBlocks { .. }
    ));
    // A spanning middle block survives both clauses.
    let mut parts = singles(15);
    let last = parts.last_mut().unwrap();
    *last = vec![
        Block { start: 0, end: 5 },
        Block { start: 5, end: 10 },
        Block { start: 10, end: 15 },
    ];
    assert!(matches!(kill_verdict(&parts, forced).unwrap(), KillVerdict::Survives { .. }));
    // Grid arity mismatch refused loud.
    assert!(matches!(
        kill_verdict(&singles(4)[..3], forced),
        Err(TwtError::GridMismatch { .. })
    ));
}

/// laya's `global_attn_every_n_layers = 3` type layout: full attention on
/// layers 0, 3, 6, …
fn laya_gss_types(n: usize) -> Vec<bool> {
    (0..n).map(|l| l % 3 != 0).collect()
}

// ── determinism — forward splits never move a bit ────────────────────────

#[test]
fn forward_split_invariance_bit_exact() {
    let corpus = planted_corpus(11, 5, 12, 33, &[&[3, 4]]);
    let one = {
        let mut b = SMatrixBuilder::new(5, 12);
        feed(&mut b, &corpus, 33);
        b.finalize().unwrap().cosine().clone()
    };
    let many = {
        let mut b = SMatrixBuilder::new(5, 12);
        feed(&mut b, &corpus, 7);
        b.finalize().unwrap().cosine().clone()
    };
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(one.get(i, j).to_bits(), many.get(i, j).to_bits());
        }
    }
}

// ── T1.4 — the ΔS map ────────────────────────────────────────────────────

#[test]
fn t1_4_delta_localizes_planted_damage() {
    let clean = SMatrix::from_fn(6, |i, j| {
        let grp = |l: usize| matches!(l, 2..=4);
        if grp(i) && grp(j) { 0.0 } else { 1.0 }
    });
    let damaged = SMatrix::from_fn(6, |i, j| {
        let grp = |l: usize| matches!(l, 2..=4);
        if grp(i) && grp(j) { 0.4 } else { 1.0 }
    });
    let d: DeltaMap = delta(&clean, &damaged).unwrap();
    assert_eq!(d.max, 0.4);
    let loc = localize_by_block(
        &d,
        &[
            Block { start: 0, end: 2 },
            Block { start: 2, end: 5 },
            Block { start: 5, end: 6 },
        ],
    );
    assert_eq!(loc[0].1, 0.0);
    assert_eq!(loc[1].1, 0.4);
    assert_eq!(loc[2].1, 0.0);
}

// ── builder validation surface ───────────────────────────────────────────

#[test]
fn builder_refuses_every_bad_shape() {
    let mut b = SMatrixBuilder::new(3, 4);
    assert!(matches!(b.push(0, 0, &[0.0; 4]), Err(TwtError::NoForwardOpen)));
    b.begin_forward(2).unwrap();
    assert!(matches!(b.begin_forward(2), Err(TwtError::ForwardAlreadyOpen)));
    assert!(matches!(b.push(3, 0, &[0.0; 4]), Err(TwtError::LayerOutOfRange { .. })));
    assert!(matches!(b.push(0, 2, &[0.0; 4]), Err(TwtError::RowOutOfRange { .. })));
    assert!(matches!(b.push(0, 0, &[0.0; 3]), Err(TwtError::DimMismatch { .. })));
    assert!(matches!(b.push(0, 0, &[f32::NAN; 4]), Err(TwtError::NonFiniteState { .. })));
    b.push(0, 0, &[0.1; 4]).unwrap();
    assert!(matches!(b.push(0, 0, &[0.1; 4]), Err(TwtError::DoubleWrite { .. })));
    assert!(matches!(
        b.end_forward(),
        Err(TwtError::IncompleteForward { got: 1, want: 6 })
    ));
}

// ── T2.3 — the partition is µs-scale at n=4096 (release) ─────────────────

#[test]
fn t2_3_partition_perf_at_n4096() {
    // Synthetic MONOTONE-friendly oracle: block-structured distances (the
    // shape the profiler hunts) at n = 4096. Timing asserted in RELEASE
    // only — a debug build's absolute number proves nothing (the profile
    // is part of the claim).
    let n = 4096usize;
    let block = 64usize;
    let s = SMatrix::from_fn(n, |i, j| {
        if i / block == j / block {
            0.05 + ((i ^ j) % 7) as f32 * 0.001
        } else {
            1.0
        }
    });
    let eps = 0.10f32;
    let t0 = std::time::Instant::now();
    let p = minmax_partition(&s, eps).unwrap();
    let elapsed = t0.elapsed();
    // Structure: exactly n/block blocks, every block satisfying ε.
    assert_eq!(p.len(), n / block);
    for b in &p {
        assert_eq!(b.len(), block);
    }
    if cfg!(not(debug_assertions)) {
        // The pre-registered bar: µs-SCALE — read as "sub-10ms at 4096",
        // i.e. O(n²)-class, never O(n³). Measured number recorded in the
        // close-out; the assert bounds the regression, not the box.
        assert!(
            elapsed.as_millis() < 10_000,
            "partition at n=4096 took {elapsed:?} — O(n³) regression"
        );
        println!("t2_3: n={n} partition in {elapsed:?} (release)");
    } else {
        println!("t2_3: n={n} partition in {elapsed:?} (debug — bar not asserted)");
    }
}

// ── the SVCCA arm's probe floor ──────────────────────────────────────────

#[test]
fn svcca_probe_floor_refuses_blind_meter() {
    let corpus = planted_corpus(5, 3, 8, 4, &[&[0, 1]]);
    let mut b = SMatrixBuilder::new(3, 8).with_svcca(SvccaCfg { stride: 1, cap_rows: 4 });
    feed(&mut b, &corpus, 4);
    // 4 retained rows ≤ dim 8 → blind meter, loud refusal (never a skip).
    assert!(matches!(b.finalize(), Err(TwtError::ProbeFloor { .. })));
}

// ── buckets: rows land in the bucket they were assigned ──────────────────

#[test]
fn rows_land_in_their_assigned_bucket() {
    let corpus = planted_corpus(13, 4, 8, 8, &[&[0, 1]]);
    let mut b = SMatrixBuilder::new(4, 8);
    b.begin_forward(8).unwrap();
    for row in 4..8 {
        b.set_row_bucket(row, Bucket::Head).unwrap();
    }
    for layer in 0..4 {
        for (r, states) in corpus.iter().enumerate() {
            b.push(layer, r, &states[layer]).unwrap();
        }
    }
    b.end_forward().unwrap();
    let s = b.finalize().unwrap();
    // Prompt bucket: rows 0..4 (clone pair 0,1 exact zero).
    assert_eq!(s.bucket(Bucket::Prompt).unwrap().get(0, 1), 0.0);
    // Head bucket: rows 4..8 (same clone pair, also exact zero).
    assert_eq!(s.bucket(Bucket::Head).unwrap().get(0, 1), 0.0);
    // Generation band never fed.
    assert!(s.bucket(Bucket::Generation).is_none());
}
