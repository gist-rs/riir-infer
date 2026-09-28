//! TWT Phase-4 gate battery (Issue 022 T4.1/T4.2) — the re-ternarization
//! arms + the κ budget in `twt::ternarize`, whole-file `#![cfg]`-gated on
//! `twt_collapse` (the Cargo row is the reader protection: a partial
//! feature selection SKIPS this target loudly instead of printing a
//! green zero).
//!
//! What these gates pin:
//! - **the arm-C bit-exact round trip**: dequant → re-quant recovers the
//!   source ternary container byte-for-byte (scales included) — the
//!   source-quantizer arm is the identity on already-ternary input;
//! - **the arm-B majority semantics** on hand-checked codes (majority
//!   survives, tie/weak votes zero, the scale is the supported-position
//!   amax of the real sum);
//! - **the arm-A f16 rounding** (known-answer round-trip on f16 values);
//! - **determinism**: every arm is bit-identical across runs;
//! - **the T4.2 budget machinery**: pass on an exact materialization,
//!   loud breach on a hostile operator, NaN refusal, ratio semantics;
//! - **refusals**: shape mismatch, non-finite input, arm-A pack refusal,
//!   Q2_0 wire round-trip losslessness (pack → repack = identity).

#![cfg(feature = "twt_collapse")]

use katgpt_core::TernaryGroupWeights;
use riir_infer_core::quant::q2_0::{dequantize_row_q2_0, repack_q2_0_to_ternary_group, BlockQ2_0};
use riir_infer_core::twt::ternarize::{
    arm_dense_f16, arm_sign_majority, arm_source_quant, budget_ok, budget_ratio,
    materialization_rel_err, Materialized, KAPPA_BUDGET,
};
use riir_infer_core::twt::synth::Lcg;
use riir_infer_core::twt::TwtError;

/// Build a small random-ish ternary container with per-group scales
/// (deterministic LCG; rows×cols, cols % 128 == 0).
fn ternary_fixture(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
    let mut rng = Lcg::new(seed);
    let mut w = TernaryGroupWeights::new(rows, cols);
    for r in 0..rows {
        for g in 0..w.groups_per_row {
            // scale in [0.01, 2.0) — f16-representable
            w.group_scale[r * w.groups_per_row + g] =
                half::f16::from_f32(0.01 + 1.99 * rng.next_centered().abs());
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            let roll = rng.next_u64() % 3;
            let v: i8 = match roll {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            w.set(r, c, v);
        }
    }
    w
}

#[test]
fn arm_c_is_a_bit_exact_round_trip_on_ternary_input() {
    let src = ternary_fixture(2, 256, 0xA11CE);
    // dequant → arm C → compare every container field.
    let dense =
        riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(
            &src,
        );
    let mat = arm_source_quant(&dense, 2, 256).unwrap();
    let Materialized::Ternary(out) = mat else {
        panic!("arm C must produce a ternary container");
    };
    assert_eq!(out.pos_bits, src.pos_bits, "pos plane drift");
    assert_eq!(out.neg_bits, src.neg_bits, "neg plane drift");
    assert_eq!(out.group_scale, src.group_scale, "scale drift");
}

#[test]
fn arm_b_majority_known_answers() {
    // 3 members × 4 positions (cols 256 to satisfy the group layout;
    // only the first 4 positions are asserted).
    let rows = 1usize;
    let cols = 256usize;
    let mut a = vec![0f32; rows * cols];
    let mut b = vec![0f32; rows * cols];
    let mut c = vec![0f32; rows * cols];
    // position 0: [+1, +1, -1] → vote +2 → sign +
    // position 1: [+1, -1, -1] → vote -1 → sign −
    // position 2: [+1, -1,  0] → vote  0 → zeroed (weak)
    // position 3: [+1,  0,  0] → vote +1 → sign + (lone majority)
    a[0] = 1.0; b[0] = 1.0; c[0] = -1.0;
    a[1] = 1.0; b[1] = -1.0; c[1] = -1.0;
    a[2] = 1.0; b[2] = -1.0; c[2] = 0.0;
    a[3] = 1.0; b[3] = 0.0; c[3] = 0.0;
    let mat = arm_sign_majority(&[&a, &b, &c], rows, cols).unwrap();
    let Materialized::Ternary(w) = mat else {
        panic!("arm B must produce a ternary container");
    };
    assert_eq!(w.get(0, 0), 1);
    assert_eq!(w.get(0, 1), -1);
    assert_eq!(w.get(0, 2), 0, "a zero vote must zero the position");
    assert_eq!(w.get(0, 3), 1, "a lone +1 is a strict majority over zeros");
    // The scale is the supported-position amax of the REAL sum:
    // |Σw| at pos 0 = 1, pos 1 = -1 → 1, pos 3 = 1 → d = 1.0.
    assert_eq!(w.scale_at(0, 0), 1.0);
    // Unsupported positions (pos 2) stay zero-valued.
    let dense = w.pos_bits[0] | w.neg_bits[0];
    let _ = dense;
}

#[test]
fn arm_b_scale_is_the_supported_amax_of_the_real_sum() {
    // Members with DIFFERENT scales at position 0: a = +4.0, b = -0.5,
    // c = -0.25 → the CODE vote is +1-1-1 = -1 (two members against one)
    // → supported, sign NEGATIVE — the vote is scale-free by pin. The
    // |Σw| = 3.25 still sets the magnitude scale.
    let rows = 1usize;
    let cols = 256usize;
    let mut am = vec![0f32; rows * cols]; am[0] = 4.0;
    let mut bm = vec![0f32; rows * cols]; bm[0] = -0.5;
    let mut cm = vec![0f32; rows * cols]; cm[0] = -0.25;
    let mat = arm_sign_majority(&[&am, &bm, &cm], rows, cols).unwrap();
    let Materialized::Ternary(w) = mat else { panic!("ternary") };
    assert_eq!(w.get(0, 0), -1, "the scale-free vote is +1-1-1 = -1");
    let d = w.scale_at(0, 0);
    assert!(
        (d - 3.25f32).abs() < 0.01,
        "scale {d} should be f16(3.25), the supported-position amax of the real sum"
    );
}

#[test]
fn arm_a_round_trips_f16_values_exactly() {
    // Values that ARE f16-representable survive arm A bit-exactly.
    let a = vec![0.5f32, -2.0, 0.25];
    let b = vec![0.5f32, -2.0, 0.25];
    let c = vec![0.5f32, -2.0, 0.25];
    let mat = arm_dense_f16(&[&a, &b, &c], 1, 3).unwrap();
    let Materialized::DenseF16(d) = mat else { panic!("dense") };
    let got: Vec<f32> = d.data.iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, vec![0.5, -2.0, 0.25]);
}

#[test]
fn every_arm_is_deterministic_across_runs() {
    let m1 = ternary_fixture(2, 256, 7);
    let m2 = ternary_fixture(2, 256, 0xBEEF);
    let d1 = qwen_dequant(&m1);
    let d2 = qwen_dequant(&m2);
    let arms1 = (
        arm_dense_f16(&[&d1, &d2], 2, 256).unwrap(),
        arm_sign_majority(&[&d1, &d2], 2, 256).unwrap(),
        arm_source_quant(&qwen_dequant_mean(&[&d1, &d2]), 2, 256).unwrap(),
    );
    let arms2 = (
        arm_dense_f16(&[&d1, &d2], 2, 256).unwrap(),
        arm_sign_majority(&[&d1, &d2], 2, 256).unwrap(),
        arm_source_quant(&qwen_dequant_mean(&[&d1, &d2]), 2, 256).unwrap(),
    );
    for (x, y) in arms1.0.to_dense_f32().iter().zip(arms2.0.to_dense_f32()) {
        assert_eq!(x.to_bits(), y.to_bits(), "arm A drifted");
    }
    match (&arms1.1, &arms2.1) {
        (Materialized::Ternary(x), Materialized::Ternary(y)) => {
            assert_eq!(x.pos_bits, y.pos_bits);
            assert_eq!(x.neg_bits, y.neg_bits);
            assert_eq!(x.group_scale, y.group_scale);
        }
        _ => panic!("arm B shape"),
    }
    match (&arms1.2, &arms2.2) {
        (Materialized::Ternary(x), Materialized::Ternary(y)) => {
            assert_eq!(x.pos_bits, y.pos_bits);
            assert_eq!(x.neg_bits, y.neg_bits);
            assert_eq!(x.group_scale, y.group_scale);
        }
        _ => panic!("arm C shape"),
    }
}

fn qwen_dequant(w: &TernaryGroupWeights) -> Vec<f32> {
    riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w)
}

fn qwen_dequant_mean(members: &[&[f32]]) -> Vec<f32> {
    riir_infer_core::twt::audition::merge_mean(members.iter().copied()).unwrap()
}

#[test]
fn the_q2_0_wire_round_trip_is_lossless() {
    // arm C output → pack → repack → identical container.
    let src = ternary_fixture(2, 256, 0x5EED);
    let dense = qwen_dequant(&src);
    let mat = arm_source_quant(&dense, 2, 256).unwrap();
    let mut blocks: Vec<BlockQ2_0> = Vec::new();
    mat.pack_q2_0(&mut blocks).unwrap();
    let repacked = repack_q2_0_to_ternary_group(&blocks, 2, 256).unwrap();
    let Materialized::Ternary(w) = mat else { panic!("ternary") };
    assert_eq!(repacked.pos_bits, w.pos_bits);
    assert_eq!(repacked.neg_bits, w.neg_bits);
    assert_eq!(repacked.group_scale, w.group_scale);
    // And the wire dequant agrees with the container's eval view.
    let mut deq = vec![0f32; 2 * 256];
    dequantize_row_q2_0(&blocks, &mut deq);
    let eval = riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(&w);
    for (x, y) in deq.iter().zip(eval) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn the_budget_machinery_passes_exact_and_refuses_hostile() {
    // Exact materialization: dense_err == 0, arm_err == 0 → pass.
    assert!(budget_ok(0.0, 0.0));
    // Hostile: arm damage double the f16 floor → breach at κ = 2.
    let dense_err = 0.01f64;
    let arm_err = 3.0 * dense_err;
    assert!(!budget_ok(arm_err, dense_err), "3× must breach κ=2");
    assert!(budget_ok(2.0 * dense_err, dense_err), "exactly 2× sits AT the ceiling");
    // NaN refuses both directions.
    assert!(!budget_ok(f64::NAN, 0.01));
    assert!(!budget_ok(0.01, f64::NAN));
    // Ratio semantics.
    assert_eq!(budget_ratio(0.03, 0.01), 3.0);
    assert_eq!(budget_ratio(0.0, 0.0), 0.0);
    assert_eq!(budget_ratio(0.01, 0.0), f64::INFINITY);
    // κ is what the pre-registration says it is.
    assert_eq!(KAPPA_BUDGET, 2.0);
}

#[test]
fn the_budget_separates_by_surrogate_strength_not_by_arm_damage_alone() {
    // T4.2's baseline is the f16 SURROGATE's end-to-end mapping error
    // against the PARENT mapping (E_dense) — not arm A's f16 rounding
    // floor (which is ~1e-8 and would make κ·E_dense unfailably tight).
    // The budget reads E_arm ≤ κ · E_dense where both errors are measured
    // the audition's way: surrogate vs parent on the same rows.
    //
    // Construction: parent W_p = f̄ + merge noise; the f16 surrogate is
    // f̄'s f16 rounding (its error IS the merge noise); the ternary arm
    // adds its materialization damage on top.
    let rows = 4usize;
    let cols = 256usize;
    let m1 = ternary_fixture(rows, cols, 11);
    let m2 = ternary_fixture(rows, cols, 23);
    let d1 = qwen_dequant(&m1);
    let d2 = qwen_dequant(&m2);
    let fbar = qwen_dequant_mean(&[&d1, &d2]);

    let mut rng = Lcg::new(99);
    let n_x = 16usize;
    let xs: Vec<f32> = (0..n_x * cols).map(|_| rng.next_centered()).collect();
    let mut ya = vec![0f32; rows];
    let mut yb = vec![0f32; rows];

    // Materialization damage alone (the honest datum this gate records):
    // on a mean-of-two-ternary f̄, arm C's 5→3 level reduction is NOT
    // small — quoted in the issue, never gated here (the audition's
    // E_dense decides per block).
    let arm_c = arm_source_quant(&fbar, rows, cols).unwrap();
    let arm_a = arm_dense_f16(&[&fbar.clone()], rows, cols).unwrap();
    let c_damage = materialization_rel_err(
        &arm_c.to_dense_f32(), &fbar, &xs, rows, cols, n_x, &mut ya, &mut yb,
    )
    .unwrap();
    let a_damage = materialization_rel_err(
        &arm_a.to_dense_f32(), &fbar, &xs, rows, cols, n_x, &mut ya, &mut yb,
    )
    .unwrap();
    assert!(a_damage < 1e-6, "arm A's f16 rounding floor moved: {a_damage}");
    assert!(
        c_damage > 0.001,
        "arm C damage on a merged f̄ unexpectedly vanished ({c_damage}) — the 5→3 level reduction must be visible"
    );

    // Strong surrogate: the parent differs from f̄ by an equal-energy
    // noise (‖noise‖ = ‖f̄‖) — E_dense ≈ 0.4 and ratio ≈ 1 + damage/frac²
    // ≈ 1.24, well under κ=2 → the ternary arm passes. Weak surrogate:
    // 5% noise — ratio ≈ 1 + 97 → breaches by miles. The surrogate
    // strength, not the arm, decides. (The ratio law is why the pass
    // margin is the WIDE one: at frac=0.5 the ratio sits AT κ.)
    for (frac, should_pass) in [(1.0f32, true), (0.05, false)] {
        // Noise with EXACTLY frac²·‖f̄‖_F² energy (normalized post-hoc —
        // next_centered is not unit-variance, so no a-priori g).
        let fnorm2: f32 = fbar.iter().map(|v| v * v).sum::<f32>();
        let mut rng = Lcg::new(7);
        let mut noise: Vec<f32> = (0..rows * cols).map(|_| rng.next_centered()).collect();
        let n2: f32 = noise.iter().map(|v| v * v).sum::<f32>();
        let scale = frac * fnorm2.sqrt() / n2.sqrt();
        for v in &mut noise {
            *v *= scale;
        }
        let parent_op: Vec<f32> = fbar.iter().zip(&noise).map(|(f, n)| f + n).collect();
        let e_dense = materialization_rel_err(
            &arm_a.to_dense_f32(), &parent_op, &xs, rows, cols, n_x, &mut ya, &mut yb,
        )
        .unwrap();
        let e_arm = materialization_rel_err(
            &arm_c.to_dense_f32(), &parent_op, &xs, rows, cols, n_x, &mut ya, &mut yb,
        )
        .unwrap();
        assert_eq!(
            budget_ok(e_arm, e_dense),
            should_pass,
            "E_arm {e_arm} vs κ·E_dense {} (frac {frac})",
            KAPPA_BUDGET * e_dense
        );
    }
}

#[test]
fn refusals_are_loud() {
    let a = vec![1.0f32, 2.0];
    let b = vec![1.0f32, 2.0, 3.0];
    assert!(matches!(
        arm_dense_f16(&[&a, &b], 1, 2),
        Err(TwtError::ArmShapeMismatch { .. })
    ));
    assert!(matches!(arm_dense_f16(&[], 1, 2), Err(TwtError::ArmEmptyPool)));
    let mut nan = vec![0f32; 256];
    nan[0] = f32::NAN;
    assert!(matches!(
        arm_sign_majority(&[&nan, &nan], 1, 256),
        Err(TwtError::NonFiniteMerged)
    ));
    assert!(matches!(
        arm_source_quant(&nan, 1, 256),
        Err(TwtError::NonFiniteMerged)
    ));
    // Arm A has no Q2_0 wire payload — refuse, never silently skip.
    let mat = arm_dense_f16(&[&a, &a], 1, 2).unwrap();
    let mut blocks = Vec::new();
    assert!(mat.pack_q2_0(&mut blocks).is_err());
    // Degenerate baseline: zero operator on zero inputs.
    let zeros = vec![0f32; 2];
    let mut ya = vec![0f32; 1];
    let mut yb = vec![0f32; 1];
    assert!(matches!(
        materialization_rel_err(&zeros, &zeros, &[0.0, 0.0], 1, 2, 1, &mut ya, &mut yb),
        Err(TwtError::DegenerateBaseline)
    ));
}

#[test]
fn non_multiple_of_128_cols_are_refused_at_the_wire_only() {
    // The container accepts arbitrary cols; the Q2_0 WIRE refuses them.
    let dense = vec![0.5f32; 3 * 2]; // 3 rows × 2 cols
    let mat = arm_source_quant(&dense, 3, 2).unwrap();
    let mut blocks = Vec::new();
    assert!(mat.pack_q2_0(&mut blocks).is_err());
}
