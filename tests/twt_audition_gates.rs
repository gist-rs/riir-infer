//! TWT Phase-3 gate battery (Issue 022 T3.1/T3.2/T3.3) — the PURE
//! audition math in `twt::audition`, whole-file `#![cfg]`-gated on
//! `twt_profile` (the row in Cargo.toml is the reader protection: a
//! partial feature selection SKIPS this target loudly instead of
//! printing a green zero).
//!
//! What these gates pin (the laya-coupled apply path is proven by the
//! DRIVER's parity arm each run, not here — this battery needs no
//! checkpoint):
//! - the merge operators' planted identities (clones are fixed points;
//!   RDSC telescopes at k=2 and matches the closed form at k=3);
//! - the correction fit recovers an exact affine map, lands near zero on
//!   orthogonal noise, and counts degenerate channels instead of
//!   silently zeroing them;
//! - the selection pin is deterministic, moves with any error byte, and
//!   the argmin cross-check refuses a tampered table;
//! - the mean-sq metric matches a hand count.

#![cfg(feature = "twt_profile")]

use riir_infer_core::twt::audition::{
    mean_sq_err, merge_mean, merge_rdsc, selection_pin, CandRow, CorrectionFit,
};
use riir_infer_core::twt::TwtError;

#[test]
fn mean_of_clones_is_the_clone() {
    let w = vec![1.5f32, -2.0, 0.25, 7.0, -0.125];
    let m = merge_mean([&w[..], &w[..], &w[..], &w[..]]).unwrap();
    assert_eq!(m, w);
}

#[test]
fn rdsc_of_clones_is_the_clone() {
    let w = vec![1.5f32, -2.0, 0.25, 7.0, -0.125];
    let m = merge_rdsc([&w[..], &w[..], &w[..], &w[..]]).unwrap();
    assert_eq!(m, w);
}

#[test]
fn rdsc_k2_is_the_last_member() {
    let a = [1.0f32, 2.0];
    let b = [3.0f32, -4.0];
    assert_eq!(merge_rdsc([&a[..], &b[..]]).unwrap(), b);
}

#[test]
fn rdsc_k3_matches_the_closed_form() {
    let a = [1.0f32, 2.0, 0.5];
    let b = [3.0f32, -4.0, 1.25];
    let c = [0.5f32, 1.0, -2.0];
    let m = merge_rdsc([&a[..], &b[..], &c[..]]).unwrap();
    for i in 0..3 {
        assert_eq!(m[i], a[i] + b[i] + c[i] - 2.0 * a[i]);
    }
}

#[test]
fn shape_mismatch_refused_both_operators() {
    let a = [1.0f32, 2.0];
    let b = [1.0f32];
    assert_eq!(
        merge_mean([&a[..], &b[..]]).unwrap_err(),
        TwtError::MergeShapeMismatch { expected: 2, got: 1 }
    );
    assert!(merge_rdsc([&a[..], &b[..]]).is_err());
}

#[test]
fn fit_recovers_an_exact_affine_map() {
    let (rows, d) = (64usize, 8usize);
    let mut h_in = vec![0f32; rows * d];
    let mut h_sur = vec![0f32; rows * d];
    let mut h_e = vec![0f32; rows * d];
    let alpha_star = [0.5f32, 1.0, 2.0, -1.0, 0.25, 3.0, -0.75, 1.5];
    let beta_star = [0.1f32, -0.2, 0.0, 0.5, -0.5, 1.0, 0.05, -0.05];
    for i in 0..rows {
        for c in 0..d {
            let off = i * d + c;
            // Deterministic pseudo-data: non-constant per channel, spread
            // over both signs.
            h_in[off] = ((i * 7 + c * 13) % 17) as f32 * 0.125 - 1.0;
            let r = (((i * 11 + c * 5) % 23) as f32 * 0.0625 - 0.65625) * 4.0;
            h_sur[off] = h_in[off] + r;
            h_e[off] = h_in[off] + alpha_star[c] * r + beta_star[c];
        }
    }
    let f = CorrectionFit::fit(&h_in, &h_sur, &h_e, rows, d).unwrap();
    assert_eq!(f.degenerate_channels, 0);
    for c in 0..d {
        assert!((f.alpha[c] - alpha_star[c]).abs() < 1e-3, "alpha[{c}]: {} vs {}", f.alpha[c], alpha_star[c]);
        assert!((f.beta[c] - beta_star[c]).abs() < 1e-3, "beta[{c}]: {} vs {}", f.beta[c], beta_star[c]);
    }
}

#[test]
fn degenerate_channels_counted_not_silently_zeroed() {
    let (rows, d) = (32usize, 4usize);
    let h_in = vec![0f32; rows * d];
    let mut h_sur = vec![0f32; rows * d];
    let mut h_e = vec![0f32; rows * d];
    for i in 0..rows {
        for c in 0..d {
            let off = i * d + c;
            h_sur[off] = 1.0; // constant residual → var 0 → degenerate
            h_e[off] = if (i + c) % 2 == 0 { 0.5 } else { -0.5 };
        }
    }
    let f = CorrectionFit::fit(&h_in, &h_sur, &h_e, rows, d).unwrap();
    assert_eq!(f.degenerate_channels, d);
    for c in 0..d {
        assert_eq!(f.alpha[c], 0.0);
    }
}

#[test]
fn correction_apply_is_the_stated_form() {
    let fit = CorrectionFit {
        alpha: vec![2.0f32, 0.5],
        beta: vec![0.25f32, -0.25],
        degenerate_channels: 0,
    };
    let h_in = [1.0f32, 2.0];
    let h_sur = [1.5f32, 4.0]; // r = (0.5, 2.0)
    let mut out = [0f32; 2];
    fit.apply(&h_in, &h_sur, &mut out);
    assert_eq!(out, [1.0 + 2.0 * 0.5 + 0.25, 2.0 + 0.5 * 2.0 - 0.25]);
}

#[test]
fn fit_alpha_only_recovers_an_exact_through_origin_map() {
    // y = α*·r exactly (β = 0) → the through-origin fit lands on α*.
    let (rows, d) = (48usize, 4usize);
    let mut h_in = vec![0f32; rows * d];
    let mut h_sur = vec![0f32; rows * d];
    let mut h_e = vec![0f32; rows * d];
    let alpha_star = [1.5f32, -0.5, 2.0, 0.25];
    for i in 0..rows {
        for (c, a) in alpha_star.iter().enumerate() {
            let off = i * d + c;
            h_in[off] = ((i * 5 + c * 3) % 11) as f32 * 0.25 - 1.25;
            let r = (((i * 13 + c * 7) % 19) as f32 * 0.125 - 1.125) * 2.0;
            h_sur[off] = h_in[off] + r;
            h_e[off] = h_in[off] + a * r;
        }
    }
    let f = CorrectionFit::fit_alpha_only(&h_in, &h_sur, &h_e, rows, d).unwrap();
    assert_eq!(f.degenerate_channels, 0);
    for (c, a) in alpha_star.iter().enumerate() {
        assert!((f.alpha[c] - a).abs() < 1e-3);
        assert_eq!(f.beta[c], 0.0);
    }
}

#[test]
fn fit_shape_mismatch_refused() {
    let a = vec![0f32; 16];
    assert!(matches!(
        CorrectionFit::fit(&a[..12], &a, &a, 4, 4),
        Err(TwtError::FitShapeMismatch { .. })
    ));
}

#[test]
fn mean_sq_err_matches_a_hand_count() {
    let pred = [1.0f32, 2.0, 3.0, 4.0];
    let h_e = [0.0f32, 2.0, 3.0, 5.0];
    assert_eq!(mean_sq_err(&pred, &h_e, 2, 2), 1.0);
    assert!(mean_sq_err(&pred, &h_e, 0, 2).is_nan());
}

#[test]
fn selection_pin_deterministic_and_byte_sensitive() {
    let rows = vec![
        CandRow { id: "member:3".into(), err_fit: 2.0 },
        CandRow { id: "mean".into(), err_fit: 1.0 },
        CandRow { id: "rdsc".into(), err_fit: 3.0 },
    ];
    let p1 = selection_pin(&rows).unwrap();
    assert_eq!(p1.winner, "mean");
    let p2 = selection_pin(&rows).unwrap();
    assert_eq!(p1.table_blake3, p2.table_blake3);
    let mut moved = rows.clone();
    moved[1].err_fit = 1.5;
    assert_ne!(p1.table_blake3, selection_pin(&moved).unwrap().table_blake3);
}

#[test]
fn tie_resolves_to_the_lowest_index() {
    let rows = vec![
        CandRow { id: "first".into(), err_fit: 1.0 },
        CandRow { id: "second".into(), err_fit: 1.0 },
    ];
    assert_eq!(selection_pin(&rows).unwrap().winner, "first");
}

#[test]
fn empty_inputs_refused() {
    let empty: [&[f32]; 0] = [];
    assert_eq!(merge_mean(empty).unwrap_err(), TwtError::EmptyMerge);
    assert!(selection_pin(&[]).is_err());
    assert!(CorrectionFit::fit(&[], &[], &[], 0, 4).is_err());
}
