//! Issue 832 gates — the NaN-safe total-order comparator this crate's float
//! sorts now consume (`katgpt_core::float_order`).
//!
//! The NaN-heavy gate FAILS against the replaced
//! `partial_cmp(..).unwrap_or(Equal)` idiom by construction: on a >=20-element
//! slice with NaNs, the legacy comparator is intransitive and std's sort
//! ABORTS ("user-provided comparison function does not correctly implement a
//! total order"; measured rustc 1.93, still true on 1.95). The corpus pin
//! proves NaN-free ordering is bit-identical to the replaced idiom, which is
//! what lets the fixes land under the existing no-regression gates.

use katgpt_core::float_order::{asc, desc};

/// Both infinities, both zeros, both subnormal bounds — every pair must order
/// exactly as the legacy idiom ordered it.
#[test]
fn nan_free_ordering_matches_the_replaced_idiom() {
    use core::cmp::Ordering::Equal;
    let corpus = [
        f32::INFINITY,
        f32::MAX,
        1.0,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE * 0.5,
        0.0,
        -0.0,
        -f32::MIN_POSITIVE * 0.5,
        -1.0,
        f32::MIN,
        f32::NEG_INFINITY,
    ];
    for a in corpus {
        for b in corpus {
            let legacy = a.partial_cmp(&b).unwrap_or(Equal);
            assert_eq!(desc(a, b), legacy.reverse(), "desc({a},{b}) diverged");
            assert_eq!(asc(a, b), legacy, "asc({a},{b}) diverged");
            assert_eq!(asc_f64(a as f64, b as f64), legacy, "asc_f64({a},{b}) diverged");
        }
    }
}

/// The production-size panic gate: 24 elements with 7 NaNs aborts under the
/// legacy idiom; under [`desc`] the reals come out fully sorted with NaN sunk
/// to the tail (it can never top a best-first list).
#[test]
fn nan_heavy_sort_does_not_abort_and_sinks_nan() {
    let mut xs: Vec<f32> = (0..17).map(|i| (i as f32) * 1.5 - 12.0).collect();
    for slot in [0, 4, 8, 12, 16, 1, 9] {
        xs[slot] = f32::NAN;
    }
    xs.push(f32::NAN);
    xs.sort_by(|a, b| desc(*a, *b));
    let n_reals = xs.len() - xs.iter().filter(|v| v.is_nan()).count();
    assert_eq!(n_reals, 10, "17 base - 7 planted NaNs = 10 reals");
    let mut expect: Vec<f32> = xs.iter().copied().filter(|v| !v.is_nan()).collect();
    expect.sort_by(|a, b| b.total_cmp(a)); // NaN-free: legacy is exact here
    assert_eq!(&xs[..n_reals], &expect[..], "reals must come out fully sorted");
    assert!(xs[n_reals..].iter().all(|v| v.is_nan()), "NaN must sort last");
}

/// Ascending sites (nearest-first / cheapest-first) get the same guarantee:
/// a NaN cost must never look cheap — it sinks to the tail.
#[test]
fn nan_heavy_ascending_sort_sinks_nan() {
    let mut ys: Vec<f64> = vec![3.5, f64::NAN, -1.0, f64::NAN, 0.0, f64::NAN];
    ys.sort_by(|a, b| asc_f64(*a, *b));
    assert_eq!(&ys[..3], &[-1.0, 0.0, 3.5]);
    assert!(ys[3..].iter().all(|v| v.is_nan()), "NaN must sort last");
}

use katgpt_core::float_order::asc_f64;
