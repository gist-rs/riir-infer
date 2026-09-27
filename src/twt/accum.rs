//! `PairCosineAccum` — the streaming pair-similarity accumulator (Issue 022
//! T1.1).
//!
//! Follows the `katgpt_core::canvas::transfer::cosine_distance` reduction
//! DISCIPLINE (that fn is `pub(crate)`: a pattern to follow, not consumable
//! substrate — single-pass dot + squared norms accumulated in **f64**, no
//! allocation, overflow-safe for f32 inputs, conservative zero-norm). This
//! accumulator is the STREAMING generalization: instead of one pair, it
//! folds *every* calibration state pair `(h_i, h_j)` observed across the
//! whole corpus into one `(dot, na, nb)` triple, and the finalized cosine is
//! taken once — the pooled "expected cosine distance" meter of Research 594
//! §1 item 1 (TWT Eq. 4).
//!
//! # Determinism
//!
//! f64 addition order is part of the result: callers must feed pairs in a
//! canonical order (the S-matrix builder folds rows ascending, forwards in
//! corpus order). Identical feeds are bit-identical cross-platform.
//!
//! # Zero-allocation
//!
//! [`Self::add`] touches only the three f64 fields — the accumulate path
//! allocates nothing (G4, gate-tested).

/// Streaming dot+norm accumulator over `(a, b)` state pairs.
///
/// `distance()` finalizes `1 - cos` in `[0, 2]`. The zero-norm /
/// non-finite case returns `1.0` (maximally distant — the conservative
/// choice of the cosine_distance discipline: no information → treat as
/// different, never as similar; a merge decision made on a fabricated 0.0
/// would collapse layers that were never observed to agree).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairCosineAccum {
    dot: f64,
    na: f64,
    nb: f64,
    positions: u64,
}

impl PairCosineAccum {
    pub const fn new() -> Self {
        Self {
            dot: 0.0,
            na: 0.0,
            nb: 0.0,
            positions: 0,
        }
    }

    /// Fold one `(a, b)` state pair (one calibration position).
    ///
    /// Mismatched lengths fold the common prefix (the cosine_distance
    /// discipline's `min`); the S-matrix builder asserts equality upstream,
    /// so this arm is defensive only.
    #[inline]
    pub fn add(&mut self, a: &[f32], b: &[f32]) {
        debug_assert_eq!(a.len(), b.len(), "pair state lengths");
        let len = a.len().min(b.len());
        for k in 0..len {
            let x = a[k] as f64;
            let y = b[k] as f64;
            self.dot += x * y;
            self.na += x * x;
            self.nb += y * y;
        }
        self.positions += 1;
    }

    /// Calibration positions folded so far.
    pub fn positions(&self) -> u64 {
        self.positions
    }

    /// Finalize `1 - cos` in `[0, 2]`; `1.0` when either pooled side is
    /// zero-norm or the reduction went non-finite (conservative).
    pub fn distance(&self) -> f32 {
        let denom = (self.na * self.nb).sqrt();
        if denom == 0.0 || !denom.is_finite() {
            return 1.0;
        }
        let cos = (self.dot / denom).clamp(-1.0, 1.0);
        if cos.is_nan() {
            return 1.0;
        }
        (1.0 - cos) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_pairs_distance_zero() {
        let mut acc = PairCosineAccum::new();
        let v = [1.0f32, -2.0, 3.0, 0.5];
        for _ in 0..7 {
            acc.add(&v, &v);
        }
        // dot == na == nb exactly (same bytes both sides) → cos 1.0 exactly.
        assert_eq!(acc.distance(), 0.0);
        assert_eq!(acc.positions(), 7);
    }

    #[test]
    fn parallel_large_magnitude_is_exact_zero_via_f64() {
        // The cosine_distance overflow class: 1e20 squares to 1e40, which
        // overflows f32 but is exact in f64 — a pure-f32 reduction yields
        // inf/inf = NaN, the f64 accumulator stays well-defined.
        let mut acc = PairCosineAccum::new();
        let a = [1e20f32, 0.0];
        let b = [2e20f32, 0.0];
        acc.add(&a, &b);
        assert_eq!(acc.distance(), 0.0);
    }

    #[test]
    fn orthogonal_is_one_opposite_is_two() {
        let mut acc = PairCosineAccum::new();
        acc.add(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((acc.distance() - 1.0).abs() < 1e-6);

        let mut acc = PairCosineAccum::new();
        acc.add(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((acc.distance() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn zero_norm_is_conservative_one() {
        let mut acc = PairCosineAccum::new();
        acc.add(&[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(acc.distance(), 1.0);

        let mut acc = PairCosineAccum::new();
        assert_eq!(acc.distance(), 1.0, "empty accumulator is conservative");
    }

    #[test]
    fn non_finite_input_is_conservative_one() {
        let mut acc = PairCosineAccum::new();
        acc.add(&[f32::NAN, 1.0], &[1.0, 1.0]);
        assert_eq!(acc.distance(), 1.0);
    }
}
