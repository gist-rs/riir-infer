//! f64 golden-NLL readout (Issue 959 T6) — the tail-faithful per-token
//! oracle for lossy-quant gates, consuming katgpt-core's
//! `fixed_log_softmax_target_f64` (the argmax-excluded-LSE softplus
//! form, arXiv:2610.31642 Eq. 5).
//!
//! **Why a golden oracle.** The lossy-surface promotion rule distrusts
//! flat aggregate perplexity (family-conditional behavior can flip
//! under flat aggregates), and quant deltas of 1e-6..1e-3 nats on
//! CONFIDENT tokens are exactly where an incumbent fp32 log-softmax
//! reads a false zero (the Softmax Collapse cliff `c_d = p·ln2`). The
//! oracle resolves the target slot's log-prob from the cliff out to
//! exponent underflow (~745 nats f64) with a PINNED fold order
//! (first-index argmax, sequential ascending-j `+=` — the determinism
//! law pinned at the katgpt-core site), so a two-variant per-token NLL
//! diff is structural, not an artifact of the readout form.
//!
//! **Consumption shape.** `read_target` takes the raw f32 logits of one
//! forward, widens them ONCE into caller-owned scratch (zero
//! steady-state alloc — the scratch rides the harness loop), and
//! returns the per-token NLL plus the top-2 logit margin (the
//! confidence coordinate the band disclosures key on).
//! Instrument-grade only — never a serving path; the logits are READ,
//! the forward is untouched.

use katgpt_core::softmax_fix::fixed_log_softmax_target_f64;

/// One scored position's golden reading: the per-token NLL in f64
/// (never saturating past any margin) and the top-2 logit margin in
/// nats (the confidence coordinate; band thresholds live with the
/// consumer, read from `katgpt_core::softmax_fix::precision_cliff`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoldenReading {
    pub nll: f64,
    pub top2_margin: f64,
}

/// Reusable widen scratch: allocate once per harness, reuse per scored
/// position (no per-position allocation in the harness loop).
#[derive(Debug, Default, Clone)]
pub struct GoldenNllScratch {
    widened: Vec<f64>,
}

impl GoldenNllScratch {
    /// Pre-size for a vocab (an optimization, never required — the
    /// scratch grows on first use otherwise).
    pub fn with_vocab(vocab: usize) -> Self {
        Self {
            widened: Vec::with_capacity(vocab),
        }
    }

    /// The golden per-token reading at `target`. Panics if `target` is
    /// out of range or `logits` is empty (the primitive's contract).
    pub fn read_target(&mut self, logits: &[f32], target: usize) -> GoldenReading {
        assert!(target < logits.len(), "golden_nll: target out of range");
        self.widened.clear();
        self.widened.extend(logits.iter().map(|&l| f64::from(l)));
        // Top-2 margin: FIRST-index strict-greater walk for the max (the
        // same tie rule the primitive pins), then a strict-greater walk
        // for the runner-up skipping the max index — a fixed, documented
        // rule (the confidence band is a measurement, but its fold order
        // is pinned all the same).
        let (mut max_i, mut top) = (0usize, f64::NEG_INFINITY);
        for (i, &v) in self.widened.iter().enumerate() {
            if v > top {
                top = v;
                max_i = i;
            }
        }
        let mut second = f64::NEG_INFINITY;
        for (i, &v) in self.widened.iter().enumerate() {
            if i != max_i && v > second {
                second = v;
            }
        }
        GoldenReading {
            nll: -fixed_log_softmax_target_f64(&self.widened, target),
            top2_margin: top - second,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::softmax_fix::{FloatFormat, precision_cliff};

    fn incumbent_nll_f64(logits: &[f32], target: usize) -> f64 {
        // The standard max-subtract form in f64 — the shape the
        // existing harness rows use (ad-hoc fold, absorption-bounded).
        let mut max_l = f32::NEG_INFINITY;
        for &l in logits {
            max_l = max_l.max(l);
        }
        let mut sum = 0.0f64;
        for &l in logits {
            sum += (f64::from(l) - f64::from(max_l)).exp();
        }
        let lse = f64::from(max_l) + sum.ln();
        lse - f64::from(logits[target])
    }

    /// At ordinary margins the golden oracle agrees with the incumbent
    /// f64 form to f64 noise (both compute the same number; the forms
    /// differ only in structure).
    #[test]
    fn agrees_with_the_incumbent_f64_form_at_ordinary_margins() {
        let logits = [2.0f32, -1.5, -0.3, -17.0, 0.7];
        let mut s = GoldenNllScratch::default();
        for target in 0..logits.len() {
            let g = s.read_target(&logits, target);
            let inc = incumbent_nll_f64(&logits, target);
            assert!(
                (g.nll - inc).abs() < 1e-12,
                "target {target}: golden {} vs incumbent {inc}",
                g.nll
            );
        }
    }

    /// The resolution demo: at a 100-nat margin the incumbent f64 form
    /// absorbs the tail (its `1 + e^{−m}` addition rounds — the Softmax
    /// Collapse class at the f64 accumulator's 36.7-nat cliff) and reads
    /// EXACTLY 0.0, while the golden oracle resolves the true
    /// ~e^{−100}. This is the delta a lossy readout would have missed.
    #[test]
    fn resolves_the_tail_where_the_incumbent_f64_absorbs() {
        let m = 100.0f64;
        let logits = [0.0f32, -m as f32];
        let mut s = GoldenNllScratch::default();
        let g = s.read_target(&logits, 0);
        assert_eq!(incumbent_nll_f64(&logits, 0), 0.0, "incumbent must absorb");
        let truth = f64::exp(-m).ln_1p();
        assert!(
            (g.nll - truth).abs() < 1e-12,
            "golden {} vs truth {truth:e}",
            g.nll
        );
        assert!((g.top2_margin - m).abs() < 1e-6);
    }

    /// The confidence coordinate: a plain gap reads top-2 = margin; the
    /// band thresholds come from the shared cliff table (fp32 16.64 /
    /// f64 36.73 — asserted here so the module's band doc cannot drift
    /// from the crate's).
    #[test]
    fn margin_coordinate_and_band_thresholds() {
        let logits = [5.0f32, 1.0, -30.0];
        let mut s = GoldenNllScratch::default();
        let g = s.read_target(&logits, 1);
        assert!((g.top2_margin - 4.0).abs() < 1e-6);
        assert!(
            (precision_cliff(FloatFormat::Fp32) - 16.6355).abs() < 0.1,
            "the fp32 band threshold must track the shared cliff table"
        );
        assert!((precision_cliff(FloatFormat::Fp64) - 36.7368).abs() < 0.1);
    }
}
