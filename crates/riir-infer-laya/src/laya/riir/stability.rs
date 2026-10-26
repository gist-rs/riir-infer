//! HEADLESS decision-depth readout over the capture seam (riir-infer
//! Issue 1106) — the residual-stability analog of the paper's "decision
//! depth" (arXiv:2610.17495: the earliest layer from which the final
//! top prediction stays unchanged).
//!
//! The laya lane ships no LM head (no vocabulary unembed), so lens top-1
//! depth is not computable here. The head-free analog is RESIDUAL-STATE
//! STABILIZATION: the earliest capture stage after which the
//! last-position residual stops moving materially. The headful lens
//! variant stays deferred until a served checkpoint ships a head — its
//! definition (per-stage unembed top-1, first-index depth, disclosed
//! tie band, never silently resolved) is pinned in the issue and
//! riir-reflex `.research/011`.
//!
//! Consumption shape (calibration only, never a serving path — the
//! [`super::encoder::Encoder::forward_capture`] cost class carries):
//!
//! ```text
//! let mut tr = StabilityTracker::with_stage_capacity(n_layers + 1, d);
//! encoder.forward_capture(&backend, &ids, &[ids.len()], &mut |stage, row, state| {
//!     if row == ids.len() - 1 {
//!         tr.observe(stage, state); // the LAST position of the case
//!     }
//! })?;
//! let depth = tr.stabilization_depth(&StabilityBand { cos: 1e-3, l2: 1e-2 });
//! ```
//!
//! The stage ordering is the capture order: `Embedding` is stage 0,
//! `AfterLayer(i)` is stage `i + 1` ([`stage_ordinal`]). A depth value
//! `s` certifies that every subsequent transition (stage `s → s+1 → …
//! → final`) moves the last-position residual by no more than the band;
//! the FINAL stage can never certify (nothing follows it), so the
//! candidate range excludes it — an uncertified read is `None`, never a
//! forced last index.
//!
//! Determinism: every reduction is a sequential f32 fold (plain `+=`,
//! no platform lane, no reassociation) so the same states produce the
//! same deltas bit-for-bit on every host — reflex publishes these
//! numbers beside each other across boxes. Two disclosed conventions:
//! bitwise-equal states yield the EXACT identity delta (`cos = 1.0`,
//! `l2 = 0.0` — `sqrt(na)·sqrt(na)` loses a bit in f32, and identical
//! consecutive stages are the convergence point the read exists for);
//! a zero-norm pair (either state all-zero, not both) yields
//! `cos = 0.0` (no-similarity-evidence; NaN would poison the band
//! comparison silently while the L2 delta still carries signal).

use super::encoder::CaptureStage;

/// One consecutive-stage movement of the last-position residual:
/// from the previous observed stage INTO this one.
///
/// `cos` is the cosine similarity of the two stage states (1.0 =
/// unchanged direction); `l2` is the Euclidean distance between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageDelta {
    pub cos: f32,
    pub l2: f32,
}

/// The pinned "materially stopped moving" band. A stage transition is
/// sub-band when BOTH `cos >= cos_band` (direction settled — near 1.0)
/// and `l2 <= l2_band` (magnitude settled — near 0.0). The consumer
/// pins the values; the substrate never hardcodes a threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilityBand {
    pub cos: f32,
    pub l2: f32,
}

/// The incremental headless stability tracker. One allocation at
/// construction (the previous-stage buffer + the delta series
/// capacity); [`observe`](Self::observe) and
/// [`stabilization_depth`](Self::stabilization_depth) never allocate
/// (the G4 gate pins it).
#[derive(Debug, Clone)]
pub struct StabilityTracker {
    d: usize,
    prev: Vec<f32>,
    have_prev: bool,
    last_ordinal: Option<usize>,
    deltas: Vec<StageDelta>,
}

/// Capture order ordinal of a stage: `Embedding` = 0, `AfterLayer(i)` =
/// `i + 1`. A depth value is one of these ordinals — map it back to a
/// layer with `depth.saturating_sub(1)`.
pub fn stage_ordinal(stage: CaptureStage) -> usize {
    match stage {
        CaptureStage::Embedding => 0,
        CaptureStage::AfterLayer(i) => i + 1,
    }
}

impl StabilityTracker {
    /// Pre-allocates for `stage_capacity` stages of width `d`. The
    /// capture produces exactly `n_layers + 1` stages; a smaller
    /// capacity only costs an amortized growth, never correctness.
    pub fn with_stage_capacity(stage_capacity: usize, d: usize) -> Self {
        Self {
            d,
            prev: vec![0.0; d],
            have_prev: false,
            last_ordinal: None,
            deltas: Vec::with_capacity(stage_capacity.saturating_sub(1)),
        }
    }

    /// Records ONE stage's chosen row (the consumer selects the
    /// last-position row of the case). The stages MUST arrive in
    /// capture order — strictly increasing ordinal, exactly one call
    /// per stage, STARTING at the embedding (ordinal 0: a FULL
    /// capture) — (a `debug_assert!` guards it; the release build trusts
    /// the capture contract, which [`forward_capture`](super::encoder::Encoder::forward_capture)
    /// provides by construction). A feed that filters or offsets stages
    /// reports INDEX-space depth — never feed one.
    ///
    /// Panics (debug builds) on a state whose width differs from the
    /// construction width.
    pub fn observe(&mut self, stage: CaptureStage, state: &[f32]) {
        debug_assert_eq!(state.len(), self.d, "stage state width");
        let ordinal = stage_ordinal(stage);
        debug_assert!(
            self.last_ordinal.is_none_or(|last| ordinal > last),
            "stages must arrive in capture order"
        );
        if !self.have_prev {
            debug_assert_eq!(
                ordinal, 0,
                "first observed stage must be Embedding (ordinal 0) — a partial-capture feed reports index-space depth"
            );
            self.prev[..self.d].copy_from_slice(state);
            self.have_prev = true;
            self.last_ordinal = Some(ordinal);
            return;
        }
        let (mut dot, mut na, mut nb, mut sq) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (a, b) in self.prev[..self.d].iter().zip(state.iter()) {
            dot += a * b;
            na += a * a;
            nb += b * b;
            let diff = a - b;
            sq += diff * diff;
        }
        let denom = na.sqrt() * nb.sqrt();
        // Exact-identity fast path: bitwise-equal states have cosine
        // exactly 1 (sqrt(na)·sqrt(na) loses a bit in f32; identical
        // consecutive stages are THE convergence point, so the identity
        // delta is pinned exact — including the zero→zero case, where
        // nothing moved).
        let cos = if sq == 0.0 {
            1.0
        } else if denom == 0.0 {
            0.0
        } else {
            dot / denom
        };
        let l2 = sq.sqrt();
        self.deltas.push(StageDelta { cos, l2 });
        self.prev[..self.d].copy_from_slice(state);
        self.last_ordinal = Some(ordinal);
    }

    /// The per-transition movement series: entry `i` is the movement
    /// LEAVING stage `i` (between stage `i` and stage `i + 1`), so the
    /// series is one shorter than the stage count.
    pub fn deltas(&self) -> &[StageDelta] {
        &self.deltas
    }

    /// Stages observed so far.
    pub fn stages_observed(&self) -> usize {
        self.deltas.len() + usize::from(self.have_prev)
    }

    /// The stabilization depth: the EARLIEST stage ordinal `s` such
    /// that every transition leaving `s` or later is sub-band — first
    /// index over the suffix scan, per the `float_order` convention.
    /// The returned value is an ORDINAL only when the feed started at
    /// the embedding (ordinal 0 — the `observe` full-capture
    /// precondition; a partial feed returns index-space values).
    /// All transitions sub-band ⇒ `Some(0)` (stable from the
    /// embedding); the final stage is NEVER a candidate — its leaving
    /// transition does not exist, so it could only certify vacuously —
    /// which includes the every-transition-breaks case ⇒ `None`, and
    /// [`Self::max_leaving_delta`] names the blocker.
    pub fn stabilization_depth(&self, band: &StabilityBand) -> Option<usize> {
        // Find the LAST transition that breaks the band; everything
        // after it is the certified stabilized suffix, and its FROM
        // stage is the depth — capped so the depth's own leaving
        // transition exists (the final stage certifies nothing).
        if self.deltas.is_empty() {
            return None;
        }
        let last = self.deltas.len() - 1;
        let mut last_break: Option<usize> = None;
        for (i, d) in self.deltas.iter().enumerate() {
            if !(d.cos >= band.cos && d.l2 <= band.l2) {
                last_break = Some(i);
            }
        }
        match last_break {
            None => Some(0),
            Some(i) if i < last => Some(i + 1),
            Some(_) => None,
        }
    }

    /// The largest leaving transition (by L2, then cosine distance
    /// `1 - cos` as the strict tiebreak) — the disclosure for a `None`
    /// depth: WHERE the residual is still moving, and by how much.
    /// The tie rule is fixed (first index on exact ties) — never
    /// iteration-order dependent.
    pub fn max_leaving_delta(&self) -> Option<(usize, StageDelta)> {
        let mut best: Option<(usize, StageDelta)> = None;
        for (i, d) in self.deltas.iter().enumerate() {
            let better = match best {
                None => true,
                Some((_, b)) => d.l2 > b.l2 || (d.l2 == b.l2 && (1.0 - d.cos) > (1.0 - b.cos)),
            };
            if better {
                best = Some((i, *d));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(i: usize) -> CaptureStage {
        if i == 0 {
            CaptureStage::Embedding
        } else {
            CaptureStage::AfterLayer(i - 1)
        }
    }

    fn feed(tr: &mut StabilityTracker, states: &[&[f32]]) {
        for (i, s) in states.iter().enumerate() {
            tr.observe(stage(i), s);
        }
    }

    /// G1 — hand-computed delta: a = [1, 0], b = [1, 1]. dot = 1,
    /// |a| = 1, |b| = √2 ⇒ cos = 1/√2; L2 = 1 exactly.
    #[test]
    fn delta_matches_hand_computed_cosine_and_l2() {
        let mut tr = StabilityTracker::with_stage_capacity(2, 2);
        feed(&mut tr, &[&[1.0, 0.0], &[1.0, 1.0]]);
        let d = &tr.deltas()[0];
        assert_eq!(d.cos, 1.0 / 2.0f32.sqrt());
        assert_eq!(d.l2, 1.0);
    }

    /// Identical consecutive states: cos = 1, L2 = 0 — the exact
    /// "stopped moving" point.
    #[test]
    fn identical_states_delta_is_exact_identity() {
        let mut tr = StabilityTracker::with_stage_capacity(2, 3);
        feed(&mut tr, &[&[0.5, -1.25, 2.0], &[0.5, -1.25, 2.0]]);
        let d = &tr.deltas()[0];
        assert_eq!(d.cos, 1.0);
        assert_eq!(d.l2, 0.0);
    }

    /// Depth semantics: [big, small, small, small] ⇒ the earliest
    /// all-sub-band suffix starts at stage 1.
    #[test]
    fn depth_is_first_stage_of_the_stable_suffix() {
        let band = StabilityBand {
            cos: 0.999,
            l2: 0.1,
        };
        let mut tr = StabilityTracker::with_stage_capacity(5, 2);
        feed(
            &mut tr,
            &[
                &[1.0, 0.0],
                &[0.0, 5.0],
                &[0.0, 5.001],
                &[0.0, 5.001],
                &[0.0, 5.001],
            ],
        );
        assert_eq!(tr.stabilization_depth(&band), Some(1));
    }

    /// Depth 0: stable from the embedding itself.
    #[test]
    fn depth_zero_when_stable_from_embedding() {
        let band = StabilityBand {
            cos: 0.999,
            l2: 0.1,
        };
        let mut tr = StabilityTracker::with_stage_capacity(3, 2);
        feed(&mut tr, &[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]]);
        assert_eq!(tr.stabilization_depth(&band), Some(0));
    }

    /// Never certifies: the movement series keeps breaking the band ⇒
    /// `None`, and the max-leaving disclosure names the blocker.
    #[test]
    fn depth_none_when_no_suffix_qualifies() {
        let band = StabilityBand {
            cos: 0.999,
            l2: 0.1,
        };
        let mut tr = StabilityTracker::with_stage_capacity(4, 1);
        feed(&mut tr, &[&[0.0], &[1.0], &[0.0], &[1.0]]);
        assert_eq!(tr.stabilization_depth(&band), None);
        let (at, d) = tr.max_leaving_delta().expect("deltas exist");
        assert_eq!(at, 0, "exact L2 ties resolve to the FIRST index");
        assert_eq!(d.l2, 1.0);
    }

    /// The final stage is never a depth candidate: a single stage has
    /// no transitions and certifies nothing.
    #[test]
    fn single_stage_certifies_nothing() {
        let band = StabilityBand {
            cos: 0.0,
            l2: f32::MAX,
        };
        let mut tr = StabilityTracker::with_stage_capacity(1, 2);
        feed(&mut tr, &[&[1.0, 2.0]]);
        assert_eq!(tr.stabilization_depth(&band), None);
        assert_eq!(tr.max_leaving_delta(), None);
        assert_eq!(tr.stages_observed(), 1);
    }

    /// Zero-norm convention: a zero state yields cos = 0.0 (never NaN)
    /// while the L2 delta still carries the movement signal.
    #[test]
    fn zero_norm_state_yields_zero_cosine_not_nan() {
        let mut tr = StabilityTracker::with_stage_capacity(2, 2);
        feed(&mut tr, &[&[0.0, 0.0], &[3.0, 4.0]]);
        let d = &tr.deltas()[0];
        assert_eq!(d.cos, 0.0);
        assert_eq!(d.l2, 5.0);
    }

    /// G1 — parity vs an independently coded brute-force recompute:
    /// seeded pseudo-random stage vectors, deltas recomputed from the
    /// retained vectors with a second implementation (per-metric two
    /// passes instead of the tracker's fused one-pass folds) must match
    /// BIT-FOR-BIT, and the depth must equal a naive suffix scan.
    #[test]
    fn parity_with_brute_force_recompute_on_seeded_fixture() {
        // Deterministic LCG (no unseeded global draws).
        let mut seed: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX >> 1) as f32 - 1.0
        };
        let (s_count, d) = (7usize, 16usize);
        let states: Vec<Vec<f32>> = (0..s_count)
            .map(|_| (0..d).map(|_| next()).collect())
            .collect();
        // Late-stage stabilization: the tail states converge exactly.
        let states: Vec<Vec<f32>> = states
            .into_iter()
            .enumerate()
            .map(|(i, mut v)| {
                if i >= 5 {
                    v.iter_mut().for_each(|x| *x = 0.25);
                }
                v
            })
            .collect();

        let mut tr = StabilityTracker::with_stage_capacity(s_count, d);
        let refs: Vec<&[f32]> = states.iter().map(|v| v.as_slice()).collect();
        feed(&mut tr, &refs);

        // Brute-force recompute (independent shape: two passes per
        // metric over cloned f64-free f32 math, same fold ORDER, same
        // exact-identity + zero-norm conventions).
        let mut expected: Vec<StageDelta> = Vec::with_capacity(s_count - 1);
        for w in states.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..d {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            let denom = na.sqrt() * nb.sqrt();
            let mut sq = 0.0f32;
            for i in 0..d {
                sq += (a[i] - b[i]) * (a[i] - b[i]);
            }
            let cos = if sq == 0.0 {
                1.0
            } else if denom == 0.0 {
                0.0
            } else {
                dot / denom
            };
            expected.push(StageDelta { cos, l2: sq.sqrt() });
        }
        assert_eq!(tr.deltas(), expected.as_slice(), "delta series bit-parity");

        // Naive suffix scan for the depth.
        let band = StabilityBand { cos: 0.9, l2: 0.5 };
        let sub = |dl: &StageDelta| dl.cos >= band.cos && dl.l2 <= band.l2;
        let mut want: Option<usize> = None;
        for s in 0..expected.len() {
            if expected[s..].iter().all(sub) {
                want = Some(s);
                break;
            }
        }
        assert_eq!(tr.stabilization_depth(&band), want);
        assert_eq!(want, Some(5), "fixture pins the stabilized tail");
    }

    /// Capture-order guard (debug builds): an out-of-order stage is a
    /// caller bug and must trip the assert, not silently produce a
    /// meaningless delta series.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "capture order")]
    fn out_of_order_stage_trips_the_debug_guard() {
        let mut tr = StabilityTracker::with_stage_capacity(3, 1);
        tr.observe(CaptureStage::Embedding, &[1.0]);
        tr.observe(CaptureStage::AfterLayer(0), &[2.0]);
        tr.observe(CaptureStage::Embedding, &[3.0]);
    }

    /// First-ordinal guard (debug builds): the feed must START at the
    /// embedding — a partial/offset capture (e.g. starting at
    /// `AfterLayer(30)`) passes the order guard and would report
    /// index-space depth.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "first observed stage must be Embedding")]
    fn first_ordinal_must_be_embedding_trips_the_debug_guard() {
        let mut tr = StabilityTracker::with_stage_capacity(3, 1);
        tr.observe(CaptureStage::AfterLayer(1), &[1.0]);
    }
}
