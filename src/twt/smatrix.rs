//! Streaming S-matrix build over a calibration corpus (Issue 022 T1.3).
//!
//! `S ∈ R^{L×L}` — the expected pairwise discrepancy between layer outputs:
//! `S[i][j]` pools `1 - cos(h_i(x), h_j(x))` over every calibration state
//! pair. Two meters arm the builder:
//!
//! - **Cosine** (always on): [`PairCosineAccum`] pooled moments per pair —
//!   the streaming meter, f64-canonical order.
//! - **SVCCA** (opt-in `with_svcca`): retained stride-subsampled rows fed
//!   to the public `katgpt_core::data_probe::cca::svcca_into` at finalize —
//!   the discrepancy METER is an ARM (riir-train Issue 494 T2.1 measured
//!   cosine against SVCCA as a merge-safety meter; import, don't
//!   re-litigate). Rows are RMS-normalized before the CCA-class meter
//!   (Plan 349 T4.1's recorded pitfall #1) and the probe count is refused
//!   below `d + 1` (pitfall #2: keep n_probe > d).
//!
//! # Arrival order and determinism
//!
//! The forward is walked LAYER-major (one layer's pass over all rows, then
//! the next), so [`SMatrixBuilder::push`] receives `(layer, row, state)` in
//! that order. The builder stages ONE forward's rows in a frame buffer
//! sized at [`begin_forward`](SMatrixBuilder::begin_forward), and
//! [`end_forward`](SMatrixBuilder::end_forward) folds every row's
//! `C(L,2)` pairs in ascending row order — the canonical f64 accumulation
//! order. Same corpus + same forward splits ⇒ bit-identical S (gate-tested).
//!
//! # Position buckets
//!
//! Causal hidden states are position-dependent (Research 594 caveats), so
//! rows carry a [`Bucket`] (default [`Bucket::Prompt`]; deterministic
//! boundaries are the driver's job — the builder only aggregates per
//! bucket). `finalize` emits one S per bucket; the DP runs per bucket (the
//! conservative-min-across-bands reading is the driver's call).

use super::accum::PairCosineAccum;
use super::TwtError;

/// Position band of a calibration row (T1.2: prompt / head / generation,
/// deterministic boundaries — causal states are position-dependent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    Prompt,
    Head,
    Generation,
}

impl Bucket {
    pub const ALL: [Self; 3] = [Self::Prompt, Self::Head, Self::Generation];
    fn index(self) -> usize {
        match self {
            Self::Prompt => 0,
            Self::Head => 1,
            Self::Generation => 2,
        }
    }
}

/// A finalized L×L symmetric discrepancy matrix (row-major, diagonal 0,
/// upper triangle computed and mirrored — symmetry holds BY CONSTRUCTION,
/// the gate asserts it anyway).
#[derive(Debug, Clone)]
pub struct SMatrix {
    n: usize,
    data: Vec<f32>,
}

impl SMatrix {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.n + j]
    }

    /// Synthetic construction for gates and perf probes (T2.2/T2.3) — not
    /// a production path; the diagonal is forced to 0, the fn is only
    /// read for `i < j` in fixed row-major order, and NaN/Inf inputs
    /// panic (the DP requires a finite oracle; `minmax_partition` would
    /// misread NaN as feasible).
    pub fn from_fn(n: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = f(i, j);
                assert!(v.is_finite(), "SMatrix::from_fn: non-finite at ({i},{j})");
                data[i * n + j] = v;
                data[j * n + i] = v;
            }
        }
        Self { n, data }
    }

    /// Upper-triangle iteration (i < j) — the DP's only read pattern.
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        let n = self.n;
        (0..n).flat_map(move |i| ((i + 1)..n).map(move |j| (i, j, self.data[i * n + j])))
    }
}

/// SVCCA-arm retention config: keep rows whose GLOBAL index (monotonic
/// across forwards, corpus order) satisfies `idx % stride == 0`, up to
/// `cap_rows` rows (first-come retention — deterministic; a corpus too
/// small to fill the probe floor is a finalize error, never a silent
/// skip).
#[derive(Debug, Clone, Copy)]
pub struct SvccaCfg {
    pub stride: usize,
    pub cap_rows: usize,
}

/// Finalized matrices: the pooled (cosine) meter per bucket, plus the
/// opt-in SVCCA arm's pooled matrix — BOTH coexist when armed (the meter
/// is an arm; the partition-stability gate reads both).
#[derive(Debug, Clone)]
pub struct SMatrices {
    /// Pooled-cosine matrices per position bucket.
    pub per_bucket: [Option<SMatrix>; 3],
    /// The SVCCA arm's pooled matrix (None when the arm is off).
    pub svcca: Option<SMatrix>,
    /// Rows that entered the pooled (cosine) accumulators.
    pub rows_folded: u64,
    /// Rows retained for the SVCCA arm (0 when the arm is off).
    pub probe_rows: usize,
    /// Layer dimension the builder was constructed with.
    pub dim: usize,
}

impl SMatrices {
    pub fn cosine(&self) -> &SMatrix {
        self.per_bucket[0]
            .as_ref()
            .expect("Prompt bucket: the default bucket of every row")
    }

    pub fn bucket(&self, b: Bucket) -> Option<&SMatrix> {
        self.per_bucket[b.index()].as_ref()
    }

    /// The SVCCA arm's pooled matrix (None unless the arm was armed AND
    /// finalize succeeded past the probe floor).
    pub fn svcca(&self) -> Option<&SMatrix> {
        self.svcca.as_ref()
    }
}

/// Streaming builder — one instance per (checkpoint, corpus, meter set).
pub struct SMatrixBuilder {
    n_layers: usize,
    dim: usize,
    svcca: Option<SvccaCfg>,
    // Pair accumulators: upper triangle, [PairCosineAccum; 3 buckets].
    acc: Vec<[PairCosineAccum; 3]>,
    // Frame staging for the forward in flight.
    frame: Vec<f32>,           // rows × n_layers × dim
    frame_bucket: Vec<Bucket>, // rows
    rows: usize,
    filled: usize, // (row, layer) cells written — layer-major arrival
    forwards_open: u64,
    // SVCCA retention (layer-major frames: slot × n_layers × dim).
    retained: Vec<f32>,
    retained_rows: usize,
    rows_folded: u64,
}

impl SMatrixBuilder {
    pub fn new(n_layers: usize, dim: usize) -> Self {
        let pairs = n_layers * (n_layers - 1) / 2;
        Self {
            n_layers,
            dim,
            svcca: None,
            acc: vec![[const { PairCosineAccum::new() }; 3]; pairs],
            frame: Vec::new(),
            frame_bucket: Vec::new(),
            rows: 0,
            filled: 0,
            forwards_open: 0,
            retained: Vec::new(),
            retained_rows: 0,
            rows_folded: 0,
        }
    }

    /// Arm the SVCCA meter (see [`SvccaCfg`]).
    pub fn with_svcca(mut self, cfg: SvccaCfg) -> Self {
        assert!(cfg.stride >= 1, "svcca stride must be >= 1");
        self.svcca = Some(cfg);
        self
    }

    pub fn svcca_armed(&self) -> bool {
        self.svcca.is_some()
    }

    /// Declare the next forward's row count and (re)size the staging
    /// frame. One allocation per new size — the push/fold path itself is
    /// allocation-free (G4).
    pub fn begin_forward(&mut self, rows: usize) -> Result<(), TwtError> {
        if rows == 0 {
            return Err(TwtError::EmptyForward);
        }
        if self.rows != 0 {
            return Err(TwtError::ForwardAlreadyOpen);
        }
        let cells = rows * self.n_layers * self.dim;
        if self.frame.len() != cells {
            self.frame.clear();
            self.frame.resize(cells, 0.0);
        } else {
            self.frame.fill(0.0);
        }
        self.frame_bucket.clear();
        self.frame_bucket.resize(rows, Bucket::Prompt);
        self.rows = rows;
        self.filled = 0;
        self.forwards_open += 1;
        Ok(())
    }

    /// Override a row's position bucket (default [`Bucket::Prompt`]).
    pub fn set_row_bucket(&mut self, row: usize, bucket: Bucket) -> Result<(), TwtError> {
        if row >= self.rows {
            return Err(TwtError::RowOutOfRange { row, rows: self.rows });
        }
        self.frame_bucket[row] = bucket;
        Ok(())
    }

    /// Fold one (layer, row, state) observation. Zero-allocation.
    pub fn push(&mut self, layer: usize, row: usize, state: &[f32]) -> Result<(), TwtError> {
        if self.rows == 0 {
            return Err(TwtError::NoForwardOpen);
        }
        if layer >= self.n_layers {
            return Err(TwtError::LayerOutOfRange {
                layer,
                layers: self.n_layers,
            });
        }
        if row >= self.rows {
            return Err(TwtError::RowOutOfRange { row, rows: self.rows });
        }
        if state.len() != self.dim {
            return Err(TwtError::DimMismatch {
                got: state.len(),
                want: self.dim,
            });
        }
        let off = (row * self.n_layers + layer) * self.dim;
        if self.frame[off..off + self.dim].iter().any(|&v| v != 0.0) {
            return Err(TwtError::DoubleWrite { layer, row });
        }
        if state.iter().any(|v| !v.is_finite()) {
            return Err(TwtError::NonFiniteState { layer, row });
        }
        self.frame[off..off + self.dim].copy_from_slice(state);

        // SVCCA retention: stride-selected GLOBAL rows (global index =
        // rows folded by earlier forwards + this forward's row index —
        // exact and deterministic given corpus order), first-come up to
        // the cap.
        if let Some(cfg) = &self.svcca {
            let global = self.rows_folded + row as u64;
            if global.is_multiple_of(cfg.stride as u64) {
                // Slot = the row's rank among stride-selected rows —
                // derived from the index, never a running count (a count
                // only advances at the last layer, so every earlier-layer
                // push would overwrite slot 0).
                let slot = (global / cfg.stride as u64) as usize;
                if slot < cfg.cap_rows {
                    let need = (slot + 1) * self.n_layers * self.dim;
                    if self.retained.len() < need {
                        self.retained.resize(need, 0.0);
                    }
                    let dst = slot * self.n_layers * self.dim + layer * self.dim;
                    self.retained[dst..dst + self.dim].copy_from_slice(state);
                    if layer + 1 == self.n_layers {
                        self.retained_rows = self.retained_rows.max(slot + 1);
                    }
                }
            }
        }
        self.filled += 1;
        Ok(())
    }

    /// Fold the staged rows into the pair accumulators (ascending row
    /// order — the canonical accumulation order) and close the forward.
    pub fn end_forward(&mut self) -> Result<(), TwtError> {
        if self.rows == 0 {
            return Err(TwtError::NoForwardOpen);
        }
        if self.filled != self.rows * self.n_layers {
            return Err(TwtError::IncompleteForward {
                got: self.filled,
                want: self.rows * self.n_layers,
            });
        }
        let (n_layers, dim) = (self.n_layers, self.dim);
        for row in 0..self.rows {
            let bucket = self.frame_bucket[row].index();
            let base = row * n_layers * dim;
            for i in 0..n_layers {
                let a = &self.frame[base + i * dim..base + (i + 1) * dim];
                for j in (i + 1)..n_layers {
                    let b = &self.frame[base + j * dim..base + (j + 1) * dim];
                    let pair = i * (2 * n_layers - i - 1) / 2 + (j - i - 1);
                    self.acc[pair][bucket].add(a, b);
                }
            }
            self.rows_folded += 1;
        }
        self.rows = 0;
        self.filled = 0;
        self.forwards_open -= 1;
        Ok(())
    }

    /// Finalize per-bucket cosine matrices + the SVCCA arm (consumes the
    /// builder).
    pub fn finalize(self) -> Result<SMatrices, TwtError> {
        if self.rows != 0 || self.forwards_open != 0 {
            return Err(TwtError::ForwardAlreadyOpen);
        }
        let n = self.n_layers;
        let mut per_bucket: [Option<SMatrix>; 3] = [None, None, None];
        for b in Bucket::ALL {
            let bi = b.index();
            let mut data = vec![0.0f32; n * n];
            let mut any = false;
            for i in 0..n {
                for j in (i + 1)..n {
                    let pair = i * (2 * n - i - 1) / 2 + (j - i - 1);
                    let d = self.acc[pair][bi].distance();
                    data[i * n + j] = d;
                    data[j * n + i] = d;
                    if self.acc[pair][bi].positions() > 0 {
                        any = true;
                    }
                }
            }
            if any {
                per_bucket[bi] = Some(SMatrix { n, data });
            }
        }

        // SVCCA arm: the meter is an arm, and its shape adapter is part of
        // its definition — katgpt-core's `svcca_into` caps sides at
        // `MAX_K = 64` latents (its intended regime), while model states
        // are d ∈ {768, 1024, …}. A SHARED seeded projection d → MAX_K
        // (the SAME basis for every layer, so identical states project to
        // identical rows — the planted-recovery property survives) adapts
        // the shape; RMS-normalize the projected rows (pitfall #1) and
        // refuse a probe set below dim+1 (pitfall #2, the ORIGINAL d —
        // stricter than the projected 64).
        let mut probe_rows = 0usize;
        let mut svcca: Option<SMatrix> = None;
        if self.svcca.is_some() {
            probe_rows = self.retained_rows;
            if probe_rows <= self.dim {
                return Err(TwtError::ProbeFloor {
                    got: probe_rows,
                    need: self.dim + 1,
                });
            }
            let max_k = katgpt_core::data_probe::cca::MAX_K;
            // CCA's honest sample regime: inflation E[ρ_max] ≈ √(2d/n)
            // saturates when d approaches n — the projection dim adapts to
            // the probe count (n ≥ 16k) so independent layers read far and
            // only true agreement reads near 1. Deterministic given the
            // corpus (probe_rows is a count, not a draw).
            let k = (probe_rows / 16).clamp(4, max_k);
            let basis = seeded_projection_basis(self.dim, k);
            let mut mats: Vec<Vec<f32>> = vec![vec![0.0; probe_rows * k]; n];
            for slot in 0..probe_rows {
                for (layer, mat) in mats.iter_mut().enumerate().take(n) {
                    let src = slot * n * self.dim + layer * self.dim;
                    let row = &self.retained[src..src + self.dim];
                    let dst = &mut mat[slot * k..(slot + 1) * k];
                    for (kk, o) in dst.iter_mut().enumerate() {
                        let mut acc = 0.0f64;
                        for (j, &v) in row.iter().enumerate() {
                            acc += (v as f64) * basis[j * k + kk];
                        }
                        *o = acc as f32;
                    }
                }
            }
            for m in &mut mats {
                for slot in 0..probe_rows {
                    rms_normalize(&mut m[slot * k..(slot + 1) * k]);
                }
            }
            let mut scratch = katgpt_core::data_probe::cca::CcaScratch::with_capacity(
                k,
                k,
                probe_rows,
            );
            // Retained rows carry no bucket split (retention is row-level),
            // so the SVCCA meter pools all buckets — the cosine arm owns
            // the bucketed reading.
            let mut data = vec![0.0f32; n * n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let rep = katgpt_core::data_probe::cca::svcca_into(
                        &mats[i],
                        &mats[j],
                        k,
                        k,
                        probe_rows,
                        0.99,
                        1e-4,
                        &mut scratch,
                    );
                    let d = if rep.degenerate {
                        1.0
                    } else {
                        (1.0 - rep.mean_rho).clamp(0.0, 2.0)
                    };
                    data[i * n + j] = d;
                    data[j * n + i] = d;
                }
            }
            svcca = Some(SMatrix { n, data });
        }

        Ok(SMatrices {
            per_bucket,
            svcca,
            rows_folded: self.rows_folded,
            probe_rows,
            dim: self.dim,
        })
    }
}

/// In-place RMS normalization (mean-square over the row); a ~zero row is
/// left untouched (it carries no direction to normalize).
fn rms_normalize(row: &mut [f32]) {
    let sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let r = (sum / row.len() as f64).sqrt();
    if r < 1e-12 {
        return;
    }
    for v in row.iter_mut() {
        *v = (*v as f64 / r) as f32;
    }
}

/// The SVCCA arm's shape adapter: a fixed, seeded `d → k` projection basis
/// (row-major `[d × k]`), the SAME matrix for every layer, so identical
/// states project to identical rows and the planted-recovery property
/// survives the projection. Entries are iid uniform in `[-1, 1]` from an
/// owned LCG — deterministic on every platform, never global RNG. A random
/// projection is the honest shape adapter here (the SVCCA paper's own
/// protocol is reduce-then-CCA; CCA sees through any FIXED linear map
/// shared by both sides up to the projection's rank).
fn seeded_projection_basis(d: usize, k: usize) -> Vec<f64> {
    let mut rng = super::synth::Lcg::new(0x7A6F_6E65_5052_4F4Au64); // 'ZoneProJ'
    (0..d * k).map(|_| (2.0 * rng.next_centered()) as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(builder: &mut SMatrixBuilder, corpus: &[Vec<Vec<f32>>], rows_per_forward: usize) {
        // corpus[row][layer] — feed forwards of `rows_per_forward` rows.
        let n_layers = corpus[0].len();
        for chunk in corpus.chunks(rows_per_forward) {
            builder.begin_forward(chunk.len()).unwrap();
            for layer in 0..n_layers {
                for (r, states) in chunk.iter().enumerate() {
                    builder.push(layer, r, &states[layer]).unwrap();
                }
            }
            builder.end_forward().unwrap();
        }
    }

    fn corpus(rows: usize, layers: usize, dim: usize, seed: u64) -> Vec<Vec<Vec<f32>>> {
        let mut st = seed;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        (0..rows)
            .map(|_| {
                (0..layers)
                    .map(|_| (0..dim).map(|_| draw()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn symmetry_diagonal_and_nonnegativity() {
        let c = corpus(16, 5, 12, 0xABCD);
        let mut b = SMatrixBuilder::new(5, 12);
        feed(&mut b, &c, 16);
        let s = b.finalize().unwrap();
        let m = s.cosine();
        for i in 0..5 {
            assert_eq!(m.get(i, i), 0.0);
            for j in 0..5 {
                assert!(m.get(i, j) >= 0.0);
                assert_eq!(m.get(i, j), m.get(j, i), "symmetry ({i},{j})");
            }
        }
    }

    #[test]
    fn forward_split_does_not_change_bits() {
        let c = corpus(9, 4, 8, 0x1234);
        let one = {
            let mut b = SMatrixBuilder::new(4, 8);
            feed(&mut b, &c, 9);
            b.finalize().unwrap().cosine().clone()
        };
        let split = {
            let mut b = SMatrixBuilder::new(4, 8);
            feed(&mut b, &c, 4);
            b.finalize().unwrap().cosine().clone()
        };
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(one.get(i, j).to_bits(), split.get(i, j).to_bits());
            }
        }
    }

    #[test]
    fn validation_errors() {
        let mut b = SMatrixBuilder::new(3, 4);
        assert!(matches!(b.push(0, 0, &[0.0; 4]), Err(TwtError::NoForwardOpen)));
        b.begin_forward(2).unwrap();
        assert!(matches!(
            b.push(3, 0, &[0.0; 4]),
            Err(TwtError::LayerOutOfRange { .. })
        ));
        assert!(matches!(
            b.push(0, 2, &[0.0; 4]),
            Err(TwtError::RowOutOfRange { .. })
        ));
        assert!(matches!(
            b.push(0, 0, &[0.0; 3]),
            Err(TwtError::DimMismatch { .. })
        ));
        assert!(matches!(
            b.push(0, 0, &[f32::NAN; 4]),
            Err(TwtError::NonFiniteState { .. })
        ));
        b.push(0, 0, &[0.1; 4]).unwrap();
        assert!(matches!(
            b.push(0, 0, &[0.1; 4]),
            Err(TwtError::DoubleWrite { .. })
        ));
        assert!(matches!(
            b.end_forward(),
            Err(TwtError::IncompleteForward { got: 1, want: 6 })
        ));
        b.push(1, 0, &[0.1; 4]).unwrap();
        b.push(2, 0, &[0.1; 4]).unwrap();
        b.push(0, 1, &[0.1; 4]).unwrap();
        b.push(1, 1, &[0.1; 4]).unwrap();
        b.push(2, 1, &[0.1; 4]).unwrap();
        b.end_forward().unwrap();
        assert!(matches!(
            b.begin_forward(0),
            Err(TwtError::EmptyForward)
        ));
    }

    #[test]
    fn svcca_arm_planted_recovery() {
        // Layers 1,2 clones → SVCCA distance ≈ 0 there, ≈ 1 elsewhere;
        // probe rows 64 > dim 12 (the pitfall-2 floor).
        let mut st = 0x5EED_5EED_5EEDu64;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let clone = |l: usize| matches!(l, 1 | 2);
        let corpus: Vec<Vec<Vec<f32>>> = (0..64)
            .map(|_| {
                let base: Vec<f32> = (0..12).map(|_| draw()).collect();
                (0..4)
                    .map(|l| {
                        if clone(l) {
                            base.clone()
                        } else {
                            (0..12).map(|_| draw()).collect::<Vec<_>>()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut b = SMatrixBuilder::new(4, 12)
            .with_svcca(SvccaCfg { stride: 1, cap_rows: 64 });
        feed(&mut b, &corpus, 64);
        let s = b.finalize().unwrap();
        assert_eq!(s.probe_rows, 64);
        let m = s.svcca().unwrap();
        // Threshold above the primitive's NS-7 whitening floor (~1e-3-ish,
        // measured 1.3e-4 at this shape).
        assert!(m.get(1, 2) < 0.01, "clone svcca distance {}", m.get(1, 2));
        assert!(m.get(0, 1) > 0.5, "distinct svcca distance {}", m.get(0, 1));
    }

    #[test]
    fn svcca_probe_floor_refuses() {
        let mut b = SMatrixBuilder::new(3, 8)
            .with_svcca(SvccaCfg { stride: 1, cap_rows: 4 });
        let c = corpus(4, 3, 8, 7);
        feed(&mut b, &c, 4);
        // 4 retained rows ≤ dim 8 → blind meter, refuse.
        assert!(matches!(b.finalize(), Err(TwtError::ProbeFloor { .. })));
    }
}
