//! Issue 027 / Plan 615 — the offline LUT grid solver (Lloyd-Max under our
//! per-128 f16 single-level scale).
//!
//! UNGATED pure math (the `dq_fakequant` spec precedent): the solver, the
//! histogram and the evaluator are lib-tested at default features. The BIN
//! (`lut_grid_solve`, feature `lut_grid`) does artifact extraction + eval.
//!
//! The grid is FOUR levels in d-units, sign-free: `{l0, 0, l2, +2}`. The
//! zero level and the `+2` anchor are PINNED — the anchor is what makes the
//! sign-absorbing scale well-defined (the block's largest-magnitude element
//! must land on it), and zero is the format's structural level. The free
//! levels `l0 < 0` and `0 < l2 < 2` are what Lloyd-Max optimizes. The
//! encoder absorbs the sign: `d = sign(argmax|w|)·f16(amax/2)` flips the
//! decoded ladder so the real-world side holding the block max gets the
//! double step.
//!
//! Determinism: the solver is a pure function of (histogram, init) — fixed
//! f64 arithmetic, fixed iteration order, fixed iteration cap — and the
//! output is BLAKE3-committed (`grid_digest`) so a cross-box re-run
//! (M3 ↔ 4090) verifies byte-identity.

use blake3::Hasher;

/// One quantization grid: 4 levels in d-units. `l0 < 0 < l2 < 2`; the zero
/// level and the `+2` anchor are structural (pinned).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q2Grid {
    /// The negative free level (d-units).
    pub l0: f32,
    /// The positive free level (d-units).
    pub l2: f32,
}

/// The incumbent T0 uniform grid `{−1, 0, +1, +2}` — what the round-clamp
/// encoder emits, and the Lloyd-Max initialization (EM starts from the
/// incumbent so measured improvement over T0 is monotone by construction).
pub const GRID_T0_UNIFORM: Q2Grid = Q2Grid { l0: -1.0, l2: 1.0 };

/// Histogram bin count over `[BIN_LO, BIN_HI]`.
pub const HISTOGRAM_BINS: usize = 4096;

/// Histogram range. `x = w/d` under the T0 scale rule is bounded by ±2 plus
/// the f16 rounding excursion of `d` (≈ +1.5% at the anchor), so ±2.125
/// covers the support with slack; out-of-range values clamp into the edge
/// bins (conservative — they only ever shrink the free levels' mass).
pub const BIN_LO: f64 = -2.125;
pub const BIN_HI: f64 = 2.125;

/// Empirical weight distribution, pooled over blocks: fixed-bin f64 MASS.
///
/// Seeded from real weights via the bin's extraction pass; the solver is a
/// pure function of this. Each sample carries a WEIGHT — the extraction
/// pass uses `d²` (the block scale squared) so the solver's objective is
/// the ENERGY-weighted MSE `Σ d²·(x − ℓ(x))²` — the same quantity the
/// round-trip SNR `Σw²/Σ(w−q)²` measures. (Measured 2026-10-01: an
/// unweighted pool solves a grid that is −15.7% on the histogram yet
/// LOSES ~0.7 dB to T0 on real energy-weighted SNR — the objective must
/// match the metric.)
#[derive(Clone, Debug)]
pub struct WeightHistogram {
    pub counts: Vec<f64>,
    pub total: f64,
}

impl WeightHistogram {
    /// Empty histogram with the fixed bin geometry.
    pub fn new() -> Self {
        Self {
            counts: vec![0.0; HISTOGRAM_BINS],
            total: 0.0,
        }
    }

    /// Add one normalized sample (clamped into the edge bins), unit mass.
    #[inline]
    pub fn record(&mut self, x: f64) {
        self.record_weighted(x, 1.0);
    }

    /// Add one sample with an explicit mass (the extraction pass uses the
    /// block's `d²` — see the type doc).
    #[inline]
    pub fn record_weighted(&mut self, x: f64, mass: f64) {
        let frac = (x - BIN_LO) / (BIN_HI - BIN_LO);
        let bin = if frac <= 0.0 {
            0usize
        } else if frac >= 1.0 {
            HISTOGRAM_BINS - 1
        } else {
            (frac * HISTOGRAM_BINS as f64) as usize
        };
        self.counts[bin] += mass;
        self.total += mass;
    }

    /// Midpoint value of bin `b` (the discretization the solver optimizes
    /// over; 4096 bins ≪ block granularity, so binning error is negligible
    /// against the level spacings it resolves).
    #[inline]
    pub fn bin_center(&self, b: usize) -> f64 {
        BIN_LO + (b as f64 + 0.5) * (BIN_HI - BIN_LO) / HISTOGRAM_BINS as f64
    }
}

impl Default for WeightHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision-boundary index for level `l` against its neighbor `r` (both in
/// d-units): the first bin whose center reaches the midpoint.
#[inline]
fn boundary_bin(l: f64, r: f64) -> usize {
    let mid = (l + r) / 2.0;
    let frac = (mid - BIN_LO) / (BIN_HI - BIN_LO);
    let b = (frac * HISTOGRAM_BINS as f64).ceil() as isize;
    b.clamp(0, HISTOGRAM_BINS as isize) as usize
}

/// Assign each bin to its nearest level (ties → the LOWER level, matching
/// the encoder's lowest-code tie-break); returns region mass sums and the
/// region count, plus the squared-error sum.
struct Regions {
    /// [count, sum_of_center] per region, regions ordered l0, 0, l2, +2.
    count: [f64; 4],
    sum: [f64; 4],
    sq_err: f64,
}

fn assign_regions(hist: &WeightHistogram, grid: Q2Grid) -> Regions {
    let levels = [grid.l0 as f64, 0.0, grid.l2 as f64, 2.0];
    // Boundaries between adjacent levels, as bin indices: region r spans
    // [bound[r], bound[r+1]).
    let mut bound = [0usize; 5];
    bound[0] = 0;
    bound[1] = boundary_bin(levels[0], levels[1]);
    bound[2] = boundary_bin(levels[1], levels[2]);
    bound[3] = boundary_bin(levels[2], levels[3]);
    bound[4] = HISTOGRAM_BINS;

    let mut out = Regions {
        count: [0.0; 4],
        sum: [0.0; 4],
        sq_err: 0.0,
    };
    for r in 0..4 {
        for b in bound[r]..bound[r + 1] {
            let c = hist.counts[b];
            if c == 0.0 {
                continue;
            }
            let center = hist.bin_center(b);
            let f = c;
            out.count[r] += f;
            out.sum[r] += f * center;
            let d = center - levels[r];
            out.sq_err += f * d * d;
        }
    }
    out
}

/// Solver statistics (deterministic; part of the committed artifact).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveStats {
    pub iterations: usize,
    /// MSE (d-units²) on the histogram at the STARTING (T0) grid.
    pub mse_start: f64,
    /// MSE at the solved grid.
    pub mse_end: f64,
}

/// Lloyd-Max EM over the empirical histogram with `{0, +2}` pinned.
///
/// Starts from [`GRID_T0_UNIFORM`] (the incumbent), alternates
/// nearest-level assignment and centroid updates of the two free levels,
/// stops at `|Δl0| + |Δl2| < 1e−7` or 256 iterations. An empty region keeps
/// its previous level (a degenerate histogram slice must not collapse a
/// level onto another). Pure f64 accumulation, narrowed to f32 once per
/// update — bit-identical across machines for the same histogram.
pub fn solve_lloyd_max(hist: &WeightHistogram) -> (Q2Grid, SolveStats) {
    let mut grid = GRID_T0_UNIFORM;
    let mse_start = histogram_mse(hist, grid);
    let mut stats = SolveStats {
        iterations: 0,
        mse_start,
        mse_end: mse_start,
    };
    if hist.total == 0.0 {
        return (grid, stats);
    }

    for _ in 0..256 {
        let regions = assign_regions(hist, grid);
        // Centroid of region 0 → new l0; region 2 → new l2. Empty region
        // keeps the incumbent level.
        let new_l0 = if regions.count[0] > 0.0 {
            (regions.sum[0] / regions.count[0]) as f32
        } else {
            grid.l0
        };
        let new_l2 = if regions.count[2] > 0.0 {
            (regions.sum[2] / regions.count[2]) as f32
        } else {
            grid.l2
        };
        // Structural ordering guard: l0 < 0 < l2 < 2. A centroid crossing a
        // pinned level means the histogram slice is degenerate for that
        // level; clamp inside the valid open interval (one ulp inside 0/2).
        let new_l0 = new_l0.clamp(-2.0, -f32::MIN_POSITIVE);
        let new_l2 = new_l2.clamp(f32::MIN_POSITIVE, 2.0 - f32::EPSILON * 2.0);

        let delta = (new_l0 - grid.l0).abs() + (new_l2 - grid.l2).abs();
        grid = Q2Grid {
            l0: new_l0,
            l2: new_l2,
        };
        stats.iterations += 1;
        if delta < 1e-7 {
            break;
        }
    }
    stats.mse_end = histogram_mse(hist, grid);
    (grid, stats)
}

/// Per-bin energy-weighted MSE of the histogram under the grid's
/// nearest-level mapping (mass units — comparable across grids, not
/// across histograms).
pub fn histogram_mse(hist: &WeightHistogram, grid: Q2Grid) -> f64 {
    if hist.total == 0.0 {
        return 0.0;
    }
    assign_regions(hist, grid).sq_err / hist.total
}

/// BLAKE3 commitment over (histogram geometry + counts, solved levels,
/// stats) — the cross-box determinism gate. Two boxes that disagree on any
/// bit produce different digests.
pub fn grid_digest(hist: &WeightHistogram, grid: Q2Grid, stats: &SolveStats) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(&HISTOGRAM_BINS.to_le_bytes());
    h.update(&BIN_LO.to_le_bytes());
    h.update(&BIN_HI.to_le_bytes());
    for c in &hist.counts {
        h.update(&c.to_le_bytes());
    }
    h.update(&grid.l0.to_le_bytes());
    h.update(&grid.l2.to_le_bytes());
    h.update(&stats.iterations.to_le_bytes());
    h.update(&stats.mse_start.to_le_bytes());
    h.update(&stats.mse_end.to_le_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(h.finalize().as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-point distribution exactly at {-1, +1}: T0 is optimal, EM holds.
    #[test]
    fn two_point_mass_at_t0_levels_is_a_fixed_point() {
        let mut h = WeightHistogram::new();
        for _ in 0..1000 {
            h.record(-1.0);
            h.record(1.0);
        }
        let (grid, stats) = solve_lloyd_max(&h);
        assert!(
            (grid.l0 + 1.0).abs() < 0.05 && (grid.l2 - 1.0).abs() < 0.05,
            "two-point at ±1 should stay near ±1, got {grid:?}"
        );
        assert!(stats.mse_end <= stats.mse_start + 1e-12);
    }

    /// Symmetric zero-mean histogram: free levels stay symmetric to each
    /// other (l0 ≈ −l2), because the region geometry is symmetric.
    #[test]
    fn symmetric_histogram_keeps_symmetric_free_levels() {
        let mut h = WeightHistogram::new();
        // Triangular mass centered at 0.
        for i in -500..=500isize {
            let w = 1000 - i.unsigned_abs();
            for _ in 0..w {
                h.record(i as f64 / 500.0);
            }
        }
        let (grid, _) = solve_lloyd_max(&h);
        assert!(
            (grid.l0 + grid.l2).abs() < 0.02,
            "symmetric histogram should give symmetric free levels, got {grid:?}"
        );
        assert!(grid.l2 < 2.0 && grid.l2 > 0.0);
    }

    /// Anchor-heavy histogram (mass at +2, the T0 anchor excursion): the
    /// solved grid must not be WORSE than T0 — EM is monotone from the
    /// incumbent.
    #[test]
    fn solved_mse_never_exceeds_t0() {
        let mut h = WeightHistogram::new();
        // Gaussian-ish body + one anchor point per block.
        let mut state: u64 = 0x243F_6A88_85A3_08D3;
        for _ in 0..200_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Box-Muller-lite: sum of 3 uniforms − 1.5 ≈ N(0, 0.5).
            let mut s = 0.0;
            for _ in 0..3 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s += ((state >> 11) as f64) / ((1u64 << 53) as f64);
            }
            h.record((s - 1.5) * 0.6);
        }
        for _ in 0..1600 {
            h.record(2.0);
        }
        let (grid, stats) = solve_lloyd_max(&h);
        assert!(
            stats.mse_end <= stats.mse_start,
            "EM must be monotone from T0: start {} end {}",
            stats.mse_start,
            stats.mse_end
        );
        assert!(grid.l0 < 0.0 && grid.l2 > 0.0 && grid.l2 < 2.0);
    }

    /// Determinism: two solves on identical histograms are bit-identical,
    /// and the digest is stable.
    #[test]
    fn solve_is_bit_identical_across_runs() {
        let mk = || {
            let mut h = WeightHistogram::new();
            for i in 0..10_000 {
                h.record(((i * 2654435761u64 % 9973) as f64 / 9973.0 - 0.5) * 3.0);
            }
            h
        };
        let (g1, s1) = solve_lloyd_max(&mk());
        let (g2, s2) = solve_lloyd_max(&mk());
        assert_eq!(g1.l0.to_bits(), g2.l0.to_bits());
        assert_eq!(g1.l2.to_bits(), g2.l2.to_bits());
        assert_eq!(s1, s2);
        let h = mk();
        let (g3, s3) = solve_lloyd_max(&h);
        assert_eq!(
            grid_digest(&h, g1, &s1),
            grid_digest(&h, g3, &s3),
            "digest must be a pure function of (histogram, grid, stats)"
        );
    }

    /// Empty histogram: returns the T0 incumbent, zero iterations.
    #[test]
    fn empty_histogram_returns_incumbent() {
        let h = WeightHistogram::new();
        let (grid, stats) = solve_lloyd_max(&h);
        assert_eq!(grid, GRID_T0_UNIFORM);
        assert_eq!(stats.iterations, 0);
    }

    /// Plan 615 probe — the bin's exact pipeline on synthetic Gaussian
    /// weights: per-block T0 scale, d²-weighted histogram, solve, then TRUE
    /// element-wise MSE of T0 vs solved on the same samples. The histogram
    /// objective and the true MSE must AGREE in direction (the live run
    /// contradicted — this test is where the contradiction reproduces).
    #[test]
    fn weighted_histogram_objective_tracks_true_mse() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Sum of 3 uniforms − 1.5 ≈ Gaussian-ish.
            let mut s = 0.0f32;
            for _ in 0..3 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s += ((state >> 40) & 0xFFFFFF) as f32 / 0x1000000 as f32;
            }
            (s - 1.5) * 0.4
        };
        let blocks = 8_000;
        let w: Vec<f32> = (0..blocks * 128).map(|_| next()).collect();

        // The bin's histogram pass, verbatim.
        let mut hist = WeightHistogram::new();
        for block in w.chunks(128) {
            let Some((_, d)) = crate::quant::q2_0::t0_block_scale(block) else {
                continue;
            };
            let mass = (d as f64) * (d as f64);
            for &v in block {
                hist.record_weighted((v / d) as f64, mass);
            }
        }
        let (grid, stats) = solve_lloyd_max(&hist);

        // True element-wise MSE, both arms — from the DEFINITION (region
        // logic recomputed here, not via the encoders, so encoder bugs and
        // histogram bugs cannot cancel).
        let true_mse = |g: Q2Grid| -> f64 {
            let levels = [g.l0 as f64, 0.0, g.l2 as f64, 2.0];
            let mut err = 0f64;
            for block in w.chunks(128) {
                let Some((_, d)) = crate::quant::q2_0::t0_block_scale(block) else {
                    continue;
                };
                let df = d as f64;
                for &v in block {
                    let x = v as f64 / df;
                    // nearest level, ties → lowest
                    let mut best = 0usize;
                    let mut bd = f64::INFINITY;
                    for (i, &l) in levels.iter().enumerate() {
                        let dist = (x - l).abs();
                        if dist < bd {
                            bd = dist;
                            best = i;
                        }
                    }
                    let e = x - levels[best];
                    err += df * df * e * e;
                }
            }
            err / w.len() as f64
        };
        let mse_t0 = true_mse(GRID_T0_UNIFORM);
        let mse_solved = true_mse(grid);
        println!(
            "probe: grid l0={} l2={} iters={} hist {:#.3e}→{:#.3e} true t0 {mse_t0:.3e} solved {mse_solved:.3e}",
            grid.l0, grid.l2, stats.iterations, stats.mse_start, stats.mse_end
        );
        assert!(
            mse_solved <= mse_t0,
            "solved grid must not lose to T0 on TRUE mse: {mse_solved} vs {mse_t0}"
        );
    }

    /// Empty-region guard: a histogram with no mass below the l0 boundary
    /// keeps l0 at its incumbent (no collapse onto 0).
    #[test]
    fn empty_negative_region_keeps_incumbent_l0() {
        let mut h = WeightHistogram::new();
        for i in 0..10_000 {
            h.record(0.1 + (i % 100) as f64 / 1000.0); // strictly positive mass
        }
        let (grid, _) = solve_lloyd_max(&h);
        assert_eq!(grid.l0, GRID_T0_UNIFORM.l0, "empty region 0 must keep the incumbent l0");
    }
}
