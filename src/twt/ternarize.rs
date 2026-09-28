//! The Phase-4 re-ternarization arms + the T4.2 κ budget (riir-infer
//! Issue 022) — the deterministic materializers that turn a merged block
//! operator `f̄_j` (the Phase-3 audition's `merge_mean` / `merge_rdsc`
//! output, or any dense merged view) into a DEPLOYABLE tensor:
//!
//! - [`TwtArm::DenseF16`] — the f16-dense surrogate (arm A). One dense
//!   operator replaces the block's k ternary matvec layers; the wire type
//!   is F16.
//! - [`TwtArm::SignMajority`] — sign/majority re-ternarize (arm B). The
//!   PINNED convention is the integer code vote: each member contributes
//!   its ternary code `{-1, 0, +1}` (the dequantized value's signum —
//!   exact, because dequant values are exactly `0` or `±f16`), the vote
//!   `Σ codeᵢ` is an integer, and the merged code is `sign(vote)` when
//!   `|vote| ≥ [`ARM_B_TAU_CODE`]` else `0`. The issue's literal
//!   `sign(Σwᵢ)` (scale-weighted) was REJECTED at pre-registration: with
//!   per-layer f16 scales differing wildly, the largest-scale member
//!   dominates the sum for scale reasons, not agreement reasons — the
//!   scale-free vote is the "majority" reading of the arm. The arm's
//!   magnitude still uses the real-valued sum: the group scale is
//!   `d = amax(|Σwᵢ|)` over SUPPORTED positions (nonzero merged code), so
//!   the emitted tensor represents `±d` exactly.
//! - [`TwtArm::SourceQuant`] — source-quantizer re-quant (arm C): the
//!   merged dense values through the SAME formula the checkpoint's
//!   encoder used (the PrismML reference `quantize_row_q2_0_ref`): per
//!   128-weight group `d = amax(|v|)` (emitted as f16), each code
//!   `clamp(round(v / d_f16), -1, +1)` — the divisor is the f16-ROUNDED
//!   scale (the value the wire carries), and the clamp pins that no code
//!   3 (+2d) is ever emitted. On already-ternary input this is a
//!   BIT-EXACT round trip (gate-tested).
//!
//! All arms are deterministic (IEEE fixed order, no RNG) and refuse
//! non-finite input loud. Slices are flat row-major `rows × cols`; shape
//! is a caller-supplied parameter (flat slices do not carry it).
//!
//! ## The T4.2 budget (PRE-REGISTERED)
//!
//! [`KAPPA_BUDGET`] is pinned BEFORE any real-block measurement — this
//! file's git history is the pre-registration. A materialized arm's
//! mapping error on the calibration rows must stay ≤ `κ ×` the f16 dense
//! arm's error on the same rows; an arm above the budget is dead (the
//! GOAT gate never sees it). The comparator is [`budget_ok`]; the
//! operator-level damage proxy an audition-free measurement can run
//! today is [`materialization_rel_err`] (the activation-space half needs
//! the Phase-5 apply path).
//!
//! What does NOT live here: the surrogate-POOL math (`twt::audition`),
//! the apply path, the GGUF writer (`twt::collapse_writer`).

use half::f16;

use super::TwtError;
use crate::quant::q2_0::{BlockQ2_0, Q2oRepackError, Q2_0_BLOCK_SIZE};

/// The T4.2 budget multiple — PRE-REGISTERED before any real-block
/// measurement (Issue 022 T4.2). A materialized arm survives only if its
/// calibration mapping error stays ≤ 2× the f16 dense arm's on the same
/// rows. Rationale: the arms exist to make the surrogate deployable, not
/// to change its answer — materialization damage beyond 2× the f16 floor
/// means the arm is re-deciding the operator, and the Phase-5 GOAT gate
/// (absolute top-1 agreement ≥ 0.9) would inherit a deficit the
/// surrogate never earned. The budget is a fail-loud CEILING on arm
/// selection, never a quality claim — Phase 5 adjudicates quality.
pub const KAPPA_BUDGET: f64 = 2.0;

/// Arm B's integer vote admission threshold (pre-registered alongside κ).
/// `1` = strict majority of nonzero member codes: a position survives
/// only when more members agree on a sign than disagree. `2` (a
/// supermajority read) is the recorded alternative — sweepable per run,
/// pinned here as the default.
pub const ARM_B_TAU_CODE: i32 = 1;

/// Arm vocabulary — the plan-level identity of a materialization (the
/// collapsed-GGUF writer records one code per emitted block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwtArm {
    /// Keep one member layer verbatim (no arm applied — byte-copy).
    Member,
    /// f16-dense surrogate (arm A).
    DenseF16,
    /// Sign/majority re-ternarize (arm B).
    SignMajority,
    /// Source-quantizer re-quant (arm C).
    SourceQuant,
}

impl TwtArm {
    /// The machine-readable code in the `twt.arm_codes` metadata array.
    pub fn code(self) -> u8 {
        match self {
            Self::Member => 0,
            Self::DenseF16 => 1,
            Self::SignMajority => 2,
            Self::SourceQuant => 3,
        }
    }

    /// The legend written beside the codes (a code table without a legend
    /// is a guess — the reader's disambiguation ships in the file).
    pub const LEGEND: &'static str =
        "twt.arm_codes: 0=member_passthrough, 1=dense_f16, 2=sign_majority, 3=source_quant";
}

/// A materialized operator — the arm's output in deployment form.
#[derive(Clone, Debug)]
pub enum Materialized {
    /// Arm A: row-major f16 dense (`rows × cols`).
    DenseF16(DenseF16Weights),
    /// Arms B/C: the ternary group container (bit-planes + f16 scales) —
    /// executable by the stock ternary forward and packable to Q2_0 wire.
    Ternary(Box<katgpt_core::TernaryGroupWeights>),
}

/// Arm A's container: row-major f16 dense weights.
#[derive(Clone, Debug)]
pub struct DenseF16Weights {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f16>,
}

impl Materialized {
    /// The operator's values as dense f32 row-major — the eval view the
    /// error proxies and the budget read (f16→f32 widening is exact; the
    /// ternary dequant is the container's own semantics).
    pub fn to_dense_f32(&self) -> Vec<f32> {
        match self {
            Self::DenseF16(d) => d.data.iter().map(|v| v.to_f32()).collect(),
            Self::Ternary(w) => {
                crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w)
            }
        }
    }

    /// The Q2_0 wire payload (arms B/C, write-into `out`). Arm A has no
    /// Q2_0 payload — the writer emits it as F16 wire directly; this
    /// refuses with [`Q2oRepackError::NotTernary`] rather than silently
    /// doing nothing.
    pub fn pack_q2_0(&self, out: &mut Vec<BlockQ2_0>) -> Result<(), Q2oRepackError> {
        match self {
            Self::DenseF16(_) => Err(Q2oRepackError::NotTernary),
            Self::Ternary(w) => crate::quant::q2_0::pack_ternary_group_to_q2_0(w, out),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::DenseF16(d) => (d.rows, d.cols),
            Self::Ternary(w) => (w.rows, w.cols),
        }
    }
}

fn validate_members(members: &[&[f32]], expected: usize) -> Result<(), TwtError> {
    if members.is_empty() {
        return Err(TwtError::ArmEmptyPool);
    }
    for m in members {
        if m.len() != expected {
            return Err(TwtError::ArmShapeMismatch {
                expected,
                got: m.len(),
            });
        }
        for &v in *m {
            if !v.is_finite() {
                return Err(TwtError::NonFiniteMerged);
            }
        }
    }
    Ok(())
}

/// Arm A — f16-dense surrogate: the members' mean, rounded to f16
/// (round-to-nearest-even; deterministic). Reuses
/// [`super::audition::merge_mean`] so the arm materializes EXACTLY the
/// operator the audition scored.
pub fn arm_dense_f16(members: &[&[f32]], rows: usize, cols: usize) -> Result<Materialized, TwtError> {
    validate_members(members, rows * cols)?;
    let dense = super::audition::merge_mean(members.iter().copied())?;
    let data = dense.iter().map(|&v| f16::from_f32(v)).collect();
    Ok(Materialized::DenseF16(DenseF16Weights { rows, cols, data }))
}

/// Arm C — source-quantizer re-quant of an already-merged dense operator
/// `merged` (`rows × cols` row-major; the `merge_mean`/`merge_rdsc`
/// output).
///
/// Per group of [`Q2_0_BLOCK_SIZE`] (128): `d = amax(|v|)`; codes
/// `clamp(round(v / d_f16), -1, +1)` where `d_f16` is the f16-ROUNDED
/// scale. On ternary input this recovers the source container bit-exactly.
pub fn arm_source_quant(merged: &[f32], rows: usize, cols: usize) -> Result<Materialized, TwtError> {
    if rows == 0 || cols == 0 {
        return Err(TwtError::ArmShapeMismatch {
            expected: rows * cols,
            got: merged.len(),
        });
    }
    if merged.len() != rows * cols {
        return Err(TwtError::ArmShapeMismatch {
            expected: rows * cols,
            got: merged.len(),
        });
    }
    for &v in merged {
        if !v.is_finite() {
            return Err(TwtError::NonFiniteMerged);
        }
    }
    let mut w = katgpt_core::TernaryGroupWeights::new(rows, cols);
    let group = Q2_0_BLOCK_SIZE;
    for r in 0..rows {
        let row_base = r * cols;
        for g in 0..w.groups_per_row {
            let g_lo = g * group;
            let g_hi = (g_lo + group).min(cols);
            // d = amax over the group (may be a partial tail group).
            let mut d = 0.0f32;
            for &v in &merged[row_base + g_lo..row_base + g_hi] {
                let a = v.abs();
                if a > d {
                    d = a;
                }
            }
            let d16 = f16::from_f32(d);
            if !d16.is_finite() {
                return Err(TwtError::NonFiniteScale { row: r, group: g });
            }
            w.group_scale[r * w.groups_per_row + g] = d16;
            if d16.to_f32() == 0.0 {
                continue; // all-zero group: codes stay 0
            }
            let d_f32 = d16.to_f32();
            // Every group spans exactly GROUP_SIZE/64 whole u64 words
            // (the container's alignment law), so the group's first word
            // is `2g` in row-local word coordinates.
            let b0 = r * w.blocks64 + g * (group / 64);
            for (j, &v) in merged[row_base + g_lo..row_base + g_hi].iter().enumerate() {
                // f32::round = round-half-away-from-zero — deterministic;
                // the clamp pins the ternary alphabet (no code 3 ever).
                let q = (v / d_f32).round().clamp(-1.0, 1.0) as i8;
                if q == 0 {
                    continue;
                }
                let word = if j < 64 { b0 } else { b0 + 1 };
                let mask = 1u64 << (j & 63);
                if q > 0 {
                    w.pos_bits[word] |= mask;
                } else {
                    w.neg_bits[word] |= mask;
                }
            }
        }
    }
    Ok(Materialized::Ternary(Box::new(w)))
}

/// Arm B — sign/majority re-ternarize over the dequantized members (see
/// the module doc for the pinned integer code-vote convention).
pub fn arm_sign_majority(
    members: &[&[f32]],
    rows: usize,
    cols: usize,
) -> Result<Materialized, TwtError> {
    validate_members(members, rows * cols)?;
    if rows == 0 || cols == 0 {
        return Err(TwtError::ArmShapeMismatch {
            expected: rows * cols,
            got: rows * cols,
        });
    }
    let mut w = katgpt_core::TernaryGroupWeights::new(rows, cols);
    let group = Q2_0_BLOCK_SIZE;

    // The real-valued per-position sum Σwᵢ (IEEE fixed member order —
    // deterministic; NOT claimed exact, only stable). The group scales
    // read from it; the signs never do.
    let mut sum = vec![0.0f32; rows * cols];
    for m in members {
        for (s, &v) in sum.iter_mut().zip(m.iter()) {
            *s += v;
        }
    }

    let mut codes = [0i8; Q2_0_BLOCK_SIZE];
    for r in 0..rows {
        let row_base = r * cols;
        for g in 0..w.groups_per_row {
            let g_lo = g * group;
            let g_hi = (g_lo + group).min(cols);
            // Pass 1: codes + supported-position amax of |Σw|.
            let mut d = 0.0f32;
            for (j, code) in codes.iter_mut().enumerate().take(g_hi - g_lo) {
                let idx = row_base + g_lo + j;
                let mut vote = 0i32;
                for m in members {
                    // signum of a dequant value: exact (0 or ±f16).
                    let v = m[idx];
                    vote += match v {
                        v if v > 0.0 => 1,
                        v if v < 0.0 => -1,
                        _ => 0,
                    };
                }
                let c = if vote.abs() >= ARM_B_TAU_CODE {
                    vote.signum() as i8
                } else {
                    0
                };
                if c != 0 {
                    let a = sum[idx].abs();
                    if a > d {
                        d = a;
                    }
                }
                *code = c;
            }
            let d16 = f16::from_f32(d);
            if !d16.is_finite() {
                return Err(TwtError::NonFiniteScale { row: r, group: g });
            }
            w.group_scale[r * w.groups_per_row + g] = d16;
            if codes[..g_hi - g_lo].iter().all(|&c| c == 0) {
                continue;
            }
            let b0 = r * w.blocks64 + g * (group / 64);
            for (j, &c) in codes.iter().enumerate().take(g_hi - g_lo) {
                if c == 0 {
                    continue;
                }
                let word = if j < 64 { b0 } else { b0 + 1 };
                let mask = 1u64 << (j & 63);
                if c > 0 {
                    w.pos_bits[word] |= mask;
                } else {
                    w.neg_bits[word] |= mask;
                }
            }
        }
    }
    Ok(Materialized::Ternary(Box::new(w)))
}

/// The T4.2 budget verdict: `arm_err ≤ κ × dense_err` (both mapping
/// errors on the SAME calibration rows, same metric). A NaN on either
/// side refuses (a NaN error is a broken instrument, never a pass);
/// `dense_err == 0` admits the arm only at exactly 0 (an exact
/// materialization — the arm-C ternary round trip).
#[inline]
pub fn budget_ok(arm_err: f64, dense_err: f64) -> bool {
    if arm_err.is_nan() || dense_err.is_nan() {
        return false;
    }
    arm_err <= KAPPA_BUDGET * dense_err
}

/// The budget ratio `arm_err / dense_err` (report-only; the gate reads
/// [`budget_ok`]). `dense_err == 0` → `INFINITY` unless `arm_err == 0`.
#[inline]
pub fn budget_ratio(arm_err: f64, dense_err: f64) -> f64 {
    if dense_err == 0.0 {
        if arm_err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        arm_err / dense_err
    }
}

/// Operator-level materialization-damage proxy, usable WITHOUT a forward:
/// the pooled relative square error of operator `a` against operator `b`
/// over calibration inputs `xs` (`n_x × cols`, row-major):
///
/// `Σ_rows ‖(A − B)x‖² / Σ_rows ‖Bx‖²`
///
/// The T4.2 budget reads this when the baseline is the f16 dense arm and
/// the candidate is a re-ternarized arm, both compared against the SAME
/// f32 merged operator. Scratch buffers (`ya`, `yb`, `rows` each) make
/// the loop allocation-free (the G4 gate holds this).
#[allow(
    clippy::too_many_arguments,
    reason = "the caller-supplied scratch buffers ARE the alloc-free contract; bundling them into a struct would just move the count"
)]
pub fn materialization_rel_err(
    a: &[f32],
    b: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    n_x: usize,
    ya: &mut [f32],
    yb: &mut [f32],
) -> Result<f64, TwtError> {
    if a.len() != rows * cols || b.len() != rows * cols {
        return Err(TwtError::ArmShapeMismatch {
            expected: rows * cols,
            got: a.len().min(b.len()),
        });
    }
    if xs.len() != n_x * cols {
        return Err(TwtError::ArmShapeMismatch {
            expected: n_x * cols,
            got: xs.len(),
        });
    }
    if ya.len() < rows || yb.len() < rows {
        return Err(TwtError::ArmShapeMismatch {
            expected: rows,
            got: ya.len().min(yb.len()),
        });
    }
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for xi in 0..n_x {
        let x = &xs[xi * cols..(xi + 1) * cols];
        dense_matvec(ya, a, x, rows, cols);
        dense_matvec(yb, b, x, rows, cols);
        for j in 0..rows {
            let diff = (ya[j] - yb[j]) as f64;
            num += diff * diff;
            let bv = yb[j] as f64;
            den += bv * bv;
        }
    }
    if den == 0.0 {
        return Err(TwtError::DegenerateBaseline);
    }
    Ok(num / den)
}

/// Row-major f32 matvec, accumulator-ordered (ascending k within the
/// row) — the reduction order is part of the determinism contract.
#[inline]
fn dense_matvec(y: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    for j in 0..rows {
        let row = &w[j * cols..(j + 1) * cols];
        let mut acc = 0.0f32;
        for k in 0..cols {
            acc += row[k] * x[k];
        }
        y[j] = acc;
    }
}
