//! Min-max DP partition — Plan 616 PROMOTION SHIM (2026-10-02).
//!
//! The partition primitives moved VERBATIM to katgpt-core
//! (`katgpt_core::partition`, feature `minmax_partition`, forwarded by
//! this crate's `twt_profile`): [`minmax_partition`],
//! [`minmax_partition_typed`], [`partition_worst`],
//! [`brute_force_optimal`], [`brute_force_optimal_typed`],
//! [`forced_min_blocks`], [`Block`], plus the pair-distance substrate
//! ([`SMatrix`](crate::twt::smatrix::SMatrix) via the smatrix shim,
//! `PairCosineAccum` via the accum shim) and `PartitionError`. This
//! module re-exports them at their historical paths (every call site —
//! the audition/collapse examples, the gates, the kill rule — resolves
//! unchanged) and keeps ONLY the lane-specific machinery that is not a
//! generic primitive: the T1.6 PRE-REGISTERED ε grid + kill rule, whose
//! pre-registration record is this file's git history (pinned
//! 2026-09-27, BEFORE the first real S — do not re-pin after a
//! measurement; a change is a new pre-registration).
//!
//! # DRY cross-link (T2.1 generalize-or-neighbor)
//!
//! `katgpt-core/src/ugc_schedule.rs` ships `dp_partition(profile, k)` — a
//! contiguous K-block DP with a SUM-of-costs edge oracle at FIXED k. The
//! promoted DP answers a different objective (min-MAX worst case under an
//! ε constraint, k FREE); the T0.1 decision was NEIGHBOR (a generalized
//! form would move every shipped call site), and both module docs now
//! cross-link inside katgpt-core.

pub use katgpt_core::partition::{
    Block, PartitionError, brute_force_optimal, brute_force_optimal_typed, forced_min_blocks,
    minmax_partition, minmax_partition_typed, partition_worst,
};

use super::TwtError;

// ── T1.6 — the pre-registered grid + kill rule (LANE-LOCAL; stays) ───────
//
// Pinned 2026-09-27, BEFORE the first real S was computed (the issue's
// own ordering requirement). Cosine distances live in [0, 2]; on
// held-out synthetic corpora distinct layers read ≈ 1 and true clones
// read exactly 0, so the grid spans sub-clone noise to mid-distances
// and its largest point (1.2) is generous on purpose — the kill rule
// runs AT it.

/// The ε grid every real S is swept over, in ascending order. Never
/// re-pinned after a measurement — a change is a new pre-registration.
pub const PRE_REGISTERED_EPS_GRID: [f32; 7] = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20];

/// Kill fraction of the type-split-forced block count (T1.6, with the
/// achievable-L amendment: laya's bar is 0.8·(2L/3), not 0.8·L).
pub const KILL_FRACTION: f32 = 0.8;

/// A middle block smaller than this contributes nothing mergeable —
/// "only size-1/size-2 blocks in the middle layers" is the second kill
/// clause (T1.6).
pub const KILL_MIN_MIDDLE_BLOCK: usize = 3;

/// The T1.6 verdict at the largest grid ε.
#[derive(Debug, Clone, PartialEq)]
pub enum KillVerdict {
    /// The DP got below the bar AND a middle block is mergeable — the
    /// phase structure is usable; proceed to audition (Phase 3).
    Survives {
        m_at_max_eps: usize,
        bar: usize,
        max_middle_block: usize,
    },
    /// m at max ε ≥ 0.8 × forced — no usable phase structure beyond the
    /// type split. Phase-1 negative (close before any audition work).
    KillBlockCount { m_at_max_eps: usize, bar: usize },
    /// The middle third holds only size-1/2 blocks — nothing mergeable
    /// where depth reduction would pay. Phase-1 negative.
    KillMiddleBlocks { max_middle_block: usize },
}

/// Adjudicate the kill rule against partitions aligned with
/// [`PRE_REGISTERED_EPS_GRID`] (same order — the LAST one is the max-ε
/// partition the rule reads). `forced` = [`forced_min_blocks`] of the
/// lane's type layout.
pub fn kill_verdict(
    partitions_by_eps: &[Vec<Block>],
    forced: usize,
) -> Result<KillVerdict, TwtError> {
    if partitions_by_eps.len() != PRE_REGISTERED_EPS_GRID.len() {
        return Err(TwtError::GridMismatch {
            got: partitions_by_eps.len(),
            want: PRE_REGISTERED_EPS_GRID.len(),
        });
    }
    let p = &partitions_by_eps[partitions_by_eps.len() - 1];
    let m = p.len();
    let bar = (KILL_FRACTION * forced as f32).ceil() as usize;
    if m >= bar {
        return Ok(KillVerdict::KillBlockCount {
            m_at_max_eps: m,
            bar,
        });
    }
    // Middle-third clause: blocks FULLY inside [n/3, 2n/3). A block
    // spanning the whole band is structure (the clause passes); only a
    // middle filled with tiny blocks kills.
    let n = p.last().map(|b| b.end).unwrap_or(0);
    let lo = n / 3;
    let hi = n - n / 3;
    let mut max_middle = 0usize;
    let mut saw_middle = false;
    for b in p {
        if b.start >= lo && b.end <= hi {
            saw_middle = true;
            max_middle = max_middle.max(b.len());
        }
    }
    if saw_middle && max_middle < KILL_MIN_MIDDLE_BLOCK {
        return Ok(KillVerdict::KillMiddleBlocks {
            max_middle_block: max_middle,
        });
    }
    Ok(KillVerdict::Survives {
        m_at_max_eps: m,
        bar,
        max_middle_block: max_middle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twt::smatrix::SMatrix;

    #[test]
    fn kill_verdict_arms() {
        // 6 layers, forced = 6 (all distinct types) → bar = ceil(4.8) = 5.
        let parts = |m: usize| -> Vec<Vec<Block>> {
            PRE_REGISTERED_EPS_GRID
                .iter()
                .map(|_| {
                    (0..m)
                        .map(|k| Block {
                            start: k,
                            end: k + 1,
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        // All singletons at max ε, forced 6 → bar 5 → m=6 ≥ 5 → kill.
        assert_eq!(
            kill_verdict(&parts(6), 6).unwrap(),
            KillVerdict::KillBlockCount {
                m_at_max_eps: 6,
                bar: 5
            }
        );
        // m = 4 < 5, middle band [2, 4) holds only singletons → middle kill.
        assert_eq!(
            kill_verdict(&parts(4), 6).unwrap(),
            KillVerdict::KillMiddleBlocks {
                max_middle_block: 1
            }
        );
        // A spanning block over the middle passes the clause.
        let mut parts_span = parts(4);
        let last = &mut parts_span[PRE_REGISTERED_EPS_GRID.len() - 1];
        *last = vec![
            Block { start: 0, end: 2 },
            Block { start: 2, end: 5 },
            Block { start: 5, end: 6 },
        ];
        assert!(matches!(
            kill_verdict(&parts_span, 6).unwrap(),
            KillVerdict::Survives { .. }
        ));
        // Grid-length mismatch refused.
        assert!(matches!(
            kill_verdict(&parts(4)[..3], 6),
            Err(TwtError::GridMismatch { .. })
        ));
    }

    /// The promoted DP resolves through the shim and answers the same on
    /// the lane's planted shape — the re-export is live, not nominal.
    #[test]
    fn shim_reexport_is_live() {
        let s = SMatrix::from_fn(6, |i, j| {
            let grp = |l: usize| matches!(l, 2..=4);
            if grp(i) && grp(j) { 0.0 } else { 1.0 }
        });
        let p = minmax_partition(&s, 0.1).unwrap();
        assert_eq!(
            p,
            vec![
                Block { start: 0, end: 1 },
                Block { start: 1, end: 2 },
                Block { start: 2, end: 5 },
                Block { start: 5, end: 6 },
            ]
        );
        assert!(matches!(
            minmax_partition(&s, -1.0),
            Err(PartitionError::InvalidEps(_))
        ));
    }
}
