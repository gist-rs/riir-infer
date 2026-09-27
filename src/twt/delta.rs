//! ΔS quant-damage map (Issue 022 T1.4) — a read-only diagnostic.
//!
//! Profiles the SAME accumulator over the SAME corpus on the parent and a
//! quantized run, then localizes where quantization moved the phase
//! geometry: `ΔS[i][j] = |S_q[i][j] - S_p[i][j]|`. The localization
//! property is the gate — ΔS says WHERE the damage sits, it does not
//! adjudicate the collapse 2×2 (T5.2 does that). Valuable even if the
//! collapse lane dies.

use super::partition::Block;
use super::smatrix::SMatrix;
use super::TwtError;

/// Sorted ΔS report (descending magnitude on the upper triangle).
#[derive(Debug, Clone)]
pub struct DeltaMap {
    pub n: usize,
    /// `(i, j, |ΔS|)` — descending magnitude, ties by (i, j) ascending.
    pub entries: Vec<(usize, usize, f32)>,
    pub max: f32,
    /// RMS of the upper-triangle ΔS values.
    pub rms: f32,
}

/// Build the ΔS map from parent + quantized S matrices (same L).
pub fn delta(parent: &SMatrix, quant: &SMatrix) -> Result<DeltaMap, TwtError> {
    if parent.n() != quant.n() {
        return Err(TwtError::DimMismatch {
            got: quant.n(),
            want: parent.n(),
        });
    }
    let n = parent.n();
    let mut entries: Vec<(usize, usize, f32)> = Vec::with_capacity(n * (n - 1) / 2);
    let mut sum_sq = 0.0f64;
    let mut max = 0.0f32;
    for (i, j, p) in parent.entries() {
        let d = (quant.get(i, j) - p).abs();
        if d > max {
            max = d;
        }
        sum_sq += (d as f64) * (d as f64);
        entries.push((i, j, d));
    }
    entries.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    let cnt = entries.len().max(1) as f64;
    Ok(DeltaMap {
        n,
        max,
        rms: (sum_sq / cnt).sqrt() as f32,
        entries,
    })
}

/// Max ΔS within each block (the per-block damage view); blocks from
/// either profile's partition — the caller picks which partition to read
/// the damage against and records which.
pub fn localize_by_block(delta_map: &DeltaMap, blocks: &[Block]) -> Vec<(Block, f32)> {
    blocks
        .iter()
        .map(|b| {
            let mut worst = 0.0f32;
            for &(i, j, d) in &delta_map.entries {
                if i >= b.start && j < b.end && d > worst {
                    worst = d;
                }
            }
            (*b, worst)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twt::smatrix::SMatrix;

    #[test]
    fn delta_localizes_the_damaged_block() {
        // Parent: one cloned block (2..5). Quant: same, except the
        // intra-block distances moved to 0.4 — damage must localize
        // inside block [2,5).
        let p = SMatrix::from_fn(6, |i, j| {
            let grp = |l: usize| matches!(l, 2 | 3 | 4);
            if grp(i) && grp(j) { 0.0 } else { 1.0 }
        });
        let q = SMatrix::from_fn(6, |i, j| {
            let grp = |l: usize| matches!(l, 2 | 3 | 4);
            if grp(i) && grp(j) {
                0.4
            } else {
                1.0
            }
        });
        let d = delta(&p, &q).unwrap();
        assert_eq!(d.max, 0.4);
        let blocks = vec![
            Block { start: 0, end: 2 },
            Block { start: 2, end: 5 },
            Block { start: 5, end: 6 },
        ];
        let loc = localize_by_block(&d, &blocks);
        assert_eq!(loc[0].1, 0.0);
        assert_eq!(loc[1].1, 0.4);
        assert_eq!(loc[2].1, 0.0);
        // Top entry is an intra-block pair.
        assert!(matches!(d.entries[0], (2, _, 0.4) | (3, _, 0.4) | (2, 4, 0.4)));
    }

    #[test]
    fn dim_mismatch_refused() {
        let a = SMatrix::from_fn(4, |_, _| 0.5);
        let b = SMatrix::from_fn(5, |_, _| 0.5);
        assert!(matches!(delta(&a, &b), Err(TwtError::DimMismatch { .. })));
    }

    #[test]
    fn identical_profiles_zero_delta() {
        let a = SMatrix::from_fn(4, |i, j| (i * 7 + j) as f32 * 0.01);
        let d = delta(&a, &a).unwrap();
        assert_eq!(d.max, 0.0);
        assert_eq!(d.rms, 0.0);
        assert!(d.entries.is_empty() || d.entries[0].2 == 0.0);
    }
}
