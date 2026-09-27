//! Deterministic synthetic corpora for the TWT gates (Issue 022 T1.5).
//!
//! Own 64-bit LCG — never global RNG (the workspace global-RNG gate) and
//! never `fastrand` (the G1 fixtures must not premise a dependency's
//! algorithm stability; 16 lines of owned LCG are the fixture truth).

/// Explicit-state 64-bit LCG (the knuth constants; never `Default` — a
/// seed is REQUIRED, so an unseeded draw cannot exist).
#[derive(Debug, Clone)]
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in `[-0.5, 0.5)`.
    pub fn next_centered(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    }
}

/// A planted calibration corpus: `corpus[row][layer] -> state`.
///
/// Every layer in a planted group emits the SAME per-row state (exact
/// byte clones — layer `j` in a group is functionally indistinguishable
/// from the group anchor on every input); all other layers emit
/// independent draws. This is the G1 ground truth: intra-group S entries
/// must finalize at exactly 0.0.
pub fn planted_corpus(
    seed: u64,
    n_layers: usize,
    dim: usize,
    rows: usize,
    groups: &[&[usize]],
) -> Vec<Vec<Vec<f32>>> {
    let mut rng = Lcg::new(seed);
    for g in groups {
        for &l in *g {
            assert!(l < n_layers, "planted layer {l} out of range");
        }
    }
    let mut anchor_of = vec![usize::MAX; n_layers];
    for (gi, g) in groups.iter().enumerate() {
        for &l in *g {
            anchor_of[l] = gi;
        }
    }
    let mut row_anchor: Vec<Vec<Vec<f32>>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        // One base draw per group per row + fresh draws elsewhere.
        let mut group_base: Vec<Vec<f32>> = vec![Vec::new(); groups.len()];
        for gb in group_base.iter_mut() {
            *gb = (0..dim).map(|_| rng.next_centered()).collect();
        }
        let mut states: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for anchor in &anchor_of {
            match *anchor {
                usize::MAX => states.push((0..dim).map(|_| rng.next_centered()).collect()),
                gi => states.push(group_base[gi].clone()),
            }
        }
        row_anchor.push(states);
    }
    row_anchor
}

/// Near-clone variant: GROUP members carry small per-layer noise
/// (`σ·draw`) on top of the shared base — the meter-sensitivity fixture
/// (exact zero is the clone case; ε-neighborhoods are the DP's real
/// diet). Unplanted layers stay exact (they are already independent
/// draws).
pub fn planted_corpus_noisy(
    seed: u64,
    n_layers: usize,
    dim: usize,
    rows: usize,
    groups: &[&[usize]],
    sigma: f32,
) -> Vec<Vec<Vec<f32>>> {
    let mut c = planted_corpus(seed, n_layers, dim, rows, groups);
    let mut rng = Lcg::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut planted = vec![false; n_layers];
    for g in groups {
        for &l in *g {
            planted[l] = true;
        }
    }
    for row in &mut c {
        for (l, state) in row.iter_mut().enumerate() {
            if planted[l] {
                for v in state.iter_mut() {
                    *v += sigma * rng.next_centered();
                }
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_groups_are_byte_clones() {
        let c = planted_corpus(42, 5, 8, 6, &[&[1, 2, 3]]);
        for row in &c {
            assert_eq!(row[1], row[2]);
            assert_eq!(row[2], row[3]);
            assert_ne!(row[0], row[1]);
            assert_ne!(row[3], row[4]);
        }
    }

    #[test]
    fn same_seed_same_corpus() {
        let a = planted_corpus(7, 4, 6, 5, &[&[0, 1]]);
        let b = planted_corpus(7, 4, 6, 5, &[&[0, 1]]);
        assert_eq!(a, b);
    }

    #[test]
    fn noisy_variant_is_near_not_exact() {
        let exact = planted_corpus(7, 4, 6, 5, &[&[0, 1]]);
        let noisy = planted_corpus_noisy(7, 4, 6, 5, &[&[0, 1]], 0.01);
        for (r_exact, r_noisy) in exact.iter().zip(&noisy) {
            assert_ne!(r_exact[0], r_noisy[0], "noise must move the bytes");
            assert_ne!(r_exact[1], r_noisy[1]);
            assert_eq!(r_exact[2], r_noisy[2], "unplanted layers untouched");
        }
    }
}
