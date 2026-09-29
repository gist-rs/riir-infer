//! Min-max DP partition over the depthwise discrepancy matrix (Issue 022
//! T2.1) + the pre-registered ε grid and kill rule (T1.6).
//!
//! Objective (TWT §3.1 / Research 594): `min_P (m, max_j S[s_j,e_j])` s.t.
//! `S[s_j,e_j] ≤ ε` — pass 1 minimizes the block count under the ε
//! constraint, pass 2 tie-breaks on the worst-case intra-block discrepancy.
//! Blocks are CONTIGUOUS (depth is ordered) and half-open `[start, end)`.
//!
//! # DRY cross-link (T2.1 generalize-or-neighbor)
//!
//! `katgpt-core/src/ugc_schedule.rs` ships `dp_partition(profile, k)` — a
//! contiguous K-block DP with a SUM-of-costs edge oracle at FIXED k. This
//! DP answers a different objective (min-MAX worst case under an ε
//! constraint, k FREE), so it lands beside it with this cross-link rather
//! than inside it; the two must never drift apart silently. If the PoC
//! floor passes, T6.2 files the single-home promotion to katgpt-core.
//!
//! # Determinism
//!
//! Every comparison is a total-order f32 compare (no sort, no partial_cmp
//! on NaN — NaN in S makes a block infeasible via `>` being false→no,
//! wait: NaN > eps is false, so NaN would read FEASIBLE; therefore S
//! matrices are validated non-NaN at construction). Argmin keeps the first
//! index on ties (index-ordered), so identical inputs give bit-identical
//! partitions on every platform.

use super::smatrix::SMatrix;
use super::TwtError;

/// One contiguous block of layers, half-open `[start, end)` — `end` is
/// EXCLUSIVE (Rust range convention); a block covers layers
/// `start ..= end-1` and its surrogate maps the stream at `start-1` to the
/// stream at `end-1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub start: usize,
    pub end: usize,
}

impl Block {
    /// Number of layers in the block.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// `max S[q][p]` over `i ≤ p < q < j` — the worst intra-block discrepancy
/// for the block `[i, j)`. Returned as a full `n×n` row-major table
/// (`[i*n+j]`, `j ≥ i`; diagonal 0). O(n²) time via the descending-i
/// column-prefix recurrence, O(n²) memory.
fn worst_table(s: &SMatrix) -> Vec<f32> {
    let n = s.n();
    let mut w = vec![0.0f32; n * n];
    for i in (0..n).rev() {
        // `run` = max over q in [i+1, j] of S[q][i] — a column prefix,
        // extended in O(1) as j grows.
        let mut run = 0.0f32;
        for j in (i + 1)..n {
            let v = s.get(j, i);
            run = if v > run { v } else { run };
            let below = w[(i + 1) * n + j];
            w[i * n + j] = if run > below { run } else { below };
        }
    }
    w
}

/// The two-pass min-max DP. Returns the count-optimal, worst-case-minimal
/// contiguous partition. Feasibility monotonicity (worst(i,j) is
/// non-decreasing as the block grows leftward) lets each candidate scan
/// stop at the first infeasible boundary.
pub fn minmax_partition(s: &SMatrix, eps: f32) -> Result<Vec<Block>, TwtError> {
    partition_dp(s, eps, None)
}

/// [`minmax_partition`] under the TYPE constraint (T2.1's tier-(i) rule
/// made structural): a block may only span layers of ONE type, because a
/// GDN layer and an attention layer carry different tensor sets and
/// shapes — there is no operator to average across the boundary. This is
/// the merge-feasible partition: every multi-layer block it emits is a
/// REAL merge candidate (homogeneous), which the unconstrained DP only
/// produced by accident (on the qwen35 interval-4 rhythm it produced
/// almost none — every block spanned an attention layer).
///
/// `types[i]` = true for sliding-window/DeltaNet layers, false for
/// full-attention — the same convention [`forced_min_blocks`] documents.
/// Monotone feasibility carries over: the block `[i, j]` grows leftward
/// in the scans, so the first `types[i] != types[j]` boundary ends the
/// scan (every deeper `i` keeps the mismatched layer `i` inside).
pub fn minmax_partition_typed(
    s: &SMatrix,
    eps: f32,
    types: &[bool],
) -> Result<Vec<Block>, TwtError> {
    if types.len() != s.n() {
        return Err(TwtError::GgufWrite(format!(
            "types length {} != S size {} — the typed partition needs one flag per layer",
            types.len(),
            s.n()
        )));
    }
    partition_dp(s, eps, Some(types))
}

/// The two-pass min-max DP core. `types = None` is the unconstrained
/// original; `Some(types)` restricts every block to one type (see
/// [`minmax_partition_typed`]).
fn partition_dp(s: &SMatrix, eps: f32, types: Option<&[bool]>) -> Result<Vec<Block>, TwtError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(TwtError::InvalidEps(eps));
    }
    let n = s.n();
    if n == 0 {
        return Ok(Vec::new());
    }
    let worst = worst_table(s);
    // The per-scan feasibility guard: a type-constrained scan stops at the
    // first mismatched layer (monotone — see minmax_partition_typed).
    let type_ok = |i: usize, j: usize| -> bool {
        match types {
            None => true,
            Some(t) => t[i] == t[j],
        }
    };

    const INF: usize = usize::MAX;
    // Pass 1 — count[j]: minimum blocks covering [0, j).
    let mut count = vec![INF; n];
    for j in 0..n {
        let mut i = j + 1;
        while i > 0 {
            i -= 1;
            if !type_ok(i, j) || worst[i * n + j] > eps {
                break; // feasibility is monotone in i — everything further left is worse
            }
            let prev = if i == 0 { 0 } else { count[i - 1] };
            if prev == INF {
                continue;
            }
            if prev + 1 < count[j] {
                count[j] = prev + 1;
            }
        }
    }
    debug_assert_ne!(count[n - 1], INF, "eps >= 0: singletons always feasible");

    // Pass 2 — worst_pref[j]: the minimum worst-case discrepancy over
    // count-optimal partitions of [0, j).
    let mut worst_pref = vec![f32::INFINITY; n];
    for j in 0..n {
        let target = count[j] - 1;
        let mut i = j + 1;
        while i > 0 {
            i -= 1;
            if !type_ok(i, j) || worst[i * n + j] > eps {
                break;
            }
            let prev_count = if i == 0 { 0 } else { count[i - 1] };
            if prev_count != target {
                continue;
            }
            let prev_worst = if i == 0 { 0.0 } else { worst_pref[i - 1] };
            let cand = if prev_worst > worst[i * n + j] {
                prev_worst
            } else {
                worst[i * n + j]
            };
            if cand < worst_pref[j] {
                worst_pref[j] = cand;
            }
        }
    }

    // Reconstruct: walk boundaries backward, picking the first boundary
    // achieving the stored optimum (index-ordered argmin — deterministic).
    let mut blocks = Vec::with_capacity(count[n - 1]);
    let mut j = n - 1;
    loop {
        let target = count[j] - 1;
        let target_worst = worst_pref[j];
        let mut i = j + 1;
        let mut chosen = 0usize;
        while i > 0 {
            i -= 1;
            if !type_ok(i, j) || worst[i * n + j] > eps {
                break;
            }
            let prev_count = if i == 0 { 0 } else { count[i - 1] };
            if prev_count != target {
                continue;
            }
            let prev_worst = if i == 0 { 0.0 } else { worst_pref[i - 1] };
            let cand = if prev_worst > worst[i * n + j] {
                prev_worst
            } else {
                worst[i * n + j]
            };
            if cand == target_worst {
                chosen = i;
                break;
            }
        }
        blocks.push(Block { start: chosen, end: j + 1 });
        if chosen == 0 {
            break;
        }
        j = chosen - 1;
    }
    blocks.reverse();
    Ok(blocks)
}

/// Worst intra-block discrepancy of a CONCRETE partition (the G1
/// post-condition arm + the brute-force comparator's objective).
pub fn partition_worst(s: &SMatrix, blocks: &[Block]) -> f32 {
    let mut worst = 0.0f32;
    for b in blocks {
        for p in b.start..b.end {
            for q in (p + 1)..b.end {
                let v = s.get(q, p);
                if v > worst {
                    worst = v;
                }
            }
        }
    }
    worst
}

/// Brute-force lexicographic optimum `(count, worst)` over ALL 2^(L-1)
/// contiguous partitions. Exponential — gate instrument only (T2.2, L ≤ 12).
pub fn brute_force_optimal(s: &SMatrix, eps: f32) -> (usize, f32) {
    let n = s.n();
    assert!(n <= 14, "brute force is 2^(n-1) — gate instrument only");
    let mut best: Option<(usize, f32)> = None;
    for mask in 0u32..(1u32 << (n - 1)) {
        // bit c set ⇒ a cut before layer c+1.
        let mut blocks: Vec<Block> = Vec::with_capacity(n);
        let mut start = 0usize;
        for c in 0..(n - 1) {
            if mask & (1 << c) != 0 {
                blocks.push(Block { start, end: c + 1 });
                start = c + 1;
            }
        }
        blocks.push(Block { start, end: n });
        let mut feasible = true;
        let mut worst = 0.0f32;
        'blk: for b in &blocks {
            for p in b.start..b.end {
                for q in (p + 1)..b.end {
                    let v = s.get(q, p);
                    if v > eps {
                        feasible = false;
                        break 'blk;
                    }
                    if v > worst {
                        worst = v;
                    }
                }
            }
        }
        if !feasible {
            continue;
        }
        let cand = (blocks.len(), worst);
        let better = match best {
            None => true,
            Some((bc, bw)) => bc > cand.0 || (bc == cand.0 && bw > cand.1),
        };
        if better {
            best = Some(cand);
        }
    }
    best.expect("eps >= 0: singleton partition always feasible")
}

/// The typed twin of [`brute_force_optimal`] — the G1 comparator for
/// [`minmax_partition_typed`]. Same enumeration, feasibility adds the
/// same-type requirement per block.
pub fn brute_force_optimal_typed(s: &SMatrix, eps: f32, types: &[bool]) -> (usize, f32) {
    let n = s.n();
    assert!(n <= 14, "brute force is 2^(n-1) — gate instrument only");
    assert_eq!(types.len(), n, "one flag per layer");
    let mut best: Option<(usize, f32)> = None;
    for mask in 0u32..(1u32 << (n - 1)) {
        let mut blocks: Vec<Block> = Vec::with_capacity(n);
        let mut start = 0usize;
        for c in 0..(n - 1) {
            if mask & (1 << c) != 0 {
                blocks.push(Block { start, end: c + 1 });
                start = c + 1;
            }
        }
        blocks.push(Block { start, end: n });
        let mut feasible = true;
        let mut worst = 0.0f32;
        'blk: for b in &blocks {
            for p in b.start..b.end {
                for q in (p + 1)..b.end {
                    let v = s.get(q, p);
                    if v > eps || types[p] != types[q] {
                        feasible = false;
                        break 'blk;
                    }
                    if v > worst {
                        worst = v;
                    }
                }
            }
        }
        if !feasible {
            continue;
        }
        let cand = (blocks.len(), worst);
        let better = match best {
            None => true,
            Some((bc, bw)) => bc > cand.0 || (bc == cand.0 && bw > cand.1),
        };
        if better {
            best = Some(cand);
        }
    }
    best.expect("eps >= 0: singleton partition always feasible")
}

/// The number of blocks the TYPE-SPLIT forces (T1.6 / T2.1): maximal runs
/// of the same layer type. `types[k]` = true for sliding-window layers.
/// Hard-infeasible pairs (distinct RoPE thetas, GDN vs attention) never
/// merge, so runs are the ACHIEVABLE floor — the kill rule measures
/// against this, never against raw L (a constraint-forced m is structure,
/// not absence-of-phase).
pub fn forced_min_blocks(types: &[bool]) -> usize {
    let mut runs = 0usize;
    let mut prev: Option<bool> = None;
    for &t in types {
        if prev != Some(t) {
            runs += 1;
        }
        prev = Some(t);
    }
    runs
}

// ── T1.6 — the pre-registered grid + kill rule ───────────────────────────
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
pub fn kill_verdict(partitions_by_eps: &[Vec<Block>], forced: usize) -> Result<KillVerdict, TwtError> {
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
        return Ok(KillVerdict::KillMiddleBlocks { max_middle_block: max_middle });
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

    fn s_of(n: usize, f: impl FnMut(usize, usize) -> f32) -> SMatrix {
        SMatrix::from_fn(n, f)
    }

    #[test]
    fn dp_recovers_one_planted_block() {
        // Layers 2,3,4 identical (distance 0); everything else ≈ 1.
        let s = s_of(6, |i, j| {
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
    }

    #[test]
    fn count_and_worst_match_brute_force_on_random_matrices() {
        // Deterministic LCG "random" S matrices — total-order compares all
        // the way, so exact agreement is the assertion.
        let mut st: u64 = 0x243F_6A88_85A3_08D3;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32
        };
        for n in [2usize, 5, 8, 11] {
            let s = s_of(n, |i, j| {
                if i == j {
                    0.0
                } else {
                    draw() * 1.5
                }
            });
            for eps in [0.05f32, 0.3, 0.7, 1.2] {
                let p = minmax_partition(&s, eps).unwrap();
                let got = (p.len(), partition_worst(&s, &p));
                let want = brute_force_optimal(&s, eps);
                assert_eq!(got, want, "n={n} eps={eps}");
                for b in &p {
                    for x in b.start..b.end {
                        for y in (x + 1)..b.end {
                            assert!(s.get(y, x) <= eps, "constraint violated n={n} eps={eps}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn m_monotone_non_increasing_in_eps() {
        let mut st: u64 = 0xDEAD_BEEF_CAFE_F00Du64;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32
        };
        let s = s_of(12, |i, j| if i == j { 0.0 } else { draw() });
        let mut prev = usize::MAX;
        for eps in PRE_REGISTERED_EPS_GRID {
            let m = minmax_partition(&s, eps).unwrap().len();
            assert!(m <= prev, "m grew at eps={eps}");
            prev = m;
        }
    }

    #[test]
    fn invalid_eps_refused() {
        let s = s_of(3, |_, _| 0.5);
        assert!(matches!(
            minmax_partition(&s, -0.1),
            Err(TwtError::InvalidEps(_))
        ));
        assert!(matches!(
            minmax_partition(&s, f32::NAN),
            Err(TwtError::InvalidEps(_))
        ));
    }

    #[test]
    fn typed_partition_matches_typed_brute_force_and_enforces_types() {
        // Alternating types + identical layer clones across types: the
        // unconstrained DP merges everything, the typed DP may not —
        // exact agreement with the typed brute force is the assertion.
        let mut st: u64 = 0x0B0B_5EED_1D1E_CAFEu64;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32
        };
        for n in [2usize, 5, 8, 11] {
            let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() * 1.5 });
            let types: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
            for eps in [0.05f32, 0.3, 0.7, 1.2] {
                let p = minmax_partition_typed(&s, eps, &types).unwrap();
                let got = (p.len(), partition_worst(&s, &p));
                let want = brute_force_optimal_typed(&s, eps, &types);
                assert_eq!(got, want, "typed n={n} eps={eps}");
                for b in &p {
                    for x in b.start..b.end {
                        assert_eq!(
                            types[x], types[b.start],
                            "mixed-type block n={n} eps={eps}"
                        );
                        for y in (x + 1)..b.end {
                            assert!(s.get(y, x) <= eps);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn typed_partition_grows_no_fewer_blocks_than_unconstrained() {
        // The type constraint only REMOVES feasible partitions, so m is
        // pointwise ≥ the unconstrained m at every ε.
        let mut st: u64 = 0x1234_ABCD_5678_EF90u64;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32
        };
        let n = 12usize;
        let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() });
        let types: Vec<bool> = (0..n).map(|i| i % 4 != 3).collect();
        for eps in PRE_REGISTERED_EPS_GRID {
            let un = minmax_partition(&s, eps).unwrap().len();
            let ty = minmax_partition_typed(&s, eps, &types).unwrap().len();
            assert!(ty >= un, "typed m {ty} < unconstrained {un} at eps={eps}");
        }
    }

    #[test]
    fn typed_partition_m_monotone_non_increasing_in_eps() {
        let mut st: u64 = 0xFEED_FACE_DADA_5501u64;
        let mut draw = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32
        };
        let n = 12usize;
        let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() });
        let types: Vec<bool> = (0..n).map(|i| i % 4 != 3).collect();
        let mut prev = usize::MAX;
        for eps in PRE_REGISTERED_EPS_GRID {
            let m = minmax_partition_typed(&s, eps, &types).unwrap().len();
            assert!(m <= prev, "typed m grew at eps={eps}");
            prev = m;
        }
    }

    #[test]
    fn typed_partition_refuses_length_mismatch() {
        let s = s_of(3, |_, _| 0.5);
        assert!(minmax_partition_typed(&s, 0.5, &[true, false]).is_err());
    }

    #[test]
    fn typed_partition_bonsai_rhythm_structure() {
        // The qwen35 interval-4 rhythm (3 DeltaNet runs + 1 attention) at
        // generous ε: the typed DP cannot cross types, so every attention
        // layer is its own singleton block — m = 16 singletons + merged
        // GDN triples, strictly less than L.
        let n = 16usize;
        let types: Vec<bool> = (0..n).map(|i| (i + 1) % 4 != 0).collect();
        let s = s_of(n, |i, j| {
            if i == j { 0.0 } else { 0.01 } // distances don't matter: the
            // type constraint, not ε, is what forces the cuts here
        });
        let p = minmax_partition_typed(&s, 0.1, &types).unwrap();
        for b in &p {
            for x in b.start..b.end {
                assert_eq!(types[x], types[b.start]);
            }
        }
        let attn_singletons = p
            .iter()
            .filter(|b| !types[b.start] && b.len() == 1)
            .count();
        assert_eq!(attn_singletons, 4, "every attention layer a singleton");
        assert!(p.len() < n, "merges happened: m {} < {n}", p.len());
    }

    #[test]
    fn forced_min_blocks_counts_runs() {
        // laya's G-S-S pattern over 7 layers: G S S G S S G → 5 runs.
        let types = [false, true, true, false, true, true, false];
        assert_eq!(forced_min_blocks(&types), 5);
        assert_eq!(forced_min_blocks(&[true; 4]), 1);
        assert_eq!(forced_min_blocks(&[]), 0);
    }

    #[test]
    fn kill_verdict_arms() {
        // 6 layers, forced = 6 (all distinct types) → bar = ceil(4.8) = 5.
        let parts = |m: usize| -> Vec<Vec<Block>> {
            PRE_REGISTERED_EPS_GRID
                .iter()
                .map(|_| {
                    (0..m)
                        .map(|k| Block { start: k, end: k + 1 })
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        // All singletons at max ε, forced 6 → bar 5 → m=6 ≥ 5 → kill.
        assert_eq!(
            kill_verdict(&parts(6), 6).unwrap(),
            KillVerdict::KillBlockCount { m_at_max_eps: 6, bar: 5 }
        );
        // m = 4 < 5, middle band [2, 4) holds only singletons → middle kill.
        assert_eq!(
            kill_verdict(&parts(4), 6).unwrap(),
            KillVerdict::KillMiddleBlocks { max_middle_block: 1 }
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
}
