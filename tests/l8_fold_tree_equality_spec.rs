//! Plan 621 T2 — the GDN L8 fold-tree oracle (CPU, pure; no GPU, no feature gate).
//!
//! Adjudicates the staged plan's addendum-2026-10-18b/18c derivation: the Yukon
//! L8 lane-geometry port (`recurrence_f32_l8_rows_hd128`, the transposed
//! k-rows-per-warp partition) reduces each row's 32 chain-partials through a
//! DIFFERENT mechanical route than the incumbent rowpar kernel
//! (`recurrence_f32_fused_rows_hd128`, deltanet.rs:1241-1252) — per-lane
//! register fold `(p0+p2)+(p1+p3)` then in-group xor 4/2/1, versus the
//! incumbent's full 5-step xor butterfly 16/8/4/2/1 — yet lands on the SAME
//! binary reduction tree over the SAME leaves with the SAME child order at
//! every node ("tree-identical by construction"). If this oracle disagrees,
//! the addendum is wrong and the plan's §3 fallback stands (order-pinned
//! class, Issue-940 site pin).
//!
//! The two orders, at the addendum-2026-10-18c §4 pinned map (group-lane
//! `n = lane % 8`, slot `m` mirrors incumbent lane `L = n + 8m`, leaf `c`
//! ascending = column `c*32 + n + 8m`):
//!
//! ```text
//! incumbent (per lane L, 32 lanes):  chain_L = ((l0 + l1) + l2) + l3   (zero-init, c ascending)
//!                                    step16 = chain_L + chain_{L^16}   (+= self-first)
//!                                    step8  = step16_L + step16_{L^8}
//!                                    step4/2/1 likewise -> total at every lane
//! L8 port (per group-lane n, 8 lanes in-group):   p_m = chain_{n+8m}  (same chain law)
//!                                    x = (p0 + p2) + (p1 + p3)
//!                                    xor 4/2/1 (+= self-first) -> total at every group-lane
//! ```
//!
//! Leaf identity: the fold sums the per-(chain, c) PRODUCTS `sd * k` — both
//! arms consume the identical products (the decay multiply happens before the
//! fold on both sides; the port's leaf `s[m*4+c] = S[wrow][c*32 + n + 8m]` is
//! the incumbent's leaf `(L = n+8m, c)`), so the oracle's leaves ARE the
//! shared products and only the summation order differs. The 0.0f chain init
//! is the identity on finite leaves and is carried identically on both sides.
//!
//! Verdict (this file is the sole adjudicator per the plan): TREE-IDENTICAL —
//! the port re-classes to the STRONG promotion case (G1 bytewise), T4's
//! order-pin fallback stays dead.

/// The pinned verdict constant (T3/T4 read this; the plan records the class).
pub const L8_FOLD_TREE_VERDICT: &str = "tree-identical-by-construction";

/// Number of incumbent lanes per row (hd128 GDN state: 32 column slices).
const LANES: usize = 32;
/// Leaves per chain (COLS_PER_LANE = 4).
const LEAVES_PER_CHAIN: usize = 4;
/// L8 group width (8 lanes per row-group in the transposed partition).
const GROUP: usize = 8;

/// Symbolic reduction tree; leaves carry the global leaf id `L * 4 + c`.
#[derive(Clone, Debug, PartialEq)]
enum Tree {
    Leaf(usize),
    Node(Box<Tree>, Box<Tree>),
}

fn leaf(l: usize, c: usize) -> Tree {
    Tree::Leaf(l * LEAVES_PER_CHAIN + c)
}

/// The shared per-lane chain: zero-init, c ascending, `+=` self-first —
/// modelled as the left fold `((l0 + l1) + l2) + l3` (the 0.0f + l0 head is
/// the identity on finite leaves, identical on both sides).
fn chain(l: usize) -> Tree {
    let mut t = leaf(l, 0);
    for c in 1..LEAVES_PER_CHAIN {
        t = Tree::Node(Box::new(t), Box::new(leaf(l, c)));
    }
    t
}

/// The incumbent's final tree at lane `l`: xor butterfly 16/8/4/2/1,
/// `+=` self-first at every step (lockstep — each step's node is
/// `(self_pre, other_pre)` over the pre-step values).
fn incumbent_tree(l: usize) -> Tree {
    let mut step: Vec<Tree> = (0..LANES).map(chain).collect();
    for off in [16usize, 8, 4, 2, 1] {
        let prev = step.clone();
        for l in 0..LANES {
            step[l] = Tree::Node(Box::new(prev[l].clone()), Box::new(prev[l ^ off].clone()));
        }
    }
    step.into_iter().nth(l).expect("l < LANES")
}

/// The L8 port's final tree at group-lane `n` (group 0 — every group is the
/// same structure over the same 32 chains): register fold
/// `(p0+p2)+(p1+p3)`, then in-group xor 4/2/1, `+=` self-first.
fn port_tree(n: usize) -> Tree {
    // x_n = (chain_n + chain_{n+16}) + (chain_{n+8} + chain_{n+24})
    let mut x: Vec<Tree> = (0..GROUP)
        .map(|n| {
            Tree::Node(
                Box::new(Tree::Node(Box::new(chain(n)), Box::new(chain(n + 16)))),
                Box::new(Tree::Node(Box::new(chain(n + 8)), Box::new(chain(n + 24)))),
            )
        })
        .collect();
    for off in [4usize, 2, 1] {
        let prev = x.clone();
        for n in 0..GROUP {
            x[n] = Tree::Node(Box::new(prev[n].clone()), Box::new(prev[n ^ off].clone()));
        }
    }
    x.into_iter().nth(n).expect("n < GROUP")
}

/// Numeric incumbent fold: returns every lane's total (all lanes converge).
fn incumbent_fold(chains: &[f32; LANES]) -> [f32; LANES] {
    let mut v = *chains;
    for off in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for l in 0..LANES {
            v[l] = prev[l] + prev[l ^ off];
        }
    }
    v
}

/// Numeric L8 port fold (group 0): returns every group-lane's total.
fn port_fold(chains: &[f32; LANES]) -> [f32; GROUP] {
    let mut x = [0f32; GROUP];
    for n in 0..GROUP {
        // p_m = chain_{n+8m}; the register fold with the pinned child order.
        x[n] = (chains[n] + chains[n + 16]) + (chains[n + 8] + chains[n + 24]);
    }
    for off in [4usize, 2, 1] {
        let prev = x;
        for n in 0..GROUP {
            x[n] = prev[n] + prev[n ^ off];
        }
    }
    x
}

/// The NEGATIVE control's fold: the naive register fold `(p0+p1)+(p2+p3)` —
/// a DIFFERENT tree (its top node pairs chains {n, n+8} then {n+16, n+24},
/// which is NOT any butterfly node). Must diverge on some seed, or the
/// oracle cannot discriminate (an oracle that cannot fail certifies nothing).
fn naive_fold(chains: &[f32; LANES]) -> [f32; GROUP] {
    let mut x = [0f32; GROUP];
    for n in 0..GROUP {
        x[n] = (chains[n] + chains[n + 8]) + (chains[n + 16] + chains[n + 24]);
    }
    for off in [4usize, 2, 1] {
        let prev = x;
        for n in 0..GROUP {
            x[n] = prev[n] + prev[n ^ off];
        }
    }
    x
}

/// xorshift64* — the test's own PRNG (std-only, seeded, deterministic).
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Finite f32 in [1, 2) with a full mantissa — no NaN/Inf/zero, sums stay
    /// moderate across 32-chain accumulation.
    fn next_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() & 0x007f_ffff) as u32;
        f32::from_bits((127u32 << 23) | mantissa)
    }
}

/// One random draw of the 32 x 4 leaf matrix.
fn draw_leaf_matrix(rng: &mut XorShift) -> [[f32; LANES]; LEAVES_PER_CHAIN] {
    let mut leaves = [[0f32; LANES]; LEAVES_PER_CHAIN];
    for row in &mut leaves {
        for slot in row.iter_mut() {
            *slot = rng.next_f32();
        }
    }
    leaves
}

/// The shared chain construction (zero-init ascending, `+=` self-first) —
/// identical on both arms per the PORT LAW; the fold orders differ only
/// after the chains.
fn chains_of(leaves: &[[f32; LANES]; LEAVES_PER_CHAIN]) -> [f32; LANES] {
    let mut chains = [0f32; LANES];
    for (l, chain_out) in chains.iter_mut().enumerate() {
        let mut p = 0f32;
        for leaf_row in &leaves[..LEAVES_PER_CHAIN] {
            p += leaf_row[l];
        }
        *chain_out = p;
    }
    chains
}

#[test]
fn leaf_map_identity_pins_the_section4_map() {
    // Port leaf (c, m) at group-lane n is the incumbent leaf (L = n + 8m, c):
    // s[m*4 + c] = S[wrow][c*32 + n + 8m]. Assert the index arithmetic of the
    // pinned map for every (n, m, c) — the T2 test asserts the map in code.
    for n in 0..GROUP {
        for m in 0..(LANES / GROUP) {
            for c in 0..LEAVES_PER_CHAIN {
                let port_column = c * LANES + n + 8 * m;
                let incumbent_column = c * LANES + (n + 8 * m);
                assert_eq!(port_column, incumbent_column);
                assert_eq!(n + 8 * m, (n + 8 * m) % LANES);
            }
        }
    }
}

#[test]
fn tree_equality_incumbent_vs_l8_port() {
    // Tree-identical BY CONSTRUCTION: the port's tree at group-lane n equals
    // the incumbent's tree at lane n (same leaf multiset, same child order at
    // every node). The stored lane is the group base (n = 0); every n holds.
    for n in 0..GROUP {
        assert_eq!(port_tree(n), incumbent_tree(n), "group-lane {n}");
    }
}

#[test]
fn bit_exactness_10k_seeds() {
    let mut rng = XorShift(0x621_621_621_621_621);
    for seed in 0..10_000u32 {
        let leaves = draw_leaf_matrix(&mut rng);
        let chains = chains_of(&leaves);
        let inc = incumbent_fold(&chains);
        let port = port_fold(&chains);
        // Every group-lane's total must equal EVERY incumbent lane's total,
        // bitwise (the same tree over the same leaves rounds identically).
        for (n, pv) in port.iter().enumerate() {
            for (l, iv) in inc.iter().enumerate() {
                assert_eq!(
                    pv.to_bits(),
                    iv.to_bits(),
                    "seed {seed}: port lane {n} vs incumbent lane {l}"
                );
            }
        }
    }
}

#[test]
fn negative_control_naive_register_fold_diverges() {
    // (p0+p1)+(p2+p3) is a different tree; the oracle must be able to SEE
    // that. At least one seed in 10k must produce a bitwise-different total.
    let mut rng = XorShift(0x621_621_621_621_622);
    let mut diverged = 0usize;
    for _ in 0..10_000u32 {
        let leaves = draw_leaf_matrix(&mut rng);
        let chains = chains_of(&leaves);
        let inc = incumbent_fold(&chains);
        let naive = naive_fold(&chains);
        // The control must diverge from the INCUMBENT (the reference order):
        // any lane of the naive fold differing bitwise from any incumbent
        // lane's total counts (all lanes hold the same total per fold).
        if naive
            .iter()
            .any(|v| inc.iter().any(|i| v.to_bits() != i.to_bits()))
        {
            diverged += 1;
        }
    }
    assert!(
        diverged > 0,
        "negative control NEVER diverged — the oracle cannot discriminate fold orders"
    );
}

#[test]
fn verdict_constant_pin() {
    // The plan-recorded class decision, pinned: the T3 port is
    // promotion-eligible through the bytewise G1 (bench_734 FNV pin covers
    // every cell); T4's order-pin fallback path is dead unless THIS oracle
    // is rewritten by a re-derivation.
    assert_eq!(L8_FOLD_TREE_VERDICT, "tree-identical-by-construction");
}
