// Issue 035 — minimizer test battery.
//
// Language equality is pinned by exhaustive enumeration on small automata
// (every token string over a 3-symbol alphabet to length 6); the
// exact-joint distribution preservation is pinned by frequency comparison
// (same logits, N draws from raw vs minimized, per-sequence counts within
// sampling noise); determinism and idempotence are pinned structurally.

use super::*;
use crate::fa_posterior::{FREE, FaScratch, SplitMix64};

type EdgeList = Vec<(usize, usize, Vec<u32>)>;

/// Structural fingerprint — what "identical automaton shape" means here.
fn shape(a: &Automaton) -> (usize, usize, usize, Vec<usize>, EdgeList) {
    let mut accepts: Vec<usize> = (0..a.n_nodes()).filter(|&n| a.is_accept(n)).collect();
    let mut edges: EdgeList = Vec::new();
    for src in 0..a.n_nodes() {
        for e in a.out_range(src) {
            edges.push((src, a.edge_dst(e), a.edge_tokens(e).to_vec()));
        }
    }
    edges.sort();
    accepts.sort_unstable();
    (a.start(), a.n_nodes(), a.n_edges(), accepts, edges)
}

/// Exhaustive acceptance comparison over every string of `alphabet` to
/// `max_len` (walk ∧ final — the compiler's acceptance law).
fn languages_agree(a: &Automaton, b: &Automaton, alphabet: &[u32], max_len: usize) {
    fn rec(a: &Automaton, b: &Automaton, alphabet: &[u32], max_len: usize, seq: &mut Vec<u32>) {
        let acc = |fa: &Automaton| fa.walk(seq).is_some_and(|n| fa.is_accept(n));
        assert_eq!(acc(a), acc(b), "language mismatch at {seq:?}");
        if seq.len() == max_len {
            return;
        }
        for &t in alphabet {
            seq.push(t);
            rec(a, b, alphabet, max_len, seq);
            seq.pop();
        }
    }
    let mut seq = Vec::new();
    rec(a, b, alphabet, max_len, &mut seq);
}

fn builder_chain() -> Automaton {
    // 0 -0-> 1 -1-> 2(accept) -0-> 2 (self-loop tail)
    AutomatonBuilder::new(3, 3, 0)
        .accept(2)
        .edge(0, 1, &[0])
        .edge(1, 2, &[1])
        .edge(2, 2, &[0, 2])
        .build()
        .unwrap()
}

#[test]
fn minimize_preserves_language_small_automata() {
    let alphabet = [0u32, 1, 2];

    // Chain with a self-loop tail.
    let a = builder_chain();
    let m = minimize(&a).unwrap();
    languages_agree(&a, &m, &alphabet, 6);

    // Parity: `0` self-loops in place, `1` toggles; accept on an even
    // count of 1s.
    let parity = AutomatonBuilder::new(2, 3, 0)
        .accept(0)
        .edge(0, 0, &[0])
        .edge(0, 1, &[1])
        .edge(1, 0, &[1])
        .edge(1, 1, &[0])
        .build()
        .unwrap();
    let mp = minimize(&parity).unwrap();
    // 0 and 1 are NOT equivalent (accept differs) — nothing merges.
    assert_eq!(mp.n_nodes(), 2);
    languages_agree(&parity, &mp, &alphabet, 6);

    // Dead branch + unreachable accepting state: both must vanish.
    let messy = AutomatonBuilder::new(5, 3, 0)
        .accept(2)
        .accept(4) // unreachable from start
        .edge(0, 1, &[0])
        .edge(1, 2, &[1]) // the live path
        .edge(0, 3, &[2]) // dead branch: 3 has no path to accept
        .build()
        .unwrap();
    let mm = minimize(&messy).unwrap();
    assert_eq!(
        (mm.n_nodes(), mm.n_edges()),
        (3, 2),
        "dead branch + unreachable accept dropped"
    );
    languages_agree(&messy, &mm, &alphabet, 6);

    // Everything-accepting single node with a self-loop.
    let single = AutomatonBuilder::new(1, 3, 0)
        .accept(0)
        .edge(0, 0, &[0, 1, 2])
        .build()
        .unwrap();
    let ms = minimize(&single).unwrap();
    assert_eq!(ms.n_nodes(), 1);
    languages_agree(&single, &ms, &alphabet, 5);
}

#[test]
fn minimize_merges_equivalent_branches() {
    // 0 -0-> 1 -2-> 3(accept) and 0 -1-> 2 -2-> 3: 1 and 2 are equivalent.
    let a = AutomatonBuilder::new(4, 3, 0)
        .accept(3)
        .edge(0, 1, &[0])
        .edge(1, 3, &[2])
        .edge(0, 2, &[1])
        .edge(2, 3, &[2])
        .build()
        .unwrap();
    let m = minimize(&a).unwrap();
    assert_eq!((m.n_nodes(), m.n_edges()), (3, 2), "1 and 2 merge");
    languages_agree(&a, &m, &[0u32, 1, 2], 6);
}

#[test]
fn minimize_is_deterministic_and_idempotent() {
    // A 2-prop-object shape with a real merge: any-order members over
    // equivalent value fragments — branches 1/2 collapse.
    let a = AutomatonBuilder::new(7, 4, 0)
        .accept(6)
        .edge(0, 1, &[0])
        .edge(0, 2, &[1])
        .edge(1, 5, &[2])
        .edge(2, 5, &[2])
        .edge(5, 6, &[3])
        .edge(6, 6, &[0, 1, 2, 3])
        .build()
        .unwrap();

    let (m1, s1) = minimize_with_stats(&a).unwrap();
    let (m2, s2) = minimize_with_stats(&a).unwrap();
    assert_eq!(s1.nodes_out, s2.nodes_out);
    assert_eq!(s1.edges_out, s2.edges_out);
    assert_eq!(shape(&m1), shape(&m2), "two runs must be byte-identical");
    assert!(s1.nodes_out < s1.nodes_in, "test premise: merging happened");

    let (m3, s3) = minimize_with_stats(&m1).unwrap();
    assert_eq!(shape(&m1), shape(&m3), "minimize is idempotent");
    assert_eq!(s3.nodes_in, s1.nodes_out);
    assert_eq!(s3.nodes_out, s1.nodes_out, "no further shrink on re-run");
    languages_agree(&a, &m1, &[0u32, 1, 2, 3], 6);
}

#[test]
fn minimize_dead_start_errors() {
    // Start has no path to any accepting state.
    let a = AutomatonBuilder::new(2, 3, 0)
        .accept(1)
        .edge(0, 0, &[0]) // 1 is unreachable from 0
        .build()
        .unwrap();
    let err = minimize(&a).unwrap_err();
    assert!(matches!(err, FaError::DeadStart(0)));
}

#[test]
fn minimize_stats_report_the_shape_delta() {
    let a = builder_chain();
    let (_, s) = minimize_with_stats(&a).unwrap();
    assert_eq!(s.nodes_in, 3);
    assert_eq!(s.edges_in, 3);
    assert_eq!(s.nodes_out, 3); // nothing equivalent — a chain stays
    assert_eq!(s.edges_out, 3);
}

#[test]
fn joint_distribution_is_preserved_through_merging() {
    // The load-bearing claim: same logits → the SAME token-sequence
    // distribution from raw and minimized automata (the module doc's
    // e_log·β argument). Merging must actually happen for the test to
    // bite: branches 1/2 are equivalent, so they merge.
    let raw = AutomatonBuilder::new(4, 3, 0)
        .accept(3)
        .edge(0, 1, &[0])
        .edge(1, 3, &[2])
        .edge(0, 2, &[1])
        .edge(2, 3, &[2])
        .build()
        .unwrap();
    let min = minimize(&raw).unwrap();
    assert!(
        min.n_nodes() < raw.n_nodes(),
        "test premise: merging happened"
    );

    // Length 2 is exactly an accepting walk (0 → branch → accept) — no
    // padding language needed on this hand-built fixture.
    let len = 2usize;
    let vocab = raw.vocab();
    // Logits that leave both branches live with distinct probabilities.
    let logits: Vec<f32> = (0..len * vocab)
        .map(|i| (((i * 7919) % 23) as f32 - 8.0) / 4.0)
        .collect();
    let forced = vec![FREE; len];

    let draws = 20_000u64;
    let count = |fa: &Automaton| -> HashMap<Vec<u32>, usize> {
        let mut scratch = FaScratch::new();
        let mut counts: HashMap<Vec<u32>, usize> = HashMap::new();
        for seed in 0..draws {
            let mut rng = SplitMix64::new(seed);
            let mut out = vec![0u32; len];
            fa.sample_joint(
                &mut scratch,
                &logits,
                len,
                &forced,
                1.0,
                false,
                &mut rng,
                &mut out,
            )
            .expect("draw");
            *counts.entry(out).or_default() += 1;
        }
        counts
    };
    let c_raw = count(&raw);
    let c_min = count(&min);
    assert_eq!(
        c_raw.len(),
        c_min.len(),
        "the merged automaton admits exactly the same sequences"
    );
    for (k, &n1) in c_raw.iter() {
        let n2 = c_min[k] as f64;
        let n1 = n1 as f64;
        let p = (n1 + n2) / (2.0 * draws as f64);
        // 5σ of the difference of two independent N-draw binomial COUNTS
        // (σ = √(2·N·p·(1−p))), floored so tiny-probability sequences
        // cannot flake.
        let tol = 5.0 * (2.0 * p * (1.0 - p) * draws as f64).sqrt() + 6.0;
        assert!(
            (n1 - n2).abs() <= tol,
            "sequence {k:?} freq drift: raw {n1} vs min {n2} (tol {tol:.1})"
        );
    }
}

#[test]
fn single_non_accepting_start_with_edges_to_accept_survives() {
    // Start accepts nothing itself but leads to accept — the trim must
    // keep the start (it is the entry) and the language is non-empty.
    let a = AutomatonBuilder::new(2, 3, 0)
        .accept(1)
        .edge(0, 1, &[1])
        .build()
        .unwrap();
    let m = minimize(&a).unwrap();
    languages_agree(&a, &m, &[0u32, 1, 2], 5);
    assert!(m.is_accept(m.walk(&[1]).unwrap()));
}
