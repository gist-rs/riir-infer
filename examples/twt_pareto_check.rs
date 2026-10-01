//! T5.5 — the ε-sweep Pareto CHECK (Issue 022): re-runs the min-max DP at
//! every pre-registered ε grid point plus the fine-end probes, over the
//! profile artifact's stored S matrix, and ASSERTS the monotonicity of the
//! block count m in ε — "monotonicity of m in ε asserted (a violation is a
//! Phase 2 bug)". Larger ε merges more ⇒ m must never grow.
//!
//! The eight MEASURED points (each carries an agreement row in
//! `.benchmarks/022_t5_collapsed_goat_agreement.md`) are additionally
//! pinned as known answers — the DP is deterministic over a fixed artifact,
//! so the pinned counts re-verify the writer's determinism input from the
//! cheap side (no GGUF emit). A deliberate Phase 2 change that moves a
//! count re-pins this list IN THE SAME COMMIT, with the reason.
//!
//! No GGUF is written. Runtime: the DP over a 64×64 S matrix, milliseconds.
//!
//! Usage:
//!   cargo run --release -p riir-infer-core --features twt_collapse \
//!     --example twt_pareto_check -- \
//!     --profile .raw/twt/bonsai_ultrachat_profile.json

#![cfg(feature = "twt_collapse")]

use riir_infer_core::twt::{SMatrix, minmax_partition};

/// The pre-registered coarse grid (Phase 2) + the fine-end probes (T5.0),
/// ascending — the sweep order.
const EPS_POINTS: [f32; 11] = [0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2];

/// The measured points (index into EPS_POINTS) and the block counts the
/// agreement sweep emitted for them (Bench 022 table; the emit stdout is
/// the per-point receipt).
const MEASURED: [(usize, usize); 8] = [
    (0, 61), // ε=0.01 — agreement 0.9486 PASS
    (1, 57), // ε=0.015 — 0.8955
    (2, 49), // ε=0.02 — 0.5247
    (3, 35), // ε=0.03 — 0.0301
    (4, 25), // ε=0.05 — 0.1945
    (5, 14), // ε=0.1 — 0.0051
    (6, 8),  // ε=0.2 — 0.0029
    (7, 5),  // ε=0.3 — 0.0000
];

fn main() {
    let mut profile_path = String::from(".raw/twt/bonsai_ultrachat_profile.json");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--profile" => profile_path = args.next().expect("--profile needs a path"),
            other => panic!("unknown arg {other}"),
        }
    }

    let text = std::fs::read_to_string(&profile_path)
        .unwrap_or_else(|e| panic!("read {profile_path}: {e}"));
    let profile: serde_json::Value = serde_json::from_str(&text).expect("profile json");
    let n_layer = profile["n_layers"].as_u64().expect("n_layers") as usize;
    let entries = profile["S_upper"].as_array().expect("S_upper");
    let s = SMatrix::from_fn(n_layer, |i, j| {
        entries
            .iter()
            .find_map(|e| {
                let a = e[0].as_u64().unwrap() as usize;
                let b = e[1].as_u64().unwrap() as usize;
                if (a, b) == (i, j) {
                    Some(e[2].as_f64().unwrap() as f32)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("S_upper missing entry ({i},{j})"))
    });

    let mut counts = Vec::with_capacity(EPS_POINTS.len());
    println!("eps\tblocks\tdepth%");
    for &eps in &EPS_POINTS {
        let dp = minmax_partition(&s, eps).expect("minmax_partition");
        let m = dp.len();
        counts.push(m);
        println!("{eps}\t{m}\t{:.1}", 100.0 * m as f32 / n_layer as f32);
    }

    // ── monotonicity of m in ε (the Phase 2 bug detector) ──
    for w in counts.windows(2) {
        assert!(
            w[0] >= w[1],
            "m(ε) is NOT monotone: a larger ε produced MORE blocks \
             ({} at the previous point vs {} here) — a Phase 2 partition bug",
            w[0],
            w[1],
        );
    }

    // ── the measured known answers ──
    for (idx, expected) in MEASURED {
        assert_eq!(
            counts[idx], expected,
            "measured point ε={} re-derived m={} but the agreement sweep \
             emitted m={expected} — the DP moved under a fixed artifact \
             (a Phase 2 bug, or a deliberate change that must re-pin this \
             table in the same commit)",
            EPS_POINTS[idx], counts[idx],
        );
    }

    println!(
        "# twt_pareto_check — m(ε) monotone across {} points; {} measured known answers hold",
        EPS_POINTS.len(),
        MEASURED.len(),
    );
}
