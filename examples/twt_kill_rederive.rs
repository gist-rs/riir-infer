//! T5.0 ⚠-finding follow-up — re-derive the Bonsai profile artifact's
//! kill-rule verdict with the CURRENT rule over the artifact's OWN stored
//! S matrix (Issue 022: the recorded `SURVIVES` was found inconsistent
//! with the current rule; T5.5 refuses to cite it until this re-derivation
//! exists).
//!
//! Reads the Phase-1 profile artifact, partitions at every point of the
//! pre-registered ε grid with the crate's own unconstrained DP (the
//! Phase-2 protocol the kill rule was written against), re-derives
//! `forced` from the interval-derived type layout, and adjudicates
//! [`kill_verdict`]. Prints the re-derived verdict beside the artifact's
//! recorded one; exits non-zero when they disagree, so the inconsistency
//! is a loud fact rather than a prose claim.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_collapse \
//!   --example twt_kill_rederive -- --profile .raw/twt/bonsai_ultrachat_profile.json
//! ```

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::twt::partition::{
    forced_min_blocks, kill_verdict, minmax_partition, partition_worst, KillVerdict,
    PRE_REGISTERED_EPS_GRID,
};
use riir_infer_core::twt::smatrix::SMatrix;

fn main() {
    let mut profile_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--profile" => {
                profile_path = Some(std::path::PathBuf::from(args.next().expect("--profile needs a path")))
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let profile_path = profile_path.expect("--profile is required");

    let text = std::fs::read_to_string(&profile_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", profile_path.display()));
    let profile: serde_json::Value = serde_json::from_str(&text).expect("profile json");
    let n_layer = profile["n_layers"].as_u64().expect("n_layers") as usize;
    let recorded_verdict = profile["verdict"].as_str().expect("verdict");
    let recorded_forced = profile["forced_blocks"].as_u64().expect("forced_blocks") as usize;

    // The interval-derived type layout (the same derivation every other
    // instrument in this lane uses; the qwen35 parent carries no explicit
    // per-layer types — `full_attention_interval` index arithmetic is the
    // parent's truth, `twt.layer_types` is the COLLAPSED file's).
    let parent = GgufFile::open(&std::path::PathBuf::from(
        std::env::var("TWT_PARENT").unwrap_or_else(|_| {
            "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_owned()
        }),
    ))
    .expect("open parent for full_attention_interval");
    let interval = parent
        .metadata_u64("qwen35.full_attention_interval")
        .unwrap_or(4) as usize;
    let types: Vec<bool> = (0..n_layer).map(|i| (i + 1) % interval != 0).collect();

    let entries = profile["S_upper"].as_array().expect("S_upper");
    let s = SMatrix::from_fn(n_layer, |i, j| {
        entries
            .iter()
            .find_map(|e| {
                let (a, b) = (e[0].as_u64().unwrap() as usize, e[1].as_u64().unwrap() as usize);
                ((a, b) == (i, j)).then(|| e[2].as_f64().unwrap() as f32)
            })
            .unwrap_or_else(|| panic!("S_upper missing entry ({i},{j})"))
    });

    // The kill rule reads the PRE-REGISTERED grid, unconstrained DP.
    let mut partitions = Vec::with_capacity(PRE_REGISTERED_EPS_GRID.len());
    for &eps in PRE_REGISTERED_EPS_GRID.iter() {
        partitions.push(minmax_partition(&s, eps).expect("minmax_partition"));
    }
    let last = partitions.last().expect("non-empty grid");
    let m_at_max = last.len();
    let worst_at_max = partition_worst(&s, last);
    let forced = forced_min_blocks(&types);
    let verdict = kill_verdict(&partitions, forced).expect("kill_verdict");

    println!("# twt_kill_rederive — current kill rule over the artifact's own S");
    println!("# profile: {}", profile_path.display());
    println!("# corpus BLAKE3: {}", profile["corpus_blake3"].as_str().unwrap_or("?"));
    println!("# forced (re-derived from interval {interval}): {forced}  (recorded: {recorded_forced})");
    println!("# m at max ε {}: {m_at_max}, partition worst {worst_at_max:.4}", PRE_REGISTERED_EPS_GRID[PRE_REGISTERED_EPS_GRID.len() - 1]);
    for (eps, p) in PRE_REGISTERED_EPS_GRID.iter().zip(partitions.iter()) {
        println!("#   ε={eps}: m={}", p.len());
    }
    let verdict_str = match &verdict {
        KillVerdict::Survives { .. } => "SURVIVES",
        KillVerdict::KillBlockCount { .. } => "KILL_BLOCK_COUNT",
        KillVerdict::KillMiddleBlocks { .. } => "KILL_MIDDLE_BLOCKS",
    };
    println!("# re-derived verdict: {verdict_str}  ({verdict:?})");
    println!("# recorded verdict:   {recorded_verdict}");
    if verdict_str != recorded_verdict {
        eprintln!(
            "⛔ re-derived verdict {verdict_str} != recorded {recorded_verdict} — the artifact's \
             verdict is stale under the current rule; cite the RE-DERIVED one (T5.5), never the \
             recorded field"
        );
        std::process::exit(1);
    }
    if forced != recorded_forced {
        eprintln!(
            "⚠ re-derived forced {forced} != recorded {recorded_forced} — the capture's forced \
             came from a different types vector (the Issue 022 ⚠ finding); the verdict agrees, \
             the provenance does not"
        );
    }
    println!("# verdict: MATCH — the recorded verdict is reproducible under the current rule");
}
