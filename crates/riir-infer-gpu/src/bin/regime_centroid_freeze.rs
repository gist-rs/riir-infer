//! Issue 1156 T4.4 / Plan 625 — the regime-centroid freeze CLI.
//!
//! Consumes the capture JSONL the forward's freeze-pass seam writes
//! (`RIIR_REGIME_CAPTURE_OUT` rows: `{tag, prompt_len, p, features}` — see
//! [`riir_infer_gpu::regime_gate`]) and computes the TWO frozen regime
//! centroids deterministically (per-class f64-accumulated means — no
//! training, the latent-space freeze law). Writes the sidecar PAIR the gate
//! loads: `<out>.json` + `<out>.json.blake3` (the digest-sidecar convention).
//!
//! Fail-closed: an unknown family tag, a feature-dim disagreement, an odd
//! (non-2·d) dim, or a class with zero rows refuses — a mislabeled run must
//! never become a centroid. The gate REFUSES TO ARM until this pair exists
//! and verifies; nothing derives a centroid at runtime.
//!
//! ```sh
//! # 1. run the feature dump (riir-ai bench_1156_regime_feature_dump, ~15 min
//! #    dense on amo; rows land at the capture path the test sets)
//! # 2. freeze:
//! cargo run --release -p riir-infer-gpu --features cubecl_runtime --bin \
//!   regime_centroid_freeze -- /tmp/regime_features.jsonl \
//!   ../riir-infer/crates/riir-infer-gpu/tests/fixtures/regime_centroids.json \
//!   --gain 10.0 --margin 0.1
//! ```
//!
//! `--gain`/`--margin` default to the named consts (`REGIME_GAIN` is a
//! PLACEHOLDER pending this pass — the dump's feature scale is what
//! calibrates them; the calibration read is: the per-class cosine spread in
//! the dump vs the margin band).

#![cfg(feature = "cubecl_runtime")]

use riir_infer_gpu::regime_gate::{
    FreezeRow, REGIME_GAIN, REGIME_MARGIN, centroids_body_json, compute_centroids, parse_freeze_row,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut input, mut out, mut gain, mut margin) = (None, None, REGIME_GAIN, REGIME_MARGIN);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gain" => {
                gain = args
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|g| *g > 0.0)
                    .expect("--gain needs a positive f32");
            }
            "--margin" => {
                margin = args
                    .next()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|m| (0.0..=0.5).contains(m))
                    .expect("--margin needs an f32 in [0, 0.5]");
            }
            other if input.is_none() => input = Some(other.to_string()),
            other if out.is_none() => out = Some(other.to_string()),
            other => {
                eprintln!("unexpected argument `{other}`");
                std::process::exit(2);
            }
        }
    }
    let (Some(input), Some(out)) = (input, out) else {
        eprintln!(
            "usage: regime_centroid_freeze <features.jsonl> <out.json> [--gain F32] [--margin F32]"
        );
        std::process::exit(2);
    };

    let body = std::fs::read_to_string(&input).unwrap_or_else(|e| panic!("read {input}: {e}"));
    let mut rows: Vec<FreezeRow> = Vec::new();
    for (i, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(parse_freeze_row(line).unwrap_or_else(|e| panic!("row {i}: {e}")));
    }
    // The per-family census goes to stderr BEFORE the computation — a dump
    // that captured one family twice and another zero times is visible here,
    // not just in the refusal.
    let mut census: std::collections::BTreeMap<String, usize> = Default::default();
    for r in &rows {
        *census
            .entry(riir_infer_gpu::regime_gate::family_of_tag(&r.tag).to_string())
            .or_insert(0) += 1;
    }
    for (family, count) in &census {
        eprintln!("[freeze] {family}: {count} rows");
    }

    let centroids = compute_centroids(&rows, gain, margin).unwrap_or_else(|e| {
        eprintln!("[freeze] REFUSED: {e}");
        std::process::exit(1);
    });
    let json = centroids_body_json(&centroids);
    let pretty = serde_json::to_string_pretty(&json).expect("centroids serialize");
    std::fs::write(&out, &pretty).unwrap_or_else(|e| panic!("write {out}: {e}"));
    let digest = riir_infer_gpu::sparse_omega_msr::blake3_hex(pretty.as_bytes());
    let sidecar = format!("{out}.blake3");
    std::fs::write(&sidecar, &digest).unwrap_or_else(|e| panic!("write {sidecar}: {e}"));
    println!(
        "[freeze] {} rows -> {} classes, dim {}",
        rows.len(),
        2,
        centroids.dim
    );
    println!("[freeze] wrote {out}");
    println!("[freeze] wrote {sidecar}");
    println!("[freeze] digest {digest}");
    println!(
        "[freeze] next: arm the gate with RIIR_PREFILL_REGIME_GATE=1 \
         RIIR_REGIME_CENTROIDS={} RIIR_OMSR_MANIFEST=<the v2 fixture> \
         and calibrate gain/margin on the dump's cosine spread",
        out
    );
}
