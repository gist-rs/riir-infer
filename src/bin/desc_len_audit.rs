//! Issue 040 — the description-length (MDL floor) audit report front.
//!
//! Feature `desc_len` (repo-birth gate discipline: `[[bin]]` carries its
//! `required-features` row). Report-only: opens each `--gguf` via mmap
//! (header plus tensor table), scans the stored symbols of every tensor in
//! ONE O(N) pass, and prints the two-part MDL floor table per
//! arXiv:2509.22445 §B.9.3.
//!
//! Output: per-tensor rows sorted by HONEST slack (side-info-inclusive),
//! per-format and model rollups, measured MB/s (G2).
//!
//! Usage:
//! ```text
//! cargo run --release --features desc_len --bin desc_len_audit -- \
//!     --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf [--json] [--min-mb 1.0]
//! ```

use anyhow::{Context, Result, bail};
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::quant::desc_len::{ModelAudit, Rollup, TensorAudit, audit_file};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Serialize)]
struct JsonRow<'a> {
    name: &'a str,
    format: &'a str,
    n_symbols: u64,
    stored_bpw: f64,
    floor_bpw: Option<f64>,
    side_info_bpw: f64,
    honest_floor_bpw: Option<f64>,
    slack_bpw: Option<f64>,
    honest_slack_bpw: Option<f64>,
    alphabet_k: usize,
    distinct_symbols: u64,
    stored_mb: f64,
    advisory: Option<&'a str>,
}

fn row_json(row: &TensorAudit) -> JsonRow<'_> {
    JsonRow {
        name: &row.name,
        format: row.format,
        n_symbols: row.n_symbols,
        stored_bpw: row.stored_bpw(),
        floor_bpw: row.floor_bpw(),
        side_info_bpw: row.side_info_bpw(),
        honest_floor_bpw: row.honest_floor_bpw(),
        slack_bpw: row.slack_bpw(),
        honest_slack_bpw: row.honest_slack_bpw(),
        alphabet_k: row.alphabet_k,
        distinct_symbols: row.distinct_symbols,
        stored_mb: row.stored_bits as f64 / 8.0 / (1 << 20) as f64,
        advisory: row.advisory,
    }
}

#[derive(Serialize)]
struct JsonRollup {
    tensors: usize,
    advisory_tensors: usize,
    weights: u64,
    stored_bpw: f64,
    honest_floor_bpw: f64,
    honest_slack_bpw: f64,
    stored_mb: f64,
    floor_mb: f64,
    side_info_mb: f64,
}

fn rollup_json(r: &Rollup) -> JsonRollup {
    JsonRollup {
        tensors: r.tensors,
        advisory_tensors: r.advisory_tensors,
        weights: r.weights,
        stored_bpw: r.stored_bpw(),
        honest_floor_bpw: r.honest_floor_bpw(),
        honest_slack_bpw: r.honest_slack_bpw(),
        stored_mb: r.stored_bits as f64 / 8.0 / (1 << 20) as f64,
        floor_mb: r.floor_bits / 8.0 / (1 << 20) as f64,
        side_info_mb: r.side_info_bits as f64 / 8.0 / (1 << 20) as f64,
    }
}

fn fmt_row(row: &TensorAudit) -> String {
    let name = if row.name.len() > 52 {
        format!("…{}", &row.name[row.name.len() - 51..])
    } else {
        row.name.clone()
    };
    match row.floor_bpw() {
        Some(h) => format!(
            "{name:<52} {:<7} {:>9.1} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>+8.4} {:>+8.4}",
            row.format,
            row.n_symbols as f64 / 1e6,
            row.stored_bpw(),
            h,
            row.side_info_bpw(),
            row.honest_floor_bpw().unwrap_or(f64::NAN),
            row.slack_bpw().unwrap_or(f64::NAN),
            row.honest_slack_bpw().unwrap_or(f64::NAN),
        ),
        None => format!(
            "{name:<52} {:<7} {:>9.1} {:>8.4}  ADVISORY: {}",
            row.format,
            row.n_symbols as f64 / 1e6,
            row.stored_bpw(),
            row.advisory.unwrap_or("?"),
        ),
    }
}

fn print_table(audit: &ModelAudit, title: &str) {
    println!("== {title} ==");
    println!(
        "{:<52} {:<7} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "tensor", "format", "Msym", "stored", "H(p)", "side", "honest", "slack", "slack*"
    );
    println!("{}", "-".repeat(122));
    for row in &audit.rows {
        println!("{}", fmt_row(row));
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut ggufs: Vec<PathBuf> = Vec::new();
    let mut json = false;
    let mut min_mb = 0.0f64;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--min-mb" => {
                min_mb = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .context("--min-mb needs a number")?;
            }
            "--help" | "-h" => {
                println!(
                    "usage: desc_len_audit --gguf <model.gguf> [... --gguf <more>] [--json] [--min-mb <mb>]"
                );
                return Ok(());
            }
            "--gguf" => {
                let p = args.next().context("--gguf needs a path")?;
                ggufs.push(PathBuf::from(p));
            }
            other => bail!("unknown arg {other:?} (try --help)"),
        }
    }
    if ggufs.is_empty() {
        bail!("no --gguf given (try --help)");
    }

    for path in &ggufs {
        let file = GgufFile::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_mb = path.metadata().map(|m| m.len()).unwrap_or(0) as f64 / (1 << 20) as f64;

        let t_scan = Instant::now();
        let mut audit = audit_file(&file).context("audit pass")?;
        let scan_s = t_scan.elapsed().as_secs_f64();
        // G2: throughput of the symbol scan over the file's tensor payload
        // (the open/mmap walk is reported separately — the audit is the
        // O(N) pass being priced).
        let mb_s = if scan_s > 0.0 {
            file_mb / scan_s
        } else {
            f64::NAN
        };

        audit.sort_by_honest_slack();
        let rollup = audit.rollup();

        if json {
            #[derive(Serialize)]
            struct JsonOut<'a> {
                path: String,
                file_mb: f64,
                scan_seconds: f64,
                scan_mb_per_s: f64,
                rollup: JsonRollup,
                rows: Vec<JsonRow<'a>>,
            }
            let out = JsonOut {
                path: path.display().to_string(),
                file_mb,
                scan_seconds: scan_s,
                scan_mb_per_s: mb_s,
                rollup: rollup_json(&rollup),
                rows: audit
                    .rows
                    .iter()
                    .filter(|r| r.stored_bits as f64 / 8.0 / (1 << 20) as f64 >= min_mb)
                    .map(row_json)
                    .collect(),
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
            continue;
        }

        print_table(&audit, &format!("{}", path.display()));
        println!("{}", "-".repeat(122));
        // Per-format rollup (the shipped-format spectrum the issue asks for).
        let mut formats: Vec<(&'static str, Rollup)> = Vec::new();
        for row in &audit.rows {
            match formats.iter_mut().find(|(n, _)| *n == row.format) {
                Some((_, r)) => accumulate(r, row),
                None => {
                    let mut r = Rollup::default();
                    accumulate(&mut r, row);
                    formats.push((row.format, r));
                }
            }
        }
        formats.sort_by(|a, b| a.1.stored_bpw().total_cmp(&b.1.stored_bpw()));
        println!(
            "{:<10} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "format", "Msym", "stored_bp", "floor_bp*", "slack_bp*", "stored_MB", "floor_MB*"
        );
        for (name, r) in &formats {
            println!(
                "{name:<10} {:>9.1} {:>10.4} {:>10.4} {:>+10.4} {:>10.1} {:>10.1}",
                r.weights as f64 / 1e6,
                r.stored_bpw(),
                r.honest_floor_bpw(),
                r.honest_slack_bpw(),
                r.stored_bits as f64 / 8.0 / (1 << 20) as f64,
                (r.floor_bits + r.side_info_bits as f64) / 8.0 / (1 << 20) as f64,
            );
        }
        println!("{}", "-".repeat(122));
        println!(
            "MODEL: {} tensors ({} advisory) · {:.1} M weights · stored {:.4} bpw ({:.1} MB) · \
             honest floor {:.4} bpw ({:.1} MB incl. {:.1} MB side-info) · honest slack {:+.4} bpw \
             → floor/stored = {:.3}",
            rollup.tensors,
            rollup.advisory_tensors,
            rollup.weights as f64 / 1e6,
            rollup.stored_bpw(),
            rollup.stored_bits as f64 / 8.0 / (1 << 20) as f64,
            rollup.honest_floor_bpw(),
            (rollup.floor_bits + rollup.side_info_bits as f64) / 8.0 / (1 << 20) as f64,
            rollup.side_info_bits as f64 / 8.0 / (1 << 20) as f64,
            rollup.honest_slack_bpw(),
            if rollup.stored_bits > 0 {
                (rollup.floor_bits + rollup.side_info_bits as f64) / rollup.stored_bits as f64
            } else {
                f64::NAN
            },
        );
        println!(
            "scan: {:.2} s for {:.1} MB → {:.0} MB/s (single O(N) metadata pass, mmap read-only)",
            scan_s, file_mb, mb_s
        );
        println!(
            "conventions: floor = N·H(p) over stored symbols (arXiv:2509.22445 §B.9.3); \
             honest floor adds side-info (per-block scales); freq table not re-priced \
             (adaptive-coder convention); f16/f32 = advisory rows, never floors"
        );
    }
    Ok(())
}

fn accumulate(r: &mut Rollup, row: &TensorAudit) {
    r.tensors += 1;
    match row.advisory {
        Some(_) => r.advisory_tensors += 1,
        None => {
            r.weights += row.n_symbols;
            r.stored_bits += row.stored_bits;
            r.value_bits_nominal += row.value_bits_nominal;
            r.side_info_bits += row.side_info_bits;
            r.floor_bits += row.floor_bits.unwrap_or(0.0);
        }
    }
}
