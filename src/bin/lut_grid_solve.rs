//! Issue 027 / Plan 615 — the offline LUT-grid solver + weight-space eval
//! bin over the DENSE GGUF artifacts.
//!
//! Feature `lut_grid` (the repo-birth gate discipline: `[[bin]]` carries its
//! `required-features` row). Pure offline tool — no training, no model
//! forward. Reads each `--gguf` (mmap), pools the empirical distribution of
//! `x = w/d` under the T0 scale rule, solves the Lloyd-Max grid, and runs
//! the 3-arm weight-space eval (symmetric | T0 | solved grid) per tensor,
//! aggregated per family — the WEIGHT-SPACE half of 027-T3. The GOAT gate
//! is the model-level per-family retention run (Plan 615 T6, deferred);
//! this table is the proxy that decides whether the model-level run is
//! worth wiring at all.
//!
//! Usage:
//! ```text
//! cargo run --release --features lut_grid --bin lut_grid_solve -- \
//!     --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf \
//!     --gguf ../riir-train/data/MiniCPM5-1B-F16.gguf \
//!     --out .benchmarks/024_lut_grid_tables
//! ```

use anyhow::{bail, Context, Result};
use half::f16;
use riir_infer_core::gguf_loader::{GgmlType, GgufFile};
use riir_infer_core::quant::lut_grid::{grid_digest, solve_lloyd_max};
use riir_infer_core::quant::q2_0::{
    dequantize_row_q2_0, dequantize_row_q2_0_grid, quantize_row_q2_0_asymmetric,
    quantize_row_q2_0_grid, quantize_row_q2_0_symmetric, t0_block_scale, Q2_0_BLOCK_SIZE,
};
use riir_infer_core::quant::lut_grid::WeightHistogram;
use serde::Serialize;

#[derive(Serialize)]
struct FamilyRow {
    family: String,
    tensors: usize,
    weights: u64,
    snr_db: [f64; 3],
    mse: [f64; 3],
    worst_tensor_snr_db: [f64; 3],
}

#[derive(Serialize)]
struct SolveReport {
    gguf: String,
    tensors_used: usize,
    tensors_skipped: usize,
    weights: u64,
    solved_grid: [f32; 2],
    iterations: usize,
    hist_mse_t0: f64,
    hist_mse_solved: f64,
    /// BLAKE3 over (histogram geometry + counts, solved levels, stats) —
    /// the artifact commitment for the cross-box re-run diff.
    grid_digest_hex: String,
    families: Vec<FamilyRow>,
    granularity: Vec<GranularityRow>,
}

#[derive(Serialize)]
struct GranularityRow {
    group: usize,
    bpw: f64,
    snr_db: f64,
    mse: f64,
}

/// Family bucket from a GGUF tensor name: strips the layer prefix
/// (`blk.N.` / `model.layers.N.` / `layers.N.`), strips the `.weight`
/// suffix, normalizes HF-style separators to the llama.cpp-ish short form.
fn family_of(name: &str) -> String {
    let mut s = name;
    for p in [
        "model.layers.",
        "layers.",
        "blk.",
        "encoder.layer.",
    ] {
        if let Some(rest) = s.strip_prefix(p) {
            // Skip the numeric segment + the following dot.
            if let Some(dot) = rest.find('.') {
                s = &rest[dot + 1..];
            }
            break;
        }
    }
    let s = s.strip_suffix(".weight").unwrap_or(s);
    // HF style: self_attn.q_proj → attn_q-shaped short token.
    let s = s
        .replace("self_attn.", "attn_")
        .replace("_proj", "")
        .replace("post_attention_layernorm", "attn_norm")
        .replace("pre_feedforward_layernorm", "mlp_norm")
        .replace("post_feedforward_layernorm", "mlp_norm");
    s.to_string()
}

fn f16_tensor_to_f32(gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let slice = gguf
        .tensor_slice(name)
        .with_context(|| format!("tensor slice missing for {name}"))?;
    let bits: &[u16] = bytemuck::cast_slice(slice);
    Ok(bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
}

/// Weight-space SNR in dB.
fn snr_db(w: &[f32], q: &[f32]) -> (f64, f64) {
    let mut sig = 0f64;
    let mut err = 0f64;
    for (a, b) in w.iter().zip(q) {
        let d = (*a - b) as f64;
        sig += (*a as f64) * (*a as f64);
        err += d * d;
    }
    if err == 0.0 {
        (f64::INFINITY, 0.0)
    } else {
        (10.0 * (sig / err).log10(), err / w.len() as f64)
    }
}

fn main() -> Result<()> {
    let mut ggufs: Vec<String> = Vec::new();
    let mut out_dir: Option<String> = None;
    let mut granularity_sweep = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => ggufs.push(args.next().context("--gguf needs a path")?),
            "--out" => out_dir = Some(args.next().context("--out needs a dir")?),
            "--granularity" => granularity_sweep = true,
            other => bail!("unknown arg {other} (usage: --gguf <p>… [--out <dir>] [--granularity])"),
        }
    }
    if ggufs.is_empty() {
        bail!("no --gguf given");
    }

    for path in &ggufs {
        let report = solve_one(path, granularity_sweep)?;
        println!("{}", render_report(&report));
        if let Some(dir) = &out_dir {
            std::fs::create_dir_all(dir)?;
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            let json_path = format!("{dir}/lut_grid_solve_{stem}.json");
            std::fs::write(
                &json_path,
                serde_json::to_string_pretty(&report).context("serialize report")?,
            )?;
            println!("[lut-grid] wrote {json_path}");
        }
    }
    Ok(())
}

fn solve_one(path: &str, granularity_sweep: bool) -> Result<SolveReport> {
    let gguf = GgufFile::open(std::path::Path::new(path))
        .with_context(|| format!("open {path}"))?;

    // Eligible tensors: F16, 2-D, name carries "weight", whole blocks.
    let mut eligible: Vec<(String, Vec<usize>)> = Vec::new();
    let mut skipped = 0usize;
    for info in &gguf.tensor_infos {
        if info.ggml_type != GgmlType::F16 || info.shape.len() != 2 {
            continue;
        }
        if !info.name.contains("weight") {
            continue;
        }
        if info.n_elements() % Q2_0_BLOCK_SIZE != 0 {
            skipped += 1;
            continue;
        }
        eligible.push((info.name.clone(), info.shape.clone()));
    }
    if eligible.is_empty() {
        bail!("no eligible F16 weight tensors in {path}");
    }

    // ── Pass 1: pooled histogram of x = w/d under the T0 rule ──────
    // Mass-weighted by d² per element — the solver's objective is the
    // ENERGY-weighted MSE the round-trip SNR measures (see the
    // WeightHistogram doc for the measured unweighted-pool failure).
    let mut hist = WeightHistogram::new();
    let mut total_weights = 0u64;
    for (name, _) in &eligible {
        let w = f16_tensor_to_f32(&gguf, name)?;
        for block in w.as_chunks::<Q2_0_BLOCK_SIZE>().0 {
            let Some((_, d)) = t0_block_scale(block) else {
                continue; // zero block: x undefined, contributes no mass
            };
            let mass = (d as f64) * (d as f64);
            for &v in block {
                hist.record_weighted((v / d) as f64, mass);
            }
        }
        total_weights += w.len() as u64;
        eprintln!("[lut-grid] hist {name} ({} weights)", w.len());
    }

    let (grid, stats) = solve_lloyd_max(&hist);
    let digest = grid_digest(&hist, grid, &stats);

    println!(
        "[lut-grid] solved grid l0={} l2={} (iters {}, hist MSE {:.6e} → {:.6e}, −{:.2}%), digest {}",
        grid.l0,
        grid.l2,
        stats.iterations,
        stats.mse_start,
        stats.mse_end,
        (1.0 - stats.mse_end / stats.mse_start) * 100.0,
        hex(&digest)
    );

    // ── Pass 2: per-tensor 3-arm eval, aggregated per family ───────
    struct Acc {
        tensors: u64,
        weights: u64,
        sig: f64,
        err: [f64; 3],
        worst: [f64; 3],
    }
    let mut families: std::collections::BTreeMap<String, Acc> = Default::default();
    let mut gran: [Acc2; 3] = [Acc2::default(), Acc2::default(), Acc2::default()];

    for (idx, (name, _)) in eligible.iter().enumerate() {
        let w = f16_tensor_to_f32(&gguf, name)?;
        let fam = family_of(name);
        let acc = families.entry(fam).or_insert(Acc {
            tensors: 0,
            weights: 0,
            sig: 0.0,
            err: [0.0; 3],
            worst: [f64::INFINITY; 3],
        });
        acc.tensors += 1;
        acc.weights += w.len() as u64;

        let mut out = Vec::new();
        let mut dq = vec![0f32; w.len()];

        quantize_row_q2_0_symmetric(&w, &mut out);
        dequantize_row_q2_0(&out, &mut dq);
        let (snr, mse) = snr_db(&w, &dq);
        acc.sig += sig_energy(&w);
        acc.err[0] += mse * w.len() as f64;
        acc.worst[0] = acc.worst[0].min(snr);

        out.clear();
        quantize_row_q2_0_asymmetric(&w, &mut out);
        dequantize_row_q2_0(&out, &mut dq);
        let (snr, mse) = snr_db(&w, &dq);
        acc.err[1] += mse * w.len() as f64;
        acc.worst[1] = acc.worst[1].min(snr);

        out.clear();
        quantize_row_q2_0_grid(&w, grid, &mut out);
        dequantize_row_q2_0_grid(&out, &mut dq, grid);
        let (snr, mse) = snr_db(&w, &dq);
        acc.err[2] += mse * w.len() as f64;
        acc.worst[2] = acc.worst[2].min(snr);

        if granularity_sweep {
            sweep_granularity(&w, &mut gran);
        }

        if idx % 25 == 0 {
            eprintln!("[lut-grid] eval {idx}/{}", eligible.len());
        }
    }

    let fam_rows: Vec<FamilyRow> = families
        .into_iter()
        .map(|(family, a)| {
            let snr = |err: f64| 10.0 * ((a.sig / err).log10());
            FamilyRow {
                family,
                tensors: a.tensors as usize,
                weights: a.weights,
                snr_db: [snr(a.err[0]), snr(a.err[1]), snr(a.err[2])],
                mse: [a.err[0] / a.weights as f64, a.err[1] / a.weights as f64, a.err[2] / a.weights as f64],
                worst_tensor_snr_db: a.worst,
            }
        })
        .collect();

    let granularity = if granularity_sweep {
        vec![
            GranularityRow { group: 128, bpw: 2.0 + 16.0 / 128.0, snr_db: gran[0].snr(), mse: gran[0].mse() },
            GranularityRow { group: 64, bpw: 2.0 + 16.0 / 64.0, snr_db: gran[1].snr(), mse: gran[1].mse() },
            GranularityRow { group: 32, bpw: 2.0 + 16.0 / 32.0, snr_db: gran[2].snr(), mse: gran[2].mse() },
        ]
    } else {
        Vec::new()
    };

    Ok(SolveReport {
        gguf: path.to_string(),
        tensors_used: eligible.len(),
        tensors_skipped: skipped,
        weights: total_weights,
        solved_grid: [grid.l0, grid.l2],
        iterations: stats.iterations,
        hist_mse_t0: stats.mse_start,
        hist_mse_solved: stats.mse_end,
        grid_digest_hex: hex(&digest),
        families: fam_rows,
        granularity,
    })
}

struct Acc2 {
    sig: f64,
    err: f64,
    n: u64,
}

impl Default for Acc2 {
    fn default() -> Self {
        Self { sig: 0.0, err: 0.0, n: 0 }
    }
}

impl Acc2 {
    fn snr(&self) -> f64 {
        10.0 * (self.sig / self.err).log10()
    }
    fn mse(&self) -> f64 {
        self.err / self.n as f64
    }
}

/// T5 weight-space granularity sweep: re-encode per-64 / per-32 groups with
/// the T0 rule (block math identical, group size differs). These are NOT
/// wire-format layouts (the wire is per-128 only) — the sweep answers
/// whether finer granularity buys more accuracy than the scale bytes cost.
fn sweep_granularity(w: &[f32], gran: &mut [Acc2; 3]) {
    for (gi, &group) in [128usize, 64, 32].iter().enumerate() {
        let mut sig = 0f64;
        let mut err = 0f64;
        for block in w.chunks_exact(group) {
            let Some((_, d)) = t0_block_scale(block) else {
                continue;
            };
            for &v in block {
                let q = (v / d).round().clamp(-1.0, 2.0) * d;
                sig += (v as f64) * (v as f64);
                let e = (v - q) as f64;
                err += e * e;
            }
        }
        gran[gi].sig += sig;
        gran[gi].err += err;
        gran[gi].n += w.len() as u64;
    }
}

fn sig_energy(w: &[f32]) -> f64 {
    w.iter().map(|&v| (v as f64) * (v as f64)).sum()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn render_report(r: &SolveReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "═══ LUT grid solve — {} ═══\n",
        r.gguf
    ));
    s.push_str(&format!(
        "tensors used {} (skipped partial {}), weights {}\n",
        r.tensors_used, r.tensors_skipped, r.weights
    ));
    s.push_str(&format!(
        "grid l0={:+.6} l2={:+.6} · iters {} · hist MSE {:.6e} → {:.6e} (−{:.2}%) · commitment {}\n",
        r.solved_grid[0],
        r.solved_grid[1],
        r.iterations,
        r.hist_mse_t0,
        r.hist_mse_solved,
        (1.0 - r.hist_mse_solved / r.hist_mse_t0) * 100.0,
        r.grid_digest_hex
    ));
    s.push_str("{:<28} {:>7} {:>14} {:>14} {:>14}  worst-tensor SNR\n");
    s.push_str(&format!(
        "{:<28} {:>7} {:>14} {:>14} {:>14}  {}\n",
        "family", "tensors", "SNR sym dB", "SNR t0 dB", "SNR solved dB", "(sym/t0/solved)"
    ));
    for f in &r.families {
        s.push_str(&format!(
            "{:<28} {:>7} {:>14.4} {:>14.4} {:>14.4}  {:.2}/{:.2}/{:.2}\n",
            f.family,
            f.tensors,
            f.snr_db[0],
            f.snr_db[1],
            f.snr_db[2],
            f.worst_tensor_snr_db[0],
            f.worst_tensor_snr_db[1],
            f.worst_tensor_snr_db[2]
        ));
    }
    if !r.granularity.is_empty() {
        s.push_str("── granularity sweep (T0 rule) ──\n");
        for g in &r.granularity {
            s.push_str(&format!(
                "per-{:>3}  {:>5.3} bpw  SNR {:.4} dB  MSE {:.6e}\n",
                g.group, g.bpw, g.snr_db, g.mse
            ));
        }
    }
    s
}
