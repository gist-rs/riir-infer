//! svd_lbit_eval — Issue 036 T1/T2 (the T0 eval loop): the init-only
//! Dual-SVID PPL ladder on gemma-2-2b-it f16 (the house fixture).
//!
//! The paper's unmeasured point: every published LittleBit PPL is a QAT
//! output — this bin measures the CLOSED-FORM INIT ALONE, against
//! training-free baselines computed on the SAME checkpoint at matched BPW
//! (never against published tables):
//!
//! | arm | rule | passes |
//! |---|---|---|
//! | `anchor` | the untouched f16 parent | 1 |
//! | `lbit1` | Dual-SVID init, single path (paths=1 rank) | per target |
//! | `lbit2` | 2-stage residual restack (paths=2 rank each) | per target |
//! | `rtn` | naive row-α binary (`α_i = mean|W_i,:|`) — floor ~1.03 bpw, so it arms ONLY at target 1.0 | 1 |
//! | `svd` | plain truncated low-rank, f16 factors, NO binarization (the SVD-LLM/ASVD class) | per target |
//!
//! Default arms × targets = the 14-pass ladder. `wte`/norms are never
//! transformed; only the 7 linear tensors per layer.
//!
//! # Disclosures (recorded in the bench, per the T0 decision)
//!
//! - **Shared basis:** ONE `k_max` Halko factorization per tensor serves
//!   `lbit1`, `lbit2` path 1, and `svd` at ALL targets (k = r_max + 16,
//!   oversampled; truncated per use). Exact per-r independent SVD is a
//!   documented approximation class. `lbit2` path 2 factorizes the
//!   per-target residual freshly.
//! - **Eval shape:** non-overlapping `--seq-len` windows (≤ 4096 — the
//!   gemma-2 sliding-window law, the `act_ptq_gemma2` precedent),
//!   teacher-forced NLL via `quant::kvq_harness::nll` (the ungated twin of
//!   `vk_harness::nll`, which sits behind `fitted_v_tables`), f64
//!   accumulation, PPL = exp(ΣNLL/N).
//! - **Determinism:** every transform seed derives from `--seed` through
//!   `svd_lbit::mix_seed`; per-window NLL sums reduce in window order.
//! - **Box state:** a PROVENANCE-style line (loadavg, workers, os) is
//!   printed best-effort at start and captured per pass — std-only.
//!
//! MEASUREMENT-ONLY: the bin NEVER writes the model; the only artifacts are
//! the optional `--out` report and `--json` sidecar. Peak RSS ≈ live f16
//! (5.3 GB) + originals (4.1 GB) + shared bases (~5 GB) + per-worker
//! transform temps + bounded KV caches (~0.21 GB/worker) — sized for the
//! 64 GB M3, disclosed here because the ladder is a one-box instrument.
//!
//! Usage:
//! ```text
//! svd_lbit_eval --gguf <model.gguf> --corpus <dir-or-txt>
//!     [--targets 1.0,0.55,0.3,0.1] [--arms anchor,lbit1,lbit2,rtn,svd]
//!     [--seq-len 1024] [--max-docs 48] [--seed 0x036]
//!     [--out <report.md>] [--json <results.json>]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use half::f16;
use katgpt_transformer::MultiLayerKVCache;
use rayon::prelude::*;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gemma_layer::GemmaTransformerWeightsF16;
use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::quant::kvq_harness::nll;
use riir_infer_core::quant::svd_lbit::{
    TruncSvd, bpw_for_rank, dual_svid_from_svd, mix_seed, plain_rank_for_bpw, rank_for_bpw,
    rand_trunc_svd, rand_trunc_svd_oversampled, reconstruct, reconstruct_lowrank, rtn_binary_rows,
};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::{ForwardContext, forward_gemma2_f16};
use riir_infer_core::types::{Config, kv_dim};

/// Default seed (issue number as hex).
const DEFAULT_SEED: u64 = 0x036;
/// Seed salt for the shared per-tensor bases.
const SALT_BASIS: u64 = 0x036_BA5E;
/// Seed salt for lbit2 path-2 (residual) factorizations.
const SALT_PATH2: u64 = 0x036_0F02;

/// Which arms the ladder runs (parsed from `--arms`).
#[derive(Clone, Copy)]
struct ArmSet {
    anchor: bool,
    lbit1: bool,
    lbit2: bool,
    rtn: bool,
    svd: bool,
}

impl ArmSet {
    fn parse(spec: &str) -> Result<Self> {
        let mut set = ArmSet {
            anchor: false,
            lbit1: false,
            lbit2: false,
            rtn: false,
            svd: false,
        };
        for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match name {
                "anchor" => set.anchor = true,
                "lbit1" => set.lbit1 = true,
                "lbit2" => set.lbit2 = true,
                "rtn" => set.rtn = true,
                "svd" => set.svd = true,
                other => bail!("unknown arm '{other}' (anchor|lbit1|lbit2|rtn|svd)"),
            }
        }
        Ok(set)
    }
}

/// One full-model PPL pass.
enum PassSpec {
    Anchor,
    Rtn,
    Lbit1(f64),
    Lbit2(f64),
    Svd(f64),
}

impl PassSpec {
    fn arm_name(&self) -> &'static str {
        match self {
            PassSpec::Anchor => "anchor",
            PassSpec::Rtn => "rtn",
            PassSpec::Lbit1(_) => "lbit1",
            PassSpec::Lbit2(_) => "lbit2",
            PassSpec::Svd(_) => "svd",
        }
    }

    fn target(&self) -> Option<f64> {
        match self {
            PassSpec::Anchor | PassSpec::Rtn => None,
            PassSpec::Lbit1(t) | PassSpec::Lbit2(t) | PassSpec::Svd(t) => Some(*t),
        }
    }

    /// Achieved BPW for one tensor shape under this pass's rule.
    fn bpw_for_shape(&self, dout: usize, din: usize) -> f64 {
        match self {
            PassSpec::Anchor => 16.0,
            PassSpec::Rtn => 1.0 + 32.0 / din as f64,
            PassSpec::Lbit1(t) => bpw_for_rank(rank_for_bpw(*t, dout, din, 1), dout, din, 1),
            PassSpec::Lbit2(t) => bpw_for_rank(rank_for_bpw(*t, dout, din, 2), dout, din, 2),
            PassSpec::Svd(t) => {
                let r = plain_rank_for_bpw(*t, dout, din);
                16.0 * r as f64 * (dout + din) as f64 / (dout * din) as f64
            }
        }
    }
}

/// One recorded pass row.
struct PassRow {
    arm: &'static str,
    target: Option<f64>,
    ppl: f64,
    nll_count: usize,
    transform_s: f32,
    ppl_s: f32,
    loadavg: String,
}

fn parse_u64(s: &str) -> Result<u64> {
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<u64>(),
    };
    parsed.with_context(|| format!("parse u64 from '{s}'"))
}

fn loadavg_best_effort() -> String {
    if let Ok(txt) = std::fs::read_to_string("/proc/loadavg") {
        let first = txt.lines().next().unwrap_or("").to_string();
        return first;
    }
    match std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unavailable".to_string(),
    }
}

fn main() -> Result<()> {
    // ── CLI ──
    let args: Vec<String> = std::env::args().collect();
    let mut gguf_path: Option<PathBuf> = None;
    let mut corpus_path: Option<PathBuf> = None;
    let mut targets = vec![1.0f64, 0.55, 0.3, 0.1];
    let mut arms = ArmSet {
        anchor: true,
        lbit1: true,
        lbit2: true,
        rtn: true,
        svd: true,
    };
    let mut seq_len = 1024usize;
    let mut max_docs = 48usize;
    let mut seed = DEFAULT_SEED;
    let mut out_path: Option<PathBuf> = None;
    let mut json_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        let take = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match args[i].as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(take(&mut i))),
            "--corpus" => corpus_path = Some(PathBuf::from(take(&mut i))),
            "--targets" => {
                let raw = take(&mut i);
                let parsed: Vec<f64> = raw
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<f64>().with_context(|| format!("target '{s}'")))
                    .collect::<Result<_>>()?;
                if parsed.is_empty() {
                    bail!("--targets must list at least one value");
                }
                for &t in &parsed {
                    if !(0.0 < t && t <= 16.0) {
                        bail!("target {t} outside (0, 16] — sanity ceiling; sub-1-bit ladder expected");
                    }
                }
                targets = parsed;
            }
            "--arms" => arms = ArmSet::parse(&take(&mut i))?,
            "--seq-len" => seq_len = take(&mut i).parse().context("--seq-len N")?,
            "--max-docs" => max_docs = take(&mut i).parse().context("--max-docs N")?,
            "--seed" => seed = parse_u64(&take(&mut i))?,
            "--out" => out_path = Some(PathBuf::from(take(&mut i))),
            "--json" => json_path = Some(PathBuf::from(take(&mut i))),
            other => bail!("unknown arg {other}"),
        }
        i += 1;
    }
    let gguf_path = gguf_path.context("--gguf <path> is required")?;
    let corpus_path = corpus_path.context("--corpus <dir-or-txt> is required")?;
    if !(16..=4096).contains(&seq_len) {
        bail!("--seq-len must be 16..=4096 (gemma-2 sliding window; the act_ptq_gemma2 law)");
    }
    if max_docs == 0 {
        bail!("--max-docs must be >= 1");
    }

    // ── Load model + tokenizer from ONE open GGUF (the vk shape) ──
    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let mut weights = load_gemma2_f16_direct(&gguf, &config)?;
    drop(gguf);
    let q_dim = config.n_head * config.head_dim;
    let kvd = kv_dim(&config);
    let n_embd = config.n_embd;
    println!(
        "# svd_lbit_eval: gemma-2 f16 | layers={} n_embd={} q_dim={} kv_dim={} mlp_hidden={} vocab={} | load {:.1}s",
        config.n_layer,
        n_embd,
        q_dim,
        kvd,
        config.mlp_hidden,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );

    // Per-tensor shapes, the act_ptq_gemma2 table exactly: (name, dout, din).
    let shapes: [(&'static str, usize, usize); 7] = [
        ("attn_wq", q_dim, n_embd),
        ("attn_wk", kvd, n_embd),
        ("attn_wv", kvd, n_embd),
        ("attn_wo", n_embd, q_dim),
        ("gate_proj", config.mlp_hidden, n_embd),
        ("up_proj", config.mlp_hidden, n_embd),
        ("down_proj", n_embd, config.mlp_hidden),
    ];

    // ── Corpus → windows (non-overlapping, capped) ──
    let t1 = Instant::now();
    let text = load_corpus_text(&corpus_path)?;
    let all_tokens = tok.encode(&text);
    let windows: Vec<&[usize]> = all_tokens.chunks(seq_len).take(max_docs).collect();
    let token_count: usize = windows.iter().map(|w| w.len()).sum();
    let nll_total: usize = windows.iter().map(|w| w.len().saturating_sub(1)).sum();
    println!(
        "# corpus: {} chars → {} tokens ({:.1}s) | windows {} × ≤{seq_len} (cap {max_docs}) | eval tokens {} | nll {}",
        text.len(),
        all_tokens.len(),
        t1.elapsed().as_secs_f32(),
        windows.len(),
        token_count,
        nll_total,
    );
    if windows.is_empty() || nll_total == 0 {
        bail!("corpus produced no eval windows");
    }

    // ── f16 originals of the 7 linears per layer (the restore source) ──
    let originals: Vec<[Vec<f16>; 7]> = weights
        .layers
        .iter()
        .map(|ly| {
            [
                ly.attn_wq.clone(),
                ly.attn_wk.clone(),
                ly.attn_wv.clone(),
                ly.attn_wo.clone(),
                ly.gate_proj.clone(),
                ly.up_proj.clone(),
                ly.down_proj.clone(),
            ]
        })
        .collect();

    // ── Pass schedule ──
    let mut passes: Vec<PassSpec> = Vec::new();
    if arms.anchor {
        passes.push(PassSpec::Anchor);
    }
    if arms.lbit1 {
        for &t in &targets {
            passes.push(PassSpec::Lbit1(t));
        }
    }
    if arms.lbit2 {
        for &t in &targets {
            passes.push(PassSpec::Lbit2(t));
        }
    }
    if arms.svd {
        for &t in &targets {
            passes.push(PassSpec::Svd(t));
        }
    }
    if arms.rtn {
        passes.push(PassSpec::Rtn); // floor ~1.03 bpw: only meaningful at 1.0
    }

    // ── Shared k_max bases (the T0 decision) ──
    let k_max: [usize; 7] = {
        let mut km = [1usize; 7];
        for (ti, &(_, dout, din)) in shapes.iter().enumerate() {
            let mut need = 1usize;
            for &t in &targets {
                if arms.lbit1 {
                    need = need.max(rank_for_bpw(t, dout, din, 1));
                }
                if arms.lbit2 {
                    need = need.max(rank_for_bpw(t, dout, din, 2));
                }
                if arms.svd {
                    need = need.max(plain_rank_for_bpw(t, dout, din));
                }
            }
            km[ti] = (need + 16).min(dout.min(din));
        }
        km
    };
    let t2 = Instant::now();
    let mut bases: Vec<Vec<TruncSvd>> = (0..config.n_layer).map(|_| Vec::new()).collect();
    bases
        .par_iter_mut()
        .enumerate()
        .for_each(|(l, layer_bases)| {
            *layer_bases = shapes
                .iter()
                .enumerate()
                .map(|(ti, &(_, dout, din))| {
                    let w32: Vec<f32> =
                        originals[l][ti].iter().map(|&v| v.to_f32()).collect();
                    rand_trunc_svd_oversampled(
                        &w32,
                        dout,
                        din,
                        k_max[ti],
                        mix_seed(seed, l as u64, ti as u64, SALT_BASIS),
                    )
                })
                .collect();
        });
    println!(
        "# shared bases: {} layers × 7 tensors, k_max {} in {:.0}s (T0: one basis serves all targets + the svd arm)",
        config.n_layer,
        k_max.iter().map(|k| k.to_string()).collect::<Vec<_>>().join("/"),
        t2.elapsed().as_secs_f32(),
    );

    // ── The ladder ──
    let provenance_start = loadavg_best_effort();
    println!(
        "# PROVENANCE (best-effort, std-only): loadavg {} | workers {} | os {} | seed {seed:#x}",
        provenance_start,
        rayon::current_num_threads(),
        std::env::consts::OS,
    );
    let kv_dims = vec![kvd; config.n_layer];
    let mut rows: Vec<PassRow> = Vec::new();
    for (pass_idx, spec) in passes.iter().enumerate() {
        let label = match spec.target() {
            Some(t) => format!("{}@{t}", spec.arm_name()),
            None => spec.arm_name().to_string(),
        };
        let t3 = Instant::now();
        transform_weights(&mut weights, &originals, &bases, spec, &shapes, seed, pass_idx as u64);
        let transform_s = t3.elapsed().as_secs_f32();
        let t4 = Instant::now();
        let (ppl, nll_count) = eval_ppl(&weights, &config, &kv_dims, &windows, seq_len);
        let ppl_s = t4.elapsed().as_secs_f32();
        let loadavg = loadavg_best_effort();
        println!(
            "# pass {}/{} {:>12}: PPL {ppl:.4} | nll {nll_count} | transform {transform_s:.0}s | ppl {ppl_s:.0}s | loadavg {loadavg}",
            pass_idx + 1,
            passes.len(),
            label,
        );
        rows.push(PassRow {
            arm: spec.arm_name(),
            target: spec.target(),
            ppl,
            nll_count,
            transform_s,
            ppl_s,
            loadavg,
        });
    }

    // ── Report ──
    let mut out = String::new();
    out.push_str("# Issue 036 T1/T2 — svd_lbit init-only PPL ladder (gemma-2-2b-it f16)\n\n");
    out.push_str(&format!(
        "model: {} | layers {} | n_embd {} | q_dim {} | kv_dim {} | mlp_hidden {} | vocab {}\n\n",
        gguf_path.display(),
        config.n_layer,
        n_embd,
        q_dim,
        kvd,
        config.mlp_hidden,
        config.vocab_size,
    ));
    out.push_str(&format!(
        "corpus: {} | windows {} × ≤{seq_len} (cap {max_docs}, non-overlapping) | eval tokens {token_count} | nll {nll_total} | seed {seed:#x}\n\n",
        corpus_path.display(),
        windows.len(),
    ));
    out.push_str(&format!(
        "PROVENANCE (best-effort, std-only): start loadavg {provenance_start} | workers {} | os {} — per-pass loadavg in the table\n\n",
        rayon::current_num_threads(),
        std::env::consts::OS,
    ));
    out.push_str("## PPL ladder\n\n| arm | target | PPL | nll n | transform s | ppl s | loadavg |\n|---|---|---|---|---|---|---|\n");
    for r in &rows {
        let target = match r.target {
            Some(t) => format!("{t}"),
            None => "—".to_string(),
        };
        out.push_str(&format!(
            "| {} | {} | {:.4} | {} | {:.0} | {:.0} | {} |\n",
            r.arm, target, r.ppl, r.nll_count, r.transform_s, r.ppl_s, r.loadavg,
        ));
    }
    out.push_str("\n## Achieved bpw per tensor (by pass)\n\n| pass | attn_wq | attn_wk | attn_wv | attn_wo | gate_proj | up_proj | down_proj |\n|---|---|---|---|---|---|---|---|\n");
    for spec in &passes {
        let label = match spec.target() {
            Some(t) => format!("{}@{t}", spec.arm_name()),
            None => spec.arm_name().to_string(),
        };
        let cells: Vec<String> = shapes
            .iter()
            .map(|&(_, dout, din)| format!("{:.3}", spec.bpw_for_shape(dout, din)))
            .collect();
        out.push_str(&format!("| {label} | {} |\n", cells.join(" | ")));
    }
    out.push_str(
        "\nDisclosures: ONE k_max Halko basis per tensor serves lbit1 + lbit2-path1 + svd at ALL targets (T0 — exact per-r SVD is a documented approximation class); lbit2 path 2 factorizes the per-target residual freshly; rtn arms ONLY at 1.0 (its floor is ~1.03 bpw, printed per shape above); eval is non-overlapping teacher-forced windows, f64 accumulation.\n\nMEASUREMENT-ONLY (Issue 036 P0 law): the bin never writes the model; artifacts are this report + the JSON sidecar only.\n",
    );
    print!("{out}");
    if let Some(p) = &out_path {
        std::fs::write(p, &out).with_context(|| format!("write report {}", p.display()))?;
        println!("# report written: {}", p.display());
    }
    if let Some(p) = &json_path {
        let passes_json: Vec<serde_json::Value> = passes
            .iter()
            .zip(&rows)
            .map(|(spec, r)| {
                let bpw: serde_json::Map<String, serde_json::Value> = shapes
                    .iter()
                    .map(|&(name, dout, din)| {
                        (name.to_string(), serde_json::json!(spec.bpw_for_shape(dout, din)))
                    })
                    .collect();
                serde_json::json!({
                    "arm": r.arm,
                    "target": r.target,
                    "ppl": r.ppl,
                    "nll_count": r.nll_count,
                    "transform_s": r.transform_s,
                    "ppl_s": r.ppl_s,
                    "loadavg": r.loadavg,
                    "bpw": bpw,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "issue": "riir-infer 036 T1/T2",
            "gguf": gguf_path.display().to_string(),
            "corpus": corpus_path.display().to_string(),
            "seed": format!("{seed:#x}"),
            "seq_len": seq_len,
            "max_docs": max_docs,
            "windows": windows.len(),
            "tokens": token_count,
            "nll_total": nll_total,
            "model": {
                "n_layer": config.n_layer,
                "n_embd": n_embd,
                "q_dim": q_dim,
                "kv_dim": kvd,
                "mlp_hidden": config.mlp_hidden,
                "vocab_size": config.vocab_size,
            },
            "provenance_start": provenance_start,
            "passes": passes_json,
        });
        std::fs::write(p, serde_json::to_string_pretty(&doc)?)
            .with_context(|| format!("write json {}", p.display()))?;
        println!("# json written: {}", p.display());
    }
    Ok(())
}

/// Transform all 7 linear tensors of every layer in place (rayon across
/// layers — pure functions per tensor, disjoint writes). The anchor arm is
/// the f16→f32→f16 roundtrip (exact for f16), i.e. identity.
fn transform_weights(
    weights: &mut GemmaTransformerWeightsF16,
    originals: &[[Vec<f16>; 7]],
    bases: &[Vec<TruncSvd>],
    spec: &PassSpec,
    shapes: &[(&'static str, usize, usize); 7],
    seed: u64,
    pass_idx: u64,
) {
    weights
        .layers
        .par_iter_mut()
        .enumerate()
        .for_each(|(l, layer)| {
            let mut tensors: [(&mut Vec<f16>, usize, usize); 7] = [
                (&mut layer.attn_wq, shapes[0].1, shapes[0].2),
                (&mut layer.attn_wk, shapes[1].1, shapes[1].2),
                (&mut layer.attn_wv, shapes[2].1, shapes[2].2),
                (&mut layer.attn_wo, shapes[3].1, shapes[3].2),
                (&mut layer.gate_proj, shapes[4].1, shapes[4].2),
                (&mut layer.up_proj, shapes[5].1, shapes[5].2),
                (&mut layer.down_proj, shapes[6].1, shapes[6].2),
            ];
            for (ti, (live, dout, din)) in tensors.iter_mut().enumerate() {
                let (dout, din) = (*dout, *din);
                let w32: Vec<f32> = originals[l][ti].iter().map(|&v| v.to_f32()).collect();
                let out32: Vec<f32> = match spec {
                    PassSpec::Anchor => w32,
                    PassSpec::Rtn => rtn_binary_rows(&w32, dout, din),
                    PassSpec::Lbit1(t) => {
                        let r = rank_for_bpw(*t, dout, din, 1);
                        reconstruct(&dual_svid_from_svd(&bases[l][ti], r, 1))
                    }
                    PassSpec::Lbit2(t) => {
                        let r = rank_for_bpw(*t, dout, din, 2);
                        let w1 = reconstruct(&dual_svid_from_svd(&bases[l][ti], r, 2));
                        let mut residual = w32;
                        for (rr, &a) in residual.iter_mut().zip(w1.iter()) {
                            *rr -= a;
                        }
                        let svd2 = rand_trunc_svd(
                            &residual,
                            dout,
                            din,
                            r,
                            mix_seed(seed, l as u64, ti as u64, SALT_PATH2 ^ pass_idx),
                        );
                        let w2 = reconstruct(&dual_svid_from_svd(&svd2, r, 2));
                        let mut out = w1;
                        for (o, &b) in out.iter_mut().zip(w2.iter()) {
                            *o += b;
                        }
                        out
                    }
                    PassSpec::Svd(t) => {
                        let rp = plain_rank_for_bpw(*t, dout, din);
                        reconstruct_lowrank(&bases[l][ti], rp)
                    }
                };
                live.clear();
                live.extend(out32.iter().map(|&x| f16::from_f32(x)));
            }
        });
}

/// Teacher-forced PPL over the windows (rayon across windows; each worker
/// owns a ForwardContext + a seq_len-BOUNDED KV cache — 0.21 GB vs 1.7 GB
/// at full block_size, positions stay < seq_len so the bounded cache is
/// exact). Per-window sums reduce in window order (deterministic).
fn eval_ppl(
    weights: &GemmaTransformerWeightsF16,
    config: &Config,
    kv_dims: &[usize],
    windows: &[&[usize]],
    seq_len: usize,
) -> (f64, usize) {
    let sums: Vec<(f64, usize)> = windows
        .par_iter()
        .map_init(
            || {
                (
                    ForwardContext::new(config),
                    MultiLayerKVCache::new_with_per_layer_kv_dim_bounded(config, kv_dims, seq_len),
                )
            },
            |(ctx, cache), &win| {
                cache.reset();
                let mut s = 0.0f64;
                let mut n = 0usize;
                for (pos, &token) in win.iter().enumerate() {
                    let logits =
                        forward_gemma2_f16(ctx, weights, cache, token, pos, config);
                    if pos + 1 < win.len() {
                        s += nll(logits, win[pos + 1]);
                        n += 1;
                    }
                }
                (s, n)
            },
        )
        .collect();
    let total: f64 = sums.iter().map(|(s, _)| *s).sum();
    let count: usize = sums.iter().map(|(_, n)| *n).sum();
    ((total / count.max(1) as f64).exp(), count)
}
