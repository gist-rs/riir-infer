//! Issue 027 / Plan 615 T6 — the MODEL-level per-family retention gate for
//! the Q2_0 grid lane.
//!
//! Arms are WEIGHT-QUANTIZATION postures over the same token sequences:
//! `base` (untouched f32), `sym` (the reference RTN encoder), `t0` (the
//! asymmetric 4th-state encoder), `q2_0a` (the Lloyd-Max grid via
//! `--l0`/`--l2`, decoded through `dequantize_row_q2_0_grid`). Each quant
//! arm round-trips every 2-D weight tensor through the wire format and
//! runs the standard f32 forward on the DEQUANTIZED weights — the
//! fakequant-at-load pattern (error injection location = the weights; the
//! forward is identical). Every quant arm builds from the PRISTINE weights
//! (postures never compound).
//!
//! Per-family retention is the lossy-surface law's requirement (never
//! aggregate ppl alone): every `--corpus` file is its own FAMILY, chunked
//! into `--seq-len` sequences; each family reports its own Δppl beside the
//! aggregate, so a family-conditional regression cannot vanish into the
//! mean.
//!
//! Vehicle: the llama path (f32 `LlamaTransformerWeights`), e.g.
//! MiniCPM5-1B. The gemma2 f16 path is a later arm (its weights would
//! convert f16→f32 first — same math, more memory).
//!
//! Usage:
//! ```text
//! cargo run --release --features lut_grid --bin lut_grid_ppl -- \
//!     --gguf ../riir-train/data/MiniCPM5-1B-F16.gguf \
//!     --corpus <chat.txt> --corpus <docs.txt> \
//!     --tokens 4096 --seq-len 512 --l0 -0.816637 --l2 0.708580 \
//!     --arms base,sym,t0,q2_0a
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::gguf_loader::{GgufFile, load_llama_weights_gguf};
use riir_infer_core::llama_layer::{LlamaLayerWeights, LlamaTransformerWeights};
use riir_infer_core::quant::lut_grid::Q2Grid;
use riir_infer_core::quant::q2_0::{
    dequantize_row_q2_0, dequantize_row_q2_0_grid, quantize_row_q2_0_asymmetric,
    quantize_row_q2_0_grid, quantize_row_q2_0_symmetric,
};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::transformer::{ForwardContext, forward_llama};
use riir_infer_core::types::Config;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Posture {
    Base,
    Sym,
    T0,
    Q2oa,
    /// T0 math at a FINER group (fakequant only — the wire is per-128; the
    /// posture exists to test whether granularity rescues the class at the
    /// model level, the T5 model-level half).
    T0g64,
    T0g32,
}

impl Posture {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "base" => Ok(Posture::Base),
            "sym" => Ok(Posture::Sym),
            "t0" => Ok(Posture::T0),
            "q2_0a" => Ok(Posture::Q2oa),
            "t0g64" => Ok(Posture::T0g64),
            "t0g32" => Ok(Posture::T0g32),
            other => bail!("unknown posture {other} (base|sym|t0|q2_0a|t0g64|t0g32)"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Posture::Base => "base",
            Posture::Sym => "sym",
            Posture::T0 => "t0",
            Posture::Q2oa => "q2_0a",
            Posture::T0g64 => "t0g64",
            Posture::T0g32 => "t0g32",
        }
    }

    /// The quantization group for the round-trip (the wire is per-128; the
    /// granularity postures quantize per-64/per-32 as a fakequant probe).
    const fn group(self) -> usize {
        match self {
            Posture::T0g64 => 64,
            Posture::T0g32 => 32,
            _ => 128,
        }
    }
}

/// Round-trip one 2-D weight tensor through a posture's wire format (or
/// the granularity postures' fakequant group).
fn roundtrip(t: &mut [f32], posture: Posture, grid: Q2Grid) -> Result<()> {
    let group = posture.group();
    if !t.len().is_multiple_of(group) {
        bail!("weight tensor length {} is not a multiple of {group}", t.len());
    }
    let mut blocks = Vec::new();
    let t0_group = |chunk: &[f32], out: &mut Vec<f32>| {
        // T0 math at an arbitrary group (fakequant; no wire layout).
        let Some((_, d)) = riir_infer_core::quant::q2_0::t0_block_scale(chunk) else {
            out.iter_mut().for_each(|o| *o = 0.0);
            return;
        };
        for (o, &v) in out.iter_mut().zip(chunk) {
            let q = (v / d).round().clamp(-1.0, 2.0);
            *o = q * d;
        }
    };
    match posture {
        Posture::Base => unreachable!("base never round-trips"),
        Posture::Sym => {
            quantize_row_q2_0_symmetric(t, &mut blocks);
            dequantize_row_q2_0(&blocks, t);
        }
        Posture::T0 => {
            quantize_row_q2_0_asymmetric(t, &mut blocks);
            dequantize_row_q2_0(&blocks, t);
        }
        Posture::Q2oa => {
            quantize_row_q2_0_grid(t, grid, &mut blocks);
            dequantize_row_q2_0_grid(&blocks, t, grid);
        }
        Posture::T0g64 | Posture::T0g32 => {
            // Group-wise in place (chunks of `group`; group divides 128 so
            // every wire-block length is also a multiple of these).
            let mut fixed = vec![0f32; group];
            for chunk_start in (0..t.len()).step_by(group) {
                let chunk = &t[chunk_start..chunk_start + group];
                fixed.copy_from_slice(chunk);
                t0_group(chunk, &mut fixed);
                t[chunk_start..chunk_start + group].copy_from_slice(&fixed);
            }
        }
    }
    Ok(())
}

/// Apply a posture to every 2-D weight tensor (norms are 1-D and skip).
/// `skip_embed` leaves wte/lm_head at full precision (the standard
/// practice the ablation arm exists to test — 2-bit logits projections
/// are the prime damage-locus suspect).
fn quantize_weights(
    w: &mut LlamaTransformerWeights,
    posture: Posture,
    grid: Q2Grid,
    skip_embed: bool,
) -> Result<()> {
    if posture == Posture::Base {
        return Ok(());
    }
    if !skip_embed {
        roundtrip(&mut w.wte, posture, grid)?;
        roundtrip(&mut w.lm_head, posture, grid)?;
    };
    for layer in &mut w.layers {
        let LlamaLayerWeights {
            attn_wq,
            attn_wk,
            attn_wv,
            attn_wo,
            gate_proj,
            up_proj,
            down_proj,
            ..
        } = layer;
        for t in [attn_wq, attn_wk, attn_wv, attn_wo, gate_proj, up_proj, down_proj] {
            roundtrip(t, posture, grid)?;
        }
    }
    Ok(())
}

fn nll(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    -((logits[target] - max).exp() as f64 / sum as f64).ln()
}

struct Family {
    name: String,
    seqs: Vec<Vec<usize>>,
}

#[derive(Default)]
struct Acc {
    sum_nll: f64,
    abs_sum: f64,
    k: usize,
    fwd: usize,
}

/// Score every family's every sequence with one weight set. `base` is the
/// pristine run's per-token NLL keyed by the SAME global index (family-
/// major, seq-major); the base run fills `base_out` instead.
fn run_posture(
    w: &LlamaTransformerWeights,
    cfg: &Config,
    families: &[Family],
    acc: &mut Acc,
    base: Option<&[f64]>,
    base_out: &mut Vec<f64>,
) -> Vec<f64> {
    let mut ctx = ForwardContext::new(cfg);
    let mut cache = MultiLayerKVCache::new(cfg);
    let mut fam_ppl: Vec<f64> = Vec::with_capacity(families.len());
    for f in families {
        let mut fam_nll = 0f64;
        let mut fam_k = 0usize;
        for seq in &f.seqs {
            cache.reset();
            for pos in 0..seq.len() - 1 {
                let logits = forward_llama(&mut ctx, w, &mut cache, seq[pos], pos, cfg);
                acc.fwd += 1;
                let l = nll(logits, seq[pos + 1]);
                acc.sum_nll += l;
                fam_nll += l;
                fam_k += 1;
                match base {
                    None => base_out.push(l),
                    Some(b) => {
                        acc.abs_sum += (l - b[acc.k]).abs();
                    }
                }
                acc.k += 1;
            }
        }
        fam_ppl.push((fam_nll / fam_k as f64).exp());
    }
    fam_ppl
}

fn main() -> Result<()> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut corpora: Vec<PathBuf> = Vec::new();
    let mut n_tokens = 4096usize;
    let mut seq_len = 512usize;
    let mut l0 = 0f32;
    let mut l2 = 0f32;
    let mut postures = vec![Posture::Base, Posture::Sym, Posture::T0, Posture::Q2oa];
    let mut skip_embed = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(args.next().context("--gguf needs a path")?)),
            "--corpus" => corpora.push(PathBuf::from(args.next().context("--corpus needs a path")?)),
            "--tokens" => n_tokens = args.next().context("--tokens needs N")?.parse()?,
            "--seq-len" => seq_len = args.next().context("--seq-len needs N")?.parse()?,
            "--l0" => l0 = args.next().context("--l0 needs f32")?.parse()?,
            "--l2" => l2 = args.next().context("--l2 needs f32")?.parse()?,
            "--skip-embed" => skip_embed = args.next().context("--skip-embed needs bool")?.parse()?,
            "--arms" => {
                let spec = args.next().context("--arms needs a list")?;
                postures = spec.split(',').map(Posture::parse).collect::<Result<_>>()?;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    let gguf_path = gguf_path.context("--gguf required")?;
    if corpora.is_empty() {
        bail!("at least one --corpus required (each file is one family)");
    }
    if !seq_len.is_multiple_of(128) {
        bail!("--seq-len must be a multiple of 128 (chunk = family granularity)");
    }
    let grid = Q2Grid { l0, l2 };
    if postures.contains(&Posture::Q2oa) && (l0 == 0.0 || l2 == 0.0) {
        bail!("q2_0a arm needs --l0/--l2 (the committed grid for this artifact)");
    }
    if !postures.contains(&Posture::Base) {
        bail!("the first arm must be base (paired deltas are against it)");
    }

    // ── Load: config + f32 weights + tokenizer ─────────────────────
    let tok = {
        let gguf = GgufFile::open(&gguf_path)?;
        if gguf.architecture() != Some("llama") {
            bail!("lut_grid_ppl v1 is the llama path (e.g. MiniCPM5-1B)");
        }
        BpeTokenizer::from_gguf(&gguf)?
    };
    let bos = tok.bos_id();
    let (mut cfg, pristine) = load_llama_weights_gguf(&gguf_path)?;
    let longest = seq_len + 1;
    if longest > cfg.block_size {
        bail!("seq-len {} exceeds block_size {}", seq_len, cfg.block_size);
    }
    cfg.block_size = longest;

    // ── Families: each corpus file = one family, chunked ───────────
    let mut families: Vec<Family> = Vec::new();
    for path in &corpora {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let ids = tok.encode(&text);
        let seqs: Vec<Vec<usize>> = ids[..ids.len().min(n_tokens)]
            .chunks(seq_len)
            .map(|c| std::iter::once(bos).chain(c.iter().copied()).collect())
            .collect();
        if seqs.is_empty() {
            bail!("family {} produced no chunks", path.display());
        }
        families.push(Family {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("family")
                .to_string(),
            seqs,
        });
    }
    let total_seqs: usize = families.iter().map(|f| f.seqs.len()).sum();
    println!(
        "# lut_grid_ppl: llama path | layers={} heads={}/{} | {} posture(s) × {} families, {} seqs of {} tok | grid l0={l0} l2={l2} | skip_embed={skip_embed}",
        cfg.n_layer,
        cfg.n_head,
        cfg.n_kv_head,
        postures.len(),
        families.len(),
        total_seqs,
        seq_len
    );

    let mut base_nll: Vec<f64> = Vec::new();
    let mut results: Vec<(String, Acc, Vec<f64>)> = Vec::new();

    for posture in postures {
        let t = std::time::Instant::now();
        let mut acc = Acc::default();
        let fam_ppl = match posture {
            Posture::Base => run_posture(&pristine, &cfg, &families, &mut acc, None, &mut base_nll),
            p => {
                // Every quant posture builds from the PRISTINE weights —
                // postures never compound.
                let mut w = pristine.clone();
                quantize_weights(&mut w, p, grid, skip_embed)?;
                run_posture(&w, &cfg, &families, &mut acc, Some(&base_nll), &mut Vec::new())
            }
        };
        println!(
            "# posture {} done in {:.1}s ({} fwd, {:.2} tok/s)",
            posture.name(),
            t.elapsed().as_secs_f64(),
            acc.fwd,
            acc.fwd as f64 / t.elapsed().as_secs_f64().max(1e-9)
        );
        results.push((posture.name().to_string(), acc, fam_ppl));
    }

    let base_acc = &results[0].1;
    let base_ppl = (base_acc.sum_nll / base_acc.k as f64).exp();
    println!("\n| posture | ppl | Δppl | mean |ΔNLL| vs base |");
    println!("|---|---|---|---|");
    for (name, acc, _) in &results {
        let ppl = (acc.sum_nll / acc.k as f64).exp();
        if *name == "base" {
            println!("| {name} | {ppl:.4} | — | — |");
        } else {
            println!(
                "| {name} | {ppl:.4} | {:+.3}% | {:.5} |",
                100.0 * (ppl / base_ppl - 1.0),
                acc.abs_sum / acc.k as f64
            );
        }
    }
    println!("\n# per-family retention (Δppl% vs base; base row = the family's own ppl) — a family-conditional regression must not hide here");
    print!("| posture");
    for f in &families {
        print!(" | {}", f.name);
    }
    println!(" |");
    print!("|---");
    for _ in &families {
        print!("|---");
    }
    println!("|");
    for (name, _, fam_ppl) in &results {
        print!("| {name}");
        for (i, fp) in fam_ppl.iter().enumerate() {
            let bp = results[0].2[i];
            if *name == "base" {
                print!(" | {bp:.4}");
            } else {
                print!(" | {:+.3}%", 100.0 * (fp / bp - 1.0));
            }
        }
        println!(" |");
    }
    Ok(())
}
