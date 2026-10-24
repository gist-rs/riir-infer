#![cfg(all(
    feature = "flashmemory_budget",
    feature = "deltanet_ternary_inference",
    feature = "bonsai2_hadamard"
))]
//! Issue 1117 T3/T4 — the FlashMemory budget (top-ρ) decode gate.
//!
//! The end-to-end quality + perf read the per-layer fidelity numbers
//! (riir-train Bench 671) cannot answer: does per-layer block selection
//! survive every attention layer compounding through the real hybrid stack?
//!
//! # Rig
//!
//! Teacher-forced decode over a long-context token slice: both arms step the
//! REAL forward one token at a time over the same ground-truth token stream
//! (no sampling — the comparison isolates the attention-read pattern). The
//! DENSE arm (`forward_qwen_deltanet_ternary`, bounded cache) runs once; each
//! ρ arm (`forward_qwen_deltanet_ternary_flashmemory`, block 64, refresh 64,
//! incremental centroids) runs against its own fresh cache + state.
//!
//! # Metrics (per position, bucketed)
//!
//! - `ΔNLL = NLL_sparse − NLL_dense` per position (mean / p95 / max, with the
//!   bucket's tail support reported beside the p95 — a percentile over a
//!   handful of rows is the max wearing a name).
//! - top-1 agreement: argmax_sparse == argmax_dense per position.
//! - retention: sampled mean selected-token fraction over the attention
//!   layers (every `--retain-every` steps).
//! - tok/s per arm (the T4 read; CPU posture, box state recorded in the
//!   bench doc by the operator — this binary prints timing only).
//!
//! # Corpus
//!
//! `--corpus-file` is REQUIRED — pass a plain-text file (≥ ctx+1 tokens after
//! BPE). The token stream is the file's first ctx+1 tokens; positions
//! compared start at `--compare-from` (0 = compare everything).
//!
//! # Example (the league posture)
//!
//! ```sh
//! cargo run --release --features flashmemory_budget,deltanet_ternary_inference,bonsai2_hadamard \
//!   --bin flashmemory_decode_gate -- \
//!   --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --corpus-file /tmp/corpus.txt --ctx 32768 --rho-list 0.05,0.10,0.15,0.30 \
//!   --bucket 4096 --out /tmp/fm_decode_gate
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use katgpt_attn::dash_attn::flashmemory_sparse::FlashMemoryConfig;
use riir_infer_core::deltanet::flashmemory_gqa::{FlashMemoryDecodeState, kv_reduction_ratio};
use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_flashmemory,
};
use riir_infer_core::deltanet::{HybridCache, HybridForwardScratch};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::quant::kvq_harness::nll;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

const DEFAULT_GGUF: &str = "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

struct Args {
    gguf: PathBuf,
    corpus_file: PathBuf,
    ctx: usize,
    rho_list: Vec<f32>,
    block: usize,
    refresh: usize,
    bucket: usize,
    compare_from: usize,
    max_steps: usize,
    retain_every: usize,
    out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        gguf: PathBuf::from(
            std::env::var("BONSAI_PQ2_0_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string()),
        ),
        corpus_file: PathBuf::new(),
        ctx: 32_768,
        rho_list: vec![0.05, 0.10, 0.15, 0.30],
        block: 64,
        refresh: 64,
        bucket: 4096,
        compare_from: 0,
        max_steps: 0,
        retain_every: 16,
        out: PathBuf::from("/tmp/fm_decode_gate"),
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let name: &str = &args[i];
        let value = || args[i + 1].clone();
        match name {
            "--gguf" => a.gguf = value().into(),
            "--corpus-file" => a.corpus_file = value().into(),
            "--ctx" => a.ctx = value().parse().unwrap(),
            "--rho-list" => {
                a.rho_list = value()
                    .split(',')
                    .map(|s| s.trim().parse().unwrap())
                    .collect();
            }
            "--block" => a.block = value().parse().unwrap(),
            "--refresh" => a.refresh = value().parse().unwrap(),
            "--bucket" => a.bucket = value().parse().unwrap(),
            "--compare-from" => a.compare_from = value().parse().unwrap(),
            "--max-steps" => a.max_steps = value().parse().unwrap(),
            "--retain-every" => a.retain_every = value().parse().unwrap(),
            "--out" => a.out = value().into(),
            _ => anyhow::bail!("unknown arg {name} (see the module doc)"),
        }
        i += 2;
    }
    if a.corpus_file.as_os_str().is_empty() {
        anyhow::bail!("--corpus-file is required (plain text, >= ctx+1 tokens)");
    }
    if a.rho_list.is_empty() {
        anyhow::bail!("--rho-list must name at least one rho");
    }
    for &rho in &a.rho_list {
        assert!(rho > 0.0 && rho <= 1.0, "rho {rho} outside (0, 1]");
    }
    assert!(a.bucket > 0 && a.block > 0 && a.refresh > 0);
    if a.max_steps == 0 {
        a.max_steps = a.ctx;
    }
    assert!(
        a.max_steps <= a.ctx,
        "--max-steps {} exceeds --ctx {}",
        a.max_steps,
        a.ctx
    );
    Ok(a)
}

/// One arm's per-position observation.
struct ArmScore {
    nll: Vec<f32>,
    argmax: Vec<usize>,
    /// Sampled mean selected-token fraction (sparse arms only; dense = 1.0).
    mean_selected_fraction: f64,
    wall_secs: f64,
}

fn percentile_with_support(rows: &mut [f32], p: f64) -> (f32, usize) {
    rows.sort_by(|a, b| a.total_cmp(b));
    let n = rows.len();
    let idx = (((n as f64) * p) as usize).min(n.saturating_sub(1));
    // Tail support = observations at or above the reported order statistic —
    // under ~10 the "p95" is the max wearing a name and is reported as such.
    (rows[idx], n - idx)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;

    // ── Model + tokenizer ──────────────────────────────────────────────────
    println!("# loading {}", args.gguf.display());
    let load_start = Instant::now();
    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    println!(
        "# loaded in {:.1}s — layers={} attn={} embd={} vocab={} kv_heads={} head_dim={}",
        load_start.elapsed().as_secs_f64(),
        config.n_layer,
        weights
            .layer_types
            .iter()
            .filter(|&&lt| lt == riir_infer_core::types::DeltaNetLayerType::Attention)
            .count(),
        config.n_embd,
        config.vocab_size,
        config.n_kv_head,
        config.head_dim,
    );

    let gguf = GgufFile::open(&args.gguf).context("re-open gguf for tokenizer")?;
    let tok = BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?;
    let text = std::fs::read_to_string(&args.corpus_file)
        .with_context(|| format!("read corpus {}", args.corpus_file.display()))?;
    let all_tokens = tok.encode(&text);
    let need = args.ctx + 1;
    anyhow::ensure!(
        all_tokens.len() >= need,
        "corpus tokenizes to {} tokens < ctx+1 = {} — pass a longer file",
        all_tokens.len(),
        need
    );
    let tokens: Vec<usize> = all_tokens[..need].to_vec();
    println!(
        "# corpus {} → {} tokens (need {need}); stepped positions 0..{}",
        args.corpus_file.display(),
        all_tokens.len(),
        args.max_steps
    );

    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let n_attn = weights
        .layer_types
        .iter()
        .filter(|&&lt| lt == riir_infer_core::types::DeltaNetLayerType::Attention)
        .count();
    let last_attn_layer = weights
        .layer_types
        .iter()
        .rposition(|&lt| lt == riir_infer_core::types::DeltaNetLayerType::Attention)
        .expect("at least one attention layer");

    let steps = args.max_steps;
    let x_len = config.vocab_size.max(config.n_embd);

    // ── Dense reference arm ────────────────────────────────────────────────
    println!("# arm: dense (reference)");
    let t0 = Instant::now();
    let mut cache = HybridCache::with_layer_types_bounded(&config, &weights.layer_types, steps + 2);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; x_len];
    let mut dense_nll = vec![0f32; steps];
    let mut dense_argmax = vec![0usize; steps];
    for pos in 0..steps {
        let logits = forward_qwen_deltanet_ternary(
            &mut x,
            &weights,
            &mut cache,
            tokens[pos],
            pos,
            &config,
            &mut scratch,
            &rope,
        );
        dense_nll[pos] = nll(logits, tokens[pos + 1]) as f32;
        dense_argmax[pos] = argmax(logits);
    }
    let dense_wall = t0.elapsed().as_secs_f64();
    drop(cache);
    let dense = ArmScore {
        nll: dense_nll,
        argmax: dense_argmax,
        mean_selected_fraction: 1.0,
        wall_secs: dense_wall,
    };
    println!(
        "#   wall {:.1}s — {:.1} tok/s",
        dense_wall,
        steps as f64 / dense_wall
    );

    // ── Sparse arms ────────────────────────────────────────────────────────
    let mut arms: Vec<(f32, ArmScore)> = Vec::new();
    for &rho in &args.rho_list {
        println!(
            "# arm: flashmemory rho={rho} (block {} refresh {})",
            args.block, args.refresh
        );
        let t0 = Instant::now();
        let mut cache =
            HybridCache::with_layer_types_bounded(&config, &weights.layer_types, steps + 2);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut fm = FlashMemoryDecodeState::new(
            &config,
            &weights.layer_types,
            steps + 2,
            FlashMemoryConfig {
                block_size: args.block,
                refresh_period: args.refresh,
                threshold: 0.5, // unused by the budget rule
            },
            rho,
        );
        let mut x = vec![0.0f32; x_len];
        let mut arm_nll = vec![0f32; steps];
        let mut arm_argmax = vec![0usize; steps];
        let mut selected_sum = 0f64;
        let mut selected_n = 0f64;
        for pos in 0..steps {
            let logits = forward_qwen_deltanet_ternary_flashmemory(
                &mut x,
                &weights,
                &mut cache,
                &mut fm,
                tokens[pos],
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            arm_nll[pos] = nll(logits, tokens[pos + 1]) as f32;
            arm_argmax[pos] = argmax(logits);
            if pos % args.retain_every == 0
                && pos > 0
                && let Some(st) = &fm.layers[last_attn_layer]
            {
                let selection = st.selector.selection();
                let (selected, total, _) = kv_reduction_ratio(selection, &st.block_cache, pos + 1);
                if total > 0 {
                    selected_sum += selected as f64 / total as f64;
                    selected_n += 1.0;
                }
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        drop(cache);
        let mean_frac = if selected_n > 0.0 {
            selected_sum / selected_n
        } else {
            1.0
        };
        println!(
            "#   wall {:.1}s — {:.1} tok/s — mean selected fraction {:.4} (last attn layer, sampled every {})",
            wall,
            steps as f64 / wall,
            mean_frac,
            args.retain_every
        );
        arms.push((
            rho,
            ArmScore {
                nll: arm_nll,
                argmax: arm_argmax,
                mean_selected_fraction: mean_frac,
                wall_secs: wall,
            },
        ));
    }

    // ── Comparison + report ────────────────────────────────────────────────
    let mut md = String::new();
    let mut json_rows = Vec::new();
    md.push_str(&format!(
        "# FlashMemory budget decode gate (Issue 1117 T3/T4)\n\n\
         model `{}` · block {} · refresh {} · steps {steps} · bucket {} · compare-from {} · incremental centroids\n\n\
         | arm | mean sel frac | tok/s | pooled ΔNLL mean | pooled ΔNLL p95 (support) | pooled ΔNLL max | top-1 agree |\n\
         |---|---|---|---|---|---|---|\n",
        args.gguf.display(),
        args.block,
        args.refresh,
        args.bucket,
        args.compare_from,
    ));
    for (rho, arm) in &arms {
        let deltas: Vec<f32> = arm.nll[args.compare_from..]
            .iter()
            .zip(&dense.nll[args.compare_from..])
            .map(|(a, b)| a - b)
            .collect();
        let agree = arm.argmax[args.compare_from..]
            .iter()
            .zip(&dense.argmax[args.compare_from..])
            .filter(|(a, b)| a == b)
            .count();
        let n_cmp = deltas.len();
        let mean: f32 = deltas.iter().sum::<f32>() / n_cmp as f32;
        let mut sorted = deltas.clone();
        let (p95, support) = percentile_with_support(&mut sorted, 0.95);
        let max = sorted[n_cmp - 1];
        let agree_pct = 100.0 * agree as f64 / n_cmp as f64;
        md.push_str(&format!(
            "| ρ {rho} | {:.4} | {:.1} | {mean:+.4} | {p95:+.4} ({support}) | {max:+.4} | {agree_pct:.2}% |\n",
            arm.mean_selected_fraction,
            steps as f64 / arm.wall_secs,
        ));
        json_rows.push(serde_json::json!({
            "rho": rho,
            "mean_selected_fraction": arm.mean_selected_fraction,
            "tok_per_s": steps as f64 / arm.wall_secs,
            "delta_nll_mean": mean,
            "delta_nll_p95": p95,
            "delta_nll_p95_support": support,
            "delta_nll_max": max,
            "top1_agreement_pct": agree_pct,
        }));

        // Per-bucket rows (sparse arm detail).
        md.push_str(&format!(
            "\n### ρ {rho} — per-bucket (width {})\n\n| bucket | n | ΔNLL mean | ΔNLL p95 (support) | ΔNLL max | top-1 agree |\n|---|---|---|---|---|---|\n",
            args.bucket
        ));
        let mut bstart = args.compare_from - args.compare_from % args.bucket;
        while bstart < steps {
            let bend = (bstart + args.bucket).min(steps);
            let lo = bstart.max(args.compare_from);
            if bend > lo {
                let mut bdelta: Vec<f32> = arm.nll[lo..bend]
                    .iter()
                    .zip(&dense.nll[lo..bend])
                    .map(|(a, b)| a - b)
                    .collect();
                let bn = bdelta.len();
                let bmean: f32 = bdelta.iter().sum::<f32>() / bn as f32;
                let (bp95, bsupport) = percentile_with_support(&mut bdelta, 0.95);
                let bmax = bdelta.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let bagree = arm.argmax[lo..bend]
                    .iter()
                    .zip(&dense.argmax[lo..bend])
                    .filter(|(a, b)| a == b)
                    .count();
                md.push_str(&format!(
                    "| [{bstart}, {bend}) | {bn} | {bmean:+.4} | {bp95:+.4} ({bsupport}) | {bmax:+.4} | {:.2}% |\n",
                    100.0 * bagree as f64 / bn as f64
                ));
            }
            bstart += args.bucket;
        }
    }

    // Dense arm summary line + run config.
    md.push_str(&format!(
        "\nDense reference: {:.1} tok/s; pooled NLL mean {:.4}.\n",
        steps as f64 / dense.wall_secs,
        dense.nll[args.compare_from..].iter().sum::<f32>() as f64
            / (steps - args.compare_from) as f64
    ));

    println!("{md}");
    std::fs::write(args.out.join("report.md"), &md)?;
    std::fs::write(
        args.out.join("report.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "issue": 1117,
            "gguf": args.gguf.display().to_string(),
            "corpus": args.corpus_file.display().to_string(),
            "ctx": args.ctx,
            "steps": steps,
            "block": args.block,
            "refresh": args.refresh,
            "bucket": args.bucket,
            "compare_from": args.compare_from,
            "attn_layers": n_attn,
            "dense_tok_per_s": steps as f64 / dense.wall_secs,
            "arms": json_rows,
        }))?,
    )?;
    println!("# wrote {}", args.out.join("report.md").display());
    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
