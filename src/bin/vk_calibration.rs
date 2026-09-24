//! vk_calibration — Issue 883 P0 model-side harness (katgpt-rs
//! `.issues/883_fitted_value_anchor_tables.md`, Research 587): the
//! offline R² dashboard for fitted token-value tables on gemma-2-2b-it.
//!
//! One pass over a natural-text corpus through the frozen checkpoint
//! with V/K taps at every layer (pre-RoPE K, post-W_V V — the tap-point
//! law) feeding `katgpt_core::fitted_anchor_table::StreamingMeanTable`
//! per (layer, signal ∈ {V, K, V−K}). Output: the go/no-go dashboard —
//! per-layer ρ_l(V), ρ_l(K), ρ_l(V−K) (variance-weighted token-explained
//! fractions), Zipf coverage, per-token n_s histogram head — the
//! offline-computable predictor for P1 (mean-removed V quant), P2
//! (fitted K=V+ retrofit), P3 (V-cache halving).
//!
//! **MEASUREMENT-ONLY (the issue's P0 law): no quality claim is made.**
//!
//! ## Fixture caveats (recorded beside every number this bin prints)
//!
//! 1. **This GGUF is a 288-tensor conversion — NO `attn_q_norm`/
//!    `attn_k_norm` tensors** (llama.cpp's standard gemma-2 conversion
//!    carries 340; upstream Gemma-2 has QK-norm). The workspace's gemma-2
//!    stack (loader + forward) is self-consistent on this artifact, and
//!    V never passes through QK-norm even upstream, but K magnitudes
//!    differ from upstream gemma-2 — ρ_l(K) and ρ_l(V−K) describe THIS
//!    artifact. Trap 1 re-arms for any standard 340-tensor conversion.
//! 2. First-slice corpus size is honest (`--max-tokens`); the issue's
//!    full spec is 10⁸–10⁹ tokens — the first slice is a reduced-scale
//!    read with measured coverage, never presented as the full pass.
//! 3. Sequences chunk at 1024 « gemma-2's 4096 sliding window, so SWA
//!    semantics are not exercised.
//!
//! Usage:
//! ```text
//! vk_calibration <gguf> <corpus-dir-or-txt> [--top-k N] [--max-tokens N]
//!                [--seq-len N] [--report PATH]
//! ```
//! Default corpus: the sibling riir-train `chat_probe` HF-pages dir
//! (natural chat text). Box state is printed beside every figure (the
//! G2 law): token throughput, wall time, table memory.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use katgpt_core::fitted_anchor_table::StreamingMeanTable;
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::gguf_loader::{config_from_gguf_metadata, GgufFile};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::ForwardContext;
use riir_infer_core::transformer::gemma2_calibration::{
    forward_gemma2_f16_tapped, load_gemma2_f16_direct, CalibrationTables,
};
use riir_infer_core::types::kv_dim;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: vk_calibration <model.gguf> <corpus-dir-or-txt> [--top-k N] \
             [--max-tokens N] [--seq-len N] [--report PATH]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let mut top_k: usize = 8192;
    let mut max_tokens: usize = 200_000;
    let mut seq_len: usize = 1024;
    let mut report_path: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--top-k" => {
                top_k = args[i + 1].parse().context("--top-k N")?;
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().context("--max-tokens N")?;
                i += 2;
            }
            "--seq-len" => {
                seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if seq_len > 4096 {
        bail!("--seq-len must stay <= 4096 (gemma-2 sliding window; see the module doc)");
    }

    // ── Load model + tokenizer from ONE open GGUF (mmap-backed reads) ──
    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    println!(
        "# vk_calibration: gemma-2-2b-it f16 | layers={} n_embd={} kv_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        kv_dim(&config),
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    println!(
        "# fixture caveat 1: 288-tensor conversion, NO q/k-norm tensors \
         (llama.cpp standard = 340) — rho(K)/rho(V-K) describe THIS artifact"
    );
    drop(gguf);

    // ── Corpus: natural text → one token stream (+ frequency counts) ──
    let t1 = Instant::now();
    let text = load_corpus_text(&corpus_path)?;
    let all_tokens = tok.encode(&text);
    println!(
        "# corpus: {} chars → {} tokens ({:.1}s) from {}",
        text.len(),
        all_tokens.len(),
        t1.elapsed().as_secs_f32(),
        corpus_path.display()
    );
    let take = all_tokens.len().min(max_tokens);
    let tokens: Vec<usize> = all_tokens[..take].to_vec();

    let mut counts = vec![0u64; config.vocab_size];
    for &t in &tokens {
        counts[t] += 1;
    }

    // ── Tables (memory disclosed — the box-state law) ──
    let kvd = kv_dim(&config);
    let mut tables = CalibrationTables::from_counts(config.n_layer, kvd, counts, top_k);
    let table_bytes = config.n_layer * 3 * (tables.top_k + 1) * kvd * 4;
    println!(
        "# tables: top_k={} of {} seen | {:.2} GiB",
        tables.top_k,
        tables.token_counts.iter().filter(|&&c| c > 0).count(),
        table_bytes as f64 / (1 << 30) as f64
    );

    // Tracked-mass preview from the frequency pre-pass (the coverage the
    // run WILL have — computed before spending hours on the forward).
    let total_n: u64 = tables.token_counts.iter().sum();
    let tracked_preview: u64 = {
        let mut c: Vec<u64> = tables.token_counts.clone();
        c.sort_unstable_by(|a, b| b.cmp(a));
        c.iter().take(tables.top_k).sum()
    };
    println!(
        "# coverage preview: top-{} tokens hold {:.1}% of the {}-token slice",
        tables.top_k,
        100.0 * tracked_preview as f64 / total_n as f64,
        total_n
    );

    // ── The calibration pass (chunked causal forward with taps) ──────
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let t2 = Instant::now();
    let mut done = 0usize;
    let mut next_report = 0usize;
    for chunk in tokens.chunks(seq_len) {
        cache.reset();
        for (pos, &token) in chunk.iter().enumerate() {
            forward_gemma2_f16_tapped(&mut ctx, &weights, &mut cache, &mut tables, token, pos, &config);
        }
        done += chunk.len();
        if done >= next_report {
            let el = t2.elapsed().as_secs_f32();
            let rate = done as f32 / el.max(1e-6);
            println!(
                "# progress {}/{} tokens | {:.0} tok/s | eta {:.0} min",
                done,
                tokens.len(),
                rate,
                (tokens.len() - done) as f32 / (rate + 1e-6) / 60.0
            );
            next_report = (done + tokens.len() / 20).max(done + 1);
        }
    }
    let pass_s = t2.elapsed().as_secs_f32();
    println!(
        "# pass done: {} tokens in {:.0}s ({:.0} tok/s) — box: 4090 workstation i7-13700K, CPU lane",
        tokens.len(),
        pass_s,
        tokens.len() as f32 / pass_s.max(1e-6)
    );

    // ── The dashboard ─────────────────────────────────────────────────
    let mut out = String::new();
    out.push_str("# Issue 883 P0 — R² dashboard: gemma-2-2b-it-f16.gguf (288-tensor conversion, no q/k-norm — caveat 1)\n\n");
    out.push_str(&format!(
        "slice: {} tokens | top_k={} | seq_len={} | tap: pre-RoPE K, post-W_V V\n\n",
        tokens.len(),
        tables.top_k,
        seq_len
    ));
    out.push_str("| layer | ρ_l(V) | ρ_l(K) | ρ_l(V−K) | V mass | K mass | V−K mass |\n");
    out.push_str("|---|---|---|---|---|---|---|\n");
    let mut r_v_all = Vec::new();
    let mut r_k_all = Vec::new();
    let mut r_vk_all = Vec::new();
    for lt in tables.layers.iter() {
        let r_v = lt.v.r_squared();
        let r_k = lt.k.r_squared();
        let r_vk = lt.vk.r_squared();
        r_v_all.push(r_v);
        r_k_all.push(r_k);
        r_vk_all.push(r_vk);
    }
    for (l, ((r_v, r_k), r_vk)) in r_v_all
        .iter()
        .zip(r_k_all.iter())
        .zip(r_vk_all.iter())
        .enumerate()
    {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.3} | {:.3} | {:.3} |\n",
            l, r_v.aggregate, r_k.aggregate, r_vk.aggregate, r_v.tracked_mass, r_k.tracked_mass,
            r_vk.tracked_mass
        ));
    }
    // Head-level ρ for layer 0 + a mid layer (per-head slices — P1's quantity).
    out.push_str("\n## per-head ρ_l(V) (layer 0 and 13)\n\n");
    for l in [0, config.n_layer / 2] {
        let hd = config.head_dim;
        let row: Vec<String> = (0..config.n_kv_head)
            .map(|h| format!("{:.4}", r_v_all[l].aggregate_over(h * hd, (h + 1) * hd)))
            .collect();
        out.push_str(&format!("| layer {l} | {} |\n", row.join(" | ")));
    }
    // Zipf coverage curve at checkpoints (the storage dial, P4's law).
    let cov = coverage_of(&tables.layers[0].v);
    out.push_str("\n## coverage(K) — layer-0 V table (storage dial P(K)=b_w·L·K·d_v)\n\n");
    for &k in &[16usize, 64, 256, 1024, 4096, tables.top_k.min(8192)] {
        if k <= cov.len() && k > 0 {
            out.push_str(&format!("- K={k}: {:.4}\n", cov[k - 1]));
        }
    }
    // Top-token n_s head (the histogram).
    let sorted = tables.layers[0].v.sorted_counts_desc();
    out.push_str("\n## top-10 n_s (layer 0)\n\n");
    out.push_str(&format!("- {:?}\n", &sorted[..sorted.len().min(10)]));

    // Aggregate verdict block.
    let mean_rho = |rs: &[katgpt_core::fitted_anchor_table::R2Report]| {
        rs.iter().map(|r| r.aggregate as f64).sum::<f64>() / rs.len() as f64
    };
    out.push_str(&format!(
        "\nmean over layers: ρ(V)={:.4} ρ(K)={:.4} ρ(V−K)={:.4}\n",
        mean_rho(&r_v_all),
        mean_rho(&r_k_all),
        mean_rho(&r_vk_all)
    ));
    out.push_str("\nMEASUREMENT-ONLY (P0 law): no quality claim. rho_l ≈ 0 on this GQA fixture would be a legitimate recorded negative (trap 4 exempts only fixture-class nulls — this IS the mechanism-bearing fixture).\n");

    print!("{out}");
    if let Some(p) = report_path {
        std::fs::write(&p, &out).with_context(|| format!("write report {}", p.display()))?;
        eprintln!("# report written: {}", p.display());
    }
    Ok(())
}

/// Layer-0 V-table coverage (tracked keys' cumulative share of N).
fn coverage_of(t: &StreamingMeanTable) -> Vec<f64> {
    t.coverage_curve()
}

/// Load the corpus text: a `.txt`/`.md` file directly, or a directory of
/// HF datasets-server `page_*.json` files (the sibling riir-train
/// `chat_probe` shape: rows[].row.messages[].content + prompt).
fn load_corpus_text(path: &std::path::Path) -> Result<String> {
    if path.is_file() {
        return Ok(std::fs::read_to_string(path)?);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("page_") && n.ends_with(".json"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no page_*.json under {}", path.display());
    }
    let mut text = String::new();
    for f in &files {
        let raw = std::fs::read_to_string(f)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        if let Some(rows) = v.get("rows").and_then(|r| r.as_array()) {
            for row in rows {
                if let Some(msgs) = row.pointer("/row/messages").and_then(|m| m.as_array()) {
                    for m in msgs {
                        if let Some(c) = m.get("content").and_then(|c| c.as_str()) {
                            text.push_str(c);
                            text.push('\n');
                        }
                    }
                }
                if let Some(p) = row.pointer("/row/prompt").and_then(|p| p.as_str()) {
                    text.push_str(p);
                    text.push('\n');
                }
            }
        }
    }
    if text.is_empty() {
        bail!("corpus text empty from {}", path.display());
    }
    Ok(text)
}
