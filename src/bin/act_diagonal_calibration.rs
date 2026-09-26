//! act_diagonal_calibration — riir-infer Issue 014 T1: the activation-diagonal
//! collector over the Bonsai-2 ternary lane (the `vk_calibration` shape at the
//! Issue-886 tap points).
//!
//! One pass over a natural-text corpus through the frozen checkpoint
//! (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, qwen35 hybrid DeltaNet/attention,
//! Hadamard-folded) observing the INPUT of every ternary projection beside
//! each matvec, feeding `katgpt_core::act_channel_moments::ActChannelMoments`.
//! Output: the BLAKE3-committed `ActChannelDiagonal` artifact (the T2 refit
//! input) + the flatness dashboard the issue asks for — per-tap `max/median`
//! of `E[x²]` and the top-1% channel share. A near-uniform diagonal predicts
//! T2–T3 NULL (the rotation already flattened what the ternary matvec sees),
//! and saying so up front is part of the result.
//!
//! ## Tap points (the forward's own hook seam, no forward edits)
//!
//! The collector rides `TernaryMatvecHook`: `bitlinear` passes the hook the
//! exact input slice the ternary matvec consumes, so on a folded model the
//! observation is the **post-rotation** input (the issue's trap 1) for free.
//! The hook runs the same `simd_ternary_group_matvec_parallel` the unhooked
//! path runs, then observes — the forward stays bit-identical (same kernel,
//! same order; the observation is side-band).
//!
//! Per token (deterministic call order, asserted every token):
//! - DeltaNet layer: `in_proj_qkv` → *attn_in*, `in_proj_z` → *attn_in*,
//!   `out_proj` → *layer_out*; then `gate_proj`/`up_proj` → *ffn_in*,
//!   `down_proj` → *swiglu*.
//! - Attention layer: `attn_wq`/`attn_wk`/`attn_wv` → *attn_in*,
//!   `attn_wo` → *layer_out*; then the same FFN trio.
//! - Final: `lm_head` → *final_in*.
//!
//! `attn_in` and `ffn_in` are DIFFERENT tensors (the FFN input is the
//! re-normed post-attention residual), hence two taps. Duplicated inputs
//! (qkv = z, wq = wk = wv, gate = up) observe into the SAME tap — identical
//! vectors, so the sums stay exact and the table keeps one entry per distinct
//! input (the T2 mapping). `in_proj_a`/`in_proj_b` are the Issue-980 DENSE
//! (BF16) escape set on Bonsai-2 — not ternary, never quantized, so their
//! input diagonal is out of scope by construction (and they carry no hook
//! seam: `matvec_into` bypasses `bitlinear`).
//!
//! ## MEASUREMENT-ONLY (the issue's P0 law, same as `vk_calibration`)
//!
//! No quality claim is made. The artifact binds only itself (BLAKE3 over the
//! canonical image); binding it to the checkpoint is the consumer's job when
//! T2 loads it (record the GGUF name + size beside the digest, as this run's
//! header does).
//!
//! Usage:
//! ```text
//! act_diagonal_calibration <gguf> <corpus-dir-or-txt>
//!     [--max-tokens N] [--seq-len N] [--report PATH] [--artifact PATH]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use katgpt_core::act_channel_moments::ActChannelMoments;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::act_taps::{ActTapHook, ActTapPlan};
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, effective_rotary_dim,
};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::DeltaNetLayerType;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: act_diagonal_calibration <model.gguf> <corpus-dir-or-txt> \
             [--max-tokens N] [--seq-len N] [--report PATH] [--artifact PATH]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let mut max_tokens: usize = 8_192;
    let mut seq_len: usize = 1_024;
    let mut report_path: Option<PathBuf> = None;
    let mut artifact_path: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
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
            "--artifact" => {
                artifact_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if seq_len == 0 {
        bail!("--seq-len must be >= 1");
    }

    // ── Load model. The loader refuses a folded file without bonsai2_hadamard
    //    (which this feature implies) — loud by construction. ──
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    let rotated = weights.rotation.is_some();
    let n_gdn = weights
        .layer_types
        .iter()
        .filter(|&&t| t == DeltaNetLayerType::DeltaNet)
        .count();
    let gguf_meta = std::fs::metadata(&gguf_path)?;
    println!(
        "# act_diagonal_calibration: qwen35 hybrid | layers={} (gdn {n_gdn}, attn {}) \
         n_embd={} mlp_hidden={} vocab={} | hadamard={rotated} | gguf {} bytes | load {:.1}s",
        config.n_layer,
        config.n_layer - n_gdn,
        config.n_embd,
        config.mlp_hidden,
        config.vocab_size,
        gguf_meta.len(),
        t0.elapsed().as_secs_f32(),
    );

    // Tokenizer from the same GGUF (a second cheap mmap open — the loader
    // consumed its own handle).
    let tok = {
        let gguf = GgufFile::open(&gguf_path).context("re-open gguf for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };

    // ── Corpus → one token stream ──
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

    // Cap the KV cache / score buffers at what the run needs (the
    // row_logit_floor_ppl law — a 262K-context model would otherwise allocate
    // its whole advertised window per attention layer).
    config.block_size = seq_len;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);

    // ── The tap plan (dimensions read off the weights, never computed) ──
    let plan = ActTapPlan::build(&config, &weights);
    println!(
        "# taps: {} distinct inputs, {} matvec calls/token | accumulator {:.2} MiB",
        plan.moment_widths.len(),
        plan.steps.len(),
        plan.moment_bytes() as f64 / (1 << 20) as f64,
    );

    let mut moments = ActChannelMoments::new(&plan.moment_widths);
    let hook = ActTapHook::new(&plan, &mut moments);

    // ── The calibration pass (chunked causal forward, hook-observed) ──
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
    let t2 = Instant::now();
    let mut done = 0usize;
    let mut next_report = 0usize;
    for chunk in tokens.chunks(seq_len) {
        cache.reset();
        for (pos, &token) in chunk.iter().enumerate() {
            hook.begin_token();
            forward_qwen_deltanet_ternary_with_hook(
                &mut x,
                &weights,
                &mut cache,
                token,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
                None,
                Some(&hook),
                None,
                None,
            );
            hook.end_token();
        }
        done += chunk.len();
        if done >= next_report {
            let el = t2.elapsed().as_secs_f32();
            let rate = done as f32 / el.max(1e-6);
            println!(
                "# progress {done}/{} tokens | {:.0} tok/s | eta {:.0} min",
                tokens.len(),
                rate,
                (tokens.len() - done) as f32 / (rate + 1e-6) / 60.0,
            );
            next_report = (done + tokens.len() / 10).max(done + 1);
        }
    }
    let pass_s = t2.elapsed().as_secs_f32();
    println!(
        "# pass done: {done} tokens in {pass_s:.0}s ({:.0} tok/s) — box: 4090 \
         workstation i7-13700K, CPU lane, AC power",
        done as f32 / pass_s.max(1e-6),
    );

    // ── Freeze + artifact ──
    let diagonal = moments.freeze();
    let commitment = diagonal.commitment();
    let digest_hex = hex(&commitment);
    let channels: usize = plan.moment_widths.iter().sum();
    println!(
        "# diagonal: {} taps, {channels} channels, BLAKE3 {digest_hex}",
        plan.moment_widths.len(),
    );
    assert!(
        diagonal.verify(),
        "frozen diagonal failed its own digest check"
    );
    if let Some(p) = artifact_path {
        let bytes = diagonal.to_bytes();
        std::fs::write(&p, &bytes).with_context(|| format!("write artifact {}", p.display()))?;
        println!("# artifact written: {} ({} bytes)", p.display(), bytes.len());
    }

    // ── The flatness dashboard (the T1 readout) ──
    let mut out = String::new();
    out.push_str(&format!(
        "# Issue 014 T1 — activation-diagonal flatness: qwen35 hybrid, \
         hadamard={rotated}, slice {done} tokens (seq_len {seq_len})\n\n"
    ));
    out.push_str(&format!("digest: {digest_hex}\n\n"));
    out.push_str("| tap | width | n_obs | med E[x²] | max/med | top-1% share |\n|---|---|---|---|---|---|\n");
    // Per-tap rows, aggregated per kind across layers.
    let mut agg: Vec<(&'static str, Vec<f64>, Vec<f64>)> = Vec::new();
    for (t, tap) in plan.taps.iter().enumerate() {
        let n = diagonal.count(t);
        if n == 0 {
            out.push_str(&format!("| {} | {} | 0 | — | — | — |\n", tap.label, tap.width));
            continue;
        }
        let sq = diagonal.mean_sq(t);
        let mut sorted: Vec<f64> = sq.iter().map(|&v| f64::from(v)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = percentile(&sorted, 0.5);
        let max = sorted[sorted.len() - 1];
        let total: f64 = sorted.iter().sum();
        let k = (sorted.len() / 100).max(1); // top 1%
        let top1: f64 = sorted[sorted.len() - k..].iter().sum();
        let ratio = if med > 0.0 { max / med } else { f64::INFINITY };
        out.push_str(&format!(
            "| {} | {} | {n} | {med:.3e} | {ratio:.1} | {:.3} |\n",
            tap.label,
            tap.width,
            if total > 0.0 { top1 / total } else { f64::NAN },
        ));
        if let Some(slot) = agg.iter_mut().find(|(kind, _, _)| *kind == tap.kind) {
            slot.1.push(med);
            slot.2.push(ratio);
        } else {
            agg.push((tap.kind, vec![med], vec![ratio]));
        }
    }
    out.push_str("\n## aggregate per tap kind (across layers)\n\n");
    out.push_str("| kind | taps | median-of-med | median max/med | max max/med |\n|---|---|---|---|---|\n");
    for (kind, meds, ratios) in &agg {
        let mut r = ratios.clone();
        r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut m = meds.clone();
        m.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.push_str(&format!(
            "| {kind} | {} | {:.3e} | {:.1} | {:.1} |\n",
            meds.len(),
            percentile(&m, 0.5),
            percentile(&r, 0.5),
            r[r.len() - 1],
        ));
    }
    out.push_str(
        "\nMEASUREMENT-ONLY (Issue 014 P0 law): no quality claim. A max/med near 1 \
         with a small top-1% share reads \"rotation flattened the diagonal\" → T2–T3 \
         predicted null (close the issue honestly). Heavy tails read the opposite — \
         proceed to the T2 scale refit.\n",
    );
    print!("{out}");
    if let Some(p) = report_path {
        std::fs::write(&p, &out).with_context(|| format!("write report {}", p.display()))?;
        println!("# report written: {}", p.display());
    }
    Ok(())
}

// ── Small helpers ─────────────────────────────────────────────────────────────

/// Median-index law with the clamp (the repo's percentile trap): floor(n·p)
/// clamped to n-1 — a rank, never a smooth percentile; the tail support is
/// n − idx and is disclosed by printing medians of medians, never p99s.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (((sorted.len() as f64) * p).floor() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}
