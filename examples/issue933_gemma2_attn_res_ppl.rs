//! `issue933_gemma2_attn_res_ppl` — katgpt-rs Issue 933 T4: the fixed-query
//! modelless-AttnRes ppl A/B on gemma-2-2b-it-f16 (the E5 probe, Research
//! 619 MSAR — "deterministic exponential-depth profile replacing learned
//! `score_weight` for checkpoints WITHOUT AttnRes weights").
//!
//! Paired frozen-read: every arm scores the IDENTICAL `[BOS] + seq_len`
//! corpus chunks (KV reset per chunk — the row_logit_floor_ppl law), the
//! base arm running the stock `forward_gemma2_f16` and each λ arm running
//! the AttnRes fork at `--block-size` layers per block. Per-token NLL pairs
//! → mean ΔNLL, Δppl%, worse-token share.
//!
//! Pre-registered bars (Issue 933 T4):
//! - λ=∞ must be bit-identical to stock (gates: the module test; the bin
//!   re-checks ΔNLL == 0 exactly as a wiring sanity).
//! - GREEN  = some finite λ with real mixing keeps Δppl ≤ 1.0% → the
//!   modelless posture is viable (opt-in posture candidate).
//! - RED    = only the disarmed λ stays within tolerance → record and close
//!   (the learned queries ARE the mechanism for real checkpoints).
//!
//! Usage:
//! ```text
//! issue933_gemma2_attn_res_ppl <gemma2-f16.gguf> <corpus> [--tokens N]
//!                              [--seq-len N] [--block-size N]
//!                              [--lambdas inf,4,2,1,0.5,0]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::gemma2_attn_res::{
    AttnResDepthProbe, GemmaAttnResState, forward_gemma2_f16_attn_res,
};
use riir_infer_core::transformer::{ForwardContext, forward_gemma2_f16};

struct Arm {
    label: String,
    lambda: Option<f32>, // None = base (stock forward)
    nll_sum: f64,
    scored: usize,
    worse: usize,   // NLL above base for this token
    delta_sum: f64, // Σ (nll_arm − nll_base), f64 order-exact bookkeeping
    delta_abs_max: f64,
    wall: std::time::Duration,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: issue933_gemma2_attn_res_ppl <gemma2-f16.gguf> <corpus> \
             [--tokens N] [--seq-len N] [--block-size N] [--lambdas inf,4,2,1,0.5,0]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let (mut n_tokens, mut seq_len, mut block_size) = (8192usize, 1024usize, 4usize);
    let mut lambdas = "inf,4,2,1,0.5,0".to_string();
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context("flag needs a value")?;
        match args[i].as_str() {
            "--tokens" => n_tokens = v.parse()?,
            "--seq-len" => seq_len = v.parse()?,
            "--block-size" => block_size = v.parse()?,
            "--lambdas" => lambdas = v.clone(),
            other => bail!("unknown arg {other}"),
        }
        i += 2;
    }
    if !(2..=4096).contains(&seq_len) {
        bail!("--seq-len must be in [2, 4096] (below the gemma-2 sliding window)");
    }
    if block_size == 0 {
        bail!("--block-size must be ≥ 1");
    }
    let lambda_list: Vec<Option<f32>> = lambdas
        .split(',')
        .map(|s| -> Result<Option<f32>> {
            let v = match s.trim() {
                "inf" | "INF" | "∞" => f32::INFINITY,
                other => other
                    .parse::<f32>()
                    .with_context(|| format!("bad λ '{other}'"))?,
            };
            Ok(Some(v))
        })
        .collect::<Result<_>>()?;

    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let mut config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    drop(gguf);
    let bos = tok.bos_id();

    let corpus_text = load_corpus_text(&corpus_path)
        .with_context(|| format!("read corpus {}", corpus_path.display()))?;
    let corpus_blake3 = blake3::Hasher::new()
        .update(corpus_text.as_bytes())
        .finalize()
        .to_hex()
        .to_string();
    let all = tok.encode(&corpus_text);
    let take = all.len().min(n_tokens);
    let tokens = &all[..take];
    let chunks: Vec<&[usize]> = tokens.chunks(seq_len).collect();
    let longest = chunks.iter().map(|c| c.len() + 1).max().unwrap_or(0);
    // Cap the KV cache at what the run needs (the row_logit_floor_ppl law).
    config.block_size = longest;
    let scored_total: usize = chunks.iter().map(|c| c.len()).sum(); // every next-token incl. after BOS

    println!(
        "# issue933_attn_res_ppl: gemma-2 f16 | layers={} n_embd={} vocab={} | block S={block_size} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    println!(
        "# corpus {} | blake3 {corpus_blake3} | tokens {}/{} | {} chunk(s), longest {longest} | scored {scored_total}",
        corpus_path.display(),
        take,
        all.len(),
        chunks.len()
    );

    // Arms: base + λ list. Each arm gets its own ctx + KV cache + state
    // (reused across chunks — re-running a chunk from pos 0 overwrites
    // exactly its rows, the stock reuse contract).
    let mut arms: Vec<Arm> = Vec::with_capacity(lambda_list.len() + 1);
    arms.push(Arm {
        label: "base".into(),
        lambda: None,
        nll_sum: 0.0,
        scored: 0,
        worse: 0,
        delta_sum: 0.0,
        delta_abs_max: 0.0,
        wall: Default::default(),
    });
    for l in &lambda_list {
        let label = match l {
            Some(v) if v.is_infinite() => "inf".to_string(),
            Some(v) => format!("{v}"),
            None => unreachable!(),
        };
        arms.push(Arm {
            label,
            lambda: *l,
            nll_sum: 0.0,
            scored: 0,
            worse: 0,
            delta_sum: 0.0,
            delta_abs_max: 0.0,
            wall: Default::default(),
        });
    }

    // Per chunk: score base first (keeping per-token NLLs), then each arm
    // against the same positions.
    let mut base_nlls: Vec<f64> = Vec::with_capacity(longest);
    for chunk in &chunks {
        let seq: Vec<usize> = std::iter::once(bos).chain(chunk.iter().copied()).collect();
        // base
        {
            let arm = &mut arms[0];
            let t = Instant::now();
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            base_nlls.clear();
            for pos in 0..seq.len() - 1 {
                let logits =
                    forward_gemma2_f16(&mut ctx, &weights, &mut cache, seq[pos], pos, &config);
                let nll = next_token_nll(&logits[..config.vocab_size], seq[pos + 1]);
                base_nlls.push(nll);
                arm.nll_sum += nll;
                arm.scored += 1;
            }
            arm.wall += t.elapsed();
        }
        // λ arms
        for arm in arms.iter_mut().skip(1) {
            let lambda = arm.lambda.expect("λ arm");
            let t = Instant::now();
            let probe = AttnResDepthProbe { block_size, lambda };
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut state = GemmaAttnResState::new(config.n_layer, config.n_embd, block_size);
            for pos in 0..seq.len() - 1 {
                let logits = forward_gemma2_f16_attn_res(
                    &mut ctx, &weights, &mut cache, seq[pos], pos, &config, &probe, &mut state,
                );
                let nll = next_token_nll(&logits[..config.vocab_size], seq[pos + 1]);
                let base = base_nlls[pos];
                let d = nll - base;
                arm.delta_sum += d;
                if d > 0.0 {
                    arm.worse += 1;
                }
                arm.delta_abs_max = arm.delta_abs_max.max(d.abs());
                arm.nll_sum += nll;
                arm.scored += 1;
            }
            arm.wall += t.elapsed();
        }
    }

    // Report.
    let base_nll_mean = arms[0].nll_sum / arms[0].scored as f64;
    let base_ppl = base_nll_mean.exp();
    println!();
    println!("| arm | mean-NLL | ppl | Δppl% | mean-ΔNLL | worse% | max|ΔNLL| | tok/s |");
    println!("|---|---|---|---|---|---|---|---|");
    let tok_s = |arm: &Arm| {
        let secs = arm.wall.as_secs_f64();
        if secs > 0.0 {
            arm.scored as f64 / secs
        } else {
            0.0
        }
    };
    println!(
        "| base | {:.5} | {:.4} | — | — | — | — | {:.0} |",
        base_nll_mean,
        base_ppl,
        tok_s(&arms[0])
    );
    for arm in arms.iter().skip(1) {
        let nll_mean = arm.nll_sum / arm.scored as f64;
        let ppl = nll_mean.exp();
        let dppl = (ppl / base_ppl - 1.0) * 100.0;
        let dnll = arm.delta_sum / arm.scored as f64;
        let worse = 100.0 * arm.worse as f64 / arm.scored as f64;
        println!(
            "| λ={} | {:.5} | {:.4} | {:+.2}% | {:+.5} | {:.1} | {:.4} | {:.0} |",
            arm.label,
            nll_mean,
            ppl,
            dppl,
            dnll,
            worse,
            arm.delta_abs_max,
            tok_s(arm)
        );
    }
    println!();
    // Verdict lines (the pre-registered bars).
    for arm in arms.iter().skip(1) {
        let nll_mean = arm.nll_sum / arm.scored as f64;
        let ppl = nll_mean.exp();
        let dppl = (ppl / base_ppl - 1.0) * 100.0;
        let is_disarmed = arm.label == "inf";
        let within = dppl.abs() <= 1.0;
        let verdict = if is_disarmed {
            if dppl.abs() > 1e-9 {
                "WIRED-DIFF (λ=∞ must be EXACT — instrument defect)".to_string()
            } else {
                "disarmed-identity OK".to_string()
            }
        } else if within {
            "GREEN-candidate (Δppl ≤ 1%)".to_string()
        } else {
            "RED (outside tolerance)".to_string()
        };
        println!("VERDICT λ={}: Δppl {dppl:+.3}% — {verdict}", arm.label);
    }
    Ok(())
}

/// Teacher-forced next-token NLL via log-softmax (f64 accumulation over the
/// max-shifted exp-sum).
fn next_token_nll(logits: &[f32], next: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) as f64).exp();
    }
    let lse = max as f64 + sum.ln();
    lse - logits[next] as f64
}
