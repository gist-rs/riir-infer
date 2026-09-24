//! `row_logit_floor_ppl` — riir-infer Issue 011 T2/T4: perplexity of the
//! gemma-2 f16 forward with katgpt-core's sink-exempt row-logit floor
//! (b-bit coded, exp-table softmax) on every attention row, against the
//! plain softmax on the SAME token stream.
//!
//! Each chunk is `[BOS] + seq_len` corpus tokens (the BOS is the natural
//! sink at position 0), the KV cache is reset per chunk, and the NLL of
//! every next token inside the chunk is scored. Arms run sequentially on
//! identical input; the base arm's per-token NLL and argmax are kept, so
//! every other arm reports a PAIRED delta (mean |ΔNLL|, top-1 flip rate)
//! next to its perplexity and the mean closed-form envelope its rows
//! paid.
//!
//! Gemma-2 facts that matter here: attention logits are tanh-softcapped
//! at 50 before the softmax (so a row's range is ≤ 100 nats), and there is
//! no QK-norm (riir-infer HISTORY.md, Issue 010).
//!
//! Usage:
//! ```text
//! row_logit_floor_ppl <gemma2-f16.gguf> <corpus.txt> [--tokens N] [--seq-len N]
//!                     [--tv EPS] [--n-sink N] [--arms base,b8,b6,b6s0,...]
//! ```
//! Arm grammar: `base` (plain softmax) or `b<bits>` (floor with the
//! `--n-sink` exemption) or `b<bits>s<n>` (explicit sink count; `s0` is the
//! no-exemption negative arm, Issue 011 T4).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::attention_floor::{RowLogitFloorPolicy, RowLogitFloorStats};
use riir_infer_core::transformer::{ForwardContext, forward_gemma2_f16};

/// One measured arm.
struct Arm {
    name: String,
    policy: Option<RowLogitFloorPolicy>,
}

fn parse_arm(spec: &str, n_sink: usize, tv: f32) -> Result<Arm> {
    let policy = match spec {
        "base" => None,
        s => {
            let rest = s
                .strip_prefix('b')
                .context("arm must be `base` or `b<bits>[s<n>]`")?;
            let (bits, sink) = match rest.split_once('s') {
                Some((b, n)) => (b.parse::<u8>()?, n.parse::<usize>()?),
                None => (rest.parse::<u8>()?, n_sink),
            };
            if !(2..=8).contains(&bits) {
                bail!("bits must be 2..=8, got {bits}");
            }
            Some(RowLogitFloorPolicy {
                n_sink: sink,
                bits,
                tv,
            })
        }
    };
    Ok(Arm {
        name: spec.to_string(),
        policy,
    })
}

/// `log Σ exp(logits) − logits[target]` in f64.
fn nll(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&l| (l as f64 - m).exp()).sum();
    m + z.ln() - logits[target] as f64
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| match v > bv {
            true => (i, v),
            false => (bi, bv),
        })
        .0
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: row_logit_floor_ppl <gguf> <corpus.txt> [--tokens N] [--seq-len N] \
             [--tv EPS] [--n-sink N] [--arms base,b8,b6,b6s0]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let (mut n_tokens, mut seq_len, mut tv, mut n_sink) = (2048usize, 1024usize, 1e-3f32, 4usize);
    let mut arm_specs = "base,b8,b6,b6s0".to_string();
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context("flag needs a value")?;
        match args[i].as_str() {
            "--tokens" => n_tokens = v.parse()?,
            "--seq-len" => seq_len = v.parse()?,
            "--tv" => tv = v.parse()?,
            "--n-sink" => n_sink = v.parse()?,
            "--arms" => arm_specs = v.clone(),
            other => bail!("unknown arg {other}"),
        }
        i += 2;
    }
    let arms: Vec<Arm> = arm_specs
        .split(',')
        .map(|s| parse_arm(s.trim(), n_sink, tv))
        .collect::<Result<_>>()?;
    if arms.first().is_none_or(|a| a.policy.is_some()) {
        bail!("the first arm must be `base` (the paired deltas are against it)");
    }

    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    if gguf.architecture() != Some("gemma2") {
        bail!("expected a gemma2 GGUF");
    }
    let config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    drop(gguf);
    if seq_len + 1 > config.block_size {
        bail!(
            "--seq-len {seq_len} + BOS exceeds block_size {}",
            config.block_size
        );
    }
    let bos = *tok
        .encode_with_bos("")
        .first()
        .context("tokenizer has no BOS")?;
    let text = std::fs::read_to_string(&corpus_path).context("read corpus")?;
    let all = tok.encode(&text);
    let take = all.len().min(n_tokens);
    let chunks: Vec<Vec<usize>> = all[..take]
        .chunks(seq_len)
        .map(|c| std::iter::once(bos).chain(c.iter().copied()).collect())
        .collect();
    let scored: usize = chunks.iter().map(|c| c.len() - 1).sum();
    println!(
        "# row_logit_floor_ppl: gemma-2 f16 | layers={} heads={} softcap={} | load {:.1}s",
        config.n_layer,
        config.n_head,
        config.attn_logit_softcapping,
        t0.elapsed().as_secs_f32()
    );
    println!(
        "# corpus {} → {} tokens in {} chunk(s) of ≤{} (+BOS), {} scored | tv ε={tv:e}",
        corpus_path.display(),
        take,
        chunks.len(),
        seq_len,
        scored
    );

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut base_nll: Vec<f64> = Vec::with_capacity(scored);
    let mut base_top: Vec<usize> = Vec::with_capacity(scored);
    println!(
        "| arm | ppl | Δppl | mean |ΔNLL| | top-1 flip | mean env TV | floored | mean w (nats) | tok/s |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut base_ppl = 0.0f64;
    for arm in &arms {
        ctx.logit_floor = arm.policy;
        ctx.logit_floor_stats = RowLogitFloorStats::default();
        let (mut sum_nll, mut sum_abs, mut flips, mut k) = (0.0f64, 0.0f64, 0usize, 0usize);
        let t = Instant::now();
        for chunk in &chunks {
            cache.reset();
            for pos in 0..chunk.len() - 1 {
                let logits =
                    forward_gemma2_f16(&mut ctx, &weights, &mut cache, chunk[pos], pos, &config);
                let l = nll(logits, chunk[pos + 1]);
                let top = argmax(logits);
                sum_nll += l;
                match arm.policy {
                    None => {
                        base_nll.push(l);
                        base_top.push(top);
                    }
                    Some(_) => {
                        sum_abs += (l - base_nll[k]).abs();
                        flips += usize::from(top != base_top[k]);
                    }
                }
                k += 1;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let ppl = (sum_nll / k as f64).exp();
        let st = ctx.logit_floor_stats;
        match arm.policy {
            None => {
                base_ppl = ppl;
                println!(
                    "| {} | {ppl:.4} | — | — | — | — | — | — | {:.2} |",
                    arm.name,
                    k as f64 / secs
                );
            }
            Some(_) => println!(
                "| {} | {ppl:.4} | {:+.3}% | {:.5} | {:.2}% | {:.4} | {:.3}% | {:.2} | {:.2} |",
                arm.name,
                100.0 * (ppl / base_ppl - 1.0),
                sum_abs / k as f64,
                100.0 * flips as f64 / k as f64,
                st.mean_envelope_tv(),
                100.0 * st.floored as f64 / st.ctx_keys.max(1) as f64,
                st.width_sum / st.rows.max(1) as f64,
                k as f64 / secs
            ),
        }
    }
    Ok(())
}
