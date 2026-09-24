//! `row_logit_floor_ppl` — riir-infer Issue 011 T2/T3/T4: the gemma-2 f16
//! forward with katgpt-core's sink-exempt row-logit floor (b-bit coded,
//! exp-table softmax) on every attention row, against the plain softmax on
//! the SAME token sequences.
//!
//! Two modes, one arm loop (every arm scores identical sequences; the base
//! arm's per-token NLL and argmax are kept so every other arm reports a
//! PAIRED delta next to the mean closed-form envelope its rows paid):
//!
//! - **ppl** (default, T2/T4): `[BOS] + seq_len` corpus tokens per chunk,
//!   KV cache reset per chunk, every next token scored.
//! - **needle** (`--needle N`, the T3 proxy): N passkey prompts of `--ctx`
//!   tokens — a 5-digit key stated once at depth `(i + ½)/N` inside corpus
//!   filler, then asked for. Only the answer tokens are scored (teacher
//!   forced); `seq-exact` = the fraction of prompts whose every answer
//!   token is the argmax, i.e. what greedy decoding would return.
//!
//! Gemma-2 facts that matter here: attention logits are tanh-softcapped at
//! 50 before the softmax (a row's range is ≤ 100 nats); there is no QK-norm
//! (riir-infer HISTORY.md, Issue 010); the context is 8K with a 4K sliding
//! window, so a true 64K needle is out of this fixture's reach — the T3
//! proxy runs the 64K WIDTH (`n65536`) on real rows instead.
//!
//! Usage:
//! ```text
//! row_logit_floor_ppl <gemma2-f16.gguf> <corpus.txt> [--tokens N] [--seq-len N]
//!                     [--needle N] [--ctx N] [--tv EPS] [--n-sink N]
//!                     [--arms base,b8,b6,b6s0,b6n65536,...]
//! ```
//! Arm grammar: `base` (plain softmax) or `b<bits>` followed by optional
//! `s<n>` (sink count; default `--n-sink`, `s0` = no exemption, T4) and
//! `n<ctx>` (fixed width `ln(ctx/ε)` for every row; default per-row).

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

/// One scored sequence: the forward runs over all of `tokens`, and the
/// prediction of `tokens[p + 1]` is scored for every `p >= score_from`.
struct Seq {
    tokens: Vec<usize>,
    score_from: usize,
}

/// `b<bits>[s<n>][n<ctx>]` → policy; `base` → none.
fn parse_arm(spec: &str, n_sink: usize, tv: f32) -> Result<Arm> {
    if spec == "base" {
        return Ok(Arm {
            name: spec.into(),
            policy: None,
        });
    }
    let rest = spec
        .strip_prefix('b')
        .context("arm must be `base` or `b<bits>[s<n>][n<ctx>]`")?;
    let mut policy = RowLogitFloorPolicy {
        n_sink,
        bits: 0,
        tv,
        width_ctx: None,
    };
    // `split_inclusive` on letters yields "6s", "0n", "65536": each piece's
    // trailing letter names the NEXT field.
    let mut key = 'b';
    for piece in rest.split_inclusive(|c: char| c.is_ascii_alphabetic()) {
        let (digits, next) = match piece.chars().last() {
            Some(c) if c.is_ascii_alphabetic() => (&piece[..piece.len() - 1], c),
            _ => (piece, '\0'),
        };
        match key {
            'b' => policy.bits = digits.parse()?,
            's' => policy.n_sink = digits.parse()?,
            'n' => policy.width_ctx = Some(digits.parse()?),
            k => bail!("unknown arm field `{k}` in `{spec}`"),
        }
        key = next;
    }
    if !(2..=8).contains(&policy.bits) {
        bail!("bits must be 2..=8 in `{spec}`");
    }
    Ok(Arm {
        name: spec.into(),
        policy: Some(policy),
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

/// The T3 passkey prompts: `[BOS] intro filler₁ needle filler₂ question answer`.
fn needle_seqs(
    tok: &SentencePieceGgufTokenizer,
    bos: usize,
    filler: &[usize],
    n: usize,
    ctx: usize,
) -> Result<Vec<Seq>> {
    let intro = tok.encode(
        "There is an important pass key hidden inside a lot of irrelevant text. \
         Find it and memorize it.\n\n",
    );
    let question = tok.encode("\n\nWhat is the pass key? The pass key is");
    let mut seqs = Vec::with_capacity(n);
    let mut state = 0x9E37_79B9u32;
    for i in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = 10_000 + state % 90_000;
        let needle = tok.encode(&format!(
            "\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
        ));
        let answer = tok.encode(&format!(" {key}."));
        let fixed = 1 + intro.len() + needle.len() + question.len() + answer.len();
        let fill = ctx
            .checked_sub(fixed)
            .context("--ctx too small for the prompt")?;
        if fill > filler.len() {
            bail!("corpus too short for --ctx {ctx}");
        }
        let before = fill * (2 * i + 1) / (2 * n);
        let off = (i * 997) % (filler.len() - fill + 1);
        let body = &filler[off..off + fill];
        let mut t = Vec::with_capacity(ctx);
        t.push(bos);
        t.extend_from_slice(&intro);
        t.extend_from_slice(&body[..before]);
        t.extend_from_slice(&needle);
        t.extend_from_slice(&body[before..]);
        t.extend_from_slice(&question);
        let score_from = t.len() - 1;
        t.extend_from_slice(&answer);
        seqs.push(Seq {
            tokens: t,
            score_from,
        });
    }
    Ok(seqs)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: row_logit_floor_ppl <gguf> <corpus.txt> [--tokens N] [--seq-len N] \
             [--needle N] [--ctx N] [--tv EPS] [--n-sink N] [--arms base,b8,b6,b6s0]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let (mut n_tokens, mut seq_len, mut tv, mut n_sink) = (2048usize, 1024usize, 1e-3f32, 4usize);
    let (mut needle, mut needle_ctx) = (0usize, 1536usize);
    let mut arm_specs = "base,b8,b6,b6s0".to_string();
    let mut i = 3;
    while i < args.len() {
        let v = args.get(i + 1).context("flag needs a value")?;
        match args[i].as_str() {
            "--tokens" => n_tokens = v.parse()?,
            "--seq-len" => seq_len = v.parse()?,
            "--needle" => needle = v.parse()?,
            "--ctx" => needle_ctx = v.parse()?,
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
    let bos = *tok
        .encode_with_bos("")
        .first()
        .context("tokenizer has no BOS")?;
    let text = std::fs::read_to_string(&corpus_path).context("read corpus")?;
    let all = tok.encode(&text);
    let seqs: Vec<Seq> = match needle {
        0 => all[..all.len().min(n_tokens)]
            .chunks(seq_len)
            .map(|c| Seq {
                tokens: std::iter::once(bos).chain(c.iter().copied()).collect(),
                score_from: 0,
            })
            .collect(),
        n => needle_seqs(&tok, bos, &all, n, needle_ctx)?,
    };
    let longest = seqs.iter().map(|s| s.tokens.len()).max().unwrap_or(0);
    if longest > config.block_size {
        bail!(
            "a sequence of {longest} tokens exceeds block_size {}",
            config.block_size
        );
    }
    let scored: usize = seqs.iter().map(|s| s.tokens.len() - 1 - s.score_from).sum();
    println!(
        "# row_logit_floor_ppl: gemma-2 f16 | layers={} heads={} softcap={} | load {:.1}s",
        config.n_layer,
        config.n_head,
        config.attn_logit_softcapping,
        t0.elapsed().as_secs_f32()
    );
    println!(
        "# mode {} | {} sequence(s), longest {} tokens, {} scored | corpus {} | tv ε={tv:e}",
        match needle {
            0 => "ppl",
            _ => "needle",
        },
        seqs.len(),
        longest,
        scored,
        corpus_path.display()
    );

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut base_nll: Vec<f64> = Vec::with_capacity(scored);
    let mut base_top: Vec<usize> = Vec::with_capacity(scored);
    println!(
        "| arm | ppl | Δppl | mean |ΔNLL| | top-1 flip | seq-exact | mean env TV | floored | mean w (nats) | tok/s |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut base_ppl = 0.0f64;
    for arm in &arms {
        ctx.logit_floor = arm.policy;
        ctx.logit_floor_stats = RowLogitFloorStats::default();
        let (mut sum_nll, mut sum_abs, mut flips, mut k) = (0.0f64, 0.0f64, 0usize, 0usize);
        let (mut exact, mut fwd) = (0usize, 0usize);
        let t = Instant::now();
        for seq in &seqs {
            cache.reset();
            let mut all_right = true;
            for pos in 0..seq.tokens.len() - 1 {
                let logits = forward_gemma2_f16(
                    &mut ctx,
                    &weights,
                    &mut cache,
                    seq.tokens[pos],
                    pos,
                    &config,
                );
                fwd += 1;
                if pos < seq.score_from {
                    continue;
                }
                let target = seq.tokens[pos + 1];
                let l = nll(logits, target);
                let top = argmax(logits);
                sum_nll += l;
                all_right &= top == target;
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
            exact += usize::from(all_right);
        }
        let secs = t.elapsed().as_secs_f64();
        let ppl = (sum_nll / k as f64).exp();
        let seq_exact = 100.0 * exact as f64 / seqs.len() as f64;
        let st = ctx.logit_floor_stats;
        match arm.policy {
            None => {
                base_ppl = ppl;
                println!(
                    "| {} | {ppl:.4} | — | — | — | {seq_exact:.1}% | — | — | — | {:.2} |",
                    arm.name,
                    fwd as f64 / secs
                );
            }
            Some(_) => println!(
                "| {} | {ppl:.4} | {:+.3}% | {:.5} | {:.2}% | {seq_exact:.1}% | {:.4} | {:.3}% | {:.2} | {:.2} |",
                arm.name,
                100.0 * (ppl / base_ppl - 1.0),
                sum_abs / k as f64,
                100.0 * flips as f64 / k as f64,
                st.mean_envelope_tv(),
                100.0 * st.floored as f64 / st.ctx_keys.max(1) as f64,
                st.width_sum / st.rows.max(1) as f64,
                fwd as f64 / secs
            ),
        }
    }
    Ok(())
}
