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
//! Two fixtures, picked by the GGUF's `general.architecture`:
//!
//! - **gemma2** (f16 weights): attention logits are tanh-softcapped at 50
//!   before the softmax (a row's range is ≤ 100 nats); there is no QK-norm
//!   (riir-infer HISTORY.md, Issue 010); the context is 8K with a 4K sliding
//!   window, so a true 64K needle is out of its reach — the T3 proxy runs the
//!   64K WIDTH (`n65536`) on real rows instead.
//! - **llama** (f32 weights, e.g. MiniCPM5-1B: 131K context, GQA 16/2, no
//!   softcap): the true long-context needle — `--ctx 65536` puts 64K REAL
//!   keys under every late row. The tokenizer is the GGUF's BPE.
//!
//! `block_size` is capped at the longest sequence, so a 131K-context model
//! allocates a KV cache for the run, not for its advertised window.
//!
//! Usage:
//! ```text
//! row_logit_floor_ppl <gemma2-f16.gguf> <corpus.txt> [--tokens N] [--seq-len N]
//!                     [--needle N] [--ctx N] [--tv EPS] [--n-sink N]
//!                     [--arms base,b8,b6,b6s0,b6n65536,...] [--dump-tokens N]
//!                     [--decode-floor true]
//! ```
//! Arm grammar: `base` (plain softmax) or `b<bits>` followed by optional
//! `s<n>` (sink count; default `--n-sink`, `s0` = no exemption, T4) and
//! `n<ctx>` (fixed width `ln(ctx/ε)` for every row; default per-row).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::gemma_layer::GemmaTransformerWeightsF16;
use riir_infer_core::gguf_loader::{
    GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct, load_llama_weights_gguf,
};
use riir_infer_core::llama_layer::LlamaTransformerWeights;
use riir_infer_core::tokenizer::{BpeTokenizer, SentencePieceGgufTokenizer};
use riir_infer_core::transformer::attention_floor::{RowLogitFloorPolicy, RowLogitFloorStats};
use riir_infer_core::transformer::attention_probe::AttnSpanProbe;
use riir_infer_core::transformer::{ForwardContext, forward_gemma2_f16, forward_llama};
use riir_infer_core::types::Config;

/// The fixture: one decode forward per architecture.
enum Model {
    Gemma2F16(GemmaTransformerWeightsF16),
    Llama(LlamaTransformerWeights),
}

impl Model {
    fn forward<'a>(
        &self,
        ctx: &'a mut ForwardContext,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        match self {
            Self::Gemma2F16(w) => forward_gemma2_f16(ctx, w, cache, token, pos, config),
            Self::Llama(w) => forward_llama(ctx, w, cache, token, pos, config),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Gemma2F16(_) => "gemma-2 f16",
            Self::Llama(_) => "llama f32",
        }
    }
}

/// The fixture's tokenizer, erased to the one call this bin makes.
type Encode = Box<dyn Fn(&str) -> Vec<usize>>;

/// `(config, model, encode, bos)` for the GGUF at `path`.
fn load(path: &std::path::Path) -> Result<(Config, Model, Encode, usize)> {
    let gguf = GgufFile::open(path).context("open gguf")?;
    match gguf.architecture() {
        Some("gemma2") => {
            let config = config_from_gguf_metadata(&gguf)?;
            let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
            let weights = load_gemma2_f16_direct(&gguf, &config)?;
            let bos = tok.bos_id();
            Ok((
                config,
                Model::Gemma2F16(weights),
                Box::new(move |t| tok.encode(t)),
                bos,
            ))
        }
        Some("llama") => {
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            drop(gguf);
            let (config, weights) = load_llama_weights_gguf(path)?;
            let bos = tok.bos_id();
            Ok((
                config,
                Model::Llama(weights),
                Box::new(move |t| tok.encode(t)),
                bos,
            ))
        }
        other => bail!("unsupported architecture {other:?} (gemma2 | llama)"),
    }
}

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
    /// The labeled answer span (needle mode): the m_Y probe's `Y`.
    span: Option<std::ops::Range<usize>>,
}

/// One arm's running totals (sequence-major loop).
#[derive(Default)]
struct ArmAcc {
    sum_nll: f64,
    sum_abs: f64,
    flips: usize,
    k: usize,
    exact: usize,
    fwd: usize,
    secs: f64,
    stats: RowLogitFloorStats,
    probe: Option<AttnSpanProbe>,
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
    encode: &dyn Fn(&str) -> Vec<usize>,
    bos: usize,
    filler: &[usize],
    n: usize,
    ctx: usize,
) -> Result<Vec<Seq>> {
    let intro = encode(
        "There is an important pass key hidden inside a lot of irrelevant text. \
         Find it and memorize it.\n\n",
    );
    let question = encode("\n\nWhat is the pass key? The pass key is");
    let mut seqs = Vec::with_capacity(n);
    let mut state = 0x9E37_79B9u32;
    for i in 0..n {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = 10_000 + state % 90_000;
        let needle = encode(&format!(
            "\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
        ));
        let answer = encode(&format!(" {key}."));
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
        let span = t.len()..t.len() + needle.len();
        t.extend_from_slice(&needle);
        t.extend_from_slice(&body[before..]);
        t.extend_from_slice(&question);
        let score_from = t.len() - 1;
        t.extend_from_slice(&answer);
        seqs.push(Seq {
            tokens: t,
            score_from,
            span: Some(span),
        });
    }
    Ok(seqs)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: row_logit_floor_ppl <gguf> <corpus.txt> [--tokens N] [--seq-len N] \
             [--needle N] [--ctx N] [--tv EPS] [--n-sink N] [--arms base,b8,b6,b6s0] \
             [--dump-tokens N] [--decode-floor true]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let (mut n_tokens, mut seq_len, mut tv, mut n_sink) = (2048usize, 1024usize, 1e-3f32, 4usize);
    let (mut needle, mut needle_ctx) = (0usize, 1536usize);
    let mut arm_specs = "base,b8,b6,b6s0".to_string();
    let mut dump_tokens = 0usize;
    let mut decode_floor = false;
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
            "--dump-tokens" => dump_tokens = v.parse()?,
            "--decode-floor" => decode_floor = v.parse()?,
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
    let (mut config, model, encode, bos) = load(&gguf_path)?;
    let text = std::fs::read_to_string(&corpus_path).context("read corpus")?;
    let all = encode(&text);
    let seqs: Vec<Seq> = match needle {
        0 => all[..all.len().min(n_tokens)]
            .chunks(seq_len)
            .map(|c| Seq {
                tokens: std::iter::once(bos).chain(c.iter().copied()).collect(),
                score_from: 0,
                span: None,
            })
            .collect(),
        n => needle_seqs(&*encode, bos, &all, n, needle_ctx)?,
    };
    if dump_tokens > 0 {
        // Fidelity check against a reference implementation (HF `tokenizers`
        // / `transformers`) before trusting a fixture: ppl mode prints the
        // first N corpus ids, needle mode each prompt as `score_from|ids`.
        let join = |t: &[usize]| t.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
        match needle {
            0 => println!("{}", join(&all[..all.len().min(dump_tokens)])),
            _ => seqs
                .iter()
                .for_each(|q| println!("{}|{}", q.score_from, join(&q.tokens))),
        }
        return Ok(());
    }
    let longest = seqs.iter().map(|s| s.tokens.len()).max().unwrap_or(0);
    if longest > config.block_size {
        bail!(
            "a sequence of {longest} tokens exceeds block_size {}",
            config.block_size
        );
    }
    // Cap the KV cache / score buffers at what the run needs (a 131K-context
    // model would otherwise allocate its whole advertised window).
    config.block_size = longest;
    let scored: usize = seqs.iter().map(|s| s.tokens.len() - 1 - s.score_from).sum();
    println!(
        "# row_logit_floor_ppl: {} | layers={} heads={}/{} softcap={} | load {:.1}s",
        model.label(),
        config.n_layer,
        config.n_head,
        config.n_kv_head,
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

    // Sequence-major: per sequence, the positions before `floor_from` run
    // ONCE with the plain softmax (shared by every arm), then each arm runs
    // the rest under its policy. The forward writes KV at `pos` and reads only
    // `0..=pos`, so re-running from `floor_from` overwrites exactly the rows
    // an arm owns and never the shared prefix. `floor_from = 0` (the default,
    // every row floored) is the arm-independent whole sequence;
    // `--decode-floor true` floors only the scored rows — the decode-time
    // low-bit consumer, whose prompt KV is dense.
    let mut ctx = ForwardContext::new(&config);
    let new_probe = || match needle {
        0 => None,
        _ => Some(AttnSpanProbe::new(
            config.n_layer,
            config.n_head,
            config.block_size,
        )),
    };
    let mut accs: Vec<ArmAcc> = arms
        .iter()
        .map(|_| ArmAcc {
            probe: new_probe(),
            ..ArmAcc::default()
        })
        .collect();
    let mut cache = MultiLayerKVCache::new(&config);
    let mut base_nll: Vec<f64> = Vec::with_capacity(scored);
    let mut base_top: Vec<usize> = Vec::with_capacity(scored);
    let mut prefix_fwd = 0usize;
    let t_prefix = Instant::now();
    let mut prefix_secs = 0.0f64;
    for seq in &seqs {
        cache.reset();
        let floor_from = match decode_floor {
            true => seq.score_from,
            false => 0,
        };
        let tp = Instant::now();
        ctx.logit_floor = None;
        ctx.attn_probe = None;
        for pos in 0..floor_from {
            model.forward(&mut ctx, &mut cache, seq.tokens[pos], pos, &config);
            prefix_fwd += 1;
        }
        prefix_secs += tp.elapsed().as_secs_f64();
        for (arm, acc) in arms.iter().zip(accs.iter_mut()) {
            ctx.logit_floor = arm.policy;
            std::mem::swap(&mut ctx.logit_floor_stats, &mut acc.stats);
            std::mem::swap(&mut ctx.attn_probe, &mut acc.probe);
            if let (Some(p), Some(span)) = (ctx.attn_probe.as_mut(), &seq.span) {
                p.span = span.clone();
            }
            let t = Instant::now();
            let mut all_right = true;
            for pos in floor_from..seq.tokens.len() - 1 {
                if let Some(p) = ctx.attn_probe.as_mut() {
                    p.armed = seq.span.is_some() && pos >= seq.score_from;
                    p.rows += u64::from(p.armed);
                }
                let logits = model.forward(&mut ctx, &mut cache, seq.tokens[pos], pos, &config);
                acc.fwd += 1;
                if pos < seq.score_from {
                    continue;
                }
                let target = seq.tokens[pos + 1];
                let l = nll(logits, target);
                let top = argmax(logits);
                acc.sum_nll += l;
                all_right &= top == target;
                match arm.policy {
                    None => {
                        base_nll.push(l);
                        base_top.push(top);
                    }
                    Some(_) => {
                        acc.sum_abs += (l - base_nll[acc.k]).abs();
                        acc.flips += usize::from(top != base_top[acc.k]);
                    }
                }
                acc.k += 1;
            }
            acc.secs += t.elapsed().as_secs_f64();
            acc.exact += usize::from(all_right);
            std::mem::swap(&mut ctx.logit_floor_stats, &mut acc.stats);
            std::mem::swap(&mut ctx.attn_probe, &mut acc.probe);
        }
    }
    if decode_floor {
        println!(
            "# decode-floor: {prefix_fwd} shared dense prefix rows in {:.1}s ({:.2} tok/s); \
             arms floor the scored rows only | total {:.1}s",
            prefix_secs,
            prefix_fwd as f64 / prefix_secs.max(1e-9),
            t_prefix.elapsed().as_secs_f64()
        );
    }

    let mut probe_lines: Vec<String> = Vec::new();
    println!(
        "| arm | ppl | Δppl | mean |ΔNLL| | top-1 flip | seq-exact | mean env TV | floored | mean w (nats) | tok/s |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut base_ppl = 0.0f64;
    for (arm, acc) in arms.iter().zip(&accs) {
        let (k, secs, fwd) = (acc.k, acc.secs, acc.fwd);
        let ppl = (acc.sum_nll / k as f64).exp();
        let seq_exact = 100.0 * acc.exact as f64 / seqs.len() as f64;
        let st = acc.stats;
        if let Some(p) = acc.probe.as_ref() {
            let (l, h, m) = p.top_head();
            let layers: Vec<String> = p.layer_means().iter().map(|m| format!("{m:.3}")).collect();
            probe_lines.push(format!(
                "| {} | {:.4} | L{l}H{h} {m:.3} | {} |",
                arm.name,
                p.m_y(),
                layers.join(" ")
            ));
        }
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
                acc.sum_abs / k as f64,
                100.0 * acc.flips as f64 / k as f64,
                st.mean_envelope_tv(),
                100.0 * st.floored as f64 / st.ctx_keys.max(1) as f64,
                st.width_sum / st.rows.max(1) as f64,
                fwd as f64 / secs
            ),
        }
    }
    if !probe_lines.is_empty() {
        println!("\n# m_Y — attention mass on the needle span, from the question + answer rows");
        println!("| arm | m_Y (all heads) | top head | per-layer mean (L0..) |");
        println!("|---|---|---|---|");
        for l in &probe_lines {
            println!("{l}");
        }
    }
    Ok(())
}
