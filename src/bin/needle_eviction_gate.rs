#![cfg(feature = "kv_eviction")]

//! Multi-needle @ long-context differential-KV-eviction gate — riir-infer
//! Issue 012 T2/T4/T5, the model-bound quality gate katgpt-rs Issue 882 P3
//! cannot run (Bench 894's synthetic hub fixture has no model behind it).
//!
//! # The rig
//!
//! `haystack (filler text with needle sentences spliced at fixed depths) +
//! question + teacher-forced answer`. The model is the qwen35-hybrid
//! Qwen3.5-0.8B-Base Q8_0 lane (ctx 262K, so 64K is in-distribution; the
//! 27B Bonsai lane needs hours per prefill on this CPU box and is outside
//! the gate's compute envelope — the arithmetic is recorded in the issue).
//! Every arm runs the ARMED forward (`EvictorState`); the full-cache
//! reference is armed with `budget = usize::MAX`, which is T3-bit-identical
//! to the unarmed path, so all arms share one code path and one observation
//! substrate.
//!
//! # Arms (T2)
//!
//! differential (λ, β, W) · max-recent (λ = 0 — the bit-identical baseline)
//! · usage-rate (the shipped H2O-class score) · prompt-pinned random (the
//! `beats_random_prompt_pin` null, seeded). Each at every `--budget-frac`
//! of the context, plus the full-cache reference.
//!
//! # Retrieval metric (pre-registered before any budget arm ran)
//!
//! - Per needle i: `delta_i = NLL_i(arm) − NLL_i(full)` — the mean
//!   teacher-forced NLL over the needle's CODE tokens in the answer, vs the
//!   same span under the full-cache arm.
//! - Needle RETAINED ⟺ `delta_i ≤ EPS_NATS` (1.0 nats/token; the full
//!   delta distribution is reported either way).
//! - `retrieval(arm) = retained / K`; the G1 bar is `≥ 1 − 1/K` (Bench
//!   894's "≥ full − 1/16" analog).
//! - Secondary: needle-row survival — whether the needle's code K/V rows
//!   are still in the compacted cache when the answer starts (the policy's
//!   own claim, independent of the model's use of them).
//!
//! ⚠ The differential table's SPECIFICITY is deliberately NOT a retrieval
//! readout here: it is a recency instrument (a two-bucket window of `W`
//! queries), so at prefill end it holds only the question's last tokens'
//! evidence — reading it would measure the question's recency, not
//! retrieval. The NLL metric is the model-bound signal.
//!
//! # T4 (trap 4) — the generic-continuation control
//!
//! The same arms score a NEUTRAL continuation's NLL (text the question does
//! not reference). Bench 894's measured negative: differential was 1.35–1.47×
//! the baseline's output error when the observation window held one-off
//! spikes the future did not need. Report `generic_NLL(arm) /
//! generic_NLL(max-recent)` — above 1.0 with needle retrieval still passing
//! is the trap firing on real text.
//!
//! # Runaway probe (the lossy-KV promotion gate)
//!
//! Per arm: greedy free decode to the `--gen` cap; `output_len` and the
//! at-cap flag feed the `RunawayStats` readout (generation-runaway is the
//! failure mode perplexity-style metrics read as FINE — katgpt-core
//! `runaway_gate`).
//!
//! # Box state
//!
//! The bin prints wall times per arm and refuses to summarize without
//! them; the run log records free RAM / load / power next to the figures
//! (the Feature-Flag-Discipline G2 law — box state is part of the claim).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::kv_evict::{EvictLayerConfig, EvictPolicy, EvictorState};
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim,
    forward_qwen_deltanet_evictable, prefill_qwen_deltanet_chunk_into,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::{Config, DeltaNetLayerType};

use katgpt_core::kv_eviction::differential::DiffEvictConfig;
use katgpt_core::kv_sink_window::SinkWindowPolicy;

/// Pre-registered retention threshold (nats/token over the needle's code
/// span). Frozen before any budget arm ran; see the module doc.
const EPS_NATS: f32 = 1.0;

/// Sinks: the kv_sink_window default convention (first 4 positions).
const N_SINK: usize = 4;

/// T4 control text: a neutral continuation (nothing to do with the needles).
const GENERIC_CONTINUATION: &str = "\nThe weather reports suggest routine conditions for the coming week, with mild temperatures expected throughout the region.";

fn needle_sentence(i: usize, code: u64) -> String {
    format!("Registry entry {i}: the access phrase is QWARF{code:04}, repeat, QWARF{code:04}. ")
}

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    context: usize,
    budget_fracs: Vec<f32>,
    needles: usize,
    lambda: f32,
    beta: f32,
    window: u32,
    cadence: usize,
    free_cap: usize,
    seed: u64,
    chunk: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf"),
        corpus: PathBuf::from("../riir-train/data/chat_probe"),
        context: 65_536,
        budget_fracs: vec![0.25, 0.50],
        needles: 8,
        lambda: 1.0,
        beta: 0.1,
        window: 256,
        cadence: 512,
        free_cap: 96,
        seed: 1337,
        chunk: 8192,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut val = |name: &str| -> String {
            it.next().unwrap_or_else(|| panic!("--{name} needs a value"))
        };
        match f.as_str() {
            "--model" => a.model = val("model").into(),
            "--corpus" => a.corpus = val("corpus").into(),
            "--context" => a.context = val("context").parse().unwrap(),
            "--budget-frac" => {
                a.budget_fracs = val("budget-frac")
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--needles" => a.needles = val("needles").parse().unwrap(),
            "--lambda" => a.lambda = val("lambda").parse().unwrap(),
            "--beta" => a.beta = val("beta").parse().unwrap(),
            "--window" => a.window = val("window").parse().unwrap(),
            "--cadence" => a.cadence = val("cadence").parse().unwrap(),
            "--gen" => a.free_cap = val("gen").parse().unwrap(),
            "--seed" => a.seed = val("seed").parse().unwrap(),
            "--chunk" => a.chunk = val("chunk").parse().unwrap(),
            "--out" => a.out = Some(val("out").into()),
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

/// One planted needle.
struct Needle {
    /// Prompt token index of the needle sentence's start (depth).
    depth_tokens: usize,
    /// Prompt token indices of the needle's CODE run.
    prompt_code_span: (usize, usize),
    code: u64,
}

/// Per-arm result (also the JSON report row).
#[derive(serde::Serialize)]
struct ArmReport {
    name: String,
    /// NLL per needle (nats/token over the code span).
    needle_nll: Vec<f32>,
    /// delta vs the full arm — filled after all arms ran.
    delta_nll: Vec<f32>,
    mean_delta: f32,
    retained: usize,
    n_needles: usize,
    /// Fraction of needle code rows still resident at answer start,
    /// averaged over attention layers.
    needle_row_survival: f32,
    /// T4: neutral continuation NLL per token.
    generic_nll: Option<f32>,
    /// Runaway probe: greedy decode hit the cap without EOS.
    capped: bool,
    output_len: usize,
    evicted_rows: u64,
    evict_events: u64,
    idle_events: u64,
    wall_seconds: f32,
}

impl ArmReport {
    fn retrieval(&self) -> f32 {
        self.retained as f32 / self.n_needles.max(1) as f32
    }
}

fn main() -> Result<()> {
    let args = parse_args();
    let t_start = Instant::now();

    // ── Model + tokenizer ──
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_weights_gguf(&args.model)
        .with_context(|| format!("load {}", args.model.display()))?;
    println!(
        "# model: qwen35 hybrid | layers={} n_embd={} n_head={} n_kv_head={} head_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    let tok = {
        let gguf = GgufFile::open(&args.model).context("re-open gguf for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };

    // ── Cache ceiling: context + answer + generic + free-decode slack ──
    config.block_size = args.context + 8 * args.needles * 16 + args.free_cap + 256;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));

    // ── Rig assembly ──
    let filler_text = load_corpus_text(&args.corpus)
        .with_context(|| format!("load corpus {}", args.corpus.display()))?;
    let filler_tokens = tok.encode(&filler_text);
    if filler_tokens.len() < args.context {
        bail!(
            "corpus too short: {} tokens < context {}",
            filler_tokens.len(),
            args.context
        );
    }

    // Needle sentences + their code spans (one pass).
    let mut needles: Vec<Needle> = Vec::with_capacity(args.needles);
    let mut needle_sent_toks: Vec<Vec<usize>> = Vec::with_capacity(args.needles);
    for i in 0..args.needles {
        let code = 1000 + (i as u64) * 977; // deterministic, distinct
        let sent_toks = tok.encode(&needle_sentence(i, code));
        let phrase_toks = tok.encode(&format!("QWARF{code:04}"));
        let span = find_subspan(&sent_toks, &phrase_toks)
            .with_context(|| format!("locate phrase in needle sentence {i}"))?;
        needles.push(Needle {
            depth_tokens: 0,
            prompt_code_span: span,
            code,
        });
        needle_sent_toks.push(sent_toks);
    }

    // Question + answer.
    let question = "\n\nQuestion: What are the access phrases of all the registry entries above? Answer with each entry number and its phrase.\nAnswer:";
    let question_toks = tok.encode(question);
    let mut full_answer = String::new();
    for (i, n) in needles.iter().enumerate() {
        full_answer.push_str(&format!(" Entry {}: QWARF{:04}.", i, n.code));
    }
    let answer_toks = tok.encode(&full_answer);
    // Answer code spans, located inside the WHOLE tokenized answer (BPE can
    // merge across the written boundaries).
    let mut answer_code_spans: Vec<(usize, usize)> = Vec::with_capacity(needles.len());
    for (i, n) in needles.iter().enumerate() {
        let phrase_toks = tok.encode(&format!("QWARF{:04}", n.code));
        let span = find_subspan(&answer_toks, &phrase_toks)
            .with_context(|| format!("locate phrase in answer for needle {i}"))?;
        answer_code_spans.push(span);
    }

    // Prompt: filler with needle sentences spliced at even depths.
    let needle_budget = needle_sent_toks.iter().map(|s| s.len()).sum::<usize>();
    let usable = args
        .context
        .saturating_sub(question_toks.len() + needle_budget + 64);
    let mut prompt: Vec<usize> = Vec::with_capacity(args.context + 64);
    let mut filler_at = 0usize;
    {
        // Fill so far; splice; the final tail brings the total to `context`.
        let mut filler_cursor = filler_at;
        for (i, sent) in needle_sent_toks.iter().enumerate() {
            let depth = usable * (i + 1) / (args.needles + 1) + N_SINK + 8;
            while prompt.len() < depth && filler_cursor < filler_tokens.len() {
                prompt.push(filler_tokens[filler_cursor]);
                filler_cursor += 1;
            }
            let sent_start = prompt.len();
            prompt.extend_from_slice(sent);
            let phrase_toks = tok.encode(&format!("QWARF{:04}", needles[i].code));
            let span = find_subspan(sent, &phrase_toks)?;
            needles[i].depth_tokens = sent_start;
            needles[i].prompt_code_span = (sent_start + span.0, sent_start + span.1);
        }
        filler_at = filler_cursor;
    }
    while prompt.len() < args.context && filler_at < filler_tokens.len() {
        prompt.push(filler_tokens[filler_at]);
        filler_at += 1;
    }
    prompt.truncate(args.context);
    // The question begins exactly at `args.context` — the prefill boundary.
    prompt.extend_from_slice(&question_toks);

    let n_attn_layers = weights
        .layer_types
        .iter()
        .filter(|&&t| t == DeltaNetLayerType::Attention)
        .count();
    println!(
        "# rig: context={} needles={} attn_layers={} prompt={} question={} answer={} fracs={:?} λ={} β={} W={} cadence={}",
        args.context,
        args.needles,
        n_attn_layers,
        prompt.len(),
        question_toks.len(),
        answer_toks.len(),
        args.budget_fracs,
        args.lambda,
        args.beta,
        args.window,
        args.cadence,
    );
    for (i, n) in needles.iter().enumerate() {
        println!(
            "# needle {i}: depth={} code_span={:?} answer_span={:?}",
            n.depth_tokens, n.prompt_code_span, answer_code_spans[i],
        );
    }

    // ── Arms ──
    struct Arm {
        name: String,
        /// None = the full-cache reference (armed, budget MAX).
        policy: Option<EvictPolicy>,
        budget: usize,
    }
    let mut arms: Vec<Arm> = vec![Arm {
        name: "full".into(),
        policy: None,
        budget: usize::MAX,
    }];
    for &frac in &args.budget_fracs {
        let budget = (args.context as f32 * frac) as usize;
        let tag = format_frac(frac);
        arms.push(Arm {
            name: format!("diff_l{:.2}@{tag}", args.lambda),
            policy: Some(EvictPolicy::Differential(DiffEvictConfig::new(
                args.lambda, args.beta, args.window,
            ))),
            budget,
        });
        arms.push(Arm {
            name: format!("maxrecent@{tag}"),
            policy: Some(EvictPolicy::Differential(DiffEvictConfig::max_recent(
                args.window,
            ))),
            budget,
        });
        arms.push(Arm {
            name: format!("usage@{tag}"),
            policy: Some(EvictPolicy::UsageRate),
            budget,
        });
        arms.push(Arm {
            name: format!("random@{tag}"),
            policy: Some(EvictPolicy::Random { seed: args.seed }),
            budget,
        });
    }

    // ── Run ──
    let mut reports: Vec<ArmReport> = Vec::with_capacity(arms.len());
    for arm in &arms {
        let layer_cfg = arm.policy.map(|policy| EvictLayerConfig {
            policy,
            budget: arm.budget,
            cadence: args.cadence,
        });
        let report = run_arm(
            &args,
            &config,
            &weights,
            &rope_freq,
            &prompt,
            &needles,
            &answer_code_spans,
            &answer_toks,
            layer_cfg,
            &arm.name,
            &tok,
        )?;
        println!(
            "# arm {:<18} {:.1}s | evicted={:>6} events={:>3} idle={:>3} | genNLL={:.3} | out_len={}{}",
            report.name,
            report.wall_seconds,
            report.evicted_rows,
            report.evict_events,
            report.idle_events,
            report.generic_nll.unwrap_or(f32::NAN),
            report.output_len,
            if report.capped { " (CAPPED)" } else { "" },
        );
        reports.push(report);
    }

    // ── Deltas vs full + the verdict table ──
    let full_nll = reports[0].needle_nll.clone();
    let n_needles = args.needles;
    let bar = 1.0 - 1.0 / n_needles as f32;
    println!("\n# ── retrieval (bar ≥ {}/{} = {bar:.3}, EPS_NATS = {EPS_NATS}) ──", n_needles - 1, n_needles);
    println!(
        "# {:<18} {:>9} {:>10} {:>10} {:>9}",
        "arm", "retrieval", "meanΔnats", "survival", "genNLL"
    );
    for r in &mut reports {
        r.delta_nll = r
            .needle_nll
            .iter()
            .zip(&full_nll)
            .map(|(&a, &f)| a - f)
            .collect();
        r.mean_delta = if r.delta_nll.is_empty() {
            f32::NAN
        } else {
            r.delta_nll.iter().sum::<f32>() / r.delta_nll.len() as f32
        };
        r.retained = r.delta_nll.iter().filter(|&&d| d <= EPS_NATS).count();
        println!(
            "# {:<18} {:>9.3} {:>+10.3} {:>10.3} {:>9.3}",
            r.name,
            r.retrieval(),
            r.mean_delta,
            r.needle_row_survival,
            r.generic_nll.unwrap_or(f32::NAN),
        );
    }

    // T4 readout: generic NLL ratio vs the max-recent baseline at the same
    // budget (the Bench-894 trap-4 comparator).
    println!("\n# ── trap-4 control (generic NLL / max-recent's) ──");
    for frac in &args.budget_fracs {
        let tag = format_frac(*frac);
        let base = reports
            .iter()
            .find(|r| r.name == format!("maxrecent@{tag}"))
            .and_then(|r| r.generic_nll);
        let d = reports
            .iter()
            .find(|r| r.name.starts_with("diff_") && r.name.ends_with(&format!("@{tag}")))
            .and_then(|r| r.generic_nll);
        if let (Some(b), Some(d)) = (base, d) {
            println!("# @{tag}: differential/maxrecent = {:.3}  (>1 ⇒ the trap fires on real text)", d / b);
        }
    }

    // Per-needle deltas for the differential arm(s).
    for r in &reports {
        if r.name.starts_with("diff_") {
            println!("\n# per-needle NLL delta vs full — {}:", r.name);
            for (i, n) in needles.iter().enumerate() {
                println!(
                    "# needle {i} depth {:>6}: Δ={:+.3} nats (arm {:.3} / full {:.3})",
                    n.depth_tokens,
                    r.delta_nll[i],
                    r.needle_nll[i],
                    full_nll[i],
                );
            }
        }
    }

    println!("\n# wall total {:.1}s | rig tokens {} | λ={} β={} W={} cadence={}",
        t_start.elapsed().as_secs_f32(), prompt.len(), args.lambda, args.beta, args.window, args.cadence);

    if let Some(out) = &args.out {
        std::fs::create_dir_all(out)?;
        let path = out.join("needle_gate_report.json");
        let json = serde_json::to_string_pretty(&reports)?;
        std::fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
        println!("# report: {}", path.display());
    }
    Ok(())
}

fn format_frac(f: f32) -> String {
    format!("{:.0}pct", f * 100.0)
}

/// Locate `needle` as a contiguous sub-slice of `hay` (token sequences).
fn find_subspan(hay: &[usize], needle: &[usize]) -> Result<(usize, usize)> {
    if needle.is_empty() || needle.len() > hay.len() {
        bail!("empty/oversized needle span");
    }
    for start in 0..=(hay.len() - needle.len()) {
        if &hay[start..start + needle.len()] == needle {
            return Ok((start, start + needle.len()));
        }
    }
    bail!("needle span not found");
}

/// NLL of `tok_id` under the categorical `logits` (nats).
fn nats_log_softmax(logits: &[f32], tok_id: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - max).exp();
    }
    -(logits[tok_id] - (max + sum.ln()))
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    args: &Args,
    config: &Config,
    weights: &riir_infer_core::deltanet::weights::QwenDeltaNetWeights,
    rope_freq: &RopeFreqTable,
    prompt: &[usize],
    needles: &[Needle],
    answer_code_spans: &[(usize, usize)],
    answer_toks: &[usize],
    layer_cfg: Option<EvictLayerConfig>,
    name: &str,
    tok: &BpeTokenizer,
) -> Result<ArmReport> {
    let t_arm = Instant::now();
    let layer_types = weights.layer_types.clone();
    let mut cache = HybridCache::with_layer_types(config, &layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let mut pctx = PrefillContext::new(config, args.chunk);
    let layer_attn: Vec<bool> = layer_types
        .iter()
        .map(|&t| t == DeltaNetLayerType::Attention)
        .collect();
    // The full-cache reference is ARMED with unlimited budget: one code
    // path for every arm (T3 pins armed-headroom ≡ unarmed).
    let cfg = layer_cfg.unwrap_or(EvictLayerConfig {
        policy: EvictPolicy::Differential(DiffEvictConfig::new(0.0, args.beta, args.window)),
        budget: usize::MAX,
        cadence: args.cadence,
    });
    let mut evictor = EvictorState::new(
        Some(&cfg),
        SinkWindowPolicy::new(N_SINK, usize::MAX),
        &layer_attn,
        config.n_head,
        config.block_size,
    );

    let v = config.vocab_size;
    let mut logits = vec![0.0f32; v];
    let mut x = vec![0.0f32; v.max(config.n_embd)];

    // ── Prefill in chunks (eviction runs from the first overshoot) ──
    for (c, chunk) in prompt.chunks(args.chunk).enumerate() {
        prefill_qwen_deltanet_chunk_into(
            weights,
            config,
            &mut cache,
            chunk,
            c * args.chunk,
            &mut scratch,
            rope_freq,
            &mut pctx,
            &mut logits,
            Some(&mut evictor),
        );
    }
    let prefill_done = t_arm.elapsed().as_secs_f32();

    // Secondary readout: are the needle's code rows still RESIDENT when the
    // answer starts? (policy claim, model-independent) — averaged over
    // attention layers, via the slot→logical map.
    let mut surv_layers = 0usize;
    let mut surv_acc = 0.0f32;
    for (li, &is_attn) in layer_attn.iter().enumerate() {
        if !is_attn {
            continue;
        }
        let layer = evictor.layer(li);
        let mut hit = 0usize;
        let mut total = 0usize;
        for n in needles {
            let (s0, s1) = n.prompt_code_span;
            for pos in s0..s1 {
                total += 1;
                if layer.slot_of_logical(pos as u64).is_some() {
                    hit += 1;
                }
            }
        }
        if total > 0 {
            surv_acc += hit as f32 / total as f32;
            surv_layers += 1;
        }
    }
    let needle_row_survival = if surv_layers > 0 { surv_acc / surv_layers as f32 } else { f32::NAN };
    let survival_done = t_arm.elapsed().as_secs_f32();

    // ── Teacher-forced answer decode ──
    // `logits` (from the prefill's last position) predict answer_toks[0].
    let mut needle_nll = vec![0.0f32; needles.len()];
    let mut needle_cnt = vec![0u32; needles.len()];
    let mut prev_logits = logits;
    let mut seq_len = prompt.len();
    for (t, &tok_id) in answer_toks.iter().enumerate() {
        let nll = nats_log_softmax(&prev_logits, tok_id);
        for (i, span) in answer_code_spans.iter().enumerate() {
            if t >= span.0 && t < span.1 {
                needle_nll[i] += nll;
                needle_cnt[i] += 1;
            }
        }
        let out = forward_qwen_deltanet_evictable(
            &mut x,
            weights,
            &mut cache,
            tok_id,
            seq_len,
            config,
            &mut scratch,
            rope_freq,
            &mut evictor,
        );
        prev_logits.copy_from_slice(&out[..v]);
        seq_len += 1;
    }
    for i in 0..needles.len() {
        if needle_cnt[i] > 0 {
            needle_nll[i] /= needle_cnt[i] as f32;
        } else {
            bail!("needle {i} has an empty answer code span — rig assembly bug");
        }
    }
    let answer_done = t_arm.elapsed().as_secs_f32();
    let _ = (prefill_done, survival_done, answer_done);

    // ── T4: generic continuation (teacher-forced, same cache) ──
    let generic_toks = tok.encode(GENERIC_CONTINUATION);
    let mut generic_acc = 0.0f32;
    for &tok_id in &generic_toks {
        generic_acc += nats_log_softmax(&prev_logits, tok_id);
        let out = forward_qwen_deltanet_evictable(
            &mut x,
            weights,
            &mut cache,
            tok_id,
            seq_len,
            config,
            &mut scratch,
            rope_freq,
            &mut evictor,
        );
        prev_logits.copy_from_slice(&out[..v]);
        seq_len += 1;
    }
    let generic_nll = (!generic_toks.is_empty())
        .then(|| generic_acc / generic_toks.len() as f32);

    // ── Runaway probe: greedy free decode to the cap ──
    let eos = tok.eos_id();
    let mut output_len = 0usize;
    let mut capped = false;
    for step in 0..args.free_cap {
        let next = argmax(&prev_logits);
        if next == eos {
            break;
        }
        let out = forward_qwen_deltanet_evictable(
            &mut x,
            weights,
            &mut cache,
            next,
            seq_len,
            config,
            &mut scratch,
            rope_freq,
            &mut evictor,
        );
        prev_logits.copy_from_slice(&out[..v]);
        seq_len += 1;
        output_len += 1;
        if step + 1 == args.free_cap {
            capped = true;
        }
    }

    let stats = evictor.total_stats();
    Ok(ArmReport {
        name: name.to_string(),
        needle_nll,
        delta_nll: vec![f32::NAN; needles.len()],
        mean_delta: f32::NAN,
        retained: 0,
        n_needles: needles.len(),
        needle_row_survival,
        generic_nll,
        capped,
        output_len,
        evicted_rows: stats.evicted_rows,
        evict_events: stats.evict_events,
        idle_events: stats.idle_events,
        wall_seconds: t_arm.elapsed().as_secs_f32(),
    })
}
