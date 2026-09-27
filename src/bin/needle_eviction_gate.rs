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
//! # T7 — the hub-distractor regime (`--hubs N`)
//!
//! Bench 008 measured differential ≡ max-recent on natural text: nothing in
//! the filler carries persistent attention mass (μ ≈ 0 outside the sinks,
//! which are exempt), so the λ common-mode correction has nothing to
//! demote — the two policies' selections were identical to 3 decimals.
//! This mode plants the regime: N hub codes (`QZARF####`, "sealed archive"
//! notices — code-shaped like the needles so the question's queries attend
//! them, semantically excluded from the question), each planted as 4
//! repeated blocks through the haystack with the code repeated 3× per
//! block. Within-block repetition is what builds μ: the block's own
//! queries (recency + induction) continuously attend the earlier code
//! occurrences, so hub code rows carry SUSTAINED mass while needle code
//! rows carry only the question-time spike.
//!
//! Pre-registered separation readout: hub-row survival must FALL with λ
//! while needle-row survival does not (the μ-correction demotes hubs
//! specifically). If no λ in the grid moves hub survival at any budget,
//! the μ mechanism does not bind on real text even under planted hub
//! repetition — Issue 012 T7 closes as a recorded NEGATIVE (the primitive
//! is synthetic-regime-bound at this observation state).
//!
//! # T5 — the sink A/B (`--sinks N`)
//!
//! The pinned sink count (kv_sink_window n_sink; default 4 = the shipped
//! convention). The Bench 008 pilot's λ grid was flat with n_sink=4
//! pinned throughout, so the sink A/B had no signal; rerun the winning
//! hub-regime cell with `--sinks 0` vs `--sinks 4` once λ sensitivity
//! exists.
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

/// Sinks: the kv_sink_window default convention (first 4 positions) — the
/// `--sinks` default. The runtime value is `args.sinks` (T5 A/B lever).
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
    /// T7: distinct hub codes, each planted as 4 repeated "sealed archive"
    /// blocks through the haystack. 0 = the Bench 008 fixture,
    /// byte-identical (no hub splices, no hub rows).
    hubs: usize,
    /// Differential λ grid — one diff arm per value (CSV). All share the
    /// single full-cache reference, so every λ's delta is against the same
    /// full run. max-recent is the λ=0 baseline arm regardless.
    lambdas: Vec<f32>,
    beta: f32,
    window: u32,
    cadence: usize,
    free_cap: usize,
    seed: u64,
    chunk: usize,
    /// 1 = the deferred protocol (default): the prompt prefills into the
    /// FULL cache and the first compression fires at decode start — the
    /// regime where the question's queries re-arm needle evidence before
    /// the budget applies. 0 = pure streaming (eviction during the
    /// haystack too; needles die young — the pilot's honest negative).
    defer: u8,
    /// Arm filter: comma-separated substrings; an arm runs when its name
    /// contains ANY of them (the full reference always runs — the deltas
    /// need it).
    arms_filter: Option<Vec<String>>,
    /// T5: pinned sink rows at the head of the cache (kv_sink_window).
    sinks: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf"),
        corpus: PathBuf::from("../riir-train/data/chat_probe"),
        context: 65_536,
        budget_fracs: vec![0.25, 0.50],
        needles: 8,
        hubs: 0,
        lambdas: vec![1.0],
        beta: 0.1,
        window: 256,
        cadence: 512,
        free_cap: 96,
        seed: 1337,
        chunk: 8192,
        defer: 1,
        arms_filter: None,
        sinks: N_SINK,
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
            "--hubs" => a.hubs = val("hubs").parse().unwrap(),
            "--sinks" => a.sinks = val("sinks").parse().unwrap(),
            "--lambda" => {
                a.lambdas = val("lambda")
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--beta" => a.beta = val("beta").parse().unwrap(),
            "--window" => a.window = val("window").parse().unwrap(),
            "--cadence" => a.cadence = val("cadence").parse().unwrap(),
            "--gen" => a.free_cap = val("gen").parse().unwrap(),
            "--seed" => a.seed = val("seed").parse().unwrap(),
            "--chunk" => a.chunk = val("chunk").parse().unwrap(),
            "--defer" => a.defer = val("defer").parse().unwrap(),
            "--arms" => {
                a.arms_filter = Some(
                    val("arms")
                        .split(',')
                        .map(str::to_string)
                        .collect(),
                )
            }
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
    /// T7: same readout over the hub-code rows (NaN when `--hubs 0`).
    /// The separation instrument: must FALL with λ while needle-row
    /// survival does not.
    hub_row_survival: f32,
    /// T7: the WALL hub rows only (the last ~200 tokens before the
    /// question — where μ is live at eviction time). The primary
    /// separation readout.
    wall_row_survival: f32,
    /// Total hub code rows in the prompt (0 when `--hubs 0`).
    hub_rows: usize,
    /// Wall hub code rows (0 when `--hubs 0`).
    wall_rows: usize,
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
        let phrase_toks = tok.encode(&format!(" QWARF{code:04}"));
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
        let phrase_toks = tok.encode(&format!(" QWARF{:04}", n.code));
        let span = find_subspan(&answer_toks, &phrase_toks)
            .with_context(|| format!("locate phrase in answer for needle {i}"))?;
        answer_code_spans.push(span);
    }

    // ── T7 hub blocks (distractor regime; none when --hubs 0) ──
    //
    // TWO placements, and the distinction is the instrument:
    // - MID blocks: the even interleave through the haystack. Their μ decays
    //   (β=0.1 EMA horizon ≈ 10 queries) long before eviction, so they
    //   compete on question-time mass alone under BOTH scores — distractor
    //   pressure, no λ separation expected.
    // - WALL blocks: the last ~200 tokens before the question. Their own
    //   queries sit inside the observation window (W), so their rows carry
    //   SUSTAINED mass at eviction time: a moderate, μ ≈ a ⇒ d = a − λμ ≈ 0
    //   (demoted at λ ≥ 1) while λ = 0 keeps them on raw a. Needle rows in
    //   the same window carry the question spike with μ ≈ 0.1·spike. The
    //   pre-registered separation readout is wall-hub survival FALLING with
    //   λ while needle survival does not.
    const HUB_BLOCKS_PER_CODE: usize = 4;
    const HUB_WALL_BLOCKS: usize = 4;
    struct HubBlock {
        toks: Vec<usize>,
        /// Relative spans of EVERY hub-code occurrence inside `toks`.
        code_spans: Vec<(usize, usize)>,
    }
    let make_hub_block = |c: usize, code: u64| -> Result<HubBlock> {
        let text = format!(
            "Sealed archive notice {c}: restricted code QZARF{code:04} stays sealed; do not disclose QZARF{code:04} before the audit closes; reference QZARF{code:04} only through the records office. "
        );
        let toks = tok.encode(&text);
        let phrase = tok.encode(&format!(" QZARF{code:04}"));
        let code_spans = find_all_subspans(&toks, &phrase)
            .with_context(|| format!("hub code spans (code {code})"))?;
        Ok(HubBlock { toks, code_spans })
    };
    let hub_codes: Vec<u64> = (0..args.hubs).map(|i| 20_000 + i as u64 * 977).collect();
    let mut hub_blocks: Vec<HubBlock> =
        Vec::with_capacity(hub_codes.len() * HUB_BLOCKS_PER_CODE);
    for _b in 0..HUB_BLOCKS_PER_CODE {
        for (c, &code) in hub_codes.iter().enumerate() {
            hub_blocks.push(make_hub_block(c, code)?);
        }
    }
    let mut wall_blocks: Vec<HubBlock> = Vec::with_capacity(HUB_WALL_BLOCKS);
    for wb in 0..HUB_WALL_BLOCKS {
        if hub_codes.is_empty() {
            break;
        }
        let c = wb % hub_codes.len();
        wall_blocks.push(make_hub_block(c, hub_codes[c])?);
    }

    // Prompt: filler with needle sentences + hub blocks spliced at even
    // depths (needles first at evenly-strided slots, hub blocks filling the
    // remaining slots in order — each code's 4 blocks land ~1/4 of the
    // usable depth apart).
    let needle_budget = needle_sent_toks.iter().map(|s| s.len()).sum::<usize>();
    let hub_budget: usize = hub_blocks.iter().map(|b| b.toks.len()).sum();
    let usable = args.context.saturating_sub(
        question_toks.len() + needle_budget + hub_budget + 64,
    );
    enum Splice<'a> {
        Needle(usize),
        Hub(&'a HubBlock),
        /// A wall block — hub rows measured SEPARATELY (the separation
        /// instrument; see the block comment above).
        HubWall(&'a HubBlock),
    }
    let n_slots = args.needles + hub_blocks.len();
    let mut slot_of: Vec<Option<Splice>> = (0..n_slots).map(|_| None).collect();
    for i in 0..args.needles {
        let s = ((i + 1) * n_slots / (args.needles + 1)).saturating_sub(1).min(n_slots - 1);
        slot_of[s] = Some(Splice::Needle(i));
    }
    {
        let mut hb = hub_blocks.iter();
        for slot in slot_of.iter_mut() {
            if slot.is_none()
                && let Some(block) = hb.next()
            {
                *slot = Some(Splice::Hub(block));
            }
        }
    }
    // Wall depths: HUB_WALL_BLOCKS blocks ending `16` tokens before the
    // question, stride block_len+8 — the final ~200 prompt tokens are hub
    // blocks, all inside/at the edge of the W-token observation window.
    let mut splices: Vec<(usize, Splice)> = Vec::with_capacity(n_slots + wall_blocks.len());
    for (s, item) in slot_of.into_iter().enumerate() {
        if let Some(item) = item {
            let depth = usable * (s + 1) / (n_slots + 1) + args.sinks + 8;
            splices.push((depth, item));
        }
    }
    for (wb, block) in wall_blocks.iter().enumerate() {
        let blen = block.toks.len();
        let depth = args
            .context
            .saturating_sub(16 + blen + wb * (blen + 8));
        splices.push((depth, Splice::HubWall(block)));
    }
    splices.sort_by_key(|(d, _)| *d);
    let mut prompt: Vec<usize> = Vec::with_capacity(args.context + 64);
    let mut filler_cursor = 0usize;
    let mut hub_spans: Vec<(usize, usize)> = Vec::new();
    let mut wall_spans: Vec<(usize, usize)> = Vec::new();
    for (depth, item) in &splices {
        while prompt.len() < *depth && filler_cursor < filler_tokens.len() {
            prompt.push(filler_tokens[filler_cursor]);
            filler_cursor += 1;
        }
        match item {
            Splice::Needle(i) => {
                let sent = &needle_sent_toks[*i];
                let sent_start = prompt.len();
                prompt.extend_from_slice(sent);
                let phrase_toks = tok.encode(&format!(" QWARF{:04}", needles[*i].code));
                let span = find_subspan(sent, &phrase_toks)?;
                needles[*i].depth_tokens = sent_start;
                needles[*i].prompt_code_span = (sent_start + span.0, sent_start + span.1);
            }
            Splice::Hub(block) => {
                let start = prompt.len();
                prompt.extend_from_slice(&block.toks);
                for (r0, r1) in &block.code_spans {
                    hub_spans.push((start + r0, start + r1));
                }
            }
            Splice::HubWall(block) => {
                let start = prompt.len();
                prompt.extend_from_slice(&block.toks);
                for (r0, r1) in &block.code_spans {
                    let span = (start + r0, start + r1);
                    wall_spans.push(span);
                    hub_spans.push(span);
                }
            }
        }
    }
    while prompt.len() < args.context && filler_cursor < filler_tokens.len() {
        prompt.push(filler_tokens[filler_cursor]);
        filler_cursor += 1;
    }
    // A splice is never placed close enough to the tail for the truncate to
    // cut it (last slot sits ~usable/(n_slots+1) before `context`, and one
    // block is far smaller) — but a corpus shorter than `context` would
    // shift every depth, so the spans are checked, not assumed.
    for (i, n) in needles.iter().enumerate() {
        if n.prompt_code_span.1 > args.context {
            bail!("needle {i} code span {:?} truncated (corpus shorter than context?)", n.prompt_code_span);
        }
    }
    for (i, (s0, s1)) in hub_spans.iter().enumerate() {
        if *s1 > args.context {
            bail!("hub span {i} ({s0}..{s1}) truncated (corpus shorter than context?)");
        }
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
        "# rig: context={} needles={} hubs={} hub_code_rows={} wall_rows={} attn_layers={} prompt={} question={} answer={} fracs={:?} λ={} β={} W={} cadence={} defer={} sinks={}",
        args.context,
        args.needles,
        args.hubs,
        hub_spans.len(),
        wall_spans.len(),
        n_attn_layers,
        prompt.len(),
        question_toks.len(),
        answer_toks.len(),
        args.budget_fracs,
        args.lambdas
            .iter()
            .map(|l| format!("{l}"))
            .collect::<Vec<_>>()
            .join(","),
        args.beta,
        args.window,
        args.cadence,
        args.defer,
        args.sinks,
    );
    for (i, n) in needles.iter().enumerate() {
        println!(
            "# needle {i}: depth={} code_span={:?} answer_span={:?}",
            n.depth_tokens, n.prompt_code_span, answer_code_spans[i],
        );
    }
    if !hub_spans.is_empty() {
        println!(
            "# hub spans: {} rows (wall {}), first={:?} last={:?}",
            hub_spans.len(),
            wall_spans.len(),
            hub_spans.first(),
            hub_spans.last(),
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
        for &lambda in &args.lambdas {
            arms.push(Arm {
                name: format!("diff_l{lambda:.2}@{tag}"),
                policy: Some(EvictPolicy::Differential(DiffEvictConfig::new(
                    lambda, args.beta, args.window,
                ))),
                budget,
            });
        }
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
    let mut full_nll_opt: Option<Vec<f32>> = None;
    let mut reports: Vec<ArmReport> = Vec::with_capacity(arms.len());
    for arm in &arms {
        if let Some(fs) = &args.arms_filter
            && arm.name != "full"
            && !fs.iter().any(|f| arm.name.contains(f.as_str()))
        {
            continue;
        }
        let defer_until = if args.defer != 0 { prompt.len() as u64 } else { 0 };
        let layer_cfg = arm.policy.map(|policy| EvictLayerConfig {
            policy,
            budget: arm.budget,
            cadence: args.cadence,
            defer_until,
        });
        let report = run_arm(
            &args,
            &config,
            &weights,
            &rope_freq,
            &prompt,
            &needles,
            &hub_spans,
            &wall_spans,
            &answer_code_spans,
            &answer_toks,
            layer_cfg,
            &arm.name,
            &tok,
        )?;
        println!(
            "# arm {:<18} {:.1}s | evicted={:>6} events={:>3} idle={:>3} | genNLL={:.3} | out_len={}{} | surv n/h/w = {:.3}/{:.3}/{:.3}",
            report.name,
            report.wall_seconds,
            report.evicted_rows,
            report.evict_events,
            report.idle_events,
            report.generic_nll.unwrap_or(f32::NAN),
            report.output_len,
            if report.capped { " (CAPPED)" } else { "" },
            report.needle_row_survival,
            report.hub_row_survival,
            report.wall_row_survival,
        );
        // Incremental delta vs the full arm (it always runs first) — the
        // verdict survives even if a later arm or the summary is lost.
        if report.name == "full" {
            full_nll_opt = Some(report.needle_nll.clone());
        } else if let Some(full_nll) = &full_nll_opt {
            let deltas: Vec<f32> = report
                .needle_nll
                .iter()
                .zip(full_nll)
                .map(|(&a, &f)| a - f)
                .collect();
            let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
            let retained = deltas.iter().filter(|&&d| d <= EPS_NATS).count();
            println!(
                "#   -> {} retrieval {}/{} (meanΔ {mean:+.3}) per-needle Δ: {:?}",
                report.name,
                retained,
                deltas.len(),
                deltas,
            );
        }
        reports.push(report);
    }

    // ── Deltas vs full + the verdict table ──
    let full_nll = reports[0].needle_nll.clone();
    let n_needles = args.needles;
    let bar = 1.0 - 1.0 / n_needles as f32;
    println!("\n# ── retrieval (bar ≥ {}/{} = {bar:.3}, EPS_NATS = {EPS_NATS}) ──", n_needles - 1, n_needles);
    println!(
        "# {:<18} {:>9} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "arm", "retrieval", "meanΔnats", "n_surv", "hub_surv", "wall_surv", "genNLL"
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
            "# {:<18} {:>9.3} {:>+10.3} {:>10.3} {:>10.3} {:>10.3} {:>9.3}",
            r.name,
            r.retrieval(),
            r.mean_delta,
            r.needle_row_survival,
            r.hub_row_survival,
            r.wall_row_survival,
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

    println!("\n# wall total {:.1}s | rig tokens {} | λ={} β={} W={} cadence={} hubs={} sinks={}",
        t_start.elapsed().as_secs_f32(), prompt.len(),
        args.lambdas.iter().map(|l| format!("{l}")).collect::<Vec<_>>().join(","),
        args.beta, args.window, args.cadence, args.hubs, args.sinks);

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

/// All non-overlapping occurrences of `needle` in `hay`, left to right
/// (the hub block repeats its code, so one block carries several spans).
fn find_all_subspans(hay: &[usize], needle: &[usize]) -> Result<Vec<(usize, usize)>> {
    if needle.is_empty() || needle.len() > hay.len() {
        bail!("empty/oversized needle span");
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + needle.len() <= hay.len() {
        if hay[start..start + needle.len()] == *needle {
            out.push((start, start + needle.len()));
            start += needle.len();
        } else {
            start += 1;
        }
    }
    if out.is_empty() {
        bail!("needle span not found");
    }
    Ok(out)
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
    hub_spans: &[(usize, usize)],
    wall_spans: &[(usize, usize)],
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
        defer_until: 0,
    });
    let mut evictor = EvictorState::new(
        Some(&cfg),
        SinkWindowPolicy::new(args.sinks, usize::MAX),
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
    eprintln!("# [{name}] phases: prefill={prefill_done:.1}s");

    // Secondary readout: are the needle / hub / wall-hub code rows still
    // RESIDENT when the answer starts? (policy claim, model-independent) —
    // averaged over attention layers, via the slot→logical map.
    fn row_survival(
        evictor: &mut EvictorState,
        layer_attn: &[bool],
        spans: &[(usize, usize)],
    ) -> f32 {
        if spans.is_empty() {
            return f32::NAN;
        }
        let mut layers = 0usize;
        let mut acc = 0.0f32;
        for (li, &is_attn) in layer_attn.iter().enumerate() {
            if !is_attn {
                continue;
            }
            let layer = evictor.layer(li);
            let mut hit = 0usize;
            let mut total = 0usize;
            for (s0, s1) in spans {
                for pos in *s0..*s1 {
                    total += 1;
                    if layer.slot_of_logical(pos as u64).is_some() {
                        hit += 1;
                    }
                }
            }
            if total > 0 {
                acc += hit as f32 / total as f32;
                layers += 1;
            }
        }
        if layers > 0 {
            acc / layers as f32
        } else {
            f32::NAN
        }
    }
    let needle_spans: Vec<(usize, usize)> = needles
        .iter()
        .map(|n| n.prompt_code_span)
        .collect();
    let needle_row_survival = row_survival(&mut evictor, &layer_attn, &needle_spans);
    let hub_row_survival = row_survival(&mut evictor, &layer_attn, hub_spans);
    let wall_row_survival = row_survival(&mut evictor, &layer_attn, wall_spans);
    let survival_done = t_arm.elapsed().as_secs_f32();
    eprintln!("# [{name}] phases: +survival={:.1}s", survival_done - prefill_done);

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
    eprintln!("# [{name}] phases: +answer={:.1}s", answer_done - survival_done);

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
    let generic_done = t_arm.elapsed().as_secs_f32();
    eprintln!("# [{name}] phases: +generic={:.1}s", generic_done - answer_done);

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
    eprintln!(
        "# [{name}] phases: +runaway={:.1}s",
        t_arm.elapsed().as_secs_f32() - generic_done
    );
    Ok(ArmReport {
        name: name.to_string(),
        needle_nll,
        delta_nll: vec![f32::NAN; needles.len()],
        mean_delta: f32::NAN,
        retained: 0,
        n_needles: needles.len(),
        needle_row_survival,
        hub_row_survival,
        wall_row_survival,
        hub_rows: hub_spans.iter().map(|(s0, s1)| s1 - s0).sum(),
        wall_rows: wall_spans.iter().map(|(s0, s1)| s1 - s0).sum(),
        generic_nll,
        capped,
        output_len,
        evicted_rows: stats.evicted_rows,
        evict_events: stats.evict_events,
        idle_events: stats.idle_events,
        wall_seconds: t_arm.elapsed().as_secs_f32(),
    })
}
