#![cfg(all(feature = "tokengate", feature = "deltanet_inference"))]

//! Plan 622 T3.4 — the four-arm acc-vs-K harness: the lane's GOAT gate.
//!
//! Four selector arms measured on the SAME seam at the SAME layer L
//! (the T2.5 pin: L = the middle FULL-ATTENTION layer, arm-(a) mass class):
//!
//! 1. **ceiling** — no prune (the full prefill + full decode), K ≡ 1.
//! 2. **random** — the seeded uniform-K floor: [`random_scores_into`] writes
//!    a random prunable subset through the SAME scorer seam; the incumbent
//!    selection machinery does the rest. `--rand-reps` seeds per cell, the
//!    BEST kept for the floor (the most favorable reading of random-K — if
//!    derived still wins against it, the gate is the stronger for it).
//! 3. **derived** — the T3.3 default arm (a): the layer-L attention-mass
//!    tap, sum-pooled over all queries.
//! 4. **fastkv** — the paper-described FastKV-class selector, REPORTED and
//!    NON-GATING: token scored by attention received from a window of the
//!    LAST queries, max-pooled, top-K. Rides the SAME tap with
//!    `tap_window = Some(w)` — a measured column, never an adjective.
//!
//! # Instruments
//!
//! ONE teacher-forced decode per cell carries BOTH metrics: the needle
//! spans (the T2.5 NIAH metric, `delta_i ≤ 1.0 nat/token`, retrieval =
//! retained/needles) and a corpus continuation (`--ppl-tokens`, the
//! long-ctx ppl axis — NLL of the continuation given prompt + answer,
//! `ppl = exp(mean nats/token)`).
//!
//! # Postures
//!
//! - `sel` — selection-only diagnostic: no bias, no scale anywhere (the
//!   T2.5 posture) — what the SELECTION alone buys.
//! - `gate` — the full T3 overlay: the pruned-PREFILL steering window (the
//!   packed kept-order rows through `KeyBiasLayout`) + the T2.3 decode gate
//!   (`build_decode_gate_into`: magnitude × sequence-RMSNorm(score) rows,
//!   scale ≡ 1.0). The steering table must exist before layer L+1 runs, so
//!   the CPU harness builds it from a first (selection) pass and re-runs —
//!   two honest passes, asserted to select identically; the production
//!   wiring fills the table at the prune point (T2.4's capture lane).
//!   The v0 overlay law: bias AND scale BOTH armed (the diagnostic is
//!   `sel`; upstream measured the drift when only one is cleared — this
//!   harness keeps the two endpoints, not the mixed postures).
//!   Bucket = None: no pads, so the PAD_KEY_BIAS extension is inert here.
//!
//! # The GOAT gate
//!
//! **derived ≥ random-K on EVERY cell** — the retrieval axis (≥) and the
//! ppl axis (≤) at every K in both postures. The verdict prints per axis;
//! a FAIL is recorded honestly (it refuses the lane's promotion, it does
//! not refuse this process — the numbers ARE the record).
//!
//! # Box state
//!
//! Wall seconds per cell + the total, loadavg + free RAM in the header, a
//! loud refusal on a zero wall (the box-state law). OPT-IN,
//! measurement-only: the lane stays behind `tokengate`, never default.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, PrunedPrefillPlan, forward_qwen_deltanet,
    forward_qwen_deltanet_pruned, prefill_qwen_deltanet_into, prefill_qwen_deltanet_pruned_into,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokengate::{
    PrunedDecodeGate, PrunedDecodeLayout, build_decode_gate_into, derived_magnitudes_into,
    random_scores_into, scorer_attention_mass, scorer_hidden_norm,
};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::transformer::KeyBiasLayout;
use riir_infer_core::types::DeltaNetLayerType;

/// The gate's pre-registered retention threshold (nats/token over the
/// needle's answer code span) — reused verbatim so the three instruments'
/// numbers stay comparable.
const EPS_NATS: f32 = 1.0;
/// The T3.2 frozen sequence-RMS epsilon (upstream `_RMS_EPS`).
const GATE_RMS_EPS: f64 = 1e-6;
/// The lane's scaffold (`kv_evict` sink convention).
const N_SINK: usize = 4;

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    context: usize,
    needles: usize,
    k_fracs: Vec<f32>,
    seed: u64,
    fastkv_window: usize,
    prune_layer: Option<usize>,
    ppl_tokens: usize,
    rand_reps: usize,
    posture: Posture,
    /// The selector arms to run (subset of random,derived,fastkv,hiddennorm;
    /// default the four-arm set). `hiddennorm` is the bench-888 Finding-4
    /// ablation (the mixer-agnostic arm at the attention layer L) — an
    /// ablation-only run that excludes derived/random skips the GOAT verdict.
    arms: Vec<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum Posture {
    Sel,
    Gate,
    Both,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf"),
        corpus: PathBuf::from("../riir-train/data/chat_probe"),
        context: 2048,
        needles: 4,
        // Log-uniform K sweep (powers of two halving + the 3/4 cell the
        // T2.5 pin measured at).
        k_fracs: vec![0.0625, 0.125, 0.25, 0.5, 0.75],
        seed: 622,
        fastkv_window: 32,
        prune_layer: None,
        ppl_tokens: 128,
        rand_reps: 3,
        posture: Posture::Both,
        arms: vec![
            "random".to_string(),
            "derived".to_string(),
            "fastkv".to_string(),
        ],
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut val = |name: &str| -> Result<String> {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("--{name} needs a value"))
        };
        match f.as_str() {
            "--model" => a.model = PathBuf::from(val("model")?),
            "--corpus" => a.corpus = PathBuf::from(val("corpus")?),
            "--context" => a.context = val("context")?.parse()?,
            "--needles" => a.needles = val("needles")?.parse()?,
            "--k-fracs" => {
                a.k_fracs = val("k-fracs")?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<std::result::Result<Vec<f32>, _>>()?;
            }
            "--seed" => a.seed = val("seed")?.parse()?,
            "--fastkv-window" => a.fastkv_window = val("fastkv-window")?.parse()?,
            "--prune-layer" => a.prune_layer = Some(val("prune-layer")?.parse()?),
            "--ppl-tokens" => a.ppl_tokens = val("ppl-tokens")?.parse()?,
            "--rand-reps" => a.rand_reps = val("rand-reps")?.parse()?,
            "--arms" => {
                a.arms = val("arms")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
            }
            "--posture" => {
                a.posture = match val("posture")?.as_str() {
                    "sel" => Posture::Sel,
                    "gate" => Posture::Gate,
                    "both" => Posture::Both,
                    other => bail!("--posture: sel|gate|both, got {other}"),
                }
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if a.k_fracs.is_empty() {
        bail!("--k-fracs needs at least one fraction");
    }
    if a.rand_reps == 0 {
        bail!("--rand-reps needs at least one seed");
    }
    if a.arms.is_empty() {
        bail!("--arms needs at least one arm");
    }
    for arm in &a.arms {
        if !matches!(arm.as_str(), "random" | "derived" | "fastkv" | "hiddennorm") {
            bail!("--arms: random|derived|fastkv|hiddennorm, got {arm}");
        }
    }
    if a.ppl_tokens == 0 {
        bail!("--ppl-tokens needs at least one continuation token");
    }
    Ok(a)
}

fn needle_sentence(i: usize, code: u64) -> String {
    format!("Registry entry {i}: the access phrase is QWARF{code:04}, repeat, QWARF{code:04}. ")
}

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

/// NLL of `tok_id` under the categorical `logits` (nats) — the gate's form.
fn nats_log_softmax(logits: &[f32], tok_id: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - max).exp();
    }
    -(logits[tok_id] - (max + sum.ln()))
}

fn free_ram_pct() -> f32 {
    let out = std::process::Command::new("memory_pressure")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok());
    let Some(text) = out else { return f32::NAN };
    for line in text.lines() {
        if let Some(pct) = line.strip_prefix("System-wide memory free percentage: ") {
            let v: f32 = pct.trim().trim_end_matches('%').parse().unwrap_or(f32::NAN);
            return v;
        }
    }
    f32::NAN
}

fn loadavg() -> f32 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or_else(|| {
            let out = std::process::Command::new("sysctl")
                .args(["-n", "vm.loadavg"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok());
            out.and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
                .unwrap_or(f32::NAN)
        })
}

/// One cell's both-instrument outcome.
struct CellReport {
    posture: &'static str,
    arm: &'static str,
    frac: f32,
    real_len: usize,
    needle_nll: Vec<f32>,
    /// mean nats/token over the continuation span.
    cont_nats: f32,
    wall_s: f32,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let t_start = Instant::now();

    // ── Model + tokenizer ──
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_weights_gguf(&args.model)
        .with_context(|| format!("load {}", args.model.display()))?;
    println!(
        "# model: {} | layers={} n_embd={} n_head={} n_kv_head={} head_dim={} vocab={} | load {:.1}s",
        args.model.display(),
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
    println!(
        "# box: loadavg={:.2} mem_free_pct={:.0} ctx={} needles={} k_fracs={:?} fastkv_w={} ppl_tokens={} rand_reps={}",
        loadavg(),
        free_ram_pct(),
        args.context,
        args.needles,
        args.k_fracs,
        args.fastkv_window,
        args.ppl_tokens,
        args.rand_reps
    );

    // ── The prune layer: the measured pin (T2.5) — the middle full-attention
    //    layer — unless the caller overrides.
    let attention_layers: Vec<usize> = (0..weights.layers.len())
        .filter(|&l| weights.layer_types[l] != DeltaNetLayerType::DeltaNet)
        .collect();
    if attention_layers.is_empty() {
        bail!("model has no full-attention layers — nothing to prune onto");
    }
    let prune_at = args.prune_layer.unwrap_or_else(|| {
        let mid = attention_layers[attention_layers.len() / 2];
        println!("# prune layer: L={mid} (the T2.5 pin: middle full-attention)");
        mid
    });
    if prune_at + 1 >= weights.layers.len() {
        bail!("prune layer {prune_at} leaves no layer above it");
    }
    if weights.layer_types[prune_at] == DeltaNetLayerType::DeltaNet {
        bail!("prune layer {prune_at} is GDN — the mass/fastkv taps need attention there");
    }

    // ── Rig assembly (the T2.5 shape + the ppl continuation) ──
    config.block_size = args.context + 8 * args.needles * 16 + 256;
    let rope_freq = RopeFreqTable::new(
        config.rope_theta,
        riir_infer_core::deltanet::effective_rotary_dim(&config),
    );

    let filler_text = load_corpus_text(&args.corpus)
        .with_context(|| format!("load corpus {}", args.corpus.display()))?;
    let filler_tokens = tok.encode(&filler_text);
    if filler_tokens.len() < args.context + args.ppl_tokens {
        bail!(
            "corpus too short: {} tokens < context {} + ppl {}",
            filler_tokens.len(),
            args.context,
            args.ppl_tokens
        );
    }

    let question = "\n\nQuestion: What are the access phrases of all the registry entries above? Answer with each entry number and its phrase.\nAnswer:";
    let question_toks = tok.encode(question);
    let mut needle_sent_toks = Vec::with_capacity(args.needles);
    let mut answer = String::new();
    for i in 0..args.needles {
        let code = 1000 + (i as u64) * 977;
        needle_sent_toks.push(tok.encode(&needle_sentence(i, code)));
        answer.push_str(&format!(" Entry {i}: QWARF{code:04}."));
    }
    let answer_toks = tok.encode(&answer);
    let mut answer_code_spans = Vec::with_capacity(args.needles);
    for i in 0..args.needles {
        let code = 1000 + (i as u64) * 977;
        let phrase = tok.encode(&format!(" QWARF{code:04}"));
        answer_code_spans.push(find_subspan(&answer_toks, &phrase)?);
    }

    // Splice needles at even depths (direct by needle index — strictly
    // increasing; the gate bin's slot indirection collides when the slot
    // count equals the needle count).
    let needle_budget: usize = needle_sent_toks.iter().map(|s| s.len()).sum();
    let usable = args
        .context
        .saturating_sub(question_toks.len() + needle_budget + 64);
    let mut splices: Vec<(usize, usize)> = Vec::with_capacity(args.needles);
    for i in 0..args.needles {
        let depth = usable * (i + 1) / (args.needles + 1) + 8;
        splices.push((depth, i));
    }
    splices.sort_by_key(|(d, _)| *d);
    let mut prompt: Vec<usize> = Vec::with_capacity(args.context + 64);
    let mut prompt_code_spans = vec![(0usize, 0usize); args.needles];
    let mut filler_cursor = 0usize;
    for (depth, i) in &splices {
        while prompt.len() < *depth && filler_cursor < filler_tokens.len() {
            prompt.push(filler_tokens[filler_cursor]);
            filler_cursor += 1;
        }
        let sent = &needle_sent_toks[*i];
        let start = prompt.len();
        prompt.extend_from_slice(sent);
        let phrase = tok.encode(&format!(" QWARF{:04}", 1000 + (*i as u64) * 977));
        let span = find_subspan(sent, &phrase)?;
        prompt_code_spans[*i] = (start + span.0, start + span.1);
    }
    while prompt.len() < args.context && filler_cursor < filler_tokens.len() {
        prompt.push(filler_tokens[filler_cursor]);
        filler_cursor += 1;
    }
    for (i, span) in prompt_code_spans.iter().enumerate() {
        if span.1 >= prompt.len() {
            bail!("needle {i} truncated by the context cap — corpus/context mismatch");
        }
    }
    // The ppl continuation: the corpus tokens AFTER the prompt's filler (the
    // decode continuation reads them teacher-forced, context = prompt +
    // answer).
    let cont_toks: Vec<usize> =
        filler_tokens[filler_cursor..filler_cursor + args.ppl_tokens].to_vec();
    // Decode span indices (inside the combined teacher-forced sequence).
    let answer_len = answer_toks.len();
    let cont_span = (answer_len, answer_len + cont_toks.len());
    println!(
        "# prompt: {} tokens | answer {} | cont {} | needle depths {:?}",
        prompt.len(),
        answer_len,
        cont_toks.len(),
        splices.iter().map(|(d, _)| *d).collect::<Vec<_>>()
    );

    // ── The derived magnitude table (T3.2), once ──
    let mut magnitudes = vec![0.0f32; weights.layers.len()];
    derived_magnitudes_into(&mut magnitudes, &weights);
    println!(
        "# magnitudes: L={prune_at} → {:.3} (mid-attention table, first 8: {:?})",
        magnitudes[prune_at],
        &magnitudes[..magnitudes.len().min(8)]
    );

    // ── The teacher-forced decode over answer ++ continuation; returns the
    //    per-needle NLLs + the continuation mean nats/token. Shared by the
    //    ceiling row and every cell (the pruned variant takes the gate).
    enum DecodeGate<'a> {
        None,
        Armed(&'a PrunedDecodeGate<'a>),
    }
    #[allow(clippy::too_many_arguments)] // the wire-window shape (mirror of the kernel fns')
    fn teacher_forced(
        weights: &riir_infer_core::deltanet::weights::QwenDeltaNetWeights,
        config: &riir_infer_core::types::Config,
        cache: &mut HybridCache,
        scratch: &mut HybridForwardScratch,
        rope_freq: &riir_infer_core::rope::RopeFreqTable,
        layout: &PrunedDecodeLayout,
        first_logits: &[f32],
        seq: &[usize],
        needle_spans: &[(usize, (usize, usize))],
        cont_span: (usize, usize),
        gate: DecodeGate<'_>,
    ) -> Result<(Vec<f32>, f32)> {
        let v = config.vocab_size;
        let mut x = vec![0.0f32; v.max(config.n_embd)];
        let mut needle_nll = vec![0.0f32; needle_spans.len()];
        let mut needle_cnt = vec![0u32; needle_spans.len()];
        let mut cont_sum = 0.0f64;
        let mut prev_logits = first_logits.to_vec();
        for (t, &tok_id) in seq.iter().enumerate() {
            let nll = nats_log_softmax(&prev_logits, tok_id);
            for (i, (_, span)) in needle_spans.iter().enumerate() {
                if t >= span.0 && t < span.1 {
                    needle_nll[i] += nll;
                    needle_cnt[i] += 1;
                }
            }
            if t >= cont_span.0 && t < cont_span.1 {
                cont_sum += nll as f64;
            }
            let out = match gate {
                DecodeGate::None => forward_qwen_deltanet(
                    &mut x,
                    weights,
                    cache,
                    tok_id,
                    layout.full_rows + t,
                    config,
                    scratch,
                    rope_freq,
                ),
                DecodeGate::Armed(g) => forward_qwen_deltanet_pruned(
                    &mut x,
                    weights,
                    cache,
                    tok_id,
                    layout,
                    t,
                    Some(g),
                    config,
                    scratch,
                    rope_freq,
                ),
            };
            prev_logits.copy_from_slice(&out[..v]);
        }
        for (n, c) in needle_nll.iter_mut().zip(&needle_cnt) {
            if *c == 0 {
                bail!("empty needle answer span — rig bug");
            }
            *n /= *c as f32;
        }
        let cont_nats = (cont_sum / (cont_span.1 - cont_span.0) as f64) as f32;
        Ok((needle_nll, cont_nats))
    }

    // ── The ceiling row: the full prefill + the unpruned decode (once) ──
    let t_full = Instant::now();
    let (full_needle, full_cont) = {
        let layer_types = weights.layer_types.clone();
        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut pctx = PrefillContext::new(&config, prompt.len());
        let v = config.vocab_size;
        let mut logits = vec![0.0f32; v];
        prefill_qwen_deltanet_into(
            &weights,
            &config,
            &mut cache,
            &prompt,
            &mut scratch,
            &rope_freq,
            &mut pctx,
            &mut logits,
            None,
        );
        let layout = PrunedDecodeLayout {
            prune_at_layer: prune_at,
            full_rows: prompt.len(),
            kept_rows: prompt.len(),
        };
        let mut seq = answer_toks.clone();
        seq.extend_from_slice(&cont_toks);
        let spans: Vec<(usize, (usize, usize))> =
            answer_code_spans.iter().copied().enumerate().collect();
        teacher_forced(
            &weights,
            &config,
            &mut cache,
            &mut scratch,
            &rope_freq,
            &layout,
            &logits,
            &seq,
            &spans,
            cont_span,
            DecodeGate::None,
        )?
    };
    let full_ppl = full_cont.exp();
    println!(
        "# [ceiling] {:.1}s | needle NLL {full_needle:?} | ppl {full_ppl:.3}",
        t_full.elapsed().as_secs_f32()
    );

    // ── The cell grid ──
    let prunable = prompt.len().saturating_sub(N_SINK + 1).max(1);
    let postures: &[&str] = match args.posture {
        Posture::Sel => &["sel"],
        Posture::Gate => &["gate"],
        Posture::Both => &["sel", "gate"],
    };
    let mut cells: Vec<CellReport> = Vec::new();

    // One arm/K/posture cell. `rand_seed` arms the random arm's scorer
    // (re-created identically for the gate posture's second pass — the
    // selection must not move).
    let run_cell = |posture: &'static str,
                    arm: &'static str,
                    frac: f32,
                    keep: usize,
                    rand_seed: u64|
     -> Result<CellReport> {
        let t_cell = Instant::now();

        enum Selector {
            Random,
            Derived,
            FastKv,
            HiddenNorm,
        }
        let selector = match arm {
            "random" => Selector::Random,
            "derived" => Selector::Derived,
            "fastkv" => Selector::FastKv,
            "hiddennorm" => Selector::HiddenNorm,
            other => bail!("unknown arm {other}"),
        };
        let tap_window = match selector {
            Selector::FastKv => Some(args.fastkv_window),
            _ => None,
        };
        // The scorer + tap arm per selector: the derived/fastkv arms copy
        // whichever statistic the tap wrote (the mass sum or the fastkv
        // max-pool — the arm IS the tap kind); random writes its own scores.
        type ScorerBox<'x> = Box<dyn FnMut(&[f32], Option<&[f32]>, &mut [f32]) + 'x>;
        let make_scorer = |seed: u64| -> ScorerBox<'_> {
            match selector {
                Selector::Random => {
                    Box::new(move |_h: &[f32], _m: Option<&[f32]>, s: &mut [f32]| {
                        random_scores_into(s, N_SINK, keep, seed)
                    })
                }
                Selector::HiddenNorm => {
                    Box::new(|_h: &[f32], _m: Option<&[f32]>, s: &mut [f32]| {
                        scorer_hidden_norm(_h, _m, s)
                    })
                }
                _ => Box::new(|_h: &[f32], m: Option<&[f32]>, s: &mut [f32]| {
                    scorer_attention_mass(_h, m, s)
                }),
            }
        };

        // One bias-free-or-steered pruned-prefill run; returns the cache
        // (post-prefill), the prefill logits, and the plan.
        let run_prefill = |key_bias: Option<(&[f32], KeyBiasLayout)>,
                           seed: u64|
         -> Result<(
            HybridCache,
            HybridForwardScratch,
            Vec<f32>,
            PrunedPrefillPlan,
        )> {
            let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
            let mut scratch = HybridForwardScratch::new(&config);
            let mut pctx = PrefillContext::new(&config, prompt.len());
            let mut logits = vec![0.0f32; config.vocab_size];
            let mut plan = PrunedPrefillPlan::new(prune_at, N_SINK, keep);
            let mut tap_buf = if matches!(selector, Selector::Random) {
                Vec::new()
            } else {
                vec![0.0f32; prompt.len()]
            };
            let mass_tap = (!tap_buf.is_empty()).then(|| &mut tap_buf[..]);
            let mut scorer = make_scorer(seed);
            prefill_qwen_deltanet_pruned_into(
                &weights,
                &config,
                &mut cache,
                &prompt,
                &mut scratch,
                &rope_freq,
                &mut pctx,
                &mut logits,
                key_bias,
                &mut plan,
                mass_tap,
                tap_window,
                &mut scorer,
            );
            Ok((cache, scratch, logits, plan))
        };

        // The gate posture's overlay tables, built from the selection pass.
        // head_stride covers BOTH consumers' reads: the prefill at layers <= L
        // reads rows up to t_n = prompt_len (zero rows there — inert adds),
        // the decode reads kept + steps. Bucket = None: no pads, so the
        // PAD_KEY_BIAS extension is inert in this harness.
        let total_steps = answer_len + cont_toks.len();
        let head_stride = prompt.len() + total_steps;
        let n_head = config.n_head;
        let n_layers = weights.layers.len();
        let mut gate_bias = vec![0.0f32; n_layers * n_head * head_stride];
        let mut gate_scale = vec![1.0f32; n_layers * n_head];

        let (mut cache, mut scratch, logits, plan) = if posture == "gate" {
            // Pass 1: the selection run (the steering table does not exist
            // yet — bias-free everywhere).
            let (_, _, _, plan1) = run_prefill(None, rand_seed)?;
            build_decode_gate_into(
                &mut gate_bias,
                Some(&mut gate_scale),
                &plan1.scores,
                &plan1.kept,
                N_SINK,
                &magnitudes,
                &weights.layer_types,
                prune_at,
                n_head,
                head_stride,
                GATE_RMS_EPS,
            );
            // Pass 2: the steering-armed prefill (the same scorer/seed — the
            // selection must reproduce bit-for-bit; the tap is bias-blind by
            // construction, and the arm asserts it rather than assuming).
            let (cache, scratch, logits, plan2) = run_prefill(
                Some((&gate_bias[..], KeyBiasLayout { head_stride })),
                rand_seed,
            )?;
            if plan2.kept != plan1.kept {
                bail!(
                    "pass 2 selected a different kept set ({} vs {} rows) — the steering must not move selection",
                    plan2.kept.len(),
                    plan1.kept.len()
                );
            }
            (cache, scratch, logits, plan2)
        } else {
            let (cache, scratch, logits, plan) = run_prefill(None, rand_seed)?;
            (cache, scratch, logits, plan)
        };

        // The teacher-forced decode carries both instruments.
        let layout = PrunedDecodeLayout::from_plan(prompt.len(), &plan);
        let gate = PrunedDecodeGate {
            bias: &gate_bias,
            head_stride,
            scale: Some(&gate_scale),
        };
        let mut seq = answer_toks.clone();
        seq.extend_from_slice(&cont_toks);
        let spans: Vec<(usize, (usize, usize))> =
            answer_code_spans.iter().copied().enumerate().collect();
        let (needle_nll, cont_nats) = teacher_forced(
            &weights,
            &config,
            &mut cache,
            &mut scratch,
            &rope_freq,
            &layout,
            &logits,
            &seq,
            &spans,
            cont_span,
            if posture == "gate" {
                DecodeGate::Armed(&gate)
            } else {
                DecodeGate::None
            },
        )?;

        Ok(CellReport {
            posture,
            arm,
            frac,
            real_len: plan.real_len,
            needle_nll,
            cont_nats,
            wall_s: t_cell.elapsed().as_secs_f32(),
        })
    };

    // The random floor's per-axis bests: (posture, frac) → (best retrieval,
    // best ppl nats over the reps) — the GOAT comparison reads the floor at
    // its per-axis BEST (the maximally defensible reading: if derived still
    // wins, the gate is the stronger for it). The table shows the best-ppl
    // rep; the GOAT detail discloses the per-axis bests.
    // (f32 keys via to_bits — f32 itself is not Ord.)
    let mut rand_axis_best: std::collections::BTreeMap<(&'static str, u32), (f32, f32)> =
        Default::default();
    let retrieval_of = |c: &CellReport| -> f32 {
        c.needle_nll
            .iter()
            .zip(&full_needle)
            .filter(|&(n, f)| n - f <= EPS_NATS)
            .count() as f32
            / c.needle_nll.len() as f32
    };

    for &posture in postures {
        for &frac in &args.k_fracs {
            let keep = ((prunable as f32) * frac).round() as usize;
            // ── random: best-of-R seeds (the most favorable floor) — only
            // when the arms list configures it ──
            if args.arms.iter().any(|a| a == "random") {
                let mut best: Option<CellReport> = None;
                let mut best_retrieval = 0.0f32;
                let mut best_ppl_nats = f32::INFINITY;
                for rep in 0..args.rand_reps {
                    let seed = args
                        .seed
                        .wrapping_add(0x9E37_79B9_7F4A_7C15u64)
                        .wrapping_mul((rep as u64) + 1);
                    let c = run_cell(posture, "random", frac, keep, seed)?;
                    best_retrieval = best_retrieval.max(retrieval_of(&c));
                    best_ppl_nats = best_ppl_nats.min(c.cont_nats);
                    let score = c.cont_nats; // the table's rep: lowest ppl wins
                    if best.as_ref().map(|b| score < b.cont_nats).unwrap_or(true) {
                        best = Some(c);
                    }
                }
                rand_axis_best.insert((posture, frac.to_bits()), (best_retrieval, best_ppl_nats));
                let c = best.expect("at least one random rep");
                eprintln!(
                    "# [{posture} random K={frac:.4}] {:.1}s | kept {} | ppl {:.3} | NLL {:?}",
                    c.wall_s,
                    c.real_len,
                    c.cont_nats.exp(),
                    c.needle_nll
                );
                cells.push(c);
            }

            // ── the remaining configured arms ──
            for arm in args.arms.iter().filter(|a| a.as_str() != "random") {
                let arm: &'static str = match arm.as_str() {
                    "derived" => "derived",
                    "fastkv" => "fastkv",
                    "hiddennorm" => "hiddennorm",
                    _ => continue,
                };
                let c = run_cell(posture, arm, frac, keep, 0)?;
                eprintln!(
                    "# [{posture} {arm} K={frac:.4}] {:.1}s | kept {} | ppl {:.3} | NLL {:?}",
                    c.wall_s,
                    c.real_len,
                    c.cont_nats.exp(),
                    c.needle_nll
                );
                cells.push(c);
            }
        }
    }

    // ── The report ──
    let needle_names: Vec<String> = (0..args.needles).map(|i| format!("n{i}")).collect();
    for &posture in postures {
        println!(
            "\n## posture = {posture} (ceiling: needle NLL {full_needle:?}, ppl {full_ppl:.3})"
        );
        println!(
            "| {:<18} | {:>5} | {:>8} | {:>9} | {:>8} | {:>8} | {:>7} |",
            "cell", "kept", "retriev", "meanΔnat", "ppl", "Δppl", "wall_s"
        );
        println!(
            "|{:-<20}|{:-<7}|{:-<10}|{:-<11}|{:-<10}|{:-<8}|{:-<9}|",
            "", "", "", "", "", "", ""
        );
        for c in cells.iter().filter(|c| c.posture == posture) {
            let mut acc = 0.0f32;
            let mut retained = 0usize;
            for (n, f) in c.needle_nll.iter().zip(&full_needle) {
                let d = n - f;
                acc += d;
                if d <= EPS_NATS {
                    retained += 1;
                }
            }
            let mean_d = acc / c.needle_nll.len() as f32;
            let retrieval = retained as f32 / c.needle_nll.len() as f32;
            let ppl = c.cont_nats.exp();
            println!(
                "| {:<18} | {:>5} | {:>9.3} | {:>9.3} | {:>8.3} | {:>+8.3} | {:>7.1} |",
                format!("{} K={:.4}", c.arm, c.frac),
                c.real_len,
                retrieval,
                mean_d,
                ppl,
                ppl - full_ppl,
                c.wall_s
            );
        }
    }

    // The GOAT verdict, per posture and per axis: derived ≥ random on
    // retrieval AND derived ≤ random on ppl, at EVERY K. An ablation-only
    // run (no derived/random in --arms) skips the verdict — there is no
    // pair to compare.
    let goat_evaluable =
        args.arms.iter().any(|a| a == "derived") && args.arms.iter().any(|a| a == "random");
    if goat_evaluable {
        println!("\n# GOAT gate: derived ≥ random-K on every cell");
        let mut goat_all = true;
        for &posture in postures {
            let mut goat = true;
            let mut detail: Vec<String> = Vec::new();
            for &frac in &args.k_fracs {
                let pick = |arm: &str| -> Option<&CellReport> {
                    cells
                        .iter()
                        .find(|c| c.posture == posture && c.arm == arm && c.frac == frac)
                };
                let d = match pick("derived") {
                    Some(d) => d,
                    None => {
                        goat = false;
                        detail.push(format!("K={frac:.4}: MISSING CELL"));
                        continue;
                    }
                };
                // The floor reads its per-axis BEST over the random reps.
                let (r_ret, r_ppl_nats) = match rand_axis_best.get(&(posture, frac.to_bits())) {
                    Some(v) => *v,
                    None => {
                        goat = false;
                        detail.push(format!("K={frac:.4}: MISSING FLOOR"));
                        continue;
                    }
                };
                let d_ret = retrieval_of(d);
                let ret_ok = d_ret >= r_ret;
                let ppl_ok = d.cont_nats <= r_ppl_nats;
                if !(ret_ok && ppl_ok) {
                    goat = false;
                }
                detail.push(format!(
                    "K={frac:.4}: retrieval {d_ret:.3}vs{r_ret:.3} {} ppl {:.4}vs{:.4} {}",
                    if ret_ok { "≥✓" } else { "≥✗" },
                    d.cont_nats.exp(),
                    r_ppl_nats.exp(),
                    if ppl_ok { "≤✓" } else { "≤✗" },
                ));
            }
            goat_all &= goat;
            println!("# {posture}: {}", if goat { "PASS" } else { "FAIL" });
            for d in &detail {
                println!("#   {d}");
            }
        }
        println!(
            "# GOAT VERDICT: {}",
            if goat_all {
                "PASS — derived ≥ random-K on every cell (both axes, both postures)"
            } else {
                "FAIL — the lane's promotion is refused; the numbers above are the record"
            }
        );
    } else {
        println!(
            "\n# GOAT gate: SKIPPED — ablation run (arms = {:?}); no derived/random pair to compare",
            args.arms
        );
    }

    // The FastKV honesty column: per cell, fastkv vs derived, plus the
    // Pareto-vs-ceiling view (what each arm gives up relative to no prune).
    println!("\n# FastKV column (reported, non-gating) vs derived");
    for &posture in postures {
        for &frac in &args.k_fracs {
            let pick = |arm: &str| {
                cells
                    .iter()
                    .find(|c| c.posture == posture && c.arm == arm && c.frac == frac)
            };
            if let (Some(d), Some(f)) = (pick("derived"), pick("fastkv")) {
                let d_ret = d
                    .needle_nll
                    .iter()
                    .zip(&full_needle)
                    .filter(|&(n, f)| n - f <= EPS_NATS)
                    .count();
                let f_ret = f
                    .needle_nll
                    .iter()
                    .zip(&full_needle)
                    .filter(|&(n, f)| n - f <= EPS_NATS)
                    .count();
                let winner = if f.cont_nats < d.cont_nats {
                    "fastkv"
                } else if d.cont_nats < f.cont_nats {
                    "derived"
                } else {
                    "tie"
                };
                println!(
                    "# {posture} K={frac:.4}: needle retained derived {d_ret}/{} fastkv {f_ret}/{} | ppl derived {:.4} fastkv {:.4} → {winner} | Δppl-vs-ceiling derived {:+.3} fastkv {:+.3}",
                    args.needles,
                    args.needles,
                    d.cont_nats.exp(),
                    f.cont_nats.exp(),
                    d.cont_nats.exp() - full_ppl,
                    f.cont_nats.exp() - full_ppl,
                );
            }
        }
    }

    let total_s = t_start.elapsed().as_secs_f32();
    let any_zero_wall = cells
        .iter()
        .any(|c| c.wall_s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater));
    if any_zero_wall || total_s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        bail!("missing wall times — refusing to summarize (box-state law)");
    }
    println!(
        "# total wall {total_s:.1}s | seed {} | needle spans {:?}",
        args.seed, needle_names
    );
    Ok(())
}
