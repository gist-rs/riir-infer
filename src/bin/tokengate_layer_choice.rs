#![cfg(all(feature = "tokengate", feature = "deltanet_inference"))]

//! Plan 622 T2.5 — the layer-choice measurement: acc-vs-K at
//! `L ∈ {0, 1, mid}` on ONE suite, pinning the prune-layer constant for OUR
//! models (upstream used L=1 dense / L=3 hybrid; measured here, never
//! copied). One suite = the NIAH rig (`needle_eviction_gate`'s, minus the
//! hub/wall distractor regimes): haystack filler with needle sentences
//! spliced at even depths + question + teacher-forced answer, the metric
//! being the gate's pre-registered per-needle NLL readout
//! (`delta_i = NLL_i(arm) − NLL_i(full)`, retained iff ≤ 1.0 nat/token,
//! retrieval = retained/K).
//!
//! # Arms
//!
//! The full-cache reference (no prune) once, then every (L, K) cell: the
//! pruned prefill (`prefill_qwen_deltanet_pruned_into`) + the pruned decode
//! (`forward_qwen_deltanet_pruned`). SELECTION-ALONE posture: the decode
//! gate is `None` — no bias rows, no scale (the steering overlay's own
//! L-sensitivity is T3.4's A/B, a separate axis; this measurement isolates
//! which L keeps the retrieval-bearing tokens).
//!
//! # The scorer at each L
//!
//! L=1 and L=mid run arm (a) attention-mass (the default; `mid` is the
//! middle FULL-ATTENTION layer — a GDN mixer has no mass). L=0 runs the
//! mixer-agnostic arm (b) hidden-norm when layer 0 is GDN (upstream's
//! "gate at L0 too weak" datum, measured not assumed) and arm (a) when it
//! is attention. The arm that ran is printed per cell.
//!
//! # Box state
//!
//! Wall seconds per arm are printed and the summary refuses without them;
//! loadavg + free RAM ride the header (the Feature-Flag-Discipline law —
//! box state is part of any number quoted beside this table). OPT-IN,
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
use riir_infer_core::tokengate::{PrunedDecodeLayout, ScorerArm, run_scorer};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::DeltaNetLayerType;

/// The gate's pre-registered retention threshold (nats/token over the
/// needle's answer code span) — reused verbatim so the two instruments'
/// numbers stay comparable.
const EPS_NATS: f32 = 1.0;

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    context: usize,
    needles: usize,
    k_fracs: Vec<f32>,
    seed: u64,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf"),
        corpus: PathBuf::from("../riir-train/data/chat_probe"),
        context: 2048,
        needles: 4,
        k_fracs: vec![0.5, 0.75],
        seed: 622,
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
            other => bail!("unknown arg {other}"),
        }
    }
    if a.k_fracs.is_empty() {
        bail!("--k-fracs needs at least one fraction");
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

fn free_ram_gb() -> f32 {
    // macOS: sysctl hw.memsize is total; the wired/free read needs vm_stat.
    // proportion_free from `memory_pressure -Q`'s free percentage when
    // available, else NaN (the header prints what it can prove).
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

/// One (scorer, L, K) cell's outcome over the needle set.
struct CellReport {
    label: String,
    arm: &'static str,
    prune_at: usize,
    keep_prunable: usize,
    prunable: usize,
    real_len: usize,
    needle_nll: Vec<f32>,
    wall_s: f32,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let t_start = Instant::now();

    // ── Model + tokenizer (the gate bin's lane) ──
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
        "# box: loadavg={:.2} mem_free_pct={:.0} ctx={} needles={} k_fracs={:?}",
        loadavg(),
        free_ram_gb(),
        args.context,
        args.needles,
        args.k_fracs
    );

    // ── Layer plan: L ∈ {0, 1, mid} with the arm each placement can run ──
    let attention_at = |l: usize| weights.layer_types[l] != DeltaNetLayerType::DeltaNet;
    let attention_layers: Vec<usize> = (0..weights.layers.len())
        .filter(|&l| attention_at(l))
        .collect();
    if attention_layers.is_empty() {
        bail!("model has no full-attention layers — nothing to prune onto");
    }
    let mid_attn = attention_layers[attention_layers.len() / 2];
    let mut placements: Vec<(usize, ScorerArm)> = Vec::with_capacity(3);
    placements.push((
        0,
        if attention_at(0) {
            ScorerArm::AttentionMass
        } else {
            ScorerArm::HiddenNorm
        },
    ));
    // L=1 only when it differs from the placements already present.
    if weights.layers.len() > 1 {
        placements.push((
            1,
            if attention_at(1) {
                ScorerArm::AttentionMass
            } else {
                ScorerArm::HiddenNorm
            },
        ));
    }
    if mid_attn > 1 {
        placements.push((mid_attn, ScorerArm::AttentionMass));
    }
    for (l, _) in &placements {
        if *l + 1 >= weights.layers.len() {
            bail!("placement L={l} leaves no layer above it");
        }
    }
    print!("# placements:");
    for (l, arm) in &placements {
        let arm_name = match arm {
            ScorerArm::AttentionMass => "mass",
            ScorerArm::HiddenNorm => "hiddennorm",
            ScorerArm::AllOnes => "allones",
        };
        print!(" L={l}({arm_name})");
    }
    println!();

    // ── Rig assembly (the gate bin's shape, no hub/wall regimes) ──
    config.block_size = args.context + 8 * args.needles * 16 + 256;
    let rope_freq = RopeFreqTable::new(
        config.rope_theta,
        riir_infer_core::deltanet::effective_rotary_dim(&config),
    );

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

    // Needle sentences at even depths; the answer carries every code once.
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

    // Splice needles at even depths of the usable filler budget (direct by
    // needle index — strictly increasing; the gate bin's slot indirection
    // collides when the slot count equals the needle count).
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
    // A splice never lands close enough to the tail to be truncated, but the
    // spans are checked, not assumed.
    for (i, span) in prompt_code_spans.iter().enumerate() {
        if span.1 >= prompt.len() {
            bail!("needle {i} truncated by the context cap — corpus/context mismatch");
        }
    }
    println!(
        "# prompt: {} tokens | answer {} tokens | needle depths {:?}",
        prompt.len(),
        answer_toks.len(),
        splices.iter().map(|(d, _)| *d).collect::<Vec<_>>()
    );

    // ── The full-cache reference (once) ──
    let t_arm = Instant::now();
    let full_nll = {
        let layer_types = weights.layer_types.clone();
        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut pctx = PrefillContext::new(&config, prompt.len());
        let v = config.vocab_size;
        let mut logits = vec![0.0f32; v];
        let mut x = vec![0.0f32; v.max(config.n_embd)];
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
        // Teacher-forced answer decode (the incumbent step).
        let mut needle_nll = vec![0.0f32; args.needles];
        let mut needle_cnt = vec![0u32; args.needles];
        let mut prev_logits = logits;
        for (seq_len, (t, &tok_id)) in (prompt.len()..).zip(answer_toks.iter().enumerate()) {
            let nll = nats_log_softmax(&prev_logits, tok_id);
            for (i, span) in answer_code_spans.iter().enumerate() {
                if t >= span.0 && t < span.1 {
                    needle_nll[i] += nll;
                    needle_cnt[i] += 1;
                }
            }
            let out = forward_qwen_deltanet(
                &mut x,
                &weights,
                &mut cache,
                tok_id,
                seq_len,
                &config,
                &mut scratch,
                &rope_freq,
            );
            prev_logits.copy_from_slice(&out[..v]);
        }
        for i in 0..args.needles {
            if needle_cnt[i] == 0 {
                bail!("needle {i} has an empty answer code span — rig bug");
            }
            needle_nll[i] /= needle_cnt[i] as f32;
        }
        println!(
            "# [full] {:.1}s | needle NLL {:?}",
            t_arm.elapsed().as_secs_f32(),
            needle_nll
        );
        needle_nll
    };

    // ── The (L, K) cells ──
    let prunable = prompt.len().saturating_sub(4 + 1).max(1); // n_sink=4 + last; the plan's scaffold
    let mut cells: Vec<CellReport> = Vec::new();
    for (prune_at, arm) in &placements {
        for &frac in &args.k_fracs {
            let keep = ((prunable as f32) * frac).round() as usize;
            let label = format!("L={prune_at}/K={frac:.2}");
            let t_cell = Instant::now();
            let layer_types = weights.layer_types.clone();
            let mut cache = HybridCache::with_layer_types(&config, &layer_types);
            let mut scratch = HybridForwardScratch::new(&config);
            let mut pctx = PrefillContext::new(&config, prompt.len());
            let v = config.vocab_size;
            let mut logits = vec![0.0f32; v];
            let mut x = vec![0.0f32; v.max(config.n_embd)];
            let mut plan = PrunedPrefillPlan::new(*prune_at, 4, keep);
            // The mass tap arms only for arm (a) (and only at an attention
            // layer — the pruned prefill asserts it).
            let mut mass = vec![0.0f32; prompt.len()];
            let mass_tap = matches!(arm, ScorerArm::AttentionMass).then_some(&mut mass[..]);
            let mut scorer = |_hidden: &[f32], m: Option<&[f32]>, scores: &mut [f32]| {
                run_scorer(*arm, _hidden, m, scores)
            };
            prefill_qwen_deltanet_pruned_into(
                &weights,
                &config,
                &mut cache,
                &prompt,
                &mut scratch,
                &rope_freq,
                &mut pctx,
                &mut logits,
                None,
                &mut plan,
                mass_tap,
                None,
                &mut scorer,
            );
            // Pruned decode, SELECTION-ALONE (gate = None).
            let layout = PrunedDecodeLayout::from_plan(prompt.len(), &plan);
            let mut needle_nll = vec![0.0f32; args.needles];
            let mut needle_cnt = vec![0u32; args.needles];
            let mut prev_logits = logits;
            for (t, &tok_id) in answer_toks.iter().enumerate() {
                let nll = nats_log_softmax(&prev_logits, tok_id);
                for (i, span) in answer_code_spans.iter().enumerate() {
                    if t >= span.0 && t < span.1 {
                        needle_nll[i] += nll;
                        needle_cnt[i] += 1;
                    }
                }
                let out = forward_qwen_deltanet_pruned(
                    &mut x,
                    &weights,
                    &mut cache,
                    tok_id,
                    &layout,
                    t,
                    None,
                    &config,
                    &mut scratch,
                    &rope_freq,
                );
                prev_logits.copy_from_slice(&out[..v]);
            }
            for i in 0..args.needles {
                needle_nll[i] /= needle_cnt[i] as f32;
            }
            let arm_name = match arm {
                ScorerArm::AttentionMass => "mass",
                ScorerArm::HiddenNorm => "hiddennorm",
                ScorerArm::AllOnes => "allones",
            };
            cells.push(CellReport {
                label,
                arm: arm_name,
                prune_at: *prune_at,
                keep_prunable: keep,
                prunable,
                real_len: plan.real_len,
                needle_nll,
                wall_s: t_cell.elapsed().as_secs_f32(),
            });
            eprintln!(
                "# [L={prune_at} K={frac:.2}] {:.1}s | kept {} | NLL {:?}",
                cells.last().unwrap().wall_s,
                plan.real_len,
                cells.last().unwrap().needle_nll
            );
        }
    }

    // ── The table: per cell, delta vs full + the gate metric ──
    println!(
        "\n| {:<16} | {:<10} | {:>6} | {:>8} | {:>10} | {:>9} |",
        "cell", "arm", "kept", "meanΔ", "retrieval", "wall_s"
    );
    println!(
        "|{:-<18}|{:-<12}|{:-<8}|{:-<10}|{:-<12}|{:-<11}|",
        "", "", "", "", "", ""
    );
    let mut by_l: std::collections::BTreeMap<usize, Vec<&CellReport>> = Default::default();
    for c in &cells {
        by_l.entry(c.prune_at).or_default().push(c);
    }
    for c in &cells {
        let mut acc = 0.0f32;
        let mut retained = 0usize;
        for (n, f) in c.needle_nll.iter().zip(&full_nll) {
            let d = n - f;
            acc += d;
            if d <= EPS_NATS {
                retained += 1;
            }
        }
        let mean_d = acc / c.needle_nll.len() as f32;
        let retrieval = retained as f32 / c.needle_nll.len() as f32;
        println!(
            "| {:<16} | {:<10} | {:>6} | {:>8.3} | {:>10.3} | {:>9.1} |",
            c.label, c.arm, c.real_len, mean_d, retrieval, c.wall_s
        );
    }
    // The L-choice readout: per L, the retrieval at its best K.
    println!("\n# layer-choice readout (retrieval at the LARGEST K per L):");
    for (l, cs) in &by_l {
        let best = cs
            .iter()
            .max_by(|a, b| a.keep_prunable.cmp(&b.keep_prunable))
            .unwrap();
        let mut acc = 0.0f32;
        let mut retained = 0usize;
        for (n, f) in best.needle_nll.iter().zip(&full_nll) {
            let d = n - f;
            acc += d;
            if d <= EPS_NATS {
                retained += 1;
            }
        }
        println!(
            "# L={l}: kept {}/{} retrieval {:.3} meanΔ {:.3} (arm {})",
            best.real_len - 4 - 1,
            best.prunable,
            retained as f32 / best.needle_nll.len() as f32,
            acc / best.needle_nll.len() as f32,
            best.arm
        );
    }
    let total_s = t_start.elapsed().as_secs_f32();
    let any_zero_wall = cells
        .iter()
        .any(|c| c.wall_s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater));
    if any_zero_wall || total_s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        bail!("missing wall times — refusing to summarize (box-state law)");
    }
    println!("# total wall {total_s:.1}s | seed {}", args.seed);
    Ok(())
}
