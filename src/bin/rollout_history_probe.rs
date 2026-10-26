//! Issue 1122 T2 — acceptance-rate probe for the cross-call history
//! n-gram drafter on a REAL repeated-prompt workload (Bonsai-27B-PQ2,
//! the league model, on the CPU reference forward).
//!
//! ## What it measures
//!
//! The NgramRL class drafts decode call N from the model's own outputs on
//! calls N-k..N-1 for the SAME prompt. Two workload shapes are measured
//! against the family's K=2 breakeven (α ≥ 0.58, `ngram_drafter.rs`):
//!
//! 1. **Re-roll shape** (the paper's envelope): one prompt, R samples at
//!    temperature — each sample k is measured against samples 1..k-1.
//!    This is the GRPO rollout-loop consumer (riir-train Plan 472) and
//!    any stochastic serving re-roll.
//! 2. **Retry shape** (the serving candidates: refine fixer retries,
//!    harness re-runs): the same problem template with one operand
//!    perturbed — history from the unperturbed prompt's samples drafts
//!    the perturbed prompt's sample.
//!
//! Controls: a greedy ×2 pair MUST be byte-identical (pinned determinism —
//! divergence is a probe bug, refused loud).
//!
//! ## Protocol notes
//!
//! - TRUE acceptance (the Bench-696 lesson): matched against the target's
//!   own token stream, never a corpus proxy.
//! - Every sample runs its OWN fresh cache from the prompt (state never
//!   carries across samples — the history table is the ONLY cross-call
//!   channel, exactly the deployment shape).
//! - Seeds pinned per (prompt, sample) — Issue-809 discipline; the run is
//!   reproducible from the committed log.
//! - Artifacts land in `.raw/issue1122/` (gitignored): prompts.tsv +
//!   samples.jsonl, so the estimator can be re-run offline without the
//!   model (`measure_alpha` over samples.jsonl).
//!
//! Usage (release; the 27B CPU forward is minutes-scale):
//!   cargo run --release --features deltanet_ternary_inference,bonsai2_hadamard \
//!     --bin rollout_history_probe -- --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     [--prompts 4] [--samples 5] [--max-new 96] [--temp 0.7] [--out .raw/issue1122]

use std::path::PathBuf;

use anyhow::Context;
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, QwenDeltaNetTernaryWeights, forward_qwen_deltanet_ternary,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rollout_history::{
    BREAKEVEN_ALPHA_K2, DEFAULT_HISTORY_CAPACITY, DEFAULT_MAX_DRAFT, DEFAULT_MAX_LOOKUP_ORDER,
    PromptHistory, measure_alpha,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;

/// Pinned probe arguments (all defaulted except the gguf; a run is
/// reproducible from this line + the committed issue).
struct Args {
    gguf: PathBuf,
    /// Prompts in the re-roll grid (each gets `--samples` runs).
    prompts: usize,
    /// Samples per prompt in the re-roll grid (incl. the cold-start one).
    samples: usize,
    /// New tokens per sample.
    max_new: usize,
    /// Sampling temperature (0 = greedy; the greedy control always runs).
    temp: f32,
    /// Artifact output dir (created; gitignored).
    out: PathBuf,
    /// Master seed (per-sample seeds derive deterministically).
    seed: u64,
}

fn parse_args() -> anyhow::Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    let mut a = Args {
        gguf: PathBuf::new(),
        prompts: 4,
        samples: 5,
        max_new: 96,
        temp: 0.7,
        out: PathBuf::from(".raw/issue1122"),
        seed: 0x1122,
    };
    let mut i = 1;
    while i < args.len() {
        let name: &str = &args[i];
        let value = || args[i + 1].clone();
        match name {
            "--gguf" => a.gguf = value().into(),
            "--prompts" => a.prompts = value().parse()?,
            "--samples" => a.samples = value().parse()?,
            "--max-new" => a.max_new = value().parse()?,
            "--temp" => a.temp = value().parse()?,
            "--out" => a.out = value().into(),
            "--seed" => a.seed = value().parse()?,
            _ => anyhow::bail!("unknown arg {name} (see the module doc)"),
        }
        i += 2;
    }
    if a.gguf.as_os_str().is_empty() {
        anyhow::bail!("--gguf is required (the Bonsai-27B-PQ2 GGUF)");
    }
    anyhow::ensure!(
        a.prompts > 0 && a.samples >= 2,
        "need prompts > 0, samples >= 2"
    );
    anyhow::ensure!(a.max_new > 4, "max-new must exceed a handful of tokens");
    Ok(a)
}

// ── the qwen3 chat template shape plan-461's collect used (real prompts) ──
const TEMPLATE: &str = "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n";

/// Arithmetic word problems in the plan-461 collect shape; `{a}`/`{b}` are
/// the operands the retry shape perturbs.
const PROBLEM_SEEDS: [(&str, i64, i64); 4] = [
    (
        "Nadia has {a} marbles. She buys {b} more bags with 6 marbles each, then gives 9 away. How many marbles does she have now? End with `Answer: <number>`",
        24,
        3,
    ),
    (
        "A crate holds {a} bottles. {b} full crates are delivered, then 17 bottles break. How many bottles are there? End with `Answer: <number>`",
        12,
        5,
    ),
    (
        "Liam walks {a} meters, then {b} times that distance again, then 15 meters back. How far is he from the start? End with `Answer: <number>`",
        40,
        2,
    ),
    (
        "A printer outputs {a} pages per minute for {b} minutes, then jams for 4 minutes. How many pages did it print? End with `Answer: <number>`",
        18,
        7,
    ),
];

/// Splitmix64 — a tiny seeded RNG for the probe's sampling draws (pinned
/// seeds; no global RNG state, the Issue-809 discipline).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform [0, 1) in f64.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Temperature sample from raw logits: max-subtracted exp with a
/// cumulative draw (deterministic given the RNG state; temp→0 walks to
/// argmax through the same code path, first-index tie rule).
fn sample_token(logits: &[f32], temp: f32, rng: &mut SplitMix64) -> usize {
    if temp <= 1e-4 {
        let mut best = 0usize;
        for (i, l) in logits.iter().enumerate() {
            if *l > logits[best] {
                best = i;
            }
        }
        return best;
    }
    let inv = 1.0_f64 / f64::from(temp);
    let mut max = f32::NEG_INFINITY;
    for l in logits {
        if *l > max {
            max = *l;
        }
    }
    // Two passes: total, then cumulative exit (deterministic sequential
    // fold — the Issue-940 posture for a local instrument).
    let mut total = 0f64;
    for l in logits {
        total += (f64::from(*l - max) * inv).exp();
    }
    let pick = rng.next_f64() * total;
    let mut acc = 0f64;
    for (i, l) in logits.iter().enumerate() {
        acc += (f64::from(*l - max) * inv).exp();
        if acc >= pick {
            return i;
        }
    }
    logits.len() - 1
}

/// Per-prompt forward driver (a FRESH bounded cache from the prompt every
/// call — the history table is the only cross-call channel).
struct ProbeCtx {
    config: Config,
    weights: QwenDeltaNetTernaryWeights,
    tok: BpeTokenizer,
    rope: RopeFreqTable,
}

impl ProbeCtx {
    fn sample(
        &self,
        prompt: &str,
        max_new: usize,
        temp: f32,
        seed: u64,
    ) -> anyhow::Result<Vec<usize>> {
        let prompt_ids: Vec<usize> = self.tok.encode(prompt);
        let end_id: usize = self
            .tok
            .encode("<|im_end|>")
            .first()
            .copied()
            .unwrap_or(151_645);
        let mut cache = HybridCache::with_layer_types_bounded(
            &self.config,
            &self.weights.layer_types,
            prompt_ids.len() + max_new + 2,
        );
        let mut scratch = HybridForwardScratch::new(&self.config);
        let mut x = vec![0.0f32; self.config.vocab_size.max(self.config.n_embd)];
        let mut rng = SplitMix64(seed);
        let mut ids = prompt_ids.clone();
        for _step in 0..max_new {
            let last_token = *ids.last().expect("non-empty ids");
            let logits = forward_qwen_deltanet_ternary(
                &mut x,
                &self.weights,
                &mut cache,
                last_token,
                ids.len() - 1,
                &self.config,
                &mut scratch,
                &self.rope,
            );
            let next = sample_token(logits, temp, &mut rng);
            ids.push(next);
            if next == end_id {
                break;
            }
        }
        Ok(ids[prompt_ids.len()..].to_vec())
    }
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;
    println!("[1122] gguf: {}", args.gguf.display());
    let gguf = GgufFile::open(&args.gguf)?;
    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let tok = BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?;
    println!(
        "[1122] loaded: layers={} embd={} vocab={}",
        config.n_layer, config.n_embd, config.vocab_size
    );
    let ctx = ProbeCtx {
        config,
        weights,
        tok,
        rope,
    };

    // ── prompt grid (real problems, plan-461 shape) ──
    let mut prompts: Vec<(String, String)> = Vec::new();
    for (tpl, a, b) in PROBLEM_SEEDS {
        prompts.push((
            tpl.replacen("{a}", &a.to_string(), 1)
                .replacen("{b}", &b.to_string(), 1),
            format!("{a}x{b}"),
        ));
    }
    let mut k = 1usize;
    while prompts.len() < args.prompts {
        let (tpl, a, b) = PROBLEM_SEEDS[(k + 1) % PROBLEM_SEEDS.len()];
        let a2 = a + (k as i64) * 3;
        let b2 = b + (k as i64);
        prompts.push((
            tpl.replacen("{a}", &a2.to_string(), 1)
                .replacen("{b}", &b2.to_string(), 1),
            format!("{a2}x{b2}"),
        ));
        k += 1;
    }
    prompts.truncate(args.prompts);

    // ── greedy control (temp 0, twice, same prompt): MUST be identical ──
    let control_prompt = TEMPLATE.replacen("{s}", &prompts[0].0, 1);
    let g1 = ctx.sample(&control_prompt, args.max_new, 0.0, args.seed)?;
    let g2 = ctx.sample(&control_prompt, args.max_new, 0.0, args.seed)?;
    if g1 != g2 {
        anyhow::bail!(
            "greedy control DIVERGED (len {} vs {}) — the forward is nondeterministic; refusing the measurement",
            g1.len(),
            g2.len()
        );
    }
    let to_u32 = |v: &[usize]| -> Vec<u32> { v.iter().map(|&t| t as u32).collect() };
    let mut control_hist = PromptHistory::new(DEFAULT_HISTORY_CAPACITY);
    control_hist.record(&to_u32(&g1));
    let control_report = measure_alpha(
        &control_hist,
        &to_u32(&g2),
        DEFAULT_MAX_DRAFT,
        DEFAULT_MAX_LOOKUP_ORDER,
    );
    println!(
        "[1122] CONTROL greedy x2: alpha={:.4} tpf={:.3} len={} OK",
        control_report.alpha,
        control_report.tokens_per_forward,
        g2.len()
    );

    // ── re-roll grid: per prompt, samples at --temp ──
    let mut rolls: Vec<(String, Vec<Vec<usize>>)> = Vec::new();
    for (idx, (problem, tag)) in prompts.iter().enumerate() {
        let prompt = TEMPLATE.replacen("{s}", problem, 1);
        let mut outs = Vec::new();
        for s in 0..args.samples {
            let seed = args.seed ^ ((idx as u64) << 32) ^ (s as u64);
            let toks = ctx.sample(&prompt, args.max_new, args.temp, seed)?;
            outs.push(toks.clone());
            println!("[1122] p{s:>02}/{idx} seed={seed:#x} len={}", toks.len());
        }
        rolls.push((tag.clone(), outs));
    }

    // ── measure: sample k vs history 1..k-1 (history grows with k) ──
    let mut alphas: Vec<f64> = Vec::new();
    let mut tpf_sum = 0f64;
    let mut n_meas = 0usize;
    for (_tag, outs) in &rolls {
        let mut hist = PromptHistory::new(DEFAULT_HISTORY_CAPACITY);
        hist.record(&to_u32(&outs[0])); // cold start: α=0 by construction
        for t in outs.iter().skip(1) {
            let r = measure_alpha(
                &hist,
                &to_u32(t),
                DEFAULT_MAX_DRAFT,
                DEFAULT_MAX_LOOKUP_ORDER,
            );
            alphas.push(r.alpha);
            tpf_sum += r.tokens_per_forward;
            n_meas += 1;
            hist.record(&to_u32(t));
        }
    }
    let mean = alphas.iter().sum::<f64>() / alphas.len().max(1) as f64;
    let mut sorted = alphas.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];

    // ── retry shape: one operand perturbed, history from the original ──
    let (tpl, a, b) = PROBLEM_SEEDS[0];
    let retry_prompt = TEMPLATE.replacen(
        "{s}",
        &tpl.replacen("{a}", &(a + 1).to_string(), 1)
            .replacen("{b}", &b.to_string(), 1),
        1,
    );
    let mut retry_hist = PromptHistory::new(DEFAULT_HISTORY_CAPACITY);
    let orig_samples = &rolls[0].1;
    for t in orig_samples {
        retry_hist.record(&to_u32(t));
    }
    let retry_toks = ctx.sample(&retry_prompt, args.max_new, args.temp, args.seed ^ 0xfeed)?;
    let retry_report = measure_alpha(
        &retry_hist,
        &to_u32(&retry_toks),
        DEFAULT_MAX_DRAFT,
        DEFAULT_MAX_LOOKUP_ORDER,
    );

    // ── verdict ──
    let clears = mean >= BREAKEVEN_ALPHA_K2;
    println!("\n================ ISSUE 1122 T2 VERDICT ================");
    println!(
        "posture: temp={} samples={} prompts={} max_new={}",
        args.temp, args.samples, args.prompts, args.max_new
    );
    println!(
        "re-roll shape : alpha mean={:.4} median={:.4} n={} tpf_mean={:.3}",
        mean,
        median,
        n_meas,
        tpf_sum / n_meas.max(1) as f64
    );
    println!(
        "retry shape   : alpha={:.4} tpf={:.3} (history = {} original samples)",
        retry_report.alpha,
        retry_report.tokens_per_forward,
        orig_samples.len()
    );
    println!(
        "breakeven K=2 : alpha >= {:.2}  →  {}",
        BREAKEVEN_ALPHA_K2,
        if clears {
            "CLEARS — T1 wiring may proceed behind a claimed consumer"
        } else {
            "NOT CLEARED — lane is net-negative at K=2 at this posture; the honest verdict is NO-GO for wiring"
        }
    );
    println!("=======================================================\n");

    // ── artifacts for offline re-runs (gitignored .raw/) ──
    let mut prompt_rows = String::new();
    for (p, tag) in &prompts {
        prompt_rows.push_str(&format!("{tag}\t{p}\n"));
    }
    std::fs::write(args.out.join("prompts.tsv"), prompt_rows)?;
    let mut log = String::new();
    for (tag, outs) in &rolls {
        for (s, t) in outs.iter().enumerate() {
            log.push_str(&format!(
                "{{\"prompt_tag\":\"{tag}\",\"sample\":{s},\"tokens\":{t:?}}}\n"
            ));
        }
    }
    log.push_str(&format!("{{\"control_greedy_tokens\":{g1:?}}}\n"));
    log.push_str(&format!("{{\"retry_tokens\":{retry_toks:?}}}\n"));
    std::fs::write(args.out.join("samples.jsonl"), log)?;
    println!("[1122] artifacts: {}", args.out.display());
    Ok(())
}
