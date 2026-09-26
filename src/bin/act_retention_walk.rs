//! act_retention_walk — riir-infer Issue 014 T2/T3: the act-aware ternary
//! scale-refit arms + the per-family conditional retention walk that gates
//! them (the katgpt-rs Issue 886 P1 model-bound quality gate).
//!
//! ## T2 — the refit arms (all in-process; the checkpoint is never written)
//!
//! Starting from the SHIPPED `Q2_0_g128` payloads of the Bonsai-2 ternary
//! lane, each arm dequantizes every refit-able ternary matvec tensor to f32
//! and requantizes it under its scale rule:
//!
//! | arm | rule | diagonal |
//! |---|---|---|
//! | `shipped` | (none — the deployed reference) | — |
//! | `mean_abs` | `quantize_from_f32` (activation-blind) | — |
//! | `wma_ex2` | `WeightedMeanAbs` | per-site `E[x²]` (T1 artifact) |
//! | `ws_ex2` | `WeightedSearch` | per-site `E[x²]` |
//! | `ws_uniform` | `WeightedSearch` | uniform (the blind-search control) |
//! | `zeroqat` | mean-abs codes + multiplier GD (see below) | per-site `E[x²]` |
//!
//! The born-ternary requant is NOT the identity (the issue's trap 2): the
//! group's mean-abs is `s·nnz/128`, not `s`, and the carry loop re-derives
//! codes at the new threshold — so every arm is compared against BOTH the
//! shipped tensor (deployed reference) and, implicitly, its own requant
//! baseline (`mean_abs`).
//!
//! **The `zeroqat` arm (the incumbent-class comparator, honestly scoped).**
//! riir-train's `ZeroQatCalibrator` (Plan 255 Ph4) does central-finite-
//! difference GD on an injected loss at the group-scale insertion point;
//! riir-infer cannot depend on the training repo (boundary fence), so the
//! CLASS is mirrored here at its DEFAULT knobs (100 steps, lr 0.01, ε 0.01,
//! clamp [1e-6, 1e6]), multiplier-parameterized (init m = 1.0 ⇒ no change,
//! the model-agnostic init). Two structural facts are recorded either sign:
//! (1) on SHIPPED born-ternary codes the layer-local weighted-reconstruction
//! surrogate is exactly stationary — every nonzero |w| equals the group
//! scale, so `∂/∂s Σu(w−s·q)² = 0` at `s = s_shipped`: GD cannot move the
//! payload, which is why the arm runs at the mean-abs-REQUANT insertion
//! point (codes fixed at the requant's, scales GD-refined against the
//! SHIPPED weights) — the non-degenerate form of the same class; (2) the
//! loss is an exact parabola in the multiplier (fixed codes), evaluated in
//! closed form from precomputed per-group `(Σuw², Σuw·q, Σu·q²)` — float
//! association differs from a per-element loop, the GD dynamics do not.
//!
//! ## T3 — the walk (Bench 948 pattern; aggregate PPL is disqualified)
//!
//! Per arm, teacher-forced next-token scoring over N content families ×
//! items × L tokens (families = deterministic text files; the diagonal is
//! the T1 artifact). Against the `shipped` reference, per family: argmax
//! flips, top-k retention (ref argmax ∈ arm top-k), the reference margin at
//! flips (split at margin 1.0 — high-margin flips are the scary class), and
//! gold NLL (context column only).
//!
//! MEASUREMENT-ONLY (Issue 014 P0 law): the walk reads refit payloads built
//! in memory; nothing is written back to the checkpoint.
//!
//! Usage:
//! ```text
//! act_retention_walk <model.gguf> <diagonal.bin> <families-dir>
//!     [--items N] [--item-tokens N] [--topk N] [--out PATH] [--report PATH]
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use half::f16;
use katgpt_core::act_channel_moments::ActChannelDiagonal;
use katgpt_core::TernaryGroupWeights;
use katgpt_types::ternary_group_act_aware::ActAwareScaleFit;

use riir_infer_core::corpus_text::load_corpus_pages;
use riir_infer_core::deltanet::act_taps::{ActTapPlan, for_each_ternary_site_mut};
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

/// `TernaryGroupWeights::GROUP_SIZE` (katgpt-types; hardcoded here the same
/// way the dequant path hardcodes it — see `dequant_wte_row_into`).
const GROUP: usize = 128;

/// The arms, in run order. `shipped` FIRST — it is the reference the later
/// arms compare against.
const ARMS: [&str; 6] = ["shipped", "mean_abs", "wma_ex2", "ws_ex2", "ws_uniform", "zeroqat"];

/// riir-train `ZeroQatConfig::default()` (Plan 255 Ph4) — the comparator's
/// knobs, mirrored verbatim.
const ZQ_STEPS: usize = 100;
const ZQ_LR: f32 = 0.01;
const ZQ_EPS: f32 = 0.01;
const ZQ_MIN_M: f32 = 1e-6;
const ZQ_MAX_M: f32 = 1e6;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: act_retention_walk <model.gguf> <diagonal.bin> <families-dir> \
             [--items N] [--item-tokens N] [--topk N] [--out PATH] [--report PATH]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let diag_path = PathBuf::from(&args[2]);
    let families_dir = PathBuf::from(&args[3]);
    let mut items_per_family: usize = 12;
    let mut item_tokens: usize = 64;
    let mut topk: usize = 8;
    let mut out_path: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--items" => {
                items_per_family = args[i + 1].parse().context("--items N")?;
                i += 2;
            }
            "--item-tokens" => {
                item_tokens = args[i + 1].parse().context("--item-tokens N")?;
                i += 2;
            }
            "--topk" => {
                topk = args[i + 1].parse().context("--topk N")?;
                i += 2;
            }
            "--out" => {
                out_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if topk < 2 {
        bail!("--topk must be >= 2 (the margin is top1 - top2)");
    }
    if item_tokens < 8 {
        bail!("--item-tokens must be >= 8");
    }
    for arm in ARMS {
        match arm {
            "shipped" | "mean_abs" | "wma_ex2" | "ws_ex2" | "ws_uniform" | "zeroqat" => {}
            other => bail!("unknown arm {other}"),
        }
    }

    // ── The T1 diagonal artifact (the refit input) ──
    let diag_bytes = std::fs::read(&diag_path)
        .with_context(|| format!("read diagonal {}", diag_path.display()))?;
    let diagonal = ActChannelDiagonal::from_bytes(&diag_bytes).context("decode diagonal")?;
    assert!(
        diagonal.verify(),
        "the diagonal artifact failed its own BLAKE3 commitment"
    );
    println!(
        "# diagonal: {} taps, commitment {}",
        diagonal.layers(),
        hex(&diagonal.commitment()),
    );

    // ── Families: deterministic text buckets ──
    let pages = load_corpus_pages(&families_dir)
        .with_context(|| format!("load families from {}", families_dir.display()))?;
    let fam_names: Vec<String> = pages.iter().map(|(n, _)| n.clone()).collect();
    println!("# families: {} ({:?})", pages.len(), fam_names);

    // ── Load model (arm 0) + tokenizer ──
    let t0 = Instant::now();
    let (mut config, mut weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    println!(
        "# model: qwen35 hybrid | layers={} n_embd={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    assert!(
        weights.invariants_hold(),
        "shipped payloads violate the bit-plane invariant"
    );
    let tok = {
        let gguf = GgufFile::open(&gguf_path).context("re-open gguf for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };

    // ── Items: tokenize each family once, cut deterministic windows ──
    let t1 = Instant::now();
    let bos = tok.bos_id();
    let mut family_items: Vec<Vec<Vec<usize>>> = Vec::with_capacity(pages.len());
    for (name, text) in &pages {
        let toks = tok.encode(text);
        let mut items = Vec::with_capacity(items_per_family);
        for j in 0..items_per_family {
            let start = 1 + j * item_tokens; // skip the family's first token (post-BOS)
            let end = start + item_tokens;
            if end >= toks.len() {
                break; // family exhausted; shorter families contribute fewer items
            }
            let mut t = Vec::with_capacity(item_tokens + 1);
            t.push(bos);
            t.extend_from_slice(&toks[start..end]);
            items.push(t);
        }
        println!(
            "# family {name}: {} chars → {} tokens → {} items",
            text.len(),
            toks.len(),
            items.len()
        );
        family_items.push(items);
    }
    let total_items: usize = family_items.iter().map(|f| f.len()).sum();
    if total_items == 0 {
        bail!("no family produced an item — corpus too short for --items/--item-tokens");
    }
    let positions_per_arm: usize = total_items * item_tokens; // bos + L tokens ⇒ L predictions
    println!(
        "# items: {total_items} × {item_tokens} tokens = {positions_per_arm} scored positions/arm ({:.1}s tokenize)",
        t1.elapsed().as_secs_f32()
    );

    // Cap the KV cache at what one item needs (the row_logit_floor law).
    config.block_size = item_tokens;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let plan = ActTapPlan::build(&config, &weights);

    // The uniform-diagonal control slices, one per distinct width, built on
    // demand (the widths repeat across layers/sites).
    let mut uniform_diag: HashMap<usize, Vec<f32>> = HashMap::new();

    let mut out = out_path
        .as_ref()
        .map(|p| std::fs::File::create(p).with_context(|| format!("create {}", p.display())))
        .transpose()?;

    // ── The reference records (arm `shipped`) ──
    let mut reference: Vec<PosRec> = Vec::with_capacity(positions_per_arm);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    let mut report = String::new();
    report.push_str(&format!(
        "# Issue 014 T2/T3 — act-aware ternary scale refit + per-family retention walk\n\n\
         model: {} | diagonal commitment {} | families {:?} | {} items × {} tokens\n\n",
        gguf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?"),
        hex(&diagonal.commitment()),
        fam_names,
        total_items,
        item_tokens,
    ));

    for (arm_idx, &arm) in ARMS.iter().enumerate() {
        // ── Refit (arms > 0 reload the checkpoint first — every refit
        //    re-derives from the SHIPPED payloads, never from a prior arm) ──
        if arm_idx > 0 {
            let t_reload = Instant::now();
            let (_cfg, w) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
                .with_context(|| format!("reload for arm {arm}"))?;
            assert!(
                w.invariants_hold(),
                "reloaded payloads violate the bit-plane invariant"
            );
            weights = w;
            println!(
                "# [{arm}] reloaded in {:.1}s",
                t_reload.elapsed().as_secs_f32()
            );
        }

        let refit_stats = if arm_idx == 0 {
            None
        } else {
            let t_refit = Instant::now();
            let stats = apply_arm(arm, &mut weights, &plan, &diagonal, &mut uniform_diag)?;
            println!(
                "# [{arm}] refit {:.1}s | changed {}/{} weights ({:.3}%) | scale ratio med {:.4} [p10 {:.4}, p90 {:.4}] max {:.3}",
                t_refit.elapsed().as_secs_f32(),
                stats.changed_weights,
                stats.total_weights,
                100.0 * stats.changed_weights as f64 / stats.total_weights.max(1) as f64,
                stats.ratio.p50(),
                stats.ratio.p10(),
                stats.ratio.p90(),
                stats.ratio.max,
            );
            Some(stats)
        };

        // ── The walk ──
        let t_walk = Instant::now();
        let mut cmp = WalkCmp::new(&fam_names, topk);
        let mut gold_nll_sum = 0.0f64;
        let mut gold_hits = 0usize;
        let mut positions = 0usize;
        for (fam_idx, items) in family_items.iter().enumerate() {
            for item in items {
                cache.reset();
                for (pos, &token) in item.iter().enumerate() {
                    if pos + 1 >= item.len() {
                        break; // no next token to score
                    }
                    let fwd = forward_qwen_deltanet_ternary_with_hook(
                        &mut x,
                        &weights,
                        &mut cache,
                        token,
                        pos,
                        &config,
                        &mut scratch,
                        &rope_freq,
                        None,
                        None,
                        None,
                        None,
                    );
                    let logits = &fwd[..config.vocab_size];
                    let gold = item[pos + 1];
                    let rec = scan_position(logits, gold, topk);
                    if arm_idx == 0 {
                        reference.push(rec.clone());
                    } else {
                        let rec_ref = &reference[positions];
                        cmp.observe(fam_idx, rec_ref, &rec);
                    }
                    gold_nll_sum += f64::from(rec.gold_nll);
                    if rec.argmax == gold as u32 {
                        gold_hits += 1;
                    }
                    positions += 1;
                }
            }
        }
        assert_eq!(
            positions, positions_per_arm,
            "arm {arm} walked {positions} positions, the plan expected \
             {positions_per_arm} — the item structure changed under the walk"
        );
        let walk_secs = t_walk.elapsed().as_secs_f32();
        println!(
            "# [{arm}] walked {positions} positions in {walk_secs:.0}s ({:.0} tok/s) | \
             gold NLL/token {:.4} | gold-hit {:.2}%",
            positions as f32 / walk_secs.max(1e-6),
            gold_nll_sum / positions.max(1) as f64,
            100.0 * gold_hits as f64 / positions.max(1) as f64,
        );

        // ── Record (append-per-arm: a kill loses at most one arm) ──
        let line = cmp.jsonl_line(
            arm,
            refit_stats.as_ref(),
            walk_secs,
            positions,
            gold_nll_sum,
            gold_hits,
        );
        println!("{line}");
        if let Some(f) = out.as_mut() {
            writeln!(f, "{line}").context("write jsonl")?;
            f.flush().ok();
        }
        report.push_str(&cmp.markdown_table(
            arm,
            refit_stats.as_ref(),
            positions,
            gold_nll_sum,
        ));
        report.push('\n');
    }

    report.push_str(
        "\nMEASUREMENT-ONLY (Issue 014 P0 law): in-process refits; the checkpoint is \
         never written. Aggregate gold NLL is a context column — the gate is the \
         per-family conditional walk (flips / top-k retention / margin at flips), \
         recorded either sign.\n",
    );
    print!("{report}");
    if let Some(p) = report_path {
        std::fs::write(&p, &report).with_context(|| format!("write report {}", p.display()))?;
        println!("# report written: {}", p.display());
    }
    Ok(())
}

/// One scored position: top-k + margins + gold NLL.
#[derive(Clone)]
struct PosRec {
    argmax: u32,
    ids: Vec<u32>,
    /// top1 − top2 logit margin.
    margin: f32,
    gold_nll: f32,
}

/// One fused pass for top-k + argmax, then one exp pass for the gold NLL.
fn scan_position(logits: &[f32], gold: usize, k: usize) -> PosRec {
    let mut ids: Vec<u32> = Vec::with_capacity(k);
    let mut vals: Vec<f32> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        let n = vals.len();
        if n < k || l > vals[n - 1] {
            // Insertion into the descending top-k.
            let mut pos = n.min(k - 1);
            if n < k {
                vals.push(l);
                ids.push(i as u32);
            } else {
                vals[pos] = l;
                ids[pos] = i as u32;
            }
            while pos > 0 && vals[pos] > vals[pos - 1] {
                vals.swap(pos, pos - 1);
                ids.swap(pos, pos - 1);
                pos -= 1;
            }
        }
    }
    let m = vals[0];
    let mut sum = 0.0f64;
    for &l in logits {
        sum += f64::from((l - m).exp());
    }
    // nll = ln Σ exp(l − m) − (gold − m); the max element guarantees sum ≥ 1
    // so the natural log is ≥ 0.
    let nll = (sum.ln() - f64::from(logits[gold] - m)) as f32;
    PosRec {
        argmax: ids[0],
        margin: m - vals[1],
        ids,
        gold_nll: nll,
    }
}

/// Per-family comparison state vs the reference arm.
struct WalkCmp {
    names: Vec<String>,
    topk: usize,
    positions: Vec<usize>,
    flips: Vec<usize>,
    flips_high_margin: Vec<usize>,
    topk_retain: Vec<usize>,
    ref_margin_at_flips: Vec<f64>,
    gold_nll_ref: Vec<f64>,
    gold_nll_arm: Vec<f64>,
}

impl WalkCmp {
    fn new(names: &[String], topk: usize) -> Self {
        let n = names.len();
        Self {
            names: names.to_vec(),
            topk,
            positions: vec![0; n],
            flips: vec![0; n],
            flips_high_margin: vec![0; n],
            topk_retain: vec![0; n],
            ref_margin_at_flips: vec![0.0; n],
            gold_nll_ref: vec![0.0; n],
            gold_nll_arm: vec![0.0; n],
        }
    }

    fn observe(&mut self, fam: usize, rec_ref: &PosRec, rec_arm: &PosRec) {
        self.positions[fam] += 1;
        self.gold_nll_ref[fam] += f64::from(rec_ref.gold_nll);
        self.gold_nll_arm[fam] += f64::from(rec_arm.gold_nll);
        if rec_arm.argmax != rec_ref.argmax {
            self.flips[fam] += 1;
            self.ref_margin_at_flips[fam] += f64::from(rec_ref.margin);
            if rec_ref.margin > 1.0 {
                self.flips_high_margin[fam] += 1;
            }
        }
        if rec_arm.ids[..self.topk].contains(&rec_ref.argmax) {
            self.topk_retain[fam] += 1;
        }
    }

    fn jsonl_line(
        &self,
        arm: &str,
        refit: Option<&RefitStats>,
        walk_secs: f32,
        positions: usize,
        gold_nll_sum: f64,
        gold_hits: usize,
    ) -> String {
        let fams: Vec<String> = (0..self.names.len())
            .map(|f| {
                format!(
                    "{{\"name\":{},\"positions\":{},\"flips\":{},\"flips_high_margin\":{},\
                     \"topk_retain\":{},\"ref_margin_at_flips_mean\":{:.4},\
                     \"gold_nll\":{:.6},\"gold_nll_ref\":{:.6}}}",
                    json_str(&self.names[f]),
                    self.positions[f],
                    self.flips[f],
                    self.flips_high_margin[f],
                    self.topk_retain[f],
                    self.ref_margin_at_flips[f] / self.flips[f].max(1) as f64,
                    self.gold_nll_arm[f] / self.positions[f].max(1) as f64,
                    self.gold_nll_ref[f] / self.positions[f].max(1) as f64,
                )
            })
            .collect();
        let refit_json = match refit {
            None => "null".to_string(),
            Some(r) => format!(
                "{{\"changed_weights\":{},\"total_weights\":{},\"ratio\":{}}}",
                r.changed_weights,
                r.total_weights,
                r.ratio.json(),
            ),
        };
        format!(
            "{{\"arm\":\"{arm}\",\"refit\":{refit_json},\"walk_secs\":{walk_secs:.1},\
             \"positions\":{positions},\"gold_nll\":{:.6},\"gold_hits\":{gold_hits},\
             \"families\":[{}]}}",
            gold_nll_sum / positions.max(1) as f64,
            fams.join(","),
        )
    }

    fn markdown_table(
        &self,
        arm: &str,
        refit: Option<&RefitStats>,
        positions: usize,
        gold_nll_sum: f64,
    ) -> String {
        let mut s = String::new();
        s.push_str(&format!("## arm `{arm}`\n\n"));
        if let Some(r) = refit {
            s.push_str(&format!(
                "refit: changed {}/{} weights ({:.3}%), scale ratio med {:.4} \
                 [p10 {:.4} · p90 {:.4}] max {:.3}\n\n",
                r.changed_weights,
                r.total_weights,
                100.0 * r.changed_weights as f64 / r.total_weights.max(1) as f64,
                r.ratio.p50(),
                r.ratio.p10(),
                r.ratio.p90(),
                r.ratio.max,
            ));
        }
        s.push_str(&format!(
            "gold NLL/token {:.4} ({} positions)\n\n",
            gold_nll_sum / positions.max(1) as f64,
            positions,
        ));
        s.push_str(
            "| family | pos | flips | flip% | hi-margin flips | top-k retain | retain% | ref-margin@flip | gold NLL (arm/ref) |\n\
             |---|---|---|---|---|---|---|---|---|\n",
        );
        for f in 0..self.names.len() {
            let pos = self.positions[f].max(1);
            s.push_str(&format!(
                "| {} | {} | {} | {:.2}% | {} | {} | {:.2}% | {:.3} | {:.4} / {:.4} |\n",
                self.names[f],
                self.positions[f],
                self.flips[f],
                100.0 * self.flips[f] as f64 / pos as f64,
                self.flips_high_margin[f],
                self.topk_retain[f],
                100.0 * self.topk_retain[f] as f64 / pos as f64,
                self.ref_margin_at_flips[f] / self.flips[f].max(1) as f64,
                self.gold_nll_arm[f] / pos as f64,
                self.gold_nll_ref[f] / pos as f64,
            ));
        }
        s
    }
}

/// Log-spaced ratio histogram over `[1e-3, 1e3]` with exact min/max and
/// count — quantiles read off the cumulative bins (bin resolution, not a
/// smooth percentile; the repo's percentile law, disclosed).
struct RatioHist {
    n: u64,
    min: f32,
    max: f32,
    log_sum: f64,
    bins: [u64; N_BINS],
    skipped_zero: u64,
}

const N_BINS: usize = 120;
const LO_EXP: f64 = -3.0; // 1e-3
const HI_EXP: f64 = 3.0; // 1e3

impl Default for RatioHist {
    fn default() -> Self {
        Self {
            n: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            log_sum: 0.0,
            bins: [0; N_BINS],
            skipped_zero: 0,
        }
    }
}

impl RatioHist {
    fn observe(&mut self, ratio: f32) {
        if ratio <= 0.0 || !ratio.is_finite() {
            self.skipped_zero += 1;
            return;
        }
        self.n += 1;
        self.min = self.min.min(ratio);
        self.max = self.max.max(ratio);
        self.log_sum += f64::from(ratio.log10());
        let e = f64::from(ratio.log10()).clamp(LO_EXP, HI_EXP - 1e-9);
        let bin = (((e - LO_EXP) / (HI_EXP - LO_EXP)) * N_BINS as f64) as usize;
        self.bins[bin.min(N_BINS - 1)] += 1;
    }

    fn quantile(&self, q: f64) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        let target = (q * self.n as f64).ceil().max(1.0) as u64;
        let mut cum = 0u64;
        for (i, &b) in self.bins.iter().enumerate() {
            cum += b;
            if cum >= target {
                let e = LO_EXP + (i as f64 + 0.5) / N_BINS as f64 * (HI_EXP - LO_EXP);
                return 10f64.powf(e);
            }
        }
        10f64.powf(HI_EXP)
    }

    fn p10(&self) -> f64 {
        self.quantile(0.10)
    }
    fn p50(&self) -> f64 {
        self.quantile(0.50)
    }
    fn p90(&self) -> f64 {
        self.quantile(0.90)
    }

    fn json(&self) -> String {
        format!(
            "{{\"n\":{},\"min\":{:.6},\"max\":{:.6},\"log_mean\":{:.4},\"p10\":{:.5},\
             \"p50\":{:.5},\"p90\":{:.5},\"skipped_zero\":{}}}",
            self.n,
            self.min,
            self.max,
            self.log_sum / self.n.max(1) as f64,
            self.p10(),
            self.p50(),
            self.p90(),
            self.skipped_zero,
        )
    }
}

struct RefitStats {
    changed_weights: usize,
    total_weights: usize,
    ratio: RatioHist,
}

/// Apply one refit arm to every refit-able ternary tensor, in site order.
///
/// `weights` arrives freshly loaded = SHIPPED payloads; each tensor is
/// dequantized from those (the reference values), requantized under the arm's
/// rule, and swapped in. Stats aggregate the payload delta vs shipped.
fn apply_arm(
    arm: &str,
    weights: &mut QwenDeltaNetTernaryWeights,
    plan: &ActTapPlan,
    diagonal: &ActChannelDiagonal,
    uniform_diag: &mut HashMap<usize, Vec<f32>>,
) -> Result<RefitStats> {
    let mut stats = RefitStats {
        changed_weights: 0,
        total_weights: 0,
        ratio: RatioHist::default(),
    };
    let fit = match arm {
        "wma_ex2" => Some(ActAwareScaleFit::WeightedMeanAbs),
        "ws_ex2" | "ws_uniform" => Some(ActAwareScaleFit::WeightedSearch),
        _ => None, // mean_abs / zeroqat / shipped: the act-aware fit is unused
    };
    for_each_ternary_site_mut(weights, |site, w| {
        // Dequantize the SHIPPED payload — the reference values every arm
        // requantizes from (never a prior arm's output).
        let dense = QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w);
        let cols = w.cols;
        let rows = w.rows;
        let new = match arm {
            "mean_abs" => TernaryGroupWeights::quantize_from_f32(&dense, rows, cols),
            "wma_ex2" | "ws_ex2" => {
                let diag_slice = plan.diag_for(diagonal, site);
                TernaryGroupWeights::quantize_from_f32_act_aware(
                    &dense,
                    rows,
                    cols,
                    diag_slice,
                    fit.expect("act-aware arms carry a fit"),
                )
            }
            "ws_uniform" => {
                let d = uniform_diag
                    .entry(cols)
                    .or_insert_with(|| vec![1.0f32; cols]);
                TernaryGroupWeights::quantize_from_f32_act_aware(
                    &dense,
                    rows,
                    cols,
                    d,
                    fit.expect("act-aware arms carry a fit"),
                )
            }
            "zeroqat" => zeroqat_refit(w, &dense, plan.diag_for(diagonal, site)),
            _ => unreachable!("arm names validated in main"),
        };
        // Stats: payload delta vs shipped. A WEIGHT changed when either of
        // its two plane bits differs (counted per u64 word pair, popcount of
        // the OR of the two XORs).
        stats.total_weights += rows * cols;
        for ((po, pn), (no, nn)) in w
            .pos_bits
            .iter()
            .zip(&new.pos_bits)
            .zip(w.neg_bits.iter().zip(&new.neg_bits))
        {
            stats.changed_weights +=
                ((po ^ pn) | (no ^ nn)).count_ones() as usize;
        }
        for r in 0..rows {
            let ob = r * w.groups_per_row;
            for g in 0..w.groups_per_row {
                let os = f32::from(w.group_scale[ob + g]);
                let ns = f32::from(new.group_scale[ob + g]);
                if os > 0.0 {
                    stats.ratio.observe(ns / os);
                }
            }
        }
        *w = new;
    });
    Ok(stats)
}

/// The ZeroQAT-class arm: codes fixed at the mean-abs requant's, scales
/// GD-refined (multiplier parameterization, riir-train default knobs) against
/// the SHIPPED weights under the diagonal-weighted reconstruction loss.
fn zeroqat_refit(
    shipped: &TernaryGroupWeights,
    dense: &[f32],
    diag_slice: &[f32],
) -> TernaryGroupWeights {
    let rows = shipped.rows;
    let cols = shipped.cols;
    let mut rq = TernaryGroupWeights::quantize_from_f32(dense, rows, cols);
    for r in 0..rows {
        for g in 0..rq.groups_per_row {
            let g_start = g * GROUP;
            let g_end = (g_start + GROUP).min(cols);
            // u = normalized diagonal over the group's slice (the act-aware
            // fit's normalization; no-information ⇒ uniform).
            let h = &diag_slice[g_start..g_end];
            let hmax = h.iter().copied().fold(0.0f32, f32::max);
            let s_rq = f32::from(rq.group_scale[r * rq.groups_per_row + g]);
            // (A, B, C) of the exact parabola L(m) = A − 2B·s·m + C·s²·m².
            let (mut a, mut b, mut c) = (0.0f32, 0.0f32, 0.0f32);
            for (j, &wij) in dense[r * cols + g_start..r * cols + g_end]
                .iter()
                .enumerate()
            {
                let u = if hmax > 0.0 { h[j] / hmax } else { 1.0 };
                let q = ternary_sign(&rq, r, g_start + j) as f32;
                a += u * wij * wij;
                b += u * wij * q;
                c += u * q * q;
            }
            let l = |m: f32| {
                let sm = s_rq * m;
                a - 2.0 * b * sm + c * sm * sm
            };
            let mut m = 1.0f32;
            for _ in 0..ZQ_STEPS {
                let grad = (l(m + ZQ_EPS) - l(m - ZQ_EPS)) / (2.0 * ZQ_EPS);
                m = (m - ZQ_LR * grad).clamp(ZQ_MIN_M, ZQ_MAX_M);
            }
            rq.group_scale[r * rq.groups_per_row + g] = f16::from_f32(m * s_rq);
        }
    }
    rq
}

/// The ternary sign at `(row, col)`: +1 / −1 / 0 from the bit-planes.
#[inline]
fn ternary_sign(w: &TernaryGroupWeights, row: usize, col: usize) -> i8 {
    let idx = row * w.blocks64 + (col >> 6);
    let mask = 1u64 << (col & 63);
    if w.pos_bits[idx] & mask != 0 {
        1
    } else if w.neg_bits[idx] & mask != 0 {
        -1
    } else {
        0
    }
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
