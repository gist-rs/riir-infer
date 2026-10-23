//! `si_zero_shot_ppl` — Plan 622 Phase 7 G3: the Bonsai zero-shot split.
//!
//! The scale-invariant schedule (`transformer::attention_si`, opt-in
//! `scale_invariant_attn`) measured at the MODEL level on the packed
//! Bonsai ternary decode path — zero weight edits, one loaded checkpoint
//! shared by every arm, the si session swapped per arm on the forward
//! scratch:
//!
//! | arm | session | pre-recorded expectation |
//!|---|---|---|
//!| `base` | `None` (the plain path) | the reference curve |
//!| `si` | `Armed` (uncalibrated) | near-unit-variance logits (QK-norm) → close to base |
//!| `si-cal` | probe → frozen σ table → `Armed` (calibrated) | the schedule's premise, measured |
//!| `si-pprope` | `Armed` + the pp-RoPE table | **EXPECTED FAIL** (G3 (iii), recorded in advance — a full-RoPE checkpoint loses its trained position representation) |
//!
//! Protocol (per arm): 16 log-spaced length rungs from `--min-length`
//! (128) to `--max-length` (2048); each rung resets the cache, feeds the
//! sequence prefix `0..=L`, and accumulates the NLL of the last
//! `--score-window` (128) targets — the same scored span under a growing
//! context, so the curve reads length-scaling, not text difficulty.
//! Per-rung ppl + the pooled ppl + per-rung Δ vs the `base` arm; the
//! verdict bar is `--tolerance` (default 5%) on EVERY rung.
//!
//! The `si-cal` arm first runs the load-time σ probe (`--probe-tokens`,
//! default 512) — plain math, per-`(layer, head)` raw-logit Welford, sink
//! dropped — and prints the σ diagnostics beside the frozen table digest.
//!
//! `--retrieval` adds the needle probe (G3's "retrieval probe"): a fact
//! at the head of the prompt, filler to the rung length, the question at
//! the end, 6 greedy decode tokens, substring match — at 1024 and 2048.
//!
//! The `si-pprope` arm on an UNMARKED checkpoint requires
//! `--pp-train-len` (loud measurement bypass naming the law); a MARKED
//! checkpoint (Plan-449 marks) goes through `pp_rope_table_from_marks`
//! (the serving path). Either way the row is reported, never asserted.
//!
//! Usage:
//! ```text
//! cargo run --release --features scale_invariant_attn,bonsai2_hadamard \
//!   --bin si_zero_shot_ppl -- \
//!   --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --corpus <text> [--arms base,si,si-cal,si-pprope] [--max-length 2048]
//!   [--score-window 128] [--tolerance 0.05] [--probe-tokens 512]
//!   [--tau 10.0] [--retrieval] [--pp-train-len 1024]
//!   [--save-cal <path>] [--probe-only]   (Plan 620 — freeze the σ table
//!   for the GPU G3 lane; e.g. --arms si-cal,base --probe-only --save-cal c.bin)
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use katgpt_core::scale_invariant::{HeadCalibration, ScaleInvariantLut, SpreadAccumulator};
#[cfg(feature = "golden_nll")]
use katgpt_core::softmax_fix::{FloatFormat, precision_cliff};
#[cfg(feature = "golden_nll")]
use riir_infer_core::golden_nll::{GoldenNllScratch, GoldenReading};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{
    GgufFile, load_qwen_deltanet_ternary_weights_gguf, pp_rope_table_from_marks,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::transformer::attention_si::{SiAttentionPlan, SiAttentionSession};
use riir_infer_core::types::{Config, DeltaNetLayerType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Base,
    Si,
    SiCal,
    SiPprope,
}

impl Arm {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "base" => Ok(Arm::Base),
            "si" => Ok(Arm::Si),
            "si-cal" => Ok(Arm::SiCal),
            "si-pprope" => Ok(Arm::SiPprope),
            other => bail!("unknown arm {other} (base|si|si-cal|si-pprope)"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Arm::Base => "base",
            Arm::Si => "si",
            Arm::SiCal => "si-cal",
            Arm::SiPprope => "si-pprope",
        }
    }
}

/// One length rung's scoring result.
#[derive(Debug, Clone)]
struct Rung {
    length: usize,
    nll_sum: f64,
    targets: usize,
    /// Issue 959 T6 golden readout (cfg'd `golden_nll`): the same
    /// scored positions through the argmax-excluded-LSE f64 oracle.
    #[cfg(feature = "golden_nll")]
    golden_sum: f64,
    /// Positions in the confident band (top-2 margin ≥ the fp32 cliff
    /// 16.64 nats — where an fp32-consumer readout reads false zeros).
    #[cfg(feature = "golden_nll")]
    confident: usize,
    #[cfg(feature = "golden_nll")]
    golden_confident_sum: f64,
    /// Positions in the deep band (margin ≥ the f64 cliff 36.73 — where
    /// even an f64 max-subtract LSE absorbs the tail).
    #[cfg(feature = "golden_nll")]
    deep: usize,
    #[cfg(feature = "golden_nll")]
    golden_deep_sum: f64,
    /// Positions where the golden NLL and the incumbent f64 form disagree
    /// AT ALL (bit-wise) — the incumbent's resolution-loss census.
    #[cfg(feature = "golden_nll")]
    disagrees: usize,
    #[cfg(feature = "golden_nll")]
    max_abs_delta: f64,
}

impl Rung {
    fn ppl(&self) -> f64 {
        (self.nll_sum / self.targets.max(1) as f64).exp()
    }
}

/// The golden-NLL accumulation state for one rung (Issue 959 T6).
/// Under `cfg(not(feature = "golden_nll"))` this is a zero-sized inert
/// placeholder — the armed path is compile-time unreachable and the
/// runtime flag refuses loud (the harness law: never a silent skip).
#[cfg(feature = "golden_nll")]
#[derive(Debug)]
struct GoldenAccum {
    scratch: GoldenNllScratch,
    sum: f64,
    confident: usize,
    confident_sum: f64,
    deep: usize,
    deep_sum: f64,
    disagrees: usize,
    max_abs_delta: f64,
}

#[cfg(feature = "golden_nll")]
impl GoldenAccum {
    fn new(vocab: usize) -> Self {
        Self {
            scratch: GoldenNllScratch::with_vocab(vocab),
            sum: 0.0,
            confident: 0,
            confident_sum: 0.0,
            deep: 0,
            deep_sum: 0.0,
            disagrees: 0,
            max_abs_delta: 0.0,
        }
    }

    fn record(
        &mut self,
        logits: &[f32],
        target: usize,
        incumbent_nll: f64,
        fp32_cliff: f64,
        f64_cliff: f64,
    ) {
        let GoldenReading { nll, top2_margin } = self.scratch.read_target(logits, target);
        self.sum += nll;
        if top2_margin >= fp32_cliff {
            self.confident += 1;
            self.confident_sum += nll;
        }
        if top2_margin >= f64_cliff {
            self.deep += 1;
            self.deep_sum += nll;
        }
        let delta = (nll - incumbent_nll).abs();
        if nll.to_bits() != incumbent_nll.to_bits() {
            self.disagrees += 1;
        }
        if delta > self.max_abs_delta {
            self.max_abs_delta = delta;
        }
    }
}

#[cfg(not(feature = "golden_nll"))]
#[derive(Debug, Default)]
struct GoldenAccum;

#[cfg(not(feature = "golden_nll"))]
impl GoldenAccum {
    fn inert() -> Self {
        Self
    }
}

/// The state one arm runs its rungs with (cache + scratch are shared and
/// reset per rung; the session is the arm's identity).
struct Rig<'a> {
    config: Config,
    weights: &'a QwenDeltaNetTernaryWeights,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    rope_freq: &'a RopeFreqTable,
}

impl<'a> Rig<'a> {
    fn new(
        config: &Config,
        weights: &'a QwenDeltaNetTernaryWeights,
        rope_freq: &'a RopeFreqTable,
    ) -> Self {
        let layer_types = weights.layer_types.clone();
        Self {
            config: config.clone(),
            weights,
            cache: HybridCache::with_layer_types(config, &layer_types),
            scratch: HybridForwardScratch::new(config),
            rope_freq,
        }
    }

    /// Set the arm's session on the scratch (the G3 A/B swap).
    fn set_session(&mut self, session: Option<SiAttentionSession>) {
        self.scratch.attention.scale_invariant = session;
    }

    /// Feed `seq[from..=to]`; accumulate the NLL of `seq[p+1]` for scored
    /// positions `p >= score_from`. Zero-alloc hot loop (the buffers ride
    /// the rig). `golden` accumulates the Issue-959-T6 oracle readout at
    /// the same scored positions (inert under cfg-off).
    #[allow(clippy::too_many_arguments)]
    fn feed(
        &mut self,
        seq: &[usize],
        from: usize,
        to: usize,
        score_from: usize,
        x: &mut [f32],
        nll: &mut f64,
        targets: &mut usize,
        golden: &mut GoldenAccum,
    ) {
        let n = self.config.n_embd;
        let v = self.config.vocab_size;
        #[cfg(feature = "golden_nll")]
        let fp32_cliff = precision_cliff(FloatFormat::Fp32);
        #[cfg(feature = "golden_nll")]
        let f64_cliff = precision_cliff(FloatFormat::Fp64);
        #[cfg(not(feature = "golden_nll"))]
        let _ = golden; // inert ZST under cfg-off (the armed path is compile-time unreachable)
        for pos in from..=to {
            let logits = forward_qwen_deltanet_ternary(
                x,
                self.weights,
                &mut self.cache,
                seq[pos],
                pos,
                &self.config,
                &mut self.scratch,
                self.rope_freq,
            );
            if pos >= score_from && pos + 1 < seq.len() {
                let target = seq[pos + 1];
                // f64 logsumexp over the raw logits (stable, exact enough
                // for the ppl rows; the exp pass is once per SCORED
                // position, never per fed token).
                let mut max_l = f32::NEG_INFINITY;
                for &l in logits.iter() {
                    max_l = max_l.max(l);
                }
                let mut sum = 0.0f64;
                for &l in logits.iter() {
                    sum += (f64::from(l) - f64::from(max_l)).exp();
                }
                let lse = f64::from(max_l) + sum.ln();
                let inc_nll = lse - f64::from(logits[target]);
                *nll += inc_nll;
                *targets += 1;
                #[cfg(feature = "golden_nll")]
                golden.record(logits, target, inc_nll, fp32_cliff, f64_cliff);
            }
        }
        let _ = (n, v);
    }

    /// Greedy-decode `steps` tokens from the fed prompt (argmax each step).
    fn greedy(
        &mut self,
        next_token: usize,
        next_pos: usize,
        steps: usize,
        x: &mut [f32],
    ) -> Vec<usize> {
        let mut out = Vec::with_capacity(steps);
        let mut token = next_token;
        for pos in next_pos..next_pos + steps {
            let logits = forward_qwen_deltanet_ternary(
                x,
                self.weights,
                &mut self.cache,
                token,
                pos,
                &self.config,
                &mut self.scratch,
                self.rope_freq,
            );
            let mut best = 0usize;
            let mut best_l = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                if l > best_l {
                    best_l = l;
                    best = i;
                }
            }
            out.push(best);
            token = best;
        }
        out
    }
}

/// The load-time σ probe: plain math over `n_tokens`, per-`(layer, head)`
/// raw-logit Welford, sink dropped (the session does the contract).
fn run_sigma_probe(
    rig: &mut Rig,
    seq: &[usize],
    n_tokens: usize,
    x: &mut [f32],
) -> Result<SpreadAccumulator> {
    let slots = rig.weights.layers.len() * rig.config.n_head;
    rig.cache.reset();
    rig.set_session(Some(SiAttentionSession::Probe(SpreadAccumulator::new(
        slots,
    ))));
    let to = n_tokens.min(seq.len()) - 1;
    let mut nll = 0.0;
    let mut targets = 0usize;
    #[cfg(feature = "golden_nll")]
    let mut probe_golden = GoldenAccum::new(0);
    #[cfg(not(feature = "golden_nll"))]
    let mut probe_golden = GoldenAccum::inert();
    rig.feed(
        seq,
        0,
        to,
        usize::MAX,
        x,
        &mut nll,
        &mut targets,
        &mut probe_golden,
    );
    match rig.scratch.attention.scale_invariant.take() {
        Some(SiAttentionSession::Probe(acc)) => Ok(acc),
        _ => bail!("sigma probe: session vanished — internal wiring bug"),
    }
}

/// Extract the frozen table from the probe. `expected_observed` is the
/// attention-layer slot count (`attention_layers × n_head`): every one of
/// those MUST have been observed (loud on shortfall — a probe that measured
/// nothing is a broken probe), while the GDN slots have no per-key logits
/// (the Plan-622 audited discard) and ride the NEUTRAL σ = 1.0 — a value
/// the armed path never reads (the dispatch fires only in attention
/// layers). Degenerate σ on an OBSERVED slot stays loud — a zero-σ head is
/// a wrong population, never a divide.
fn calibration_from_probe(
    acc: &SpreadAccumulator,
    slots: usize,
    expected_observed: usize,
) -> Result<HeadCalibration> {
    // `heads()` is the PRE-SIZED slot count — the observed set is the slots
    // carrying a std estimate. Every attention-layer slot MUST have one
    // (loud on shortfall — a probe that measured nothing is a broken probe,
    // not a calibration); the GDN slots have no per-key logits (the
    // Plan-622 audited discard) and ride the NEUTRAL σ = 1.0, a value the
    // armed path never reads (the dispatch fires only in attention layers).
    let mut neutral = 0usize;
    let mut scales = Vec::with_capacity(slots);
    for slot in 0..slots {
        match acc.std_estimate(slot) {
            Some(sigma) => {
                if !(sigma.is_finite() && sigma > 0.0) {
                    bail!("sigma probe: slot {slot} σ = {sigma} — degenerate population");
                }
                scales.push(sigma as f32);
            }
            None => {
                scales.push(1.0);
                neutral += 1;
            }
        }
    }
    let observed = slots - neutral;
    ensure!(
        observed == expected_observed,
        "sigma probe: observed {observed} of {expected_observed} attention-layer slots — \
         a shortfall is a broken probe, not a calibration"
    );
    println!(
        "# calibration table: {observed} measured + {neutral} neutral (GDN, never read) = {slots} slots",
    );
    HeadCalibration::build(scales).map_err(anyhow::Error::msg)
}

fn main() -> Result<()> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut arms = vec![Arm::Base, Arm::Si, Arm::SiCal, Arm::SiPprope];
    let mut max_length = 2048usize;
    let mut min_length = 128usize;
    let mut rungs_n = 16usize;
    let mut score_window = 128usize;
    let mut tolerance = 0.05f64;
    let mut probe_tokens = 512usize;
    let mut tau = 10.0f32;
    let mut retrieval = false;
    let mut pp_train_len: Option<usize> = None;
    // Plan 620 — the GPU G3 lane consumes the frozen σ table: `--save-cal`
    // writes the HeadCalibration sidecar right after the probe (BLAKE3-checked
    // format, `HeadCalibration::to_sidecar`), `--probe-only` stops there.
    let mut save_cal: Option<PathBuf> = None;
    let mut probe_only = false;
    // Issue 959 T6 — the f64 golden-NLL readout over the scored spans
    // (reference-oracle mode: base arm only, one forward, two readouts).
    let mut golden_nll = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(args.next().expect("path"))),
            "--corpus" => corpus = Some(PathBuf::from(args.next().expect("path"))),
            "--arms" => {
                arms = args
                    .next()
                    .expect("list")
                    .split(',')
                    .map(Arm::parse)
                    .collect::<Result<Vec<_>>>()?;
                if !arms.contains(&Arm::Base) {
                    bail!("the base arm is the reference side — always included");
                }
            }
            "--max-length" => max_length = args.next().expect("n").parse()?,
            "--min-length" => min_length = args.next().expect("n").parse()?,
            "--rungs" => rungs_n = args.next().expect("n").parse()?,
            "--score-window" => score_window = args.next().expect("n").parse()?,
            "--tolerance" => tolerance = args.next().expect("x").parse()?,
            "--probe-tokens" => probe_tokens = args.next().expect("n").parse()?,
            "--tau" => tau = args.next().expect("x").parse()?,
            "--retrieval" => retrieval = true,
            "--pp-train-len" => pp_train_len = Some(args.next().expect("n").parse()?),
            "--save-cal" => save_cal = Some(PathBuf::from(args.next().expect("path"))),
            "--probe-only" => probe_only = true,
            "--golden-nll" => golden_nll = true,
            other => bail!("unknown arg {other}"),
        }
    }
    let gguf_path = gguf_path.context("--gguf is required")?;
    let corpus = corpus.context("--corpus is required")?;
    if golden_nll && !cfg!(feature = "golden_nll") {
        bail!(
            "--golden-nll needs the golden_nll feature (loud refusal, never a silent skip): \
             cargo build --release --features scale_invariant_attn,bonsai2_hadamard,golden_nll --bin si_zero_shot_ppl"
        );
    }
    if golden_nll {
        if arms.len() != 1 || arms[0] != Arm::Base {
            println!(
                "# golden-nll: reference-oracle mode — the readout rides the BASE arm's forward \
                 (one forward, two readouts); overriding --arms to base"
            );
            arms = vec![Arm::Base];
        }
        println!(
            "# golden-nll: Issue 959 T6 — per-token NLL via the argmax-excluded-LSE f64 oracle \
             (katgpt-core fixed_log_softmax_target_f64); confidence bands at the fp32 cliff \
             16.64 / f64 cliff 36.73 nats"
        );
    }
    assert!(min_length >= 4 && max_length > min_length, "4 <= min < max");
    assert!(
        score_window >= 8 && score_window <= min_length,
        "8 <= score-window <= min-length"
    );

    // ── load once, share across arms ──
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    println!(
        "# si_zero_shot_ppl: {} | layers={} heads={}/{} rot={} | load {:.1}s",
        gguf_path.display(),
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        weights.rotation.is_some(),
        t0.elapsed().as_secs_f32()
    );

    // The row_logit_floor law: cap the cache window at what the run needs,
    // BEFORE any cache construction.
    config.block_size = max_length + 32;

    let gguf = GgufFile::open(&gguf_path).context("re-open for tokenizer + marks")?;
    let tok = BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?;

    // The pp-RoPE table for the si-pprope arm: marks first (the serving
    // path), else the loud measurement bypass.
    let pp_table = match pp_rope_table_from_marks(&gguf, &config) {
        Ok(t) => {
            println!(
                "# pp-RoPE: Plan-449 marks present — kept {} of {} pairs (base-marked)",
                t.kept(),
                effective_rotary_dim(&config) / 2
            );
            Some(t)
        }
        Err(e) => match pp_train_len {
            Some(len) => {
                println!(
                    "# pp-RoPE: UNMARKED CHECKPOINT — measurement bypass at --pp-train-len {len} \
                     (Plan 622 G3 (iii): the zero-shot pairing is the pre-recorded expected FAIL; \
                     loader verdict was: {e})"
                );
                Some(RopeFreqTable::new_pp(
                    config.rope_theta,
                    effective_rotary_dim(&config),
                    len,
                ))
            }
            None => {
                println!(
                    "# pp-RoPE: unmarked checkpoint and no --pp-train-len — the si-pprope arm \
                     will refuse if requested ({e})"
                );
                None
            }
        },
    };
    drop(gguf);

    // ── the frozen sequence (byte-identical for every arm) ──
    let text = load_corpus_text(&corpus)?;
    let mut seq = tok.encode_with_bos(&text);
    seq.truncate(max_length + 1);
    if seq.len() < min_length + 1 {
        bail!(
            "corpus too short: {} tokens (BOS incl.) < min-length {} + 1",
            seq.len(),
            min_length
        );
    }
    println!(
        "# corpus: {} tokens (BOS + text) | seq[0]=BOS={} sink carve-out armed | τ = {tau}",
        seq.len(),
        tok.bos_id()
    );

    // The 16 log-spaced rungs (the schedule is logarithmic in t; the curve
    // reads its shape, and the feed cost stays under the linear ladder's).
    let rungs: Vec<usize> = (0..rungs_n)
        .map(|k| {
            let l = min_length as f64
                * (max_length as f64 / min_length as f64).powf(k as f64 / (rungs_n - 1) as f64);
            l.round() as usize
        })
        .collect();
    println!(
        "# rungs ({}): {:?} | scored span {score_window} tokens per rung",
        rungs.len(),
        rungs
    );

    let plain_table = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));

    // The session LUT is arm-independent — one build at the block size.
    let lut = Arc::new(ScaleInvariantLut::build(tau, config.block_size));
    println!(
        "# LUT: max_distance {} (block {}) | ατ·ln(1+T/τ) at T={max_length} ≈ {:.1}",
        lut.max_distance(),
        config.block_size,
        (config.block_size as f64 / tau as f64).ln() * core::f64::consts::E.sqrt() * tau as f64
    );

    let mut rig = Rig::new(&config, &weights, &plain_table);
    let mut x = vec![0.0f32; config.n_embd.max(config.vocab_size)];

    // σ table built once (the si-cal arm), the diagnostics beside it.
    let mut calibration: Option<HeadCalibration> = None;
    if arms.contains(&Arm::SiCal) {
        let t = Instant::now();
        let acc = run_sigma_probe(&mut rig, &seq, probe_tokens, &mut x)?;
        let slots = weights.layers.len() * config.n_head;
        let attention_slots = weights
            .layer_types
            .iter()
            .filter(|t| **t == DeltaNetLayerType::Attention)
            .count()
            * config.n_head;
        let cal = calibration_from_probe(&acc, slots, attention_slots)?;
        // Diagnostics: the σ spread across slots is the premise check —
        // near-unity σ (QK-norm) says the uncalibrated arm should already
        // track the schedule's unit-variance premise.
        let mut sig: Vec<f64> = (0..slots).map(|s| f64::from(cal.scale_for(s))).collect();
        sig.sort_by(|a, b| a.partial_cmp(b).expect("finite σ"));
        let q = |p: f64| sig[((sig.len() - 1) as f64 * p) as usize];
        println!(
            "# sigma probe: {probe_tokens} tokens, {slots} slots in {:.1}s | σ min {:.4} p25 {:.4} \
             median {:.4} p75 {:.4} max {:.4} | mean {:.4}",
            t.elapsed().as_secs_f32(),
            sig[0],
            q(0.25),
            q(0.5),
            q(0.75),
            sig[sig.len() - 1],
            sig.iter().sum::<f64>() / sig.len() as f64
        );
        println!(
            "# calibration table: digest {:02x}… over {slots} slots",
            cal.digest()[..4]
                .iter()
                .fold(0u64, |a, b| (a << 8) | u64::from(*b))
                & 0xffff
        );
        // Plan 620 — freeze the table for the GPU G3 lane (the sidecar is
        // BLAKE3-checked on load; quote host + digest beside every row).
        if let Some(path) = &save_cal {
            std::fs::write(path, cal.to_sidecar())
                .with_context(|| format!("write cal sidecar {}", path.display()))?;
            println!("# cal sidecar written: {}", path.display());
        }
        if probe_only {
            println!("# --probe-only: stopping after the probe (the GPU lane owns the arms)");
            return Ok(());
        }
        calibration = Some(cal);
    }

    // ── arms ──
    let mut results: Vec<(Arm, Vec<Rung>)> = Vec::new();
    for arm in &arms {
        let t = Instant::now();
        match *arm {
            Arm::Base => rig.set_session(None),
            Arm::Si => rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
            )))),
            Arm::SiCal => {
                let cal = calibration
                    .as_ref()
                    .expect("calibration built above")
                    .clone();
                let plan = SiAttentionPlan::calibrated(
                    lut.clone(),
                    cal,
                    config.n_head,
                    weights.layers.len(),
                )
                .map_err(anyhow::Error::msg)?;
                rig.set_session(Some(SiAttentionSession::Armed(Arc::new(plan))));
            }
            Arm::SiPprope => {
                let Some(table) = pp_table.as_ref() else {
                    bail!(
                        "si-pprope requested but no pp table: the checkpoint is unmarked and \
                         --pp-train-len was not given (the serving path loud-fails too)"
                    );
                };
                rig.rope_freq = table;
                rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                    SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
                ))));
            }
        }
        let mut rung_results = Vec::with_capacity(rungs.len());
        #[cfg(feature = "golden_nll")]
        let mut golden = if golden_nll {
            GoldenAccum::new(config.vocab_size)
        } else {
            GoldenAccum::new(0)
        };
        #[cfg(not(feature = "golden_nll"))]
        let mut golden = GoldenAccum::inert();
        for &l in &rungs {
            rig.cache.reset();
            let mut nll = 0.0f64;
            let mut targets = 0usize;
            #[cfg(feature = "golden_nll")]
            {
                golden.sum = 0.0;
                golden.confident = 0;
                golden.confident_sum = 0.0;
                golden.deep = 0;
                golden.deep_sum = 0.0;
                golden.disagrees = 0;
                golden.max_abs_delta = 0.0;
            }
            let score_from = l.saturating_sub(score_window);
            rig.feed(
                &seq,
                0,
                l,
                score_from,
                &mut x,
                &mut nll,
                &mut targets,
                &mut golden,
            );
            rung_results.push(Rung {
                length: l,
                nll_sum: nll,
                targets,
                #[cfg(feature = "golden_nll")]
                golden_sum: golden.sum,
                #[cfg(feature = "golden_nll")]
                confident: golden.confident,
                #[cfg(feature = "golden_nll")]
                golden_confident_sum: golden.confident_sum,
                #[cfg(feature = "golden_nll")]
                deep: golden.deep,
                #[cfg(feature = "golden_nll")]
                golden_deep_sum: golden.deep_sum,
                #[cfg(feature = "golden_nll")]
                disagrees: golden.disagrees,
                #[cfg(feature = "golden_nll")]
                max_abs_delta: golden.max_abs_delta,
            });
        }
        // Restore the plain rope table for the next arm (the pprope arm's
        // swap must not leak).
        if *arm == Arm::SiPprope {
            rig.rope_freq = &plain_table;
        }
        println!(
            "# arm {} done in {:.1}s",
            arm.name(),
            t.elapsed().as_secs_f32()
        );
        results.push((*arm, rung_results));
    }

    // ── the G3 table ──
    println!("\n# G3 zero-shot split — pooled ppl over {score_window}-token scored spans");
    println!(
        "# {:>8} {:>14}",
        "length",
        rungs
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(" ")
            .truncate_to_width(14)
    );
    let base = &results[0].1;
    for (arm, rung_list) in &results {
        let total_nll: f64 = rung_list.iter().map(|r| r.nll_sum).sum();
        let total_t: usize = rung_list.iter().map(|r| r.targets).sum();
        let pooled = (total_nll / total_t.max(1) as f64).exp();
        let worst_delta: f64 = rung_list
            .iter()
            .zip(base.iter())
            .map(|(r, b)| (r.ppl() - b.ppl()) / b.ppl())
            .fold(0.0f64, |a, d| a.max(d.abs()));
        println!(
            "# {:>8} pooled_ppl {:.4} worst |Δppl/ppl_base| {:.4} ({})",
            arm.name(),
            pooled,
            worst_delta,
            match *arm {
                Arm::Base => "reference".to_string(),
                Arm::Si | Arm::SiCal => {
                    if worst_delta <= tolerance {
                        "PASS".to_string()
                    } else {
                        "FAIL".to_string()
                    }
                }
                Arm::SiPprope => format!(
                    "{} (EXPECTED FAIL pre-recorded — G3 (iii))",
                    if worst_delta > tolerance {
                        "FAIL"
                    } else {
                        "unexpected PASS"
                    }
                ),
            }
        );
        for (r, b) in rung_list.iter().zip(base.iter()) {
            println!(
                "#   L={:>5} ppl {:.4} Δ {:+.4} ({} tokens)",
                r.length,
                r.ppl(),
                (r.ppl() - b.ppl()) / b.ppl(),
                r.targets
            );
        }
    }

    // ── the golden-NLL report (Issue 959 T6) ──
    #[cfg(feature = "golden_nll")]
    if golden_nll {
        let (base_arm, rung_list) = &results[0];
        debug_assert_eq!(base_arm, &Arm::Base);
        let total_t: usize = rung_list.iter().map(|r| r.targets).sum();
        let total_inc: f64 = rung_list.iter().map(|r| r.nll_sum).sum();
        let total_golden: f64 = rung_list.iter().map(|r| r.golden_sum).sum();
        let conf_t: usize = rung_list.iter().map(|r| r.confident).sum();
        let conf_golden: f64 = rung_list.iter().map(|r| r.golden_confident_sum).sum();
        let deep_t: usize = rung_list.iter().map(|r| r.deep).sum();
        let deep_golden: f64 = rung_list.iter().map(|r| r.golden_deep_sum).sum();
        let disagrees: usize = rung_list.iter().map(|r| r.disagrees).sum();
        let max_delta: f64 = rung_list
            .iter()
            .map(|r| r.max_abs_delta)
            .fold(0.0f64, f64::max);
        println!(
            "\n# golden-NLL readout (Issue 959 T6, arXiv:2610.31642 Eq. 5) — the tail-faithful \
oracle over the same scored positions; the lossy-quant gates diff THEIR variant \
checkpoints through this readout"
        );
        println!(
            "#   pooled: incumbent f64-lse ppl {:.6} · golden f64-oracle ppl {:.6} · \
Δ {:+.3e} nats/token (mean over {total_t} scored)",
            (total_inc / total_t.max(1) as f64).exp(),
            (total_golden / total_t.max(1) as f64).exp(),
            (total_golden - total_inc) / total_t.max(1) as f64
        );
        println!(
            "#   confident band (margin ≥ 16.64 nats, the fp32 cliff): {conf_t}/{total_t} positions, \
pooled golden ppl {} — the band where an fp32-consumer readout reads false zeros",
            if conf_t > 0 {
                format!("{:.6}", (conf_golden / conf_t as f64).exp())
            } else {
                "n/a (empty band)".to_string()
            }
        );
        println!(
            "#   deep band (margin ≥ 36.73 nats, the f64 cliff): {deep_t}/{total_t} positions, \
pooled golden ppl {} — the band where even an f64 max-subtract LSE absorbs the tail",
            if deep_t > 0 {
                format!("{:.6}", (deep_golden / deep_t as f64).exp())
            } else {
                "n/a (empty band)".to_string()
            }
        );
        println!(
            "#   incumbent-vs-golden: {disagrees}/{total_t} positions disagree AT ALL, \
max |Δnll| {max_delta:.3e} nats — on a healthy model this is f64-composition noise; \
a lossy variant shows its true deltas here"
        );
        for r in rung_list {
            println!(
                "#   L={:>5} golden_ppl {:.4} confident {}/{}, deep {}/{}",
                r.length,
                (r.golden_sum / r.targets.max(1) as f64).exp(),
                r.confident,
                r.targets,
                r.deep,
                r.targets
            );
        }
    }
    #[cfg(not(feature = "golden_nll"))]
    let _ = golden_nll; // the runtime flag already refused above

    // ── the retrieval probe ──
    if retrieval {
        println!("\n# retrieval probe (needle at the head, question at the tail, 6 greedy tokens)");
        const FACTS: [(&str, &str, &str); 2] = [
            (
                "The codeword of the amber lantern is ZEBRA.",
                "zebra",
                "Question: What is the codeword of the amber lantern? Answer:",
            ),
            (
                "The password of the blue gate is RIVER.",
                "river",
                "Question: What is the password of the blue gate? Answer:",
            ),
        ];
        const FILLER: &str =
            " The merchant walked along the road and counted his coins under the afternoon sun.";
        const RETRIEVAL_RUNGS: [usize; 2] = [1024, 2048];
        for &l in &RETRIEVAL_RUNGS {
            let filler_tokens = tok.encode(FILLER);
            for (arm, _) in &results {
                if *arm != Arm::Base && !matches!(arm, Arm::Si | Arm::SiCal | Arm::SiPprope) {
                    continue;
                }
                // Re-arm the session exactly as the ppl loop did.
                match *arm {
                    Arm::Base => rig.set_session(None),
                    Arm::Si => rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                        SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
                    )))),
                    Arm::SiCal => {
                        let cal = calibration.as_ref().expect("built").clone();
                        let plan = SiAttentionPlan::calibrated(
                            lut.clone(),
                            cal,
                            config.n_head,
                            weights.layers.len(),
                        )
                        .map_err(anyhow::Error::msg)?;
                        rig.set_session(Some(SiAttentionSession::Armed(Arc::new(plan))));
                    }
                    Arm::SiPprope => {
                        let table = pp_table.as_ref().expect("checked above");
                        rig.rope_freq = table;
                        rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                            SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
                        ))));
                    }
                }
                if *arm == Arm::SiPprope && pp_table.is_none() {
                    continue;
                }
                let mut row = format!("#   L={l:>5} {}", arm.name());
                for (fact, want, question) in FACTS {
                    let fact_tokens = tok.encode(fact);
                    let question_tokens = tok.encode(question);
                    let budget = l.saturating_sub(fact_tokens.len() + question_tokens.len() + 1);
                    let k = budget / filler_tokens.len().max(1);
                    let mut prompt = vec![tok.bos_id()];
                    prompt.extend(fact_tokens.iter());
                    for _ in 0..k {
                        prompt.extend(filler_tokens.iter());
                    }
                    prompt.extend(question_tokens.iter());
                    if prompt.len() > config.block_size - 8 {
                        row.push_str("  SKIP(prompt>block)");
                        continue;
                    }
                    rig.cache.reset();
                    let feed_to = prompt.len() - 1;
                    let mut nll = 0.0;
                    let mut targets = 0usize;
                    #[cfg(feature = "golden_nll")]
                    let mut ret_golden = GoldenAccum::new(0);
                    #[cfg(not(feature = "golden_nll"))]
                    let mut ret_golden = GoldenAccum::inert();
                    rig.feed(
                        &prompt,
                        0,
                        feed_to,
                        usize::MAX,
                        &mut x,
                        &mut nll,
                        &mut targets,
                        &mut ret_golden,
                    );
                    let generated = rig.greedy(prompt[feed_to], feed_to, 6, &mut x);
                    let text_out = tok.decode(&generated).to_lowercase();
                    let hit = text_out.contains(want);
                    row.push_str(&format!(
                        "  [{}: {}]",
                        want,
                        if hit { "HIT" } else { "MISS" }
                    ));
                }
                if *arm == Arm::SiPprope {
                    rig.rope_freq = &plain_table;
                }
                println!("{row}");
            }
        }
    }

    println!("\n# verdict law: si / si-cal PASS iff every rung's |Δppl|/ppl_base ≤ {tolerance}");
    println!("# si-pprope: the FAIL is the PRE-RECORDED expectation (Plan 622 G3 (iii)) —");
    println!("#   the quality claim lives in riir-train Plan 449, not in zero-shot pairing.");

    Ok(())
}

/// Tiny width clamp for the rung header row.
trait TruncateWidth {
    fn truncate_to_width(self, w: usize) -> String;
}
impl TruncateWidth for String {
    fn truncate_to_width(self, w: usize) -> String {
        if self.len() <= w {
            self
        } else {
            self.chars().take(w).collect()
        }
    }
}
