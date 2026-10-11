//! `si_zero_shot_ppl_gpu` — Plan 620 (Plan 622 Phase 7): the GPU G3 harness.
//!
//! The CPU `si_zero_shot_ppl` port driving the cudarc whole-forward
//! (`forward_token_graph`, the ~90 tok/s class) — the FAST re-gate lane for
//! the G3 zero-shot split and the App-H τ-sweep enabler (a full CPU run is
//! ~2 h/arm; this one is minutes).
//!
//! Arms: `base` (the plain path) · `si` (armed, uncalibrated) · `si-cal`
//! (armed from a frozen `HeadCalibration` sidecar — the σ probe is a
//! CPU-forward instrument, never re-run here) · `si-pprope` (SKIPped with
//! the reason on this lane: the GPU rope kernels compute `inv_freq` inline
//! from `theta_base` — there is no device rope table for the pp-RoPE
//! mask to ride; the CPU run carries the pre-recorded expected-FAIL row).
//!
//! Same protocol as the CPU harness: 16 log-spaced rungs 128→2048, the last
//! `--score-window` (128) targets scored per rung, pooled ppl + worst
//! per-rung Δ vs `base`, the `--tolerance` (5%) verdict on every rung.
//!
//! **Bit-identity gate** (the Phase-A law): `--arms base --dump-logits N`
//! prints the first four logits' f32 bits after each of the first N fed
//! tokens. This bin requires ONLY `cuda_graphs_forward` + the folded loader
//! — NOT `scale_invariant_attn` — so the same command runs on an UNHOOKED
//! build and the hex rows compare byte-for-byte across builds.
//!
//! Usage:
//! ```text
//! cargo build --release -p riir-infer-gpu --features cuda_graphs_forward,riir-infer-core/bonsai2_hadamard \
//!   --bin si_zero_shot_ppl_gpu
//! ./target/release/si_zero_shot_ppl_gpu --gguf <pq2_0.gguf> --corpus <text> \
//!   [--arms base,si,si-cal] [--cal-sidecar <head_calibration.bin>] \
//!   [--max-length 2048] [--score-window 128] [--tau 10.0] [--retrieval] \
//!   [--dump-logits 8]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;
use riir_infer_gpu::ternary_deltanet_gpu_forward_cudarc::TernaryDeltanetGpuForwardCudarc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Base,
    Si,
    SiCal,
    SiPprope,
}

impl Arm {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "base" => Ok(Arm::Base),
            "si" => Ok(Arm::Si),
            "si-cal" => Ok(Arm::SiCal),
            "si-pprope" => Ok(Arm::SiPprope),
            other => Err(format!("unknown arm {other} (base|si|si-cal|si-pprope)")),
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

/// One length rung's scoring result (the CPU harness's row).
#[derive(Debug, Clone)]
struct Rung {
    length: usize,
    nll_sum: f64,
    targets: usize,
}

impl Rung {
    fn ppl(&self) -> f64 {
        (self.nll_sum / self.targets.max(1) as f64).exp()
    }
}

/// The raw-arg set the main parsed off argv, threaded to `run_arms`.
struct Args {
    gguf_path: PathBuf,
    corpus: PathBuf,
    arms: Vec<Arm>,
    max_length: usize,
    min_length: usize,
    rungs_n: usize,
    score_window: usize,
    tolerance: f64,
    tau: f32,
    retrieval: bool,
    cal_sidecar: Option<PathBuf>,
    dump_logits: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut arms = vec![Arm::Base, Arm::Si, Arm::SiCal];
    let mut max_length = 2048usize;
    let mut min_length = 128usize;
    let mut rungs_n = 16usize;
    let mut score_window = 128usize;
    let mut tolerance = 0.05f64;
    let mut tau = 10.0f32;
    let mut retrieval = false;
    let mut cal_sidecar: Option<PathBuf> = None;
    let mut dump_logits = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(args.next().ok_or("--gguf needs a path")?)),
            "--corpus" => corpus = Some(PathBuf::from(args.next().ok_or("--corpus needs a path")?)),
            "--arms" => {
                arms = args
                    .next()
                    .ok_or("--arms needs a list")?
                    .split(',')
                    .map(Arm::parse)
                    .collect::<Result<Vec<_>, String>>()?;
                if !arms.contains(&Arm::Base) {
                    return Err("the base arm is the reference side — always included".into());
                }
            }
            "--max-length" => {
                max_length = args.next().ok_or("--max-length needs n")?.parse().map_err(|_| "--max-length: bad usize")?
            }
            "--min-length" => {
                min_length = args.next().ok_or("--min-length needs n")?.parse().map_err(|_| "--min-length: bad usize")?
            }
            "--rungs" => rungs_n = args.next().ok_or("--rungs needs n")?.parse().map_err(|_| "--rungs: bad usize")?,
            "--score-window" => {
                score_window = args
                    .next()
                    .ok_or("--score-window needs n")?
                    .parse()
                    .map_err(|_| "--score-window: bad usize")?
            }
            "--tolerance" => {
                tolerance = args
                    .next()
                    .ok_or("--tolerance needs x")?
                    .parse()
                    .map_err(|_| "--tolerance: bad f64")?
            }
            "--tau" => tau = args.next().ok_or("--tau needs x")?.parse().map_err(|_| "--tau: bad f32")?,
            "--retrieval" => retrieval = true,
            "--cal-sidecar" => {
                cal_sidecar = Some(PathBuf::from(args.next().ok_or("--cal-sidecar needs a path")?))
            }
            "--dump-logits" => {
                dump_logits = args.next().ok_or("--dump-logits needs n")?.parse().map_err(|_| "--dump-logits: bad usize")?
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        gguf_path: gguf_path.ok_or("--gguf is required")?,
        corpus: corpus.ok_or("--corpus is required")?,
        arms,
        max_length,
        min_length,
        rungs_n,
        score_window,
        tolerance,
        tau,
        retrieval,
        cal_sidecar,
        dump_logits,
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    assert!(
        args.min_length >= 4 && args.max_length > args.min_length,
        "4 <= min < max"
    );
    assert!(
        args.score_window >= 8 && args.score_window <= args.min_length,
        "8 <= score-window <= min-length"
    );
    if args.arms.contains(&Arm::SiCal) && args.cal_sidecar.is_none() {
        return Err(
            "si-cal needs --cal-sidecar <head_calibration.bin> — the sigma probe is a \
             CPU-forward instrument (riir-infer si_zero_shot_ppl --save-cal); the GPU lane \
             consumes the frozen sidecar, never per-head score downloads"
                .into(),
        );
    }

    // ── load once, share across arms (the CPU harness's law) ──
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.gguf_path)
        .map_err(|e| format!("load {}: {e}", args.gguf_path.display()))?;
    println!(
        "# si_zero_shot_ppl_gpu: {} | layers={} heads={}/{} rot={} | load {:.1}s",
        args.gguf_path.display(),
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        weights.rotation.is_some(),
        t0.elapsed().as_secs_f32()
    );

    // The row_logit_floor law: cap the cache window at what the run needs,
    // BEFORE the forward (and its device caches) is constructed.
    config.block_size = args.max_length + 32;

    let gguf = GgufFile::open(&args.gguf_path).map_err(|e| format!("re-open: {e}"))?;
    let tok = BpeTokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
    drop(gguf);

    // ── the frozen sequence (byte-identical for every arm) ──
    let text = load_corpus_text(&args.corpus).map_err(|e| format!("corpus: {e}"))?;
    let mut seq = tok.encode_with_bos(&text);
    seq.truncate(args.max_length + 1);
    if seq.len() < args.min_length + 1 {
        return Err(format!(
            "corpus too short: {} tokens (BOS incl.) < min-length {} + 1",
            seq.len(),
            args.min_length
        ));
    }
    println!(
        "# corpus: {} tokens (BOS + text) | seq[0]=BOS={} sink carve-out armed | τ = {}",
        seq.len(),
        tok.bos_id(),
        args.tau
    );

    // The 16 log-spaced rungs — identical arithmetic to the CPU harness.
    let rungs: Vec<usize> = (0..args.rungs_n)
        .map(|k| {
            let l = args.min_length as f64
                * (args.max_length as f64 / args.min_length as f64)
                    .powf(k as f64 / (args.rungs_n - 1) as f64);
            l.round() as usize
        })
        .collect();
    println!(
        "# rungs ({}): {:?} | scored span {} tokens per rung",
        rungs.len(),
        rungs,
        args.score_window
    );

    // ── the GPU forward (one construction, shared by every arm) ──
    let t_gpu = Instant::now();
    let mut fwd = TernaryDeltanetGpuForwardCudarc::new(&config, &weights)
        .map_err(|e| format!("GPU forward construct: {e}"))?;
    println!("# gpu forward up in {:.1}s", t_gpu.elapsed().as_secs_f32());

    // ── optional: the bit-identity dump (base posture, then exit) ──
    if args.dump_logits > 0 {
        dump_logits(&mut fwd, &seq, args.dump_logits)?;
        return Ok(());
    }

    let results = run_arms(&args, &mut fwd, &config, &weights, &tok, &seq, &rungs)?;

    // ── the G3 table (the CPU harness's printer, verbatim shape) ──
    println!("\n# G3 zero-shot split — pooled ppl over {}-token scored spans", args.score_window);
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
                    if worst_delta <= args.tolerance {
                        "PASS".to_string()
                    } else {
                        "FAIL".to_string()
                    }
                }
                // Unreachable on this lane today (the arm SKIPs at plan
                // build) — kept for parity with the CPU printer.
                Arm::SiPprope => "SKIP (no device rope table — see the module docs)".to_string(),
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

    if args.retrieval {
        println!("# retrieval probe: not ported to the GPU lane yet (Plan 620 Phase C follow-up)");
    }
    Ok(())
}

/// The arm loop: one session state per arm, the rung ladder per arm. The GPU
/// rung reset is `reset_state` (in-place zeroing — the captured graph stays
/// valid, Issue 696).
fn run_arms(
    args: &Args,
    fwd: &mut TernaryDeltanetGpuForwardCudarc,
    config: &Config,
    weights: &QwenDeltaNetTernaryWeights,
    tok: &BpeTokenizer,
    seq: &[usize],
    rungs: &[usize],
) -> Result<Vec<(Arm, Vec<Rung>)>, String> {
    let _ = tok;
    // Both postures reference these only inside cfg'd blocks; the discards
    // keep the off-posture build warning-free.
    let _ = config;
    let _ = weights;

    #[cfg(feature = "scale_invariant_attn")]
    let lut = katgpt_core::scale_invariant::ScaleInvariantLut::build(args.tau, config.block_size);
    #[cfg(feature = "scale_invariant_attn")]
    println!(
        "# LUT: max_distance {} (block {})",
        lut.max_distance(),
        config.block_size
    );

    // si-cal: the frozen sidecar (CPU-probe instrument; GPU consumes).
    #[cfg(feature = "scale_invariant_attn")]
    let calibration = if args.arms.contains(&Arm::SiCal) {
        let path = args.cal_sidecar.as_ref().expect("checked at parse");
        let bytes = std::fs::read(path).map_err(|e| format!("cal sidecar {}: {e}", path.display()))?;
        let cal = katgpt_core::scale_invariant::HeadCalibration::from_sidecar(&bytes)
            .map_err(|e| format!("cal sidecar: {e}"))?;
        println!(
            "# cal sidecar: {} slots, digest {:02x}… (host-frozen — quote host + digest beside every row)",
            cal.heads(),
            cal.digest()[..4]
                .iter()
                .fold(0u64, |a, b| (a << 8) | u64::from(*b))
                & 0xffff
        );
        Some(cal)
    } else {
        None
    };

    // Build every arm's session up front; attach per arm (attach/detach
    // invalidate the graph — one recapture per arm, amortized over the rungs).
    #[cfg(feature = "scale_invariant_attn")]
    let sessions: Vec<(Arm, Option<riir_infer_gpu::SiGpuSession>)> = {
        let n_head = config.n_head;
        let n_layers = weights.layers.len();
        let stream = fwd.infra.stream.clone();
        let mut v = Vec::with_capacity(args.arms.len());
        for arm in &args.arms {
            let s = match arm {
                Arm::Base => None,
                Arm::Si | Arm::SiPprope => Some(
                    riir_infer_gpu::SiGpuSession::attach(&stream, &lut, None, n_layers, n_head)
                        .map_err(|e| format!("si session: {e}"))?,
                ),
                Arm::SiCal => Some(
                    riir_infer_gpu::SiGpuSession::attach(
                        &stream,
                        &lut,
                        calibration.as_ref(),
                        n_layers,
                        n_head,
                    )
                    .map_err(|e| format!("si-cal session: {e}"))?,
                ),
            };
            v.push((*arm, s));
        }
        v
    };
    #[cfg(not(feature = "scale_invariant_attn"))]
    let sessions: Vec<(Arm, ())> = args
        .arms
        .iter()
        .map(|arm| {
            if *arm != Arm::Base {
                eprintln!(
                    "# arm {}: SKIP — built without scale_invariant_attn (add the feature for the si arms)",
                    arm.name()
                );
            }
            (*arm, ())
        })
        .collect();

    let mut results: Vec<(Arm, Vec<Rung>)> = Vec::new();
    for (arm, session) in sessions {
        // The pp-RoPE arm: honest SKIP on the GPU lane (no device rope table
        // for the pp mask to ride — the CPU run carries the expected-FAIL row).
        if arm == Arm::SiPprope {
            println!(
                "# arm {}: SKIP — the GPU rope kernels compute inv_freq inline from theta_base; \
                 there is no device rope table for the pp-RoPE mask to ride (Plan 620). \
                 The CPU harness's pre-recorded expected-FAIL row stands.",
                arm.name()
            );
            continue;
        }

        // Session attach/detach (graph invalidates — one recapture per arm).
        #[cfg(feature = "scale_invariant_attn")]
        match session {
            Some(s) => {
                fwd.attach_si(s);
                fwd.arm_si(true).map_err(|e| e.to_string())?;
            }
            None => fwd.detach_si(),
        }
        #[cfg(not(feature = "scale_invariant_attn"))]
        let _ = &session; // the off-arm tuple's () — no session to consume

        let t = Instant::now();
        let mut rung_results = Vec::with_capacity(rungs.len());
        for &l in rungs {
            fwd.reset_state().map_err(|e| e.to_string())?;
            let mut nll = 0.0f64;
            let mut targets = 0usize;
            let score_from = l.saturating_sub(args.score_window);
            for pos in 0..=l {
                let logits = fwd
                    .forward_token_graph(seq[pos])
                    .map_err(|e| e.to_string())?;
                if pos >= score_from && pos + 1 < seq.len() {
                    let target = seq[pos + 1];
                    // f64 logsumexp over the raw logits (the CPU harness's
                    // scoring, byte-for-byte shape).
                    let mut max_l = f32::NEG_INFINITY;
                    for &lg in &logits {
                        max_l = max_l.max(lg);
                    }
                    let mut sum = 0.0f64;
                    for &lg in &logits {
                        sum += (f64::from(lg) - f64::from(max_l)).exp();
                    }
                    let lse = f64::from(max_l) + sum.ln();
                    nll += lse - f64::from(logits[target]);
                    targets += 1;
                }
            }
            rung_results.push(Rung {
                length: l,
                nll_sum: nll,
                targets,
            });
        }
        println!("# arm {} done in {:.1}s", arm.name(), t.elapsed().as_secs_f32());
        results.push((arm, rung_results));
    }

    // Leave the incumbent posture behind (the next consumer starts clean).
    #[cfg(feature = "scale_invariant_attn")]
    fwd.detach_si();
    Ok(results)
}

/// The Phase-A bit-identity dump: feed the first `n_tokens` of the corpus
/// at the base posture, print the first four logits' f32 bits after each —
/// deterministic rows that must compare BYTE-EQUAL across the incumbent and
/// hooked builds (same feature set, same corpus, same checkpoint).
fn dump_logits(
    fwd: &mut TernaryDeltanetGpuForwardCudarc,
    seq: &[usize],
    n_tokens: usize,
) -> Result<(), String> {
    println!("# dump-logits: first 4 logits' f32 bits after each of the first {n_tokens} fed tokens");
    for (pos, &tok_id) in seq.iter().enumerate().take(n_tokens) {
        let logits = fwd
            .forward_token_graph(tok_id)
            .map_err(|e| e.to_string())?;
        let bits: Vec<String> = logits[..4.min(logits.len())]
            .iter()
            .map(|f| format!("{:08x}", f.to_bits()))
            .collect();
        println!("pos={pos} tok={tok_id} {}", bits.join(" "));
    }
    Ok(())
}
