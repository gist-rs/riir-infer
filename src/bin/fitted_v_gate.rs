//! fitted_v_gate — riir-infer Issue 013 T1: the model-bound P1 gate for
//! katgpt-core's token-mean-removed V quantization (the
//! `MeanRemovedValueCache` product, katgpt-rs `0b768e95d` / Bench 895) on
//! the REAL gemma-2-2b decode path — the gates katgpt-rs cannot run.
//!
//! One process, four phases:
//!
//! - **A calibrate** — `forward_gemma2_f16_tapped` over `tokens[0..cal_n]`
//!   (chunks of `seq_len`, fresh cache per chunk) feeding
//!   `LayeredVkCalibration` (the Bench-004 protocol; pre-RoPE K / post-W_V
//!   V taps — the tap-point law).
//! - **B freeze** — `FittedTokenTable::from_calibration(ValueMean, λ_js=0)`
//!   at `top_k`, plus the truncated dial-k table (rows are rank-ordered, so
//!   the dial is a remap, not a second pass). Prints per-layer ρ_l(V) —
//!   Gate 1's predictor — and the sink-norm evidence (Gate 4a).
//! - **C eval** — `tokens[cal_n..cal_n+eval_n]`, seq = BOS + (seq_len−1)
//!   corpus tokens per chunk, one full pass per arm through
//!   `forward_gemma2_f16_hk` (arm-major: the base arm completes first and
//!   fills the paired-delta record every later arm indexes). Arms:
//!   `f16` (base) | `p-b{b}` | `mr-b{b}` for each `b ∈ bits` |
//!   `mr-b{last}-k{dial}` (the coverage dial). Per scored position: paired
//!   ΔNLL + top-1 flip vs base; per TARGET-token frequency band (Gate 3);
//!   per position bin `t_n < 32` (the sink window — the `kv_sink_window`
//!   n_sink convention) vs body (Gate 4b); per-layer quantizer telemetry
//!   (Gate 1). The report is rewritten after every arm (a mid-run death
//!   keeps every completed arm's measurements — the Bench-009-scan-1
//!   lesson, applied at arm granularity).
//! - **D gates** — the pre-registered thresholds (the issue file carries
//!   the registering commit; echoed in the report).
//!
//! ## The four gates (pre-registered)
//!
//! 1. **Prediction vs measurement**: measured MSE ratio mr/plain vs
//!    `1 − ρ_l(V)`; ±0.20 per layer on ≥ 20/26 layers, ±0.10 on the layer
//!    mean, at 3 and 4 bits (2-bit: direction-only — the absmax caveat is
//!    the arbiter; `max_abs_enc` vs `max_abs` recorded per layer).
//! 2. **PPL at matched bits**: mean paired ΔNLL(mr − plain) < 0 at every
//!    bit width; secondary: chunk-paired win share > 0.5.
//! 3. **Per-band conditional retention** (the Orthrus law): flips/ΔNLL by
//!    token-frequency band — the recorded conditional view; the Bench-895
//!    caveat predicts 2-bit off-mean worsening concentrated in the top band.
//! 4. **Sinks**: `‖E^V[BOS]‖` vs the median tracked-row norm (the table
//!    absorbs the sink mean), and the sink-window bin must not flip worse
//!    under mr than plain at any bits.
//!
//! MEASUREMENT-ONLY: no promotion claim lives here (the promotion is
//! katgpt-rs-side and waits on these gates).
//!
//! Usage:
//! ```text
//! fitted_v_gate <gguf> <corpus> [--cal-tokens N] [--eval-tokens N]
//!               [--bits 2,3,4] [--top-k N] [--dial-k N] [--seq-len N]
//!               [--report PATH] [--smoke]
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use katgpt_core::fitted_value_table::{FittedTokenTable, VkSignal};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::{config_from_gguf_metadata, GgufFile};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::{
    forward_gemma2_f16, forward_gemma2_f16_hk, ForwardContext, NoHook, NoVQuant,
};
use riir_infer_core::transformer::gemma2_calibration::{
    forward_gemma2_f16_tapped, load_gemma2_f16_direct, CalibrationTables,
};
use riir_infer_core::transformer::gemma2_vquant::{LayerMse, VQuantState};
use riir_infer_core::types::kv_dim;

/// Frequency-band edges over the tracked rows (rows are rank-ordered by
/// the calibration's count-desc order): top-64 / 64–1024 / 1024–top_k.
const BAND_EDGES: [usize; 2] = [64, 1024];

/// The sink-window position bin (the `kv_sink_window::SinkWindowPolicy`
/// n_sink convention puts sinks at positions `[0, n_sink)`; the bin is the
/// wider early-position window that leans sink-mass).
const SINK_BIN_POS: usize = 32;

/// The KVarN tile (the recorded instantiation; whole tiles per chunk).
const TILE: usize = 128;

/// Report-time arm descriptor.
#[derive(Clone, Copy)]
enum ArmKind {
    /// Unquantized f16 V — the paired base.
    Base,
    /// Plain KVarN V quant at `bits` — the matched-bits control.
    Plain(u8),
    /// The P1 token-mean arm over KVarN at `bits` (`dial` selects the
    /// truncated table — the coverage dial).
    MeanRemoved { bits: u8, dial: bool },
}

struct ArmSpec {
    name: String,
    kind: ArmKind,
}

/// One arm's running totals.
struct Acc {
    /// Scored positions (every arm scores the same set).
    k: usize,
    sum_nll: f64,
    flips: usize,
    sum_abs: f64,
    /// Per-band [flips, Σ|ΔNLL|, ΣΔNLL] (paired, quant arms only).
    band: [[f64; 3]; 4],
    /// Per position-bin [flips, Σ|ΔNLL|] (0 = sink window, 1 = body).
    posbin: [[f64; 2]; 2],
    /// Per-chunk ΣNLL (quant arms only; the paired win-share axis).
    chunk_d: Vec<f64>,
    /// Per-layer quantizer telemetry (quant arms only).
    tel: Vec<LayerMse>,
    fwd: usize,
    secs: f64,
}

impl Acc {
    fn new(n_layers: usize) -> Self {
        Self {
            k: 0,
            sum_nll: 0.0,
            flips: 0,
            sum_abs: 0.0,
            band: [[0.0; 3]; 4],
            posbin: [[0.0; 2]; 2],
            chunk_d: Vec::new(),
            tel: vec![LayerMse::default(); n_layers],
            fwd: 0,
            secs: 0.0,
        }
    }
}

/// The run-static inputs the report renderer reads.
struct ReportCtx<'a> {
    r2: &'a [f32],
    kvd: usize,
    total_scored: usize,
    band_n: &'a [f64; 4],
    posbin_n: &'a [f64; 2],
    sink_block: &'a str,
    top_k: usize,
    dial_k: usize,
    seq_len: usize,
    cal_n: usize,
    eval_n: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: fitted_v_gate <gguf> <corpus> [--cal-tokens N] [--eval-tokens N] \
             [--bits 2,3,4] [--top-k N] [--dial-k N] [--seq-len N] [--report PATH] [--smoke]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let smoke = args.iter().any(|a| a == "--smoke");
    let (mut cal_n, mut eval_n) = if smoke { (2048, 2048) } else { (61_440, 12_288) };
    let mut bits: Vec<u8> = if smoke { vec![4] } else { vec![2, 3, 4] };
    let mut top_k: usize = 8192;
    let mut dial_k: usize = 1024;
    let mut seq_len: usize = 1024;
    let mut report_path: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--cal-tokens" => {
                cal_n = args[i + 1].parse().context("--cal-tokens N")?;
                i += 2;
            }
            "--eval-tokens" => {
                eval_n = args[i + 1].parse().context("--eval-tokens N")?;
                i += 2;
            }
            "--bits" => {
                bits = args[i + 1]
                    .split(',')
                    .map(|s| s.parse().context("--bits 2,3,4"))
                    .collect::<Result<_>>()?;
                i += 2;
            }
            "--top-k" => {
                top_k = args[i + 1].parse().context("--top-k N")?;
                i += 2;
            }
            "--dial-k" => {
                dial_k = args[i + 1].parse().context("--dial-k N")?;
                i += 2;
            }
            "--seq-len" => {
                seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--smoke" => i += 1,
            other => bail!("unknown arg {other}"),
        }
    }
    if seq_len > 4096 {
        bail!("--seq-len must stay <= 4096 (gemma-2 sliding window; see vk_calibration's module doc)");
    }
    if !seq_len.is_multiple_of(TILE) {
        bail!("--seq-len must be a multiple of {TILE} (whole KVarN tiles per chunk)");
    }
    if dial_k >= top_k {
        bail!("--dial-k must be < --top-k (the dial is a truncation)");
    }

    // ── Load model + tokenizer from ONE open GGUF (mmap-backed reads) ──
    let t_start = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let mut config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    let bos = tok.bos_id();
    println!(
        "# fitted_v_gate: gemma-2 f16 | layers={} n_embd={} kv_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        kv_dim(&config),
        config.vocab_size,
        t_start.elapsed().as_secs_f32()
    );
    drop(gguf);

    // ── Corpus: natural text → one token stream ──────────────────────
    let text = load_corpus_text(&corpus_path)?;
    let all_tokens = tok.encode(&text);
    if all_tokens.len() < cal_n + eval_n + seq_len {
        bail!(
            "corpus too short: {} tokens < cal {cal_n} + eval {eval_n} + seq {seq_len}",
            all_tokens.len()
        );
    }
    println!(
        "# corpus: {} chars → {} tokens | cal [0..{cal_n}) eval [{cal_n}..{}) seq_len={seq_len}",
        text.len(),
        all_tokens.len(),
        cal_n + eval_n
    );

    // The caches are sized to the chunk, not the model window (the
    // row_logit_floor_ppl cap — a 131K-context model would otherwise
    // allocate its whole advertised window).
    config.block_size = seq_len;
    let kvd = kv_dim(&config);
    let n_layers = config.n_layer;

    // ── Phase A: calibrate (the Bench-004 tapped protocol) ───────────
    let cal_tokens: Vec<usize> = all_tokens[..cal_n].to_vec();
    let mut counts = vec![0u64; config.vocab_size];
    for &t in &cal_tokens {
        counts[t] += 1;
    }
    let t_cal = Instant::now();
    let mut tables = CalibrationTables::from_counts(n_layers, kvd, counts, top_k);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut done = 0usize;
    let mut next_report = 0usize;
    for chunk in cal_tokens.chunks(seq_len) {
        cache.reset();
        for (pos, &token) in chunk.iter().enumerate() {
            forward_gemma2_f16_tapped(
                &mut ctx, &weights, &mut cache, &mut tables, token, pos, &config, &mut NoHook,
            );
        }
        done += chunk.len();
        if done >= next_report {
            let rate = done as f32 / t_cal.elapsed().as_secs_f32().max(1e-6);
            println!(
                "# calibrate {done}/{cal_n} | {:.0} tok/s | eta {:.0} min",
                rate,
                (cal_n - done) as f32 / (rate + 1e-6) / 60.0
            );
            let _ = std::io::stdout().flush();
            next_report = (done + cal_n / 20).max(done + 1);
        }
    }
    println!(
        "# calibrated in {:.0}s ({:.0} tok/s) | box: 4090 workstation i7-13700K, CPU lane",
        t_cal.elapsed().as_secs_f32(),
        cal_n as f32 / t_cal.elapsed().as_secs_f32().max(1e-6)
    );

    // ── Phase B: freeze (both tables leak — process-lifetime by design;
    //    VQuantState holds &'static and outlives nothing but the process)
    let table_full: &'static FittedTokenTable = Box::leak(Box::new(
        FittedTokenTable::from_calibration(&tables, VkSignal::ValueMean, 0.0),
    ));
    let table_dial: &'static FittedTokenTable =
        Box::leak(Box::new(truncate_table(table_full, &tables, dial_k)));
    let r2: Vec<f32> = tables
        .layers
        .iter()
        .map(|lt| lt.v.r_squared().aggregate)
        .collect();
    let mean_rho: f64 = r2.iter().map(|&r| r as f64).sum::<f64>() / r2.len() as f64;
    println!(
        "# frozen: top_k={} ({} layers × {} rows × {} wide) + dial top_k={} | mean ρ_l(V)={mean_rho:.4} (Bench-004 shape: 0.48 at 120k)",
        top_k,
        table_full.n_layer(),
        table_full.rows(),
        table_full.width(),
        table_dial.rows()
    );

    // Gate 4a — the table absorbs the sink mean.
    let sink_block = sink_norms_report(table_full, &tables, n_layers / 2, bos);
    print!("{sink_block}");
    let _ = std::io::stdout().flush();

    // ── Phase C: eval arms ────────────────────────────────────────────
    let eval_tokens: Vec<usize> = all_tokens[cal_n..cal_n + eval_n].to_vec();
    let corpus_per_chunk = seq_len - 1; // BOS + (seq_len−1) corpus tokens
    let n_chunks = eval_tokens.len() / corpus_per_chunk;
    if n_chunks == 0 {
        bail!("eval slice smaller than one chunk");
    }
    let total_scored = n_chunks * corpus_per_chunk; // targets 1..seq_len
    println!(
        "# eval: {n_chunks} chunks × {corpus_per_chunk} corpus tokens | {total_scored} scored per arm | bits {bits:?}"
    );

    // Band of each vocab id (0=top-64 1=64–1024 2=tracked 3=tail).
    let band_of = band_table(&tables);

    let mut arms: Vec<ArmSpec> = vec![ArmSpec {
        name: "f16".into(),
        kind: ArmKind::Base,
    }];
    for &b in &bits {
        arms.push(ArmSpec {
            name: format!("p-b{b}"),
            kind: ArmKind::Plain(b),
        });
    }
    for &b in &bits {
        arms.push(ArmSpec {
            name: format!("mr-b{b}"),
            kind: ArmKind::MeanRemoved { bits: b, dial: false },
        });
    }
    arms.push(ArmSpec {
        name: format!("mr-b{}-k{dial_k}", bits[bits.len() - 1]),
        kind: ArmKind::MeanRemoved {
            bits: bits[bits.len() - 1],
            dial: true,
        },
    });

    // Belt-and-braces once per run: the delegation wrapper must be
    // bit-identical to the entry point it replaced (the refactor's G3).
    {
        cache.reset();
        let (pt0, pt1) = (eval_tokens[0], eval_tokens[1]);
        let a0: Vec<u32> = forward_gemma2_f16(&mut ctx, &weights, &mut cache, pt0, 0, &config)
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let b0: Vec<u32> =
            forward_gemma2_f16_hk(&mut ctx, &weights, &mut cache, &mut NoVQuant, pt0, 0, &config)
                .iter()
                .map(|x| x.to_bits())
                .collect();
        assert_eq!(
            a0, b0,
            "forward_gemma2_f16_hk(NoVQuant) diverged from forward_gemma2_f16 at pos 0"
        );
        let a1: Vec<u32> = forward_gemma2_f16(&mut ctx, &weights, &mut cache, pt1, 1, &config)
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let b1: Vec<u32> =
            forward_gemma2_f16_hk(&mut ctx, &weights, &mut cache, &mut NoVQuant, pt1, 1, &config)
                .iter()
                .map(|x| x.to_bits())
                .collect();
        assert_eq!(
            a1, b1,
            "forward_gemma2_f16_hk(NoVQuant) diverged at pos 1 (store+seam+read path)"
        );
        cache.reset();
        println!("# probe: f16_hk(NoVQuant) bit-identical to forward_gemma2_f16 (pos 0+1)");
    }

    let mut accs: Vec<Acc> = arms.iter().map(|_| Acc::new(n_layers)).collect();
    let mut base_nll: Vec<f64> = Vec::with_capacity(total_scored);
    let mut base_top: Vec<usize> = Vec::with_capacity(total_scored);
    let mut band_n = [0f64; 4];
    let mut posbin_n = [0f64; 2];

    let t_eval = Instant::now();
    // Arm-major: the base arm fills the paired record first; every later
    // arm indexes it with its own running `k`.
    for (ai, arm) in arms.iter().enumerate() {
        let mut vq = match arm.kind {
            ArmKind::Base => None,
            ArmKind::Plain(b) => Some(VQuantState::new_plain(
                b,
                n_layers,
                kvd,
                seq_len,
                TILE,
                table_full,
            )),
            ArmKind::MeanRemoved { bits: b, dial } => Some(VQuantState::new_mean_removed(
                b,
                n_layers,
                kvd,
                seq_len,
                TILE,
                if dial { table_dial } else { table_full },
            )),
        };
        let ta = Instant::now();
        let mut k = 0usize;
        for (ci, chunk) in eval_tokens.chunks(corpus_per_chunk).enumerate() {
            if ci >= n_chunks {
                break;
            }
            cache.reset();
            if let Some(state) = vq.as_mut() {
                state.reset();
            }
            let mut seq = Vec::with_capacity(seq_len);
            seq.push(bos);
            seq.extend_from_slice(chunk);
            let mut chunk_sum = 0.0f64;
            for pos in 0..seq.len() {
                let token = seq[pos];
                if let Some(state) = vq.as_mut() {
                    state.set_token(pos, token as u32);
                }
                let logits = match vq.as_mut() {
                    None => {
                        forward_gemma2_f16(&mut ctx, &weights, &mut cache, token, pos, &config)
                    }
                    Some(state) => forward_gemma2_f16_hk(
                        &mut ctx, &weights, &mut cache, state, token, pos, &config,
                    ),
                };
                if pos + 1 >= seq.len() {
                    continue;
                }
                let target = seq[pos + 1];
                let l = nll(logits, target);
                let top = argmax(logits);
                chunk_sum += l;
                let acc = &mut accs[ai];
                acc.fwd += 1;
                acc.sum_nll += l;
                acc.k += 1;
                match arm.kind {
                    ArmKind::Base => {
                        base_nll.push(l);
                        base_top.push(top);
                        band_n[band_of[target] as usize] += 1.0;
                        posbin_n[usize::from(pos >= SINK_BIN_POS)] += 1.0;
                    }
                    _ => {
                        let bl = base_nll[k];
                        let flip = usize::from(top != base_top[k]);
                        acc.flips += flip;
                        acc.sum_abs += (l - bl).abs();
                        let b = band_of[target] as usize;
                        acc.band[b][0] += flip as f64;
                        acc.band[b][1] += (l - bl).abs();
                        acc.band[b][2] += l - bl;
                        let pb = usize::from(pos >= SINK_BIN_POS);
                        acc.posbin[pb][0] += flip as f64;
                        acc.posbin[pb][1] += (l - bl).abs();
                        k += 1;
                    }
                }
            }
            if ai > 0 {
                accs[ai].chunk_d.push(chunk_sum);
            }
        }
        accs[ai].secs += ta.elapsed().as_secs_f64();
        if let Some(state) = &vq {
            for (l, t) in state.layer_mse().iter().enumerate() {
                let d = &mut accs[ai].tel[l];
                d.sq_err += t.sq_err;
                d.sq_ref += t.sq_ref;
                d.rows += t.rows;
                d.max_abs = d.max_abs.max(t.max_abs);
                d.max_abs_enc = d.max_abs_enc.max(t.max_abs_enc);
                d.rows_tracked += t.rows_tracked;
            }
        }
        let a = &accs[ai];
        println!(
            "# arm {} done: {} scored | ppl {:.4} | flip {:.2}% | {:.0} fwd/s | {:.0}s",
            arm.name,
            a.k,
            if a.k > 0 { (a.sum_nll / a.k as f64).exp() } else { 0.0 },
            if a.k > 0 { 100.0 * a.flips as f64 / a.k as f64 } else { 0.0 },
            a.fwd as f64 / a.secs.max(1e-9),
            a.secs
        );
        let _ = std::io::stdout().flush();
        if let Some(p) = &report_path {
            let rctx = ReportCtx {
                r2: &r2,
                kvd,
                total_scored,
                band_n: &band_n,
                posbin_n: &posbin_n,
                sink_block: &sink_block,
                top_k,
                dial_k,
                seq_len,
                cal_n,
                eval_n,
            };
            let out = render_report(&rctx, &arms, &accs, false);
            std::fs::write(p, &out).with_context(|| format!("write report {}", p.display()))?;
        }
    }

    // ── Phase D: the gates (final report) ─────────────────────────────
    let rctx = ReportCtx {
        r2: &r2,
        kvd,
        total_scored,
        band_n: &band_n,
        posbin_n: &posbin_n,
        sink_block: &sink_block,
        top_k,
        dial_k,
        seq_len,
        cal_n,
        eval_n,
    };
    let report = render_report(&rctx, &arms, &accs, true);
    print!("{report}");
    if let Some(p) = &report_path {
        std::fs::write(p, &report).with_context(|| format!("write report {}", p.display()))?;
        eprintln!("# report written: {}", p.display());
    }
    println!(
        "# eval done: {} arms × {total_scored} scored in {:.0}s | box: 4090 workstation i7-13700K, CPU lane",
        arms.len(),
        t_eval.elapsed().as_secs_f64()
    );
    Ok(())
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
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Truncate the frozen table to the top-`k` rows (rows are rank-ordered,
/// so the dial table is a remap — no second calibration pass).
fn truncate_table(
    full: &FittedTokenTable,
    cal: &CalibrationTables,
    k: usize,
) -> FittedTokenTable {
    let nl = full.n_layer();
    let width = full.width();
    let k = k.min(full.rows());
    // Invert row_of_token once: rank → token (every row < top_k has one).
    let mut tok_of_row = vec![u32::MAX; full.rows()];
    for (t, &r) in cal.row_of_token.iter().enumerate() {
        if (r as usize) < tok_of_row.len() {
            tok_of_row[r as usize] = t as u32;
        }
    }
    let mut data = Vec::with_capacity(nl * k * width);
    for l in 0..nl {
        for &tok in tok_of_row.iter().take(k) {
            data.extend_from_slice(full.row(l, tok).expect("rank row present"));
        }
    }
    let row_of_token: Vec<u32> = cal
        .row_of_token
        .iter()
        .map(|&r| if (r as usize) < k { r } else { u32::MAX })
        .collect();
    FittedTokenTable::from_rows(nl, width, row_of_token, data)
}

/// Band per vocab id: 0 = rank < 64, 1 = < 1024, 2 = tracked, 3 = tail.
/// (Every tracked rank is < top_k by construction — rank == row index.)
fn band_table(cal: &CalibrationTables) -> Vec<u8> {
    cal.row_of_token
        .iter()
        .map(|&r| {
            if r == u32::MAX {
                3
            } else if (r as usize) < BAND_EDGES[0] {
                0
            } else if (r as usize) < BAND_EDGES[1] {
                1
            } else {
                2
            }
        })
        .collect()
}

/// Gate 4a: the sink-mean evidence — ‖E^V_l[BOS]‖/√d vs the median tracked
/// row norm, on the first, mid, and last layer. Returns the block (printed
/// live AND embedded in the report).
fn sink_norms_report(
    table: &FittedTokenTable,
    cal: &CalibrationTables,
    mid: usize,
    bos: usize,
) -> String {
    let ln = |r: &[f32]| (r.iter().map(|&x| x * x).sum::<f32>() / r.len() as f32).sqrt();
    // rank → token, once.
    let mut tok_of_row = vec![u32::MAX; table.rows()];
    for (t, &r) in cal.row_of_token.iter().enumerate() {
        if (r as usize) < tok_of_row.len() {
            tok_of_row[r as usize] = t as u32;
        }
    }
    let mut out = String::from(
        "\n# Gate 4a — sink-mean absorption (‖E^V[BOS]‖/√d vs median tracked row)\n\n\
         | layer | ‖E^V[BOS]‖/√d | median row ‖·‖/√d | ratio |\n|---|---|---|---|\n",
    );
    let last = table.n_layer() - 1;
    for l in [0usize, mid, last] {
        let Some(bos_row) = table.row(l, bos as u32) else {
            out.push_str(&format!("| {l} | (BOS untracked) | — | — |\n"));
            continue;
        };
        let rn = ln(bos_row);
        let mut norms: Vec<f32> = tok_of_row
            .iter()
            .filter(|&&t| t != u32::MAX)
            .filter_map(|&t| table.row(l, t))
            .map(ln)
            .collect();
        if norms.is_empty() {
            out.push_str(&format!("| {l} | (empty table) | — | — |\n"));
            continue;
        }
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = norms[norms.len() / 2];
        out.push_str(&format!(
            "| {l} | {rn:.4} | {med:.4} | {:.2}× |\n",
            rn / med.max(1e-9)
        ));
    }
    out.push_str("\n(A ratio » 1 means the BOS sink's V mean is far larger than a typical token's — the table absorbs it as `E^V[BOS]`, so the mean-removed encode quantizes the residual, not the sink.)\n");
    out
}

/// The full report: the arm table, the four gates, telemetry, caveats.
/// Rewritten after every arm (partial) and once final. All thresholds are
/// echoed so the artifact is self-contained.
fn render_report(ctx: &ReportCtx, arms: &[ArmSpec], accs: &[Acc], final_report: bool) -> String {
    let table_mib = |rows: usize| {
        4.0 * ctx.r2.len() as f64 * rows as f64 * ctx.kvd as f64 / (1 << 20) as f64
    };
    let mut out = String::new();
    out.push_str("# Issue 013 T1 — fitted-V P1 gate: gemma-2-2b decode (fitted_v_gate)\n\n");
    out.push_str(&format!(
        "status: {} ({} scored per completed arm / {} total) | arms {}/{} | \
         cal {} eval {} tokens | seq_len {} | top_k {} dial_k {}\n\n",
        if final_report { "FINAL" } else { "PARTIAL" },
        accs.iter().map(|a| a.k).max().unwrap_or(0),
        ctx.total_scored,
        accs.iter().filter(|a| a.k > 0).count(),
        arms.len(),
        ctx.cal_n,
        ctx.eval_n,
        ctx.seq_len,
        ctx.top_k,
        ctx.dial_k
    ));
    out.push_str(
        "Pre-registered gates: (1) |ratio_l − (1−ρ_l)| ≤ 0.20 on ≥ 20/26 layers AND \
         |mean ratio − mean pred| ≤ 0.10 at 3+4 bits (2-bit: direction-only, the absmax \
         caveat is the arbiter) · (2) mean paired ΔNLL(mr−plain) < 0 at every bits, \
         secondary chunk win share > 0.5 · (3) per-band conditional view, recorded \
         honestly · (4) sink-window flips(mr) ≤ flips(plain) at every bits.\n\n",
    );

    // ── Arm table ─────────────────────────────────────────────────────
    out.push_str("## Arms\n\n| arm | ppl | Δppl vs base | mean \\|ΔNLL\\| | flip | tok/s |\n|---|---|---|---|---|---|\n");
    let base_ppl = accs
        .first()
        .filter(|a| a.k > 0)
        .map(|a| (a.sum_nll / a.k as f64).exp());
    for (arm, a) in arms.iter().zip(accs) {
        if a.k == 0 {
            out.push_str(&format!("| {} | (pending) | — | — | — | — |\n", arm.name));
            continue;
        }
        let ppl = (a.sum_nll / a.k as f64).exp();
        let dppl = match base_ppl {
            Some(bp) => format!("{:+.3}%", 100.0 * (ppl / bp - 1.0)),
            None => "—".into(),
        };
        let (mean_abs, flip) = match arm.kind {
            ArmKind::Base => ("—".to_string(), "—".to_string()),
            _ => (
                format!("{:.5}", a.sum_abs / a.k as f64),
                format!("{:.2}%", 100.0 * a.flips as f64 / a.k as f64),
            ),
        };
        out.push_str(&format!(
            "| {} | {ppl:.4} | {dppl} | {mean_abs} | {flip} | {:.1} |\n",
            arm.name,
            a.fwd as f64 / a.secs.max(1e-9)
        ));
    }

    // Arm lookup helpers.
    let plain_of = |b: u8| {
        arms.iter()
            .position(|a| matches!(a.kind, ArmKind::Plain(x) if x == b))
    };
    let mr_of = |b: u8, dial: bool| {
        arms.iter().position(
            |a| matches!(a.kind, ArmKind::MeanRemoved { bits: x, dial: d } if x == b && d == dial),
        )
    };
    let bits_present: Vec<u8> = {
        let mut v: Vec<u8> = arms
            .iter()
            .filter_map(|a| match a.kind {
                ArmKind::Plain(b) => Some(b),
                _ => None,
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    // ── Gate 1 ────────────────────────────────────────────────────────
    out.push_str("\n## Gate 1 — prediction vs measurement (mr/plain MSE ratio vs 1 − ρ_l(V))\n\n");
    out.push_str("| layer | 1−ρ (pred) | ratio b2 | ratio b3 | ratio b4 |\n|---|---|---|---|---|\n");
    let mut per_bits: Vec<(u8, Vec<Option<f64>>)> = Vec::new();
    for &b in &bits_present {
        let (Some(pi), Some(mi)) = (plain_of(b), mr_of(b, false)) else {
            continue;
        };
        let (tp, tm) = (&accs[pi].tel, &accs[mi].tel);
        let mut ratios = Vec::with_capacity(tp.len());
        for l in 0..tp.len() {
            if tp[l].rows == 0 || tm[l].rows == 0 || tp[l].mse(ctx.kvd) <= 0.0 {
                ratios.push(None);
            } else {
                ratios.push(Some(tm[l].mse(ctx.kvd) / tp[l].mse(ctx.kvd)));
            }
        }
        per_bits.push((b, ratios));
    }
    for l in 0..ctx.r2.len() {
        let pred = 1.0 - ctx.r2[l] as f64;
        let cell = |b: u8| {
            per_bits
                .iter()
                .find(|(bb, _)| *bb == b)
                .and_then(|(_, r)| r[l])
                .map_or("—".into(), |x| format!("{x:.3}"))
        };
        out.push_str(&format!(
            "| {l} | {pred:.3} | {} | {} | {} |\n",
            cell(2),
            cell(3),
            cell(4)
        ));
    }
    for (b, ratios) in &per_bits {
        let pairs: Vec<(f64, f64)> = ratios
            .iter()
            .zip(ctx.r2.iter())
            .filter_map(|(r, &rho)| r.map(|x| (x, 1.0 - rho as f64)))
            .collect();
        if pairs.is_empty() {
            out.push_str(&format!("\n- b{b}: (pending)\n"));
            continue;
        }
        let mut deltas: Vec<f64> = pairs.iter().map(|(r, p)| (r - p).abs()).collect();
        deltas.sort_by(|a, b2| a.partial_cmp(b2).unwrap());
        let med = deltas[deltas.len() / 2];
        let n_le = deltas.iter().filter(|&&d| d <= 0.20).count();
        let mean_ratio = pairs.iter().map(|(r, _)| r).sum::<f64>() / pairs.len() as f64;
        let mean_pred = pairs.iter().map(|(_, p)| p).sum::<f64>() / pairs.len() as f64;
        let mean_gap = (mean_ratio - mean_pred).abs();
        let verdict = if *b == 2 {
            let below = pairs.iter().filter(|(r, _)| *r < 1.0).count();
            format!(
                "direction-only (absmax caveat is the arbiter): ratio < 1 on {below}/{} layers",
                pairs.len()
            )
        } else {
            let ok = n_le >= 20 && mean_gap <= 0.10;
            format!(
                "{} — within ±0.20 on {n_le}/{} layers (median |Δ| {med:.3}), |mean ratio − mean pred| = {mean_gap:.3}",
                if ok { "PASS" } else { "FAIL" },
                pairs.len()
            )
        };
        out.push_str(&format!("\n- b{b}: {verdict}\n"));
        // The absmax caveat arbiter: did the encode-side range grow?
        if let Some(mi) = mr_of(*b, false) {
            let tm = &accs[mi].tel;
            let grew = tm.iter().filter(|t| t.max_abs_enc > t.max_abs).count();
            let max_enc = tm.iter().map(|t| t.max_abs_enc).fold(0.0f32, f32::max);
            let max_abs = tm.iter().map(|t| t.max_abs).fold(0.0f32, f32::max);
            out.push_str(&format!(
                "- b{b} caveat arbiter: encode range (max |V−E|) > raw range (max |V|) on {grew}/{} layers; global max enc {max_enc:.3} vs raw {max_abs:.3}\n",
                tm.len()
            ));
        }
    }

    // ── Gate 2 ────────────────────────────────────────────────────────
    out.push_str("\n## Gate 2 — PPL at matched bits (mr vs plain)\n\n| bits | Δppl(mr−plain) | mean paired ΔNLL | chunk win share | verdict |\n|---|---|---|---|---|\n");
    for &b in &bits_present {
        let (Some(pi), Some(mi)) = (plain_of(b), mr_of(b, false)) else {
            continue;
        };
        let (ap, am) = (&accs[pi], &accs[mi]);
        if ap.k == 0 || am.k == 0 {
            out.push_str(&format!("| {b} | (pending) | — | — | — |\n"));
            continue;
        }
        let ppl_p = (ap.sum_nll / ap.k as f64).exp();
        let ppl_m = (am.sum_nll / am.k as f64).exp();
        let dnll = (am.sum_nll - ap.sum_nll) / am.k as f64;
        let wins = am
            .chunk_d
            .iter()
            .zip(ap.chunk_d.iter())
            .filter(|(m, p)| m < p)
            .count();
        let n_pairs = am.chunk_d.len().min(ap.chunk_d.len());
        let share = if n_pairs > 0 {
            format!("{wins}/{n_pairs} ({:.0}%)", 100.0 * wins as f64 / n_pairs as f64)
        } else {
            "—".into()
        };
        let ok = dnll < 0.0;
        out.push_str(&format!(
            "| {b} | {:+.3}% | {dnll:+.5} | {share} | {} |\n",
            100.0 * (ppl_m / ppl_p - 1.0),
            if ok { "PASS" } else { "FAIL" }
        ));
    }
    // The coverage dial arm (informational vs its full-table twin).
    let last_b = bits_present[bits_present.len() - 1];
    if let (Some(mi), Some(di)) = (mr_of(last_b, false), mr_of(last_b, true)) {
        let (am, ad) = (&accs[mi], &accs[di]);
        if am.k > 0 && ad.k > 0 {
            let d = (ad.sum_nll - am.sum_nll) / ad.k as f64;
            out.push_str(&format!(
                "\n- coverage dial (mr-b{last_b}-k{} vs k{}): ΔNLL {d:+.5} — the cost of dropping table rows {}→{} (P(K) {:.0}→{:.0} MiB)\n",
                ctx.dial_k,
                ctx.top_k,
                ctx.top_k,
                ctx.dial_k,
                table_mib(ctx.top_k),
                table_mib(ctx.dial_k)
            ));
        }
    }

    // ── Gate 3 ────────────────────────────────────────────────────────
    out.push_str("\n## Gate 3 — per-band conditional retention (target-token frequency bands)\n\n");
    const BAND_NAMES: [&str; 4] = ["top-64", "64–1024", "1024–top_k", "tail/untracked"];
    for &b in &bits_present {
        let (Some(pi), Some(mi)) = (plain_of(b), mr_of(b, false)) else {
            continue;
        };
        out.push_str(&format!(
            "### b{b}\n\n| band | n | flip p | flip mr | mean\\|Δ\\| p | mean\\|Δ\\| mr | mean Δ (mr−plain) |\n|---|---|---|---|---|---|---|\n"
        ));
        for (band, name) in BAND_NAMES.iter().enumerate() {
            let n = ctx.band_n[band];
            if n <= 0.0 {
                out.push_str(&format!("| {name} | 0 | — | — | — | — | — |\n"));
                continue;
            }
            let (bp, bm) = (&accs[pi].band[band], &accs[mi].band[band]);
            out.push_str(&format!(
                "| {name} | {:.0} | {:.2}% | {:.2}% | {:.5} | {:.5} | {:+.5} |\n",
                n,
                100.0 * bp[0] / n,
                100.0 * bm[0] / n,
                bp[1] / n,
                bm[1] / n,
                (bm[2] - bp[2]) / n
            ));
        }
        out.push('\n');
    }
    out.push_str(
        "The Bench-895 caveat predicts the mr damage concentrates in OFF-MEAN occurrences of \
         FREQUENT tokens at 2 bits — visible here as a positive top-band mean Δ at b2 even \
         when the aggregate Gate 2 read is negative.\n",
    );

    // ── Gate 4 ────────────────────────────────────────────────────────
    out.push_str("\n## Gate 4 — sinks\n\n| bits | sink-bin flip p | sink-bin flip mr | body flip mr | verdict |\n|---|---|---|---|---|\n");
    for &b in &bits_present {
        let (Some(pi), Some(mi)) = (plain_of(b), mr_of(b, false)) else {
            continue;
        };
        let (ap, am) = (&accs[pi], &accs[mi]);
        if ap.k == 0 || am.k == 0 {
            out.push_str(&format!("| {b} | (pending) | — | — | — |\n"));
            continue;
        }
        let n_sink = ctx.posbin_n[0];
        let flip_p = 100.0 * ap.posbin[0][0] / n_sink.max(1.0);
        let flip_m = 100.0 * am.posbin[0][0] / n_sink.max(1.0);
        let body_m = 100.0 * am.posbin[1][0] / ctx.posbin_n[1].max(1.0);
        let ok = flip_m <= flip_p + 1e-9;
        out.push_str(&format!(
            "| {b} | {flip_p:.2}% | {flip_m:.2}% | {body_m:.2}% | {} |\n",
            if ok { "PASS (mr ≤ plain)" } else { "FAIL" }
        ));
    }
    out.push_str(ctx.sink_block);
    out.push('\n');

    // ── Coverage + storage ────────────────────────────────────────
    out.push_str("\n## Coverage + storage\n\n");
    out.push_str(&format!(
        "- table bytes (P(K) = b_w·L·K·d_v, b_w=4): top_k {} = {:.0} MiB, dial k{} = {:.0} MiB\n",
        ctx.top_k,
        table_mib(ctx.top_k),
        ctx.dial_k,
        table_mib(ctx.dial_k)
    ));
    for (arm, a) in arms.iter().zip(accs) {
        if let ArmKind::MeanRemoved { bits: _, dial } = arm.kind {
            let rows: u64 = a.tel.iter().map(|t| t.rows).sum();
            let tracked: u64 = a.tel.iter().map(|t| t.rows_tracked).sum();
            if rows > 0 {
                let tag = if dial { ", dial table" } else { "" };
                out.push_str(&format!(
                    "- {}{}: realized table coverage on the eval slice = {:.1}% ({tracked}/{rows} flushed rows tracked)\n",
                    arm.name,
                    tag,
                    100.0 * tracked as f64 / rows as f64
                ));
            }
        }
    }
    for &b in &bits_present {
        let v_bytes = (ctx.kvd * b as usize).div_ceil(8) + 8 + 4 + 4 * ctx.kvd / TILE;
        out.push_str(&format!(
            "- V bytes/token at b{b}: {v_bytes} (packed {} + rtn scale/zp 8 + var-norm {} amortized) vs f32 {} ({:.3}×)\n",
            (ctx.kvd * b as usize).div_ceil(8),
            4 + 4 * ctx.kvd / TILE,
            ctx.kvd * 4,
            v_bytes as f64 / (ctx.kvd * 4) as f64
        ));
    }
    out.push_str(
        "- (b2 runs KVarN's skip-varn + grouped-4 RTN posture; the var-norm term above is the stored-meta figure, not per-bits work.)\n",
    );

    // ── Caveats ───────────────────────────────────────────────────────
    out.push_str("\n## Fixture caveats\n\n\
- 288-tensor conversion, NO q/k-norm (the Bench-004 fixture) — trap-1 re-arms on a standard conversion.\n\
- Simulation law: lossy rows are rewritten into the plain cache at TILE close (KVarN's own streaming visibility — within-open-tile reads raw, closed-tile reads dequantized). Storage is virtual; the quality surface is exact.\n\
- KVarN instantiation: tile 128, hadamard OFF, var-norm ON at b > 2, skip-varn + grouped-4 RTN at b2.\n\
- Tap point: pre-RoPE K / post-W_V V; RoVE OFF (the default decode path). Freeze λ_js = 0; untracked tokens take the plain path (never an error, never a zero-row guess).\n\
- Calibration and eval are DISJOINT token slices of the same corpus; realized coverage is reported above.\n\
- MEASUREMENT-ONLY — promotion is katgpt-rs-side and waits on these gates.\n");
    out
}
