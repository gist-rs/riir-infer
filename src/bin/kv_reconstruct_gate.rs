//! kv_reconstruct_gate — riir-infer Issue 013 T3: the P3 V-cache
//! reconstruction gate (katgpt-rs Issue 883 P3, the model-bound lane) on the
//! REAL gemma-2-2b decode path.
//!
//! One process, five phases:
//!
//! - **P0 load** — the model, the corpus, and T2's table artifact
//!   (`--table`, BLAKE3-verified; T3 reuses T2's 2 h calibration).
//! - **P1 G3** — the read-path seam probe: plain `forward_gemma2_f16` vs
//!   `forward_gemma2_f16_hk(NoVQuant)` logits `to_bits`-identical across a
//!   65-position decode (the seam's default IS the cache slice — this pins
//!   the wiring post-edit), plus the recon-vs-store logit delta at λ=0
//!   (the rotation-rounding class, recorded).
//! - **P2 G1** — paired store-vs-reconstruct arms at λ ∈ {0, 0.5, 1} over
//!   the first eval chunks: per-position ΔNLL(recon − store), PPL per arm,
//!   and the retention walk (ΔNLL/flips by target-token frequency band ×
//!   tracked/miss). The tolerance gate is the rotation-rounding class.
//! - **P3 G2** — tg128 paired interleave (the katgpt-rs `ab_timing`
//!   protocol): full-cache control vs reconstruct at λ=0 and λ=1, 128-token
//!   prefill + 64 timed decode steps, R interleaved ABC triples, median of
//!   per-pair ratios. Box state echoed from `--box-note`.
//! - **P4 record** — the KV bytes/token arithmetic (the 50% law: gemma-2-2b
//!   is 8q:4kv at hd 256, `n_v/(n_kv+n_v) = 1/2`; sliding-window layers
//!   scale with the window and keep the fraction).
//! - **P5 report** — pre-registered gates echoed + verdicts; rewritten
//!   after every phase (a mid-run death keeps completed phases).
//!
//! MEASUREMENT-ONLY: T3's claim is that the reconstruction serves T2's
//! quality at half the KV bytes (G1 within tolerance) and what the naive
//! read path costs (G2, recorded — the kernel levers are T4's lane).
//! Promotion is katgpt-rs-side and waits on the gates.
//!
//! Usage:
//! ```text
//! kv_reconstruct_gate <gguf> <corpus> [--table PATH] [--eval-tokens N]
//!                     [--seq-len N] [--lambdas 0,0.5,1] [--tg-pairs R]
//!                     [--tg-prefill N] [--tg-decode N] [--report PATH]
//!                     [--box-note S] [--smoke]
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use katgpt_core::fitted_value_table::FittedTokenTable;
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gemma_layer::GemmaTransformerWeightsF16;
use riir_infer_core::gguf_loader::{config_from_gguf_metadata, GgufFile};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::gemma2_calibration::load_gemma2_f16_direct;
use riir_infer_core::transformer::gemma2_ktov::KToVState;
use riir_infer_core::transformer::gemma2_vrecon::VReconState;
use riir_infer_core::transformer::vk_harness::{argmax, load_fitted_table, nll, TableMeta};
use riir_infer_core::transformer::{forward_gemma2_f16, forward_gemma2_f16_hk, ForwardContext, NoVQuant};
use riir_infer_core::types::{kv_dim, Config};

/// The G1 tolerance (pre-registered): the reconstruct arm must sit at the
/// rotation-rounding class, not model level. Mean paired |ΔNLL| ≤ 2e-3 AND
/// max |ΔNLL| ≤ 5e-2. Wiring bugs (wrong convention, wrong token map) blow
/// past 0.05 by orders of magnitude; rotation rounding sits at ~1e-5.
const G1_MEAN_ABS_TOL: f64 = 2e-3;
const G1_MAX_ABS_TOL: f64 = 5e-2;

/// The G3 recon-vs-store logit delta bound (recorded, wiring-class).
const G3_LOGIT_DELTA_TOL: f32 = 5e-2;

// ── The arm hook — the ONE forward-dispatch seam all passes share ────────

enum ArmHook<'a> {
    /// The f16 base arm — the plain forward, no hook.
    Plain,
    /// The full-cache no-op hook (the G3/G2 control; bitwise today's path).
    None,
    /// A K=V+ store arm (the T2 reference; the G1 pairing base).
    K(&'a mut KToVState),
    /// A P3 reconstruct arm.
    R(&'a mut VReconState),
}

impl ArmHook<'_> {
    fn reset(&mut self) {
        match self {
            ArmHook::K(st) => st.reset(),
            ArmHook::R(st) => st.reset(),
            _ => {}
        }
    }

    #[inline]
    fn set_token(&mut self, pos: usize, token: u32) {
        match self {
            ArmHook::K(st) => st.set_token(pos, token),
            ArmHook::R(st) => st.set_token(pos, token),
            _ => {}
        }
    }

    #[inline]
    fn step<'c>(
        &mut self,
        ctx: &'c mut ForwardContext,
        weights: &GemmaTransformerWeightsF16,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'c mut [f32] {
        match self {
            ArmHook::Plain => forward_gemma2_f16(ctx, weights, cache, token, pos, config),
            ArmHook::None => {
                forward_gemma2_f16_hk(ctx, weights, cache, &mut NoVQuant, token, pos, config)
            }
            ArmHook::K(st) => {
                forward_gemma2_f16_hk(ctx, weights, cache, &mut **st, token, pos, config)
            }
            ArmHook::R(st) => {
                forward_gemma2_f16_hk(ctx, weights, cache, &mut **st, token, pos, config)
            }
        }
    }
}

// ── The eval pass (the T2 shape: chunked teacher-forced NLL) ─────────────

#[derive(Default)]
struct PassOut {
    /// Per scored position NLL (the pairing record; arm-major).
    nll: Vec<f64>,
    /// Per scored position argmax (the flip record).
    top: Vec<usize>,
    /// Per scored position TARGET token id (the walk's banding key).
    target: Vec<u32>,
    fwd: usize,
    secs: f64,
}

/// `n_chunks` chunks of `[BOS] + (seq_len−1)` corpus tokens over `tokens`,
/// teacher-forced, per-position NLL + argmax + target id.
#[allow(clippy::too_many_arguments)]
fn run_pass(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    hook: &mut ArmHook<'_>,
    tokens: &[usize],
    corpus_per_chunk: usize,
    n_chunks: usize,
    bos: usize,
    config: &Config,
) -> PassOut {
    let mut out = PassOut::default();
    let t0 = Instant::now();
    for chunk in tokens.chunks(corpus_per_chunk).take(n_chunks) {
        cache.reset();
        hook.reset();
        let mut seq = Vec::with_capacity(corpus_per_chunk + 1);
        seq.push(bos);
        seq.extend_from_slice(chunk);
        for (pos, &token) in seq.iter().enumerate() {
            hook.set_token(pos, token as u32);
            let logits = hook.step(ctx, weights, cache, token, pos, config);
            if pos + 1 >= seq.len() {
                continue;
            }
            out.nll.push(nll(logits, seq[pos + 1]));
            out.top.push(argmax(logits));
            out.target.push(seq[pos + 1] as u32);
            out.fwd += 1;
        }
    }
    out.secs = t0.elapsed().as_secs_f64();
    out
}

fn ppl_of(nlls: &[f64]) -> f64 {
    (nlls.iter().sum::<f64>() / nlls.len().max(1) as f64).exp()
}

/// Provenance placeholder for the smoke's synthetic zero table.
fn meta_placeholder() -> TableMeta {
    TableMeta {
        n_layer: 0,
        width: 0,
        rows: 1,
        vocab: 0,
        cal_tokens: 0,
    }
}

/// Paired delta stats: (mean Δ, mean |Δ|, max |Δ|, flips) of `a − b`.
fn paired(a: &PassOut, b: &PassOut) -> (f64, f64, f64, usize) {
    let n = a.nll.len().min(b.nll.len());
    let (mut sum, mut abs, mut max, mut flips) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for i in 0..n {
        let d = a.nll[i] - b.nll[i];
        sum += d;
        abs += d.abs();
        max = max.max(d.abs());
        if a.top[i] != b.top[i] {
            flips += 1;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0.0, 0);
    }
    (sum / n as f64, abs / n as f64, max, flips)
}

// ── The retention walk ───────────────────────────────────────────────────

/// ΔNLL/flips by target-token frequency band × table-tracked. Bands are
/// count tertiles over the PASS's own target tokens (deterministic,
/// self-contained — the eval corpus is the population).
struct Walk {
    /// (band, tracked) → [count, ΣΔ, Σ|Δ|, flips]
    cell: [[[f64; 4]; 2]; 3],
}

impl Walk {
    fn new() -> Self {
        Self {
            cell: [[[0.0; 4]; 2]; 3],
        }
    }

    /// `tracked(tok)` = the table has a row for `tok` on layer 0 (coverage
    /// is layer-uniform by construction — one top_k set).
    fn accumulate(&mut self, recon: &PassOut, store: &PassOut, tracked: &dyn Fn(u32) -> bool) {
        let n = recon.nll.len().min(store.nll.len());
        // Count tertiles over the pass's targets.
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for &t in &recon.target {
            *counts.entry(t).or_insert(0) += 1;
        }
        let mut freqs: Vec<usize> = counts.values().copied().collect();
        freqs.sort_unstable();
        let (q1, q2) = (
            freqs[freqs.len() / 3],
            freqs[2 * freqs.len() / 3],
        );
        for i in 0..n {
            let t = recon.target[i];
            let c = counts[&t];
            let band = if c <= q1 {
                0
            } else if c <= q2 {
                1
            } else {
                2
            };
            let tr = usize::from(tracked(t));
            let d = recon.nll[i] - store.nll[i];
            let cell = &mut self.cell[band][tr];
            cell[0] += 1.0;
            cell[1] += d;
            cell[2] += d.abs();
            if recon.top[i] != store.top[i] {
                cell[3] += 1.0;
            }
        }
    }

    fn render(&self, title: &str) -> String {
        let mut s = format!("### {title}\n\n");
        s.push_str("| band | tracked | n | mean ΔNLL | mean \\|Δ\\| | flips |\n|---|---|---|---|---|---|\n");
        for (band, name) in [(0, "freq≤q1"), (1, "q1<freq≤q2"), (2, "freq>q2")] {
            for (tr, trn) in [(0, "miss"), (1, "tracked")] {
                let c = &self.cell[band][tr];
                if c[0] == 0.0 {
                    continue;
                }
                s.push_str(&format!(
                    "| {name} | {trn} | {} | {:+.2e} | {:.2e} | {} |\n",
                    c[0] as usize,
                    c[1] / c[0],
                    c[2] / c[0],
                    c[3] as usize
                ));
            }
        }
        s
    }
}

// ── G2: the tg128 paired interleave ──────────────────────────────────────

struct TimingArm {
    name: &'static str,
    /// Median per-step µs across pairs (the primary figure).
    med_us: f64,
    /// Min per-pair median µs (the box-quiet bound, recorded).
    min_us: f64,
    /// Median per-pair ratio vs the control (None for the control).
    ratio: Option<f64>,
    prefill_ms: f64,
}

/// One interleaved (control, recon-λ0, recon-λ1) triple: fresh cache +
/// prefill + `n_decode` timed decode steps, per arm; returns per-arm
/// (median step µs, prefill ms).
#[allow(clippy::too_many_arguments)]
fn timing_triple(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    hooks: &mut [ArmHook<'_>],
    prefill: &[usize],
    decode: &[usize],
    bos: usize,
    config: &Config,
) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(hooks.len());
    for hook in hooks.iter_mut() {
        cache.reset();
        hook.reset();
        let t_pf = Instant::now();
        let mut seq: Vec<usize> = Vec::with_capacity(prefill.len() + 1);
        seq.push(bos);
        seq.extend_from_slice(prefill);
        for (pos, &t) in seq.iter().enumerate() {
            hook.set_token(pos, t as u32);
            let _ = hook.step(ctx, weights, cache, t, pos, config);
        }
        let prefill_ms = t_pf.elapsed().as_secs_f64() * 1e3;
        let mut steps: Vec<f64> = Vec::with_capacity(decode.len());
        // Continue the SAME sequence (context keeps growing — the decode
        // regime), teacher-forcing the decode tokens.
        let base = seq.len();
        for (i, &t) in decode.iter().enumerate() {
            let pos = base + i;
            hook.set_token(pos, t as u32);
            let t0 = Instant::now();
            let _ = hook.step(ctx, weights, cache, t, pos, config);
            steps.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = steps[steps.len() / 2];
        out.push((med, prefill_ms));
    }
    out
}

// ── Report ───────────────────────────────────────────────────────────────

struct Report {
    path: PathBuf,
    buf: String,
}

impl Report {
    fn new(path: PathBuf, header: &str) -> Self {
        Self {
            path,
            buf: String::from(header),
        }
    }
    fn push(&mut self, s: &str) {
        self.buf.push_str(s);
        let _ = std::fs::write(&self.path, &self.buf);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: kv_reconstruct_gate <gguf> <corpus> [--table PATH] [--eval-tokens N] \
             [--seq-len N] [--lambdas 0,0.5,1] [--tg-pairs R] [--tg-prefill N] \
             [--tg-decode N] [--report PATH] [--box-note S] [--skip-g1] [--smoke] \
             [--recon eager|deferred]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let smoke = args.iter().any(|a| a == "--smoke");
    let mut eval_n: usize = if smoke { 512 } else { 2048 };
    let mut seq_len: usize = if smoke { 256 } else { 1024 };
    let mut table_path: Option<PathBuf> = None;
    let mut lambdas: Vec<f32> = vec![0.0, 0.5, 1.0];
    let (mut tg_pairs, mut tg_prefill, mut tg_decode) = if smoke { (2, 64, 16) } else { (12, 128, 64) };
    let mut report_path: Option<PathBuf> = None;
    let mut box_note = String::from("4090 workstation i7-13700K, CPU lane, AC");
    let mut skip_g1 = false;
    // The read posture (Issue 013 T4): `eager` = the T3 scratch (Bench 013/016
    // continuity, the default); `deferred` = the fused kernel lane — and it
    // carries its own in-run probe (the bitwise λ0 law + the regrouping
    // bound), so no separate flag.
    let mut recon_deferred = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--eval-tokens" => {
                eval_n = args[i + 1].parse().context("--eval-tokens N")?;
                i += 2;
            }
            "--seq-len" => {
                seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--table" => {
                table_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--lambdas" => {
                lambdas = args[i + 1]
                    .split(',')
                    .map(|s| s.parse().context("--lambdas 0,0.5,1"))
                    .collect::<Result<_>>()?;
                if !lambdas.contains(&0.0) {
                    bail!("--lambdas must include 0 (the V:=K control)");
                }
                i += 2;
            }
            "--tg-pairs" => {
                tg_pairs = args[i + 1].parse().context("--tg-pairs R")?;
                i += 2;
            }
            "--tg-prefill" => {
                tg_prefill = args[i + 1].parse().context("--tg-prefill N")?;
                i += 2;
            }
            "--tg-decode" => {
                tg_decode = args[i + 1].parse().context("--tg-decode N")?;
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--box-note" => {
                box_note = args[i + 1].clone();
                i += 2;
            }
            "--skip-g1" => {
                skip_g1 = true;
                i += 1;
            }
            "--recon" => {
                recon_deferred = match args[i + 1].as_str() {
                    "eager" => false,
                    "deferred" => true,
                    other => bail!("--recon expects eager|deferred, got {other}"),
                };
                i += 2;
            }
            "--smoke" => i += 1,
            other => bail!("unknown arg {other}"),
        }
    }
    if seq_len > 4096 {
        bail!("--seq-len must stay <= 4096 (gemma-2 sliding window)");
    }
    let Some(table_path) = table_path else {
        bail!("--table is required (T2's 012_kv_table_residual.bin artifact)");
    };

    // ── P0: load model + tokenizer + table ────────────────────────────
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
        "# kv_reconstruct_gate: gemma-2 f16 | layers={} n_embd={} kv_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        kv_dim(&config),
        config.vocab_size,
        t_start.elapsed().as_secs_f32()
    );
    drop(gguf);

    let text = load_corpus_text(&corpus_path)?;
    let all_tokens = tok.encode(&text);
    // Eval slice sits at T2's eval start (tokens [61440..)); G1 uses its
    // first chunks, G2's prefill/decode ride the same region.
    let eval_start = 61_440usize;
    let need = eval_start + eval_n + tg_prefill + tg_decode + 64;
    if all_tokens.len() < need {
        bail!(
            "corpus too short: {} tokens < eval_start {eval_start} + eval {eval_n} + tg {}",
            all_tokens.len(),
            tg_prefill + tg_decode
        );
    }
    println!(
        "# corpus: {} chars → {} tokens | eval [{eval_start}..{}) | seq {seq_len}",
        text.len(),
        all_tokens.len(),
        eval_start + eval_n
    );

    // The G2 prefill rides the same cache — the block must hold
    // (tg_prefill + BOS + tg_decode) positions, else the timing loop writes
    // past the allocation (the 0xC0000005 class: the Bench-016 cells died at
    // exactly one position over when block_size followed seq_len alone).
    config.block_size = seq_len.max(tg_prefill + tg_decode + 8) + 64;
    let kvd = kv_dim(&config);
    let n_layers = config.n_layer;
    let corpus_per_chunk = seq_len - 1;
    let n_chunks = eval_n / corpus_per_chunk;
    if n_chunks == 0 {
        bail!("eval slice smaller than one chunk");
    }

    let (table, meta, synthetic) = if table_path.exists() {
        let (t, m) = load_fitted_table(&table_path, Some((n_layers, kvd)))?;
        (t, m, false)
    } else if smoke {
        // Smoke-only wiring fixture: token 0 → a ZERO row, every other
        // token a miss. λ arms then serve exact G(−θp)K̂ (+ 0) — the full
        // read path (rotation, row lookup, add) with no calibration. NEVER
        // a measurement posture; the report says so.
        println!("# SYNTHETIC ZERO TABLE (smoke wiring fixture — {table_path:?} not built yet)");
        (
            FittedTokenTable::from_rows(
                n_layers,
                kvd,
                vec![0],
                vec![0.0; n_layers * kvd],
            ),
            meta_placeholder(),
            true,
        )
    } else {
        bail!(
            "--table {} not found (T2's calibration artifact; a full run REQUIRES it)",
            table_path.display()
        );
    };
    let table: &'static FittedTokenTable = Box::leak(Box::new(table));
    println!(
        "# table loaded: {} ({} rows, cal_tokens {}) — {}",
        table_path.display(),
        meta.rows,
        meta.cal_tokens,
        if synthetic {
            "SYNTHETIC ZERO (smoke)"
        } else {
            "BLAKE3 verified"
        }
    );

    // The forward's own rope table — the action's angles are THESE values.
    let mut ctx = ForwardContext::new(&config);
    let freq: Vec<f32> = ctx.rope_freq_table.as_slice().to_vec();
    let mut cache = MultiLayerKVCache::new(&config);

    let mut report = Report::new(
        report_path.unwrap_or_else(|| PathBuf::from(".benchmarks/013_t3_reconstruct_report.md")),
        &format!(
            "# Bench 013 — T3 P3 V-cache reconstruction gate (kv_reconstruct_gate)\n\n\
             **Status:** RUNNING (rewritten after every phase)\n\n\
             Box: {box_note}  \n\
             Fixture: gemma-2-2b-it-f16, chat_probe, eval tokens [{eval_start}..{}), \
             seq {seq_len}, {n_chunks} chunks, teacher-forced NLL.  \n\
             Table: {} ({} rows, cal {}) — {}.  \n\n",
            eval_start + eval_n,
            table_path.display(),
            meta.rows,
            meta.cal_tokens,
            if synthetic {
                "SYNTHETIC ZERO (smoke wiring fixture — NOT a measurement posture)"
            } else {
                "T2's artifact, BLAKE3-verified"
            }
        ),
    );

    // ── P1: the G3 seam probe ─────────────────────────────────────────
    // (a) plain vs NoVQuant: the read-path selector's default IS the cache
    // slice — logits must be to_bits-identical across a 65-position decode.
    // (b) recon-λ0 vs store-λ0: the rotation-rounding class, recorded.
    let g3 = {
        let probe_tokens: Vec<usize> = all_tokens[eval_start..eval_start + 64].to_vec();
        let mut ok;
        let mut recon_vs_store_max: f32 = 0.0;
        {
            let mut seq = vec![bos];
            seq.extend_from_slice(&probe_tokens);
            let mut plain: Vec<Vec<u32>> = Vec::new();
            let mut noop: Vec<Vec<u32>> = Vec::new();
            cache.reset();
            for (pos, &t) in seq.iter().enumerate() {
                let lg = forward_gemma2_f16(&mut ctx, &weights, &mut cache, t, pos, &config);
                plain.push(lg.iter().map(|x| x.to_bits()).collect());
            }
            cache.reset();
            for (pos, &t) in seq.iter().enumerate() {
                let lg = forward_gemma2_f16_hk(
                    &mut ctx, &weights, &mut cache, &mut NoVQuant, t, pos, &config,
                );
                noop.push(lg.iter().map(|x| x.to_bits()).collect());
            }
            ok = plain == noop;
            // (b) recon λ=0 vs store λ=0 over the same positions.
            let mut st_store = KToVState::new_uniform(table, 0.0, n_layers, kvd, seq.len() + 32);
            let mut st_recon = VReconState::new_uniform(
                table,
                0.0,
                n_layers,
                kvd,
                &freq,
                seq.len() + 32,
                config.n_head,
                false,
            );
            let mut store_lg: Vec<Vec<f32>> = Vec::new();
            let mut recon_lg: Vec<Vec<f32>> = Vec::new();
            cache.reset();
            for (pos, &t) in seq.iter().enumerate() {
                st_store.set_token(pos, t as u32);
                let lg = forward_gemma2_f16_hk(
                    &mut ctx, &weights, &mut cache, &mut st_store, t, pos, &config,
                );
                store_lg.push(lg.to_vec());
            }
            cache.reset();
            for (pos, &t) in seq.iter().enumerate() {
                st_recon.set_token(pos, t as u32);
                let lg = forward_gemma2_f16_hk(
                    &mut ctx, &weights, &mut cache, &mut st_recon, t, pos, &config,
                );
                recon_lg.push(lg.to_vec());
            }
            for (a, b) in store_lg.iter().zip(&recon_lg) {
                for (x, y) in a.iter().zip(b) {
                    recon_vs_store_max = recon_vs_store_max.max((x - y).abs());
                }
            }
            let class_ok = recon_vs_store_max <= G3_LOGIT_DELTA_TOL;
            println!(
                "# G3: seam-identity {} | recon-vs-store max |Δlogit| {recon_vs_store_max:.3e} ({})",
                if ok { "PASS" } else { "FAIL" },
                if class_ok {
                    "rotation-rounding class"
                } else {
                    "WIRING-CLASS — investigate"
                }
            );
            ok &= class_ok;
        }
        report.push(&format!(
            "## P1 — G3 seam probe\n\n\
             - plain vs `NoVQuant` (the selector's default = the cache slice): **{}** \
             (65-position decode, logits `to_bits`).  \n\
             - recon-λ0 vs store-λ0 max \\|Δlogit\\|: **{recon_vs_store_max:.3e}** \
             (bound {G3_LOGIT_DELTA_TOL} — the rotation-rounding class).  \n\n",
            if ok { "PASS" } else { "FAIL" }
        ));
        ok
    };

    // ── P1b: the deferred-lane probe (Issue 013 T4, --recon deferred) ─
    // The fused kernel's two laws, on the REAL model + table over a
    // 65-position decode:
    // (a) deferred-λ0 == eager-λ0 to_bits (the bitwise zero-λ law);
    // (b) all-miss deferred-λ1 == eager-λ1 to_bits (the miss law);
    // (c) tracked λ1: deferred vs eager max |Δlogit| (the regrouping
    //     class, recorded against the 5e-2 wiring bound) and vs store.
    let deferred_probe = if recon_deferred {
        let probe_tokens: Vec<usize> = all_tokens[eval_start..eval_start + 64].to_vec();
        let mut seq = vec![bos];
        seq.extend_from_slice(&probe_tokens);
        let max_delta = |a: &[Vec<f32>], b: &[Vec<f32>]| -> f32 {
            a.iter()
                .zip(b)
                .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
                .fold(0.0f32, f32::max)
        };
        let mut run_lg = |lam: f32, deferred: bool, miss: bool| -> Vec<Vec<f32>> {
            let mut st = VReconState::new_uniform(
                table,
                lam,
                n_layers,
                kvd,
                &freq,
                seq.len() + 32,
                config.n_head,
                deferred,
            );
            cache.reset();
            let mut lgs: Vec<Vec<f32>> = Vec::with_capacity(seq.len());
            for (pos, &t) in seq.iter().enumerate() {
                // u32::MAX is out of vocab — set_token resolves it to the
                // miss row (MAX), which IS the all-miss arm.
                st.set_token(pos, if miss { u32::MAX } else { t as u32 });
                let lg = forward_gemma2_f16_hk(
                    &mut ctx, &weights, &mut cache, &mut st, t, pos, &config,
                );
                lgs.push(lg.to_vec());
            }
            lgs
        };
        let eager0 = run_lg(0.0, false, false);
        let def0 = run_lg(0.0, true, false);
        let miss1_eager = run_lg(1.0, false, true);
        let miss1_def = run_lg(1.0, true, true);
        let eager1 = run_lg(1.0, false, false);
        let def1 = run_lg(1.0, true, false);
        let store1 = {
            let mut st = KToVState::new_uniform(table, 1.0, n_layers, kvd, seq.len() + 32);
            cache.reset();
            let mut lgs: Vec<Vec<f32>> = Vec::with_capacity(seq.len());
            for (pos, &t) in seq.iter().enumerate() {
                st.set_token(pos, t as u32);
                let lg = forward_gemma2_f16_hk(
                    &mut ctx, &weights, &mut cache, &mut st, t, pos, &config,
                );
                lgs.push(lg.to_vec());
            }
            lgs
        };
        let bits = |lgs: &[Vec<f32>]| {
            lgs.iter()
                .flat_map(|l| l.iter().map(|x| x.to_bits()))
                .collect::<Vec<_>>()
        };
        let law_a = bits(&eager0) == bits(&def0);
        let law_b = bits(&miss1_eager) == bits(&miss1_def);
        let regroup_max = max_delta(&eager1, &def1);
        let vs_store_max = max_delta(&def1, &store1);
        let ok = law_a && law_b && regroup_max <= G3_LOGIT_DELTA_TOL;
        println!(
            "# deferred probe: λ0 to_bits {} | miss-λ1 to_bits {} | \
             regroup max |Δlogit| {regroup_max:.3e} | def-λ1 vs store {vs_store_max:.3e} ({})",
            if law_a { "PASS" } else { "FAIL" },
            if law_b { "PASS" } else { "FAIL" },
            if ok { "PASS" } else { "FAIL" }
        );
        report.push(&format!(
            "## P1b — deferred-lane probe (--recon deferred)\n\n\
             65-position decode on the real model + table.  \n\n\
             - deferred-λ0 vs eager-λ0: **{}** (to_bits — the zero-λ law).  \n\
             - deferred-λ1 all-miss vs eager-λ1: **{}** (to_bits — the miss law).  \n\
             - tracked λ1, deferred vs eager max \\|Δlogit\\|: **{regroup_max:.3e}** \
             (the regrouped-association class; bound {G3_LOGIT_DELTA_TOL}).  \n\
             - tracked λ1, deferred vs STORE max \\|Δlogit\\|: **{vs_store_max:.3e}** \
             (the full P3 class: regrouping + rotation rounding).  \n\n",
            if law_a { "PASS" } else { "FAIL" },
            if law_b { "PASS" } else { "FAIL" }
        ));
        ok
    } else {
        true
    };

    // ── P2: the G1 paired store-vs-reconstruct arms ───────────────────
    let eval_tokens: Vec<usize> = all_tokens[eval_start..eval_start + eval_n].to_vec();
    let mut g1_rows = String::from(
        "## P2 — G1 paired store-vs-reconstruct\n\n\
         | λ | ppl store | ppl recon | mean ΔNLL | mean \\|Δ\\| | max \\|Δ\\| | flips | verdict |\n|---|---|---|---|---|---|---|---|\n",
    );
    let mut g1_pass = true;
    let mut walk = Walk::new();
    let mut walk_done = false;
    if skip_g1 {
        // The G2-scaling posture (--skip-g1): the G1 record lives in Bench
        // 013 (this repo) — this run measures the read-path scaling axis
        // only. The eval slice still allocates one chunk so the corpus
        // geometry (block_size, n_chunks) stays well-formed.
        g1_rows.clear();
        g1_rows.push_str(
            "## P2 — G1 paired store-vs-reconstruct\n\n\
             SKIPPED (--skip-g1): the G1 record is Bench 013's (T3, seq 1024, \
             0 flips at every λ); this run measures the G2 scaling axis only.  \n\n",
        );
    } else
    // The f16 base (context for the recorded tax; the pairing base is the
    // store arm at the same λ).
    {
        let mut hook = ArmHook::Plain;
        let po = run_pass(
            &mut ctx, &weights, &mut cache, &mut hook, &eval_tokens, corpus_per_chunk, n_chunks,
            bos, &config,
        );
        println!(
            "# arm f16: {} scored | ppl {:.4} | {:.0} fwd/s",
            po.nll.len(),
            ppl_of(&po.nll),
            po.fwd as f64 / po.secs.max(1e-9)
        );
        g1_rows.push_str(&format!(
            "| — | **{:.4}** | — | — | — | — | — | context (base) |\n",
            ppl_of(&po.nll)
        ));
    }
    if !skip_g1 {
    for &lam in lambdas.iter() {
        let po_store = {
            let mut st = KToVState::new_uniform(table, lam, n_layers, kvd, seq_len + 32);
            let mut hook = ArmHook::K(&mut st);
            run_pass(
                &mut ctx, &weights, &mut cache, &mut hook, &eval_tokens, corpus_per_chunk,
                n_chunks, bos, &config,
            )
        };
        let po_recon = {
            let mut st = VReconState::new_uniform(
                table,
                lam,
                n_layers,
                kvd,
                &freq,
                seq_len + 32,
                config.n_head,
                false,
            );
            let mut hook = ArmHook::R(&mut st);
            run_pass(
                &mut ctx, &weights, &mut cache, &mut hook, &eval_tokens, corpus_per_chunk,
                n_chunks, bos, &config,
            )
        };
        let (mean_d, mean_abs, max_abs, flips) = paired(&po_recon, &po_store);
        let row_pass = mean_abs <= G1_MEAN_ABS_TOL && max_abs <= G1_MAX_ABS_TOL;
        g1_pass &= row_pass;
        println!(
            "# arm λ={lam}: store ppl {:.4} recon ppl {:.4} | mean Δ {mean_d:+.2e} max |Δ| {max_abs:.2e} | {}",
            ppl_of(&po_store.nll),
            ppl_of(&po_recon.nll),
            if row_pass { "PASS" } else { "FAIL" }
        );
        g1_rows.push_str(&format!(
            "| k-{lam:.2} | {:.4} | {:.4} | {mean_d:+.2e} | {mean_abs:.2e} | {max_abs:.2e} | {flips}/{} | {} |\n",
            ppl_of(&po_store.nll),
            ppl_of(&po_recon.nll),
            po_recon.nll.len(),
            if row_pass { "PASS" } else { "**FAIL**" }
        ));
        // The walk rides the LARGEST λ (where the refund is biggest and any
        // band structure would show); tracked probe = layer-0 row presence.
        if lam == lambdas.iter().copied().fold(f32::NEG_INFINITY, f32::max) && !walk_done {
            walk.accumulate(&po_recon, &po_store, &|t| table.row(0, t).is_some());
            walk_done = true;
        }
        let _ = std::io::stdout().flush();
    }
    }
    g1_rows.push_str(&format!(
        "\nTolerances (pre-registered): mean \\|ΔNLL\\| ≤ {G1_MEAN_ABS_TOL}, max ≤ {G1_MAX_ABS_TOL} — the rotation-rounding class.  \n\n"
    ));
    if skip_g1 {
        g1_rows.clear();
        g1_rows.push_str(
            "## P2 — G1 paired store-vs-reconstruct\n\n\
             SKIPPED (--skip-g1): the G1 record is Bench 013's (T3, seq 1024, \
             0 flips at every λ); this run measures the G2 scaling axis only.  \n\n",
        );
    }
    g1_rows.push_str(&walk.render("Retention walk (recon − store, by target-token frequency band × tracked)"));
    g1_rows.push('\n');
    report.push(&g1_rows);

    // ── P3: the G2 tg128 paired interleave ────────────────────────────
    let tg = {
        let prefill_tokens: Vec<usize> =
            all_tokens[eval_start + eval_n..eval_start + eval_n + tg_prefill].to_vec();
        let decode_tokens: Vec<usize> = all_tokens
            [eval_start + eval_n + tg_prefill..eval_start + eval_n + tg_prefill + tg_decode]
            .to_vec();
        // Arm list: the full-cache control first (ratios read against it);
        // eager mode = the T3 triple (Bench 013/016 continuity); deferred
        // mode = the T4 four (the eager-λ1 anchor + the fused arms — the
        // decision figure).
        let mut st_l0 = VReconState::new_uniform(
            table,
            0.0,
            n_layers,
            kvd,
            &freq,
            tg_prefill + tg_decode + 8,
            config.n_head,
            false,
        );
        let mut st_l1 = VReconState::new_uniform(
            table,
            1.0,
            n_layers,
            kvd,
            &freq,
            tg_prefill + tg_decode + 8,
            config.n_head,
            false,
        );
        let mut st_d0 = VReconState::new_uniform(
            table,
            0.0,
            n_layers,
            kvd,
            &freq,
            tg_prefill + tg_decode + 8,
            config.n_head,
            true,
        );
        let mut st_d1 = VReconState::new_uniform(
            table,
            1.0,
            n_layers,
            kvd,
            &freq,
            tg_prefill + tg_decode + 8,
            config.n_head,
            true,
        );
        let (arm_names, n_arms) = if recon_deferred {
            (
                ["full-cache", "eager-λ1", "def-λ0", "def-λ1"],
                4usize,
            )
        } else {
            (["full-cache", "recon-λ0", "recon-λ1", ""], 3usize)
        };
        let mut hooks: Vec<ArmHook<'_>> = if recon_deferred {
            vec![
                ArmHook::None,
                ArmHook::R(&mut st_l1),
                ArmHook::R(&mut st_d0),
                ArmHook::R(&mut st_d1),
            ]
        } else {
            vec![
                ArmHook::None,
                ArmHook::R(&mut st_l0),
                ArmHook::R(&mut st_l1),
            ]
        };
        let mut pairs: Vec<Vec<(f64, f64)>> = Vec::with_capacity(tg_pairs);
        for _ in 0..tg_pairs {
            let triple = timing_triple(
                &mut ctx, &weights, &mut cache, &mut hooks, &prefill_tokens, &decode_tokens,
                bos, &config,
            );
            pairs.push(triple);
        }
        // Per arm: median of per-pair medians + min + median ratio vs arm 0.
        let mut arms_out: Vec<TimingArm> = Vec::new();
        for arm_idx in 0..n_arms {
            let mut meds: Vec<f64> = pairs.iter().map(|p| p[arm_idx].0).collect();
            meds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = meds[meds.len() / 2];
            let min = meds.iter().copied().fold(f64::INFINITY, f64::min);
            let prefill_ms: f64 =
                pairs.iter().map(|p| p[arm_idx].1).sum::<f64>() / pairs.len() as f64;
            let ratio = if arm_idx == 0 {
                None
            } else {
                let mut rs: Vec<f64> = pairs
                    .iter()
                    .map(|p| p[arm_idx].0 / p[0].0)
                    .collect();
                rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                Some(rs[rs.len() / 2])
            };
            arms_out.push(TimingArm {
                name: arm_names[arm_idx],
                med_us: med,
                min_us: min,
                ratio,
                prefill_ms,
            });
        }
        println!(
            "# G2 tg{tg_decode} decode ({} pairs, seq {}, {} mode):",
            tg_pairs,
            tg_prefill + 1,
            if recon_deferred { "deferred" } else { "eager" }
        );
        for a in &arms_out {
            println!(
                "  {:>10}: median {:.0} µs/step ({:.1} tok/s) | min {:.0} µs{} | prefill {:.0} ms",
                a.name,
                a.med_us,
                1e6 / a.med_us,
                a.min_us,
                a.ratio
                    .map(|r| format!(" | ratio {r:.3}×"))
                    .unwrap_or_default(),
                a.prefill_ms
            );
        }
        let mut s = format!(
            "## P3 — G2 tg{} paired interleave\n\n\
             {} interleaved triples over {} arms ({}); {}-token prefill + \
             {} timed decode steps; median of per-pair medians; ratios are medians of \
             per-pair ratios (the katgpt-rs `ab_timing` shape). Mode: **{}**.  \n\n\
             | arm | median µs/step | tok/s | min µs | ratio vs full | prefill ms |\n|---|---|---|---|---|---|\n",
            tg_decode,
            tg_pairs,
            n_arms,
            arm_names[..n_arms].join(", "),
            tg_prefill + 1,
            tg_decode,
            if recon_deferred { "deferred (the T4 fused lane)" } else { "eager (the T3 scratch)" }
        );
        for a in &arms_out {
            s.push_str(&format!(
                "| {} | {:.0} | {:.1} | {:.0} | {} | {:.0} |\n",
                a.name,
                a.med_us,
                1e6 / a.med_us,
                a.min_us,
                a.ratio
                    .map(|r| format!("{r:.3}×"))
                    .unwrap_or_else(|| "1.000×".into()),
                a.prefill_ms
            ));
        }
        s.push_str(&format!(
            "\nBox: {box_note}. Recorded, not gated{}  \n\n"
        , if recon_deferred {
            " — the decision rule reads the WINDOW-EDGE deferred ratio: ≤ 1.20 re-fires the \
             katgpt-core promotion; > 1.20 holds (riir-infer Issue 013 T4 pre-registration)."
        } else {
            "."
        }));
        s
    };
    report.push(&tg);

    // ── P4: the bytes/token record ────────────────────────────────────
    {
        let full = 2 * kvd * 4 * n_layers;
        let p3 = kvd * 4 * n_layers;
        let s = format!(
            "## P4 — KV bytes/token record\n\n\
             - Full cache: 2 × {kvd} × 4 B × {n_layers} layers = **{full} B/token** (K + V, f32).  \n\
             - P3 (key-only): {kvd} × 4 B × {n_layers} = **{p3} B/token** — the law's exact \
             **50.0%** (`n_v/(n_kv+n_v) = 1/2`; gemma-2-2b is 8q:4kv at hd 256).  \n\
             - Sliding-window layers (window 4096): K and V are window-bounded equally, so the \
             fraction holds at every context; below the window this lane's saving is uniform.  \n\
             - Instrument caveat: this lane still WRITES the raw V row at store (one memcpy/step, \
             ~0.03% of step cost); the READ path is fully reconstructed — a production P3 cache \
             drops the V allocation entirely, which is the recorded arithmetic.  \n\n"
        );
        report.push(&s);
        println!("# P4 bytes/token: full {full} B, P3 {p3} B (50.0%)");
    }

    // ── P5: verdicts ──────────────────────────────────────────
    report.push(&format!(
        "## Verdicts\n\n\
         - Read posture: **{}**  \n\
         - **G3 (seam identity): {}**  \n\
         - **Deferred-lane probe (λ0/miss to_bits + regrouping bound): {}**  \n\
         - **G1 (reconstruct == store within rotation rounding): {}**  \n\
         - **G2 (tg{} read-path cost): recorded** (see P3)  \n\
         - **Bytes/token: 50.0%** (the law, recorded)  \n\n\
         A G1 FAIL means a convention/wiring bug (the wrong rotation subgroup, a stale token \
         map), not a model effect — the reconstruction is deterministic algebra.  \n",
        if recon_deferred { "deferred (the T4 fused lane)" } else { "eager (the T3 scratch)" },
        if g3 { "PASS" } else { "FAIL" },
        if recon_deferred {
            if deferred_probe { "PASS" } else { "FAIL" }
        } else {
            "n/a (eager mode)"
        },
        if skip_g1 { "SKIPPED (--skip-g1; record: Bench 013)" } else if g1_pass { "PASS" } else { "FAIL" },
        tg_decode
    ));
    let status = if g3 && deferred_probe && (g1_pass || skip_g1) {
        if skip_g1 {
            "**Status:** COMPLETE — G3 PASS, G1 SKIPPED (G2-scaling posture; G1 record: Bench 013)"
        } else {
            "**Status:** COMPLETE — G3 PASS, G1 PASS (G2/bytes recorded)"
        }
    } else {
        "**Status:** COMPLETE — GATE FAILURE (see verdicts)"
    };
    report.buf = report.buf.replace(
        "**Status:** RUNNING (rewritten after every phase)",
        status,
    );
    report.push("");

    println!(
        "# done: G3 {} deferred-probe {} G1 {}",
        if g3 { "PASS" } else { "FAIL" },
        if recon_deferred {
            if deferred_probe { "PASS" } else { "FAIL" }
        } else {
            "n/a"
        },
        if skip_g1 { "SKIPPED" } else if g1_pass { "PASS" } else { "FAIL" }
    );
    if !(g3 && deferred_probe && (g1_pass || skip_g1)) {
        bail!("gate failure — see the report");
    }
    Ok(())
}
