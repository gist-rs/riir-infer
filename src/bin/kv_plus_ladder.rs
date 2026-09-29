//! kv_plus_ladder — riir-infer Issue 013 T2: the K=V+ λ ladder (the
//! model-bound P2 gate for katgpt-core's `v_from_k_plus`, katgpt-rs
//! `0b768e95d` / Bench 895) on the REAL gemma-2-2b decode path.
//!
//! One process, seven phases:
//!
//! - **A calibrate** — the Bench-004 tapped protocol over
//!   `tokens[0..cal_n)` (or `--table` loads a previously dumped artifact —
//!   BLAKE3-verified — and skips the 2 h pass; T3 reuses the same file).
//! - **B freeze** — `FittedTokenTable::from_calibration(Residual, λ_js=0)`
//!   (the P2 signal `E_l[s] = mean(V − K | s)`), print the per-layer
//!   ρ_l(V−K) dashboard, dump the artifact when `--dump-table` is given.
//! - **G3 probe** — the λ=0 serve hook vs a direct `V := K` copy hook:
//!   `to_bits`-identical logits across a65-position decode, plus the T1
//!   `NoVQuant` delegation probe.
//! - **C1 ladder** — the issue's λ ∈ {0, 0.5, 1} (+ f16 base) on the
//!   held-out eval chunks, arm-major, paired per-position ΔNLL + top-1
//!   flips vs f16 AND vs k-0.00, per-chunk win shares both ways.
//! - **C2 schedule grid** (only if some λ > 0 beats k-0.00) — per layer
//!   λ_l ∈ {0, 0.5, 1} \ {λ*}, all other layers at λ*, ONE sweep on the
//!   held-out search chunk; argmax per layer, ties → λ*. Direct grid
//!   evaluation, never GD. A final pass measures the CHOSEN schedule on
//!   the same chunk (interactions are never summed).
//! - **C3 validation** — the chosen schedule on the FULL eval chunks
//!   (held out from calibration, the ladder, and the grid).
//! - **C4 NIAH** (Bench-814 shape) — filler + one needle + continuation
//!   tail, password-token best rank + hit + answer NLL, teacher-forced.
//! - **D report** — pre-registered gates echoed + verdicts; rewritten
//!   after every arm (a mid-run death keeps completed arms).
//!
//! MEASUREMENT-ONLY: the only claim under test is
//! `quality(K=V+) > quality(K=V)` — no parity claim vs full V. Promotion is
//! katgpt-rs-side and waits on the gates.
//!
//! Usage:
//! ```text
//! kv_plus_ladder <gguf> <corpus> [--cal-tokens N] [--eval-tokens N]
//!                [--search-tokens N] [--lambdas 0,0.5,1] [--seq-len N]
//!                [--table PATH] [--dump-table PATH] [--niah-trials N]
//!                [--report PATH] [--box-note S] [--smoke]
//! ```

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use katgpt_core::fitted_value_table::{FittedTokenTable, VkSignal};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gemma_layer::GemmaTransformerWeightsF16;
use riir_infer_core::gguf_loader::{config_from_gguf_metadata, GgufFile};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::gemma2_calibration::{
    forward_gemma2_f16_tapped, load_gemma2_f16_direct, CalibrationTables,
};
use riir_infer_core::transformer::gemma2_ktov::{KToVState, KvLayerTel};
use riir_infer_core::transformer::vk_harness::{
    argmax, build_niah_trial, dump_fitted_table, load_fitted_table, nll, rank, NiahTrial,
};
use riir_infer_core::transformer::{
    forward_gemma2_f16, forward_gemma2_f16_hk, ForwardContext, NoVQuant, ValueStoreHook,
};
use riir_infer_core::types::{kv_dim, Config};

/// T1's recorded f16 PPL (Bench 011) — the free cross-run determinism check
/// (same fixture, same slices, same protocol ⇒ the base arm must reproduce).
const T1_F16_PPL: f64 = 6.0907;

/// The grid's per-layer candidates (the issue's λ set).
const GRID_CANDIDATES: [f32; 3] = [0.0, 0.5, 1.0];

/// Grid ties resolve to the incumbent (a win smaller than this is a tie).
const GRID_TIE_EPS: f64 = 1e-4;

// ── The arm hook — the ONE forward-dispatch seam both passes share ──────

enum ArmHook<'a> {
    /// The f16 base arm — the plain forward, no hook.
    Plain,
    /// A K=V+ serve arm (uniform or per-layer λ schedule).
    K(&'a mut KToVState),
}

impl ArmHook<'_> {
    fn reset(&mut self) {
        if let ArmHook::K(st) = self {
            st.reset();
        }
    }

    #[inline]
    fn set_token(&mut self, pos: usize, token: u32) {
        if let ArmHook::K(st) = self {
            st.set_token(pos, token);
        }
    }

    #[inline]
    fn set_lam(&mut self, layer: usize, value: f32) {
        if let ArmHook::K(st) = self {
            st.set_lam(layer, value);
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
            ArmHook::K(st) => {
                forward_gemma2_f16_hk(ctx, weights, cache, &mut **st, token, pos, config)
            }
        }
    }
}

// ── The eval pass ────────────────────────────────────────────────────────

#[derive(Default)]
struct PassOut {
    /// Per scored position NLL (the pairing record; arm-major).
    nll: Vec<f64>,
    /// Per scored position argmax (the flip record).
    top: Vec<usize>,
    chunk_sums: Vec<f64>,
    fwd: usize,
    secs: f64,
}

/// `n_chunks` chunks of `[BOS] + (seq_len−1)` corpus tokens over `tokens`,
/// teacher-forced, per-position NLL + argmax.
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
        let mut chunk_sum = 0.0f64;
        for (pos, &token) in seq.iter().enumerate() {
            hook.set_token(pos, token as u32);
            let logits = hook.step(ctx, weights, cache, token, pos, config);
            if pos + 1 >= seq.len() {
                continue;
            }
            let l = nll(logits, seq[pos + 1]);
            chunk_sum += l;
            out.nll.push(l);
            out.top.push(argmax(logits));
            out.fwd += 1;
        }
        out.chunk_sums.push(chunk_sum);
    }
    out.secs = t0.elapsed().as_secs_f64();
    out
}

// ── NIAH (the Bench-814 shape) ───────────────────────────────────────────

struct NiahOut {
    /// Best (minimum) password-token rank per trial.
    best_rank: Vec<usize>,
    /// Per-trial answer NLL (Σ over the password tokens, teacher-forced).
    answer_nll: Vec<f64>,
    fwd: usize,
    secs: f64,
}

/// One NIAH arm over the trials: prompt decode, then teacher-force the
/// password tokens one past the tail, recording each one's rank at its
/// position (Bench 814's password-token best rank; rank 1 = the model's
/// top prediction IS the password token — the greedy-hit equivalent).
fn run_niah_arm(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    hook: &mut ArmHook<'_>,
    trials: &[NiahTrial],
    config: &Config,
) -> NiahOut {
    let mut out = NiahOut {
        best_rank: Vec::with_capacity(trials.len()),
        answer_nll: Vec::with_capacity(trials.len()),
        fwd: 0,
        secs: 0.0,
    };
    let t0 = Instant::now();
    for trial in trials {
        cache.reset();
        hook.reset();
        let seq = &trial.tokens;
        let mut cur: Vec<f32> = Vec::new();
        for (pos, &token) in seq.iter().enumerate() {
            hook.set_token(pos, token as u32);
            let logits = hook.step(ctx, weights, cache, token, pos, config);
            out.fwd += 1;
            if pos == trial.answer_pos {
                cur = logits.to_vec();
            }
        }
        let mut best = usize::MAX;
        let mut sum = 0.0f64;
        for (i, &pt) in trial.password_tokens.iter().enumerate() {
            let r = rank(&cur, pt);
            sum += nll(&cur, pt);
            best = best.min(r);
            if i + 1 < trial.password_tokens.len() {
                let nxt = hook.step(ctx, weights, cache, pt, trial.answer_pos + 1 + i, config);
                cur = nxt.to_vec();
                out.fwd += 1;
            }
        }
        out.best_rank.push(best);
        out.answer_nll.push(sum);
    }
    out.secs = t0.elapsed().as_secs_f64();
    out
}

// ── The G3 probe's direct V:=K copy hook ─────────────────────────────────

/// The reference path a deployment's `VReadPath` would serve at λ=0.
struct DirectCopyState {
    k_pre: Vec<f32>,
    kvd: usize,
}

impl DirectCopyState {
    fn new(kvd: usize) -> Self {
        Self {
            k_pre: vec![0.0; kvd],
            kvd,
        }
    }
}

impl ValueStoreHook for DirectCopyState {
    fn keys_pre_rope(&mut self, _l: usize, _p: usize, k_pre: &[f32]) {
        self.k_pre.copy_from_slice(k_pre);
    }
    fn value_stored(&mut self, _layer_idx: usize, pos: usize, layer_values: &mut [f32]) {
        let off = pos * self.kvd;
        layer_values[off..off + self.kvd].copy_from_slice(&self.k_pre);
    }
}

// ── Arm accounting ───────────────────────────────────────────────────────

/// One ladder arm — raw vectors retained so every pairing (vs f16, vs
/// k-0.00, vs the λ* incumbent) is computed once, post-pass, in one place.
struct LadderArm {
    name: String,
    lam: Option<f32>,
    nlls: Vec<f64>,
    top: Vec<usize>,
    chunk_sums: Vec<f64>,
    fwd: usize,
    secs: f64,
    tel: Vec<KvLayerTel>,
}

impl LadderArm {
    fn ppl(&self) -> f64 {
        if self.nlls.is_empty() {
            0.0
        } else {
            (self.nlls.iter().sum::<f64>() / self.nlls.len() as f64).exp()
        }
    }

    fn mean_d(&self, reference: &LadderArm) -> f64 {
        let n = self.nlls.len().min(reference.nlls.len());
        if n == 0 {
            0.0
        } else {
            (0..n).map(|i| self.nlls[i] - reference.nlls[i]).sum::<f64>() / n as f64
        }
    }

    fn win_count(&self, reference: &LadderArm) -> (usize, usize) {
        let wins = self
            .chunk_sums
            .iter()
            .zip(reference.chunk_sums.iter())
            .filter(|(a, b)| a < b)
            .count();
        (wins, self.chunk_sums.len())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: kv_plus_ladder <gguf> <corpus> [--cal-tokens N] [--eval-tokens N] \
             [--search-tokens N] [--lambdas 0,0.5,1] [--seq-len N] [--table PATH] \
             [--dump-table PATH] [--niah-trials N] [--report PATH] [--box-note S] [--smoke]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let smoke = args.iter().any(|a| a == "--smoke");
    let (mut cal_n, mut eval_n, mut search_n) = if smoke {
        (1024, 512, 256)
    } else {
        (61_440, 12_288, 1024)
    };
    let mut lambdas: Vec<f32> = vec![0.0, 0.5, 1.0];
    let mut seq_len: usize = if smoke { 256 } else { 1024 };
    let mut table_path: Option<PathBuf> = None;
    let mut dump_path: Option<PathBuf> = None;
    let mut niah_trials: usize = if smoke { 1 } else { 6 };
    let mut report_path: Option<PathBuf> = None;
    let mut box_note = String::from("4090 workstation i7-13700K, CPU lane, AC");
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
            "--search-tokens" => {
                search_n = args[i + 1].parse().context("--search-tokens N")?;
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
            "--seq-len" => {
                seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--table" => {
                table_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--dump-table" => {
                dump_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--niah-trials" => {
                niah_trials = args[i + 1].parse().context("--niah-trials N")?;
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
            "--smoke" => i += 1,
            other => bail!("unknown arg {other}"),
        }
    }
    if seq_len > 4096 {
        bail!("--seq-len must stay <= 4096 (gemma-2 sliding window)");
    }

    // ── Load model + tokenizer ────────────────────────────────────────
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
        "# kv_plus_ladder: gemma-2 f16 | layers={} n_embd={} kv_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        kv_dim(&config),
        config.vocab_size,
        t_start.elapsed().as_secs_f32()
    );
    drop(gguf);

    // ── Corpus ────────────────────────────────────────────────────────
    let text = load_corpus_text(&corpus_path)?;
    let all_tokens = tok.encode(&text);
    let need = cal_n + eval_n + search_n + seq_len;
    if all_tokens.len() < need {
        bail!(
            "corpus too short: {} tokens < cal {cal_n} + eval {eval_n} + search {search_n} + seq {seq_len}",
            all_tokens.len()
        );
    }
    println!(
        "# corpus: {} chars → {} tokens | cal [0..{cal_n}) eval [{cal_n}..{}) search [{}..{})",
        text.len(),
        all_tokens.len(),
        cal_n + eval_n,
        cal_n + eval_n,
        cal_n + eval_n + search_n
    );

    // block_size sizes the KV cache rows, ctx.head_scores, and the
    // attention score scratch (attend_row requires t_n ≤ block_size). The
    // NIAH phase teacher-forces the password at positions ≥ seq_len
    // (answer_pos + 1 + i), so everything is sized seq_len + 64 up front;
    // the eval protocol never touches the extra rows, and the arithmetic
    // for positions < seq_len is unchanged (the T1 consistency gate pins
    // that per run).
    config.block_size = seq_len + 64;
    let kvd = kv_dim(&config);
    let n_layers = config.n_layer;
    let corpus_per_chunk = seq_len - 1;
    let n_chunks = eval_n / corpus_per_chunk;
    if n_chunks == 0 {
        bail!("eval slice smaller than one chunk");
    }
    let top_k = 8192usize;

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    // ── Phase A/B: calibrate (or load) + freeze ───────────────────────
    let table: &'static FittedTokenTable;
    let mut rho_vk: Vec<f64> = Vec::new();
    let mut table_loaded = false;
    let mut table_sha = String::from("(none)");
    match &table_path {
        Some(p) => {
            let (t, meta) = load_fitted_table(p, Some((n_layers, kvd)))?;
            table_sha = format!("blake3-verified artifact, cal_tokens {}", meta.cal_tokens);
            table = Box::leak(Box::new(t));
            table_loaded = true;
            println!("# table loaded: {} ({} rows)", p.display(), meta.rows);
        }
        None => {
            let cal_tokens: Vec<usize> = all_tokens[..cal_n].to_vec();
            let mut counts = vec![0u64; config.vocab_size];
            for &t in &cal_tokens {
                counts[t] += 1;
            }
            let t_cal = Instant::now();
            let mut tables = CalibrationTables::from_counts(n_layers, kvd, counts, top_k);
            let mut done = 0usize;
            let mut next_report = 0usize;
            for chunk in cal_tokens.chunks(seq_len) {
                cache.reset();
                for (pos, &token) in chunk.iter().enumerate() {
                    forward_gemma2_f16_tapped(
                        &mut ctx, &weights, &mut cache, &mut tables, token, pos, &config,
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
                "# calibrated in {:.0}s ({:.0} tok/s)",
                t_cal.elapsed().as_secs_f32(),
                cal_n as f32 / t_cal.elapsed().as_secs_f32().max(1e-6)
            );
            for l in tables.layers.iter() {
                rho_vk.push(f64::from(l.vk.r_squared().aggregate));
            }
            table = Box::leak(Box::new(FittedTokenTable::from_calibration(
                &tables,
                VkSignal::Residual,
                0.0,
            )));
            if let Some(dp) = &dump_path {
                let hex = dump_fitted_table(table, dp, cal_n as u64, config.vocab_size)
                    .context("dump table")?;
                // Load-verify the artifact in-process (the round-trip probe).
                let (back, _) = load_fitted_table(dp, Some((n_layers, kvd)))?;
                for &probe_tok in [0u32, 1, 257, 8191].iter() {
                    for l in [0, n_layers / 2, n_layers - 1] {
                        assert_eq!(
                            back.row(l, probe_tok).map(<[f32]>::to_vec),
                            table.row(l, probe_tok).map(<[f32]>::to_vec),
                            "table round-trip diverged at ({l}, {probe_tok})"
                        );
                    }
                }
                table_sha = format!("blake3:{}… (round-trip verified)", &hex[..16]);
                println!("# table dumped: {} sha {}…", dp.display(), &hex[..16]);
            }
        }
    }

    // ── G3 probe: λ=0 hook vs the direct V:=K copy ────────────────────
    let g3_pass = {
        let probe_tokens: Vec<usize> = all_tokens[cal_n..cal_n + 64].to_vec();
        let mut st_a = KToVState::new_uniform(table, 0.0, n_layers, kvd, 128);
        let mut st_b = DirectCopyState::new(kvd);
        let mut ok;
        {
            let mut seq = vec![bos];
            seq.extend_from_slice(&probe_tokens);
            let mut a_logits: Vec<Vec<u32>> = Vec::new();
            let mut b_logits: Vec<Vec<u32>> = Vec::new();
            cache.reset();
            for (pos, &t) in seq.iter().enumerate() {
                st_a.set_token(pos, t as u32);
                let lg =
                    forward_gemma2_f16_hk(&mut ctx, &weights, &mut cache, &mut st_a, t, pos, &config);
                a_logits.push(lg.iter().map(|x| x.to_bits()).collect());
            }
            cache.reset();
            for (pos, &t) in seq.iter().enumerate() {
                let lg =
                    forward_gemma2_f16_hk(&mut ctx, &weights, &mut cache, &mut st_b, t, pos, &config);
                b_logits.push(lg.iter().map(|x| x.to_bits()).collect());
            }
            ok = a_logits == b_logits;
            // Belt-and-braces: the NoVQuant delegation stays bit-identical.
            if ok {
                let t0 = probe_tokens[0];
                cache.reset();
                let a: Vec<u32> =
                    forward_gemma2_f16(&mut ctx, &weights, &mut cache, t0, 0, &config)
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                cache.reset();
                let b: Vec<u32> = forward_gemma2_f16_hk(
                    &mut ctx, &weights, &mut cache, &mut NoVQuant, t0, 0, &config,
                )
                .iter()
                .map(|x| x.to_bits())
                .collect();
                ok = a == b;
            }
        }
        println!("# G3 probe: {}", if ok { "PASS" } else { "FAIL" });
        ok
    };

    // ── Phase C1: the ladder ──────────────────────────────────────────
    let eval_tokens: Vec<usize> = all_tokens[cal_n..cal_n + eval_n].to_vec();
    let search_tokens: Vec<usize> = all_tokens[cal_n + eval_n..cal_n + eval_n + search_n].to_vec();
    println!(
        "# eval: {n_chunks} chunks × {corpus_per_chunk} = {} scored per arm | λ {lambdas:?}",
        n_chunks * corpus_per_chunk
    );
    let mut arms: Vec<LadderArm> = Vec::new();
    let t_eval = Instant::now();

    // f16 base.
    {
        let mut hook = ArmHook::Plain;
        let po = run_pass(
            &mut ctx, &weights, &mut cache, &mut hook, &eval_tokens, corpus_per_chunk, n_chunks,
            bos, &config,
        );
        println!(
            "# arm f16 done: {} scored | ppl {:.4} | {:.0} fwd/s",
            po.nll.len(),
            (po.nll.iter().sum::<f64>() / po.nll.len().max(1) as f64).exp(),
            po.fwd as f64 / po.secs.max(1e-9)
        );
        arms.push(LadderArm {
            name: "f16".into(),
            lam: None,
            nlls: po.nll,
            top: po.top,
            chunk_sums: po.chunk_sums,
            fwd: po.fwd,
            secs: po.secs,
            tel: Vec::new(),
        });
        let _ = std::io::stdout().flush();
    }

    // λ arms (0 first — the k-0 pairing base).
    for &lam in lambdas.iter() {
        let name = format!("k-{lam:.2}");
        let mut st = KToVState::new_uniform(table, lam, n_layers, kvd, seq_len + 32);
        let po = {
            let mut hook = ArmHook::K(&mut st);
            run_pass(
                &mut ctx, &weights, &mut cache, &mut hook, &eval_tokens, corpus_per_chunk,
                n_chunks, bos, &config,
            )
        };
        let tel: Vec<KvLayerTel> = st.layer_tel().to_vec();
        println!(
            "# arm {name} done: {} scored | ppl {:.4} | {:.0} fwd/s",
            po.nll.len(),
            (po.nll.iter().sum::<f64>() / po.nll.len().max(1) as f64).exp(),
            po.fwd as f64 / po.secs.max(1e-9)
        );
        arms.push(LadderArm {
            name,
            lam: Some(lam),
            nlls: po.nll,
            top: po.top,
            chunk_sums: po.chunk_sums,
            fwd: po.fwd,
            secs: po.secs,
            tel,
        });
        let _ = std::io::stdout().flush();
    }

    // λ* — the best-beating λ > 0 (mean paired ΔNLL vs k-0). Computed in a
    // scoped block so the arm borrows end before the k-sched push below.
    let (lam_star, lam_star_d): (Option<f32>, Option<f64>) = {
        let k0 = arms.iter().find(|a| a.lam == Some(0.0)).expect("k-0 arm");
        let mut best: Option<(f32, f64)> = None;
        for a in arms.iter().filter(|a| a.lam.is_some_and(|l| l > 0.0)) {
            let d = a.mean_d(k0);
            if d < 0.0 && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((a.lam.expect("filtered"), d));
            }
        }
        match best {
            Some((l, d)) => {
                println!("# λ* = {l} (mean ΔNLL vs k-0 {d:+.5}) — grid armed");
                (Some(l), Some(d))
            }
            None => {
                println!("# no λ > 0 beats k-0 — G-A FAIL, grid skipped (the recorded negative)");
                (None, None)
            }
        }
    };

    // ── Phase C2: the schedule grid ───────────────────────────────────
    let mut schedule: Vec<f32> = Vec::new();
    let mut grid_note = String::from("skipped (no λ > 0 beat k-0)");
    let mut sched_search_d: Option<f64> = None;
    let grid_armed = lam_star.filter(|_| !smoke);
    if let Some(ls) = grid_armed {
        schedule = vec![ls; n_layers];
        let mut st = KToVState::new(table, schedule.clone(), n_layers, kvd, seq_len + 32);
        let mut hook = ArmHook::K(&mut st);
        let inc = run_pass(
            &mut ctx, &weights, &mut cache, &mut hook, &search_tokens, corpus_per_chunk, 1, bos,
            &config,
        );
        println!(
            "# grid: incumbent (all λ={ls}) search-chunk ppl {:.4}",
            (inc.nll.iter().sum::<f64>() / inc.nll.len().max(1) as f64).exp()
        );
        for (l, slot) in schedule.iter_mut().enumerate() {
            let mut best_cand = ls;
            let mut best_d = 0.0f64;
            for cand in GRID_CANDIDATES {
                if (cand - ls).abs() < 1e-9 {
                    continue;
                }
                hook.set_lam(l, cand);
                let po = run_pass(
                    &mut ctx, &weights, &mut cache, &mut hook, &search_tokens, corpus_per_chunk,
                    1, bos, &config,
                );
                let n = po.nll.len().min(inc.nll.len());
                let d = (0..n).map(|i| po.nll[i] - inc.nll[i]).sum::<f64>() / n as f64;
                if d < best_d - GRID_TIE_EPS {
                    best_d = d;
                    best_cand = cand;
                }
                hook.set_lam(l, ls);
            }
            *slot = best_cand;
            hook.set_lam(l, best_cand);
            println!("# grid layer {l}: λ {best_cand} (Δ {best_d:+.5})");
            let _ = std::io::stdout().flush();
        }
        // The CHOSEN schedule, measured (never summed through layers).
        let chosen = run_pass(
            &mut ctx, &weights, &mut cache, &mut hook, &search_tokens, corpus_per_chunk, 1, bos,
            &config,
        );
        let n = chosen.nll.len().min(inc.nll.len());
        let d = (0..n).map(|i| chosen.nll[i] - inc.nll[i]).sum::<f64>() / n as f64;
        sched_search_d = Some(d);
        grid_note.clear();
        println!("# grid done: chosen ΔNLL vs incumbent {d:+.5} on the search chunk");
    } else if let Some(ls) = lam_star {
        // Smoke: the grid is priced out; the schedule degenerates to λ*.
        schedule = vec![ls; n_layers];
        grid_note = "smoke — grid skipped, schedule = uniform λ*".into();
    }

    // ── Phase C3: schedule validation on the full eval ────────────────
    if !schedule.is_empty() {
        let mut st = KToVState::new(table, schedule.clone(), n_layers, kvd, seq_len + 32);
        let po = {
            let mut hook = ArmHook::K(&mut st);
            run_pass(
                &mut ctx, &weights, &mut cache, &mut hook, &eval_tokens, corpus_per_chunk,
                n_chunks, bos, &config,
            )
        };
        let tel: Vec<KvLayerTel> = st.layer_tel().to_vec();
        println!(
            "# arm k-sched done: ppl {:.4} | {:.0} fwd/s",
            (po.nll.iter().sum::<f64>() / po.nll.len().max(1) as f64).exp(),
            po.fwd as f64 / po.secs.max(1e-9)
        );
        arms.push(LadderArm {
            name: "k-sched".into(),
            lam: None,
            nlls: po.nll,
            top: po.top,
            chunk_sums: po.chunk_sums,
            fwd: po.fwd,
            secs: po.secs,
            tel,
        });
        let _ = std::io::stdout().flush();
    }

    // ── Phase C4: NIAH ────────────────────────────────────────────────
    let mut niah: Vec<(String, NiahOut)> = Vec::new();
    if niah_trials > 0 {
        let mut trials: Vec<NiahTrial> = Vec::with_capacity(niah_trials);
        for t in 0..niah_trials {
            let password = format!("sunset{}", 1000 + 137 * t);
            let depth = [0.25f32, 0.5, 0.75][t % 3];
            let trial = build_niah_trial(&tok, bos, seq_len, depth, &password)
                .with_context(|| format!("niah trial {t} ({password})"))?;
            trials.push(trial);
        }
        println!("# niah: {} trials built", trials.len());

        // f16.
        {
            let mut hook = ArmHook::Plain;
            let o = run_niah_arm(&mut ctx, &weights, &mut cache, &mut hook, &trials, &config);
            println!("# niah f16: median rank {:?} | hits {}/{}", {
                let mut r = o.best_rank.clone();
                r.sort_unstable();
                r.get(r.len() / 2).copied().unwrap_or(0)
            }, o.best_rank.iter().filter(|&&x| x == 1).count(), o.best_rank.len());
            niah.push(("f16".into(), o));
        }
        // λ arms.
        for &lam in lambdas.iter() {
            let mut st = KToVState::new_uniform(table, lam, n_layers, kvd, seq_len + 32);
            let mut hook = ArmHook::K(&mut st);
            let o = run_niah_arm(&mut ctx, &weights, &mut cache, &mut hook, &trials, &config);
            let name = format!("k-{lam:.2}");
            println!("# niah {name}: done");
            niah.push((name, o));
        }
        // The schedule arm (when the grid ran).
        if !schedule.is_empty() {
            let mut st = KToVState::new(table, schedule.clone(), n_layers, kvd, seq_len + 32);
            let mut hook = ArmHook::K(&mut st);
            let o = run_niah_arm(&mut ctx, &weights, &mut cache, &mut hook, &trials, &config);
            println!("# niah k-sched: done");
            niah.push(("k-sched".into(), o));
        }
    }

    // ── Phase D: the report ───────────────────────────────────────────
    // Telemetry source: the λ=1 arm when present, else the largest λ.
    let tel_src = arms
        .iter()
        .filter(|a| a.lam.is_some_and(|l| l > 0.0) && !a.tel.is_empty())
        .max_by(|a, b| a.lam.unwrap().partial_cmp(&b.lam.unwrap()).unwrap());
    let tel_cos: Vec<f64> = tel_src
        .map(|a| a.tel.iter().map(|t| t.cos_kv()).collect())
        .unwrap_or_default();
    let tel_refund: Vec<f64> = tel_src
        .map(|a| a.tel.iter().map(|t| t.refund_share()).collect())
        .unwrap_or_default();

    // Pairings (computed once, here — after every push, so the borrows are
    // fresh and the base/k-0 rows are found in the FINAL arm set).
    let base_arm = arms.first().expect("base arm exists");
    let k0 = arms.iter().find(|a| a.lam == Some(0.0));
    struct Row {
        name: String,
        ppl: f64,
        dp_base: Option<f64>,
        dn_base: Option<f64>,
        fl_base: Option<f64>,
        win_base: Option<(usize, usize)>,
        dp_k0: Option<f64>,
        dn_k0: Option<f64>,
        fl_k0: Option<f64>,
        win_k0: Option<(usize, usize)>,
        tok_s: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    for a in &arms {
        let is_base = std::ptr::eq(a, base_arm);
        let (dp_base, dn_base, fl_base, win_base) = if is_base {
            (None, None, None, None)
        } else {
            match base_pair(a, base_arm) {
                Some((dp, dn, fl, w)) => (Some(dp), Some(dn), Some(fl), Some(w)),
                None => (None, None, None, None),
            }
        };
        let (dp_k0, dn_k0, fl_k0, win_k0) = if a.lam == Some(0.0) {
            (None, None, None, None)
        } else {
            match k0 {
                Some(k0a) => {
                    let dp = 100.0 * ((a.ppl() / k0a.ppl()) - 1.0);
                    let dn = a.mean_d(k0a);
                    let n = a.nlls.len().min(k0a.nlls.len());
                    let fl = 100.0
                        * (0..n).filter(|&i| a.top[i] != k0a.top[i]).count() as f64
                        / n as f64;
                    let w = a.win_count(k0a);
                    (Some(dp), Some(dn), Some(fl), Some(w))
                }
                None => (None, None, None, None),
            }
        };
        rows.push(Row {
            name: a.name.clone(),
            ppl: a.ppl(),
            dp_base,
            dn_base,
            fl_base,
            win_base,
            dp_k0,
            dn_k0,
            fl_k0,
            win_k0,
            tok_s: a.fwd as f64 / a.secs.max(1e-9),
        });
    }

    let mut out = String::new();
    out.push_str("# Issue 013 T2 — K=V+ λ ladder: gemma-2-2b decode (kv_plus_ladder)\n\n");
    out.push_str(&format!(
        "seq_len {seq_len} | cal {cal_n} eval {eval_n} search {search_n} tokens | top_k {top_k} | table {}\n\n",
        table_sha
    ));
    out.push_str(&format!("box: {box_note}\n\n"));
    if smoke {
        out.push_str("**SMOKE RUN — harness check only, never evidence.**\n\n");
    }
    out.push_str(&format!(
        "- G3 probe (hard): {}{}\n- λ*: {}\n- grid: {grid_note}\n\n",
        if g3_pass { "PASS" } else { "FAIL" },
        if table_loaded { " | table LOADED (".to_owned() + &table_sha + ")" } else { String::new() },
        lam_star.map_or("(none)".into(), |l| format!(
            "{l} (ΔNLL vs k-0 {:+.5})",
            lam_star_d.unwrap_or(f64::NAN)
        )),
    ));

    out.push_str("## Ladder (held-out eval)\n\n");
    out.push_str("| arm | ppl | Δppl f16 | ΔNLL f16 | flip f16 | win f16 | Δppl k-0 | ΔNLL k-0 | flip k-0 | win k-0 | tok/s |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in &rows {
        let dpb = r.dp_base.map_or("—".into(), |x| format!("{x:+.3}%"));
        let dnb = r.dn_base.map_or("—".into(), |x| format!("{x:+.5}"));
        let flb = r.fl_base.map_or("—".into(), |x| format!("{x:.2}%"));
        let wb = r
            .win_base
            .map_or("—".into(), |(w, n)| format!("{w}/{n}"));
        let dpk = r.dp_k0.map_or("—".into(), |x| format!("{x:+.3}%"));
        let dnk = r.dn_k0.map_or("—".into(), |x| format!("{x:+.5}"));
        let flk = r.fl_k0.map_or("—".into(), |x| format!("{x:.2}%"));
        let wk = r.win_k0.map_or("—".into(), |(w, n)| format!("{w}/{n}"));
        out.push_str(&format!(
            "| {} | {:.4} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} |\n",
            r.name, r.ppl, dpb, dnb, flb, wb, dpk, dnk, flk, wk, r.tok_s
        ));
    }
    out.push('\n');

    // Gates.
    out.push_str("## Gates (pre-registered)\n\n");
    if let Some(k0a) = k0 {
        let b = base_arm;
        let tax = 100.0 * ((k0a.ppl() / b.ppl()) - 1.0);
        out.push_str(&format!(
            "- **Tax cross-check:** Δppl(k-0 − f16) = {tax:+.3}% — the V:=K cost the refund is measured against (issue cited 2.5–3.1%).\n"
        ));
        out.push_str(&format!(
            "- **Consistency:** f16 ppl {:.4} vs T1's {T1_F16_PPL:.4} (Δ {:+.3}%).\n",
            b.ppl(),
            100.0 * ((b.ppl() / T1_F16_PPL) - 1.0)
        ));
    }
    match lam_star {
        Some(l) => {
            let best = arms
                .iter()
                .find(|a| a.lam == Some(l))
                .expect("λ* arm exists");
            let k0a = k0.expect("k-0 exists");
            out.push_str(&format!(
                "- **G-A (the claim): PASS** — k-{l:.2} mean paired ΔNLL vs k-0 = {:+.5} (< 0); win share {}/{} chunks.\n",
                best.mean_d(k0a),
                best.win_count(k0a).0,
                best.win_count(k0a).1
            ));
        }
        None => out.push_str(
            "- **G-A (the claim): FAIL** — no λ > 0 beat k-0 on mean paired ΔNLL: the table refunds nothing measurable on this fixture.\n",
        ),
    }
    out.push_str(&format!(
        "- **G-C (bit-identity): {}**\n",
        if g3_pass { "PASS" } else { "FAIL" }
    ));
    if let (Some(sd), Some(sa)) = (sched_search_d, arms.iter().find(|a| a.name == "k-sched")) {
        let k0a = k0.expect("k-0 exists");
        out.push_str(&format!(
            "- **G-D (schedule transfer):** search ΔNLL vs incumbent {sd:+.5} → validation ΔNLL vs k-0 {:+.5} (ppl {:.4}); uniform λ* arm for the same comparison: ΔNLL {:+.5}.\n",
            sa.mean_d(k0a),
            sa.ppl(),
            lam_star
                .and_then(|l| arms.iter().find(|a| a.lam == Some(l)))
                .map_or(f64::NAN, |a| a.mean_d(k0a))
        ));
    }
    out.push('\n');

    if !rho_vk.is_empty() && !rho_vk[0].is_nan() {
        out.push_str("## Calibration dashboard — ρ_l(V−K) and the λ=1 telemetry\n\n");
        out.push_str("| layer | ρ_l(V−K) | cos(K,V) | λ²‖E‖²/‖V‖² |\n|---|---|---|---|\n");
        for (l, r) in rho_vk.iter().enumerate() {
            out.push_str(&format!(
                "| {l} | {r:.4} | {:.4} | {:.4} |\n",
                tel_cos.get(l).copied().unwrap_or(f64::NAN),
                tel_refund.get(l).copied().unwrap_or(f64::NAN)
            ));
        }
        out.push('\n');
    }

    if !schedule.is_empty() {
        out.push_str(&format!(
            "## Chosen per-layer schedule\n\n`{:?}`\n\n",
            schedule
        ));
    }

    if !niah.is_empty() {
        let t_n = niah.first().map(|(_, o)| o.best_rank.len()).unwrap_or(0);
        out.push_str(&format!(
            "## NIAH (Bench-814 shape, {t_n} trials, direction-only)\n\n| arm | median best rank | hits (rank 1) | mean answer NLL |\n|---|---|---|---|\n"
        ));
        for (name, o) in &niah {
            let mut r = o.best_rank.clone();
            r.sort_unstable();
            let med = r.get(r.len() / 2).copied().unwrap_or(0);
            let hits = o.best_rank.iter().filter(|&&x| x == 1).count();
            let m = if o.answer_nll.is_empty() {
                0.0
            } else {
                o.answer_nll.iter().sum::<f64>() / o.answer_nll.len() as f64
            };
            out.push_str(&format!(
                "| {name} | {med} | {hits}/{} | {m:.3} |\n",
                o.best_rank.len()
            ));
        }
        out.push('\n');
    }

    out.push_str(
        "\n---\n*Measurement-only (issue 013 T2). The only claim under test is quality(K=V+) > quality(K=V); no parity claim vs full V. Promotion is katgpt-rs-side.*\n",
    );

    print!("{out}");
    let _ = std::io::stdout().flush();
    if let Some(p) = &report_path {
        std::fs::write(p, &out).with_context(|| format!("write report {}", p.display()))?;
        eprintln!("# report written: {}", p.display());
    }
    println!(
        "# done in {:.0}s | box: {box_note}",
        t_eval.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Base-arm pairing helper (kept at fn granularity so the row loop stays
/// linear): returns (Δppl%, ΔNLL, flip%, win) of `a` vs `base`.
fn base_pair(
    a: &LadderArm,
    base: &LadderArm,
) -> Option<(f64, f64, f64, (usize, usize))> {
    if std::ptr::eq(a, base) || a.nlls.is_empty() || base.nlls.is_empty() {
        return None;
    }
    let dp = 100.0 * ((a.ppl() / base.ppl()) - 1.0);
    let dn = a.mean_d(base);
    let n = a.nlls.len().min(base.nlls.len());
    let fl = 100.0 * (0..n).filter(|&i| a.top[i] != base.top[i]).count() as f64 / n as f64;
    let w = a.win_count(base);
    Some((dp, dn, fl, w))
}
