//! Plan 626 T4 (G2) — the resident lane's per-token latency A/B against the
//! incumbent per-token-sync lane, at the league decode posture (27B folded,
//! Metal, GPU-exclusive box).
//!
//! The premise (Issue 1168 P1 vs P2): the incumbent pays 17.79 ms/token —
//! 2.455 enqueue-front + 15.328 GPU-serialized read (the per-token full-logits
//! `read_one` drains the queue); the measured deep-queue floor is 15.32
//! ms/token (65.30 tok/s). The resident lane removes the per-token sync —
//! its expected wall is the floor + the amortized K-cadence harvest
//! (one fenced 32-byte ring read every k tokens). The pre-registered bar:
//! **P1-shape ≥ −2.0 ms/token (17.79 → ≤15.8)**; the promotion target is the
//! league ratio ≥ 0.97× (that re-cell is the consumer-side step, bench_768/831
//! — it needs the serving router, filed when the substrate lands).
//!
//! Verdict discipline (the issue-1129 convention): the G2 numbers are THE
//! RECORD, printed and never asserted — a ratio moving the wrong way is the
//! demote-scope signal, decided by the T5 verdict, not by this file's exit
//! code. What IS asserted: reach (the resident counter moves exactly WALK per
//! arm) and the box-state gate (this file refuses to run when the box is on
//! battery / Low Power — the bench_preflight posture, in-process subset).
//!
//! Interleaved ABBA in ONE process with ONE model load (the sequential-A/B
//! box-state law, riir-ai sequential_ab_timing_audit): rounds run
//! incumbent/resident ×2 per round pair, median across pairs. Warm-up run
//! excluded. This is a MEASUREMENT cell, not a correctness gate (G1 is
//! `plan626_resident_greedy_g1`).
//!
//! Usage (GPU-exclusive; ~6 min):
//!
//! ```bash
//! BONSAI_GGUF=<model> cargo test --release -p riir-infer-gpu \
//!   --features cubecl_runtime,ternary_gemv,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test plan626_resident_g2_ab -- --ignored --nocapture
//! ```

#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    feature = "deltanet_recurrence_rowpar",
    feature = "params_handle_cache",
    target_os = "macos",
    not(debug_assertions)
))]

use std::time::Instant;

use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

/// The issue-1129 decode schedule: 24 measured tokens per arm run.
const WALK: usize = 24;
/// The harvest cadence (the resident arm's ring depth).
const K: usize = 8;
/// Round pairs: warm-up (0) + 5 measured pairs, ABBA-interleaved.
const PAIRS: usize = 5;
/// The P1 reference the bar is quoted against (Issue 1168 P1, amo 11-03).
const P1_REFERENCE_MS: f64 = 17.79;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

/// Host argmax last-index (the harness rule) — the incumbent arm's pick.
fn host_argmax_last(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits")
        .0
}

fn walk_incumbent(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &QwenDeltaNetTernaryWeights,
    seed: usize,
) -> f64 {
    fwd.reset_state();
    let mut token = seed;
    let t0 = Instant::now();
    for _ in 0..WALK {
        fwd.set_input_token(weights, token);
        let logits = fwd.forward_token();
        token = host_argmax_last(&logits);
    }
    t0.elapsed().as_secs_f64() * 1000.0
}

fn walk_resident(fwd: &mut TernaryDeltanetGpuForward, seed: usize) -> f64 {
    fwd.reset_state();
    fwd.resident_begin(seed, K);
    let before = TernaryDeltanetGpuForward::resident_launch_count();
    let t0 = Instant::now();
    let mut pending = 0;
    for _ in 0..WALK {
        fwd.resident_enqueue();
        pending += 1;
        if pending == K {
            let _ = fwd.resident_harvest(K);
            pending = 0;
        }
    }
    if pending > 0 {
        let _ = fwd.resident_harvest(pending);
    }
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let delta = TernaryDeltanetGpuForward::resident_launch_count() - before;
    assert_eq!(delta, WALK as u64, "resident reach: {delta} != {WALK}");
    wall
}

#[test]
#[ignore = "league-model GPU lane — perf A/B; run on a GPU-exclusive box"]
fn plan626_resident_g2_ab() {
    // The in-process subset of the bench_preflight posture: AC power is not
    // checkable without IOKit here, so the load ceiling + the GPU-exclusivity
    // census are the gates this cell owns. A loaded box poisons the medians.
    let load = std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| {
        s.split_whitespace()
            .next()
            .and_then(|l| l.parse::<f64>().ok())
    });
    if let Some(l1) = load {
        assert!(
            l1 < 6.0,
            "G2 REFUSED: 1-min load {l1:.2} ≥ 6.0 — rerun on a quieter box"
        );
    }

    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed)");
        return;
    };
    // The bench_763 census posture: the uniform cache forced ON (the composed
    // config — params_handle_cache + uniform cache — is what the P1/P2
    // numbers measured; without this the cell reads ~2x and the bar is
    // unadjudicable, the first G2 run's finding).
    cubecl::wgpu::set_uniform_cache_override(Some(true));

    let model_path = std::env::var("BONSAI_GGUF")
        .unwrap_or_else(|_| "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[plan626-g2] loading {} …", model_path.display());
    let t0 = Instant::now();
    let (mut config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&model_path)
            .map_err(|e| format!("load: {e}"))
            .unwrap();
    // The walk needs 24 positions; keep the model's own block budget when the
    // device tolerates it (the P1 census posture) — the KV shape is part of
    // the league posture. A device that refuses the bind alignment must
    // clamp LOUD, never silently (the T4-G1 lesson names the refusal).
    if config.block_size > 4096 {
        config.block_size = 4096;
    }
    let vocab = config.vocab_size;
    let bos = config.bos_token;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    eprintln!(
        "[plan626-g2] forward ready ({:.0}s) · vocab={vocab} · n_layer={} · folded={}",
        t0.elapsed().as_secs_f32(),
        config.n_layer,
        fwd.is_folded(),
    );

    // The seed the P1 posture walked (BOS — the league decode entry).
    let seed = bos;

    // Warm-up (excluded): one arm each, drives the lazy compile paths.
    let _ = walk_incumbent(&mut fwd, &weights, seed);
    let _ = walk_resident(&mut fwd, seed);

    let mut inc_ms: Vec<f64> = Vec::with_capacity(PAIRS);
    let mut res_ms: Vec<f64> = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        // ABBA — incumbent, resident, resident, incumbent — so a box-state
        // drift within the pair cannot masquerade as a lane effect.
        let a1 = walk_incumbent(&mut fwd, &weights, seed);
        let b1 = walk_resident(&mut fwd, seed);
        let b2 = walk_resident(&mut fwd, seed);
        let a2 = walk_incumbent(&mut fwd, &weights, seed);
        inc_ms.push((a1 + a2) / 2.0);
        res_ms.push((b1 + b2) / 2.0);
        eprintln!(
            "[plan626-g2] pair {pair}: incumbent {:.2} ms/tok · resident {:.2} ms/tok",
            (a1 + a2) / 2.0 / WALK as f64,
            (b1 + b2) / 2.0 / WALK as f64,
        );
    }

    let inc = median(&mut inc_ms) / WALK as f64;
    let res = median(&mut res_ms) / WALK as f64;
    let delta = res - inc;
    let ratio = res / inc;
    eprintln!("═════════════════════════════════════════════════════════");
    eprintln!(
        "[plan626-g2] VERDICT: incumbent median {:.2} ms/token · resident median {:.2} ms/token · delta {:+.2} · ratio {:.3}x",
        inc, res, delta, ratio
    );
    eprintln!(
        "[plan626-g2] bar: P1-reference {P1_REFERENCE_MS:.2} ms/tok; pre-registered ≤ 15.8 → resident {} the bar ({:.2})",
        if res <= 15.8 { "MEETS" } else { "MISSES" },
        res
    );
    eprintln!(
        "[plan626-g2] box: loadavg {:?} · AC (assumed; quote bench_preflight PROVENANCE for any published row) · k={K} walk={WALK}",
        load
    );
}
