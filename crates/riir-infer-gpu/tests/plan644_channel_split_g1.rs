//! riir-ai Plan 644 T1/T4 — the ANE‖GPU channel-split gate on the folded
//! Bonsai-2 file (M5 Ultra, Metal + ANE).
//!
//! ONE forward is constructed with the split armed (so every site carries
//! its ANE prefix program + GPU suffix rows); the runtime flag then toggles
//! split ON/OFF per pass, so both arms share the instance, the pool, the
//! weights and the tokens. Arms run INTERLEAVED (off, on, off, on, ...) so a
//! drifting box loads both equally.
//!
//! - G1: last-position logits, split vs GPU-only — cosine ≥ 0.9999 and the
//!   same argmax (the ANE computes in fp16 with per-row int8 requant, so the
//!   bar is the Bench-1150 stretch class, not bit-identity).
//! - G2 (reported here, gated in T4): tok/s ratio split/GPU-only, best and
//!   median over the interleaved pairs.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/p644 cargo test -p riir-infer-gpu --release \
//!   --features cubecl_runtime,ternary_gemm_batched,multi_host,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard,metal_tensor_gemm,ane_zc_channel_split,deltanet_prework_fused,deltanet_recurrence_smem_staged \
//!   --test plan644_channel_split_g1 -- --ignored --nocapture --exact plan644_channel_split_g1
//! ```
//!
//! ⛔ ONE TEST PER PROCESS (`--exact <name>`; the B1 test likewise). ANE
//! program slots are device-global and a dropped forward releases its ~127
//! programs only after seconds, so a test that follows another in the same
//! process registers 0 split sites and compares GPU-only against GPU-only.
//! G1 therefore REFUSES a pass whose split arm ran zero ANE evals (measured
//! 2026-10-31: G1 after B1 read 1.000× with max|Δ| 0.0000 — a green zero).
//!
//! The last two features are the riir-gpu serving defaults — the GPU-only
//! denominator must be the shipped prefill (riir-infer Issue 1159: without
//! them a ragged p ran the incumbent prework's per-token conv1d). Production
//! posture 2026-10-31: 1.186× @2048, 1.164× @4096, 1.103× @3000.
//!
//! ⛔ The denominator is the STRICT tensor GEMM unless
//! `RIIR_METAL_TENSOR_RELAXED=1` (the dist lane's shipped posture, Issue 1146
//! rung 2b). Against RELAXED the split LOSES (riir-ai Issue 1147 B8 verdict,
//! 2026-10-31): p=2048 f=0.35 0.765×, f=0.15 1.012×; 8×128 frames over
//! [0, 32) f=0.35 0.847×, f=0.10 1.015×. A split win must be re-read there.
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `P644_P` (prompt tokens, default
//! 2048; whole 2048-token blocks split, the tail runs GPU-only), `P644_PAIRS`
//! (timed interleaved pairs, default 4), `P644_F` (ANE row fraction),
//! `P644_BLOCK` (ANE tokens per program eval, default 2048 — Issue 1147 B8),
//! `P644_WARM` (untimed warm-up pairs, default 2; 0 times the cold first
//! pass), `P644_FRAMES` / `P644_HI` (the dist shape: that many consecutive
//! p-token frames over layers `[0, hi)`; default 1 frame, the whole model).
//! Diagnostic: `RIIR_P644_GPU_HALF_ONLY=1` runs only the GPU half of each
//! split (G1 FAILS by design) — prices the GPU-side structure apart from the
//! ANE (measured 1.144× at f=0.40: the seam's GPU structure is efficient).

#![cfg(all(
    feature = "ane_zc_channel_split",
    feature = "multi_host",
    target_os = "macos",
    target_arch = "aarch64"
))]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::ane_prefill::channel_split::{
    set_prefill_ane_channel_split, set_prefill_ane_channel_split_block,
    set_prefill_ane_channel_split_fraction,
};
use riir_infer_gpu::ane_prefill::exec_zc::{
    set_zc_fault_at_eval, zc_eval_ordinal, zc_lane_fail_count, zc_stage_dump, zc_stage_reset,
};
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::set_prefill_use_metal_tensor_zerocopy;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{key}={v}: not an integer"))
        })
        .unwrap_or(default)
}

fn tokens(count: usize, bos: usize, vocab: usize) -> Vec<usize> {
    (0..count)
        .map(|i| match i {
            0 => bos,
            _ => (bos + i * 7919) % vocab,
        })
        .collect()
}

/// One whole-prompt prefill (a single frame); returns (wall, final logits).
fn pass(
    fwd: &mut TernaryDeltanetGpuForward,
    toks: &[usize],
    n_layer: usize,
) -> (Duration, Vec<f32>) {
    pass_frames(fwd, toks, toks.len(), n_layer, n_layer)
}

/// A dist-shaped prefill (riir-ai Issue 1147 B8): `frame`-token chunks at
/// advancing positions over layers `[0, hi)`. Returns (wall, the final
/// logits when `hi == n_layer`, else the last frame's hidden rows — the
/// stage's handoff).
fn pass_frames(
    fwd: &mut TernaryDeltanetGpuForward,
    toks: &[usize],
    frame: usize,
    hi: usize,
    n_layer: usize,
) -> (Duration, Vec<f32>) {
    fwd.reset_state();
    let t0 = Instant::now();
    let mut out = Vec::new();
    for (i, chunk) in toks.chunks(frame).enumerate() {
        out = fwd.prefill_range_chunk(
            chunk,
            None,
            i * frame,
            toks.len(),
            LayerRange::new(0, hi as u32),
            true,
        );
    }
    let out = match hi == n_layer {
        true => fwd.read_logits(),
        false => out,
    };
    (t0.elapsed(), out)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

/// First-index argmax (Issue 940 tie rule).
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| match x > bv {
            true => (i, x),
            false => (bi, bv),
        })
        .0
}

#[test]
#[ignore = "requires Metal + ANE + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn plan644_channel_split_g1() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let p = env_usize("P644_P", 2048);
    let pairs = env_usize("P644_PAIRS", 4);
    // Any p: whole 2048-token blocks split, the tail runs full-width on
    // the GPU (riir-ai Issue 1147 B4).
    assert!(p > 0, "P644_P must be positive");
    if let Ok(f) = std::env::var("P644_F") {
        set_prefill_ane_channel_split_fraction(f.parse().expect("P644_F: f64"));
    }
    if let Ok(b) = std::env::var("P644_BLOCK") {
        set_prefill_ane_channel_split_block(b.parse().expect("P644_BLOCK: usize"));
    }

    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("model load");
    let mut config = config;
    // Dist shape (Issue 1147 B8): `P644_FRAMES` consecutive p-token frames
    // over layers [0, `P644_HI`) — default one frame, the whole model.
    let frames = env_usize("P644_FRAMES", 1).max(1);
    config.block_size = p * frames + 1;
    let n_layer = config.n_layer;
    let hi = env_usize("P644_HI", n_layer).clamp(1, n_layer);
    let toks = tokens(p * frames, config.vocab_size / 2, config.vocab_size);
    let pass = |fwd: &mut TernaryDeltanetGpuForward, toks: &[usize], n_layer: usize| {
        pass_frames(fwd, toks, p, hi, n_layer)
    };

    // Arm BEFORE construction: the zero-copy GEMM and the split bank are
    // both built in `new()`.
    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_ane_channel_split(true);
    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    // Warm both arms (kernel JIT, ANE program first-run). `P644_WARM`
    // pairs (default 2); `P644_WARM=0` times the COLD first pass — the
    // cold-start read of riir-ai Issue 1147 B8 (a dist stage is one fresh
    // process per prompt).
    for _ in 0..env_usize("P644_WARM", 2) {
        for on in [false, true] {
            set_prefill_ane_channel_split(on);
            pass(&mut fwd, &toks, n_layer);
        }
    }

    let (mut t_off, mut t_on) = (Vec::new(), Vec::new());
    let (mut l_off, mut l_on) = (Vec::new(), Vec::new());
    zc_stage_reset();
    let e0 = zc_eval_ordinal();
    for _ in 0..pairs {
        set_prefill_ane_channel_split(false);
        let (w, l) = pass(&mut fwd, &toks, n_layer);
        t_off.push(w.as_secs_f64());
        l_off = l;
        set_prefill_ane_channel_split(true);
        let (w, l) = pass(&mut fwd, &toks, n_layer);
        t_on.push(w.as_secs_f64());
        l_on = l;
    }
    set_prefill_ane_channel_split(false);
    let split_evals = zc_eval_ordinal() - e0;
    // Split-arm stage decomposition summed over the timed pairs (the
    // GPU-only arm never enters the ZC worker).
    println!("  split stages over {pairs} passes: {}", zc_stage_dump());

    let cos = cosine(&l_off, &l_on);
    let (a_off, a_on) = (argmax(&l_off), argmax(&l_on));
    let max_abs = l_off
        .iter()
        .zip(&l_on)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let ratios: Vec<f64> = t_off.iter().zip(&t_on).map(|(o, n)| o / n).collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let best = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "═══ Plan 644 channel split @ p={p} × {frames} frame(s), layers [0, {hi}), {pairs} interleaved pairs ═══"
    );
    println!(
        "  gpu-only: best {:.1} ms ({:.1} tok/s) · split: best {:.1} ms ({:.1} tok/s)",
        best(&t_off) * 1e3,
        (p * frames) as f64 / best(&t_off),
        best(&t_on) * 1e3,
        (p * frames) as f64 / best(&t_on)
    );
    println!(
        "  per-pass ms off {:?} · on {:?}",
        t_off
            .iter()
            .map(|t| format!("{:.1}", t * 1e3))
            .collect::<Vec<_>>(),
        t_on.iter()
            .map(|t| format!("{:.1}", t * 1e3))
            .collect::<Vec<_>>()
    );
    println!(
        "  speedup per pair (off/on): {:?} → median {:.3}× (min {:.3}×, max {:.3}×)",
        ratios.iter().map(|r| format!("{r:.3}")).collect::<Vec<_>>(),
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() - 1]
    );
    println!("  G1: cosine {cos:.7} · argmax {a_off} vs {a_on} · max|Δ| {max_abs:.4}");
    assert!(
        split_evals > 0,
        "Plan 644 G1: the split arm ran 0 ANE evals — no site registered (another \
         forward still holds the device's ANE programs?); G1/G2 would compare \
         GPU-only against itself"
    );
    assert_eq!(a_off, a_on, "Plan 644 G1: split moved the argmax");
    assert!(cos >= 0.9999, "Plan 644 G1: cosine {cos} < 0.9999");
}

/// riir-ai Issue 1147 B1 — fail-open AFTER submission, EXECUTED: a lane
/// eval is made to fail mid-pass (after its seams' GPU complements and
/// consumers are queued). The chunk driver must restore the GDN state and
/// re-run the chunk GPU-only, so the final logits are BIT-IDENTICAL to the
/// GPU-only reference — stale split columns could not produce that.
#[test]
#[ignore = "requires Metal + ANE + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn plan644_b1_fail_open_after_submit() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let p = env_usize("P644_P", 2048);
    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("model load");
    let mut config = config;
    config.block_size = p + 1;
    let n_layer = config.n_layer;
    let toks = tokens(p, config.vocab_size / 2, config.vocab_size);
    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_ane_channel_split(true);
    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    set_prefill_ane_channel_split(false);
    let (_, reference) = pass(&mut fwd, &toks, n_layer);
    // One healthy split pass sizes the pass in evals.
    set_prefill_ane_channel_split(true);
    let e0 = zc_eval_ordinal();
    pass(&mut fwd, &toks, n_layer);
    let evals = zc_eval_ordinal() - e0;
    assert!(evals > 1, "B1: the split pass ran {evals} ANE evals");
    assert!(
        zc_lane_fail_count() == 0,
        "B1: lane failed before the fault"
    );

    let at = zc_eval_ordinal() + evals / 2;
    set_zc_fault_at_eval(Some(at));
    let (_, faulted) = pass(&mut fwd, &toks, n_layer);
    set_zc_fault_at_eval(None);
    set_prefill_ane_channel_split(false);
    println!(
        "  B1: fault at eval {at} ({} of {evals} in the pass) · lane fails {} · logits bit-identical to GPU-only: {}",
        evals / 2,
        zc_lane_fail_count(),
        faulted == reference
    );
    assert_eq!(
        zc_lane_fail_count(),
        1,
        "B1: the injected fault did not fire"
    );
    assert!(
        faulted
            .iter()
            .map(|f| f.to_bits())
            .eq(reference.iter().map(|f| f.to_bits())),
        "B1: the faulted pass is not the GPU-only result — the chunk was not re-run"
    );
}
