//! riir-infer Issue 032 T4/T5 — G1 for the Metal CubeCL prefill of a
//! Hadamard-folded (Bonsai-2) model.
//!
//! The reference arm is the folded DECODE eager path fed one token at a time
//! (`set_input_token` + `forward_token`): it is itself G1-gated against the
//! CPU forward (riir-ai `g1_bonsai2_metal_parity.rs`, Plan 602 B4 — top-1
//! 11751, top-5 1.00, top-20 1.00, worst rel err 6.7e-6), so a prefill that
//! agrees with it agrees with the CPU reference through one hop, at a
//! fraction of the CPU forward's cost. Prefill runs batched ternary GEMMs
//! whose reduction order differs from decode's GEMVs, so the bar is the
//! parity-gate class, not bit-identity:
//!
//! - top-1 identical (hard), top-5 overlap 1.00, top-20 overlap >= 0.80,
//!   worst relative error <= 2% over the shared top-20 probabilities;
//! - the SAME bar one step later — prefill, then one `forward_token` on the
//!   prefill's argmax, against the decode arm's next step — which is what
//!   asserts the conv / recurrence / KV state handoff, not just the logits.
//!
//! P covers the chunked-conv1d path (multiples of 64) and the sequential
//! conv1d fallback (`P % 64 != 0`). ⚠ The default list stops at 512 for
//! cost; **`I032_P=4096` is the production chunk-max arm** (~12 min, decode
//! reference included) and the one that matters for grid limits — the first
//! landing was green to 2048 while P=4096 crashed on Metal's 65535
//! workgroups-per-dimension cap (fixed `5f075ff` + `7e99f34`; 4096 G1 green
//! after, worst rel err 1.23e-5).
//!
//! T5 (no regression on the pre-rotation file) rides the same binary:
//! `prerotation_prefill_pin_unchanged` reproduces the `Q2_0` prefill pin
//! `fnv 99a0733c45a0e663` @2048 bit-exactly — every folded branch is keyed on
//! `rot_tables`, so the old path must be byte-for-byte untouched.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i032 cargo test -p riir-infer-gpu --release \
//!   --features cubecl_runtime,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,ternary_gemm_simdgroup,deltanet_recurrence_rowpar,deltanet_recurrence_parallel,riir-infer-core/bonsai2_hadamard \
//!   --test g1_032_folded_prefill_metal -- --ignored --nocapture --test-threads=1
//! ```
//! Env: `BONSAI2_G1_MODEL` / `BONSAI1_G3_MODEL` override the sibling-layout
//! defaults; `I032_P` (comma list) overrides the prompt lengths.

#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::ternary_deltanet_gpu_forward::{
    set_prefill_attention_batched, set_prefill_batch_elementwise, set_prefill_chunked,
    set_prefill_chunked_conv1d,
};
use riir_infer_gpu::TernaryDeltanetGpuForward;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const PREROT_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-27B-Q2_0.gguf";
/// The pre-rotation prefill pin at P=2048 (Issue 1004 G1 anchor class;
/// reproduced by riir-ai `bench_1004_r1_staged_recurrence_e2e`, 2026-10-01).
const PREROT_PIN_P2048: u64 = 0x99a0_733c_45a0_e663;

const TOP5_MIN: f32 = 1.0;
const TOP20_MIN: f32 = 0.80;
const REL_ERR_MAX: f32 = 0.02;

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.select_nth_unstable_by(k.min(v.len() - 1), |&x, &y| v[y].total_cmp(&v[x]));
    idx.truncate(k);
    idx.sort_by(|&x, &y| v[y].total_cmp(&v[x]));
    idx
}

fn overlap(a: &[f32], b: &[f32], k: usize) -> f32 {
    let (ta, tb) = (top_k(a, k), top_k(b, k));
    ta.iter().filter(|i| tb.contains(i)).count() as f32 / k as f32
}

/// Worst relative error between the two arms' probabilities over the shared
/// top-k (each arm normalised over its own top-k; the decode-G1 metric).
fn worst_rel_err(a: &[f32], b: &[f32], k: usize) -> f32 {
    let probs = |v: &[f32]| -> Vec<(usize, f32)> {
        let idx = top_k(v, k);
        let max = v[idx[0]];
        let e: Vec<f32> = idx.iter().map(|&i| (v[i] - max).exp()).collect();
        let s: f32 = e.iter().sum();
        idx.into_iter().zip(e).map(|(i, x)| (i, x / s)).collect()
    };
    let (pa, pb) = (probs(a), probs(b));
    pa.iter()
        .filter_map(|(i, va)| pb.iter().find(|(j, _)| j == i).map(|(_, vb)| (va - vb).abs() / va.abs().max(1e-9)))
        .fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> usize {
    top_k(v, 1)[0]
}

fn logits_fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

fn prompt(p: usize, bos: usize, vocab: usize) -> Vec<usize> {
    (0..p).map(|i| if i == 0 { bos } else { (bos + i * 7919) % vocab }).collect()
}

/// The shipping prefill knob state (riir-ai `bench_1004_r1_staged_recurrence_e2e`).
fn shipping_knobs() {
    set_prefill_batch_elementwise(true);
    set_prefill_chunked(true);
    set_prefill_chunked_conv1d(true);
    set_prefill_attention_batched(true);
}

fn gate(label: &str, prefill: &[f32], decode: &[f32]) {
    let (t5, t20, rel) = (overlap(prefill, decode, 5), overlap(prefill, decode, 20), worst_rel_err(prefill, decode, 20));
    let (ap, ad) = (argmax(prefill), argmax(decode));
    println!("{label}: argmax prefill {ap} / decode {ad} · top-5 {t5:.2} · top-20 {t20:.2} · worst rel err {rel:.3e}");
    assert_eq!(ap, ad, "{label}: top-1 differs (prefill {ap}, decode {ad})");
    assert!(t5 >= TOP5_MIN, "{label}: top-5 overlap {t5} < {TOP5_MIN}");
    assert!(t20 >= TOP20_MIN, "{label}: top-20 overlap {t20} < {TOP20_MIN}");
    assert!(rel <= REL_ERR_MAX, "{label}: worst rel err {rel} > {REL_ERR_MAX}");
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn folded_prefill_matches_folded_decode() {
    let ps: Vec<usize> = std::env::var("I032_P")
        .unwrap_or_else(|_| "64,100,128,512".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let model = env_path("BONSAI2_G1_MODEL", FOLDED_MODEL);
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    assert!(weights.rotation.is_some(), "{}: not a Hadamard-folded file", model.display());
    config.block_size = ps.iter().copied().max().unwrap_or(512) + 1;
    shipping_knobs();

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    assert!(fwd.is_folded(), "folded constructor did not build rotation tables");

    for &p in &ps {
        let tokens = prompt(p, config.bos_token, config.vocab_size);

        fwd.reset_state();
        let pre = fwd.prefill(&tokens);
        let next = argmax(&pre);
        fwd.set_input_token(&weights, next);
        let pre_step = fwd.forward_token();

        fwd.reset_state();
        let mut dec = Vec::new();
        for &t in &tokens {
            fwd.set_input_token(&weights, t);
            dec = fwd.forward_token();
        }
        fwd.set_input_token(&weights, next);
        let dec_step = fwd.forward_token();

        gate(&format!("P={p:>4} last position"), &pre, &dec);
        gate(&format!("P={p:>4} +1 decode step"), &pre_step, &dec_step);
    }
}

#[test]
#[ignore = "requires the Metal GPU + the pre-rotation Q2_0 model; run explicitly with --release"]
fn prerotation_prefill_pin_unchanged() {
    let model = env_path("BONSAI1_G3_MODEL", PREROT_MODEL);
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("pre-rotation model load");
    assert!(weights.rotation.is_none(), "{}: expected a pre-rotation file", model.display());
    config.block_size = 2048;
    shipping_knobs();

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    assert!(!fwd.is_folded());
    fwd.reset_state();
    let logits = fwd.prefill(&prompt(2048, config.bos_token, config.vocab_size));
    let fnv = logits_fnv(&logits);
    println!("pre-rotation P=2048: argmax {} · fnv {fnv:016x}", argmax(&logits));
    assert_eq!(fnv, PREROT_PIN_P2048, "pre-rotation prefill logits moved (pin {PREROT_PIN_P2048:016x})");
}

fn sh(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().replace('\n', " "))
        .unwrap_or_else(|_| "unavailable".into())
}

/// T6 — the first Bonsai-2 M3 batched-prefill throughput reading. A
/// MEASUREMENT, not a gate and never a league cell (Plan 602 C4 owns those):
/// median of `I032_ROUNDS` (default 3) timed prefills after one warm-up, with
/// the PROVENANCE line beside it. The reference rate is what a folded prompt
/// cost before this issue — token-by-token decode — timed on the same prompt.
#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn folded_prefill_throughput() {
    let p: usize = std::env::var("I032_TP_P").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let rounds: usize = std::env::var("I032_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let model = env_path("BONSAI2_G1_MODEL", FOLDED_MODEL);
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    assert!(weights.rotation.is_some(), "{}: not a Hadamard-folded file", model.display());
    config.block_size = p;
    shipping_knobs();
    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    let tokens = prompt(p, config.bos_token, config.vocab_size);
    println!(
        "PROVENANCE: model {} power [{}] powermode [{}] loadavg [{}] P={p} rounds={rounds}",
        model.display(),
        sh("pmset", &["-g", "batt"]),
        sh("sh", &["-c", "pmset -g | grep -i powermode"]),
        sh("sysctl", &["-n", "vm.loadavg"]),
    );

    fwd.reset_state();
    let warm = fwd.prefill(&tokens);
    let mut secs = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        fwd.reset_state();
        let t = std::time::Instant::now();
        let logits = fwd.prefill(&tokens);
        secs.push(t.elapsed().as_secs_f64());
        assert_eq!(logits_fnv(&logits), logits_fnv(&warm), "folded prefill is not deterministic");
    }
    secs.sort_by(f64::total_cmp);
    let prefill_tps = p as f64 / secs[secs.len() / 2];

    // The pre-Issue-032 cost of the same prompt: token-by-token decode (a
    // 256-token sample — the rate, not the full prompt).
    let sample = 256.min(p);
    fwd.reset_state();
    let t = std::time::Instant::now();
    for &tok in &tokens[..sample] {
        fwd.set_input_token(&weights, tok);
        let _ = fwd.forward_token();
    }
    let decode_tps = sample as f64 / t.elapsed().as_secs_f64();
    println!(
        "folded P={p}: batched prefill {prefill_tps:.1} tok/s (median of {rounds}; min {:.2}s max {:.2}s) · token-by-token {decode_tps:.1} tok/s ({sample}-token sample) · {:.2}x · argmax {} · fnv {:016x}",
        secs[0],
        secs[secs.len() - 1],
        prefill_tps / decode_tps,
        argmax(&warm),
        logits_fnv(&warm),
    );
}
