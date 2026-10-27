//! Issue 1139 T1 / Plan 623 T2.2c(2) pricing probe — the PRODUCTION 1-box
//! decode funnel (`forward_token`, not the stage seam) on the real
//! Bonsai-2 27B, plus the ceiling an on-GPU argmax could recover: an idle
//! second `read_logits` (pure readback, no pending compute) and the host
//! `argmax_first` over the vocab.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i623_dist cargo test -p riir-infer-dist --release \
//!   --features gpu,bonsai2_hadamard --test decode_readback_probe_gpu -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `I1139_STEPS` (timed steps, default 64).

#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::time::Instant;

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_dist::argmax_first;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const WARMUP: usize = 8;

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn production_decode_and_readback_ceiling() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let steps: usize = std::env::var("I1139_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("load");
    config.block_size = WARMUP + steps;
    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    let mut tok = 9707usize;
    let (mut fwd_s, mut read_s, mut argmax_s) = (0f64, 0f64, 0f64);
    for step in 0..WARMUP + steps {
        fwd.set_input_token(&weights, tok);
        let t = Instant::now();
        let logits = fwd.forward_token();
        let d_fwd = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let again = fwd.read_logits();
        let d_read = t.elapsed().as_secs_f64();
        let t = Instant::now();
        tok = argmax_first(&logits) as usize;
        let d_argmax = t.elapsed().as_secs_f64();
        assert_eq!(again.len(), logits.len());
        if step >= WARMUP {
            fwd_s += d_fwd;
            read_s += d_read;
            argmax_s += d_argmax;
        }
    }
    let ms = 1e3 / steps as f64;
    println!(
        "per token: forward_token {:.3} ms ({:.2} tok/s) · idle readback {:.3} ms · host argmax {:.3} ms · ceiling {:.2}%",
        fwd_s * ms,
        steps as f64 / fwd_s,
        read_s * ms,
        argmax_s * ms,
        100.0 * (read_s + argmax_s) / fwd_s
    );
}

/// Issue 1140 T1 — is decode slower after a batched `prefill` than after a
/// per-token prefill? Fresh forwarder per arm (the pool state is the
/// suspect), arms ALTERNATING (A B B A …), same prompt, `I1140_STEPS`
/// timed decode steps each.
///
/// Env: `BONSAI2_G1_MODEL`, `I1140_PROMPT` (default 256), `I1140_STEPS`
/// (default 128), `I1140_REPS` (pairs, default 3).
#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn decode_after_batched_vs_per_token_prefill_interleaved() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (len, steps, reps) = (
        env("I1140_PROMPT", 256),
        env("I1140_STEPS", 128),
        env("I1140_REPS", 3),
    );
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("load");
    config.block_size = len + steps;
    riir_infer_gpu::ternary_deltanet_gpu_forward::set_prefill_chunk_max(len);
    let prompt: Vec<usize> = (0..len).map(|i| 1000 + (i * 7919) % 30_000).collect();
    let ctx = CubeCLContext::new().expect("GPU init");

    // One arm: fresh forwarder, prompt in, `steps` timed greedy decode steps.
    let arm = |batched: bool| -> f64 {
        let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
        let mut logits = Vec::new();
        match batched {
            true => logits = fwd.prefill(&prompt),
            false => {
                for &t in &prompt {
                    fwd.set_input_token(&weights, t);
                    logits = fwd.forward_token();
                }
            }
        }
        let mut tok = argmax_first(&logits) as usize;
        let t = Instant::now();
        for _ in 0..steps {
            fwd.set_input_token(&weights, tok);
            tok = argmax_first(&fwd.forward_token()) as usize;
        }
        steps as f64 / t.elapsed().as_secs_f64()
    };
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for r in 0..reps {
        // ABBA within each pair of reps: the order alternates.
        let first_batched = r % 2 == 1;
        for batched in [first_batched, !first_batched] {
            let tok_s = arm(batched);
            println!(
                "rep {r} {}: decode {tok_s:.2} tok/s",
                if batched {
                    "after-batched"
                } else {
                    "after-per-token"
                }
            );
            match batched {
                true => b.push(tok_s),
                false => a.push(tok_s),
            }
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (ma, mb) = (med(&mut a), med(&mut b));
    println!(
        "I1140 P={len} steps={steps} reps={reps}: median decode after per-token {ma:.2} · after batched {mb:.2} tok/s · Δ {:+.2}%",
        100.0 * (mb - ma) / ma
    );
}
