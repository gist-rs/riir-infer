//! Issue 1160 G0 — the early collapse probe for training-free soft-token feedback on Bonsai-2.
//!
//! Soft-token feedback feeds the next step `E^T s` with `s = softmax(top_k(o_t) / tau)`. When
//! the decode distribution is spiky, `s` sits on its argmax and `E^T s` is the argmax
//! embedding: the arm degenerates to the incumbent (LIFT's measured k=1 failure). The issue
//! pre-registers the kill: **median top-1 mass >= 0.9 => close as k=1 collapse, no sweep.**
//!
//! This decodes greedily from fixed prompts (prefill, then `set_input_token` +
//! `forward_token` per step, the eager decode protocol) and records at every decode step:
//!
//! - `p1` — top-1 mass of the full softmax (tau = 1): the pre-registered G0 statistic;
//! - `H` — entropy of that softmax, nats;
//! - `s1` — top-1 mass of the soft token the PILOT arm would actually feed
//!   (k = 128, tau = 1.5, renormalised over the top-k: what `E^T s` weights).
//!
//! Medians and quartiles per prompt family and pooled. A report, not a gate: G0's verdict
//! is read from the pooled median against the 0.9 bar and recorded in the issue.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1160 cargo test -p riir-infer-gpu --release \
//!   --features metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1160_g0_collapse_probe -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `I1160_STEPS` (decode steps per prompt, default 64).

#![cfg(all(
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "multi_host",
    feature = "ternary_gemm_batched",
    target_os = "macos"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{load_qwen_deltanet_ternary_weights_gguf, GgufFile};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::TernaryDeltanetGpuForward;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

/// The pilot arm's soft-token shape (Issue 1160 §Arms: k = 128, tau = 1.5).
const PILOT_K: usize = 128;
const PILOT_TAU: f64 = 1.5;
/// The pre-registered collapse bar on the pooled median top-1 mass.
const COLLAPSE_BAR: f64 = 0.9;

/// Fixed prompts, frozen: (family, prompt). The arithmetic family is the G1b headline class.
const PROMPTS: [(&str, &str); 8] = [
    (
        "arith",
        "Question: A shop sells pencils at 3 for 45 cents. How much do 12 pencils cost?\nAnswer: Let's think step by step.",
    ),
    (
        "arith",
        "Question: Tom has 17 apples. He gives 5 to Mary and buys twice as many as he has left. How many apples does Tom have now?\nAnswer: Let's think step by step.",
    ),
    (
        "arith",
        "Question: What is 48 multiplied by 25, minus 137?\nAnswer: Let's think step by step.",
    ),
    (
        "arith",
        "Question: A train travels 210 kilometres in 3 hours. At the same speed, how far does it travel in 5 hours?\nAnswer: Let's think step by step.",
    ),
    (
        "code",
        "// Return the n-th Fibonacci number iteratively.\npub fn fib(n: u64) -> u64 {\n",
    ),
    (
        "prose",
        "The lighthouse keeper had kept the same routine for eleven years. Each evening, an hour before sunset, he",
    ),
    (
        "prose",
        "Write a short paragraph explaining why the sky is blue to a ten-year-old.\n\n",
    ),
    (
        "zh",
        "春天的早晨，小镇上的市场总是最热闹的。天刚亮，",
    ),
];

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One decode step's statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
    p1: f64,
    entropy: f64,
    s1: f64,
}

/// The step statistics of one logits vector (f64, max-shifted, sequential folds — a
/// measurement, so the reduction order is pinned).
fn step_stats(logits: &[f32], k: usize, tau: f64) -> (Step, usize) {
    let (arg, m) =
        logits
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &x)| {
                match (x as f64) > bv {
                    true => (i, x as f64),
                    false => (bi, bv),
                }
            });
    let z: f64 = logits.iter().map(|&x| ((x as f64) - m).exp()).sum();
    let entropy = logits
        .iter()
        .map(|&x| {
            let lp = (x as f64) - m - z.ln();
            -lp.exp() * lp
        })
        .sum::<f64>();
    let mut top: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let k = k.min(top.len());
    top.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let zk: f64 = top[..k].iter().map(|&x| ((x - m) / tau).exp()).sum();
    (
        Step {
            p1: 1.0 / z,
            entropy,
            // The max logit's own term is exp(0) = 1.
            s1: 1.0 / zk,
        },
        arg,
    )
}

/// The `q`-quantile (nearest-rank, q in [0, 1]) of `v`, which it sorts.
fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
    v[idx]
}

fn row(label: &str, steps: &[Step]) -> f64 {
    let mut p1: Vec<f64> = steps.iter().map(|s| s.p1).collect();
    let mut h: Vec<f64> = steps.iter().map(|s| s.entropy).collect();
    let mut s1: Vec<f64> = steps.iter().map(|s| s.s1).collect();
    let spiky = p1.iter().filter(|&&p| p >= COLLAPSE_BAR).count();
    let med = quantile(&mut p1, 0.5);
    println!(
        "{label:<7} n={:>4} · p1 q25 {:.3} med {:.3} q75 {:.3} · >=0.9 {:>5.1}% · H med {:.3} · s1(k={PILOT_K},tau={PILOT_TAU}) med {:.3}",
        steps.len(),
        quantile(&mut p1, 0.25),
        med,
        quantile(&mut p1, 0.75),
        100.0 * spiky as f64 / steps.len() as f64,
        quantile(&mut h, 0.5),
        quantile(&mut s1, 0.5),
    );
    med
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn g0_decode_top1_mass_census() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let steps = env_usize("I1160_STEPS", 64);
    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    config.block_size = 256 + steps;
    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    println!(
        "# issue1160 G0: greedy decode, {steps} steps/prompt, bar median p1 >= {COLLAPSE_BAR}"
    );
    let mut pooled: Vec<Step> = Vec::new();
    let mut by_family: Vec<(&str, Vec<Step>)> = Vec::new();
    for (i, (family, prompt)) in PROMPTS.iter().enumerate() {
        let toks = tok.encode(prompt);
        fwd.reset_state();
        let mut logits = fwd.prefill(&toks);
        let mut out = Vec::with_capacity(steps);
        let mut seen: Vec<Step> = Vec::with_capacity(steps);
        for _ in 0..steps {
            assert!(
                logits.iter().all(|x| x.is_finite()),
                "prompt {i}: non-finite logits"
            );
            let (s, next) = step_stats(&logits, PILOT_K, PILOT_TAU);
            seen.push(s);
            out.push(next);
            fwd.set_input_token(&weights, next);
            logits = fwd.forward_token();
        }
        let text = tok.decode(&out);
        println!(
            "## [{i}] {family}: {:?}",
            text.chars().take(160).collect::<String>()
        );
        row(&format!("  [{i}]"), &seen);
        pooled.extend_from_slice(&seen);
        match by_family.iter_mut().find(|(f, _)| f == family) {
            Some((_, v)) => v.extend(seen),
            None => by_family.push((family, seen)),
        }
    }
    println!("# per family");
    for (family, v) in &by_family {
        row(family, v);
    }
    println!("# pooled");
    let med = row("ALL", &pooled);
    println!(
        "# G0 verdict: median p1 {med:.3} {} {COLLAPSE_BAR} => {}",
        if med >= COLLAPSE_BAR { ">=" } else { "<" },
        if med >= COLLAPSE_BAR {
            "COLLAPSE — close without the sweep"
        } else {
            "PASS — the pilot arm is armed"
        }
    );
}

#[test]
fn step_stats_reads_a_spike_and_a_flat_distribution() {
    let (spike, arg) = step_stats(&[0.0, 20.0, 0.0, 0.0], 2, 1.0);
    assert_eq!(arg, 1);
    assert!(spike.p1 > 0.999 && spike.s1 > 0.999 && spike.entropy < 1e-6);
    let (flat, _) = step_stats(&[1.0; 4], 4, 1.5);
    assert!((flat.p1 - 0.25).abs() < 1e-12);
    assert!((flat.s1 - 0.25).abs() < 1e-12);
    assert!((flat.entropy - 4f64.ln()).abs() < 1e-12);
    // A temperature flattens the soft token below the plain top-1 mass.
    let (t, _) = step_stats(&[3.0, 2.0, 1.0, 0.0], 4, 1.5);
    assert!(t.s1 < t.p1);
}

#[test]
fn quantile_is_nearest_rank() {
    let mut v = vec![3.0, 1.0, 2.0, 4.0, 5.0];
    assert_eq!(quantile(&mut v, 0.5), 3.0);
    assert_eq!(quantile(&mut v, 0.0), 1.0);
    assert_eq!(quantile(&mut v, 1.0), 5.0);
}
