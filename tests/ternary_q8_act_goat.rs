//! Int8-activation ternary matvec GOAT gate (katgpt-rs Issue 935) — the real
//! Bonsai-2-27B PQ2_0 decode, f32 vs q8 activations, ONE process, ONE load.
//!
//! The q8 arm is LOSSY (activation rounding — llama.cpp's CPU Q2_0 regime), so
//! the gate is deployed-path behaviour, not bit identity. Two arms, two caches,
//! the SAME teacher-forced corpus head, alternated per position (so load drift
//! hits both arms alike) via the runtime switch `set_ternary_q8_act`:
//!
//! - **G1 quality** (per position, q8 vs the f32 reference):
//!   - top-1 agreement ≥ [`MIN_TOP1_AGREE`] overall, and ≥
//!     [`MIN_CONFIDENT_AGREE`] CONDITIONAL on the f32 arm being confident
//!     (p_top1 ≥ 0.5 — the positions where a flip changes the output);
//!   - mean truth-token NLL delta ≤ [`MAX_NLL_DELTA`] nats;
//!   - teacher-forced hit-rate delta ≥ −[`MAX_HIT_DROP`].
//! - **G2 perf**: median per-position (q8 / f32) wall ratio ≤ [`MAX_TIME_RATIO`]
//!   (`--release`; the per-position pairing is the interleaving).
//! - Greedy free decode (32 steps per arm, sequential) is RECORDED, never
//!   scored: the first divergence position is printed beside the rates.
//!
//! ```bash
//! cargo test --release --features ternary_q8_act,bonsai2_hadamard \
//!     --test ternary_q8_act_goat -- --ignored --nocapture --test-threads=1
//! ```
//! Env: `BONSAI2_GGUF` (default `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`),
//! `Q8_GATE_CORPUS` (default `.raw/twt/audition_calib.txt`), `Q8_GATE_POSITIONS`
//! (default 256).
#![cfg(all(feature = "ternary_q8_act", feature = "bonsai2_hadamard"))]

use std::time::Instant;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::deltanet::ternary_weights::{QwenDeltaNetTernaryWeights, set_ternary_q8_act};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;

const MIN_TOP1_AGREE: f64 = 0.95;
const MIN_CONFIDENT_AGREE: f64 = 0.99;
const MAX_NLL_DELTA: f64 = 0.02;
const MAX_HIT_DROP: f64 = 0.01;
const MAX_TIME_RATIO: f64 = 0.80;
const FREE_DECODE_STEPS: usize = 32;

/// One arm's decode state.
struct Arm {
    cache: HybridCache,
    scratch: HybridForwardScratch,
    x: Vec<f32>,
}

impl Arm {
    fn new(config: &Config, weights: &QwenDeltaNetTernaryWeights) -> Self {
        Self {
            cache: HybridCache::with_layer_types(config, &weights.layer_types),
            scratch: HybridForwardScratch::new(config),
            x: vec![0.0f32; config.vocab_size.max(config.n_embd)],
        }
    }

    /// One forward on the selected arm; returns the wall seconds.
    fn step(
        &mut self,
        q8: bool,
        weights: &QwenDeltaNetTernaryWeights,
        token: usize,
        pos: usize,
        config: &Config,
        rope: &RopeFreqTable,
    ) -> f64 {
        set_ternary_q8_act(q8);
        let t = Instant::now();
        forward_qwen_deltanet_ternary_with_hook(
            &mut self.x,
            weights,
            &mut self.cache,
            token,
            pos,
            config,
            &mut self.scratch,
            rope,
            None,
            None,
            None,
            None,
        );
        t.elapsed().as_secs_f64()
    }
}

/// (argmax, p_argmax, −log p(truth)) from raw logits — numerically stable.
fn read_logits(logits: &[f32], truth: usize) -> (usize, f64, f64) {
    let (am, &mx) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits");
    let lse = mx as f64
        + logits
            .iter()
            .map(|&l| ((l - mx) as f64).exp())
            .sum::<f64>()
            .ln();
    let p_top = (mx as f64 - lse).exp();
    (am, p_top, lse - logits[truth] as f64)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "real 27B model, ~10 min — run with --ignored --release"]
fn q8_act_holds_quality_and_wins_wall_on_bonsai2() {
    let gguf = std::env::var("BONSAI2_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".into());
    let corpus =
        std::env::var("Q8_GATE_CORPUS").unwrap_or_else(|_| ".raw/twt/audition_calib.txt".into());
    let n_pos: usize =
        std::env::var("Q8_GATE_POSITIONS").map_or(256, |v| v.parse().expect("usize"));
    assert!(
        std::path::Path::new(&gguf).exists() && std::path::Path::new(&corpus).exists(),
        "UNSEEN, not a pass: model {gguf} or corpus {corpus} missing"
    );

    let text = load_corpus_text(std::path::Path::new(&corpus)).expect("corpus");
    let tok = BpeTokenizer::from_gguf(&GgufFile::open(std::path::Path::new(&gguf)).expect("gguf"))
        .expect("tokenizer");
    let ids = tok.encode(&text);
    assert!(
        ids.len() > n_pos + 1,
        "corpus too short: {} tokens",
        ids.len()
    );

    let t0 = Instant::now();
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(std::path::Path::new(&gguf)).expect("load");
    eprintln!("[q8-gate] load {:.1}s", t0.elapsed().as_secs_f64());
    config.block_size = n_pos + FREE_DECODE_STEPS + 1;
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let vocab = config.vocab_size;
    let (mut f32_arm, mut q8_arm) = (Arm::new(&config, &weights), Arm::new(&config, &weights));

    // ── G1 + G2: teacher-forced, arms alternated per position ──
    let (mut agree, mut conf_n, mut conf_agree) = (0usize, 0usize, 0usize);
    let (mut hit_f, mut hit_q, mut nll_f, mut nll_q) = (0usize, 0usize, 0.0f64, 0.0f64);
    let mut ratios = Vec::with_capacity(n_pos);
    let (mut wall_f, mut wall_q) = (0.0f64, 0.0f64);
    for p in 0..n_pos {
        let truth = ids[p + 1];
        // Rotate which arm runs first so neither always gets the warmer cache.
        let (tf, tq) = if p % 2 == 0 {
            let tf = f32_arm.step(false, &weights, ids[p], p, &config, &rope);
            (tf, q8_arm.step(true, &weights, ids[p], p, &config, &rope))
        } else {
            let tq = q8_arm.step(true, &weights, ids[p], p, &config, &rope);
            (f32_arm.step(false, &weights, ids[p], p, &config, &rope), tq)
        };
        wall_f += tf;
        wall_q += tq;
        ratios.push(tq / tf);
        let (am_f, p_f, l_f) = read_logits(&f32_arm.x[..vocab], truth);
        let (am_q, _, l_q) = read_logits(&q8_arm.x[..vocab], truth);
        agree += usize::from(am_f == am_q);
        if p_f >= 0.5 {
            conf_n += 1;
            conf_agree += usize::from(am_f == am_q);
        }
        hit_f += usize::from(am_f == truth);
        hit_q += usize::from(am_q == truth);
        nll_f += l_f;
        nll_q += l_q;
    }
    let n = n_pos as f64;
    let top1 = agree as f64 / n;
    let conf = conf_agree as f64 / conf_n.max(1) as f64;
    let (hr_f, hr_q) = (hit_f as f64 / n, hit_q as f64 / n);
    let (mnll_f, mnll_q) = (nll_f / n, nll_q / n);
    let ratio = median(&mut ratios);

    // ── recorded, never scored: greedy free decode per arm ──
    let mut greedy = [Vec::new(), Vec::new()];
    for (k, (arm, q8)) in [(&mut f32_arm, false), (&mut q8_arm, true)]
        .into_iter()
        .enumerate()
    {
        let mut cur = ids[n_pos];
        for s in 0..FREE_DECODE_STEPS {
            arm.step(q8, &weights, cur, n_pos + s, &config, &rope);
            cur = read_logits(&arm.x[..vocab], 0).0;
            greedy[k].push(cur);
        }
    }
    let div = greedy[0].iter().zip(&greedy[1]).position(|(a, b)| a != b);

    eprintln!(
        "[q8-gate] positions {n_pos} | loadavg {:?}",
        sysctl_loadavg()
    );
    eprintln!(
        "[q8-gate] top-1 agree {top1:.4} (bar ≥ {MIN_TOP1_AGREE}) | confident {conf:.4} over {conf_n} (bar ≥ {MIN_CONFIDENT_AGREE})"
    );
    eprintln!(
        "[q8-gate] hit f32 {hr_f:.4} q8 {hr_q:.4} | mean NLL f32 {mnll_f:.4} q8 {mnll_q:.4} (Δ {:+.4}, bar ≤ {MAX_NLL_DELTA})",
        mnll_q - mnll_f
    );
    eprintln!(
        "[q8-gate] wall f32 {:.1} ms/tok q8 {:.1} ms/tok | median paired ratio {ratio:.3} (bar ≤ {MAX_TIME_RATIO})",
        wall_f * 1e3 / n,
        wall_q * 1e3 / n
    );
    eprintln!(
        "[q8-gate] free decode {FREE_DECODE_STEPS} steps: first divergence {div:?} (recorded, not scored)"
    );

    assert!(
        top1 >= MIN_TOP1_AGREE,
        "G1 top-1 agreement {top1:.4} < {MIN_TOP1_AGREE}"
    );
    assert!(
        conf_n > 0,
        "G1 conditional bar has no confident positions — UNSEEN"
    );
    assert!(
        conf >= MIN_CONFIDENT_AGREE,
        "G1 confident agreement {conf:.4} < {MIN_CONFIDENT_AGREE}"
    );
    assert!(
        mnll_q - mnll_f <= MAX_NLL_DELTA,
        "G1 NLL delta {:+.4} > {MAX_NLL_DELTA}",
        mnll_q - mnll_f
    );
    assert!(
        hr_q >= hr_f - MAX_HIT_DROP,
        "G1 hit-rate drop {:.4} > {MAX_HIT_DROP}",
        hr_f - hr_q
    );
    assert!(
        ratio <= MAX_TIME_RATIO,
        "G2 median ratio {ratio:.3} > {MAX_TIME_RATIO}"
    );
}

fn sysctl_loadavg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}
