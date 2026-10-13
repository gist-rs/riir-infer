//! katgpt-rs Issue 936 T3 — the decode spin pool (`deltanet::set_decode_spin_pool`):
//! real Bonsai-2-27B PQ2_0, ONE load, two decode states over the same weights,
//! the switch flipped per position (first mover rotated). With the switch off
//! the forward takes the rayon path (Bench 880's `on_rayon_worker`) and the
//! pool's workers stay parked.
//!
//! - **G1**: every logit of every teacher-forced position is bit-identical
//!   between the arms (fixed row partitions; scheduling only).
//! - **G2**: median per-position (pool / rayon) wall ratio ≤ [`MAX_TIME_RATIO`].
//!
//! ```bash
//! cargo test --release --features ternary_q2_resident,bonsai2_hadamard,decode_spin_pool \
//!     --test decode_spin_pool_goat -- --ignored --nocapture --test-threads=1
//! ```
//! Env: `BONSAI2_GGUF`, `Q8_GATE_CORPUS`, `SPIN_GATE_POSITIONS` (default 64),
//! `KATGPT_SPIN_POOL_WORKERS` (pool size; default P-cores − 1 on macOS).
#![cfg(all(
    feature = "ternary_q2_resident",
    feature = "bonsai2_hadamard",
    feature = "decode_spin_pool"
))]

use std::time::Instant;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::set_decode_spin_pool;
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::deltanet::ternary_weights::{QwenDeltaNetTernaryWeights, set_ternary_q8_act};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;

const MAX_TIME_RATIO: f64 = 0.95;

/// One arm's decode state (the weights are shared).
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

    fn step(
        &mut self,
        pool: bool,
        w: &QwenDeltaNetTernaryWeights,
        token: usize,
        pos: usize,
        config: &Config,
        rope: &RopeFreqTable,
    ) -> f64 {
        set_decode_spin_pool(pool);
        let t = Instant::now();
        forward_qwen_deltanet_ternary_with_hook(
            &mut self.x,
            w,
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

#[test]
#[ignore = "real 27B model — run with --ignored --release"]
fn decode_spin_pool_is_bit_identical_and_faster_on_bonsai2() {
    let gguf = std::env::var("BONSAI2_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".into());
    let corpus =
        std::env::var("Q8_GATE_CORPUS").unwrap_or_else(|_| ".raw/twt/audition_calib.txt".into());
    let n_pos: usize =
        std::env::var("SPIN_GATE_POSITIONS").map_or(64, |v| v.parse().expect("usize"));
    assert!(
        std::path::Path::new(&gguf).exists() && std::path::Path::new(&corpus).exists(),
        "UNSEEN, not a pass: model {gguf} or corpus {corpus} missing"
    );
    let text = load_corpus_text(std::path::Path::new(&corpus)).expect("corpus");
    let tok = BpeTokenizer::from_gguf(&GgufFile::open(std::path::Path::new(&gguf)).expect("gguf"))
        .expect("tokenizer");
    let ids = tok.encode(&text);
    assert!(ids.len() > n_pos, "corpus too short: {} tokens", ids.len());

    set_ternary_q8_act(true);
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(std::path::Path::new(&gguf)).expect("load");
    config.block_size = n_pos + 1;
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let (mut on, mut off) = (Arm::new(&config, &weights), Arm::new(&config, &weights));
    let vocab = config.vocab_size;

    let mut ratios = Vec::with_capacity(n_pos);
    let (mut wall_on, mut wall_off, mut mismatched) = (0.0f64, 0.0f64, 0usize);
    for (p, &id) in ids.iter().take(n_pos).enumerate() {
        let (t_on, t_off) = if p % 2 == 0 {
            let a = on.step(true, &weights, id, p, &config, &rope);
            (a, off.step(false, &weights, id, p, &config, &rope))
        } else {
            let b = off.step(false, &weights, id, p, &config, &rope);
            (on.step(true, &weights, id, p, &config, &rope), b)
        };
        wall_on += t_on;
        wall_off += t_off;
        ratios.push(t_on / t_off);
        let same = on.x[..vocab]
            .iter()
            .zip(&off.x[..vocab])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        mismatched += usize::from(!same);
    }
    set_decode_spin_pool(true);
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    eprintln!("[spin-gate] positions {n_pos} | logit-vector mismatches {mismatched} (bar 0)");
    eprintln!(
        "[spin-gate] wall rayon {:.1} ms/tok pool {:.1} ms/tok | median paired ratio {median:.3} (bar ≤ {MAX_TIME_RATIO})",
        wall_off / n_pos as f64 * 1e3,
        wall_on / n_pos as f64 * 1e3
    );
    assert_eq!(mismatched, 0, "G1: the spin pool changed logits");
    assert!(
        median <= MAX_TIME_RATIO,
        "G2: median ratio {median:.3} > {MAX_TIME_RATIO}"
    );
}
