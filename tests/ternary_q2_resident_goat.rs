//! Raw-`Q2_0` resident projections GOAT gate (katgpt-rs Issue 935 T5) — the
//! real Bonsai-2-27B PQ2_0 decode, bit-plane q8 vs raw-Q2 q8, ONE process,
//! TWO loads (the representation is chosen at load: `set_ternary_q2_resident`).
//!
//! The raw-Q2 kernel computes the same exact integer sums in the same f32
//! accumulation order as the bit-plane q8 kernel, so the claim is **bit
//! identity**, not a quality band:
//!
//! - **G1**: every logit of every teacher-forced position is bit-identical
//!   (full-vocab compare), and a greedy free decode of [`FREE_DECODE_STEPS`]
//!   tokens is identical.
//! - **G2**: median per-position (q2 / planes) wall ratio ≤ [`MAX_TIME_RATIO`]
//!   (`--release`; arms alternated per position, first-mover rotated).
//! - **Admissibility**: the two loads must actually differ in representation
//!   (a switch that silently did nothing would compare planes with planes).
//!
//! ```bash
//! cargo test --release --features ternary_q2_resident,bonsai2_hadamard \
//!     --test ternary_q2_resident_goat -- --ignored --nocapture --test-threads=1
//! ```
//! Env: `BONSAI2_GGUF` (default `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`),
//! `Q8_GATE_CORPUS` (default `.raw/twt/audition_calib.txt`), `Q2_GATE_POSITIONS`
//! (default 64).
#![cfg(all(feature = "ternary_q2_resident", feature = "bonsai2_hadamard"))]

use std::time::Instant;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::deltanet::ternary_weights::{
    ProjWeights, QwenDeltaNetTernaryWeights, set_ternary_q2_resident, set_ternary_q8_act,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;

const MAX_TIME_RATIO: f64 = 0.95;
const FREE_DECODE_STEPS: usize = 32;

/// One arm: its own weights (representation fixed at load) and decode state.
struct Arm {
    weights: QwenDeltaNetTernaryWeights,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    x: Vec<f32>,
}

impl Arm {
    fn load(gguf: &str, q2: bool, block_size: usize) -> (Config, Self) {
        set_ternary_q2_resident(q2);
        let t0 = Instant::now();
        let (mut config, weights) =
            load_qwen_deltanet_ternary_weights_gguf(std::path::Path::new(gguf)).expect("load");
        eprintln!("[q2-gate] load (q2={q2}) {:.1}s", t0.elapsed().as_secs_f64());
        config.block_size = block_size;
        let arm = Self {
            cache: HybridCache::with_layer_types(&config, &weights.layer_types),
            scratch: HybridForwardScratch::new(&config),
            x: vec![0.0f32; config.vocab_size.max(config.n_embd)],
            weights,
        };
        (config, arm)
    }

    /// Count of per-layer projections held as raw Q2 blocks.
    fn q2_projections(&self) -> usize {
        self.weights
            .layers
            .iter()
            .flat_map(|l| l.projections())
            .filter(|p| matches!(p, ProjWeights::TernaryQ2(_)))
            .count()
    }

    fn step(&mut self, token: usize, pos: usize, config: &Config, rope: &RopeFreqTable) -> f64 {
        let t = Instant::now();
        forward_qwen_deltanet_ternary_with_hook(
            &mut self.x,
            &self.weights,
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

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits")
        .0
}

#[allow(clippy::needless_range_loop)] // `p` is the position AND the index
#[test]
#[ignore = "real 27B model, two loads, ~6 min — run with --ignored --release"]
fn q2_resident_is_bit_identical_and_faster_on_bonsai2() {
    let gguf = std::env::var("BONSAI2_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".into());
    let corpus =
        std::env::var("Q8_GATE_CORPUS").unwrap_or_else(|_| ".raw/twt/audition_calib.txt".into());
    let n_pos: usize = std::env::var("Q2_GATE_POSITIONS").map_or(64, |v| v.parse().expect("usize"));
    assert!(
        std::path::Path::new(&gguf).exists() && std::path::Path::new(&corpus).exists(),
        "UNSEEN, not a pass: model {gguf} or corpus {corpus} missing"
    );
    let text = load_corpus_text(std::path::Path::new(&corpus)).expect("corpus");
    let tok = BpeTokenizer::from_gguf(&GgufFile::open(std::path::Path::new(&gguf)).expect("gguf"))
        .expect("tokenizer");
    let ids = tok.encode(&text);
    assert!(ids.len() > n_pos + 1, "corpus too short: {} tokens", ids.len());

    set_ternary_q8_act(true);
    let block = n_pos + FREE_DECODE_STEPS + 1;
    let (config, mut planes) = Arm::load(&gguf, false, block);
    let (_, mut q2) = Arm::load(&gguf, true, block);
    set_ternary_q2_resident(true);
    let n_proj = q2.weights.layers.iter().flat_map(|l| l.projections()).filter(|p| p.rows() > 0).count();
    assert_eq!(planes.q2_projections(), 0, "planes arm must hold no Q2 projections");
    assert!(q2.q2_projections() > 0, "q2 arm holds no Q2 projections — the switch did nothing");
    eprintln!("[q2-gate] q2 arm: {}/{n_proj} projections raw-Q2", q2.q2_projections());

    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let vocab = config.vocab_size;
    let mut ratios = Vec::with_capacity(n_pos);
    let (mut wall_p, mut wall_q, mut mismatched) = (0.0f64, 0.0f64, 0usize);
    for p in 0..n_pos {
        let (tp, tq) = if p % 2 == 0 {
            let tp = planes.step(ids[p], p, &config, &rope);
            (tp, q2.step(ids[p], p, &config, &rope))
        } else {
            let tq = q2.step(ids[p], p, &config, &rope);
            (planes.step(ids[p], p, &config, &rope), tq)
        };
        wall_p += tp;
        wall_q += tq;
        ratios.push(tq / tp);
        let same = planes.x[..vocab]
            .iter()
            .zip(&q2.x[..vocab])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        mismatched += usize::from(!same);
    }
    ratios.sort_by(f64::total_cmp);
    let ratio = ratios[ratios.len() / 2];

    let mut greedy = [Vec::new(), Vec::new()];
    for (k, arm) in [&mut planes, &mut q2].into_iter().enumerate() {
        let mut cur = ids[n_pos];
        for s in 0..FREE_DECODE_STEPS {
            arm.step(cur, n_pos + s, &config, &rope);
            cur = argmax(&arm.x[..vocab]);
            greedy[k].push(cur);
        }
    }
    let div = greedy[0].iter().zip(&greedy[1]).position(|(a, b)| a != b);

    let n = n_pos as f64;
    eprintln!("[q2-gate] positions {n_pos} | logit-vector mismatches {mismatched} (bar 0)");
    eprintln!(
        "[q2-gate] wall planes {:.1} ms/tok q2 {:.1} ms/tok | median paired ratio {ratio:.3} (bar ≤ {MAX_TIME_RATIO})",
        wall_p * 1e3 / n,
        wall_q * 1e3 / n
    );
    eprintln!("[q2-gate] greedy {FREE_DECODE_STEPS} steps: first divergence {div:?} (bar None)");

    assert_eq!(mismatched, 0, "G1: {mismatched}/{n_pos} positions not bit-identical");
    assert_eq!(div, None, "G1: greedy decode diverged at {div:?}");
    assert!(ratio <= MAX_TIME_RATIO, "G2 median ratio {ratio:.3} > {MAX_TIME_RATIO}");
}
