//! Plan 623 T2.3b — chunked prompt frames through the GPU stage's batched
//! range prefill (`prefill_range_chunk`), on the real Bonsai-2 27B.
//!
//! 1. **Anchor:** the full-range stage's chunked prefill is BIT-IDENTICAL to
//!    the production 1-box `TernaryDeltanetGpuForward::prefill` at the same
//!    chunk width (`set_prefill_chunk_max`) — the range seam adds nothing.
//! 2. **G1:** the scoped 2-stage pair over loopback TCP, chunked, is
//!    bit-identical to that oracle (token stream, the last prompt
//!    position's logits, every decode logit) at each requested split.
//! 3. **G2 read:** prefill wall time, chunked vs per-position, 1-box and
//!    2-stage (same process, same stages' weights, interleaved).
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i623_dist cargo test -p riir-infer-dist --release \
//!   --features gpu,bonsai2_hadamard --test two_stage_prefill_gpu -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `I623_K` (splits, default `32`),
//! `I623_PROMPT` (prompt length, default 64), `I623_CHUNK` (default 16),
//! `I623_NEW` (tokens generated, default 8).

#![cfg(feature = "gpu")]

use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::argmax_first;
use riir_infer_dist::runner::{HeadOptions, HeadReport, prefill_width, run_head_with, run_tail};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, stage_scope};
use riir_infer_dist::stage_gpu::TernaryGpuStage;
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::ternary_deltanet_gpu_forward::set_prefill_chunk_max;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

type Weights = Arc<QwenDeltaNetTernaryWeights>;

fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load(model: &Path, ctx: usize, scope: impl FnOnce(&Config) -> WeightScope) -> (Config, Weights) {
    let (mut c, w) = load_qwen_deltanet_ternary_weights_gguf_scoped(model, scope).expect("load");
    c.block_size = ctx;
    (c, Arc::new(w))
}

/// Deterministic in-vocab prompt (ids spread over the common range).
fn prompt(len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| 1000 + (i * 7919) % 30_000)
        .collect()
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// 1-box oracle over a full-range stage: (tokens, [last-prompt logits,
/// decode logits...], prefill wall).
fn oracle(
    s: &mut dyn StageCompute,
    prompt: &[u32],
    max_new: usize,
    chunk: usize,
) -> (Vec<u32>, Vec<Vec<f32>>, Duration, f64) {
    let empty: [f32; 0] = [];
    let mut logits = Vec::new();
    let t = Instant::now();
    let mut last = Vec::new();
    match chunk {
        1 => {
            for (pos, &tok) in prompt.iter().enumerate() {
                last = s.step(tok, pos, &empty).to_vec();
            }
        }
        _ => {
            for lo in (0..prompt.len()).step_by(chunk) {
                let hi = (lo + chunk).min(prompt.len());
                let fin = hi == prompt.len();
                s.prefill_chunk(&prompt[lo..hi], &empty, lo, prompt.len(), fin, &mut last);
            }
        }
    }
    let prefill = t.elapsed();
    let mut next = argmax_first(&last);
    logits.push(last);
    let mut toks = vec![next];
    let t = Instant::now();
    for i in 1..max_new {
        let l = s.step(next, prompt.len() + i - 1, &empty).to_vec();
        next = argmax_first(&l);
        logits.push(l);
        toks.push(next);
    }
    let decode_tok_s = (max_new - 1) as f64 / t.elapsed().as_secs_f64();
    (toks, logits, prefill, decode_tok_s)
}

#[allow(clippy::too_many_arguments)]
fn pair(
    config: &Config,
    hw: &Weights,
    tw: &Weights,
    k: u32,
    identity: ArtifactIdentity,
    prompt: &[u32],
    max_new: usize,
    chunk: usize,
) -> (HeadReport, Vec<Vec<f32>>) {
    let n = config.n_layer as u32;
    let key = PeerKey([0x23; 32]);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw, tk) = (config.clone(), tw.clone(), key.clone());
    let tail = thread::spawn(move || {
        let mut stage = TernaryGpuStage::new(&tc, tw, LayerRange::new(k, n), identity).unwrap();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, tk, Direction::Upstream).unwrap();
        let mut seen = Vec::new();
        run_tail(&mut stage, &mut link, |_, l| seen.push(l.to_vec())).map(|_| seen)
    });
    let mut head =
        TernaryGpuStage::new(config, hw.clone(), LayerRange::new(0, k), identity).unwrap();
    let mut link = TcpTransport::new(
        TcpStream::connect(addr).unwrap(),
        key,
        Direction::Downstream,
    )
    .unwrap();
    let opts = HeadOptions {
        prefill_chunk: chunk,
        min_batched: 0,
        ..HeadOptions::default()
    };
    let r = run_head_with(&mut head, &mut link, prompt, max_new, opts).expect("head");
    (r, tail.join().unwrap().expect("tail"))
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn gpu_chunked_prefill_bit_identical_and_timed() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let len: usize = env_or("I623_PROMPT", 64);
    let chunk: usize = env_or("I623_CHUNK", 16);
    let max_new: usize = env_or("I623_NEW", 8);
    let ks: Vec<u32> = std::env::var("I623_K")
        .unwrap_or_else(|_| "32".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    assert!(
        chunk > 1,
        "I623_CHUNK must be > 1 (1 is the per-position protocol)"
    );
    let prompt = prompt(len);
    let ctx = len + max_new;
    // The width the runner actually uses (the evened chunk) — the oracle
    // and the production cap must match it for the bit bar.
    let opts = HeadOptions {
        prefill_chunk: chunk,
        min_batched: 0,
        ..HeadOptions::default()
    };
    let chunk = prefill_width(len, opts);
    assert!(chunk > 1, "prefill width collapsed to per-position");
    set_prefill_chunk_max(chunk);

    let identity = ArtifactIdentity::of_gguf(&model).expect("identity");
    let (config, weights) = load(&model, ctx, |c| WeightScope::full(c.n_layer));
    let n = config.n_layer as u32;

    // 1. Anchor: production 1-box prefill (chunk-capped at `chunk`) vs the
    //    full-range stage's chunked prefill.
    let want_prod = {
        let gpu = CubeCLContext::new().expect("GPU init");
        let mut fwd = TernaryDeltanetGpuForward::new(&gpu, &config, &weights);
        let ids: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
        fwd.prefill(&ids)
    };
    let (want_toks, want_logits, t_box_chunk, dec_chunk) = {
        let mut s =
            TernaryGpuStage::new(&config, weights.clone(), LayerRange::full(n), identity).unwrap();
        oracle(&mut s, &prompt, max_new, chunk)
    };
    assert!(
        bits_eq(&want_prod, &want_logits[0]),
        "anchor: stage chunked prefill != production prefill (chunk {chunk})"
    );
    println!(
        "anchor: full-range stage prefill == production prefill (P={len}, chunk {chunk}) bit-identical"
    );
    let t_box_pos = {
        let mut s =
            TernaryGpuStage::new(&config, weights.clone(), LayerRange::full(n), identity).unwrap();
        let (toks, _, t, dec_pos) = oracle(&mut s, &prompt, max_new, 1);
        // Batched vs per-position is a different kernel set: tokens are
        // reported, never asserted bit-equal across the two.
        println!("1-box per-position stream {toks:?}");
        (t, dec_pos)
    };
    let (t_box_pos, dec_pos) = t_box_pos;
    println!(
        "1-box prefill P={len}: per-position {:.3}s · chunked({chunk}) {:.3}s = {:.2}x · decode after: per-position {dec_pos:.2} · chunked {dec_chunk:.2} tok/s",
        t_box_pos.as_secs_f64(),
        t_box_chunk.as_secs_f64(),
        t_box_pos.as_secs_f64() / t_box_chunk.as_secs_f64()
    );
    println!("1-box chunked stream {want_toks:?}");
    drop(weights);

    for k in ks.into_iter().filter(|k| (1..n).contains(k)) {
        let (_, hw) = load(&model, ctx, |c| {
            stage_scope(LayerRange::new(0, k), c.n_layer as u32)
        });
        let (_, tw) = load(&model, ctx, |c| {
            stage_scope(LayerRange::new(k, n), c.n_layer as u32)
        });
        // 2. G1 (chunked pair vs chunked oracle) + 3. the interleaved time read.
        let (rc, seen) = pair(&config, &hw, &tw, k, identity, &prompt, max_new, chunk);
        assert_eq!(rc.generated, want_toks, "split@{k}: stream diverged");
        assert_eq!(seen.len(), want_logits.len(), "split@{k}: logit count");
        for (p, (a, b)) in seen.iter().zip(&want_logits).enumerate() {
            assert!(
                bits_eq(a, b),
                "split@{k}: tail logits #{p} not bit-identical"
            );
        }
        let (rp, _) = pair(&config, &hw, &tw, k, identity, &prompt, max_new, 1);
        let (rc2, _) = pair(&config, &hw, &tw, k, identity, &prompt, max_new, chunk);
        assert_eq!(
            rc2.generated, want_toks,
            "split@{k}: re-run stream diverged"
        );
        println!(
            "split k={k}: chunked G1 bit-identical ×{} · prefill P={len}: per-position {:.3}s · chunked({chunk}) {:.3}s / {:.3}s = {:.2}x · decode {:.2} tok/s",
            seen.len(),
            rp.prefill.as_secs_f64(),
            rc.prefill.as_secs_f64(),
            rc2.prefill.as_secs_f64(),
            rp.prefill.as_secs_f64() / rc2.prefill.as_secs_f64(),
            rc.decode_tok_s()
        );
    }
}

/// Plan 623 T2.3c — the prefill_width cost curve on ONE full-range Metal
/// stage: per-position wall vs chunked wall across (P × width), best-of-2
/// after a warm-up of BOTH kernel sets (first-call Metal compiles would
/// otherwise read as the chunk floor). The rows are the raw material for
/// re-deriving `min_batched` (the per-position↔chunked break-even) and
/// the dist bin's default width — the derivation itself lives in the
/// bench record, not here.
///
/// ```sh
/// CARGO_TARGET_DIR=/tmp/i623_dist cargo test -p riir-infer-dist --release \
///   --features gpu,bonsai2_hadamard --test two_stage_prefill_gpu \
///   prefill_width_curve_grid -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn prefill_width_curve_grid() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let ps: Vec<usize> = std::env::var("I623_CURVE_P")
        .unwrap_or_else(|_| "32,48,64,96,128,256".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let ws: Vec<usize> = std::env::var("I623_CURVE_W")
        .unwrap_or_else(|_| "8,16,32,64,128".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    let identity = ArtifactIdentity::of_gguf(&model).expect("identity");
    let ctx = ps.iter().max().copied().unwrap_or(256) + 8;
    let (config, weights) = load(&model, ctx, |c| WeightScope::full(c.n_layer));
    let n = config.n_layer as u32;
    let mut stage = TernaryGpuStage::new(&config, weights, LayerRange::full(n), identity).unwrap();
    let empty: [f32; 0] = [];

    // Warm-up: one per-position pass + one chunked pass at the smallest
    // P (compiles every kernel set both paths touch).
    let warm = prompt(32);
    for (pos, &tok) in warm.iter().enumerate() {
        let _ = stage.step(tok, pos, &empty);
    }
    let mut sink = Vec::new();
    stage.prefill_chunk(&warm, &empty, 0, warm.len(), true, &mut sink);
    assert!(!sink.is_empty(), "warm-up produced no logits");

    let timed_prefill = |stage: &mut TernaryGpuStage, p: usize, w: usize| -> f64 {
        let toks = prompt(p);
        // Best-of-2; position state is overwritten from 0 each pass.
        let mut best = f64::INFINITY;
        for _ in 0..2 {
            let t = Instant::now();
            match w {
                1 => {
                    for (pos, &tok) in toks.iter().enumerate() {
                        let _ = stage.step(tok, pos, &empty);
                    }
                }
                _ => {
                    let mut last = Vec::new();
                    for lo in (0..p).step_by(w) {
                        let hi = (lo + w).min(p);
                        let fin = hi == p;
                        stage.prefill_chunk(&toks[lo..hi], &empty, lo, p, fin, &mut last);
                    }
                }
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    println!("P,width,wall_s,s_per_tok,note",);
    for &p in &ps {
        let pos_wall = timed_prefill(&mut stage, p, 1);
        println!(
            "{p},1,{pos_wall:.4},{:.6},per-position",
            pos_wall / p as f64
        );
        for &w in &ws {
            if w >= p * 2 {
                continue; // a width wider than 2P is the same single chunk
            }
            let wall = timed_prefill(&mut stage, p, w);
            println!(
                "{p},{w},{wall:.4},{:.6},chunked ratio_vs_pos {:.3}",
                wall / p as f64,
                pos_wall / wall
            );
        }
    }
}
