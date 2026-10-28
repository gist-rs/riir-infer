//! Issue 1146 rung 2a — G1 of the Metal tensor (matmul2d) zero-copy prefill
//! GEMM through the RANGE prefill seam the dist lane serves
//! (`prefill_range_chunk`), against the simdgroup incumbent.
//!
//! The league harness (riir-ai Bench 1009) proved the zerocopy arm G1-exact
//! through the whole-prompt `prefill()` API; the dist tensor-armed
//! build then generated a DIFFERENT first token on the same model. This gate
//! localises the break: three postures on ONE instance, last-position logits
//! compared BIT-EXACT (argmax + max_abs == 0 — the Bench-1009 class), the
//! zerocopy arm FIRST in every pair:
//!
//! - `prefill()` zerocopy OFF vs ON (the league path — the control);
//! - `prefill_range_chunk(0..n)` in `I1146T_W`-wide frames, OFF vs ON;
//! - the 2-stage split `[0,k)` → hidden rows → `[k,n)` per frame, OFF vs ON.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1146t cargo test -p riir-infer-gpu --release \
//!   --features metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1146_tensor_range_g1 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` (model path), `I1146T_P` (prompt length, default
//! 512), `I1146T_W` (frame width, default 128), `I1146T_K` (split, default 32).

#![cfg(all(
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "multi_host",
    feature = "ternary_gemm_batched",
    target_os = "macos"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::set_prefill_use_metal_tensor_zerocopy;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Last-position logits through `prefill_range_chunk`, either one full-range
/// stage or a `[0,k)` → `[k,n)` split per frame (the dist head/tail shape).
fn range_prefill(
    fwd: &mut TernaryDeltanetGpuForward,
    toks: &[usize],
    w: usize,
    split: Option<usize>,
    n_layer: usize,
) -> Vec<f32> {
    fwd.reset_state();
    let p = toks.len();
    let mut logits = Vec::new();
    for (ci, chunk) in toks.chunks(w).enumerate() {
        let base = ci * w;
        let last = base + chunk.len() == p;
        logits = match split {
            None => fwd.prefill_range_chunk(
                chunk,
                None,
                base,
                p,
                LayerRange::new(0, n_layer as u32),
                last,
            ),
            Some(k) => {
                let hidden = fwd.prefill_range_chunk(
                    chunk,
                    None,
                    base,
                    p,
                    LayerRange::new(0, k as u32),
                    false,
                );
                fwd.prefill_range_chunk(
                    &[],
                    Some(&hidden),
                    base,
                    p,
                    LayerRange::new(k as u32, n_layer as u32),
                    last,
                )
            }
        };
    }
    logits
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn tensor_zerocopy_range_prefill_matches_simdgroup() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let p = env_usize("I1146T_P", 512);
    let w = env_usize("I1146T_W", 128);
    let k = env_usize("I1146T_K", 32);

    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    config.block_size = p + 8;
    let n_layer = config.n_layer;
    let toks: Vec<usize> = (0..p).map(|i| 7 + (i * 7919) % 50_000).collect();

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    let mut rows: Vec<(&str, usize, f32)> = Vec::new();
    let mut check = |label: &'static str, off: Vec<f32>, on: Vec<f32>| {
        let (a0, a1, d) = (argmax(&off), argmax(&on), max_abs(&off, &on));
        println!(
            "{label}: argmax off {a0} on {a1} · max_abs {d:.3e} · len {}",
            off.len()
        );
        rows.push((label, usize::from(a0 != a1 || d != 0.0), d));
    };

    // ON runs FIRST in every pair, on scratch shapes the pool has never
    // served: the dist stage's first prefill is exactly that, and an
    // OFF-first order warms the pool and hides the first-use hazard
    // (Issue 670 T5's "first-prefill pool-state" class).
    set_prefill_use_metal_tensor_zerocopy(true);
    fwd.reset_state();
    let whole_on = fwd.prefill(&toks);
    set_prefill_use_metal_tensor_zerocopy(false);
    fwd.reset_state();
    let whole_off = fwd.prefill(&toks);
    check("prefill() whole", whole_off, whole_on);

    set_prefill_use_metal_tensor_zerocopy(true);
    let full_on = range_prefill(&mut fwd, &toks, w, None, n_layer);
    set_prefill_use_metal_tensor_zerocopy(false);
    let full_off = range_prefill(&mut fwd, &toks, w, None, n_layer);
    check("range full", full_off, full_on);

    set_prefill_use_metal_tensor_zerocopy(true);
    let split_on = range_prefill(&mut fwd, &toks, w, Some(k), n_layer);
    set_prefill_use_metal_tensor_zerocopy(false);
    let split_off = range_prefill(&mut fwd, &toks, w, Some(k), n_layer);
    check("range split", split_off, split_on);

    let bad: Vec<_> = rows.iter().filter(|(_, flip, _)| *flip != 0).collect();
    assert!(bad.is_empty(), "zerocopy diverges from simdgroup: {bad:?}");
}
