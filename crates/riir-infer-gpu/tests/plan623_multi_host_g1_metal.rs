//! Plan 623 T1.5 — G1 of the GPU range-forward (Metal/CubeCL): the staged
//! composition `stage(0..k) ∘ stage(k..n)` must be BIT-IDENTICAL to the 1-box
//! full GPU forward, for every split k, on the real model.
//!
//! This is the GPU twin of the CPU gate (`tests/plan623_multi_host_g1.rs`,
//! T1.1) and the load-bearing invariant of the whole multi-host lane: a
//! pipeline stage boundary that changes the logits is a wire break, not a
//! tolerance. The bit-exact bar is EXPECTED — the range funnel dispatches the
//! same kernels in the same order per buffer (the boundary law: the fused
//! `residual_add_rmsnorm_f32` Phase-1 add IS `a[i] + b[i]`, one rounding,
//! identical to `ResidualAddCubeCL`, and the RMSNorm reduction tree is the
//! same strided-256 walk in both kernels — verified in-source at the T1.5
//! landing), so any mismatch is a real defect, never a numerics footnote.
//!
//! **Qwen/Bonsai ONLY (owner model-priority law, 2026-10-26):** the gemma2
//! GPU seam (`forward_gpu_range`) exists as compile-only scaffolding mirroring
//! the CPU T1.1 symmetry — its EXECUTED G1 cell on the retired gemma-2-2b
//! checkpoint was RESCINDED by the owner the same day it landed. Gemma2's
//! per-layer bodies are self-contained, so its range bit-identity is
//! structural; the executed gate lives on the production lane (ternary
//! Bonsai-2) alone. No new gemma-2-2b execution cells anywhere.
//!
//! The gate runs on ONE forwarder instance (memory: the 27B is ~7 GB): the
//! reference loop runs first, [`TernaryDeltanetGpuForward::reset_state`]
//! fully resets (KV bounded by pos, GDN/conv zero-filled, pos = 0), then the
//! staged loops replay the same tokens through `[0,k)` → hidden handoff →
//! `[k,n)`. Layer state slots are indexed absolutely and the ranges are
//! disjoint per stage pass, so instance reuse cannot cross-contaminate.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i623 cargo test -p riir-infer-gpu --release \
//!   --features cubecl_runtime,ternary_gemv,multi_host,riir-infer-core/bonsai2_hadamard \
//!   --test plan623_multi_host_g1_metal -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env: `BONSAI2_G1_MODEL` overrides the sibling-layout default; `I623_T`
//! overrides the token count (default 4); `I623_K` (comma list) overrides the
//! every-k split sweep (e.g. `1,32,63` for a quick posture).

#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "multi_host"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn tokens(count: usize, bos: usize, vocab: usize) -> Vec<usize> {
    (0..count)
        .map(|i| {
            if i == 0 {
                bos
            } else {
                (bos + i * 7919) % vocab
            }
        })
        .collect()
}

fn assert_bit_exact(label: &str, t: usize, a: &[f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label} token {t}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            let rel = ((x - y).abs()) / x.abs().max(1e-30);
            panic!(
                "{label} token {t}: BIT DIVERGENCE at [{i}] — {x} vs {y} (rel {rel:.3e}); \
                 the range funnel is not dispatch-identical to the full funnel"
            );
        }
    }
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn qwen_range_forward_bit_identical_every_split() {
    let model = env_path("BONSAI2_G1_MODEL", FOLDED_MODEL);
    if !model.exists() {
        eprintln!("SKIP: model not found at {}", model.display());
        return;
    }
    let t_count: usize = std::env::var("I623_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let ks: Vec<usize> = std::env::var("I623_K")
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_default(); // empty = every split

    let (config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    assert!(
        weights.rotation.is_some(),
        "{}: not a Hadamard-folded file",
        model.display()
    );
    // Decode-only: clamp the KV-cache position capacity to the sequence bound
    // (the g1_032 / riir-train driver pattern; the Issue-994 VRAM gate
    // refuses the loader's default context-sized block).
    let mut config = config;
    config.block_size = t_count + 1;
    let n_layer = config.n_layer;
    let bos = config.vocab_size / 2;
    let toks = tokens(t_count, bos, config.vocab_size);
    println!(
        "qwen arm: n_layer {n_layer}, {t_count} tokens, splits {}",
        if ks.is_empty() {
            "1..n_layer (every)".to_string()
        } else {
            format!("{ks:?}")
        }
    );

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    // Reference: the 1-box full forward over the token sequence.
    let mut reference = Vec::with_capacity(toks.len());
    for (t, &tok) in toks.iter().enumerate() {
        fwd.set_pos(t);
        fwd.set_input_token(&weights, tok);
        reference.push(fwd.forward_token());
    }

    // Staged replay for every split (or the caller's subset).
    let split_list: Vec<usize> = if ks.is_empty() {
        (1..n_layer).collect()
    } else {
        ks.into_iter()
            .filter(|k| (1..n_layer).contains(k))
            .collect()
    };
    for k in split_list {
        fwd.reset_state();
        let staged = staged_pipeline(&mut fwd, &weights, &toks, k, n_layer);
        for (t, (r, s)) in reference.iter().zip(staged.iter()).enumerate() {
            assert_bit_exact("qwen", t, r, s);
        }
        println!("qwen split k={k}: bit-identical ×{t_count}");
    }
}

/// One pipeline pass per token: stage `[0,k)` → hidden handoff → stage
/// `[k,n)` → logits, on the shared instance (positions scheduled explicitly —
/// the range seams never advance `pos` themselves).
fn staged_pipeline(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    toks: &[usize],
    k: usize,
    n_layer: usize,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(toks.len());
    for (t, &tok) in toks.iter().enumerate() {
        fwd.set_pos(t);
        fwd.set_input_token(weights, tok);
        fwd.forward_range_dispatch(LayerRange::new(0, k as u32));
        let hidden = fwd.read_hidden_state();
        // The handoff: same position, hidden state in, fresh mark.
        fwd.set_pos(t);
        fwd.set_hidden_state(&hidden);
        fwd.forward_range_dispatch(LayerRange::new(k as u32, n_layer as u32));
        out.push(fwd.read_logits());
    }
    out
}
