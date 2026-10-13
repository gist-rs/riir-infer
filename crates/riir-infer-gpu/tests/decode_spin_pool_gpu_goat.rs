//! katgpt-rs Issue 936 T5 — the decode spin pool on the GPU-HOOKED ternary
//! decode (Metal lane). The CPU-lane gate is
//! [`decode_spin_pool_goat`](../../tests/decode_spin_pool_goat.rs); this one
//! installs the three GPU hooks (matvec / fused input-proj / fused FFN, the
//! Issue 599/602/601 lanes) so every ternary GEMV dispatches to Metal while
//! the CPU side keeps norms, conv1d, the recurrence glue and the readbacks.
//! `on_decode_pool` wraps the whole hooked body, so the pool posture is
//! ALREADY the default here (`decode_spin_pool` is implied by
//! `deltanet_ternary_inference`) — the load-bearing assertion is NO HARM:
//! between hooks the pool's workers spin ≤ 100 µs and then park, and a GPU
//! dispatch (~2 ms) is far past the idle window, so the pool must not cost
//! the hooked lane anything measurable.
//!
//! - **G1**: every logit of every teacher-forced position is bit-identical
//!   between the arms (same kernels, same dispatch order; scheduling only).
//! - **G2 (no-harm)**: median per-position (pool / rayon) wall ratio
//!   ≤ [`MAX_TIME_RATIO`]. A ratio materially BELOW 1.0 is reported, never
//!   asserted — a hooked-lane win is data for the record, not a bar.
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/t5gpu cargo test -p riir-infer-gpu --release \
//!   --features spin_pool_gpu_goat \
//!   --test decode_spin_pool_gpu_goat -- --ignored --nocapture --test-threads=1
//! ```
//! Env: `BONSAI2_GGUF`, `SPIN_GPU_POSITIONS` (default 24), plus the
//! sibling-layout model default. MEASUREMENT, never a league cell (Plan 602
//! C4 owns those).
#![cfg(feature = "spin_pool_gpu_goat")]

use std::time::Instant;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::set_decode_spin_pool;
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::deltanet::ternary_weights::{
    QwenDeltaNetTernaryWeights, set_ternary_q8_act,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::{GpuTernaryFfn, GpuTernaryInputProj, GpuTernaryMatvec};

const MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const CORPUS: &str = "../../.raw/twt/audition_calib.txt";
const DEFAULT_POSITIONS: usize = 24;
/// No-harm bar: the pool may not cost the hooked lane > 5% at the median.
const MAX_TIME_RATIO: f64 = 1.05;

fn env_path(var: &str, default: &str) -> std::path::PathBuf {
    std::env::var_os(var)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(default))
}

fn sh(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().replace('\n', " "))
        .unwrap_or_else(|_| "unavailable".into())
}

/// One arm's decode state (the weights and the hooks are shared; only the
/// cache/scratch/x are per-arm so the two postures never share KV state).
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

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        pool: bool,
        w: &QwenDeltaNetTernaryWeights,
        matvec: &GpuTernaryMatvec,
        input_proj: &GpuTernaryInputProj,
        ffn: &GpuTernaryFfn,
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
            Some(matvec),
            Some(input_proj),
            Some(ffn),
        );
        t.elapsed().as_secs_f64()
    }
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run with --ignored --release"]
fn decode_spin_pool_is_no_harm_on_the_gpu_hooked_lane() {
    let model = env_path("BONSAI2_GGUF", MODEL);
    let corpus = env_path("Q8_GATE_CORPUS", CORPUS);
    let n_pos: usize = std::env::var("SPIN_GPU_POSITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_POSITIONS);
    assert!(
        model.is_file() && corpus.is_file(),
        "UNSEEN, not a pass: model {} or corpus {} missing",
        model.display(),
        corpus.display()
    );
    println!(
        "PROVENANCE: model {} power [{}] powermode [{}] loadavg [{}] positions={n_pos}",
        model.display(),
        sh("pmset", &["-g", "batt"]),
        sh("sh", &["-c", "pmset -g | grep -i powermode"]),
        sh("sysctl", &["-n", "vm.loadavg"]),
    );

    let text = load_corpus_text(&corpus).expect("corpus");
    let tok =
        BpeTokenizer::from_gguf(&GgufFile::open(&model).expect("gguf")).expect("tokenizer");
    let ids = tok.encode(&text);
    assert!(ids.len() > n_pos, "corpus too short: {} tokens", ids.len());

    set_ternary_q8_act(true);
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&model).expect("load");
    assert!(
        weights.rotation.is_some(),
        "{}: not a Hadamard-folded file (this gate is the Bonsai-2 lane)",
        model.display()
    );
    config.block_size = n_pos + 1;
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let vocab = config.vocab_size;

    // ── GPU + hooks: upload every ternary projection once ──
    let ctx = CubeCLContext::new().expect("GPU init");
    let client = ctx.client();
    let matvec = GpuTernaryMatvec::new(client.clone());
    let input_proj = GpuTernaryInputProj::new(client.clone());
    let ffn = GpuTernaryFfn::new(client.clone());
    let upload = Instant::now();
    for layer in &weights.layers {
        for w in [
            &layer.in_proj_qkv,
            &layer.in_proj_z,
            &layer.out_proj,
            &layer.attn_wq,
            &layer.attn_wk,
            &layer.attn_wv,
            &layer.attn_wo,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
        ] {
            if let Some(t) = w.as_ternary() {
                matvec.preupload(t);
            }
        }
        // Issue-980 class: the DeltaNet gate projections are ternary in
        // pre-rotation files, DENSE BF16 in Bonsai 2 — only the ternary arm
        // is hook material.
        if let (Some(qkv), Some(z), Some(a), Some(b)) = (
            layer.in_proj_qkv.as_ternary(),
            layer.in_proj_z.as_ternary(),
            layer.in_proj_a.as_ternary(),
            layer.in_proj_b.as_ternary(),
        ) {
            input_proj.preupload(qkv, z, a, b);
        }
        if let (Some(g), Some(u), Some(d)) = (
            layer.gate_proj.as_ternary(),
            layer.up_proj.as_ternary(),
            layer.down_proj.as_ternary(),
        ) {
            ffn.preupload(g, u, d);
        }
    }
    matvec.preupload(&weights.lm_head);
    matvec.preupload(&weights.wte);
    println!(
        "[spin-gpu] hook upload done in {:.1}s",
        upload.elapsed().as_secs_f64()
    );

    let (mut on, mut off) = (
        Arm::new(&config, &weights),
        Arm::new(&config, &weights),
    );

    // Warmup: one position per arm, NOT measured (first-call JIT compile of
    // every CubeCL kernel + the handle-cache fills).
    let _ = on.step(true, &weights, &matvec, &input_proj, &ffn, ids[0], 0, &config, &rope);
    let _ = off.step(false, &weights, &matvec, &input_proj, &ffn, ids[0], 0, &config, &rope);

    let mut ratios = Vec::with_capacity(n_pos);
    let (mut wall_on, mut wall_off, mut mismatched) = (0.0f64, 0.0f64, 0usize);
    for (p, &id) in ids.iter().take(n_pos).enumerate() {
        let (t_on, t_off) = if p % 2 == 0 {
            let a = on.step(true, &weights, &matvec, &input_proj, &ffn, id, p, &config, &rope);
            let b = off.step(false, &weights, &matvec, &input_proj, &ffn, id, p, &config, &rope);
            (a, b)
        } else {
            let b = off.step(false, &weights, &matvec, &input_proj, &ffn, id, p, &config, &rope);
            let a = on.step(true, &weights, &matvec, &input_proj, &ffn, id, p, &config, &rope);
            (a, b)
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
    println!(
        "[spin-gpu] positions {n_pos} | logit-vector mismatches {mismatched} (bar 0)"
    );
    println!(
        "[spin-gpu] hooked lane: rayon {:.1} ms/tok ({:.2} tok/s) · pool {:.1} ms/tok ({:.2} tok/s) | median paired ratio {median:.3} (no-harm bar ≤ {MAX_TIME_RATIO})",
        wall_off / n_pos as f64 * 1e3,
        n_pos as f64 / wall_off,
        wall_on / n_pos as f64 * 1e3,
        n_pos as f64 / wall_on,
    );
    assert_eq!(
        mismatched, 0,
        "G1: the spin pool changed logits on the hooked lane"
    );
    assert!(
        median <= MAX_TIME_RATIO,
        "G2 no-harm: median ratio {median:.3} > {MAX_TIME_RATIO} — the pool costs the hooked lane; demote it there"
    );
}
