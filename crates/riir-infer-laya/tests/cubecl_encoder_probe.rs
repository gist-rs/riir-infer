#![cfg(feature = "laya-riir-cubecl")]
//! Issue-016/018 bisect + stability probe (the 018 instrument): run ONE
//! synthetic sequence through `Encoder::forward_probe` on the Cpu and
//! Cubecl backends and diff per-op. Three questions in one run:
//!
//! 1. cpu-vs-cubecl per-op max-abs — the FIRST tag whose diff jumps from
//!    the reduction-order baseline (~1e-5) to something structural
//!    localizes the wrong op.
//! 2. cubecl-vs-cubecl across repeated passes (a `begin_pass` between) —
//!    a nonzero diff means in-process non-determinism (the 018 wobble);
//!    zero means the run's pass stream was stable.
//!
//! Geometry is the real english checkpoint (weights are cached — no
//! download); the input is a fixed xorshift id stream, seq 400 (long
//! enough that the sliding window and odd-length edge tiles both bite).

use riir_infer_laya::laya::config::{load_checkpoint_configs, Checkpoint};
use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::cubecl::CubeclBackend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::weights as riir_weights;
use riir_infer_laya::laya::weights::weights_root;

/// Fixed xorshift id stream — deterministic across runs, no RNG crate.
fn ids_for(seq: usize, vocab: usize) -> Vec<u32> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    (0..seq)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % vocab as u64) as u32
        })
        .collect()
}

fn load_encoder(ckpt: Checkpoint) -> (Encoder, usize) {
    let root = weights_root();
    let name = ckpt.subfolder();
    let dir = root.join(name);
    let (_agent_cfg, enc_cfg) =
        load_checkpoint_configs(&dir, name).expect("checkpoint configs");
    let vocab = enc_cfg.vocab;
    let mut raw =
        riir_weights::load(&dir.join("model.safetensors"), name).expect("safetensors");
    let enc = Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder");
    (enc, vocab)
}

#[test]
fn probe_ops_cpu_vs_cubecl_and_repeat() {
    let (enc, vocab) = load_encoder(Checkpoint::English);
    let seq = 400usize;
    let ids = ids_for(seq, vocab);

    // ── CPU reference (per-op) ──
    let cpu = Cpu;
    let mut cpu_ops: Vec<(String, Vec<f32>)> = Vec::new();
    enc.forward_probe(&cpu, &ids, &mut |tag, bytes| {
        cpu_ops.push((tag.to_string(), bytes.to_vec()))
    })
    .expect("cpu probe");

    // ── CubeCL (per-op) ──
    let gpu = CubeclBackend::new().expect("cubecl backend");
    let mut gpu_ops: Vec<(String, Vec<f32>)> = Vec::new();
    enc.forward_probe(&gpu, &ids, &mut |tag, bytes| {
        gpu_ops.push((tag.to_string(), bytes.to_vec()))
    })
    .expect("cubecl probe");

    assert_eq!(cpu_ops.len(), gpu_ops.len(), "op count diverges");
    println!(
        "{:>14} {:>12} {:>8} {:>8}",
        "tag", "abs_diff", "idx", "n>1e-2"
    );
    let mut first_bad = None;
    for ((t1, a), (t2, b)) in cpu_ops.iter().zip(gpu_ops.iter()) {
        assert_eq!(t1, t2, "op tags diverge");
        let mut diff = 0.0f32;
        let mut idx = 0usize;
        let mut bad = 0usize;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            if d > diff {
                diff = d;
                idx = i;
            }
            if d > 1e-2 {
                bad += 1;
            }
        }
        println!(
            "{t1:>14} {diff:>12.3e} {idx:>8} {bad:>8}  (cpu {} gpu {})",
            a[idx], b[idx]
        );
        if diff > 1e-2 && first_bad.is_none() {
            first_bad = Some(t1.clone());
        }
    }
    println!("first structural divergence: {:?}", first_bad);

    // ── CubeCL repeat (new pass, same ids): in-process determinism ──
    gpu.begin_pass();
    let mut gpu_ops2: Vec<(String, Vec<f32>)> = Vec::new();
    enc.forward_probe(&gpu, &ids, &mut |tag, bytes| {
        gpu_ops2.push((tag.to_string(), bytes.to_vec()))
    })
    .expect("cubecl probe 2");
    let mut repeat_max = 0.0f32;
    let mut repeat_tag = String::new();
    let mut first_repeat_bad: Option<String> = None;
    for ((t1, a), (t2, b)) in gpu_ops.iter().zip(gpu_ops2.iter()) {
        assert_eq!(t1, t2);
        let mut diff = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > diff {
                diff = d;
            }
        }
        if diff > 1e-4 && first_repeat_bad.is_none() {
            first_repeat_bad = Some(format!("{t1} ({diff:.3e})"));
        }
        if diff > repeat_max {
            repeat_max = diff;
            repeat_tag = t1.clone();
        }
        if diff > 1e-4 {
            println!("  REPEAT-DIFF {t1:>14} {diff:.3e}");
        }
    }
    println!(
        "cubecl repeat max diff: {repeat_max:.3e} (at {repeat_tag}) · first: {first_repeat_bad:?}"
    );
}
