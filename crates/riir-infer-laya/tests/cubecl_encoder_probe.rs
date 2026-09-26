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
    println!("first structural divergence: {first_bad:?}");

    // Tracked-index trajectories (issue 018: the deterministic component
    // of the drift lives on a handful of large-magnitude residual
    // elements — print their per-tag diff so the injection SITE is
    // visible, not just the per-tag max).
    const TRACKED: [usize; 2] = [5499, 206203];
    for &ti in &TRACKED {
        print!("tracked[{ti:>6}]:");
        for ((t1, a), (_t2, b)) in cpu_ops.iter().zip(gpu_ops.iter()) {
            // Deep-mode tags include short buffers (the rope table is
            // seq·hd) — guard the index per tag.
            if a.len() <= ti || b.len() <= ti {
                continue;
            }
            let d = (a[ti] - b[ti]).abs();
            if d > 1e-6 {
                print!(" {t1}={d:.2e}");
            }
        }
        println!();
    }

    // ── CubeCL repeat loop (new pass each iteration, same ids): in-process
    // determinism — the 018 lever-1 instrument. Each pass is diffed against
    // the FIRST cubecl pass op-by-op; a nonzero diff means the pass read
    // content the same op produced differently in pass 1 (or read a slot
    // another op trampled). The FIRST divergent tag per fire is the
    // localization datum (the kernel or the binding path). Pass count:
    // LAYA_PROBE_PASSES, default 8 — the wobble fires ~5–10% of forwards,
    // so 8 passes ≈ coin-flip per run; run repeatedly (and under GPU load
    // to raise the fire rate) until a fire names its op.
    let passes: usize = std::env::var("LAYA_PROBE_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mut fires = 0usize;
    for p in 0..passes {
        gpu.begin_pass();
        let mut gpu_ops_p: Vec<(String, Vec<f32>)> = Vec::new();
        enc.forward_probe(&gpu, &ids, &mut |tag, bytes| {
            gpu_ops_p.push((tag.to_string(), bytes.to_vec()))
        })
        .unwrap_or_else(|e| panic!("cubecl probe pass {p}: {e}"));
        assert_eq!(gpu_ops.len(), gpu_ops_p.len(), "pass {p}: op count diverges");
        let mut pass_max = 0.0f32;
        let mut pass_tag = String::new();
        let mut first_bad: Option<(String, f32)> = None;
        for ((t1, a), (t2, b)) in gpu_ops.iter().zip(gpu_ops_p.iter()) {
            assert_eq!(t1, t2, "pass {p}: op tags diverge");
            let mut diff = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (x - y).abs();
                if d > diff {
                    diff = d;
                }
            }
            if diff > 1e-4 {
                println!("  REPEAT-DIFF p={p} {t1:>14} {diff:.3e}");
                if first_bad.is_none() {
                    first_bad = Some((t1.clone(), diff));
                }
            }
            if diff > pass_max {
                pass_max = diff;
                pass_tag = t1.clone();
            }
        }
        if first_bad.is_some() {
            fires += 1;
        }
        println!(
            "pass {p}: max {pass_max:.3e} (at {pass_tag}) · first: {:?}",
            first_bad.map(|(t, d)| format!("{t} ({d:.3e})"))
        );
    }
    println!("cubecl repeat loop: {fires}/{passes} passes fired (threshold 1e-4)");
    // The loop is a MEASUREMENT instrument, not a gate: the wobble is the
    // issue-018 open question and a fire must not fail the run (the
    // intermittently-failing-gate law). The cpu-vs-cubecl structural
    // check above stays the asserting half.
}
