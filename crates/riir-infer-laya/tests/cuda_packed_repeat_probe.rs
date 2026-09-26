//! Issue 021 T1c — the PACKED-forward repeat probe: the real english
//! checkpoint, `Encoder::forward_packed` driven directly on the CUDA
//! backend, the returned hidden handle downloaded and bit-compared across
//! REPEATS. The per-op variant (`cuda_encoder_repeat_probe`) rides
//! `forward_probe`, whose `laya-riir-cubecl` gate drags a dev-dep closure
//! this box's rustc 1.98.1 cannot currently compile (reproducible
//! STATUS_ACCESS_VIOLATION on katgpt-speculative/katgpt-forward) — this
//! slimmer probe answers the same first question under `laya-riir-cuda`
//! alone: is the packed forward bit-stable across back-to-back calls
//! (the agent's real per-answer cadence — `begin_pass` per call)?
//!
//! SKIPs loud when the english checkpoint is absent (the NDB_BIN posture).

#![cfg(all(not(target_os = "macos"), feature = "laya-riir-cuda"))]

use std::sync::{Mutex, MutexGuard, OnceLock};

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::cuda::Cuda;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::weights::weights_root;

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn ids_for(seq: usize, vocab: usize) -> Vec<u32> {
    let mut s: u64 = 0x243F_6A88_85A3_08D3;
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
    use riir_infer_laya::laya::config::load_checkpoint_configs;
    use riir_infer_laya::laya::riir::weights as riir_weights;
    let root = weights_root();
    let name = ckpt.subfolder();
    let dir = root.join(name);
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("checkpoint configs");
    let vocab = enc_cfg.vocab;
    let mut raw = riir_weights::load(&dir.join("model.safetensors"), name).expect("safetensors");
    let enc = Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder");
    (enc, vocab)
}

fn first_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32)> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x.to_bits() != y.to_bits())
        .map(|i| (i, (a[i] - b[i]).abs()))
}

const RUNS: usize = 30;
const SEQ: usize = 400;

#[test]
fn cuda_packed_forward_repeats_bit_stable() {
    let dir = weights_root().join(Checkpoint::English.subfolder());
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIP: no english checkpoint at {}", dir.display());
        return;
    }
    let _gpu = gpu_lock();

    let (enc, vocab) = load_encoder(Checkpoint::English);
    let ids = ids_for(SEQ, vocab);

    // CPU reference of the same packed forward — the sanity anchor: the
    // cuda run's FIRST hidden must be inside G5-class drift of it, so a
    // probe red reads as cuda-local wobble, never as a broken load.
    let cpu_hidden = enc
        .forward_packed(&Cpu, &ids, &[SEQ])
        .expect("cpu reference forward");

    let g = Cuda::new().expect("cuda backend");
    let download = |h: &Vec<f32>| {
        let mut out = vec![0f32; h.len()];
        g.download_into(h, &mut out);
        out
    };

    g.begin_pass();
    let golden_handle = enc.forward_packed(&g, &ids, &[SEQ]).expect("cuda forward");
    let golden = download(&golden_handle);
    let (idx, delta) = first_diff(&cpu_hidden, &golden).map_or_else(
        || {
            println!("cpu↔cuda run 0: bit-identical");
            (usize::MAX, 0.0f32)
        },
        |v| v,
    );
    if idx != usize::MAX {
        println!("cpu↔cuda run 0: first diff idx {idx}, |Δ| {delta:.3e} (G5-class, expected)");
    }

    let mut failed = false;
    for run in 1..=RUNS {
        g.begin_pass();
        let h = enc.forward_packed(&g, &ids, &[SEQ]).expect("cuda forward");
        let got = download(&h);
        if let Some((i, d)) = first_diff(&golden, &got) {
            failed = true;
            println!(
                "✗ run {run}: packed forward DIVERGED — idx {i} (of {}), max |Δ| {d:.3e}; \
                 cpu-vs-run0 at same idx |Δcpu| {:.3e}",
                golden.len(),
                (cpu_hidden[i] - golden[i]).abs()
            );
            break;
        }
        println!("✓ run {run}: {} floats bit-stable", golden.len());
    }
    assert!(!failed, "packed forward repeat probe: DIVERGED");
}
