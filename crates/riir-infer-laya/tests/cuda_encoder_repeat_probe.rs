//! Issue 021 T1b — the ENCODER-level repeat probe: the real english
//! checkpoint (ModernBERT-large), one banking77-length input (seq 400 —
//! inside the suite's long-seq class), `Encoder::forward_probe`'s per-op
//! sink bit-compared across REPEATS on the CUDA backend.
//!
//! The op-level probe (`cuda_repeat_probe`) came back GREEN — every kernel
//! is bit-stable in isolation at these shapes ×200 — while the harness
//! banking77 cell flips `determinism_ok` under EVERY kernel posture switch
//! (flash off / reg4 off / ladder off all still flip). So the wobble lives
//! in the composition above single ops. This probe walks the packed
//! forward's exact op stream (the `forward_probe` tags, one `download_into`
//! per op) and names the first tag that differs between run 0 and a later
//! run — encoder-level localization in one run.
//!
//! `begin_pass()` between repeats (the agent's real per-forward epoch
//! cadence — fresh uploads + fresh slots each call, exactly what two
//! back-to-back `system_one` calls see).
//!
//! Deep mode (`LAYA_PROBE_DEEP=1`) additionally sinks the attention block's
//! device-written internals (q/k/v/scores/ctx) per layer — the kernel-level
//! granularity step. Not armed by default (a ~15 MB/layer/pass readback).
//!
//! Runs only under `laya-riir-cuda` on a non-macOS host; skips loud (the
//! NDB_BIN posture) when the english checkpoint is absent.

#![cfg(all(
    not(target_os = "macos"),
    feature = "laya-riir-cuda",
    feature = "laya-riir-cubecl"
))]

use std::sync::{Mutex, MutexGuard, OnceLock};

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::cuda::Cuda;
use riir_infer_laya::laya::weights::weights_root;

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// LCG token ids (the `cubecl_encoder_probe` form).
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

fn load_encoder(ckpt: Checkpoint) -> (riir_infer_laya::laya::riir::encoder::Encoder, usize) {
    use riir_infer_laya::laya::config::load_checkpoint_configs;
    use riir_infer_laya::laya::riir::encoder::Encoder;
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
fn cuda_encoder_forward_probe_repeats_bit_stable() {
    let dir = weights_root().join(Checkpoint::English.subfolder());
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIP: no english checkpoint at {}", dir.display());
        return;
    }
    let _gpu = gpu_lock();

    let (enc, vocab) = load_encoder(Checkpoint::English);
    let ids = ids_for(SEQ, vocab);
    let g = Cuda::new().expect("cuda backend");

    // Run 0 — the golden tag table.
    let mut golden: Vec<(String, Vec<f32>)> = Vec::new();
    g.begin_pass();
    enc.forward_probe(&g, &ids, &mut |tag, bytes| {
        golden.push((tag.to_string(), bytes.to_vec()))
    })
    .expect("golden probe");
    println!(
        "golden run: {} tags, {} floats total",
        golden.len(),
        golden.iter().map(|(_, b)| b.len()).sum::<usize>()
    );

    let mut failed = false;
    for run in 1..=RUNS {
        let mut got: Vec<(String, Vec<f32>)> = Vec::new();
        g.begin_pass();
        enc.forward_probe(&g, &ids, &mut |tag, bytes| {
            got.push((tag.to_string(), bytes.to_vec()))
        })
        .expect("probe run");
        assert_eq!(golden.len(), got.len(), "tag count diverged at run {run}");
        for ((t0, a), (t1, b)) in golden.iter().zip(got.iter()) {
            assert_eq!(t0, t1, "tag order diverged at run {run}");
            if let Some((idx, delta)) = first_diff(a, b) {
                failed = true;
                println!(
                    "✗ run {run}: tag {t0} DIVERGED — idx {idx} (of {}), max |Δ| {delta:.3e}",
                    a.len()
                );
            }
        }
        if !failed {
            println!("✓ run {run}: {} tags bit-stable", golden.len());
        } else {
            break; // first divergent run names the tag; later runs add nothing
        }
    }
    assert!(!failed, "encoder repeat probe: a tag DIVERGED");
}
