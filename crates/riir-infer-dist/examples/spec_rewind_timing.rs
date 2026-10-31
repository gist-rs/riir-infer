//! Issue 1148 rung-2 timing probe — the measured case for the exact
//! checkpoint fast path against the `HybridStateSegment` codec.
//!
//! Times, on ONE stage at a fixed cursor, five mechanisms (median of N):
//! - `codec snapshot`  — `HybridStateSegment::encode_into` (the V0 rewind
//!   capture, wire-shaped bytes, full buffers),
//! - `codec restore`   — `restore_into`,
//! - `ck snapshot`     — `SpecStage::checkpoint_state` (retained buffers,
//!   live rows only),
//! - `ck restore`      — `restore_checkpoint`,
//! - `decode step`     — one `StageCompute::step` (the scale every rewind
//!   cost should be read against).
//!
//! Postures:
//! - default: the synth Bonsai-2 (mechanism check, always runs);
//! - `--bonsai <path>` (or `BONSAI_GGUF`): the real 27B head stage at
//!   split 63 — the Bench 917 seat. State is left at construction zeros
//!   (values are irrelevant to timing; no prefill is run).
//!
//! Report-only: numbers land in the bench note, box state rides the
//! invocation record.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use riir_infer_core::deltanet::WeightScope;
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped;
use riir_infer_core::multi_host::LayerRange;
use riir_infer_dist::spec_fill::{CheckpointLayer, SpecStage, StageCheckpoint};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage, stage_scope};

const CURSOR: usize = 600; // the Bench 917 posture (~600 positions)
const ITERS: usize = 20;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn time_stage(stage: &mut TernaryCpuStage, label: &str, kvd: usize) {
    let empty: [f32; 0] = [];
    let tok = 7u32;

    // Buffer warmup (untimed): the codec segment buffer + the checkpoint
    // buffers reach steady state — the timed loop reuses them, exactly the
    // runner's posture.
    let mut snap: Vec<u8> = Vec::new();
    stage.snapshot_state(&mut snap).expect("codec snapshot");
    let snap_mb = snap.len() as f64 / (1024.0 * 1024.0);
    let mut ck = StageCheckpoint::default();
    stage.checkpoint_state(CURSOR, &mut ck).expect("checkpoint");
    let ck_live: usize = ck
        .layers
        .iter()
        .map(|l| match l {
            CheckpointLayer::Attention { key, value } => key.len() + value.len(),
            CheckpointLayer::DeltaNet { rec, conv } => rec.len() + conv.len(),
        })
        .sum();
    let ck_mb = ck_live as f64 * 4.0 / (1024.0 * 1024.0);

    let (mut t_enc, mut t_dec, mut t_ck, mut t_ckr, mut t_step) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        let t = Instant::now();
        stage.snapshot_state(&mut snap).expect("codec snapshot");
        t_enc.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        stage.restore_state(&snap).expect("codec restore");
        t_dec.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        stage.checkpoint_state(CURSOR, &mut ck).expect("checkpoint");
        t_ck.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        stage.restore_checkpoint(&ck).expect("ck restore");
        t_ckr.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        let _ = stage.step(tok, CURSOR, &empty);
        t_step.push(t.elapsed().as_secs_f64());
    }

    let (enc, dec, cks, ckr, stp) = (
        median(&mut t_enc),
        median(&mut t_dec),
        median(&mut t_ck),
        median(&mut t_ckr),
        median(&mut t_step),
    );
    println!(
        "[{label}] cursor {CURSOR} · kv_dim {kvd} · codec segment {snap_mb:.1} MB · ck live {ck_mb:.1} MB"
    );
    println!(
        "[{label}] ms: enc {:.2} · dec {:.2} · ck {:.2} · ckr {:.2} · step {:.2} · ratios vs step: enc {:.1}x dec {:.1}x ck {:.1}x ckr {:.1}x",
        enc * 1e3,
        dec * 1e3,
        cks * 1e3,
        ckr * 1e3,
        stp * 1e3,
        enc / stp,
        dec / stp,
        cks / stp,
        ckr / stp,
    );
}

fn main() {
    let mut bonsai: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bonsai" => bonsai = args.next().map(PathBuf::from),
            other => {
                if other.starts_with("--") {
                    eprintln!("unknown arg {other}");
                }
            }
        }
    }
    if bonsai.is_none()
        && let Ok(p) = std::env::var("BONSAI_GGUF")
    {
        bonsai = Some(PathBuf::from(p));
    }

    // Synth posture (always).
    {
        let bytes = synth::build_gguf(
            &synth::synth_metadata(false),
            &synth::synth_tensors_bonsai2(),
        );
        let path = synth::write_tmp("spec_rewind_timing", &bytes);
        let (config, w) = load_qwen_deltanet_ternary_weights_gguf_scoped(&path, |cfg| {
            WeightScope::full(cfg.n_layer)
        })
        .expect("load synth");
        let id = ArtifactIdentity::of_gguf(&path).expect("identity");
        let _ = std::fs::remove_file(&path);
        let kvd = config.n_kv_head * config.head_dim;
        let n = config.n_layer as u32;
        let mut stage =
            TernaryCpuStage::new(config, Arc::new(w), LayerRange::full(n), id).expect("synth");
        time_stage(&mut stage, "synth full", kvd);
    }

    // Real posture: the Bench 917 head seat, split 63, ctx clamped to
    // 4096 (the bench posture — the GGUF's native 128k context would
    // allocate 128k-row KV buffers per layer, the Issue-929 class).
    if let Some(model) = bonsai {
        let identity = ArtifactIdentity::of_gguf(&model).expect("identity");
        let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf_scoped(&model, |c| {
            stage_scope(LayerRange::new(0, 63), c.n_layer as u32)
        })
        .map_err(|e| format!("load {model:?}: {e}"))
        .expect("load bonsai head");
        config.block_size = 4096;
        let n_layer = config.n_layer as u32;
        assert_eq!(
            n_layer, 64,
            "the split-63 seat assumes the 64-layer artifact"
        );
        let kvd = config.n_kv_head * config.head_dim;
        let mut stage =
            TernaryCpuStage::new(config, Arc::new(weights), LayerRange::new(0, 63), identity)
                .expect("bonsai head stage");
        time_stage(&mut stage, "bonsai head [0,63)", kvd);
    } else {
        println!("[bonsai] skipped — pass --bonsai <path> or set BONSAI_GGUF");
    }
}
