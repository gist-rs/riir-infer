//! Issue 1106 — the stability readout consumed END-TO-END through the
//! real capture seam: a served checkpoint on the CPU backend,
//! `Encoder::forward_capture` → `StabilityTracker` → depth, with the
//! delta series bit-parity-checked against a brute-force recompute from
//! the retained last-position states (the G1 fixture law, at the real
//! checkpoint).
//!
//! Skips LOUD when the english checkpoint is absent (a deferral, never
//! a green) — the pure-math gates in `stability.rs` run everywhere.
//!
//! Runs: `cargo test -p riir-infer-laya --features twt_profile --test
//! issue1106_stability_capture_smoke`
#![cfg(feature = "twt_profile")]

use std::collections::HashMap;

use riir_infer_laya::laya::config::{self, Checkpoint};
use riir_infer_laya::laya::riir::backend::Cpu;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::stability::{stage_ordinal, StabilityBand, StabilityTracker};
use riir_infer_laya::laya::riir::weights as lane_weights;
use riir_infer_laya::laya::tokenize::Tok;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

const PROMPTS: [&str; 3] = [
    "The customer was charged twice and wants a refund.",
    "How do I reset my password?",
    "Where is my order?",
];

#[test]
fn issue1106_end_to_end_capture_to_depth_on_real_checkpoint() {
    let ckpt = Checkpoint::English;
    let root = weights_root();
    if !root.join(ckpt.subfolder()).is_dir() {
        panic!(
            "SKIP-LOUD: checkpoint dir {} absent — the e2e stability smoke needs the \
             served weights (pure-math gates run everywhere; this leg is the wiring \
             proof at the real checkpoint)",
            root.join(ckpt.subfolder()).display()
        );
    }
    let dir = ensure_checkpoint(&root, ckpt).expect("checkpoint verify");
    let (_agent_cfg, enc_cfg) = config::load_checkpoint_configs(&dir, ckpt.subfolder())
        .expect("checkpoint configs");
    let mut map: HashMap<String, lane_weights::Weights> =
        lane_weights::load(&dir.join("model.safetensors"), ckpt.subfolder())
            .expect("weights load");
    let encoder =
        Encoder::from_map(&mut map, enc_cfg.clone(), ckpt.subfolder()).expect("encoder");
    let tok = Tok::from_dir(&dir, ckpt.subfolder()).expect("tokenizer");
    let backend = Cpu;
    encoder.warm(&backend);

    let n_stages = enc_cfg.layers + 1;
    let d = enc_cfg.hidden;
    let band = StabilityBand { cos: 0.999_9, l2: 0.05 };

    for prompt in PROMPTS {
        let ids = tok.encode(prompt).expect("encode");
        assert!(ids.len() >= 2, "fixture prompts must tokenize to >= 2 ids");
        let last = ids.len() - 1;

        // The capture: feed the tracker the LAST position of the case;
        // retain the same rows for the brute-force recompute.
        let mut tr = StabilityTracker::with_stage_capacity(n_stages, d);
        let mut retained: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n_stages);
        encoder
            .forward_capture(&backend, &ids, &[ids.len()], &mut |stage, row, state| {
                if row == last {
                    tr.observe(stage, state);
                    retained.push((stage_ordinal(stage), state[..d].to_vec()));
                }
            })
            .expect("forward_capture");

        // Stage structure: exactly n_stages stages, capture order.
        assert_eq!(tr.stages_observed(), n_stages, "stages = layers + 1");
        let ordinals: Vec<usize> = retained.iter().map(|(o, _)| *o).collect();
        let want: Vec<usize> = (0..n_stages).collect();
        assert_eq!(ordinals, want, "capture order ordinals");

        // G1 at the real checkpoint: the tracker's series must be
        // bit-identical to an independent recompute from the retained
        // states (same fold order, separately coded).
        let deltas = tr.deltas();
        assert_eq!(deltas.len(), n_stages - 1);
        for i in 0..deltas.len() {
            let (a, b) = (&retained[i].1, &retained[i + 1].1);
            let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..d {
                dot += a[j] * b[j];
                na += a[j] * a[j];
                nb += b[j] * b[j];
            }
            let mut sq = 0.0f32;
            for j in 0..d {
                sq += (a[j] - b[j]) * (a[j] - b[j]);
            }
            let denom = na.sqrt() * nb.sqrt();
            let want_cos = if sq == 0.0 {
                1.0
            } else if denom == 0.0 {
                0.0
            } else {
                dot / denom
            };
            assert_eq!(deltas[i].cos, want_cos, "cos parity at stage {i}");
            assert_eq!(deltas[i].l2, sq.sqrt(), "l2 parity at stage {i}");
        }

        // The depth read is well-formed: either a candidate in
        // 0..=n_stages-2, or an honest None with a named blocker.
        match tr.stabilization_depth(&band) {
            Some(depth) => {
                assert!(depth <= n_stages - 2, "depth below the final stage");
                println!("{prompt:?}: depth {depth}/{max} (band cos {c} l2 {l})", max = n_stages - 1, c = band.cos, l = band.l2);
            }
            None => {
                let (at, d) = tr.max_leaving_delta().expect("None names a blocker");
                println!(
                    "{prompt:?}: depth None — still moving at stage {at} (cos {:+.6} l2 {:+.6})",
                    d.cos, d.l2
                );
            }
        }
    }
}
