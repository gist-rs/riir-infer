//! Issue 020 T11 — the FOLD rungs' paired A/B, POSITION-BALANCED in one
//! process (the gates' own rule: never a first-arm-first number).
//! MEASUREMENT ONLY — the promotion gate for `LAYA_METAL_FOLD_RES` /
//! `LAYA_METAL_FOLD_GLU` (both landed default-OFF).
//!
//! ```sh
//! cargo test --release --features laya-riir-metal --test metal_fold_ab \
//!     -- --ignored --nocapture
//! ```
//!
//! Three arms against the unfused control, selected by `AB_FOLD`:
//! `res` / `glu` / `both` (default). Each round runs BOTH backends on the
//! SAME ids with the order alternating round to round; the verdict is the
//! MEDIAN of per-round on/off ratios. Pairing cancels drift the box adds
//! between rounds (contention that moves both arms of a pair together),
//! which sequential runs of two processes cannot. Record box state beside
//! the numbers — a loaded or throttled box invalidates the whole table
//! (run `scripts/bench_preflight.sh` in the reflex repo first).
//!
//! Shapes: the stable-band suite lengths (the short-sequence cells the
//! every-published-cell bar still loses) plus the crossover shapes, so the
//! table shows where the fold stops paying. The fold is a NO-OP above the
//! split rule's reach (m > 256 for the widest shape) — the 1.000 rows are
//! the control that the arm wires to nothing.
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::{Checkpoint, load_checkpoint_configs};
use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights as ckpt_weights;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

const ROUNDS: usize = 24;

/// The losing band's own lengths (bench 035: banking77 ~317 is the long
/// outlier; ag_news/sst5/emotion/massive run ~10–60 tokens; typed ~100–120)
/// plus the split-rule crossover shapes.
const SEQ_LENS: &[usize] = &[10, 24, 32, 46, 54, 80, 96, 106, 128, 188, 317];

fn fwd_ms(enc: &Encoder, b: &Metal, ids: &[u32]) -> f64 {
    b.begin_pass();
    let t = std::time::Instant::now();
    let h = enc.forward(b, ids).expect("forward");
    let mut out = vec![0f32; h.len()];
    b.download_into(&h, &mut out);
    t.elapsed().as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "measurement-only A/B (issue 020 T11 fold rungs) — run with --ignored --nocapture on a quiet box"]
fn t11_fold_paired_ab() {
    let arm = std::env::var("AB_FOLD").unwrap_or_else(|_| "both".into());
    let (res, glu) = match arm.as_str() {
        "res" => (true, false),
        "glu" => (false, true),
        "both" => (true, true),
        other => panic!("AB_FOLD must be res|glu|both, got {other}"),
    };
    let dir = ensure_checkpoint(&weights_root(), Checkpoint::English).expect("checkpoint present");
    let name = Checkpoint::English.subfolder();
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");
    let mut raw = ckpt_weights::load(&dir.join("model.safetensors"), name).expect("weights");
    let enc = Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder");
    let off = Metal::new().expect("metal").with_folds(false, false);
    let on = Metal::new().expect("metal").with_folds(res, glu);
    enc.warm(&off);
    enc.warm(&on);
    println!(
        "t11 FOLD A/B: AB_FOLD={arm} (res {res} glu {glu}) vs unfused · {ROUNDS} paired rounds/shape · \
         judge on the ratios' medians, never the absolute p50s"
    );
    let vocab = 50368usize;
    for seq in SEQ_LENS {
        let ids: Vec<u32> = (0..*seq).map(|i| ((i * 7919) % vocab) as u32).collect();
        for _ in 0..2 {
            fwd_ms(&enc, &off, &ids);
            fwd_ms(&enc, &on, &ids);
        }
        let before = on.fold_dispatches();
        let (mut a, mut b, mut r) = (Vec::new(), Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            let (x, y) = if round % 2 == 0 {
                let x = fwd_ms(&enc, &off, &ids);
                (x, fwd_ms(&enc, &on, &ids))
            } else {
                let y = fwd_ms(&enc, &on, &ids);
                (fwd_ms(&enc, &off, &ids), y)
            };
            a.push(x);
            b.push(y);
            r.push(y / x);
        }
        let folds = (on.fold_dispatches() - before) / ROUNDS as u64;
        let mut rs = r.clone();
        rs.sort_by(f64::total_cmp);
        println!(
            "seq {seq:>4}: off p50 {:7.3} ms · on p50 {:7.3} ms · paired on/off median {:.3} \
             (IQR {:.3}–{:.3}) · wins {}/{} · fold epilogues/fwd {folds}",
            median(a),
            median(b),
            median(r.clone()),
            rs[ROUNDS / 4],
            rs[3 * ROUNDS / 4],
            r.iter().filter(|v| **v < 1.0).count(),
            ROUNDS
        );
    }
}
