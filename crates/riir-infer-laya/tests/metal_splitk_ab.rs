//! Issue 020 T11 — split-K A/B, POSITION-BALANCED and PAIRED in one process
//! (the gates' own rule: never a first-arm-first number). MEASUREMENT ONLY:
//!
//! ```sh
//! cargo test --release --features laya-riir-metal --test metal_splitk_ab \
//!     -- --ignored --nocapture
//! ```
//!
//! Two backends — `Metal::with_splitk(false, …)` (off) and `(true, …)` — run
//! the SAME encoder on the SAME ids; each round runs both, the order
//! alternating round to round, and the verdict is the MEDIAN of per-round
//! `on / off` ratios. Pairing cancels drift the box adds between rounds
//! (contention that moves both arms of a pair together), which sequential
//! runs of two processes cannot. Record box state beside the numbers.
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::{Checkpoint, load_checkpoint_configs};
use riir_infer_laya::laya::riir::backend::Backend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::metal::{Metal, SplitRule};
use riir_infer_laya::laya::riir::weights as ckpt_weights;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

const ROUNDS: usize = 24;

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
#[ignore = "measurement-only A/B (issue 020 T11) — run with --ignored --nocapture"]
fn t11_splitk_paired_ab() {
    let max_tgs: u64 = std::env::var("AB_MAXTGS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let ckpt = Checkpoint::English;
    let dir = ensure_checkpoint(&weights_root(), ckpt).expect("checkpoint present");
    let name = ckpt.subfolder();
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");
    let mut raw = ckpt_weights::load(&dir.join("model.safetensors"), name).expect("weights");
    let enc = Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder");
    // AB_BASE=ceiling compares DEFAULT against the first rule instead of off.
    let base = match std::env::var("AB_BASE").as_deref() {
        Ok("ceiling") => SplitRule::TG_CEILING_ONLY,
        _ => SplitRule {
            on: false,
            ..SplitRule::DEFAULT
        },
    };
    let off = Metal::with_split_rule(base).expect("metal base");
    let on = Metal::with_split_rule(SplitRule {
        max_tgs,
        ..SplitRule::DEFAULT
    })
    .expect("metal on");
    enc.warm(&off);
    enc.warm(&on);
    println!(
        "t11 A/B: base {:?} vs DEFAULT (max_tgs {max_tgs}) · {ROUNDS} paired rounds/shape",
        std::env::var("AB_BASE").unwrap_or_else(|_| "off".into())
    );
    let vocab = 50368usize;
    for seq in [24usize, 46, 54, 80, 106, 140, 188, 256, 317, 512] {
        let ids: Vec<u32> = (0..seq).map(|i| ((i * 7919) % vocab) as u32).collect();
        for _ in 0..2 {
            fwd_ms(&enc, &off, &ids);
            fwd_ms(&enc, &on, &ids);
        }
        let before = on.splitk_dispatches();
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
        let splits = (on.splitk_dispatches() - before) / ROUNDS as u64;
        let mut rs = r.clone();
        rs.sort_by(f64::total_cmp);
        println!(
            "seq {seq:>4}: off p50 {:7.2} ms · on p50 {:7.2} ms · paired on/off median {:.3} \
             (IQR {:.3}–{:.3}) · wins {}/{} · split GEMMs/fwd {splits}",
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
