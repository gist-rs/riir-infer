//! Plan 611 S5 — the three-backend A/B (T7 op-layer unification): the
//! hand-tuned Metal MSL lane vs the portable CubeCL arm vs the CPU
//! reference, POSITION-BALANCED in one process. MEASUREMENT ONLY — the
//! verdict of plan 611 (which op layer survives), never a promotion by
//! itself.
//!
//! ```sh
//! cargo test --release -p riir-infer-laya \
//!     --features laya-riir-metal,laya-riir-cubecl \
//!     --test backend_ab -- --ignored --nocapture
//! ```
//!
//! Two tables, one process, three backends held side by side
//! (`RiirAgent::load_with_device` — no env round-trips):
//!
//! 1. **Encoder ladder**: `Encoder::forward` at the sequence lengths the
//!    lane actually serves (short suite cells to the long outlier, plus
//!    512), one `begin_pass` + forward + `download_into` per sample (the
//!    `metal_fold_ab` timing unit — the read-back is inside the timed
//!    region, so an async device cannot report unfinished work).
//! 2. **Agent cases**: `system_one` on the typed checkpoint. `5q_mid` is the
//!    typed-decisions shape of record, which each backend runs in its
//!    NATURAL posture (Metal packed; CubeCL and Cpu per-question loop,
//!    since `supports_packed_attention` is false there). `1q_ctrl` is the
//!    loop path on every backend.
//!
//! Order: each round runs all three arms, cycling through all six
//! permutations (rounds are a multiple of 6), so every arm sits in every
//! position equally often. Verdicts use the MEDIAN of per-round PAIRED
//! ratios, never the absolute p50s. Record the box state beside the numbers
//! (run riir-reflex `scripts/bench_preflight.sh` first). A loaded or
//! throttled box invalidates the table.
//!
//! Output agreement is asserted inline on the agent cases: every backend's
//! argmax (`choice`) and rounded probabilities must agree with the Cpu
//! lane's within the G5 budget. A timing win on drifting outputs is not a
//! win.
//!
//! PRE-REGISTERED verdict (plan 611 S5/S6; recorded in the S5 bench record
//! BEFORE any number was read):
//! - A "regime" is ≥ 3 ADJACENT ladder lengths, or one agent case. A win in
//!   a regime means a paired-ratio median < 1.00 with ≥ 9/12 wins in EVERY
//!   cell of it. One winning cell is never a regime.
//! - **Hand Metal lane**: deleted ONLY if CubeCL wins EVERY regime (the
//!   whole ladder AND both agent cases). Otherwise it stays the macOS
//!   default.
//! - **CubeCL arm**: DELETED if it loses to the CPU lane in every regime
//!   (cubecl/cpu median ≥ 1.00 everywhere): a GPU lane slower than the CPU
//!   lane has no portability value. Otherwise it is KEPT opt-in behind
//!   `laya-riir-cubecl` as the portability/CI arm. It is not default and
//!   not in any release set, and its G5 gate stays armed (the "maintained"
//!   half of plan 611's delete-if-slower-AND-unmaintained rule).
//! - **Promotion**: none from this table. A CubeCL regime win over Metal
//!   names a candidate for a future gated plan; it does not flip a default.
//! - The CPU lane is the reference oracle and is never deleted.
#![cfg(all(
    target_os = "macos",
    feature = "laya-riir-metal",
    feature = "laya-riir-cubecl"
))]

use riir_infer_laya::laya::config::{Checkpoint, load_checkpoint_configs};
use riir_infer_laya::laya::riir::agent::{DeviceKind, RiirAgent};
use riir_infer_laya::laya::riir::backend::{Backend, Cpu};
use riir_infer_laya::laya::riir::cubecl::CubeclBackend;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::metal::Metal;
use riir_infer_laya::laya::riir::weights as ckpt_weights;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};
use serde_json::{Value, json};

/// A multiple of 6 — every permutation of the three arms equally often.
const ROUNDS: usize = 12;

/// The served band (short suite cells ~10–60 tokens, typed ~100–120, the
/// banking77 long outlier ~317) plus 512 for the long end.
const SEQ_LENS: &[usize] = &[16, 54, 128, 317, 512];

/// The six orders of arms {0, 1, 2}.
const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
    [1, 0, 2],
    [0, 2, 1],
];

const ARMS: [&str; 3] = ["cpu", "metal", "cubecl"];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn quartiles(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 4], s[3 * s.len() / 4])
}

/// One row: absolute p50s per arm + the two paired verdict ratios
/// (cubecl/metal and cubecl/cpu — "wins" = rounds where CubeCL was faster).
fn report(label: &str, t: &[Vec<f64>; 3]) {
    let ratio = |num: usize, den: usize| -> Vec<f64> {
        t[num].iter().zip(&t[den]).map(|(a, b)| a / b).collect()
    };
    let (cm, cc) = (ratio(2, 1), ratio(2, 0));
    let (cm_lo, cm_hi) = quartiles(&cm);
    let (cc_lo, cc_hi) = quartiles(&cc);
    println!(
        "{label:>10}: p50 cpu {:9.3} · metal {:8.3} · cubecl {:8.3} ms │ cubecl/metal {:.3} \
         (IQR {:.3}–{:.3}, wins {}/{ROUNDS}) │ cubecl/cpu {:.3} (IQR {:.3}–{:.3}, wins {}/{ROUNDS})",
        median(t[0].clone()),
        median(t[1].clone()),
        median(t[2].clone()),
        median(cm.clone()),
        cm_lo,
        cm_hi,
        cm.iter().filter(|r| **r < 1.0).count(),
        median(cc.clone()),
        cc_lo,
        cc_hi,
        cc.iter().filter(|r| **r < 1.0).count(),
    );
}

/// The permutation-balanced paired loop: `run(arm)` → ms.
fn paired(mut run: impl FnMut(usize) -> f64) -> [Vec<f64>; 3] {
    for arm in 0..3 {
        run(arm);
        run(arm);
    }
    let mut t: [Vec<f64>; 3] = Default::default();
    for round in 0..ROUNDS {
        for &arm in &PERMS[round % 6] {
            let ms = run(arm);
            t[arm].push(ms);
        }
    }
    t
}

fn fwd_ms(enc: &Encoder, b: &dyn Backend, ids: &[u32]) -> f64 {
    b.begin_pass();
    let t = std::time::Instant::now();
    let h = enc.forward(b, ids).expect("forward");
    let mut out = vec![0f32; h.len()];
    b.download_into(&h, &mut out);
    t.elapsed().as_secs_f64() * 1e3
}

const INS_MID: &str = "You are the party's tactician. Read the battlefield state \
below and decide the safest advance. Weigh the reported enemy positions, the \
party's remaining supplies, the light left before dusk, and the terrain between \
here and the marked objective. Choose the corridor that keeps the wounded \
carriers out of the skirmish line while still closing the distance before \
nightfall.";

fn state() -> Value {
    json!({
        "location": "collapsed_watchtower",
        "party": {"hp": 87, "members": 6, "wounded": 2},
        "supplies": {"rations": 14, "torches": 5},
        "threats": [{"kind": "boar", "dist": 40}, {"kind": "bandit", "dist": 120}],
        "tick": 123_456
    })
}

/// The `metal_head_defer_ab` 5-q typed shape (same questions, so the two
/// harnesses' 5-q rows describe one workload).
fn mid_questions() -> Vec<(String, Value)> {
    vec![
        (
            "q_choice_a".into(),
            json!({"type": "choice", "instructions": INS_MID,
                   "criteria": {"north_ridge": "", "river_ford": "",
                                "old_road": "", "pine_hollow": ""}}),
        ),
        (
            "q_score_b".into(),
            json!({"type": "score", "instructions": INS_MID,
                   "criteria": ["scouted", "partly mapped", "unknown",
                                "dangerous", "lethal"]}),
        ),
        (
            "q_noul_c".into(),
            json!({"type": "noul", "instructions": INS_MID,
                   "criteria": {"true": "the ford is passable before dusk",
                                "false": "the ford floods by mid-afternoon"}}),
        ),
        (
            "q_choice_d".into(),
            json!({"type": "choice", "instructions": INS_MID,
                   "criteria": {"camp_here": "", "push_on": "", "retreat": ""}}),
        ),
        (
            "q_noul_e".into(),
            json!({"type": "noul", "instructions": INS_MID, "criteria": null}),
        ),
    ]
}

#[test]
#[ignore = "measurement-only three-backend A/B (plan 611 S5) — run with --ignored --nocapture on a quiet box"]
fn s5_backend_paired_ab() {
    // ── Table 1: the encoder ladder ──
    let ckpt = Checkpoint::English;
    let dir = ensure_checkpoint(&weights_root(), ckpt).expect("checkpoint present");
    let name = ckpt.subfolder();
    let (_agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name).expect("configs");
    let vocab = enc_cfg.vocab;
    let mut raw = ckpt_weights::load(&dir.join("model.safetensors"), name).expect("weights");
    let enc = Encoder::from_map(&mut raw, enc_cfg, name).expect("encoder");
    let cpu = Cpu;
    let metal = Metal::new().expect("metal");
    let cubecl = CubeclBackend::new().expect("cubecl");
    let backends: [&dyn Backend; 3] = [&cpu, &metal, &cubecl];
    for b in backends {
        enc.warm(b);
    }
    println!(
        "S5 BACKEND A/B — table 1 (encoder ladder): english checkpoint · cubecl runtime {} · \
         {ROUNDS} permutation-balanced rounds/shape · judge on the paired medians",
        cubecl.runtime_label()
    );
    for &seq in SEQ_LENS {
        let ids: Vec<u32> = (0..seq).map(|i| ((i * 7919) % vocab) as u32).collect();
        let t = paired(|arm| fwd_ms(&enc, backends[arm], &ids));
        report(&format!("seq {seq}"), &t);
    }
    drop(raw);

    // ── Table 2: the agent cases (typed checkpoint, natural postures) ──
    let root = weights_root();
    let kinds = [DeviceKind::Cpu, DeviceKind::Metal, DeviceKind::Cubecl];
    let agents: Vec<RiirAgent> = kinds
        .iter()
        .map(|k| {
            RiirAgent::load_with_device(&root, Checkpoint::TypedDecisions, *k)
                .expect("typed checkpoint loads at every posture")
        })
        .collect();
    for (a, want) in agents.iter().zip(ARMS) {
        assert_eq!(a.device(), want, "the posture must be the one requested");
    }
    let st = state();
    let probe = agents[0]
        .forward_question(&st, &mid_questions()[0].1)
        .expect("probe forward");
    println!(
        "S5 BACKEND A/B — table 2 (agent system_one): typed checkpoint · mid question seq_len {} · \
         metal packed, cubecl + cpu per-question loop",
        probe.seq_len
    );
    let cases: [(&str, Vec<(String, Value)>); 2] = [
        ("5q_mid", mid_questions()),
        ("1q_ctrl", mid_questions().into_iter().take(1).collect()),
    ];
    for (label, questions) in cases {
        // Output agreement vs the Cpu reference (the G5 1e-3 budget on the
        // rounded probabilities; exact on the argmax) before any timing.
        let answers: Vec<_> = agents
            .iter()
            .map(|a| a.system_one(&st, &questions).expect("system_one"))
            .collect();
        for (arm, got) in answers.iter().enumerate().skip(1) {
            for (x, y) in answers[0].iter().zip(got) {
                assert_eq!(x.choice, y.choice, "{label}/{}: argmax diverges from cpu", ARMS[arm]);
                for ((kx, px), (ky, py)) in x.probabilities.iter().zip(&y.probabilities) {
                    assert_eq!(kx, ky, "{label}: option order");
                    assert!(
                        (px - py).abs() <= 1e-3,
                        "{label}/{} {}: p drift {:.2e} > 1e-3",
                        ARMS[arm],
                        x.qid,
                        (px - py).abs()
                    );
                }
            }
        }
        let t = paired(|arm| {
            let t0 = std::time::Instant::now();
            let _ = agents[arm].system_one(&st, &questions).expect("system_one");
            t0.elapsed().as_secs_f64() * 1e3
        });
        report(label, &t);
    }
}
