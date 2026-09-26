//! Issue 020 T12 — the packed-head DEFER posture's paired A/B,
//! POSITION-BALANCED in one process (the gates' own rule: never a
//! first-arm-first number). MEASUREMENT ONLY — the promotion gate for the
//! `LAYA_HEAD_DEFER=1` rung (landed default-OFF).
//!
//! ```sh
//! LAYA_DEVICE=metal cargo test --release --features laya-riir-metal \
//!     --test metal_head_defer_ab -- --ignored --nocapture
//! ```
//!
//! Two arms against the composed control, toggled per round through the
//! agent's `set_head_defer_override` seam (the fold A/B's `with_folds`
//! pattern — one process, alternating order round to round, verdict =
//! the MEDIAN of per-round on/off ratios). Pairing cancels the box drift
//! that moves both arms of a round together; record box state beside the
//! numbers (run `scripts/bench_preflight.sh` in the reflex repo first —
//! a loaded or throttled box invalidates the whole table).
//!
//! Shapes:
//! - `5q_mid` — five typed-geometry questions: the typed_decisions shape
//!   of record (every one of its cases is exactly 5 questions).
//! - `5q_short` — the same questions with ~10x shorter instructions: the
//!   short end of the multi-q band (the defer's saving scales with the
//!   drain count, not the tokens, so the short end bounds it).
//! - `1q_ctrl` — one question: `packed_eligible` excludes it, so both
//!   postures run the identical loop path — the wiring control (the
//!   toggle must move nothing; median 1.000 ± noise).
//!
//! PRE-REGISTERED verdict (record it beside the numbers either way):
//! - PROMOTE to default-on: BOTH 5-q medians ≤ 0.99 with ≥ 18/24 wins
//!   each, and the 1-q control median within [0.99, 1.01].
//! - CLOSE as measured-thin (keep default-off): either 5-q median in
//!   (0.99, 1.01), or a win fraction < 18/24 with no median > 1.01.
//! - Any 5-q median > 1.01: the rung regresses — stays default-off and
//!   the knob is dead weight (note it for removal).
//!
//! Bit-identity is asserted inline per shape: both arms' PACKED_ACT_BITS
//! captures must be byte-equal (the same seam `packed_same_shape_gate`
//! reads) and the rounded Answers must match — a timing win on drifting
//! outputs is not a win.
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::agent::{RiirAgent, PACKED_ACT_BITS};
use riir_infer_laya::laya::weights::weights_root;
use serde_json::{json, Value};

const ROUNDS: usize = 24;

/// ~3.4 words per token on this tokenizer's domain text; the mid
/// instructions land each question near the typed suite's p50 geometry
/// (~1xx tokens including state + options), the short ones near the
/// band's short end.
const INS_MID: &str = "You are the party's tactician. Read the battlefield state \
below and decide the safest advance. Weigh the reported enemy positions, the \
party's remaining supplies, the light left before dusk, and the terrain between \
here and the marked objective. Choose the corridor that keeps the wounded \
carriers out of the skirmish line while still closing the distance before \
nightfall.";
const INS_SHORT: &str = "Pick the safest corridor.";

fn state() -> Value {
    json!({
        "location": "collapsed_watchtower",
        "party": {"hp": 87, "members": 6, "wounded": 2},
        "supplies": {"rations": 14, "torches": 5},
        "threats": [{"kind": "boar", "dist": 40}, {"kind": "bandit", "dist": 120}],
        "tick": 123_456
    })
}

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

fn short_questions() -> Vec<(String, Value)> {
    mid_questions()
        .into_iter()
        .map(|(qid, mut q)| {
            q["instructions"] = json!(INS_SHORT);
            (qid, q)
        })
        .collect()
}

fn run_case(agent: &RiirAgent, state: &Value, questions: &[(String, Value)]) -> f64 {
    let t = std::time::Instant::now();
    let _ = agent.system_one(state, questions).expect("system_one");
    t.elapsed().as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn captured_bits() -> Vec<([u32; 2], Vec<u32>)> {
    PACKED_ACT_BITS.lock().unwrap().clone()
}

#[test]
#[ignore = "measurement-only A/B (issue 020 T12 defer rung) — run with --ignored --nocapture on a quiet box"]
fn t12_head_defer_paired_ab() {
    let mut agent = RiirAgent::load(&weights_root(), Checkpoint::TypedDecisions)
        .expect("typed checkpoint present");
    assert_eq!(agent.device(), "metal", "run with LAYA_DEVICE=metal — a CPU \
        posture measures a different lane and the table would be fiction");
    let st = state();

    // Shape probe (the loop path — one throwaway forward reads seq_len).
    let probe = agent
        .forward_question(&st, &mid_questions()[0].1)
        .expect("probe forward");
    println!(
        "t12 DEFER A/B: typed checkpoint · metal · {ROUNDS} paired rounds/shape · \
         mid question seq_len {} · judge on the ratios' medians, never the absolute p50s",
        probe.seq_len
    );

    let cases: [(&str, Vec<(String, Value)>); 3] = [
        ("5q_mid", mid_questions()),
        ("5q_short", short_questions()),
        ("1q_ctrl", mid_questions().into_iter().take(1).collect()),
    ];
    for (name, questions) in cases {
        // One verification pass per arm BEFORE the timing loop (the timing
        // loop re-runs the same deterministic case; the gates pin
        // bit-identity separately). Answers compare through their Debug
        // forms (the envelope carries no PartialEq); the raw-bit seam is
        // the stronger check and must be byte-equal.
        let runs: [(Vec<_>, Vec<_>); 2] = [false, true].map(|defer| {
            agent.set_head_defer_override(Some(defer));
            PACKED_ACT_BITS.lock().unwrap().clear();
            let a = agent.system_one(&st, &questions).expect("system_one");
            (a, captured_bits())
        });
        agent.set_head_defer_override(None);
        assert_eq!(
            format!("{:?}", runs[0].0),
            format!("{:?}", runs[1].0),
            "{name}: the two postures must produce identical Answers"
        );
        assert_eq!(
            runs[0].1, runs[1].1,
            "{name}: PACKED_ACT_BITS must be byte-equal across the postures"
        );

        // Warm both postures (pipeline + alloc paths), then the paired
        // rounds with the order alternating round to round.
        for defer in [false, true] {
            agent.set_head_defer_override(Some(defer));
            run_case(&agent, &st, &questions);
            run_case(&agent, &st, &questions);
        }
        let (mut a, mut b, mut r) = (Vec::new(), Vec::new(), Vec::new());
        for round in 0..ROUNDS {
            let (x, y) = if round % 2 == 0 {
                agent.set_head_defer_override(Some(false));
                let x = run_case(&agent, &st, &questions);
                agent.set_head_defer_override(Some(true));
                (x, run_case(&agent, &st, &questions))
            } else {
                agent.set_head_defer_override(Some(true));
                let y = run_case(&agent, &st, &questions);
                agent.set_head_defer_override(Some(false));
                (run_case(&agent, &st, &questions), y)
            };
            a.push(x);
            b.push(y);
            r.push(y / x);
        }
        agent.set_head_defer_override(None);
        let mut rs = r.clone();
        rs.sort_by(f64::total_cmp);
        println!(
            "{name:>9}: off p50 {:7.3} ms · on p50 {:7.3} ms · paired on/off median {:.3} \
             (IQR {:.3}–{:.3}) · wins {}/{}",
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
