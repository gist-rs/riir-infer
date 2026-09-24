//! The packed-path chain-aliasing gate: two same-shape questions in ONE
//! case must answer bit-identically to the per-question loop, at the RAW
//! `act_probabilities` bits.
//!
//! The rounded [`Answer`] envelope cannot carry this gate: the aliasing
//! hazard it pins (per-question host buffers malloc-reusing an address
//! within the case's single chain epoch — `(host ptr, len, epoch)` keys
//! then alias the previous question's device buffers) can hide inside
//! 4-decimal rounding. It was measured live: a shared CLS row plus a stale
//! `act_in` for question 2, raw-bit divergence from the loop path, in a
//! layout the published fixtures never reproduced. The fix — per-question
//! slabs and [`HeadScratch`]es allocated up front and kept alive for the
//! whole case — makes every logical buffer own its address for the epoch.
//!
//! Requires the REAL typed-decisions checkpoint (`LAYA_WEIGHTS_DIR`, the
//! consumer cache layout) — synthetic weights cannot reach the aliasing
//! layout. Skips LOUD when the weights are absent; never a silent pass.
//!
//! Run: `LAYA_DEVICE=metal LAYA_WEIGHTS_DIR=~/.cache/riir-reflex/laya \
//!       cargo test --release -p riir-infer-laya --features laya-riir-metal \
//!       --test packed_same_shape_gate`
#![cfg(all(target_os = "macos", feature = "laya-riir-metal"))]

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::agent::{RiirAgent, PACKED_ACT_BITS};
use riir_infer_laya::laya::types::Forward;

fn weights_root() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("LAYA_WEIGHTS_DIR") {
        return Some(std::path::PathBuf::from(dir));
    }
    std::env::var_os("HOME").map(|h| {
        std::path::PathBuf::from(h).join(".cache/riir-reflex/laya")
    })
}

#[test]
fn packed_same_shape_matches_loop_raw_bits() {
    let Some(root) = weights_root() else {
        eprintln!(
            "SKIP LOUD: no checkpoint weights (set LAYA_WEIGHTS_DIR, or install \
             the consumer cache at ~/.cache/riir-reflex/laya) — the aliasing \
             gate needs the real typed-decisions checkpoint, never a green zero"
        );
        return;
    };
    if !root.join("typed").join("model.safetensors").exists() {
        panic!(
            "SKIP LOUD (as a failure): LAYA_WEIGHTS_DIR={} has no typed/ \
             checkpoint — point it at the real cache; an absent subject must \
             not read as a pass",
            root.display()
        );
    }
    let agent = RiirAgent::load(&root, Checkpoint::TypedDecisions).expect("agent load");
    let state = serde_json::json!({
        "topic": "packed parity",
        "text": "the quick brown fox jumps over the lazy dog near the river bank at dawn"
    });
    // Same tokenized shape (same pad count, same option-word length ⇒ same
    // marker count), different content — exactly the pair class whose
    // per-question buffers are the same size and therefore the malloc-reuse
    // candidates.
    let mk = |pad: usize, opt_word: &str| {
        serde_json::json!({
            "type": "choice",
            "instructions": format!(
                "Pick the option that best matches the state. {}",
                "filler ".repeat(pad)
            ),
            "criteria": [opt_word, "other"]
        })
    };
    let qs = vec![
        ("q1".to_string(), mk(30, "alpha")),
        ("q2".to_string(), mk(30, "omega")),
    ];

    // Warmup (kernel compile + scratch pool fill), then the packed case with
    // a cleared capture, then the loop reference question by question.
    let _ = agent.system_one(&state, &qs).expect("warmup");
    PACKED_ACT_BITS.lock().expect("act bits poison").clear();
    agent.system_one(&state, &qs).expect("packed");
    let packed_bits: Vec<([u32; 2], Vec<u32>)> =
        PACKED_ACT_BITS.lock().expect("act bits poison").clone();
    assert_eq!(packed_bits.len(), qs.len(), "raw-bits capture arity");

    for (i, (_, qdef)) in qs.iter().enumerate() {
        let q = riir_infer_laya::laya::tokenize::to_internal(qdef).expect("internal question");
        let looped: Forward = agent
            .forward_internal(&state, &q)
            .expect("loop forward");
        let loop_act: [u32; 2] = [
            looped.act_probabilities[0].to_bits(),
            looped.act_probabilities[1].to_bits(),
        ];
        let loop_logits: Vec<u32> =
            looped.logits.iter().map(|l| l.to_bits()).collect();
        let (packed_act, packed_logits) = &packed_bits[i];
        assert_eq!(
            packed_act, &loop_act,
            "question {} act_probabilities diverge from the loop at the raw-bit \
             level — the chain-aliasing hazard has regressed (packed {packed_act:?} \
             vs loop {loop_act:?}; logits packed {packed_logits:?} vs loop {loop_logits:?})",
            i + 1
        );
        assert_eq!(
            packed_logits, &loop_logits,
            "question {} scorer logits diverge from the loop at the raw-bit level",
            i + 1
        );
    }
}
