//! The prefix-state coupling gate (riir-reflex issue 054 Part 2, verdict
//! record): the laya encoder CANNOT serve an open-jev-fast-style
//! prefix-state handoff (`fla_mode="state"`) — the exact-continuation
//! premise does not transfer. Three architectural grounds, all structural:
//!
//! 1. **Bidirectional attention.** The encoder is ModernBERT/mmBERT
//!    (`encoder_config.json`), not GDN: `attention_forward_default` has no
//!    causal mask (the only mask is the symmetric sliding-window band
//!    `|q − k| ≤ window`), so EVERY row attends every row of its sequence.
//!    A shared state span's hidden rows are therefore functions of the
//!    per-question head span too — they cannot be encoded once and reused
//!    exactly across the questions of a case. GDN's delta-rule recurrence
//!    is causal: its final prefix state IS the exact continuation input.
//!    ModernBERT has no per-position state a suffix could resume from.
//! 2. **Per-question positions.** `build_sequence` renders
//!    `[CLS] {t} question: {ins} [SEP] [MASK] opt… [SEP] {state} [SEP]` —
//!    the state sits AFTER the per-question head span, at a different RoPE
//!    offset in every question's sequence (head lengths differ). A single
//!    shared encode cannot even reproduce the positional geometry.
//! 3. **Per-question truncation.** The state is truncated to
//!    `room = max_len − ids.len() − 1`, which varies per question — the
//!    "shared prefix" is not guaranteed identical tokens across questions.
//!
//! The tests hold the claim honest in BOTH directions: the coupling must
//! MEASURE above the CPU GEMM reduction-order drift budget (1e-5, the
//! `packed_forward_equiv` class). A reading at or below that budget would
//! mean the shared-span rows are suffix-independent — the exact-handoff
//! premise would transfer and this gate's refutation would be WRONG, so
//! the assert fails and the record must be revisited.
//!
//! Arm 1 (synthetic, hermetic): 4-layer mixed full/sliding geometry, a
//! shared prefix + two different tails — the shared-prefix rows must drift
//! beyond 1e-4. Plus a determinism control (the same ids forward
//! bit-identically twice, so the measured drift is attention coupling, not
//! nondeterminism).
//!
//! Arm 2 (real typed checkpoint, skip-loud without weights;
//! `PREFIX_PROBE_REQUIRE_DATA=1` turns the skip into a failure): two
//! choice questions against ONE shared state through
//! [`RiirAgent::encode_question`] — the shared state-span rows (the
//! longest common id suffix) must drift beyond 1e-4. CPU posture,
//! explicit — never the GPU (the probe is a gate, not a perf lane).
//!
//! Run: `cargo test -p riir-infer-laya --features laya-riir --test prefix_state_coupling -- --nocapture`
#![cfg(feature = "laya-riir")]

use std::collections::HashMap;

use riir_infer_laya::laya::config::{Checkpoint, EncoderConfig};
use riir_infer_laya::laya::riir::agent::{DeviceKind, RiirAgent};
use riir_infer_laya::laya::riir::backend::Cpu;
use riir_infer_laya::laya::riir::encoder::Encoder;
use riir_infer_laya::laya::riir::weights::Weights;
use riir_infer_laya::laya::weights::weights_root;

/// The coupling gate: strictly above the packed-equivalence drift budget
/// (1e-5) with headroom — 1e-4 is 10× the budget and three orders below
/// any real coupling on trained weights.
const COUPLING_GATE: f32 = 1e-4;

/// The packed-equivalence budget the gate must EXCEED to mean anything
/// (named for the failure message — a reading at or under it refutes the
/// coupling claim instead of proving it).
const DRIFT_BUDGET: f32 = 1e-5;

/// Deterministic xorshift fill (the `packed_forward_equiv` pattern — the
/// values only need coverage, not statistical quality).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn weights(shape: Vec<usize>, seed: u64) -> Weights {
    let n: usize = shape.iter().product();
    Weights {
        shape,
        data: riir_infer_laya::laya::riir::weights::WeightData::F32(fill(n, seed)),
    }
}

/// d=64 (hd 64, one head), 4 layers (full / sliding / sliding / full) —
/// full layers 0 and 3 couple every position, the sliding layers couple
/// within window 4. Window is intentionally smaller than the shared span
/// so BOTH mask classes are exercised by the coupling.
fn test_config() -> EncoderConfig {
    EncoderConfig {
        hidden: 64,
        layers: 4,
        heads: 1,
        intermediate: 32,
        vocab: 97,
        eps: 1e-5,
        global_every: 3,
        local_attention: 8, // window 4
        rope_theta_full: 10_000.0,
        rope_theta_slide: 16_000.0,
        sliding: vec![false, true, true, false],
        hidden_activation: "gelu".into(),
    }
}

fn test_encoder() -> Encoder {
    let cfg = test_config();
    let d = cfg.hidden;
    let i = cfg.intermediate;
    let mut map: HashMap<String, Weights> = HashMap::new();
    let mut seed = 1u64;
    let mut next = || {
        seed += 7;
        seed
    };
    map.insert(
        "encoder.embeddings.tok_embeddings.weight".into(),
        weights(vec![cfg.vocab, d], next()),
    );
    map.insert(
        "encoder.embeddings.norm.weight".into(),
        weights(vec![d], next()),
    );
    for idx in 0..cfg.layers {
        if idx != 0 {
            map.insert(
                format!("encoder.layers.{idx}.attn_norm.weight"),
                weights(vec![d], next()),
            );
        }
        map.insert(
            format!("encoder.layers.{idx}.attn.Wqkv.weight"),
            weights(vec![3 * d, d], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.attn.Wo.weight"),
            weights(vec![d, d], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.mlp.Wi.weight"),
            weights(vec![2 * i, d], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.mlp.Wo.weight"),
            weights(vec![d, i], next()),
        );
        map.insert(
            format!("encoder.layers.{idx}.mlp_norm.weight"),
            weights(vec![d], next()),
        );
    }
    map.insert("encoder.final_norm.weight".into(), weights(vec![d], next()));
    Encoder::from_map(&mut map, cfg, "prefix-coupling-test").expect("synthetic encoder loads")
}

fn token_ids(total: usize, vocab: usize) -> Vec<u32> {
    fill(total, 4242)
        .into_iter()
        .map(|v| ((v + 1.0) * 0.5 * (vocab - 1) as f32) as u32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// ARM 1 — the mechanism, hermetic: rows of a SHARED prefix must depend on
/// the tail that follows them (bidirectional attention), at positions the
/// two sequences agree on. A tail-independence reading (drift at or below
/// the GEMM reduction-order budget) FAILS this test — it would mean the
/// open-jev-fast exact-handoff premise transfers, contradicting the record.
#[test]
fn shared_prefix_rows_are_suffix_dependent_synthetic() {
    let enc = test_encoder();
    let b = Cpu;
    let shared = token_ids(16, 97);
    let tail_a = token_ids(8, 97);
    let tail_b: Vec<u32> = token_ids(8, 97).iter().map(|t| (t + 37) % 97).collect();
    assert_ne!(tail_a, tail_b, "the two tails must differ");

    let mut ids_a = shared.clone();
    ids_a.extend_from_slice(&tail_a);
    let mut ids_b = shared.clone();
    ids_b.extend_from_slice(&tail_b);

    // Determinism control: the same ids forward bit-identically twice, so
    // any drift measured below is attention coupling, not nondeterminism.
    let a1 = enc.forward(&b, &ids_a).expect("forward a1 runs");
    let a2 = enc.forward(&b, &ids_a).expect("forward a2 runs");
    assert_eq!(a1, a2, "the forward must be deterministic on fixed ids");

    let fb = enc.forward(&b, &ids_b).expect("forward b runs");
    let d = enc.hidden_dim();
    let off = shared.len() * d;
    let drift = max_abs_diff(&a1[..off], &fb[..off]);
    println!(
        "synthetic shared-prefix drift: {drift:.3e} over {} rows (gate > {COUPLING_GATE:e}, \
         drift budget {DRIFT_BUDGET:e})",
        shared.len()
    );
    assert!(
        drift > COUPLING_GATE,
        "shared-prefix rows read tail-INDEPENDENT (drift {drift:.3e} ≤ {COUPLING_GATE:e}): \
         the bidirectional-coupling premise behind the issue 054 Part 2 refutation does not \
         hold on this op stream — the exact-handoff lead may transfer after all; re-open the \
         record (a drift at or below {DRIFT_BUDGET:e} would be GEMM reduction-order noise, \
         not independence)"
    );
}

/// Longest common suffix length of two id streams (the shared state span
/// sits at the END of every `build_sequence` render, before the final
/// `[SEP]`).
fn common_suffix_len(a: &[u32], b: &[u32]) -> usize {
    let mut k = 0;
    while k < a.len() && k < b.len() && a[a.len() - 1 - k] == b[b.len() - 1 - k] {
        k += 1;
    }
    k
}

/// ARM 2 — the real typed checkpoint, the suite the issue names: two
/// distinct choice questions against ONE shared case state. The state
/// span's hidden rows must differ between the two forwards by more than
/// the coupling gate — the measured form of ground 1 (and the span
/// offsets differ, the measured form of ground 2).
#[test]
fn shared_state_span_is_suffix_dependent_real_typed() {
    let root = weights_root();
    let weights_file = root.join("typed").join("model.safetensors");
    if !weights_file.is_file() {
        if std::env::var("PREFIX_PROBE_REQUIRE_DATA").as_deref() == Ok("1") {
            panic!(
                "PREFIX_PROBE_REQUIRE_DATA=1 but no typed checkpoint at {}",
                weights_file.display()
            );
        }
        eprintln!(
            "SKIP (loud): no typed checkpoint at {} — arm 2 unmeasured on this box \
             (set PREFIX_PROBE_REQUIRE_DATA=1 to fail instead)",
            weights_file.display()
        );
        return;
    }

    let agent = RiirAgent::load_with_device(&root, Checkpoint::TypedDecisions, DeviceKind::Cpu)
        .expect("typed checkpoint loads on the CPU posture");

    // One case state, two genuinely different questions (different
    // instructions AND different options — head spans differ in both
    // content and length). Sized to sit well under the state-truncation
    // room, so both questions carry the IDENTICAL full state span.
    let state = serde_json::json!({
        "case_id": "wf-2026-09-30-0042",
        "workflow": "claims_processing",
        "stage": "adjudication",
        "events": [
            {"tick": 812, "kind": "intake", "actor": "adjuster_7", "detail": "claim 8841 opened, documents complete"},
            {"tick": 845, "kind": "review", "actor": "adjuster_7", "detail": "coverage confirmed, deductible applied"},
            {"tick": 901, "kind": "flag", "actor": "audit_bot", "detail": "invoice total mismatch: 4120.00 vs 3980.00"},
            {"tick": 933, "kind": "query", "actor": "adjuster_7", "detail": "vendor contacted for corrected invoice"},
            {"tick": 977, "kind": "response", "actor": "vendor_ops", "detail": "corrected invoice 3980.00 received and attached"}
        ],
        "balances": {"reserved": 3980.00, "paid": 0.00, "expense": 220.00},
        "notes": "sluice gate inspection scheduled; no prior claims on policy; fraud score 0.07",
        "sla": {"hours_remaining": 36, "priority": "standard"}
    });
    let q1 = serde_json::json!({
        "type": "choice",
        "instructions": "Pick the best next action for the current workflow state.",
        "criteria": ["approve_claim", "request_more_evidence", "escalate_to_senior"]
    });
    let q2 = serde_json::json!({
        "type": "choice",
        "instructions": "Choose the highest-priority follow-up given the recorded events.",
        "criteria": ["close_case", "notify_policyholder", "reconcile_invoice_totals"]
    });

    let enc1 = agent.encode_question(&state, &q1).expect("q1 encodes");
    let enc2 = agent.encode_question(&state, &q2).expect("q2 encodes");

    let (ids1, m1) = agent.tokenize_question(&state, &q1).expect("q1 tokenizes");
    let (ids2, m2) = agent.tokenize_question(&state, &q2).expect("q2 tokenizes");
    assert_eq!(
        ids1.len(),
        enc1.seq_len,
        "tokenize_question must agree with encode_question's stream (q1)"
    );
    assert_eq!(
        ids2.len(),
        enc2.seq_len,
        "tokenize_question must agree with encode_question's stream (q2)"
    );

    let k = common_suffix_len(&ids1, &ids2);
    let d = enc1.d;
    println!(
        "typed lane spans: seq1 {} (markers {}), seq2 {} (markers {}), shared suffix {k} tokens; \
         span offsets {} vs {} (per-question RoPE positions differ)",
        ids1.len(),
        m1.len(),
        ids2.len(),
        m2.len(),
        ids1.len() - k,
        ids2.len() - k
    );
    assert!(
        k >= 32,
        "the shared state span collapsed (k = {k}) — the probe input is degenerate, \
         not the architecture; widen the state or shorten the questions"
    );

    let span1 = &enc1.hidden[(ids1.len() - k) * d..];
    let span2 = &enc2.hidden[(ids2.len() - k) * d..];
    let drift = max_abs_diff(span1, span2);
    println!(
        "real typed shared-state drift: {drift:.3e} over {k} rows (gate > {COUPLING_GATE:e}, \
         drift budget {DRIFT_BUDGET:e})"
    );
    assert!(
        drift > COUPLING_GATE,
        "the real typed checkpoint reads the shared state span QUESTION-INDEPENDENT \
         (drift {drift:.3e} ≤ {COUPLING_GATE:e}): ground 1 of the issue 054 Part 2 refutation \
         fails on the shipped checkpoint — the exact-handoff lead may transfer after all; \
         re-open the record (a drift at or below {DRIFT_BUDGET:e} would be GEMM \
         reduction-order noise, not independence)"
    );
}
