//! Issue 021 — the AGENT-level repeat probe: the full `RiirAgent::system_one`
//! on the real english checkpoint over 12 real banking77 cases (the full
//! 77-key universe), 30 rounds, comparing the complete serde render of the
//! answers. This is the harness repeat check's exact question, outside the
//! harness: ops bit-stable (cuda_repeat_probe), the packed encoder forward
//! bit-stable (cuda_packed_repeat_probe) — if THIS flips, the composition
//! between them (head reads, act chain, answer envelope) carries the
//! wobble; if it is green, the harness-side flow (its request shape, its
//! per-case loop, its state JSON) differs from this probe's in the load-
// bearing way.
//!
//! SKIPs loud when the english checkpoint is absent (the NDB_BIN posture).

#![cfg(all(not(target_os = "macos"), feature = "laya-riir"))]

use std::sync::{Mutex, MutexGuard, OnceLock};

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::RiirAgent;
use riir_infer_laya::laya::weights::weights_root;

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const B77_KEYS: [&str; 77] = [
    "Refund not showing up",
    "activate my card",
    "age limit",
    "apple pay or google pay",
    "atm support",
    "automatic top up",
    "balance not updated after bank transfer",
    "balance not updated after cheque or cash deposit",
    "beneficiary not allowed",
    "cancel transfer",
    "card about to expire",
    "card acceptance",
    "card arrival",
    "card delivery estimate",
    "card linking",
    "card not working",
    "card payment fee charged",
    "card payment not recognised",
    "card payment wrong exchange rate",
    "card swallowed",
    "cash withdrawal charge",
    "cash withdrawal not recognised",
    "change pin",
    "compromised card",
    "contactless not working",
    "country support",
    "declined card payment",
    "declined cash withdrawal",
    "declined transfer",
    "direct debit payment not recognised",
    "disposable card limits",
    "edit personal details",
    "exchange charge",
    "exchange rate",
    "exchange via app",
    "extra charge on statement",
    "failed transfer",
    "fiat currency support",
    "get disposable virtual card",
    "get physical card",
    "getting spare card",
    "getting virtual card",
    "lost or stolen card",
    "lost or stolen phone",
    "order physical card",
    "passcode forgotten",
    "pending card payment",
    "pending cash withdrawal",
    "pending top up",
    "pending transfer",
    "pin blocked",
    "receiving money",
    "request refund",
    "reverted card payment?",
    "supported cards and currencies",
    "terminate account",
    "top up by bank transfer charge",
    "top up by card charge",
    "top up by cash or cheque",
    "top up failed",
    "top up limits",
    "top up reverted",
    "topping up by card",
    "transaction charged twice",
    "transfer fee charged",
    "transfer into account",
    "transfer not received by recipient",
    "transfer timing",
    "unable to verify identity",
    "verify my identity",
    "verify source of funds",
    "verify top up",
    "virtual card not working",
    "visa or mastercard",
    "why verify identity",
    "wrong amount of cash received",
    "wrong exchange rate for cash withdrawal",
];

const B77_CASES: [(&str, &str); 12] = [
    (
        "How do I locate my card?",
        "card arrival",
    ),
    (
        "I still have not received my new card, I ordered over a week ago.",
        "card arrival",
    ),
    (
        "I ordered a card but it has not arrived. Help please!",
        "card arrival",
    ),
    (
        "Is there a way to know when my card will arrive?",
        "card arrival",
    ),
    (
        "My card has not arrived yet.",
        "card arrival",
    ),
    (
        "When will I get my card?",
        "card arrival",
    ),
    (
        "Do you know if there is a tracking number for the new card you sent me?",
        "card arrival",
    ),
    (
        "i have not received my card",
        "card arrival",
    ),
    (
        "still waiting on that card",
        "card arrival",
    ),
    (
        "Is it normal to have to wait over a week for my new card?",
        "card arrival",
    ),
    (
        "How do I track my card?",
        "card arrival",
    ),
    (
        "How long does a card delivery take?",
        "card arrival",
    ),
];


fn question_json() -> serde_json::Value {
    let mut crit = serde_json::Map::new();
    for k in B77_KEYS {
        crit.insert(k.to_string(), serde_json::Value::Null);
    }
    serde_json::json!({
        "type": "choice",
        "instructions": "Which banking intent does `message` express?",
        "criteria": crit,
    })
}

#[test]
fn cuda_agent_system_one_repeats_stable() {
    let dir = weights_root().join(Checkpoint::English.subfolder());
    if !dir.join("model.safetensors").exists() {
        eprintln!("SKIP: no english checkpoint at {}", dir.display());
        return;
    }
    let _gpu = gpu_lock();
    let agent = RiirAgent::load(&weights_root(), Checkpoint::English).expect("agent load");
    let q = question_json();

    let render =
        |as_: &[riir_infer_laya::laya::types::Answer]| serde_json::to_string(as_).unwrap();

    // Round 0 answers for every case — the golden.
    let mut golden: Vec<String> = Vec::new();
    for (text, _gold) in B77_CASES {
        let state = serde_json::json!({ "message": text });
        let answers = agent
            .system_one(&state, &[("intent".to_string(), q.clone())])
            .expect("system_one");
        golden.push(render(&answers));
    }

    let mut failed = false;
    for round in 1..=30 {
        for (ci, (text, _gold)) in B77_CASES.iter().enumerate() {
            let state = serde_json::json!({ "message": text });
            let answers = agent
                .system_one(&state, &[("intent".to_string(), q.clone())])
                .expect("system_one repeat");
            let got = render(&answers);
            if got != golden[ci] {
                failed = true;
                println!("✗ round {round} case {ci} DIVERGED:\n  a1 = {}\n  a2 = {}", golden[ci], got);
            }
        }
        if failed {
            break;
        }
        println!("✓ round {round}: {} cases stable", B77_CASES.len());
    }
    assert!(!failed, "agent repeat probe: DIVERGED");
}
