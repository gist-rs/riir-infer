//! Plan 626 T4 — G1: the GPU-resident greedy walk is EXACTLY the incumbent
//! greedy walk (token-id sequences equal, never tolerance).
//!
//! The incumbent lane syncs every token: `set_input_token(tok)` +
//! `forward_token()` (the full 993 KB logits download). The resident lane
//! (plan 626 T3) keeps the queue deep across the token boundary: indirect
//! embed from a device id ring → all layers → on-device argmax — the id
//! never reaches the host mid-stream. Both lanes dispatch the SAME layer
//! kernels, so the logits are bit-identical by construction; the walk
//! equality therefore pins exactly the two things the resident lane adds:
//!
//! 1. **The indirect embed** — T3(a)'s kernel reads the row id the argmax
//!    WROTE (not the launch scalar); any ring-slot arithmetic bug walks a
//!    different token and the sequences diverge.
//! 2. **The tie rule** — T3a's `argmax_f32_last` twin must reproduce the
//!    harness's `max_by(total_cmp)` (LAST index on an exact-f32 tie); a
//!    first-index argmax here diverges on degenerate loops (the tie fixture
//!    at kernel level: `test_argmax_tie_rules_diverge_exactly_at_the_tie`).
//!
//! The reach half is the vacuous-guard law (Bench 996): the resident walk's
//! `resident_launch_count()` delta must equal the token count EXACTLY —
//! output equality alone can never prove the lane ran.
//!
//! Three seeds per the G1 bar (≥3 walks), each walked by BOTH arms from a
//! fresh `reset_state()` on the SAME forward instance (the Issue 1129
//! cross-run determinism precedent — every walk starts from reset state).
//!
//! Usage (GPU-exclusive on the Metal posture; ~6 min model load + ~2 s walks):
//!
//! ```bash
//! cargo test -p riir-infer-gpu --release \
//!   --features cubecl_runtime,ternary_gemv,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test plan626_resident_greedy_g1 -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI_GGUF` (default the sibling riir-train data path). Run on an
//! idle GPU (the exclusivity rule — no other compute consumer; the folded
//! league model is the posture both decode league lanes serve).

#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    target_os = "macos",
    not(debug_assertions)
))]

use std::time::Instant;

use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

/// The issue-1129 decode schedule: 24 measured tokens per walk.
const WALK: usize = 24;
/// The harvest cadence — the ring depth. 24 = 3 full batches, so the ring
/// wraps twice mid-walk and every slot is harvested before overwrite.
const K: usize = 8;

/// Host greedy argmax with the league harness tie rule: LAST maximal index
/// (`max_by(total_cmp)`). The incumbent arm feeds this; the resident arm's
/// on-device twin must reproduce it (T3a).
fn host_argmax_last(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits")
        .0
}

/// The incumbent per-token-sync walk: set_input_token + forward_token +
/// host argmax-last. Returns the produced token ids (length WALK).
fn walk_incumbent(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &QwenDeltaNetTernaryWeights,
    seed: usize,
) -> Vec<usize> {
    fwd.reset_state();
    let mut ids = Vec::with_capacity(WALK);
    let mut token = seed;
    for _ in 0..WALK {
        fwd.set_input_token(weights, token);
        let logits = fwd.forward_token();
        let next = host_argmax_last(&logits);
        ids.push(next);
        token = next;
    }
    ids
}

/// The resident walk: K-cadence enqueue with ONE fenced harvest per batch.
/// Returns the produced token ids (length WALK).
fn walk_resident(fwd: &mut TernaryDeltanetGpuForward, seed: usize) -> Vec<u32> {
    fwd.reset_state();
    fwd.resident_begin(seed, K);
    let before = TernaryDeltanetGpuForward::resident_launch_count();
    let mut ids = Vec::with_capacity(WALK);
    let mut pending = 0;
    for _ in 0..WALK {
        fwd.resident_enqueue();
        pending += 1;
        if pending == K {
            ids.extend(fwd.resident_harvest(K));
            pending = 0;
        }
    }
    if pending > 0 {
        ids.extend(fwd.resident_harvest(pending));
    }
    let delta = TernaryDeltanetGpuForward::resident_launch_count() - before;
    assert_eq!(
        delta, WALK as u64,
        "reach: resident_launch_count moved {delta}, expected exactly {WALK} — \
         the resident lane did not dispatch what the walk asked for"
    );
    ids
}

#[test]
#[ignore = "league-model GPU lane — ~6 min model load + ~2 s walks; run on an idle GPU"]
fn plan626_resident_greedy_walk_is_exact() {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed)");
        return;
    };

    let model_path = std::env::var("BONSAI_GGUF")
        .unwrap_or_else(|_| "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[plan626-g1] loading {} …", model_path.display());
    let t0 = Instant::now();
    let (mut config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&model_path)
            .map_err(|e| format!("load: {e}"))
            .unwrap();
    config.block_size = config.block_size.min(4096);
    let vocab = config.vocab_size;
    let bos = config.bos_token;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    eprintln!(
        "[plan626-g1] forward ready ({:.0}s) · vocab={vocab} · n_layer={} · folded={}",
        t0.elapsed().as_secs_f32(),
        config.n_layer,
        fwd.is_folded(),
    );

    // Three seeds per the G1 bar (≥3 walks): the BOS seed the league decode
    // serves, plus two mid-vocabulary seeds exercising different argmax
    // neighborhoods (all < vocab — every id has an embedding row).
    let seeds = [bos, vocab / 3, (vocab * 2) / 3];
    for seed in seeds {
        assert!(seed < vocab, "seed {seed} out of vocabulary");

        let incumbent = walk_incumbent(&mut fwd, &weights, seed);
        let resident = walk_resident(&mut fwd, seed);

        assert_eq!(
            resident.len(),
            WALK,
            "resident walk returned {} ids",
            resident.len()
        );
        for (step, (a, b)) in incumbent.iter().zip(resident.iter()).enumerate() {
            assert_eq!(
                *a, *b as usize,
                "seed {seed} step {step}: resident walk diverged from the incumbent \
                 (incumbent id {a}, resident id {b}) — the indirect embed ring or \
                 the argmax tie twin broke the EXACT contract"
            );
        }
        eprintln!(
            "[plan626-g1] seed {seed}: {WALK}-token walk EXACT ✓ (first ids {:?})",
            &resident[..4.min(resident.len())]
        );
    }

    // The disarm law: reset_state disarms the ring — a bare harvest after a
    // reset must panic loud, never continue the previous stream.
    fwd.reset_state();
    let disarmed =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fwd.resident_harvest(1))).is_err();
    assert!(
        disarmed,
        "reset_state must disarm the resident ring (harvest without re-begin must panic)"
    );
    eprintln!("[plan626-g1] reset_state disarm law ✓");
}
