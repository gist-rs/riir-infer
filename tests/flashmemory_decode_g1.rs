#![cfg(all(
    feature = "flashmemory_budget",
    feature = "deltanet_ternary_inference",
    feature = "bonsai2_hadamard"
))]
//! Issue 1117 — the FlashMemory budget (top-ρ) decode lane's G1 battery over
//! the synthetic folded Bonsai-2 file (4 layers: 3 DeltaNet + 1 full-attention
//! at interval 4 — the plan-622 synth kit, `tests/common/synth_bonsai2.rs`).
//!
//! Three laws, each on a REAL decode loop (teacher-forced stepping, both
//! arms through the model-level forward):
//!
//! 1. **G0 — ρ = 1.0 ≡ dense.** The budget rule at ρ 1.0 keeps every block
//!    (top `n−1` complete + the current block), so the sparse lane's reads
//!    cover exactly what the dense kernel reads — over the same per-head
//!    ascending token order, same simd dot, same max-subtract softmax form.
//!    The decode loop must therefore be BIT-IDENTICAL to
//!    `forward_qwen_deltanet_ternary` at every step, through several block
//!    boundaries AND across the dense kernel's sequential→parallel dispatch
//!    (t_n = 512). This also pins the block-open law: with the periodic
//!    refresh disabled (`refresh_period = usize::MAX`), ONLY the force-
//!    refresh at block opens can keep the newest block attended — ρ = 1.0
//!    stays bit-identical, so the fresh tokens were never stranded.
//! 2. **Incremental ≡ rebuild.** `update_last_block_from_keys` per step and
//!    a full `rebuild_from_keys` per step must produce identical caches →
//!    identical selections → BIT-IDENTICAL decode at ρ = 0.5, where the
//!    selection actually prunes.
//! 3. **ρ < 1 moves, deterministically.** The ρ = 0.5 arm's logits differ
//!    from dense somewhere (the lane is not a silent no-op), run twice gives
//!    bit-identical logits, and every logit is finite. Plus a retention
//!    sanity read: the last attention layer's selection at the end of the
//!    run attends strictly fewer tokens than exist.

use katgpt_attn::dash_attn::flashmemory_sparse::FlashMemoryConfig;
use riir_infer_core::deltanet::flashmemory_gqa::{FlashMemoryDecodeState, kv_reduction_ratio};
use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_flashmemory,
};
use riir_infer_core::deltanet::{HybridCache, HybridForwardScratch};
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;

#[path = "common/synth_bonsai2.rs"]
mod synth;

use synth::*;

/// Decode steps for the walk: ≥ 9 block boundaries at block size 64 and past
/// the dense kernel's sequential→parallel dispatch (t_n = 512).
const STEPS: usize = 700;
const BLOCK_SIZE: usize = 64;
const RHO_MID: f32 = 0.5;

fn load_synth() -> (
    Config,
    riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
) {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    // Unique name PER CALL SITE — write_tmp's path is pid-scoped, so two
    // tests in this binary racing in parallel would otherwise share one file
    // and one test's `remove_file` truncates (unmaps) the other's weights.
    let site = std::thread::current().name().unwrap_or("t").to_string();
    let path = write_tmp(&format!("riir_b2_fm_decode_{site}"), &bytes);
    let (config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&path).expect("load folded synthetic");
    let _ = std::fs::remove_file(&path);
    (config, weights)
}

/// Deterministic token stream in `[0, vocab)` (xorshift64*).
fn token_stream(len: usize) -> Vec<usize> {
    let vocab = VOCAB as usize;
    let mut state = 0x9E3779B97F4A7C15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as usize % vocab
        })
        .collect()
}

fn fm_config(refresh_period: usize) -> FlashMemoryConfig {
    FlashMemoryConfig {
        block_size: BLOCK_SIZE,
        refresh_period,
        threshold: 0.5, // unused by the budget rule
    }
}

/// Run the dense decode, returning per-step logits.
fn run_dense(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    tokens: &[usize],
) -> Vec<Vec<f32>> {
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let mut cache =
        HybridCache::with_layer_types_bounded(config, &weights.layer_types, tokens.len() + 2);
    let mut scratch = HybridForwardScratch::new(config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
    let mut logits = Vec::with_capacity(tokens.len());
    for (pos, &tok) in tokens.iter().enumerate() {
        let out = forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            tok,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        logits.push(out.to_vec());
    }
    logits
}

/// Run the FlashMemory decode at `rho`/`incremental`/`refresh_period`,
/// returning per-step logits + the final state (for the retention read).
fn run_flashmemory(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    tokens: &[usize],
    rho: f32,
    incremental: bool,
    refresh_period: usize,
) -> (Vec<Vec<f32>>, FlashMemoryDecodeState) {
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let mut cache =
        HybridCache::with_layer_types_bounded(config, &weights.layer_types, tokens.len() + 2);
    let mut scratch = HybridForwardScratch::new(config);
    let mut fm = FlashMemoryDecodeState::new(
        config,
        &weights.layer_types,
        tokens.len() + 2,
        fm_config(refresh_period),
        rho,
    );
    fm.incremental = incremental;
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
    let mut logits = Vec::with_capacity(tokens.len());
    for (pos, &tok) in tokens.iter().enumerate() {
        let out = forward_qwen_deltanet_ternary_flashmemory(
            &mut x,
            weights,
            &mut cache,
            &mut fm,
            tok,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        logits.push(out.to_vec());
    }
    (logits, fm)
}

fn assert_bit_identical(label: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
    assert_eq!(a.len(), b.len(), "{label}: step count diverged");
    for (pos, (la, lb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            la.len(),
            lb.len(),
            "{label}: vocab width diverged at step {pos}"
        );
        for (i, (x, y)) in la.iter().zip(lb.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: logit[{i}] diverged at step {pos} ({x} vs {y})"
            );
        }
    }
}

/// G0 — ρ = 1.0 is bit-identical to the dense decode, in three postures at
/// once: incremental + periodic refresh, incremental + block-open-only
/// refresh (`refresh_period = usize::MAX` — the force-refresh-at-open law is
/// the ONLY thing keeping the newest block attended), and the rebuild
/// reference posture.
#[test]
fn g0_budget_one_matches_dense() {
    let (config, weights) = load_synth();
    let tokens = token_stream(STEPS + 1);

    let dense = run_dense(&config, &weights, &tokens);

    let (inc_periodic, _) = run_flashmemory(&config, &weights, &tokens, 1.0, true, BLOCK_SIZE);
    assert_bit_identical("rho=1.0 incremental periodic", &dense, &inc_periodic);

    let (inc_open_only, _) = run_flashmemory(&config, &weights, &tokens, 1.0, true, usize::MAX);
    assert_bit_identical("rho=1.0 incremental open-only", &dense, &inc_open_only);

    let (rebuild, _) = run_flashmemory(&config, &weights, &tokens, 1.0, false, BLOCK_SIZE);
    assert_bit_identical("rho=1.0 rebuild reference", &dense, &rebuild);
}

/// Incremental per-step centroid updates produce bit-identical decode to a
/// full rebuild every step, at a ρ where the selection actually prunes.
#[test]
fn incremental_equals_rebuild() {
    let (config, weights) = load_synth();
    let tokens = token_stream(STEPS + 1);

    let (inc, _) = run_flashmemory(&config, &weights, &tokens, RHO_MID, true, BLOCK_SIZE);
    let (reb, _) = run_flashmemory(&config, &weights, &tokens, RHO_MID, false, BLOCK_SIZE);
    assert_bit_identical("rho=0.5 incremental vs rebuild", &inc, &reb);
}

/// ρ < 1 moves logits (the lane is not a silent no-op), is deterministic
/// (run twice → bit-identical), stays finite, and the selection actually
/// prunes (retention sanity read on the last attention layer).
#[test]
fn budget_moves_deterministic_and_prunes() {
    let (config, weights) = load_synth();
    let tokens = token_stream(STEPS + 1);
    let dense = run_dense(&config, &weights, &tokens);

    let (run1, fm) = run_flashmemory(&config, &weights, &tokens, RHO_MID, true, BLOCK_SIZE);
    let (run2, _) = run_flashmemory(&config, &weights, &tokens, RHO_MID, true, BLOCK_SIZE);
    assert_bit_identical("rho=0.5 determinism", &run1, &run2);

    // Moves vs dense somewhere, and never leaves the finite domain.
    let mut moved = 0usize;
    for (la, lb) in dense.iter().zip(run1.iter()) {
        for (&x, &y) in la.iter().zip(lb.iter()) {
            assert!(y.is_finite(), "flashmemory logit not finite");
            if x.to_bits() != y.to_bits() {
                moved += 1;
            }
        }
    }
    assert!(
        moved > 0,
        "rho=0.5 decode never diverged from dense — the selection is a silent no-op"
    );

    // Retention sanity: at the final step the last attention layer's
    // selection covers strictly less than the whole context.
    let last_attn = fm
        .layers
        .iter()
        .rposition(|l| l.is_some())
        .expect("at least one attention layer");
    let st = fm.layers[last_attn].as_ref().unwrap();
    let seq_len = STEPS;
    let selection = st.selector.selection();
    let (selected, total, reduction_pct) = kv_reduction_ratio(selection, &st.block_cache, seq_len);
    assert!(
        selected < total,
        "selection covers all tokens at rho=0.5 ({selected}/{total})"
    );
    assert!(
        reduction_pct > 1.0,
        "reduction {reduction_pct}% suspiciously nil at rho=0.5"
    );
}
