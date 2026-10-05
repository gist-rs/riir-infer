// Issue 035 P1 tests — the FA-constrained decode wiring.
//
// CPU-pure tests cover the schedule/scoring/config surfaces without a GPU;
// the GPU integration tests mirror `gemma2_d2f::tests`' heavy-model shape
// (Gemma2 2B fixture weights, `heavy_model_test_gate` serialization).

use super::*;
use crate::gemma2_d2f::D2fScConfig;
use crate::test_gpu_support::{gpu_release_pages, heavy_model_test_gate};
use riir_infer_core::fa_posterior::AutomatonBuilder;
use riir_infer_core::types::Config;

use crate::GpuContext;
use crate::gemma2_d2f::tests::create_test_weights;

// ── CPU-pure: the commit schedule ───────────────────────────────────

#[test]
fn commit_schedule_orders_by_confidence_and_applies_budget() {
    let rows = vec![(5, 0.2f32), (2, 0.9), (7, 0.9), (1, 0.5)];
    // No budget: every accepted position, confidence desc, ties → lower pos.
    assert_eq!(confidence_ordered(&rows, None), vec![2, 7, 1, 5]);
    // Budget 2: the two most confident.
    assert_eq!(confidence_ordered(&rows, Some(2)), vec![2, 7]);
    // Budget larger than the row count: no-op.
    assert_eq!(confidence_ordered(&rows, Some(99)), vec![2, 7, 1, 5]);
    // Empty: empty.
    assert!(confidence_ordered(&[], None).is_empty());
}

// ── CPU-pure: the RawTop1 confidence ─────────────────────────────────

#[test]
fn lm_prob_of_matches_mask_suppressed_softmax() {
    // logits over 4 tokens; mask = 1 suppressed. Row: t0 dominates.
    let logits = [10.0f32, 5.0, 1.0, -3.0];
    let p0 = lm_prob_of(&logits, 0, 1);
    let m = 10.0f32;
    let sum: f32 = [10.0, 1.0, -3.0].iter().map(|&l| (l - m).exp()).sum();
    let expected = 1.0 / sum; // exp(0)/sum
    assert!((p0 - expected).abs() < 1e-6, "{p0} vs {expected}");
    // A non-argmax token reads its own softmax mass.
    let p2 = lm_prob_of(&logits, 2, 1);
    let expected2 = (1.0f32 - m).exp() / sum;
    assert!((p2 - expected2).abs() < 1e-6);
    // Degenerate all-mask row: 0.0, not NaN.
    assert_eq!(lm_prob_of(&[0.0f32; 3], 0, 0), 0.0);
}

// ── CPU-pure: the constrained sampler ────────────────────────────────

impl ConstrainedSampler {
    fn with_marginal_weight(mut self, w: f32) -> Self {
        self.weights[6] = w;
        self
    }
}

#[test]
fn constrained_sampler_sigmoid_math() {
    // Zero weights + zero bias → sigmoid(0) = 0.5 regardless of marginal.
    let s = ConstrainedSampler::from_weights([0.0; 7], 0.0);
    assert!((s.predict(&SamplerFeatures::default(), -3.0) - 0.5).abs() < 1e-12);
    // Only the marginal weight live: sigmoid(w7 · mlog + b).
    let s = ConstrainedSampler::from_weights([0.0; 7], 0.0).with_marginal_weight(1.0);
    let z = -3.0f64;
    let expected = 1.0 / (1.0 + (-z).exp());
    assert!((s.predict(&SamplerFeatures::default(), -3.0) - expected).abs() < 1e-12);
}

// ── CPU-pure: config validation ──────────────────────────────────────

#[test]
fn validate_refuses_mask_token_bad_start_and_budget() {
    // Chain 0→1→2 over a 4-token vocab, edge tokens {1, 2}.
    let fa = AutomatonBuilder::new(3, 4, 0)
        .accept(2)
        .edge(0, 1, &[1, 2])
        .edge(1, 2, &[1, 2])
        .build()
        .unwrap();

    // Happy path.
    assert_eq!(FaConstraintConfig::new(&fa).validate(0, 4, 8), Ok(()));

    // The automaton emits the mask placeholder (token 0) → refuse.
    let fa_bad = AutomatonBuilder::new(2, 4, 0)
        .accept(1)
        .edge(0, 1, &[0, 1])
        .build()
        .unwrap();
    assert_eq!(
        FaConstraintConfig::new(&fa_bad).validate(0, 4, 8),
        Err(FaDecodeError::MaskTokenInAutomaton(0))
    );

    // Carry-over start out of range.
    let cfg = FaConstraintConfig {
        start_node: 9,
        ..FaConstraintConfig::new(&fa)
    };
    assert_eq!(
        cfg.validate(0, 4, 8),
        Err(FaDecodeError::Sampler(FaError::BadStart(9)))
    );

    // Zero denoise steps: the lane cannot propose.
    assert_eq!(
        FaConstraintConfig::new(&fa).validate(0, 4, 0),
        Err(FaDecodeError::BadConfig)
    );

    // ConstrainedMarginal tree over budget: 3 states, len_pad 4 →
    // 4 · 4 · 9 · 8 = 1152 bytes; a 1 KB budget refuses with the formula.
    let cfg = FaConstraintConfig {
        commit_by: FaCommitBy::ConstrainedMarginal,
        tree_budget_bytes: 1024,
        ..FaConstraintConfig::new(&fa)
    };
    assert_eq!(
        cfg.validate(0, 4, 8),
        Err(FaDecodeError::TreeBudgetExceeded {
            n_nodes: 3,
            len_pad: 4,
            bytes: 1152,
            budget: 1024
        })
    );
    // RawTop1 ignores the tree budget entirely.
    let cfg = FaConstraintConfig {
        tree_budget_bytes: 0,
        ..FaConstraintConfig::new(&fa)
    };
    assert_eq!(cfg.validate(0, 4, 8), Ok(()));
}

// ── CPU-pure: propose_x0 wiring ──────────────────────────────────────

/// Diamond into an accepting self-loop over a 4-token vocab (edge tokens
/// exclude 0, the mask in these tests).
fn diamond_fa() -> Automaton {
    AutomatonBuilder::new(5, 4, 0)
        .accept(4)
        .edge(0, 1, &[1])
        .edge(1, 2, &[2])
        .edge(1, 3, &[3])
        .edge(2, 4, &[1, 2])
        .edge(3, 4, &[1, 3])
        .edge(4, 4, &[1, 2, 3])
        .build()
        .unwrap()
}

#[test]
fn propose_x0_pins_committed_draws_free_and_lands_accepting() {
    let fa = diamond_fa();
    let block_len = 3;
    // Block sits at block_start 2 of a 6-row logit sheet. Pins must be
    // walk-consistent: from the start the only edge allows token 1, then
    // 1→2 allows exactly 2 — so the committed prefix (1, 2) is the walk
    // 0→1→2 and the free position draws from 2→4's {1, 2}.
    let row = [0.0f32, 5.0, 1.0, 0.5];
    let all_logits: Vec<Vec<f32>> = vec![row.to_vec(); 6];
    // Global tokens: [mask, mask, 1, 2, mask, mask] — block = positions
    // 2..5, committed 0–1, masked 2.
    let tokens = vec![0usize, 0, 1, 2, 0, 0];
    let masked = [false, false, false, false, true, false];

    let mut scratch = FaDecodeScratch::new(4, block_len);
    let mut rng = SplitMix64::new(7);
    let mut nan = 0u64;
    propose_x0(
        &fa,
        &mut scratch,
        fa.start(),
        &all_logits,
        2,
        block_len,
        &tokens,
        &masked,
        1.0,
        false,
        false,
        &mut rng,
        &mut nan,
    )
    .expect("propose");
    assert_eq!(nan, 0);
    // Pins carried verbatim; the free position drew an automaton-allowed
    // token (2→4 allows {1, 2}).
    assert_eq!(scratch.proposals[0], 1, "pin 1");
    assert_eq!(scratch.proposals[1], 2, "pin 2");
    assert!(
        scratch.proposals[2] == 1 || scratch.proposals[2] == 2,
        "free draw {} not in 2→4's token set",
        scratch.proposals[2]
    );
    // Same seed ⇒ same draw.
    let mut rng2 = SplitMix64::new(7);
    let mut scratch2 = FaDecodeScratch::new(4, block_len);
    propose_x0(
        &fa,
        &mut scratch2,
        fa.start(),
        &all_logits,
        2,
        block_len,
        &tokens,
        &masked,
        1.0,
        false,
        false,
        &mut rng2,
        &mut nan,
    )
    .expect("propose 2");
    assert_eq!(scratch.proposals, scratch2.proposals);
}

#[test]
fn carry_over_across_two_proposals_walks_to_accept() {
    // Two successive block proposals threaded by final node — the P1
    // carry-over contract at the sampler level: block 2 conditions on
    // block 1's landing node, and the concatenation walks accept.
    let fa = diamond_fa();
    let row = [0.0f32, 4.0, 2.0, 1.0];
    let all_logits: Vec<Vec<f32>> = vec![row.to_vec(); 8];
    let mut tokens: Vec<usize> = vec![0; 8];

    let mut scratch = FaDecodeScratch::new(4, 4);
    let mut rng = SplitMix64::new(11);
    let mut nan = 0u64;

    // Block 1: positions 0..4 from the designated start (4..8 pre-committed
    // placeholders — not this proposal's problem).
    let mut masked1 = [true; 8];
    masked1[4..8].fill(false);
    propose_x0(
        &fa,
        &mut scratch,
        fa.start(),
        &all_logits,
        0,
        4,
        &tokens,
        &masked1,
        1.0,
        false,
        false,
        &mut rng,
        &mut nan,
    )
    .expect("block 1");
    for (i, &p) in scratch.proposals.iter().enumerate() {
        tokens[i] = p as usize;
    }
    let block1: Vec<u32> = tokens[..4].iter().map(|&t| t as u32).collect();
    let node1 = fa.walk_from(fa.start(), &block1).expect("block 1 walks");

    // Block 2: positions 4..8 from node1, block-1 tokens pinned.
    let mut masked2 = [false; 8];
    masked2[4..8].fill(true);
    propose_x0(
        &fa,
        &mut scratch,
        node1,
        &all_logits,
        4,
        4,
        &tokens,
        &masked2,
        1.0,
        false,
        false,
        &mut rng,
        &mut nan,
    )
    .expect("block 2");
    for (i, &p) in scratch.proposals.iter().enumerate() {
        tokens[4 + i] = p as usize;
    }
    let block2: Vec<u32> = tokens[4..].iter().map(|&t| t as u32).collect();
    let node2 = fa.walk_from(node1, &block2).expect("block 2 walks");
    assert!(fa.is_accept(node2), "threaded walk lands accepting");
}

// ── GPU integration ──────────────────────────────────────────────────

/// Chain 0→1→2→3→4(accept) with a self-loop on the accept node, over the
/// model vocab; every edge allows the same small token set (1..=4) — the
/// mask placeholder (0) is excluded. The self-loop is load-bearing: block
/// carry-over starts block 2 from block 1's landing node (the accept), and
/// a start with no outgoing edges is a DeadStart refusal by design.
fn chain_fa(vocab: usize) -> Automaton {
    let allowed: Vec<u32> = (1u32..=4).collect();
    AutomatonBuilder::new(5, vocab, 0)
        .accept(4)
        .edge(0, 1, &allowed)
        .edge(1, 2, &allowed)
        .edge(2, 3, &allowed)
        .edge(3, 4, &allowed)
        .edge(4, 4, &allowed)
        .build()
        .unwrap()
}

fn fa_decode_config(steps: usize, tau: f32) -> Gemma2D2fConfig {
    Gemma2D2fConfig {
        block_size: 4,
        denoise_steps: steps,
        // The fixture's logits are ~N(0, 0.48) over a 256k vocab — raw
        // top-1 prob sits ~3.8e-6 (Issue 1001's calibration); the tests
        // pick tau relative to that.
        confidence_threshold: tau,
        temperature: 1.0,
        sampler: None,
        sc_config: D2fScConfig::default(),
    }
}

fn assert_block_walks(
    fa: &Automaton,
    start: usize,
    result: &Gemma2D2fFaResult,
    block_size: usize,
    mask: usize,
) {
    let seq = result.decode.tokens.len();
    let block = &result.decode.tokens[seq - block_size..];
    for (i, &t) in block.iter().enumerate() {
        assert_ne!(t, mask, "block position {i} still carries the mask");
    }
    let toks: Vec<u32> = block.iter().map(|&t| t as u32).collect();
    let node = fa
        .walk_from(start, &toks)
        .unwrap_or_else(|| panic!("block does not walk from {start}: {toks:?}"));
    assert!(fa.is_accept(node), "block walk lands on non-accepting node");
}

#[test]
fn test_d2f_decode_constrained_chain_grammar_converges() {
    let _heavy = heavy_model_test_gate();
    let ctx = GpuContext::new().expect("GpuContext should init");
    let client = ctx.cubecl_client();
    gpu_release_pages(&client);

    let config = Config::gemma2_2b();
    let weights = create_test_weights(&config);
    let decode_config = fa_decode_config(20, 1e-9);
    let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
    let fa = chain_fa(config.vocab_size);
    let constraint = FaConstraintConfig::new(&fa);
    let mut rng = fastrand::Rng::with_seed(42);

    let mask = 0usize;
    let prompt = vec![42usize, 7];
    let result = d2f_decode_gemma2_constrained(
        &mut d2f,
        &prompt,
        mask,
        &decode_config,
        &constraint,
        &mut rng,
    )
    .expect("constrained decode");

    assert!(result.decode.converged, "raw axis at tau 1e-9 converges");
    assert_eq!(result.decode.state, D2fBlockState::FullyActivated);
    assert_eq!(result.nan_fallbacks, 0, "no NaN fallbacks");
    assert!(result.committed > 0, "the confidence rule committed");
    assert_block_walks(
        &fa,
        constraint.start_node,
        &result,
        decode_config.block_size,
        mask,
    );
}

#[test]
fn test_d2f_decode_constrained_final_commit_never_leaves_masks() {
    // tau = 0.99 never fires on this fixture (max raw prob ~3.8e-6): every
    // step remasks, the loop exhausts — and the final x0 commit still
    // returns a fully valid, mask-free, accepted block.
    let _heavy = heavy_model_test_gate();
    let ctx = GpuContext::new().expect("GpuContext should init");
    let client = ctx.cubecl_client();
    gpu_release_pages(&client);

    let config = Config::gemma2_2b();
    let weights = create_test_weights(&config);
    let decode_config = fa_decode_config(3, 0.99);
    let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
    let fa = chain_fa(config.vocab_size);
    let constraint = FaConstraintConfig::new(&fa);
    let mut rng = fastrand::Rng::with_seed(7);

    let mask = 0usize;
    let result = d2f_decode_gemma2_constrained(
        &mut d2f,
        &[42usize, 7],
        mask,
        &decode_config,
        &constraint,
        &mut rng,
    )
    .expect("constrained decode");

    assert!(
        !result.decode.converged,
        "tau 0.99 never fires on this fixture"
    );
    assert_eq!(result.committed, 0, "no per-step commits");
    assert_eq!(result.decode.steps_used, 3, "loop exhausted");
    assert_block_walks(
        &fa,
        constraint.start_node,
        &result,
        decode_config.block_size,
        mask,
    );
}

#[test]
fn test_d2f_decode_constrained_marginal_axis() {
    // The ConstrainedMarginal axis end to end: constrained posteriors are
    // normalized over the edge's 4 allowed tokens (~0.25 each), so tau 0.05
    // commits immediately — and the tree lane's nan counter stays 0.
    let _heavy = heavy_model_test_gate();
    let ctx = GpuContext::new().expect("GpuContext should init");
    let client = ctx.cubecl_client();
    gpu_release_pages(&client);

    let config = Config::gemma2_2b();
    let weights = create_test_weights(&config);
    let decode_config = fa_decode_config(20, 0.05);
    let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
    let fa = chain_fa(config.vocab_size);
    let constraint = FaConstraintConfig {
        commit_by: FaCommitBy::ConstrainedMarginal,
        ..FaConstraintConfig::new(&fa)
    };
    let mut rng = fastrand::Rng::with_seed(13);

    let mask = 0usize;
    let result = d2f_decode_gemma2_constrained(
        &mut d2f,
        &[42usize, 7],
        mask,
        &decode_config,
        &constraint,
        &mut rng,
    )
    .expect("constrained decode");

    assert!(
        result.decode.converged,
        "marginal axis at tau 0.05 converges"
    );
    assert_eq!(result.nan_fallbacks, 0, "no NaN fallbacks on the tree lane");
    assert_block_walks(
        &fa,
        constraint.start_node,
        &result,
        decode_config.block_size,
        mask,
    );
}

#[test]
fn test_d2f_decode_constrained_two_block_carryover() {
    // The P1 carry-over contract end to end: decode block 1, thread its
    // final_node into block 2's start_node, and verify the CONCATENATED
    // blocks walk from the original start to an accepting node.
    let _heavy = heavy_model_test_gate();
    let ctx = GpuContext::new().expect("GpuContext should init");
    let client = ctx.cubecl_client();
    gpu_release_pages(&client);

    let config = Config::gemma2_2b();
    let weights = create_test_weights(&config);
    let decode_config = fa_decode_config(20, 1e-9);
    let mut d2f = GpuGemmaCubeCLD2F::new(client.clone(), &weights, &config, decode_config);
    let fa = chain_fa(config.vocab_size);
    let mask = 0usize;

    // Block 1.
    let c1 = FaConstraintConfig::new(&fa);
    let mut rng = fastrand::Rng::with_seed(42);
    let prompt = vec![42usize, 7];
    let r1 = d2f_decode_gemma2_constrained(&mut d2f, &prompt, mask, &decode_config, &c1, &mut rng)
        .expect("block 1");
    assert_block_walks(&fa, c1.start_node, &r1, decode_config.block_size, mask);

    // Block 2: prompt = block 1's full output (the function appends its own
    // masks), walk starts at r1.final_node.
    let block1: Vec<usize> = r1.decode.tokens[prompt.len()..].to_vec();
    let c2 = FaConstraintConfig {
        start_node: r1.final_node,
        ..FaConstraintConfig::new(&fa)
    };
    let r2 = d2f_decode_gemma2_constrained(
        &mut d2f,
        &r1.decode.tokens,
        mask,
        &decode_config,
        &c2,
        &mut rng,
    )
    .expect("block 2");
    assert_block_walks(&fa, r1.final_node, &r2, decode_config.block_size, mask);

    // The concatenated walk: block 1 ‖ block 2 from the ORIGINAL start.
    let block2: Vec<usize> = r2.decode.tokens[r1.decode.tokens.len()..].to_vec();
    let both: Vec<u32> = block1
        .iter()
        .chain(block2.iter())
        .map(|&t| t as u32)
        .collect();
    let node = fa.walk_from(fa.start(), &both).expect("both blocks walk");
    assert!(
        fa.is_accept(node),
        "threaded two-block walk lands accepting"
    );
}
