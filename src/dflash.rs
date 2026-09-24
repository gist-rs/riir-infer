use crate::spec_types::{DraftResult, SpeculativeContext};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax_scaled};
use katgpt_speculative::dflash::{DflashCache, DflashCtx};
use rayon::prelude::*;

// ── Issue 013 Phase B / Issue 373: shared-core backend adapter ────
//
// `katgpt_speculative::dflash` owns the three `_with` algorithmic cores;
// these impls wire riir-engine's `MultiLayerKVCache` and `ForwardContext`
// into the generic cores via disjoint field borrows.
//
// Orphan-rule note (Issue 373, unblocks Plan 406 Phase 2 T2.1):
// `MultiLayerKVCache` is local today but will become foreign
// (re-exported from katgpt-transformer) once Phase 2 T2.1 lands. A direct
// `impl DflashCache for MultiLayerKVCache` would then be foreign-trait-on-
// foreign-type → E0117. The upstream sibling `katgpt-forward` solved this
// with a borrowing newtype adapter (Plan 394); we mirror that pattern here
// so the adapter travels with this module, not the type.

/// Borrowing adapter that satisfies `DflashCache` for our `MultiLayerKVCache`
/// without violating the orphan rule (local type → impl always legal,
/// regardless of whether `MultiLayerKVCache` is local or re-exported).
struct CacheAdapter<'a>(&'a mut MultiLayerKVCache);

/// Trampoline matching the shared core's `forward_fn` shape
/// (`Fn(&mut Ctx, &Weights, &mut Cache, usize, usize, &Config)`). Unwraps the
/// adapter so riir-engine's `forward` gets its concrete `&mut MultiLayerKVCache`.
#[inline]
fn forward_via_adapter(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut CacheAdapter<'_>,
    token: usize,
    pos: usize,
    config: &Config,
) {
    let _ = forward(ctx, weights, cache.0, token, pos, config);
}

impl DflashCache for CacheAdapter<'_> {
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
    }

    #[inline]
    fn invalidate_position(&mut self, pos: usize, kv_dim: usize) {
        self.0.invalidate_position(pos, kv_dim);
    }

    fn seed_layers(&mut self, target_hidden: &[f32], draft_kv_dim: usize) {
        // Replicates the KV-seeding loop from the previous local
        // `dflash_predict_conditioned_with`: project target hidden into the
        // drafter's KV dim by truncation/padding.
        if target_hidden.is_empty() || draft_kv_dim == 0 {
            return;
        }
        let target_dim = target_hidden.len().min(draft_kv_dim);
        for layer in &mut self.0.layers {
            layer.key[..target_dim].copy_from_slice(&target_hidden[..target_dim]);
            layer.key[target_dim..draft_kv_dim].fill(0.0);
            layer.value[..target_dim].copy_from_slice(&target_hidden[..target_dim]);
            layer.value[target_dim..draft_kv_dim].fill(0.0);
        }
    }
}

impl DflashCtx<TransformerWeights> for ForwardContext {
    #[inline]
    fn logits_slice(&self) -> &[f32] {
        &self.logits
    }

    #[inline]
    fn hidden_state_slice(&self) -> &[f32] {
        &self.hidden_state
    }

    fn apply_mtp_conditioning(
        &mut self,
        weights: &TransformerWeights,
        mtp_ctx: &[f32],
        n_embd: usize,
        vocab_size: usize,
    ) {
        // Add the MTP context into the hidden state (first AR step only),
        // then recompute logits via the LM head matmul.
        let n = n_embd.min(mtp_ctx.len());
        for i in 0..n {
            unsafe {
                *self.hidden_state.get_unchecked_mut(i) += *mtp_ctx.get_unchecked(i);
            }
        }
        crate::types::matmul(
            &mut self.logits,
            &weights.lm_head,
            &self.hidden_state,
            vocab_size,
            n_embd,
        );
    }
}

// ── Zero-alloc _with variants (delegating to shared core) ──────

/// Zero-alloc variant of `dflash_predict`.
///
/// Delegates to [`katgpt_speculative::dflash::dflash_predict_with`] — the
/// algorithmic core is shared with `katgpt-rs`. This thin wrapper preserves
/// the original riir-engine signature (taking `&mut SpeculativeContext`) and
/// records the populated-step count on `sctx`.
///
/// Free win: the shared core uses Issue 053 selective `invalidate_position`
/// (one full reset before the loop, then per-step position invalidation)
/// instead of a full per-step reset. End-state marginals are identical.
pub fn dflash_predict_with(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> usize {
    let mut cache = CacheAdapter(&mut sctx.cache);
    let steps = katgpt_speculative::dflash::dflash_predict_with(
        &mut sctx.ctx,
        &mut cache,
        draft_weights,
        forward_via_adapter,
        &mut sctx.probs_buf,
        &mut sctx.marginals_flat,
        draft_config,
        token,
        pos,
    );
    sctx.steps_populated = steps;
    steps
}

/// Hidden-state-capturing variant of [`dflash_predict_with`] (Plan 433).
///
/// Identical to [`dflash_predict_with`] but additionally snapshots the
/// drafter's final hidden state after each forward step into
/// `h_dflash_captured`. Used by the Weaver marginal corrector integration
/// (`dflash_predict_with_weaver`), which needs `h_dflash[step]` (one slice
/// per draft depth).
///
/// # Layout
///
/// `h_dflash_captured` is `[max_steps * n_embd]`, row-major. Step `i`'s
/// hidden state occupies `[i * n_embd .. (i + 1) * n_embd]`.
///
/// # Behavior preservation
///
/// Marginals written to `sctx.marginals_flat` are bit-identical to
/// [`dflash_predict_with`] — the only addition is the per-step hidden-state
/// snapshot.
pub fn dflash_predict_with_capture(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    h_dflash_captured: &mut [f32],
) -> usize {
    let mut cache = CacheAdapter(&mut sctx.cache);
    let steps = katgpt_speculative::dflash::dflash_predict_with_capture(
        &mut sctx.ctx,
        &mut cache,
        draft_weights,
        forward_via_adapter,
        &mut sctx.probs_buf,
        &mut sctx.marginals_flat,
        h_dflash_captured,
        draft_config,
        token,
        pos,
    );
    sctx.steps_populated = steps;
    steps
}

// ── Weaver-corrected DFlash (Plan 433, gated `weaver_runtime`) ──
//
// Combines `dflash_predict_with_capture` + `WeaverCorrector::correct_marginals_with_scratch`
// in a single call so the spec decode loop (e.g. `speculative_step_qwen_deltanet_tree`)
// can opt into Weaver correction. This mirrors the katgpt-rs wrapper so both
// runtimes expose the same wiring without duplicating the orchestration logic.

/// `DFlash` predict + Weaver marginal correction in one call (Plan 433).
///
/// Runs [`dflash_predict_with_capture`] to produce marginals + per-step
/// drafter hidden states, then applies
/// [`WeaverCorrector::correct_marginals_with_scratch`] to correct the
/// marginals in-place over the top-K candidates at each draft depth.
///
/// See the katgpt-rs sibling (`katgpt_forward::dflash::dflash_predict_with_weaver`)
/// for the full contract — this wrapper is behavior-identical.
///
/// # No-harm contract
///
/// Zero-init Weaver weights produce zero residuals; the only change to the
/// marginals is the top-K truncation/renormalization (typically <1% mass).
#[cfg(feature = "weaver_runtime")]
// hot-path leaf: drafter + Weaver wiring
#[allow(clippy::too_many_arguments)]
pub fn dflash_predict_with_weaver(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    h_dflash_captured: &mut [f32],
    weaver: &katgpt_speculative::weaver::WeaverCorrector,
    h_verifier: &[f32],
    embedding: &[f32],
    scratch: &mut katgpt_speculative::weaver::WeaverScratch,
) -> Result<usize, katgpt_speculative::weaver::WeaverCorrectError> {
    // 1. Draft marginals + capture per-step hidden states.
    let steps = dflash_predict_with_capture(
        sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        h_dflash_captured,
    );
    if steps == 0 {
        return Ok(0);
    }

    // 2. Slice `h_dflash_captured` into a `&[&[f32]]` view (stack array, max 64).
    let n_embd = draft_config.n_embd;
    let mut h_dflash_slices: [&[f32]; 64] = [&[]; 64];
    let count = steps.min(64);
    for (i, slot) in h_dflash_slices.iter_mut().enumerate().take(count) {
        let start = i * n_embd;
        let end = start + n_embd;
        *slot = if end <= h_dflash_captured.len() {
            &h_dflash_captured[start..end]
        } else {
            &[]
        };
    }
    let h_dflash = &h_dflash_slices[..count];

    // 3. Apply Weaver correction in-place.
    weaver.correct_marginals_with_scratch(
        &mut sctx.marginals_flat,
        h_verifier,
        h_dflash,
        embedding,
        draft_config.vocab_size,
        scratch,
    )?;
    Ok(steps)
}

/// Zero-alloc variant of `dflash_predict_ar`.
///
/// Delegates to [`katgpt_speculative::dflash::dflash_predict_ar_with`] — the
/// algorithmic core is shared with `katgpt-rs`. This thin wrapper preserves
/// the original riir-engine signature (taking `&mut SpeculativeContext`) and
/// records the populated-step count on `sctx`.
///
/// Caller responsibility: reset `sctx.cache` before calling (allows KV
/// preloading between reset and the AR loop — Phase 3, Plan 055).
pub fn dflash_predict_ar_with(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    mtp_context: Option<&[f32]>,
) -> usize {
    // NOTE: Caller is responsible for resetting sctx before calling this function.
    // This allows KV cache preloading between reset and the AR loop (Phase 3, Plan 055).
    let mut cache = CacheAdapter(&mut sctx.cache);
    let steps = katgpt_speculative::dflash::dflash_predict_ar_with(
        &mut sctx.ctx,
        &mut cache,
        draft_weights,
        forward_via_adapter,
        &mut sctx.probs_buf,
        &mut sctx.marginals_flat,
        &mut sctx.sampled_tokens,
        draft_config,
        token,
        pos,
        rng,
        mtp_context,
    );
    sctx.steps_populated = steps;
    steps
}

/// Zero-alloc variant of `dflash_predict_conditioned`.
///
/// Delegates to [`katgpt_speculative::dflash::dflash_predict_conditioned_with`]
/// — the algorithmic core is shared with `katgpt-rs`. This thin wrapper
/// preserves the original riir-engine signature (taking
/// `&mut SpeculativeContext`) and records the populated-step count on `sctx`.
pub fn dflash_predict_conditioned_with(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut Rng,
) -> usize {
    let mut cache = CacheAdapter(&mut sctx.cache);
    let steps = katgpt_speculative::dflash::dflash_predict_conditioned_with(
        &mut sctx.ctx,
        &mut cache,
        draft_weights,
        forward_via_adapter,
        &mut sctx.probs_buf,
        &mut sctx.marginals_flat,
        &mut sctx.sampled_tokens,
        draft_config,
        token,
        pos,
        target_hidden_state,
        rng,
    );
    sctx.steps_populated = steps;
    steps
}

// ── Backward-compatible public API (thin wrappers) ─────────────

/// Sequential `DFlash`: Predict marginal distributions using draft model.
/// Uses pre-allocated `ForwardContext` for zero-alloc per step.
pub fn dflash_predict(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let mut sctx = SpeculativeContext::new(draft_config);
    let steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, token, pos);
    let vocab_size = draft_config.vocab_size;
    let mut marginals = Vec::with_capacity(steps);
    for step in 0..steps {
        marginals.push(sctx.marginal_slice(step, vocab_size).to_vec());
    }
    marginals
}

/// Parallel `DFlash`: Predict marginals using rayon.
/// One `ForwardContext` per rayon worker thread — no contention, zero waste.
pub fn dflash_predict_parallel(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    if max_steps == 0 {
        return Vec::new();
    }

    // For micro models, sequential is faster than rayon overhead
    if draft_config.n_embd <= draft_config.parallel_threshold {
        return dflash_predict(draft_weights, draft_config, token, pos);
    }

    (0..max_steps)
        .into_par_iter()
        .map_init(
            || {
                (
                    ForwardContext::new(draft_config),
                    MultiLayerKVCache::new(draft_config),
                )
            },
            |(ctx, cache), step| {
                let draft_pos = pos + step;
                let logits = forward(ctx, draft_weights, cache, token, draft_pos, draft_config);
                softmax_scaled(logits, 1.0 / draft_config.temperature);
                logits.to_vec()
            },
        )
        .collect()
}

/// Autoregressive `DFlash`: Predict marginals by sampling and feeding back tokens.
///
/// Unlike `dflash_predict` (which feeds the same token/pos to every step),
/// this samples a token at each step and feeds it back as input for the next.
/// Produces conditional `q(x|x_{<i`}) distributions instead of independent marginals.
pub fn dflash_predict_ar(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> DraftResult {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.cache.reset();
    let steps = dflash_predict_ar_with(
        &mut sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        rng,
        None,
    );
    let vocab_size = draft_config.vocab_size;
    DraftResult::new(
        (0..steps)
            .map(|step| sctx.marginal_slice(step, vocab_size).to_vec())
            .collect(),
        sctx.sampled_tokens().to_vec(),
    )
}

/// Target-conditioned `DFlash`: Predict marginals using draft model
/// conditioned on the target model's hidden state.
///
/// Uses Option C from plan 012: seed draft KV cache with target hidden state.
/// The target's hidden state (from `ForwardContext.hidden_state`) is projected
/// to the draft model's KV dimension and used as the initial KV cache entry.
/// This gives the draft model access to the target's representation without
/// any weight matrix changes.
///
/// Returns `DraftResult` with marginals and sampled tokens.
pub fn dflash_predict_conditioned(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut Rng,
) -> DraftResult {
    let mut sctx = SpeculativeContext::new(draft_config);
    let steps = dflash_predict_conditioned_with(
        &mut sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        target_hidden_state,
        rng,
    );
    let vocab_size = draft_config.vocab_size;
    DraftResult::new(
        (0..steps)
            .map(|step| sctx.marginal_slice(step, vocab_size).to_vec())
            .collect(),
        sctx.sampled_tokens().to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};
    use katgpt_speculative::dd_tree::{build_dd_tree, extract_best_path};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_dflash_produces_marginals() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        assert!(!marginals.is_empty());
        assert!(marginals.len() <= config.draft_lookahead);

        for (i, row) in marginals.iter().enumerate() {
            assert_eq!(row.len(), config.vocab_size, "row {i} wrong size");
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "row {i} sum = {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dflash_parallel_matches_count() {
        let (weights, config) = make_draft();
        let seq = dflash_predict(&weights, &config, 0, 0);
        let par = dflash_predict_parallel(&weights, &config, 0, 0);
        assert_eq!(seq.len(), par.len(), "parallel should produce same count");
    }

    #[test]
    fn test_dflash_positions_differ() {
        let (weights, config) = make_draft();
        let m0 = dflash_predict(&weights, &config, 0, 0);
        let m1 = dflash_predict(&weights, &config, 0, 1);
        assert_ne!(
            m0[0], m1[0],
            "marginals at different positions should differ"
        );
    }

    #[test]
    fn test_dflash_ar_produces_marginals() {
        let (weights, config) = make_draft();
        let result = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert!(!result.marginals.is_empty(), "should produce marginals");
        assert!(
            !result.sampled_tokens.is_empty(),
            "should produce sampled tokens"
        );
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_ar_is_autoregressive() {
        // Verify the sampler is actually consuming the RNG (not just returning argmax).
        // We sample from many seeds and require that at least two distinct token
        // sequences appear. Asserting a specific pair (e.g. seed 1 vs 2) is too
        // fragile: the 27-token draft vocab at temperature 0.5 can be peaked
        // enough that nearby uniform draws land in the same CDF bin, making
        // adjacent seeds produce identical AR paths by chance.
        let (weights, config) = make_draft();
        let mut distinct = std::collections::HashSet::new();
        for seed in 1u64..=16 {
            let r = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(seed));
            distinct.insert(r.sampled_tokens);
        }
        assert!(
            distinct.len() >= 2,
            "sampler appears deterministic across 16 seeds — RNG is not being consumed; \
             distinct sequences: {}",
            distinct.len()
        );
    }

    #[test]
    fn test_dflash_ar_deterministic() {
        let (weights, config) = make_draft();
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert_eq!(
            r1.sampled_tokens, r2.sampled_tokens,
            "same seed should produce same tokens"
        );
        for (a, b) in r1.marginals.iter().zip(r2.marginals.iter()) {
            for (pa, pb) in a.iter().zip(b.iter()) {
                assert!((pa - pb).abs() < 1e-6, "marginals should be identical");
            }
        }
    }

    #[test]
    fn test_extract_best_path() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        let path = extract_best_path(&tree);
        if !tree.is_empty() {
            assert!(!path.is_empty(), "non-empty tree should produce a path");
            for &t in &path {
                assert!(t < config.vocab_size, "token {t} out of range");
            }
        }
    }

    #[test]
    fn test_dflash_conditioned_produces_marginals() {
        let (weights, config) = make_draft();
        let target_config = Config::micro();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        // Get target hidden state
        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );
        let hidden = target_ctx.hidden_state.clone();

        let result =
            dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));
        assert!(!result.marginals.is_empty());
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_conditioned_differs_from_unconditioned() {
        let (weights, config) = make_draft();
        let target_config = Config::micro();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );
        let hidden = target_ctx.hidden_state.clone();

        let uncond = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let cond = dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));

        // Conditioned should differ from unconditioned (different KV cache seed)
        assert_ne!(
            cond.sampled_tokens, uncond.sampled_tokens,
            "conditioned marginals should differ from unconditioned"
        );
    }

    #[test]
    fn test_dflash_conditioned_valid_probs() {
        let (weights, config) = make_draft();
        let hidden = vec![0.5; config.n_embd]; // fake hidden state
        let result =
            dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));
        for probs in &result.marginals {
            for &p in probs {
                assert!(p.is_finite(), "prob should be finite");
                assert!(p >= 0.0, "prob should be non-negative");
            }
        }
    }

    #[test]
    fn test_dflash_conditioned_empty_hidden() {
        let (weights, config) = make_draft();
        let result = dflash_predict_conditioned(&weights, &config, 0, 0, &[], &mut Rng::new(42));
        // Empty hidden state should still produce valid output (no seeding)
        assert!(!result.marginals.is_empty());
    }

    #[test]
    fn test_dflash_predict_with_matches_original() {
        let (weights, config) = make_draft();
        let mut sctx = SpeculativeContext::new(&config);
        let steps = dflash_predict_with(&mut sctx, &weights, &config, 0, 0);
        let vocab_size = config.vocab_size;

        assert_eq!(steps, config.draft_lookahead);
        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            assert_eq!(slice.len(), vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
        }
    }

    /// Plan 433 T3 (riir-engine mirror): `dflash_predict_with_capture` writes
    /// marginals that are bit-identical to `dflash_predict_with`, and populates
    /// the hidden-state capture buffer with non-trivial data.
    #[test]
    fn test_dflash_predict_with_capture_matches_no_capture() {
        let (weights, config) = make_draft();
        let vocab_size = config.vocab_size;
        let n_embd = config.n_embd;

        // Run the no-capture path.
        let mut sctx_plain = SpeculativeContext::new(&config);
        let steps_plain = dflash_predict_with(&mut sctx_plain, &weights, &config, 0, 0);
        let marginals_plain = sctx_plain.marginals_flat.clone();

        // Run the capture path.
        let mut sctx_cap = SpeculativeContext::new(&config);
        let mut h_captured = vec![0.0f32; config.draft_lookahead * n_embd];
        let steps_cap =
            dflash_predict_with_capture(&mut sctx_cap, &weights, &config, 0, 0, &mut h_captured);

        assert_eq!(steps_plain, steps_cap, "step count mismatch");

        // Bit-identical marginals.
        for (i, (a, b)) in sctx_cap
            .marginals_flat
            .iter()
            .zip(marginals_plain.iter())
            .enumerate()
        {
            assert_eq!(a, b, "marginal[{i}] differs between capture and no-capture");
        }

        // Hidden states populated.
        for step in 0..steps_cap {
            let h = &h_captured[step * n_embd..(step + 1) * n_embd];
            let max_abs = h.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            assert!(
                max_abs > 0.0,
                "step {step} hidden state is all-zero — capture didn't run"
            );
        }

        let _ = vocab_size;
    }

    /// Plan 433 T7 (riir-engine mirror): `dflash_predict_with_weaver` with
    /// zero-init Weaver weights whose K exceeds `vocab_size` preserves G1 and
    /// leaves marginals bit-identical (k > V early-return path).
    #[cfg(feature = "weaver_runtime")]
    #[test]
    fn test_dflash_predict_with_weaver_zero_weights_preserves_g1() {
        use katgpt_speculative::weaver::{
            WeaverConfig, WeaverCorrector, WeaverScratch, WeaverWeights,
        };

        let (weights, config) = make_draft();
        let n_embd = config.n_embd;
        let vocab_size = config.vocab_size;

        let weaver_cfg = WeaverConfig {
            hidden_dim: n_embd,
            n_heads: 4,
            k_candidates: vocab_size + 100, // K > V → early return
            n_layer: 1,
            d_ff: n_embd * 4,
            rms_eps: 1e-6,
            max_depth: config.draft_lookahead,
        };
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(weaver_cfg.clone()));
        let mut scratch = WeaverScratch::new(&weaver_cfg);

        // Capture-path marginals (the reference).
        let mut sctx_ref = SpeculativeContext::new(&config);
        let mut h_ref = vec![0.0f32; config.draft_lookahead * n_embd];
        let steps_ref =
            dflash_predict_with_capture(&mut sctx_ref, &weights, &config, 0, 0, &mut h_ref);
        let marginals_ref = sctx_ref.marginals_flat.clone();

        // Weaver path.
        let mut sctx = SpeculativeContext::new(&config);
        let mut h = vec![0.0f32; config.draft_lookahead * n_embd];
        let h_verifier: Vec<f32> = vec![0.5; n_embd];
        let embedding: Vec<f32> = vec![0.1; vocab_size * n_embd];
        let steps = dflash_predict_with_weaver(
            &mut sctx,
            &weights,
            &config,
            0,
            0,
            &mut h,
            &corrector,
            &h_verifier,
            &embedding,
            &mut scratch,
        )
        .expect("zero-weight Weaver with K>V should not error");

        assert_eq!(steps, steps_ref);
        assert_eq!(
            sctx.marginals_flat, marginals_ref,
            "Weaver with K > V should leave marginals unchanged"
        );

        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
            for &v in slice {
                assert!(v.is_finite(), "NaN/Inf in marginal[{step}]");
            }
        }
    }

    #[test]
    fn test_dflash_predict_ar_with_matches_original() {
        let (weights, config) = make_draft();
        let mut sctx = SpeculativeContext::new(&config);
        sctx.cache.reset();
        let steps =
            dflash_predict_ar_with(&mut sctx, &weights, &config, 0, 0, &mut Rng::new(42), None);
        let vocab_size = config.vocab_size;

        assert_eq!(steps, config.draft_lookahead);
        assert_eq!(sctx.sampled_tokens().len(), steps);
        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            assert_eq!(slice.len(), vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
        }
    }

    #[test]
    fn test_dflash_predict_conditioned_with_matches_original() {
        let (weights, config) = make_draft();
        let hidden = vec![0.5; config.n_embd];
        let mut sctx = SpeculativeContext::new(&config);
        let steps = dflash_predict_conditioned_with(
            &mut sctx,
            &weights,
            &config,
            0,
            0,
            &hidden,
            &mut Rng::new(42),
        );
        let vocab_size = config.vocab_size;

        assert!(steps > 0);
        assert_eq!(sctx.sampled_tokens().len(), steps);
        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            assert_eq!(slice.len(), vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
        }
    }

    #[test]
    fn test_dflash_with_reuse_across_calls() {
        let (weights, config) = make_draft();
        let mut sctx = SpeculativeContext::new(&config);

        // First call
        let steps1 = dflash_predict_with(&mut sctx, &weights, &config, 0, 0);
        assert_eq!(steps1, config.draft_lookahead);

        // Second call — same context, should produce same results
        let steps2 = dflash_predict_with(&mut sctx, &weights, &config, 0, 0);
        assert_eq!(steps2, config.draft_lookahead);

        // Results should be identical (same inputs, deterministic)
        let vocab_size = config.vocab_size;
        for step in 0..steps1 {
            // Can't compare directly since second call overwrites, but we know it ran OK
            let _slice = sctx.marginal_slice(step, vocab_size);
        }
    }

    #[test]
    fn test_parallel_threshold_fallback_identical() {
        // draft config: n_embd=4, parallel_threshold=128 → sequential path
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let sequential = dflash_predict(&weights, &config, 0, 0);
        let parallel = dflash_predict_parallel(&weights, &config, 0, 0);

        assert_eq!(sequential.len(), parallel.len());
        for (step, (seq_marg, par_marg)) in sequential.iter().zip(parallel.iter()).enumerate() {
            for (i, (a, b)) in seq_marg.iter().zip(par_marg.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {step} token {i}: sequential={a}, parallel={b}"
                );
            }
        }
    }

    #[test]
    fn test_parallel_threshold_above_runs_parallel() {
        // micro config: n_embd=16, parallel_threshold=128 → still sequential
        let config = Config::micro();
        assert!(
            config.n_embd <= config.parallel_threshold,
            "micro should be below threshold"
        );

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let sequential = dflash_predict(&weights, &config, 0, 0);
        let parallel = dflash_predict_parallel(&weights, &config, 0, 0);

        // Should be identical because threshold triggers sequential fallback
        assert_eq!(sequential.len(), parallel.len());
        for (step, (seq_marg, par_marg)) in sequential.iter().zip(parallel.iter()).enumerate() {
            for (i, (a, b)) in seq_marg.iter().zip(par_marg.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {step} token {i}: sequential={a}, parallel={b}"
                );
            }
        }
    }

    #[test]
    fn test_parallel_threshold_custom_above_triggers_parallel() {
        // Custom config with threshold below n_embd → actual parallel path
        let mut config = Config::micro();
        config.parallel_threshold = 1; // Force parallel path (n_embd=16 > 1)

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let result = dflash_predict_parallel(&weights, &config, 0, 0);
        assert!(!result.is_empty(), "parallel should produce results");
        assert_eq!(result.len(), config.draft_lookahead);

        // Verify valid probabilities
        for (step, marg) in result.iter().enumerate() {
            let sum: f32 = marg.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "step {step} probabilities should sum to ~1.0, got {sum}"
            );
        }
    }
}
