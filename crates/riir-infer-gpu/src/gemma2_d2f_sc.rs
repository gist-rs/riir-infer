//! Self-Conditioning (SC) support for D2F block-causal decode (Plan 250 T1-T4).
//!
//! Implements the self-conditioning mechanism from arXiv:2604.02028 for the
//! Gemma2 D2F denoising loop. The previous step's x̂_0 prediction (expected
//! clean-token embedding) is concatenated with the current x_t embedding and
//! projected back to `n_embd` via a learnable matrix `W_sc`.
//!
//! # Architecture
//!
//! ```text
//! D2F Denoising Step (with SC)
//! ├── Embedding lookup:  x_t = wte[token] * sqrt(n_embd)
//! ├── SC projection:     h = W_sc @ [x_t ‖ x̂_0_prev]   (2·n_embd → n_embd)
//! ├── Transformer layers (unchanged)
//! └── Logits → softmax → x̂_0 estimate for next step
//! ```
//!
//! # W_sc Initialization
//!
//! `W_sc` is initialized as `[I_n | 0_n]` (identity for x_t, zeros for SC input).
//! This guarantees **zero behavioral change** when SC is enabled but untrained:
//! `W_sc @ [x_t ‖ 0] = I·x_t + 0·0 = x_t`. LoRA training updates `W_sc` to
//! learn how to incorporate the SC signal.
//!
//! # Training (T3)
//!
//! During LoRA training, with probability `sc_probability`:
//! 1. Forward pass without SC → get x̂_0 logits
//! 2. Compute x̂_0 estimate (stop-gradient — treated as constant input)
//! 3. Forward pass with SC using x̂_0 estimate → SC-conditioned logits
//! 4. Loss = MSE(SC-conditioned logits, target)
//!
//! With probability `1 - sc_probability`: SC input = zeros (no conditioning).
//!
//! # Inference (T4)
//!
//! - First denoising step: SC input = None (no previous prediction).
//! - All subsequent steps: SC input = previous step's x̂_0 estimate.
//!
//! TL;DR: SC feeds the previous step's prediction back as additional input.
//! W_sc = [I|0] init means zero overhead until trained. Paper: +5-7× quality.

// ── SC Configuration ──────────────────────────────────────────────

/// Configuration for self-conditioning in D2F decode.
///
/// When `enabled` is false, all SC machinery is bypassed (zero overhead).
/// When `enabled` is true, the decode loop tracks the previous step's x̂_0
/// estimate and feeds it through `W_sc` projection.
#[derive(Clone, Copy, Debug)]
pub struct D2fScConfig {
    /// Master switch. When false, decode behaves identically to non-SC.
    pub enabled: bool,
    /// Temperature for softmax when computing x̂_0 estimate.
    /// Default 1.0. Lower = sharper (argmax-like), higher = smoother.
    pub x0_temperature: f32,
}

impl Default for D2fScConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            x0_temperature: 1.0,
        }
    }
}

impl D2fScConfig {
    /// Create an enabled SC config with paper defaults.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Create a disabled SC config (zero overhead).
    pub fn disabled() -> Self {
        Self::default()
    }
}

// ── SC State (per decode call) ────────────────────────────────────

/// Per-decode-call state for self-conditioning.
///
/// Tracks the previous denoising step's x̂_0 estimate (expected clean-token
/// embedding per position). On the first step, `prev_x0_estimate` is `None`
/// — the forward pass runs without SC input.
///
/// After each step, call [`D2fScState::update_from_logits()`] to compute and store the
/// x̂_0 estimate for the next step.
#[derive(Clone, Debug)]
pub struct D2fScState {
    /// Previous step's x̂_0 estimate per position: `[seq_len][n_embd]`.
    /// `None` on the first step (no previous prediction available).
    pub prev_x0_estimate: Option<Vec<Vec<f32>>>,
    /// Configuration for x̂_0 computation.
    pub config: D2fScConfig,
}

impl D2fScState {
    /// Create a fresh SC state for a new decode call (first step has no SC input).
    pub fn new(config: D2fScConfig) -> Self {
        Self {
            prev_x0_estimate: None,
            config,
        }
    }

    /// Whether SC input is available for the current step.
    /// Returns `false` on the first step (no previous prediction).
    #[inline]
    pub fn has_sc_input(&self) -> bool {
        self.prev_x0_estimate.is_some()
    }

    /// Get the SC input for the forward pass, if available.
    /// Returns `None` on the first step.
    #[inline]
    pub fn sc_input(&self) -> Option<&[Vec<f32>]> {
        self.prev_x0_estimate.as_deref()
    }

    /// Update the x̂_0 estimate from the current step's logits.
    ///
    /// Computes the expected embedding per position:
    /// `x̂_0[pos] = softmax(logits[pos] / τ) @ wte`
    ///
    /// The mask token is excluded from the softmax (it represents "unknown",
    /// not a valid clean-token prediction).
    ///
    /// # Arguments
    ///
    /// * `all_logits` — Per-position logits from the forward pass.
    /// * `wte_cpu` — CPU embedding table, shape `[vocab_size * n_embd]`.
    /// * `n_embd` — Embedding dimension.
    /// * `mask_token_id` — Token ID to exclude from softmax (the `[MASK]` token).
    pub fn update_from_logits(
        &mut self,
        all_logits: &[Vec<f32>],
        wte_cpu: &[f32],
        n_embd: usize,
        mask_token_id: usize,
    ) {
        let seq_len = all_logits.len();
        if seq_len == 0 {
            self.prev_x0_estimate = Some(Vec::new());
            return;
        }

        let mut estimates = Vec::with_capacity(seq_len);
        for logits in &all_logits[..seq_len] {
            let estimate = compute_x0_estimate(logits, wte_cpu, n_embd, mask_token_id, self.config.x0_temperature);
            estimates.push(estimate);
        }
        self.prev_x0_estimate = Some(estimates);
    }

    /// Reset state for a new decode sequence.
    pub fn reset(&mut self) {
        self.prev_x0_estimate = None;
    }
}

// ── x̂_0 Estimate Computation ─────────────────────────────────────

/// Compute the x̂_0 estimate (expected embedding) from logits.
///
/// `x̂_0 = softmax(logits / τ, exclude=mask) @ wte`
///
/// This is the "soft token" representation: instead of committing to a single
/// predicted token, we take the probability-weighted average of all token
/// embeddings. This gives a smooth, differentiable estimate of the clean input.
///
/// # Stop-Gradient
///
/// In training, this estimate should be detached (treated as constant input
/// to the next forward pass). Since this is a CPU computation with no autograd
/// graph, detachment is implicit — the result is plain `Vec<f32>` data.
///
/// # Arguments
///
/// * `logits` — Vocab-sized logit vector for one position.
/// * `wte_cpu` — CPU embedding table, shape `[vocab_size * n_embd]`.
/// * `n_embd` — Embedding dimension.
/// * `mask_token_id` — Token to exclude from softmax.
/// * `temperature` — Softmax temperature (1.0 = standard).
pub fn compute_x0_estimate(
    logits: &[f32],
    wte_cpu: &[f32],
    n_embd: usize,
    mask_token_id: usize,
    temperature: f32,
) -> Vec<f32> {
    let vocab = logits.len();
    if vocab == 0 || n_embd == 0 {
        return vec![0.0f32; n_embd];
    }

    // Numerically stable softmax over non-mask tokens. The max must be
    // folded over the SAME restricted support (Issue 695 H20): the old
    // all-tokens max let a dominant [MASK] logit drive every remaining
    // exponent to underflow (sum_exp → 0), silently zeroing the estimate
    // even though the non-mask support was perfectly normal.
    let max_logit = logits
        .iter()
        .enumerate()
        .filter(|(t, _)| *t != mask_token_id)
        .map(|(_, &l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = if temperature > 0.0 { 1.0 / temperature } else { 1.0 };

    // Table-size check hoisted out of the vocab-sized hot loop (Issue 695
    // H21 — was a per-token compare across 256k iterations).
    if wte_cpu.len() < vocab * n_embd {
        // Embedding table smaller than expected — return zeros.
        return vec![0.0f32; n_embd];
    }

    let mut sum_exp = 0.0f32;
    // First pass: compute sum of exponentials.
    #[allow(clippy::needless_range_loop, reason = "skip-by-index: t == mask_token_id continue; cleaner than filter on enumerated index")]
    for t in 0..vocab {
        if t == mask_token_id {
            continue;
        }
        let scaled = (logits[t] - max_logit) * inv_temp;
        sum_exp += scaled.exp();
    }
    if sum_exp <= 0.0 {
        return vec![0.0f32; n_embd];
    }

    // Second pass: accumulate weighted embedding.
    // x̂_0[i] = Σ_t softmax_t * wte[t * n_embd + i]
    let mut estimate = vec![0.0f32; n_embd];
    let inv_sum = 1.0 / sum_exp;
    #[allow(clippy::needless_range_loop, reason = "skip-by-index: t == mask_token_id continue")]
    for t in 0..vocab {
        if t == mask_token_id {
            continue;
        }
        let scaled = (logits[t] - max_logit) * inv_temp;
        let prob = scaled.exp() * inv_sum;
        if prob <= 0.0 {
            continue;
        }
        let tok_off = t * n_embd;
        for (i, est) in estimate.iter_mut().enumerate() {
            *est += prob * wte_cpu[tok_off + i];
        }
    }

    estimate
}

// ── W_sc Projection ───────────────────────────────────────────────

/// Initialize `W_sc` as `[I_n | 0_n]` (identity for x_t, zeros for SC input).
///
/// Shape: `(n_embd, 2 * n_embd)` stored row-major.
/// Row `i`: `W_sc[i][j] = 1.0` if `j == i`, else `0.0`.
///
/// This guarantees that an untrained `W_sc` produces zero behavioral change:
/// `W_sc @ [x_t ‖ sc] = I·x_t + 0·sc = x_t`.
///
/// # Arguments
///
/// * `n_embd` — Embedding dimension (input x_t dimension).
///
/// # Returns
///
/// Flat vector of length `n_embd * 2 * n_embd`.
pub fn init_w_sc_identity_padded(n_embd: usize) -> Vec<f32> {
    let cols = 2 * n_embd;
    let mut w_sc = vec![0.0f32; n_embd * cols];
    // Identity block in the first n_embd columns.
    for i in 0..n_embd {
        w_sc[i * cols + i] = 1.0;
    }
    w_sc
}

/// Check whether a flat W_sc buffer still matches the identity-padded init
/// `[I_n | 0_n]` (row-major, shape `(n_embd, 2*n_embd)`).
///
/// Used as a fast-path gate: when W_sc is still at its untrained init, the
/// SC projection is provably the identity on `x_t` (`W_sc @ [x_t ‖ sc] = x_t`),
/// so callers can skip the O(n²) matmul entirely. After LoRA training updates
/// W_sc, this returns `false` and the full projection runs.
///
/// Complexity: O(n²) worst case. Callers should cache the verdict once per
/// W_sc allocation rather than re-checking per position/per step.
pub fn is_identity_padded(w_sc: &[f32], n_embd: usize) -> bool {
    let cols = 2 * n_embd;
    if w_sc.len() != n_embd * cols {
        return false;
    }
    for i in 0..n_embd {
        let row_off = i * cols;
        if w_sc[row_off + i] != 1.0 {
            return false;
        }
        for j in 0..cols {
            if j != i && w_sc[row_off + j] != 0.0 {
                return false;
            }
        }
    }
    true
}

/// Apply the SC projection: concatenate `x_t` and `sc_input`, project with `W_sc`.
///
/// `output[i] = Σ_{j=0}^{n-1} W_sc[i][j] * x_t[j] + Σ_{j=0}^{n-1} W_sc[i][n+j] * sc[j]`
///
/// When `W_sc = [I | 0]` (untrained), this returns `x_t` unchanged.
///
/// # Arguments
///
/// * `x_t` — Current step's token embedding (length `n_embd`).
/// * `sc_input` — Previous step's x̂_0 estimate (length `n_embd`).
/// * `w_sc` — Projection matrix, flat `(n_embd, 2*n_embd)`.
/// * `n_embd` — Embedding dimension.
///
/// # Returns
///
/// Projected embedding of length `n_embd`.
///
/// # Panics
///
/// Panics in debug if lengths don't match.
pub fn project_sc(
    x_t: &[f32],
    sc_input: &[f32],
    w_sc: &[f32],
    n_embd: usize,
) -> Vec<f32> {
    debug_assert_eq!(x_t.len(), n_embd, "x_t length must equal n_embd");
    debug_assert_eq!(sc_input.len(), n_embd, "sc_input length must equal n_embd");
    debug_assert_eq!(
        w_sc.len(),
        n_embd * 2 * n_embd,
        "w_sc length must equal n_embd * 2 * n_embd"
    );

    let cols = 2 * n_embd;
    let mut output = vec![0.0f32; n_embd];

    #[allow(clippy::needless_range_loop, reason = "matmul: i selects row i*cols of w_sc")]
    for i in 0..n_embd {
        let row_off = i * cols;
        let mut acc = 0.0f32;
        // First n_embd columns: x_t contribution.
        for j in 0..n_embd {
            acc += w_sc[row_off + j] * x_t[j];
        }
        // Second n_embd columns: sc_input contribution.
        for j in 0..n_embd {
            acc += w_sc[row_off + n_embd + j] * sc_input[j];
        }
        output[i] = acc;
    }

    output
}

/// Apply the SC projection in-place (writes into the provided output buffer).
///
/// Same as [`project_sc`] but writes into a pre-allocated buffer to avoid
/// allocation per position. The buffer must have length `n_embd`.
///
/// # Implementation note (Issue 695 H19)
///
/// The accumulation is TWO separate loops per row — one over the x_t half
/// of `w_sc[row_off..row_off+n]`, one over the sc_input half
/// `w_sc[row_off+n..row_off+2n]` — both contiguous, so W_sc access stays
/// row-major sequential across the whole row. An earlier doc claimed a
/// single linear scan against a pre-built concatenated `[x_t ‖ sc_input]`
/// scratch view; neither exists (the concatenation would be the
/// transposed-GEMV form proposed in the issue's fix direction — see
/// `expected-embedding-transposed-gemv-resident-head`).
pub fn project_sc_into(
    x_t: &[f32],
    sc_input: &[f32],
    w_sc: &[f32],
    n_embd: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), n_embd, "output length must equal n_embd");
    debug_assert_eq!(x_t.len(), n_embd, "x_t length must equal n_embd");
    debug_assert_eq!(sc_input.len(), n_embd, "sc_input length must equal n_embd");
    debug_assert_eq!(
        w_sc.len(),
        n_embd * 2 * n_embd,
        "w_sc length must equal n_embd * 2 * n_embd"
    );

    let cols = 2 * n_embd;
    #[allow(clippy::needless_range_loop, reason = "matmul: i selects row i*cols of w_sc")]
    for i in 0..n_embd {
        let row_off = i * cols;
        let mut acc = 0.0f32;
        // x_t contribution (first n_embd columns of the row).
        let w_a = &w_sc[row_off..row_off + n_embd];
        for j in 0..n_embd {
            acc += w_a[j] * x_t[j];
        }
        // sc_input contribution (second n_embd columns of the row).
        let w_b = &w_sc[row_off + n_embd..row_off + cols];
        for j in 0..n_embd {
            acc += w_b[j] * sc_input[j];
        }
        output[i] = acc;
    }
}

// ── Training Helpers (T3) ─────────────────────────────────────────

/// Result of a single SC training forward pass pair.
///
/// Implements the two-forward-pass SC training pattern from the paper:
/// 1. Forward without SC → get baseline x̂_0 logits
/// 2. Forward with SC (using x̂_0 from step 1) → get SC-conditioned logits
///
/// The SC-conditioned logits are used for loss computation. The baseline
/// logits are only used to compute the x̂_0 estimate (stop-gradient).
#[derive(Clone, Debug)]
pub struct ScTrainingForwardResult {
    /// Logits from the SC-conditioned forward pass (used for loss).
    pub sc_conditioned_logits: Vec<Vec<f32>>,
    /// Whether SC was actually applied this step.
    /// When `false`, `sc_conditioned_logits` are from a no-SC forward pass.
    pub sc_applied: bool,
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── D2fScConfig ────────────────────────────────────────────────

    #[test]
    fn config_disabled_by_default() {
        let config = D2fScConfig::default();
        assert!(!config.enabled);
        assert!((config.x0_temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn config_enabled_sets_flag() {
        let config = D2fScConfig::enabled();
        assert!(config.enabled);
    }

    #[test]
    fn config_disabled_explicit() {
        let config = D2fScConfig::disabled();
        assert!(!config.enabled);
    }

    // ── D2fScState ─────────────────────────────────────────────────

    #[test]
    fn state_new_has_no_sc_input() {
        let state = D2fScState::new(D2fScConfig::enabled());
        assert!(!state.has_sc_input());
        assert!(state.sc_input().is_none());
    }

    #[test]
    fn state_update_populates_sc_input() {
        let mut state = D2fScState::new(D2fScConfig::enabled());
        // Fake logits: 3 positions, 4 vocab.
        let logits = vec![
            vec![1.0, 2.0, 3.0, 0.1], // token 0
            vec![0.5, 0.5, 0.5, 0.5], // uniform
            vec![0.0, 0.0, 10.0, 0.0], // peaked at token 2
        ];
        // Fake wte: 4 tokens × 2 dims.
        let wte = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            1.0, 1.0, // token 2
            0.5, 0.5, // token 3 (mask)
        ];
        state.update_from_logits(&logits, &wte, 2, 3 /* mask_token_id=3 */);

        assert!(state.has_sc_input());
        let sc = state.sc_input().expect("sc input should exist");
        assert_eq!(sc.len(), 3);
        assert_eq!(sc[0].len(), 2); // n_embd = 2

        // Position 2 (peaked at token 2, embedding [1,1]) should be ≈ [1,1].
        assert!(
            (sc[2][0] - 1.0).abs() < 0.01,
            "peaked position should match token 2 embedding x, got {}",
            sc[2][0]
        );
        assert!(
            (sc[2][1] - 1.0).abs() < 0.01,
            "peaked position should match token 2 embedding y, got {}",
            sc[2][1]
        );
    }

    #[test]
    fn state_reset_clears_sc_input() {
        let mut state = D2fScState::new(D2fScConfig::enabled());
        state.prev_x0_estimate = Some(vec![vec![1.0; 4]; 2]);
        assert!(state.has_sc_input());

        state.reset();
        assert!(!state.has_sc_input());
        assert!(state.sc_input().is_none());
    }

    #[test]
    fn state_update_empty_logits() {
        let mut state = D2fScState::new(D2fScConfig::enabled());
        state.update_from_logits(&[], &[], 4, 0);
        assert!(state.has_sc_input());
        assert_eq!(state.sc_input().unwrap().len(), 0);
    }

    // ── compute_x0_estimate ────────────────────────────────────────

    #[test]
    fn x0_estimate_peaked_logits() {
        // Logits strongly peaked at token 1.
        let logits = vec![-10.0, 10.0, -10.0, -10.0];
        // wte: 4 tokens × 2 dims.
        let wte = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1 ← peaked here
            1.0, 1.0, // token 2
            0.5, 0.5, // token 3
        ];
        let estimate = compute_x0_estimate(&logits, &wte, 2, 99, 1.0);
        // Should be ≈ [0, 1] (token 1's embedding).
        assert!((estimate[0] - 0.0).abs() < 0.01, "x should be ≈0, got {}", estimate[0]);
        assert!((estimate[1] - 1.0).abs() < 0.01, "y should be ≈1, got {}", estimate[1]);
    }

    #[test]
    fn x0_estimate_uniform_logits() {
        // Uniform logits → average of all non-mask token embeddings.
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let wte = vec![
            2.0, 0.0, // token 0
            0.0, 2.0, // token 1
            2.0, 2.0, // token 2
            0.0, 0.0, // token 3 (mask — excluded)
        ];
        let estimate = compute_x0_estimate(&logits, &wte, 2, 3, 1.0);
        // Average of tokens 0,1,2: ([2,0] + [0,2] + [2,2]) / 3 = [4/3, 4/3]
        assert!(
            (estimate[0] - 4.0 / 3.0).abs() < 0.01,
            "x should be ≈4/3, got {}",
            estimate[0]
        );
        assert!(
            (estimate[1] - 4.0 / 3.0).abs() < 0.01,
            "y should be ≈4/3, got {}",
            estimate[1]
        );
    }

    #[test]
    fn x0_estimate_excludes_mask_token() {
        // Mask token has huge logit but should be excluded.
        let logits = vec![0.0, 0.0, 0.0, 100.0];
        let wte = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            1.0, 1.0, // token 2
            99.0, 99.0, // token 3 (mask — must be excluded!)
        ];
        let estimate = compute_x0_estimate(&logits, &wte, 2, 3, 1.0);
        // Should be average of tokens 0,1,2 — NOT influenced by token 3.
        // ([1,0] + [0,1] + [1,1]) / 3 = [2/3, 2/3].
        //
        // Issue 695 H20 regression pin: the OLD code folded the stability
        // max over ALL tokens including the mask — with this exact input the
        // dominant [MASK] logit drove every non-mask exponent to underflow
        // (exp(-100) = 0), sum_exp hit 0, and the estimate silently returned
        // ZEROS, which still passed the old `< 2.0` assertion vacuously.
        // The max must come from the restricted support.
        assert!(
            (estimate[0] - 2.0 / 3.0).abs() < 1e-5,
            "x should be ≈2/3 (avg of non-mask tokens), got {}",
            estimate[0]
        );
        assert!(
            (estimate[1] - 2.0 / 3.0).abs() < 1e-5,
            "y should be ≈2/3 (avg of non-mask tokens), got {}",
            estimate[1]
        );
    }

    #[test]
    fn x0_estimate_empty_logits() {
        let estimate = compute_x0_estimate(&[], &[], 4, 0, 1.0);
        assert_eq!(estimate, vec![0.0f32; 4]);
    }

    #[test]
    fn x0_estimate_zero_n_embd() {
        let estimate = compute_x0_estimate(&[1.0, 2.0], &[1.0, 2.0], 0, 0, 1.0);
        assert!(estimate.is_empty());
    }

    #[test]
    fn x0_estimate_low_temperature_sharpens() {
        // Bimodal logits: token 0 slightly higher than token 1.
        let logits = vec![1.1, 1.0, -5.0, -5.0];
        let wte = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            0.0, 0.0, // token 2
            0.0, 0.0, // token 3
        ];
        let high_temp = compute_x0_estimate(&logits, &wte, 2, 99, 10.0);
        let low_temp = compute_x0_estimate(&logits, &wte, 2, 99, 0.1);

        // Low temp → more peaked at token 0 → x closer to 1, y closer to 0.
        assert!(
            low_temp[0] > high_temp[0],
            "low temp should sharpen toward token 0 (higher x), got low={} high={}",
            low_temp[0],
            high_temp[0]
        );
    }

    // ── init_w_sc_identity_padded ──────────────────────────────────

    #[test]
    fn w_sc_identity_padded_shape() {
        let w_sc = init_w_sc_identity_padded(4);
        // Shape: 4 × 8 = 32 elements.
        assert_eq!(w_sc.len(), 4 * 2 * 4);
    }

    #[test]
    fn w_sc_identity_padded_first_block_is_identity() {
        let w_sc = init_w_sc_identity_padded(3);
        let cols = 2 * 3;
        // Check identity block (first n_embd=3 columns).
        // Row 0: [1, 0, 0, ...]
        assert!((w_sc[0] - 1.0).abs() < f32::EPSILON);
        assert!((w_sc[1] - 0.0).abs() < f32::EPSILON);
        assert!((w_sc[2] - 0.0).abs() < f32::EPSILON);
        // Row 1: [0, 1, 0, ...]
        assert!((w_sc[cols] - 0.0).abs() < f32::EPSILON);
        assert!((w_sc[cols + 1] - 1.0).abs() < f32::EPSILON);
        assert!((w_sc[cols + 2] - 0.0).abs() < f32::EPSILON);
        // Row 2: [0, 0, 1, ...]
        assert!((w_sc[2 * cols] - 0.0).abs() < f32::EPSILON);
        assert!((w_sc[2 * cols + 1] - 0.0).abs() < f32::EPSILON);
        assert!((w_sc[2 * cols + 2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn w_sc_identity_padded_second_block_is_zero() {
        let w_sc = init_w_sc_identity_padded(3);
        let cols = 2 * 3;
        // Check zero block (second n_embd=3 columns).
        for i in 0..3 {
            for j in 3..6 {
                assert!(
                    (w_sc[i * cols + j] - 0.0).abs() < f32::EPSILON,
                    "second block should be zero at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn w_sc_identity_padded_zero_n_embd() {
        let w_sc = init_w_sc_identity_padded(0);
        assert!(w_sc.is_empty());
    }

    // ── project_sc ─────────────────────────────────────────────────

    #[test]
    fn project_sc_identity_w_returns_x_t() {
        let n = 3;
        let w_sc = init_w_sc_identity_padded(n);
        let x_t = vec![1.0, 2.0, 3.0];
        let sc_input = vec![10.0, 20.0, 30.0]; // Should be zeroed out by W_sc.
        let output = project_sc(&x_t, &sc_input, &w_sc, n);
        // With [I|0], output = x_t.
        for i in 0..n {
            assert!(
                (output[i] - x_t[i]).abs() < 1e-5,
                "output[{i}] should be x_t[{i}]={}, got {}",
                x_t[i],
                output[i]
            );
        }
    }

    #[test]
    fn project_sc_zero_sc_returns_x_t() {
        let n = 4;
        let w_sc = init_w_sc_identity_padded(n);
        let x_t = vec![0.5, 1.5, 2.5, 3.5];
        let sc_input = vec![0.0; n];
        let output = project_sc(&x_t, &sc_input, &w_sc, n);
        for i in 0..n {
            assert!((output[i] - x_t[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn project_sc_full_identity_2n() {
        // W_sc = [I | I] → output = x_t + sc_input.
        let n = 2;
        let mut w_sc = vec![0.0f32; n * 2 * n];
        let cols = 2 * n;
        // First block: identity.
        for i in 0..n {
            w_sc[i * cols + i] = 1.0;
        }
        // Second block: identity.
        for i in 0..n {
            w_sc[i * cols + n + i] = 1.0;
        }
        let x_t = vec![1.0, 2.0];
        let sc_input = vec![3.0, 4.0];
        let output = project_sc(&x_t, &sc_input, &w_sc, n);
        assert!((output[0] - 4.0).abs() < 1e-5); // 1+3
        assert!((output[1] - 6.0).abs() < 1e-5); // 2+4
    }

    #[test]
    fn project_sc_scaled() {
        // W_sc = [0.5*I | 0.5*I] → output = 0.5*x_t + 0.5*sc.
        let n = 2;
        let mut w_sc = vec![0.0f32; n * 2 * n];
        let cols = 2 * n;
        for i in 0..n {
            w_sc[i * cols + i] = 0.5;
            w_sc[i * cols + n + i] = 0.5;
        }
        let x_t = vec![2.0, 4.0];
        let sc_input = vec![6.0, 8.0];
        let output = project_sc(&x_t, &sc_input, &w_sc, n);
        assert!((output[0] - 4.0).abs() < 1e-5); // 0.5*2 + 0.5*6 = 4
        assert!((output[1] - 6.0).abs() < 1e-5); // 0.5*4 + 0.5*8 = 6
    }

    #[test]
    fn project_sc_into_matches_project_sc() {
        let n = 4;
        let w_sc = init_w_sc_identity_padded(n);
        let x_t = vec![1.0, 2.0, 3.0, 4.0];
        let sc_input = vec![0.1, 0.2, 0.3, 0.4];

        let allocated = project_sc(&x_t, &sc_input, &w_sc, n);
        let mut in_place = vec![0.0f32; n];
        project_sc_into(&x_t, &sc_input, &w_sc, n, &mut in_place);

        for i in 0..n {
            assert!((allocated[i] - in_place[i]).abs() < 1e-7);
        }
    }

    // ── Integration: SC state + projection ─────────────────────────

    #[test]
    fn sc_loop_two_steps() {
        // Simulate two denoise steps:
        // Step 0: no SC input → forward → update state.
        // Step 1: SC input from step 0 → forward → update state.
        let n_embd = 2;
        let mask_id = 3;

        // Fake wte.
        let wte = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            1.0, 1.0, // token 2
            0.5, 0.5, // token 3 (mask)
        ];

        let mut state = D2fScState::new(D2fScConfig::enabled());

        // Step 0: no SC.
        assert!(!state.has_sc_input());
        let logits_step0 = vec![
            vec![10.0, -5.0, -5.0, -5.0], // peaked at token 0 → x̂_0 ≈ [1, 0]
        ];
        state.update_from_logits(&logits_step0, &wte, n_embd, mask_id);

        // Step 1: SC available.
        assert!(state.has_sc_input());
        let sc = state.sc_input().unwrap();
        assert_eq!(sc.len(), 1);
        assert_eq!(sc[0].len(), n_embd);
        // x̂_0 from step 0 should be ≈ [1, 0] (token 0's embedding).
        assert!((sc[0][0] - 1.0).abs() < 0.01);
        assert!((sc[0][1] - 0.0).abs() < 0.01);

        // Project with identity W_sc → should recover x_t, ignore SC.
        let w_sc = init_w_sc_identity_padded(n_embd);
        let x_t = vec![0.7, 0.3];
        let projected = project_sc(&x_t, &sc[0], &w_sc, n_embd);
        // With [I|0], output = x_t = [0.7, 0.3].
        assert!((projected[0] - 0.7).abs() < 1e-5);
        assert!((projected[1] - 0.3).abs() < 1e-5);
    }
}
