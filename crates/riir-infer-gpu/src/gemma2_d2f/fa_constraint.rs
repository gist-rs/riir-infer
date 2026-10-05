//! FA-constrained exact posterior decoding over the [`super::d2f_decode_gemma2`]
//! denoising loop (issue 035 P1) — the `propose_x0` arm behind the opt-in
//! `fa_constraint` feature.
//!
//! Per denoising step the loop proposes x\u{0} with ONE exact joint draw over
//! the generation block ([`riir_infer_core::fa_posterior`]): committed
//! positions are pinned (`forced`), masked positions are free. Proposals are
//! scored, ranked confidence-first, and committed while the accept rule
//! holds (fixed threshold on the axis confidence, or a learned sampler);
//! the rest remask. Because every draw is conditioned on all prior commits,
//! the block's final token assignment is a valid automaton walk from the
//! configured start node — a guarantee the unconstrained lane cannot make
//! (per-position draws do not respect joint constraints, and a
//! SemiActivated block leaves mask placeholders).
//!
//! The two commit axes (issue 035 P1's `commit_by`):
//!
//! * [`FaCommitBy::RawTop1`] — the A/B baseline: `P_lm(proposed | position)`
//!   from the mask-suppressed softmax over the full vocab. Same confidence
//!   currency as the base loop (max softmax prob), read at the proposal.
//! * [`FaCommitBy::ConstrainedMarginal`] — the constrained posterior
//!   marginal of the proposal; its natural log rides
//!   [`super::SamplerFeatures::marginal_log`] as the extra sampler input
//!   (the learned posture is [`ConstrainedSampler`], 7 params). Builds the
//!   O(log L) segment tree each step — mind the state-count budget
//!   ([`FaDecodeError::TreeBudgetExceeded`]).
//!
//! Block carry-over: the caller threads [`Gemma2D2fFaResult::final_node`]
//! into the next block's [`FaConstraintConfig::start_node`]; the sampler
//! starts the walk there instead of the automaton's designated start.
//!
//! ⚠ Deliberately a separate loop from [`super::d2f_decode_gemma2`] (the
//! byte-identical feature-off law — the base sampling step is untouched).
//! The two share the forward dispatch, SC bookkeeping and result-assembly
//! shape but not the sampling step; drift risk is carried HERE. When the
//! base loop changes for a mechanical reason, mirror it into
//! [`d2f_decode_gemma2_constrained`] and re-run both suites.

use riir_infer_core::fa_posterior::{Automaton, FREE, FaError, FaScratch, SplitMix64, TreeScratch};
use thiserror::Error;

use super::{
    D2fBlockState, D2fScState, Gemma2D2fConfig, Gemma2D2fResult, GpuGemmaCubeCLD2F, SamplerFeatures,
};

/// Default scratch budget for the [`FaCommitBy::ConstrainedMarginal`] tree
/// (the tree levels + the marginals' forward/backward prefixes, all f64
/// N×N matrices — roughly `4 · len_pad · N² · 8` bytes).
pub const DEFAULT_TREE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// The commit-confidence axis (issue 035 P1's `commit_by`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaCommitBy {
    /// Unconstrained confidence of the proposed token — `P_lm(token | pos)`
    /// from the mask-suppressed softmax over the full vocab. The A/B
    /// baseline; runs the f32 sequential joint sampler unless
    /// [`FaConstraintConfig::parallel`] is set.
    RawTop1,
    /// Constrained posterior marginal `P(token | constraint)` of the
    /// proposed token; `marginal_log` feeds [`ConstrainedSampler`] (or the
    /// fixed threshold directly). Always builds the O(log L) segment tree —
    /// marginals are a tree product.
    ConstrainedMarginal,
}

/// Constrained-decode configuration: which automaton, where the walk starts
/// (the block carry-over), and how commits are scored. Construct via
/// [`FaConstraintConfig::new`] and override fields as needed.
pub struct FaConstraintConfig<'a> {
    /// The constraint automaton over the model's token space. Its edge
    /// tokens must be valid model token ids and must NEVER include the mask
    /// placeholder (validated at decode entry).
    pub automaton: &'a Automaton,
    /// Automaton node the block's walk starts from — the state reached by
    /// the ACCEPTED PREFIX from earlier blocks (issue 035 P1's carry-over).
    /// The first block passes `automaton.start()` (the `new` default).
    pub start_node: usize,
    /// Which confidence scores the commit rule.
    pub commit_by: FaCommitBy,
    /// Draw the proposal through the O(log L) parallel tree (RawTop1 only —
    /// ConstrainedMarginal always builds the tree).
    pub parallel: bool,
    /// Per-step commit budget: `Some(k)` commits at most the k
    /// highest-confidence accepted proposals (the mosaic confidence-decoder
    /// schedule); `None` commits every accepted proposal.
    pub commit_budget: Option<usize>,
    /// Scratch budget for the ConstrainedMarginal tree, bytes. Refused at
    /// decode entry ([`FaDecodeError::TreeBudgetExceeded`]) when the
    /// automaton's state count would exceed it.
    pub tree_budget_bytes: usize,
    /// Learned constrained commit sampler (the ConstrainedMarginal axis's
    /// 7-param posture). `None` = fixed `confidence_threshold` on the axis
    /// confidence. The base config's 6-param `DiffusionSampler` is the
    /// RAW axis's learned posture and is ignored on this axis.
    pub sampler: Option<ConstrainedSampler>,
}

impl<'a> FaConstraintConfig<'a> {
    /// Default configuration over `automaton`: designated start node,
    /// RawTop1 commit axis, sequential sampler, no budget, the default tree
    /// budget, no learned sampler.
    pub fn new(automaton: &'a Automaton) -> Self {
        Self {
            start_node: automaton.start(),
            commit_by: FaCommitBy::RawTop1,
            parallel: false,
            commit_budget: None,
            tree_budget_bytes: DEFAULT_TREE_BUDGET_BYTES,
            sampler: None,
            automaton,
        }
    }

    /// Config-level refusals, checked once at decode entry (fail loud before
    /// any GPU work, never OOM at step 3 of 8).
    pub fn validate(
        &self,
        mask_token_id: usize,
        block_size: usize,
        denoise_steps: usize,
    ) -> Result<(), FaDecodeError> {
        let fa = self.automaton;
        if self.start_node >= fa.n_nodes() {
            return Err(FaDecodeError::Sampler(FaError::BadStart(self.start_node)));
        }
        if denoise_steps == 0 {
            return Err(FaDecodeError::BadConfig);
        }
        // The automaton must never emit the mask placeholder: a committed
        // proposal equal to the mask would un-commit the position.
        if fa.allows(mask_token_id as u32) {
            return Err(FaDecodeError::MaskTokenInAutomaton(mask_token_id));
        }
        if self.commit_by == FaCommitBy::ConstrainedMarginal {
            let n = fa.n_nodes();
            let len_pad = block_size.max(1).next_power_of_two();
            // Tree levels (≈ 2·len_pad N×N f64) + the marginals pass's
            // forward/backward prefixes (≈ 2·(len_pad+1)) → 4·len_pad·N²·8.
            let bytes = 4usize
                .saturating_mul(len_pad)
                .saturating_mul(n)
                .saturating_mul(n)
                .saturating_mul(8);
            if bytes > self.tree_budget_bytes {
                return Err(FaDecodeError::TreeBudgetExceeded {
                    n_nodes: n,
                    len_pad,
                    bytes,
                    budget: self.tree_budget_bytes,
                });
            }
        }
        Ok(())
    }
}

/// Constrained-lane decode errors: the sampler's own errors plus the decode
/// wiring's config/contract refusals.
#[derive(Error, Debug, PartialEq)]
pub enum FaDecodeError {
    #[error(transparent)]
    Sampler(#[from] FaError),
    #[error(
        "mask token {0} appears in the constraint automaton — the automaton must never emit the mask placeholder"
    )]
    MaskTokenInAutomaton(usize),
    #[error(
        "constrained-marginal tree too large: {n_nodes} states × len_pad {len_pad} ≈ {bytes} bytes > budget {budget} — shrink the automaton, raise tree_budget_bytes, or use RawTop1"
    )]
    TreeBudgetExceeded {
        n_nodes: usize,
        len_pad: usize,
        bytes: usize,
        budget: usize,
    },
    #[error(
        "denoise_steps must be >= 1 — the constrained lane needs at least one forward to propose x0"
    )]
    BadConfig,
    #[error("contract violation: committed block does not walk to an accepting node (sampler bug)")]
    WalkBrokeContract,
}

/// The constrained commit sampler (the `commit_by = ConstrainedMarginal`
/// learned posture): the 6 base [`super::SamplerFeatures`] plus
/// `marginal_log` (the constrained marginal's natural log) as a 7th input —
/// 7 weights + bias, O(1) inference.
#[derive(Clone, Copy, Debug)]
pub struct ConstrainedSampler {
    weights: [f32; 7],
    bias: f32,
}

impl ConstrainedSampler {
    /// Create from trained weights + bias. Weights 0..6 are the base
    /// features' (same order as `SamplerFeatures::to_array`), weight 6 is
    /// `marginal_log`.
    pub fn from_weights(weights: [f32; 7], bias: f32) -> Self {
        Self { weights, bias }
    }

    /// `P(commit | features, marginal_log)` — sigmoid(w·x + b).
    pub fn predict(&self, features: &SamplerFeatures, marginal_log: f32) -> f64 {
        let x = features.to_array();
        let z: f64 = self.weights[..6]
            .iter()
            .zip(x.iter())
            .map(|(w, f)| (*w as f64) * (*f as f64))
            .sum::<f64>()
            + (self.weights[6] as f64) * (marginal_log as f64)
            + self.bias as f64;
        1.0 / (1.0 + (-z).exp())
    }

    /// Decide whether to commit the proposed token.
    pub fn decide(&self, features: &SamplerFeatures, marginal_log: f32, threshold: f64) -> bool {
        self.predict(features, marginal_log) >= threshold
    }
}

/// Constrained decode result: the base decode result plus the FA lane's
/// carry-over and disclosure fields.
#[derive(Clone, Debug)]
pub struct Gemma2D2fFaResult {
    /// The base result. `converged`/`state` reflect the CONFIDENCE loop
    /// only: the constrained lane additionally commits the final step's x\u{0}
    /// proposal for any still-masked position (accepted by construction),
    /// so the returned block NEVER carries mask tokens even when
    /// `converged == false` — the difference from the base lane.
    pub decode: Gemma2D2fResult,
    /// Automaton node reached by the returned block's tokens — thread into
    /// the next block's [`FaConstraintConfig::start_node`].
    pub final_node: usize,
    /// Deferred NaN→uniform fallbacks across the decode (the parallel
    /// lane's end-of-run counter; 0 on the sequential path).
    pub nan_fallbacks: u64,
    /// Proposals committed by the confidence rule across all steps (the
    /// final x\u{0} commit of still-masked positions not included).
    pub committed: usize,
}

/// Per-decode-call scratch: allocated once, reused across denoising steps.
struct FaDecodeScratch {
    seq: FaScratch,
    tree: TreeScratch,
    forced: Vec<u32>,
    block_logits: Vec<f32>,
    proposals: Vec<u32>,
    marginals: Vec<Vec<f64>>,
}

impl FaDecodeScratch {
    fn new(vocab: usize, block_size: usize) -> Self {
        Self {
            seq: FaScratch::new(),
            tree: TreeScratch::new(),
            forced: vec![FREE; block_size],
            block_logits: Vec::with_capacity(block_size * vocab),
            proposals: vec![0; block_size],
            marginals: Vec::new(),
        }
    }
}

/// The per-step joint proposal (issue 035 P1's `propose_x0`): flatten the
/// block's logits into scratch, build the `forced` vector (committed =
/// pinned, masked = [`FREE`]), and draw the joint — sequential f32 sampler
/// (`use_tree == false`) or the O(log L) tree (`use_tree`), which also
/// fills the per-position token marginals when `want_marginals`.
#[allow(clippy::too_many_arguments)]
fn propose_x0(
    fa: &Automaton,
    scratch: &mut FaDecodeScratch,
    start_node: usize,
    all_logits: &[Vec<f32>],
    block_start: usize,
    block_len: usize,
    tokens: &[usize],
    masked: &[bool],
    temperature: f32,
    use_tree: bool,
    want_marginals: bool,
    rng: &mut SplitMix64,
    nan_count: &mut u64,
) -> Result<(), FaError> {
    // Forced vector: committed positions pinned to their committed token.
    for i in 0..block_len {
        let pos = block_start + i;
        scratch.forced[i] = if masked[pos] {
            FREE
        } else {
            tokens[pos] as u32
        };
    }
    // Block logits flattened row-major (the sampler's contiguous layout).
    scratch.block_logits.clear();
    for row in &all_logits[block_start..block_start + block_len] {
        scratch.block_logits.extend_from_slice(row);
    }
    let logits = &scratch.block_logits[..];
    let forced = &scratch.forced[..block_len];
    let temp = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    // T <= 0 is the greedy posture: the PATH still comes from the exact
    // joint, the token draw takes the raw argmax per crossed edge.
    let argmax_tokens = !(temperature.is_finite() && temperature > 0.0);

    let final_node = if use_tree {
        let tree = fa.build_tree_from(
            &mut scratch.tree,
            start_node,
            logits,
            block_len,
            forced,
            temp,
        )?;
        scratch.marginals = if want_marginals {
            tree.marginals()
        } else {
            Vec::new()
        };
        tree.sample(argmax_tokens, rng, nan_count, &mut scratch.proposals)?
    } else {
        scratch.marginals.clear();
        fa.sample_joint_from(
            &mut scratch.seq,
            start_node,
            logits,
            block_len,
            forced,
            temp,
            argmax_tokens,
            rng,
            &mut scratch.proposals,
        )?
    };
    debug_assert!(fa.is_accept(final_node), "joint draw lands accepting");
    Ok(())
}

/// `P_lm(token | position)` from the mask-suppressed softmax — the RawTop1
/// axis confidence. Two O(vocab) passes (max, then sum + target), zero
/// allocation. Returns 0.0 for the degenerate all-mask row (and for a
/// token equal to the mask — the automaton validation makes that
/// unreachable through the decode loop).
fn lm_prob_of(logits: &[f32], token: usize, mask_token_id: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if i != mask_token_id && l > max {
            max = l;
        }
    }
    if max == f32::NEG_INFINITY {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut px = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        if i == mask_token_id {
            continue;
        }
        let e = (l - max).exp();
        sum += e;
        if i == token {
            px = e;
        }
    }
    if sum > 0.0 { px / sum } else { 0.0 }
}

/// Commit schedule: rank accepted (position, confidence) rows by
/// confidence descending — ties break to the LOWER position first so the
/// schedule is deterministic — apply the per-step budget, return the
/// positions to commit in commit order.
fn confidence_ordered(accepted: &[(usize, f32)], budget: Option<usize>) -> Vec<usize> {
    let mut rows = accepted.to_vec();
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let take = budget.unwrap_or(rows.len()).min(rows.len());
    rows[..take].iter().map(|&(p, _)| p).collect()
}

/// Iterative mask-and-refine under an automaton constraint: each step runs
/// the block-causal forward, proposes x\u{0} with ONE exact joint draw over
/// the block (committed positions pinned), scores every masked position on
/// the configured axis, commits accepted proposals confidence-first (up to
/// `commit_budget`), and remasks the rest. After the loop the still-masked
/// positions take the final step's x\u{0} proposal — accepted by
/// construction, so the returned block is ALWAYS a valid automaton walk
/// from `start_node` (the base lane would leave mask placeholders).
///
/// The base decode path ([`super::d2f_decode_gemma2`]) is untouched by this
/// lane — see the module doc for the deliberate-divergence note.
#[allow(clippy::too_many_arguments)]
pub fn d2f_decode_gemma2_constrained(
    gpu: &mut GpuGemmaCubeCLD2F,
    prompt: &[usize],
    mask_token_id: usize,
    decode_config: &Gemma2D2fConfig,
    constraint: &FaConstraintConfig<'_>,
    rng: &mut fastrand::Rng,
) -> Result<Gemma2D2fFaResult, FaDecodeError> {
    let block_size = decode_config.block_size;
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;
    let prompt_len_raw = prompt.len();
    constraint.validate(mask_token_id, block_size, max_steps)?;
    let fa = constraint.automaton;

    // Prompt-tail clamp — mirrors the base loop (Issue 695 H7): keep the
    // prompt's TAIL so the full generation block always fits max_seq.
    let max_seq = gpu.config().block_size;
    let prompt_tail_start = prompt_len_raw.saturating_sub(max_seq.saturating_sub(block_size));
    let mut tokens: Vec<usize> = prompt[prompt_tail_start..].to_vec();
    tokens.extend(std::iter::repeat_n(mask_token_id, block_size));
    let seq_len = tokens.len();
    let block_start = prompt_len_raw - prompt_tail_start;
    let prompt_len = block_start;
    assert!(
        seq_len <= max_seq,
        "seq_len {seq_len} exceeds model block_size {max_seq}"
    );

    let mut masked: Vec<bool> = vec![false; seq_len];
    masked[block_start..seq_len].fill(true);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut final_confidence = vec![0.0f32; seq_len];

    // SC state — mirrors the base loop (Plan 250 T4).
    let sc_enabled = decode_config.sc_config.enabled;
    let mut sc_state = sc_enabled.then(|| D2fScState::new(decode_config.sc_config));
    let n_embd = gpu.config().n_embd;

    let vocab = gpu.config().vocab_size;
    let mut scratch = FaDecodeScratch::new(vocab, block_size);
    let mut nan_fallbacks = 0u64;
    let mut committed_total = 0usize;
    let mut steps_used = 0usize;

    for step in 0..max_steps {
        let all_logits = match sc_state.as_ref() {
            Some(state) => gpu.block_causal_forward_with_sc(&tokens, prompt_len, Some(state)),
            None => gpu.block_causal_forward(&tokens, prompt_len),
        };
        steps_used = step + 1;

        // One exact joint proposal over the block.
        let use_tree =
            constraint.commit_by == FaCommitBy::ConstrainedMarginal || constraint.parallel;
        let want_marginals = constraint.commit_by == FaCommitBy::ConstrainedMarginal;
        propose_x0(
            fa,
            &mut scratch,
            constraint.start_node,
            &all_logits,
            block_start,
            block_size,
            &tokens,
            &masked,
            temperature,
            use_tree,
            want_marginals,
            &mut SplitMix64::new(rng.u64(..)),
            &mut nan_fallbacks,
        )?;

        // Score every masked position on the commit axis.
        let mut accepted: Vec<(usize, f32)> = Vec::with_capacity(block_size);
        for pos in block_start..seq_len {
            if !masked[pos] {
                continue;
            }
            let i = pos - block_start;
            let proposed = scratch.proposals[i] as usize;
            let logits_row = &all_logits[pos];
            let (conf, marginal_log) = match constraint.commit_by {
                FaCommitBy::RawTop1 => (lm_prob_of(logits_row, proposed, mask_token_id), 0.0),
                FaCommitBy::ConstrainedMarginal => {
                    let m = scratch.marginals[i][proposed];
                    let mlog = if m > 0.0 { m.ln() as f32 } else { -1e30 };
                    (m as f32, mlog)
                }
            };
            final_confidence[pos] = conf;

            let mut features = SamplerFeatures::from_logits(
                logits_row,
                step,
                max_steps,
                i,
                block_size,
                mask_token_id,
            );
            features.marginal_log = marginal_log;

            let take = match constraint.commit_by {
                // RAW axis: the base config's 6-param sampler is the learned
                // posture (identical to the base loop's rule), else threshold.
                FaCommitBy::RawTop1 => match &decode_config.sampler {
                    Some(sampler) => sampler.decide(&features, f64::from(tau_conf)),
                    None => conf >= tau_conf,
                },
                // CONSTRAINED axis: the 7-param constrained sampler reads
                // marginal_log; else the fixed threshold on the marginal.
                FaCommitBy::ConstrainedMarginal => match &constraint.sampler {
                    Some(sampler) => sampler.decide(&features, marginal_log, f64::from(tau_conf)),
                    None => conf >= tau_conf,
                },
            };
            if take {
                accepted.push((pos, conf));
            }
        }

        // Confidence-ordered commit; the rest remask.
        for pos in confidence_ordered(&accepted, constraint.commit_budget) {
            tokens[pos] = scratch.proposals[pos - block_start] as usize;
            masked[pos] = false;
            committed_total += 1;
        }

        // SC update — mirrors the base loop (Plan 250 T4, Issue 695 H16).
        if let Some(state) = sc_state.as_mut()
            && !gpu.w_sc_is_identity()
        {
            state.update_from_logits(&all_logits, gpu.wte_cpu_ref(), n_embd, mask_token_id);
        }

        confidence_history.push(
            (block_size - masked[block_start..seq_len].iter().filter(|&&m| m).count()) as f32
                / block_size as f32,
        );

        if (block_start..seq_len).all(|pos| !masked[pos]) {
            break;
        }
    }

    // Final x\u{0} commit: still-masked positions take the last proposal —
    // accepted by construction, so the returned block never carries masks.
    for pos in block_start..seq_len {
        if masked[pos] {
            tokens[pos] = scratch.proposals[pos - block_start] as usize;
            masked[pos] = false;
        }
    }

    // Contract: the block walks from start_node to an ACCEPTING node.
    let block_tokens: Vec<u32> = tokens[block_start..seq_len]
        .iter()
        .map(|&t| t as u32)
        .collect();
    let final_node = fa
        .walk_from(constraint.start_node, &block_tokens)
        .filter(|&n| fa.is_accept(n))
        .ok_or(FaDecodeError::WalkBrokeContract)?;

    let all_committed_by_confidence = confidence_history
        .last()
        .is_some_and(|&c| (c - 1.0).abs() < f32::EPSILON);
    let state = if all_committed_by_confidence {
        D2fBlockState::FullyActivated
    } else {
        D2fBlockState::SemiActivated {
            step: steps_used.saturating_sub(1),
            confidence: confidence_history.last().copied().unwrap_or(0.0),
        }
    };

    Ok(Gemma2D2fFaResult {
        decode: Gemma2D2fResult {
            tokens,
            steps_used,
            confidence: final_confidence,
            converged: all_committed_by_confidence,
            state,
            confidence_history,
        },
        final_node,
        nan_fallbacks,
        committed: committed_total,
    })
}

#[cfg(test)]
mod tests;
