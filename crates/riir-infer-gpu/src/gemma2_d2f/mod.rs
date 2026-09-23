//! Gemma 2 D2F Block-Causal Decode — CubeCL GPU path (Plan 108).
//!
//! GPU-accelerated D2F (Diffusion-to-Forecast) denoising decode loop for Gemma 2.
//! Uses block-causal attention: bidirectional within generation block, causal across blocks.
//!
//! Two-phase per layer: Phase A fills K/V for all positions (RMSNorm→QKV→RoPE→Store),
//! Phase B runs block-causal attention + MLP per position.
//!
//! All code behind `#[cfg(feature = "gemma2_d2f")]`.
//! Requires: `gemma2_d2f = ["cubecl_runtime", "dllm", "dep:fastrand"]`.
//!
//! Imports `pub(crate)` types from `gemma2_cubecl`: `CpuKVCache`, `CubeCLWeightFormat`,
//! `CubeCLLayerWeights`, `NormGammas`, `GpuNormGammaHandles`, CPU helpers.

use cubecl::prelude::*;
use cubecl::server::Handle;
use crate::cubecl_runtime::ActiveRuntime;

use crate::attention_cubecl::{AttentionBlockCausalParams, AttentionCubeCL};
use crate::gemma2_cubecl::{
    CpuKVCache, CubeCLF16LayerWeights, CubeCLLayerWeights, CubeCLQ4KLayerWeights,
    CubeCLWeightBuffers, CubeCLWeightFormat, GpuGemmaCubeCL, GpuNormGammaHandles, NormGammas,
    apply_rope, geglu, rmsnorm_gamma, softcap,
};
use crate::gemv_autotune::GemvAutotune;
use crate::gemv_f16_cubecl::{F16Handle, GemvF16CubeCL};
use crate::gemv_q4k_cubecl::{GemvQ4KCubeCL, Q4KHandle};
use riir_infer_core::gemma_layer::GemmaTransformerWeights;
use riir_infer_core::types::Config;

// Self-conditioning support (Plan 250 T1-T4).
use crate::gemma2_d2f_sc::{
    D2fScConfig, D2fScState, init_w_sc_identity_padded, is_identity_padded, project_sc_into,
};

// ── D2F Block State ────────────────────────────────────────────────

/// Final state of a D2F decode block.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum D2fBlockState {
    /// Not all positions unmasked within denoise_steps.
    SemiActivated {
        /// Step at which decoding stopped.
        step: usize,
        /// Fraction of positions unmasked at final step.
        confidence: f32,
    },
    /// All positions successfully unmasked.
    FullyActivated,
}

impl D2fBlockState {
    /// Whether the block is fully activated (all positions unmasked).
    #[inline]
    pub fn is_fully_activated(&self) -> bool {
        matches!(self, D2fBlockState::FullyActivated)
    }
}

/// Number of input features per position for the diffusion sampler.
const N_SAMPLER_FEATURES: usize = 6;

/// Per-position features for D2F sampler decisions. Lightweight confidence stats.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplerFeatures {
    /// Top-1 token probability after softmax.
    pub top1_prob: f32,
    /// Margin: top-1 prob − top-2 prob. Higher = more confident.
    pub margin: f32,
    /// Sum of top-3 token probabilities. Higher = peaked distribution.
    pub top3_mass: f32,
    /// Entropy of softmax distribution. Lower = more confident.
    pub entropy: f32,
    /// Current denoising step normalized: step / max_steps.
    pub step_norm: f32,
    /// Position within block normalized: pos / block_size.
    pub pos_norm: f32,
}

impl SamplerFeatures {
    /// Extract features from logits. `mask_token_id` excluded from probabilities.
    ///
    /// Zero-allocation single-pass implementation: tracks the top-3 unnormalized
    /// exponentials for `top1`/`top2`/`top3_mass`, and accumulates the entropy
    /// sum `Σ e·ln(e)` in the same pass — no `probs` materialization and no
    /// `O(v log v)` sort. Numerically identical to the previous multi-pass
    /// version (verified by `test_sampler_features_*`).
    pub fn from_logits(
        logits: &[f32],
        step: usize,
        total_steps: usize,
        pos_in_block: usize,
        block_size: usize,
        mask_token_id: usize,
    ) -> Self {
        let vocab = logits.len();
        if vocab == 0 {
            return Self::default();
        }
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Single fused pass over the vocab:
        //   sum_exp    = Σ e_t                         (softmax denominator)
        //   sum_e_ln_e = Σ e_t · ln(e_t)               (entropy accumulator;
        //                ln(e_t) = logits[t] - max_logit, so no log() per token)
        //   t1/t2/t3   = top-3 largest e_t             (single-pass top-3)
        // All three quantities are derived from the same `e_t = exp(logits[t] - max_logit)`,
        // so we compute `e_t` exactly once per token.
        let mut sum_exp = 0.0f32;
        let mut sum_e_ln_e = 0.0f32;
        let mut t1 = 0.0f32;
        let mut t2 = 0.0f32;
        let mut t3 = 0.0f32;

        #[allow(clippy::needless_range_loop, reason = "softmax: skip-by-index t == mask_token_id continue")]
        for t in 0..vocab {
            if t == mask_token_id {
                continue;
            }
            let logit_shifted = logits[t] - max_logit;
            let e = logit_shifted.exp();
            sum_exp += e;
            // Entropy term: e · ln(e) = e · logit_shifted.
            // Guard against -inf logits (e underflows to 0.0; 0·(-inf) = NaN).
            if e > 0.0 && logit_shifted.is_finite() {
                sum_e_ln_e += e * logit_shifted;
            }
            // Insertion-sort top-3 (branchless cascade — no heap, O(v) total).
            if e > t1 {
                t3 = t2;
                t2 = t1;
                t1 = e;
            } else if e > t2 {
                t3 = t2;
                t2 = e;
            } else if e > t3 {
                t3 = e;
            }
        }

        if sum_exp <= 0.0 {
            return Self::default();
        }

        let inv_sum = sum_exp.recip();
        let top1_prob = t1 * inv_sum;
        let top2_prob = t2 * inv_sum;
        let top3_mass = (t1 + t2 + t3) * inv_sum;

        // Entropy: -Σ p·ln(p) = ln(S) - (1/S)·Σ e·ln(e).
        // Derived: p = e/S, ln(p) = ln(e) - ln(S), so
        //   -Σ p·ln(p) = -Σ (e/S)(ln(e) - ln(S)) = -(1/S)Σ e·ln(e) + ln(S).
        let entropy = sum_exp.ln() - sum_e_ln_e * inv_sum;

        Self {
            top1_prob,
            margin: top1_prob - top2_prob,
            top3_mass,
            entropy,
            step_norm: if total_steps > 0 {
                step as f32 / total_steps as f32
            } else {
                0.0
            },
            pos_norm: if block_size > 0 {
                pos_in_block as f32 / block_size as f32
            } else {
                0.0
            },
        }
    }

    /// Convert to flat feature array for model input.
    fn to_array(self) -> [f32; N_SAMPLER_FEATURES] {
        [
            self.top1_prob,
            self.margin,
            self.top3_mass,
            self.entropy,
            self.step_norm,
            self.pos_norm,
        ]
    }
}

/// Logistic regression sampler: sigmoid(w·x + b). 7 params, O(1) inference.
#[derive(Clone, Copy, Debug)]
pub struct DiffusionSampler {
    /// Feature weights.
    weights: [f32; N_SAMPLER_FEATURES],
    /// Bias term.
    bias: f32,
}

impl DiffusionSampler {
    /// Create an untrained (zero-weight) sampler that predicts ~0.5.
    pub fn untrained() -> Self {
        Self {
            weights: [0.0; N_SAMPLER_FEATURES],
            bias: 0.0,
        }
    }

    /// Create from trained weights + bias.
    pub fn from_weights(weights: [f32; N_SAMPLER_FEATURES], bias: f32) -> Self {
        Self { weights, bias }
    }

    /// Compute accept probability P(correct | features) ∈ [0, 1].
    pub fn predict(&self, features: &SamplerFeatures) -> f64 {
        let x = features.to_array();
        let z: f64 = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, f)| (*w as f64) * (*f as f64))
            .sum::<f64>()
            + self.bias as f64;
        1.0 / (1.0 + (-z).exp()) // sigmoid
    }

    /// Decide whether to accept a denoised token.
    pub fn decide(&self, features: &SamplerFeatures, threshold: f64) -> bool {
        self.predict(features) >= threshold
    }
}

// ── D2F Config ─────────────────────────────────────────────────────

/// Configuration for D2F denoising decode.
#[derive(Clone, Copy, Debug)]
pub struct Gemma2D2fConfig {
    /// Number of tokens per D2F generation block.
    pub block_size: usize,
    /// Maximum denoising iterations.
    pub denoise_steps: usize,
    /// Confidence threshold for unmasking a position.
    pub confidence_threshold: f32,
    /// Sampling temperature (0.0 = greedy, 1.0 = standard).
    pub temperature: f32,
    /// Optional diffusion sampler for per-position accept/reject decisions.
    /// When `Some`, replaces fixed `confidence_threshold` with learned predictions.
    /// When `None`, falls back to `confidence >= threshold`.
    pub sampler: Option<DiffusionSampler>,
    /// Self-conditioning configuration (Plan 250 T1-T4).
    /// When `sc_config.enabled` is false (default), SC is fully bypassed —
    /// zero overhead, decode behaves identically to non-SC.
    /// When enabled, the decode loop tracks the previous step's x̂_0 estimate
    /// and feeds it through W_sc projection into the embedding layer.
    pub sc_config: D2fScConfig,
}

impl Default for Gemma2D2fConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            denoise_steps: 8,
            confidence_threshold: 0.7,
            temperature: 1.0,
            sampler: None,
            sc_config: D2fScConfig::default(),
        }
    }
}

impl Gemma2D2fConfig {
    /// Quality preset: more steps, higher threshold.
    pub fn quality() -> Self {
        Self {
            denoise_steps: 12,
            confidence_threshold: 0.8,
            ..Self::default()
        }
    }

    /// Speed preset: fewer steps, lower threshold.
    pub fn speed() -> Self {
        Self {
            denoise_steps: 4,
            confidence_threshold: 0.5,
            ..Self::default()
        }
    }

    /// Create config with custom block size.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
}

// ── D2F Result ─────────────────────────────────────────────────────

/// D2F decode result for Gemma2.
#[derive(Clone, Debug)]
pub struct Gemma2D2fResult {
    /// Final decoded tokens (prompt + block).
    pub tokens: Vec<usize>,
    /// Number of denoising steps used (may be < max if converged early).
    pub steps_used: usize,
    /// Per-position confidence scores at final step.
    pub confidence: Vec<f32>,
    /// Whether all positions were unmasked within denoise_steps.
    pub converged: bool,
    /// Final block state.
    pub state: D2fBlockState,
    /// Confidence history across steps (for diagnostics).
    /// Each entry is the fraction of unmasked positions at that step.
    pub confidence_history: Vec<f32>,
}

// ── GPU D2F Struct ─────────────────────────────────────────────────

/// GPU-accelerated Gemma 2 D2F block decoder.
///
/// Reuses CubeCL client + weight handles from `GpuGemmaCubeCL`
/// but dispatches block-causal attention. KV cache reset per `block_causal_forward()` call.
///
/// Create via `from_gpu_gemma(gpu, d2f_config)` (consumes `GpuGemmaCubeCL`)
/// or `new(client, weights, config, d2f_config)` (standalone, F32 handles).
pub struct GpuGemmaCubeCLD2F {
    /// Shared CubeCL compute client.
    client: ComputeClient<ActiveRuntime>,
    /// Model configuration.
    config: Config,
    /// CubeCL weight handles (shared with GpuGemmaCubeCL via move).
    weights: CubeCLWeightFormat,
    /// CPU RMSNorm gamma vectors.
    norm_gammas: NormGammas,
    /// GPU RMSNorm gamma handles.
    #[allow(dead_code)] // Reserved for GPU-side RMSNorm (T3 CubeCL full GPU path).
    gpu_norm_gammas: GpuNormGammaHandles,
    /// CPU KV cache for block-causal forward (sized for full seq_len per call).
    kv_cache: CpuKVCache,
    /// CPU embedding weights.
    wte_cpu: Vec<f32>,
    /// Block-causal attention parameters.
    attn_params: AttentionBlockCausalParams,
    /// GEMV autotune cache.
    gemv_autotune: GemvAutotune,
    /// D2F configuration.
    d2f_config: Gemma2D2fConfig,
    /// Self-conditioning projection matrix W_sc (Plan 250 T2).
    /// Shape: `(n_embd, 2*n_embd)` flat. Initialized as `[I | 0]` (identity
    /// for x_t, zeros for SC input) → zero behavioral change until trained.
    /// Allocated lazily when SC is first enabled.
    w_sc: Option<Vec<f32>>,
    /// Cached verdict: `true` while `w_sc` still matches the identity-padded
    /// init. When `true`, the SC projection is provably the identity on `x_t`
    /// and the O(n²) matmul is skipped entirely (G4 at cost level). Set to
    /// `false` by `set_w_sc` after LoRA training writes a non-identity W_sc.
    w_sc_is_identity: bool,
    /// Reusable scratch buffers for `forward_layer_block_causal` — pre-allocated
    /// to model dims and cleared/reused across every (layer × position)
    /// iteration. Eliminates THESE buffers' per-position heap allocations only
    /// (Issue 695 H12: the loop still allocates per position elsewhere — the
    /// `queries` rows, KV gather Vecs, `read_handle` Vecs, and `hidden_states`
    /// slot churn; see the issue's H8/H9 for those).
    scratch_normed_a: Vec<f32>,     // [n_embd] Phase-A RMSNorm input
    scratch_residual: Vec<f32>,     // [n_embd] Phase-B residual
    scratch_residual2: Vec<f32>,    // [n_embd] Phase-B residual2
    scratch_mlp_hidden: Vec<f32>,   // [mlp_hidden] Phase-B GeGLU output
}

impl GpuGemmaCubeCLD2F {
    /// Construct from existing `GpuGemmaCubeCL` (takes ownership, no Clone needed).
    ///
    /// Moves all GPU handles, weight buffers, and CubeCL client from the existing
    /// AR decoder. The `GpuGemmaCubeCL` is consumed and cannot be used afterward.
    /// Creates a fresh KV cache for block-causal processing.
    ///
    /// # Arguments
    ///
    /// * `gpu` — Existing CubeCL Gemma2 AR decoder (consumed).
    /// * `d2f_config` — D2F hyperparameters (block size, denoise steps, etc.).
    pub fn from_gpu_gemma(gpu: GpuGemmaCubeCL, d2f_config: Gemma2D2fConfig) -> Self {
        // Destructure to move fields; remaining (kv_cache, gpu_kv_cache, attn_params) are dropped.
        let GpuGemmaCubeCL {
            client,
            config,
            weights,
            norm_gammas,
            gpu_norm_gammas,
            wte_cpu,
            gemv_autotune,
            ..
        } = gpu;

        let kv_stride = config.n_kv_head * config.head_dim;

        // Issue 695 H6: the block-causal attention kernel hardcodes
        // softcap=50.0 and scale=0.0625 (1/√256) — the `softcap`/`scale` fields
        // built below are NOT read by the kernel. Any config with
        // `attn_logit_softcapping != 50.0` or `head_dim != 256` would silently
        // diverge; fail loudly at construction instead. Remove this assert
        // only when the kernel starts consuming the params fields.
        assert_block_causal_kernel_pinned(&config);

        let attn_params = AttentionBlockCausalParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0,
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            pos: 0,
            prompt_len: 0,
            block_size: d2f_config.block_size,
        };

        Self {
            client,
            config: config.clone(),
            weights,
            norm_gammas,
            gpu_norm_gammas,
            kv_cache: CpuKVCache::new(config.n_layer, kv_stride),
            wte_cpu,
            attn_params,
            gemv_autotune,
            d2f_config,
            w_sc: None,
            w_sc_is_identity: false,
            scratch_normed_a: vec![0.0; config.n_embd],
            scratch_residual: vec![0.0; config.n_embd],
            scratch_residual2: vec![0.0; config.n_embd],
            scratch_mlp_hidden: vec![0.0; config.mlp_hidden],
        }
    }

    /// Standalone constructor — uploads fresh F32 weight handles.
    ///
    /// Equivalent to creating a `GpuGemmaCubeCL::new()` then converting,
    /// but skips the intermediate AR decoder. Uses F32 weight format.
    pub fn new(
        client: ComputeClient<ActiveRuntime>,
        weights: &GemmaTransformerWeights,
        config: &Config,
        d2f_config: Gemma2D2fConfig,
    ) -> Self {
        let cubecl_weights = CubeCLWeightBuffers::from_weights(&client, weights);
        let norm_gammas = NormGammas::from_weights(weights);
        let gpu_norm_gammas = GpuNormGammaHandles::upload(&client, weights);
        let kv_stride = config.n_kv_head * config.head_dim;

        // Issue 695 H6 — see from_gpu_gemma for the rationale.
        assert_block_causal_kernel_pinned(config);

        let attn_params = AttentionBlockCausalParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: config.head_dim,
            n_positions: 0,
            softcap: config.attn_logit_softcapping,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            pos: 0,
            prompt_len: 0,
            block_size: d2f_config.block_size,
        };

        Self {
            client,
            config: config.clone(),
            weights: CubeCLWeightFormat::F32(cubecl_weights),
            norm_gammas,
            gpu_norm_gammas,
            kv_cache: CpuKVCache::new(config.n_layer, kv_stride),
            wte_cpu: weights.wte.clone(),
            attn_params,
            gemv_autotune: GemvAutotune::new(),
            d2f_config,
            w_sc: None,
            w_sc_is_identity: false,
            scratch_normed_a: vec![0.0; config.n_embd],
            scratch_residual: vec![0.0; config.n_embd],
            scratch_residual2: vec![0.0; config.n_embd],
            scratch_mlp_hidden: vec![0.0; config.mlp_hidden],
        }
    }

    // ── Public API ─────────────────────────────────────────────────

    /// Run block-causal forward pass for all tokens (no self-conditioning).
    ///
    /// Convenience wrapper around [`Self::block_causal_forward_with_sc()`] that passes
    /// `None` for the SC input. Identical behavior to pre-Plan-250 code.
    ///
    /// # Arguments
    ///
    /// * `tokens` — Token IDs to process (prompt + generation).
    /// * `prompt_len` — Number of prompt tokens at the start of `tokens`.
    ///
    /// # Returns
    ///
    /// Per-position logits: `Vec<Vec<f32>>` of shape `[seq_len][vocab_size]`.
    pub fn block_causal_forward(&mut self, tokens: &[usize], prompt_len: usize) -> Vec<Vec<f32>> {
        self.block_causal_forward_with_sc(tokens, prompt_len, None)
    }

    /// Run block-causal forward pass with optional self-conditioning (Plan 250 T2).
    ///
    /// Processes the full token sequence through all Gemma2 layers using
    /// block-causal attention. When `sc_state` provides a previous x̂_0 estimate,
    /// the embedding for each position is projected through W_sc:
    ///
    /// `h = W_sc @ [embed(token) ‖ x̂_0_prev(pos)]`
    ///
    /// When `sc_state` is `None` or has no previous estimate (first step),
    /// behavior is identical to the non-SC forward pass.
    ///
    /// # KV Cache
    ///
    /// The KV cache is reset at the start of each call. Each call is
    /// independent (suitable for D2F denoising where each step re-runs
    /// the full sequence with different masks).
    ///
    /// # Arguments
    ///
    /// * `tokens` — Token IDs to process (prompt + generation).
    /// * `prompt_len` — Number of prompt tokens at the start of `tokens`.
    /// * `sc_state` — Optional SC state carrying the previous step's x̂_0 estimate.
    ///
    /// # Returns
    ///
    /// Per-position logits: `Vec<Vec<f32>>` of shape `[seq_len][vocab_size]`.
    pub fn block_causal_forward_with_sc(
        &mut self,
        tokens: &[usize],
        prompt_len: usize,
        sc_state: Option<&D2fScState>,
    ) -> Vec<Vec<f32>> {
        let seq_len = tokens.len();
        let n = self.config.n_embd;
        let vocab = self.config.vocab_size;
        let embed_scale = (n as f32).sqrt();
        let eps = self.config.rms_norm_eps as f32;

        // Reset KV cache for this forward pass.
        self.kv_cache = CpuKVCache::new(
            self.config.n_layer,
            self.config.n_kv_head * self.config.head_dim,
        );

        // Determine if SC projection should be applied this step.
        // Requires: SC enabled in config and SC input available (i.e. not the
        // first denoising step). When both hold, lazily allocate W_sc on the
        // first SC-applied call (identity-padded init → zero behavioral change
        // until trained), then apply the projection.
        let sc_input: Option<&[Vec<f32>]> = match sc_state {
            Some(state) if state.config.enabled => state.sc_input(),
            _ => None,
        };
        let apply_sc = if sc_input.is_some() {
            // Lazy-allocate W_sc the first time we actually have SC input.
            // Identity-padded init: W_sc @ [x_t ‖ x̂_0] = x_t when x̂_0 is zeros.
            if self.w_sc.is_none() {
                self.w_sc = Some(init_w_sc_identity_padded(n));
                self.w_sc_is_identity = true;
            }
            true
        } else {
            false
        };
        // When W_sc is still at identity init, the projection is provably the
        // identity on x_t (I·x_t + 0·x̂_0 = x_t). Skip the O(n²) matmul entirely
        // — this is the G4 property at cost level (untrained SC = zero overhead).
        // After LoRA training writes a non-identity W_sc via `set_w_sc`, the full
        // matmul runs and the real SC projection cost is incurred.
        let apply_projection = apply_sc && !self.w_sc_is_identity;

        // 1. Embedding lookup for all positions (CPU, scaled by sqrt(n_embd)).
        //    When SC is active and W_sc is trained (non-identity), project:
        //    h = W_sc @ [embed(token) ‖ x̂_0_prev(pos)]. For identity W_sc the
        //    projection is a no-op (h already equals the identity result), so
        //    we skip the matmul entirely (G4 cost property).
        //    Pre-allocate a reusable SC projection scratch buffer outside the loop
        //    (hot-loop allocation rule).
        let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut sc_scratch: Option<Vec<f32>> = if apply_projection {
            Some(vec![0.0f32; n])
        } else {
            None
        };
        for (pos, &token) in tokens.iter().enumerate() {
            let mut hidden = vec![0.0f32; n];
            let tok_off = token * n;
            for (i, h) in hidden.iter_mut().enumerate() {
                *h = self.wte_cpu[tok_off + i] * embed_scale;
            }

            // SC projection: overwrite hidden with W_sc @ [hidden ‖ sc_input[pos]].
            // Skipped when W_sc is identity (untrained) — the identity projection
            // would return `hidden` unchanged, so the matmul is wasted work.
            if apply_projection {
                let w_sc = self.w_sc.as_ref().expect("w_sc allocated above");
                let sc = sc_input.expect("sc_input checked above");
                if pos < sc.len() && sc[pos].len() == n {
                    let projected = sc_scratch.as_mut().expect("sc_scratch allocated above");
                    project_sc_into(&hidden, &sc[pos], w_sc, n, projected);
                    std::mem::swap(&mut hidden, projected);
                } else {
                    // Issue 695 H24: this position silently ran UNPROJECTED
                    // before. `update_from_logits` builds sc_input as
                    // [seq_len][n_embd] against the same seq_len the caller
                    // passes here, so a mismatch is a caller bug — fail loud
                    // in debug, keep the graceful fallback in release.
                    debug_assert!(
                        false,
                        "sc_input shape mismatch at pos {pos}: sc.len()={}, \
                         sc[{pos}].len()={} (expected {} rows of n_embd={n})",
                        sc.len(),
                        sc.get(pos).map_or(0, |r| r.len()),
                        seq_len
                    );
                }
            }

            hidden_states.push(hidden);
        }

        // 2. Process all layers with block-causal attention.
        for layer_idx in 0..self.config.n_layer {
            self.forward_layer_block_causal(&mut hidden_states, layer_idx, prompt_len, seq_len);
        }

        // 3. Final RMSNorm + tied lm_head + logit softcapping for each position.
        let mut all_logits = Vec::with_capacity(seq_len);
        for hs in &hidden_states[..seq_len] {
            let mut hidden = hs.clone();
            rmsnorm_gamma(&mut hidden, &self.norm_gammas.final_norm, n, eps);

            // LM head GEMV (tied wte): logits = wte @ hidden
            let logits = match &self.weights {
                CubeCLWeightFormat::F32(w) => self.dispatch_gemv(&w.wte, &hidden, vocab, n),
                CubeCLWeightFormat::F16(w) => self.dispatch_gemv_f16(&w.wte, &hidden),
                CubeCLWeightFormat::Q4K(w) => self.dispatch_gemv_q4k(&w.wte, &hidden),
            };

            // Final logit softcapping: cap * tanh(logits / cap)
            let mut logits = logits;
            if self.config.final_logit_softcapping > 0.0 {
                softcap(&mut logits, self.config.final_logit_softcapping);
            }
            all_logits.push(logits);
        }

        all_logits.shrink_to_fit();
        all_logits
    }

    /// Access the model config reference.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the D2F configuration.
    #[inline]
    pub fn d2f_config(&self) -> &Gemma2D2fConfig {
        &self.d2f_config
    }

    /// Access the CPU embedding table (for SC x̂_0 computation, Plan 250).
    #[inline]
    pub fn wte_cpu_ref(&self) -> &[f32] {
        &self.wte_cpu
    }

    /// Access the SC projection matrix W_sc (if allocated).
    /// Returns `None` when SC has not been used yet (lazy allocation).
    #[inline]
    pub fn w_sc_ref(&self) -> Option<&[f32]> {
        self.w_sc.as_deref()
    }

    /// Whether W_sc is still at its identity-padded init (untrained).
    /// When `true`, the SC projection is skipped (G4 cost property).
    #[inline]
    pub fn w_sc_is_identity(&self) -> bool {
        self.w_sc_is_identity
    }

    /// Overwrite the SC projection matrix W_sc with a trained (non-identity)
    /// buffer. Clears the identity flag so subsequent forwards run the full
    /// O(n²) projection. Expected shape: `n_embd * 2 * n_embd` (row-major).
    ///
    /// # Panics
    ///
    /// Panics if `w_sc.len() != n_embd * 2 * n_embd`.
    pub fn set_w_sc(&mut self, w_sc: Vec<f32>) {
        let n = self.config.n_embd;
        assert_eq!(
            w_sc.len(),
            n * 2 * n,
            "set_w_sc: expected length {} (n_embd * 2 * n_embd), got {}",
            n * 2 * n,
            w_sc.len()
        );
        // Re-check the identity verdict so a caller passing in an
        // identity-shaped buffer still gets the fast path.
        self.w_sc_is_identity = is_identity_padded(&w_sc, n);
        self.w_sc = Some(w_sc);
    }

    // ── Layer forward ──────────────────────────────────────────────

    /// Single Gemma2 transformer layer with block-causal attention.
    ///
    /// Two-phase approach ensures all K/V are computed before any attention:
    ///
    /// **Phase A — K/V Fill:**
    /// For all positions: RMSNorm → QKV GEMV → RoPE → Store K/V in cache
    ///
    /// **Phase B — Attention + MLP:**
    /// For each position: Block-causal attention + Wo → RMSNorm + residual →
    /// MLP (gate/up → GeGLU → down) → RMSNorm + residual
    fn forward_layer_block_causal(
        &mut self,
        hidden_states: &mut [Vec<f32>],
        layer_idx: usize,
        prompt_len: usize,
        seq_len: usize,
    ) {
        let n = self.config.n_embd;
        let q_dim = self.config.n_head * self.config.head_dim;
        let kv_dim = self.config.n_kv_head * self.config.head_dim;
        let mlp = self.config.mlp_hidden;
        let eps = self.config.rms_norm_eps as f32;
        let norms = &self.norm_gammas.layers[layer_idx];

        // ── Phase A: Compute Q/K/V for all positions, store K/V ────
        let mut queries: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for (pos, hidden) in hidden_states[..seq_len].iter().enumerate() {
            // CPU: RMSNorm — clone into reusable scratch instead of allocating
            // a fresh Vec per position per layer.
            let normed = {
                let scratch = std::mem::take(&mut self.scratch_normed_a);
                let mut buf = if scratch.len() == n {
                    scratch
                } else {
                    vec![0.0; n]
                };
                buf[..n].copy_from_slice(&hidden[..n]);
                rmsnorm_gamma(&mut buf, &norms.input_norm, n, eps);
                buf
            };

            // GPU: QKV GEMVs (batched, 1 sync for all 3)
            let (q, k, v) = match &self.weights {
                CubeCLWeightFormat::F32(w) => {
                    self.dispatch_qkv(&w.layers[layer_idx], &normed, q_dim, kv_dim, n)
                }
                CubeCLWeightFormat::F16(w) => {
                    self.dispatch_qkv_f16(&w.layers[layer_idx], &normed, q_dim, kv_dim, n)
                }
                CubeCLWeightFormat::Q4K(w) => {
                    self.dispatch_qkv_q4k(&w.layers[layer_idx], &normed, q_dim, kv_dim, n)
                }
            };

            // Recycle the scratch for the next position.
            self.scratch_normed_a = normed;

            // CPU: RoPE on both Q and K
            let mut q = q;
            let mut k = k;
            apply_rope(
                &mut q,
                pos,
                self.config.head_dim,
                self.config.n_head,
                self.config.rope_theta,
            );
            apply_rope(
                &mut k,
                pos,
                self.config.head_dim,
                self.config.n_kv_head,
                self.config.rope_theta,
            );

            // CPU: Store K, V in KV cache
            self.kv_cache.store(layer_idx, pos, &k, &v);
            queries.push(q);
        }

        // ── Phase B: Block-causal attention + Wo + MLP ─────────────
        for pos in 0..seq_len {
            // Copy residual into reusable scratch instead of cloning hidden_states[pos].
            let residual = {
                let s = std::mem::take(&mut self.scratch_residual);
                let mut buf = if s.len() == n { s } else { vec![0.0; n] };
                buf[..n].copy_from_slice(&hidden_states[pos][..n]);
                buf
            };
            let q = &queries[pos];

            // GPU: Block-causal attention + Wo (batched, 1 sync)
            let wo_out = match &self.weights {
                CubeCLWeightFormat::F32(w) => self.dispatch_block_causal_attention_wo(
                    &w.layers[layer_idx],
                    q,
                    layer_idx,
                    pos,
                    n,
                    prompt_len,
                    seq_len,
                ),
                CubeCLWeightFormat::F16(w) => self.dispatch_block_causal_attention_wo_f16(
                    &w.layers[layer_idx],
                    q,
                    layer_idx,
                    pos,
                    n,
                    prompt_len,
                    seq_len,
                ),
                CubeCLWeightFormat::Q4K(w) => self.dispatch_block_causal_attention_wo_q4k(
                    &w.layers[layer_idx],
                    q,
                    layer_idx,
                    pos,
                    n,
                    prompt_len,
                    seq_len,
                ),
            };

            // CPU: RMSNorm + add residual
            let mut hidden = wo_out;
            rmsnorm_gamma(&mut hidden, &norms.post_attn_norm, n, eps);
            for (h, r) in hidden.iter_mut().zip(residual.iter()) {
                *h += r;
            }

            // Recycle residual scratch; move current hidden into residual2 scratch.
            self.scratch_residual = residual;
            let mut residual2 = std::mem::take(&mut self.scratch_residual2);
            if residual2.len() != n {
                residual2 = vec![0.0; n];
            }
            residual2[..n].copy_from_slice(&hidden[..n]);

            // CPU: RMSNorm
            rmsnorm_gamma(&mut hidden, &norms.pre_mlp_norm, n, eps);

            // GPU: Gate + Up GEMVs (batched, 1 sync)
            let (gate, up) = match &self.weights {
                CubeCLWeightFormat::F32(w) => {
                    self.dispatch_gate_up(&w.layers[layer_idx], &hidden, mlp, n)
                }
                CubeCLWeightFormat::F16(w) => {
                    self.dispatch_gate_up_f16(&w.layers[layer_idx], &hidden, mlp, n)
                }
                CubeCLWeightFormat::Q4K(w) => {
                    self.dispatch_gate_up_q4k(&w.layers[layer_idx], &hidden, mlp, n)
                }
            };

            // CPU: GeGLU into reusable scratch (avoids vec![0.0f32; mlp] per position).
            let mut mlp_hidden = std::mem::take(&mut self.scratch_mlp_hidden);
            if mlp_hidden.len() != mlp {
                mlp_hidden = vec![0.0; mlp];
            } else {
                // geglu writes all mlp elements, but clear defensively in case
                // of any early-exit path change upstream.
                mlp_hidden[..mlp].fill(0.0);
            }
            geglu(&gate, &up, &mut mlp_hidden);

            // GPU: Down GEMV (1 sync)
            let down_out = match &self.weights {
                CubeCLWeightFormat::F32(w) => {
                    self.dispatch_gemv(&w.layers[layer_idx].down_proj, &mlp_hidden, n, mlp)
                }
                CubeCLWeightFormat::F16(w) => {
                    self.dispatch_gemv_f16(&w.layers[layer_idx].down_proj, &mlp_hidden)
                }
                CubeCLWeightFormat::Q4K(w) => {
                    self.dispatch_gemv_q4k(&w.layers[layer_idx].down_proj, &mlp_hidden)
                }
            };

            // Recycle mlp_hidden scratch.
            self.scratch_mlp_hidden = mlp_hidden;

            // CPU: RMSNorm + add residual2
            let mut hidden = down_out;
            rmsnorm_gamma(&mut hidden, &norms.post_mlp_norm, n, eps);
            for (h, r) in hidden.iter_mut().zip(residual2.iter()) {
                *h += r;
            }

            // Recycle residual2 scratch.
            self.scratch_residual2 = residual2;

            hidden_states[pos] = hidden;
        }
    }

    // ── CubeCL dispatch helpers ─────────────────────────────────────

    /// Launch CubeCL GEMV: `output[M] = weight[M,N] @ input[N]`.
    fn dispatch_gemv(&self, weight: &Handle, input: &[f32], m: usize, n: usize) -> Vec<f32> {
        let input_handle = self.client.create_from_slice(f32::as_bytes(input));
        let output_handle = self.client.empty(m * core::mem::size_of::<f32>());

        // SAFETY: weight has M×N elements, input has N elements, output has M elements.
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                weight.clone(),
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        self.read_handle(&output_handle)
    }

    /// Launch CubeCL f16 weight GEMV: `output[M] = weight_f16[M,N] @ input[N]`.
    fn dispatch_gemv_f16(&self, weight: &F16Handle, input: &[f32]) -> Vec<f32> {
        let input_handle = self.client.create_from_slice(f32::as_bytes(input));
        let output_handle = self.client.empty(weight.m * core::mem::size_of::<f32>());

        // SAFETY: F16Handle has correct m, n and weight buffer is m*n f16 elements.
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight.weight.clone(),
                input_handle,
                output_handle.clone(),
                weight.m,
                weight.n,
            );
        }

        self.read_handle(&output_handle)
    }

    /// Launch CubeCL Q4_K dequant+GEMV: `output[M] = dequant_q4k(weight) @ input[N]`.
    fn dispatch_gemv_q4k(&self, weight: &Q4KHandle, input: &[f32]) -> Vec<f32> {
        let input_handle = self.client.create_from_slice(f32::as_bytes(input));
        let output_handle = self.client.empty(weight.m * core::mem::size_of::<f32>());

        // SAFETY: Q4KHandle has correct m, n and buffer sizes.
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                weight,
                input_handle,
                output_handle.clone(),
            );
        }

        self.read_handle(&output_handle)
    }

    /// Launch batched Q/K/V GEMVs (F32, 3 launches, 1 sync).
    fn dispatch_qkv(
        &self,
        layer_weights: &CubeCLLayerWeights,
        hidden: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let q_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let k_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());
        let v_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());

        // SAFETY: All buffer sizes match the (m, n) dimensions.
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wq.clone(),
                input_handle.clone(),
                q_handle.clone(),
                q_dim,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wk.clone(),
                input_handle.clone(),
                k_handle.clone(),
                kv_dim,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wv.clone(),
                input_handle,
                v_handle.clone(),
                kv_dim,
                n_embd,
            );
        }

        // Single sync: read all outputs.
        let q = self.read_handle(&q_handle);
        let k = self.read_handle(&k_handle);
        let v = self.read_handle(&v_handle);

        (q, k, v)
    }

    /// Launch batched Q/K/V GEMVs (F16, 3 launches, 1 sync).
    fn dispatch_qkv_f16(
        &self,
        layer_weights: &CubeCLF16LayerWeights,
        hidden: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let q_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let k_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());
        let v_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());

        // SAFETY: F16 weight handles have correct m, n dimensions.
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wq.weight.clone(),
                input_handle.clone(),
                q_handle.clone(),
                q_dim,
                n_embd,
            );
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wk.weight.clone(),
                input_handle.clone(),
                k_handle.clone(),
                kv_dim,
                n_embd,
            );
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wv.weight.clone(),
                input_handle,
                v_handle.clone(),
                kv_dim,
                n_embd,
            );
        }

        let q = self.read_handle(&q_handle);
        let k = self.read_handle(&k_handle);
        let v = self.read_handle(&v_handle);

        (q, k, v)
    }

    /// Launch batched Q/K/V GEMVs (Q4_K, 3 launches, 1 sync).
    fn dispatch_qkv_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        hidden: &[f32],
        q_dim: usize,
        kv_dim: usize,
        _n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let q_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let k_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());
        let v_handle = self.client.empty(kv_dim * core::mem::size_of::<f32>());

        // SAFETY: Q4K weight handles have correct m, n dimensions.
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.attn_wq,
                input_handle.clone(),
                q_handle.clone(),
            );
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.attn_wk,
                input_handle.clone(),
                k_handle.clone(),
            );
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.attn_wv,
                input_handle,
                v_handle.clone(),
            );
        }

        let q = self.read_handle(&q_handle);
        let k = self.read_handle(&k_handle);
        let v = self.read_handle(&v_handle);

        (q, k, v)
    }

    /// Launch block-causal attention + Wo GEMV (F32, 2 launches, 1 sync).
    ///
    /// Key difference from causal: uses `launch_block_causal()` with
    /// block-causal masking parameters (pos, prompt_len, block_size).
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    fn dispatch_block_causal_attention_wo(
        &self,
        layer_weights: &CubeCLLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
        prompt_len: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_dim = self.config.n_head * self.config.head_dim;

        // Build combined KV buffer from CPU cache (all positions up to seq_len).
        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, seq_len);

        // Create handles.
        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        // Block-causal attention params.
        let mut params = self.attn_params;
        params.n_positions = seq_len;
        params.pos = pos;
        params.prompt_len = prompt_len;
        params.block_size = self.d2f_config.block_size;

        // Launch block-causal attention.
        AttentionCubeCL::launch_block_causal::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // Fused: attention → Wo without intermediate sync.
        // SAFETY: attn_wo is [n_embd, q_dim], attn_out is [q_dim], wo_out is [n_embd].
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wo.clone(),
                attn_out_handle,
                wo_out_handle.clone(),
                n_embd,
                q_dim,
            );
        }

        self.read_handle(&wo_out_handle)
    }

    /// Launch block-causal attention + Wo GEMV (F16).
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    fn dispatch_block_causal_attention_wo_f16(
        &self,
        layer_weights: &CubeCLF16LayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
        prompt_len: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_dim = self.config.n_head * self.config.head_dim;

        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, seq_len);

        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        let mut params = self.attn_params;
        params.n_positions = seq_len;
        params.pos = pos;
        params.prompt_len = prompt_len;
        params.block_size = self.d2f_config.block_size;

        AttentionCubeCL::launch_block_causal::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // SAFETY: attn_wo F16 handle has correct m=n_embd, n=q_dim.
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.attn_wo.weight.clone(),
                attn_out_handle,
                wo_out_handle.clone(),
                n_embd,
                q_dim,
            );
        }

        self.read_handle(&wo_out_handle)
    }

    /// Launch block-causal attention + Wo GEMV (Q4_K).
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    fn dispatch_block_causal_attention_wo_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        query: &[f32],
        layer_idx: usize,
        pos: usize,
        n_embd: usize,
        prompt_len: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_dim = self.config.n_head * self.config.head_dim;

        let combined_kv = self.kv_cache.get_combined_kv(layer_idx, seq_len);

        let query_handle = self.client.create_from_slice(f32::as_bytes(query));
        let kv_handle = self.client.create_from_slice(f32::as_bytes(&combined_kv));
        let attn_out_handle = self.client.empty(q_dim * core::mem::size_of::<f32>());
        let wo_out_handle = self.client.empty(n_embd * core::mem::size_of::<f32>());

        let mut params = self.attn_params;
        params.n_positions = seq_len;
        params.pos = pos;
        params.prompt_len = prompt_len;
        params.block_size = self.d2f_config.block_size;

        AttentionCubeCL::launch_block_causal::<ActiveRuntime>(
            &self.client,
            query_handle,
            kv_handle,
            attn_out_handle.clone(),
            &params,
        );

        // SAFETY: attn_wo Q4K handle has correct dimensions.
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.attn_wo,
                attn_out_handle,
                wo_out_handle.clone(),
            );
        }

        self.read_handle(&wo_out_handle)
    }

    /// Launch batched gate + up GEMVs (F32, 2 launches, 1 sync).
    fn dispatch_gate_up(
        &self,
        layer_weights: &CubeCLLayerWeights,
        hidden: &[f32],
        mlp: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let gate_handle = self.client.empty(mlp * core::mem::size_of::<f32>());
        let up_handle = self.client.empty(mlp * core::mem::size_of::<f32>());

        // SAFETY: gate/up weights are [mlp, n_embd], hidden is [n_embd].
        unsafe {
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.gate_proj.clone(),
                input_handle.clone(),
                gate_handle.clone(),
                mlp,
                n_embd,
            );
            self.gemv_autotune.launch::<ActiveRuntime>(
                &self.client,
                layer_weights.up_proj.clone(),
                input_handle,
                up_handle.clone(),
                mlp,
                n_embd,
            );
        }

        let gate = self.read_handle(&gate_handle);
        let up = self.read_handle(&up_handle);

        (gate, up)
    }

    /// Launch batched gate + up GEMVs (F16, 2 launches, 1 sync).
    fn dispatch_gate_up_f16(
        &self,
        layer_weights: &CubeCLF16LayerWeights,
        hidden: &[f32],
        mlp: usize,
        n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let gate_handle = self.client.empty(mlp * core::mem::size_of::<f32>());
        let up_handle = self.client.empty(mlp * core::mem::size_of::<f32>());

        // SAFETY: F16 gate/up handles have correct m=mlp, n=n_embd dimensions.
        unsafe {
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.gate_proj.weight.clone(),
                input_handle.clone(),
                gate_handle.clone(),
                mlp,
                n_embd,
            );
            GemvF16CubeCL::launch::<ActiveRuntime>(
                &self.client,
                layer_weights.up_proj.weight.clone(),
                input_handle,
                up_handle.clone(),
                mlp,
                n_embd,
            );
        }

        let gate = self.read_handle(&gate_handle);
        let up = self.read_handle(&up_handle);

        (gate, up)
    }

    /// Launch batched gate + up GEMVs (Q4_K, 2 launches, 1 sync).
    fn dispatch_gate_up_q4k(
        &self,
        layer_weights: &CubeCLQ4KLayerWeights,
        hidden: &[f32],
        mlp: usize,
        _n_embd: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let input_handle = self.client.create_from_slice(f32::as_bytes(hidden));
        let gate_handle = self.client.empty(mlp * core::mem::size_of::<f32>());
        let up_handle = self.client.empty(mlp * core::mem::size_of::<f32>());

        // SAFETY: Q4K gate/up handles have correct dimensions.
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.gate_proj,
                input_handle.clone(),
                gate_handle.clone(),
            );
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_weights.up_proj,
                input_handle,
                up_handle.clone(),
            );
        }

        let gate = self.read_handle(&gate_handle);
        let up = self.read_handle(&up_handle);

        (gate, up)
    }

    // ── Sync helpers ────────────────────────────────────────────────

    /// Read a CubeCL Handle to CPU as `Vec<f32>`.
    ///
    /// Blocks until the GPU kernel producing this handle's data completes.
    fn read_handle(&self, handle: &Handle) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(handle.clone())
            .unwrap_or_else(|e| panic!("CubeCL buffer read failed: {e}"));
        f32::from_bytes(&bytes).to_vec()
    }
}

// ── Softmax + Sampling Helpers ─────────────────────────────────────

/// In-place softmax: converts logits to probabilities that sum to 1.
///
/// Numerically stable: subtracts max before exp to avoid overflow.
fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for l in logits.iter_mut() {
        *l = (*l - max_val).exp();
    }
    let sum: f32 = logits.iter().copied().sum();
    if sum > 0.0 {
        for l in logits.iter_mut() {
            *l /= sum;
        }
    }
}

/// Sample a token index from a probability distribution using cumulative sampling.
///
/// Draws a uniform random value and walks the cumulative distribution.
fn sample_from_probs(probs: &[f32], rng: &mut fastrand::Rng) -> usize {
    let r = rng.f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i;
        }
    }
    // Fallback: return last valid index.
    probs.len().saturating_sub(1)
}

/// Temperature-scaled sampling from logits with mask token suppression.
///
/// Applies temperature scaling, softmax, then samples.
/// Returns (sampled_token, confidence) where confidence = max probability.
///
/// Writes into `probs_scratch` (caller-allocated, reused across calls) instead
/// of allocating a fresh `Vec` per invocation. `probs_scratch` is resized to
/// `logits.len()` if necessary.
fn sample_with_confidence(
    logits: &[f32],
    temperature: f32,
    mask_token_id: usize,
    rng: &mut fastrand::Rng,
    probs_scratch: &mut Vec<f32>,
) -> (usize, f32) {
    probs_scratch.clear();
    probs_scratch.extend_from_slice(logits);
    let probs = probs_scratch.as_mut_slice();

    // Suppress mask token (don't sample it).
    if mask_token_id < probs.len() {
        probs[mask_token_id] = f32::NEG_INFINITY;
    }

    // Apply temperature scaling.
    if temperature > 0.0 && temperature != 1.0 {
        let inv_temp = 1.0 / temperature;
        for p in probs.iter_mut() {
            *p *= inv_temp;
        }
    }

    softmax(probs);

    // Confidence = max probability.
    let confidence = probs.iter().copied().fold(0.0f32, f32::max);

    // Sample from distribution.
    let token = sample_from_probs(probs, rng);

    (token, confidence)
}

// ── D2F Decode Loop (T4) ───────────────────────────────────────────

/// Run D2F denoising decode loop for Gemma2.
///
/// Assert that `config` matches the values the block-causal attention kernel
/// hardcodes (Issue 695 H6).
///
/// The kernel (attention_cubecl block-causal variant) pins `softcap = 50.0`
/// and `scale = 0.0625` (1/√256) as literals; the `AttentionBlockCausalParams`
/// softcap/scale fields are never read. A config that disagrees computes
/// silently-wrong attention — this makes it fail at construction instead.
/// Lift when the kernel consumes the params fields.
fn assert_block_causal_kernel_pinned(config: &Config) {
    assert_eq!(
        config.head_dim, 256,
        "block-causal kernel hardcodes scale = 1/√256 (head_dim 256), got head_dim {}",
        config.head_dim
    );
    assert_eq!(
        config.attn_logit_softcapping, 50.0,
        "block-causal kernel hardcodes softcap = 50.0, got {} — configs that disable softcapping silently diverge",
        config.attn_logit_softcapping
    );
}

/// Iterative mask-and-refine: each step runs block-causal forward,
/// samples masked positions, unmasks those with sufficient confidence
/// (fixed threshold or learned `DiffusionSampler`).
///
/// Algorithm: `prompt + [mask; block]` → loop { forward → sample → unmask confident } → result
pub fn d2f_decode_gemma2(
    gpu: &mut GpuGemmaCubeCLD2F,
    prompt: &[usize],
    mask_token_id: usize,
    decode_config: &Gemma2D2fConfig,
    rng: &mut fastrand::Rng,
) -> Gemma2D2fResult {
    let block_size = decode_config.block_size;
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;
    let prompt_len_raw = prompt.len();

    // Initialize: prompt + mask tokens for the block.
    //
    // Issue 695 H7: `Config.block_size` is the model's MAX SEQUENCE length,
    // not the D2F block. The old `tokens.truncate(gpu.config().block_size)`
    // kept the FRONT of the sequence, which (a) cut the GENERATION block
    // whenever prompt_len + block > max — `step_confidence` divides by
    // `block_size`, so it could never reach 1.0 and the block never
    // converged — and (b) panicked at `masked[block_start..seq_len]` when
    // prompt_len >= max (slice start > end). Keep the prompt's TAIL (most
    // recent context) so the full block always fits, and clamp the prompt
    // length used downstream to what actually survived.
    let max_seq = gpu.config().block_size;
    let prompt_tail_start = prompt_len_raw.saturating_sub(max_seq.saturating_sub(block_size));
    let mut tokens: Vec<usize> = prompt[prompt_tail_start..].to_vec();
    tokens.extend(std::iter::repeat_n(mask_token_id, block_size));
    let seq_len = tokens.len();
    let block_start = prompt_len_raw - prompt_tail_start;
    let prompt_len = block_start; // clamped prompt length actually in `tokens`
    assert!(
        seq_len <= max_seq,
        "seq_len {seq_len} exceeds model block_size {max_seq}"
    );
    assert!(
        block_start <= seq_len,
        "block_start {block_start} > seq_len {seq_len}"
    );

    // Track which positions are still masked.
    let mut masked: Vec<bool> = vec![false; seq_len];
    masked[block_start..seq_len].fill(true);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut final_confidence = vec![0.0f32; seq_len];

    // SC state for self-conditioning (Plan 250 T4).
    // Initialized fresh per decode call. First step has no SC input (None).
    // After each step, update_from_logits() stores the x̂_0 estimate for
    // the next step. When SC is disabled, the state is never created.
    let sc_enabled = decode_config.sc_config.enabled;
    let mut sc_state = sc_enabled.then(|| D2fScState::new(decode_config.sc_config));
    let n_embd = gpu.config().n_embd;

    // Reusable softmax/sampling scratch buffer — allocated once, reused across
    // every (step × masked-position) iteration below. Avoids a full vocab-sized
    // heap allocation per `sample_with_confidence` call (vocab can be 256k for
    // Gemma2 → ~1MB per call previously).
    let vocab = gpu.config().vocab_size;
    let mut probs_scratch: Vec<f32> = Vec::with_capacity(vocab);

    for step in 0..max_steps {
        // Run block-causal forward for all positions.
        // Pass SC state if enabled (None on first step, x̂_0 estimate on later steps).
        let all_logits = match sc_state.as_ref() {
            Some(state) => gpu.block_causal_forward_with_sc(&tokens, prompt_len, Some(state)),
            None => gpu.block_causal_forward(&tokens, prompt_len),
        };

        let mut n_unmasked = 0usize;

        for pos in block_start..seq_len {
            if !masked[pos] {
                n_unmasked += 1;
                continue;
            }

            let logits = &all_logits[pos];

            // Sample token with confidence.
            let (sampled_token, confidence) =
                sample_with_confidence(logits, temperature, mask_token_id, rng, &mut probs_scratch);

            final_confidence[pos] = confidence;

            // Accept/reject: use sampler if available, else fixed threshold.
            let accept = match &decode_config.sampler {
                Some(sampler) => {
                    let features = SamplerFeatures::from_logits(
                        logits,
                        step,
                        max_steps,
                        pos - block_start,
                        block_size,
                        mask_token_id,
                    );
                    sampler.decide(&features, f64::from(tau_conf))
                }
                None => confidence >= tau_conf,
            };
            if accept && sampled_token != mask_token_id {
                tokens[pos] = sampled_token;
                masked[pos] = false;
                n_unmasked += 1;
            }
        }

        let step_confidence = n_unmasked as f32 / block_size as f32;
        confidence_history.push(step_confidence);

        // Update SC state from this step's logits (Plan 250 T4).
        // Computes x̂_0 estimate = softmax(logits/τ, exclude=mask) @ wte.
        // Available as SC input for the NEXT denoising step.
        // Stop-gradient is implicit: x̂_0 is plain data, not an autograd tensor.
        //
        // Issue 695 H16: gate the producer on the same cached identity verdict
        // that gates the consumer (`apply_projection = apply_sc &&
        // !self.w_sc_is_identity` in block_causal_forward_with_sc). With
        // untrained W_sc (the only in-tree-reachable state), the estimate is
        // an O(vocab × n_embd) host walk per position whose result is dropped
        // unused — 28.3 GB wte stream + 7.1 GMAC per step at the GOAT dims.
        if let Some(state) = sc_state.as_mut()
            && !gpu.w_sc_is_identity()
        {
            state.update_from_logits(&all_logits, gpu.wte_cpu_ref(), n_embd, mask_token_id);
        }

        // Early exit: all block positions unmasked.
        if (block_start..seq_len).all(|pos| !masked[pos]) {
            break;
        }
    }

    // Determine final state.
    let all_unmasked = (block_start..seq_len).all(|pos| !masked[pos]);
    let final_step_confidence = confidence_history.last().copied().unwrap_or(0.0);

    let state = if all_unmasked {
        D2fBlockState::FullyActivated
    } else {
        D2fBlockState::SemiActivated {
            step: confidence_history.len().saturating_sub(1),
            confidence: final_step_confidence,
        }
    };

    let steps_used = confidence_history.len();

    Gemma2D2fResult {
        tokens,
        steps_used,
        confidence: final_confidence,
        converged: all_unmasked,
        state,
        confidence_history,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
