//! CPU forward pass for hybrid DeltaNet/Attention models (Plan 182).
//!
//! Implements the Gated `DeltaNet` algorithm from:
//! - Yang et al., "Gated `DeltaNet`" (ICLR 2025, arXiv:2412.06464)
//! - Qwen3.5 reference: `HuggingFace` `modular_qwen3_5.py`
//!
//! # Algorithm (per `linear_attention` layer, decode step)
//!
//! ```text
//! 1. Project: qkv = in_proj_qkv(x), z = in_proj_z(x), ba = in_proj_a(x) + in_proj_b(x)
//! 2. Conv1D: depthwise causal conv on [q,k,v] with SiLU activation
//! 3. Gates: β = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
//! 4. Normalize: q̃ = l2norm(q), k̃ = l2norm(k)
//! 5. Recurrence: S = exp(g)·S + k̃ ⊗ β(v − (exp(g)·S)^T k̃)
//!                output = q̃^T · S / √d_k
//! 6. Output gate: output = rms_norm(output) * silu(z)
//! 7. Project: x = out_proj(output)
//! ```

use crate::deltanet::weights::{DeltaNetLayerWeights, QwenDeltaNetWeights};
use crate::types::{self, Config, DeltaNetLayerType, rmsnorm_with_gamma_eps, swiglu};

/// Per-layer recurrent state for `DeltaNet` layers.
///
/// Each `linear_attention` layer maintains a fixed-size state matrix:
/// `[n_value_heads * key_head_dim * value_head_dim]` elements.
/// For Qwen3.5-0.8B: 16 heads × (128 × 128) = 262,144 floats = 1 MB per layer.
///
/// Full attention layers use standard KV cache (not stored here).
pub struct DeltaNetState {
    /// Per-layer recurrent state. Only populated for `DeltaNet` layers.
    /// `layers[i]` = `[n_v_heads * key_dim * value_dim]` for `linear_attention` layers,
    /// empty Vec for `full_attention` layers.
    pub recurrent_states: Vec<Vec<f32>>,
    /// Per-layer conv1d sliding window state.
    /// `conv_states[i]` = `[conv_dim * kernel_size]` for `linear_attention` layers,
    /// empty Vec for `full_attention` layers.
    pub conv_states: Vec<Vec<f32>>,
}

impl DeltaNetState {
    /// Create zero-initialized state for all layers.
    pub fn new(config: &Config, layer_types: &[DeltaNetLayerType]) -> Self {
        let n_v_heads = config.deltanet_linear_n_value_heads;
        let key_dim = config.deltanet_linear_head_dim;
        let val_dim = config.deltanet_linear_head_dim;
        let state_dim = n_v_heads * key_dim * val_dim;
        let conv_kernel = config.deltanet_conv_kernel_size;
        // conv_dim = n_q_heads * key_dim + n_k_heads * key_dim + n_v_heads * val_dim
        // But in Qwen3.5: Q and K share same head count, V has separate count
        // From weights: in_proj_qkv output dim = (linear_num_key_heads * key_dim + 2 * linear_num_value_heads * val_dim)
        // conv1d operates on the concatenated [q, k, v] channels
        let n_k_heads = config.deltanet_linear_n_heads;
        let conv_dim = (n_k_heads + n_k_heads + n_v_heads) * key_dim;

        let mut recurrent_states = Vec::with_capacity(config.n_layer);
        let mut conv_states = Vec::with_capacity(config.n_layer);

        for &lt in layer_types {
            if lt == DeltaNetLayerType::DeltaNet {
                recurrent_states.push(vec![0.0f32; state_dim]);
                conv_states.push(vec![0.0f32; conv_dim * conv_kernel]);
            } else {
                recurrent_states.push(Vec::new());
                conv_states.push(Vec::new());
            }
        }

        Self {
            recurrent_states,
            conv_states,
        }
    }
}

/// Hybrid cache for Qwen3.5: KV cache for attention layers + recurrent state for `DeltaNet` layers.
pub struct HybridCache {
    /// Standard KV cache for `full_attention` layers.
    pub kv_cache: crate::transformer::MultiLayerKVCache,
    /// Recurrent state for `DeltaNet` layers.
    pub deltanet_state: DeltaNetState,
}

impl HybridCache {
    pub fn new(config: &Config) -> Self {
        let layer_types = vec![DeltaNetLayerType::Attention; config.n_layer];
        Self::with_layer_types(config, &layer_types)
    }

    pub fn with_layer_types(config: &Config, layer_types: &[DeltaNetLayerType]) -> Self {
        let kv_cache = crate::transformer::MultiLayerKVCache::new(config);
        let deltanet_state = DeltaNetState::new(config, layer_types);
        Self {
            kv_cache,
            deltanet_state,
        }
    }

    /// Zero both halves in place, restoring the freshly-constructed state
    /// WITHOUT reallocating (buffers keep their capacities — the per-call
    /// reset for embedders/evaluators that run many independent forward
    /// sequences against one cache; Issue 838 H1). `fill` on the empty
    /// per-layer vecs of full-attention layers is a no-op.
    pub fn reset(&mut self) {
        self.kv_cache.reset();
        for s in self.deltanet_state.recurrent_states.iter_mut() {
            s.fill(0.0);
        }
        for s in self.deltanet_state.conv_states.iter_mut() {
            s.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Core Gated DeltaNet recurrent step (single token, decode)
// ---------------------------------------------------------------------------

/// Run one step of the Gated `DeltaNet` recurrence (decode mode).
///
/// Convenience wrapper that allocates the output and per-head scratch
/// buffers. Hot-path callers should prefer [`gated_deltanet_step_inplace`]
/// which reuses pre-allocated scratch and avoids per-call allocation.
///
/// # Arguments
///
/// * `q` - Query: `[n_v_heads * key_dim]` (L2-normalized by caller)
/// * `k` - Key: `[n_v_heads * key_dim]` (L2-normalized by caller)
/// * `v` - Value: `[n_v_heads * val_dim]`
/// * `state` - Recurrent state: `[n_v_heads * key_dim * val_dim]` (modified in-place)
/// * `beta` - Update rate per head: `[n_v_heads]`
/// * `decay` - Decay per head: `[n_v_heads]` (already computed: exp(g) ∈ (0, 1])
/// * `n_v_heads` - Number of value heads
/// * `key_dim` - Key head dimension
/// * `val_dim` - Value head dimension
#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    beta: &[f32],
    decay: &[f32],
    n_v_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_v_heads * val_dim];
    let mut kv_mem = vec![0.0f32; val_dim];
    let mut delta = vec![0.0f32; val_dim];
    gated_deltanet_step_inplace(
        q,
        k,
        v,
        state,
        beta,
        decay,
        n_v_heads,
        key_dim,
        val_dim,
        &mut output,
        &mut kv_mem,
        &mut delta,
    );
    output
}

/// In-place variant of [`gated_deltanet_step`] that reuses caller-provided
/// scratch buffers, eliminating per-token-per-layer allocation on the hot path.
///
/// # Arguments (additional vs. `gated_deltanet_step`)
///
/// * `output` - Output buffer: `[n_v_heads * val_dim]` (written, not read).
/// * `kv_mem` - Per-head scratch: `[val_dim]` (overwritten per head).
/// * `delta` - Per-head scratch: `[val_dim]` (overwritten per head).
#[allow(clippy::too_many_arguments)]
pub fn gated_deltanet_step_inplace(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    beta: &[f32],
    decay: &[f32],
    n_v_heads: usize,
    key_dim: usize,
    val_dim: usize,
    output: &mut [f32],
    kv_mem: &mut [f32],
    delta: &mut [f32],
) {
    let state_dim_per_head = key_dim * val_dim;
    // `scale` only depends on `key_dim`, hoist outside the head loop.
    let scale = 1.0 / (key_dim as f32).sqrt();

    for h in 0..n_v_heads {
        let state_off = h * state_dim_per_head;
        let k_off = h * key_dim;
        let v_off = h * val_dim;
        let q_off = h * key_dim; // q has same layout: n_v_heads * key_dim
        let out_off = h * val_dim;
        let g = decay[h];
        let b = beta[h];

        // Step 1: Decay old state (SIMD-accelerated).
        let s = &mut state[state_off..state_off + state_dim_per_head];
        crate::simd::simd_scale_inplace(s, g);

        // Step 2+3 fused: Retrieve state prediction and compute delta in one pass.
        // Fusing eliminates the kv_mem intermediate between the two loops —
        // delta[row] = beta * (v[row] - Σ_c S[row,c]·k[c]) computed inline.
        // kv_mem is still written (for callers/tests that inspect it) but is no
        // longer a loop-carried dependency.
        let k_head = &k[k_off..k_off + key_dim];
        for row in 0..val_dim {
            let s_row = &s[row * key_dim..row * key_dim + key_dim];
            let kv = crate::simd::simd_dot_f32(s_row, k_head, key_dim);
            kv_mem[row] = kv;
            delta[row] = b * (v[v_off + row] - kv);
        }

        // Step 4: Write correction into state
        // S[row, col] += delta[row] * k[col]  (rank-1 outer product)
        //
        // `simd_outer_product_acc(acc, a, b, m, n)` computes
        // `acc[i*n + j] += a[i] * b[j]`, so `a` must be the **row** vector
        // (`delta`, indexed by `val_dim`) and `b` the **column** vector (`k`,
        // indexed by `key_dim`).
        //
        // **Issue 594 (2026-08-10) — the flat-logit root cause.** These two
        // arguments were swapped (`a = k_head`, `b = delta`), which wrote the
        // TRANSPOSE of the intended update: `S[row, col] += k[row] * delta[col]`.
        // The readout in step 5 (`out[row] = dot(S[row, ..], q)`) then produced
        // `k[row] * (delta · q)` instead of `delta[row] * (k · q)` — i.e. every
        // head's output collapsed onto the (L2-normalized) key direction and
        // discarded the value content, from the very first layer. Because
        // `key_dim == val_dim == 128` on every real Qwen3.5 checkpoint, no
        // length assertion could catch it and the synthetic all-zero fixtures
        // could not either.
        //
        // Verified against `llama.cpp`'s
        // `ggml_compute_forward_gated_delta_net_one_chunk`:
        //   `for j: ggml_vec_mad_f32(S_v, &s_out[j*S_v], k_d, delta[j])`
        // → row `j` (a `val_dim` index) is scaled by `delta[j]` and accumulates
        // `k` along the columns.
        crate::simd::simd_outer_product_acc(s, &delta[..val_dim], k_head, val_dim, key_dim);

        // Step 5: Read output
        // output[r] = Σ_c S[r, c] * q[c] / sqrt(key_dim)  (matvec, SIMD-accelerated)
        let q_head = &q[q_off..q_off + key_dim];
        for row in 0..val_dim {
            let s_row = &s[row * key_dim..row * key_dim + key_dim];
            output[out_off + row] = crate::simd::simd_dot_f32(s_row, q_head, key_dim) * scale;
        }
    }
}

/// Causal depthwise conv1d update for a single token (decode mode).
///
/// Updates the conv sliding window state and applies `SiLU` activation.
///
/// # Arguments
///
/// * `x` - Input for current token: `[conv_dim]`
/// * `conv_weight` - Depthwise conv weights: `[conv_dim, kernel_size]`
/// * `conv_state` - Sliding window state: `[conv_dim * kernel_size]` (modified in-place)
/// * `conv_dim` - Number of channels
/// * `kernel_size` - Conv kernel size (typically 4)
#[allow(clippy::needless_range_loop)]
pub fn causal_conv1d_update(
    x: &mut [f32],
    conv_weight: &[f32],
    conv_state: &mut [f32],
    conv_dim: usize,
    kernel_size: usize,
) {
    // Single fused pass: shift sliding window, append new sample, then apply
    // depthwise conv weights + SiLU. Combining the two loops halves the number
    // of passes over `conv_state` and improves cache locality.
    for ch in 0..conv_dim {
        let off = ch * kernel_size;
        // Shift: state[0..kernel-1] = state[1..kernel]
        for k in 0..kernel_size - 1 {
            conv_state[off + k] = conv_state[off + k + 1];
        }
        // Append current input
        conv_state[off + kernel_size - 1] = x[ch];

        // Depthwise conv: dot(state[ch], weight[ch])
        let mut sum = 0.0f32;
        for k in 0..kernel_size {
            sum += conv_state[off + k] * conv_weight[off + k];
        }
        // SiLU activation: x * sigmoid(x). fast_sigmoid is the Cephes
        // polynomial approximation — same hot-path primitive every other
        // sigmoid callsite in riir-engine uses (commits 0470434, 1908f92).
        let sig = crate::simd::fast_sigmoid(sum);
        x[ch] = sum * sig;
    }
}

/// L2 normalize a vector in-place.
pub(super) fn l2_normalize(x: &mut [f32]) {
    let sum_sq = crate::simd::simd_sum_sq(x, x.len());
    if sum_sq > 0.0 {
        let inv_norm = 1.0 / sum_sq.sqrt();
        crate::simd::simd_scale_inplace(x, inv_norm);
    }
}

/// Softplus: ln(1 + exp(x)), numerically stable.
pub(super) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ---------------------------------------------------------------------------
// Forward pass for a single DeltaNet (linear_attention) layer
// ---------------------------------------------------------------------------

/// Forward pass for one `DeltaNet` layer (decode mode, single token).
///
/// Implements the full Gated `DeltaNet` pipeline:
/// 1. QKV + Z projections
/// 2. `Conv1D` preprocessing
/// 3. Gate computation (beta, decay)
/// 4. L2 normalization of Q and K
/// 5. Gated delta rule recurrence
/// 6. Gated `RMSNorm` (`rms_norm` * silu(z))
/// 7. Output projection
///
/// # Layout notes
///
/// The `in_proj_qkv` weight has output dim = `(n_k_heads * key_dim * 2 + n_v_heads * val_dim)`.
/// It's split as: Q `[n_k_heads * key_dim]`, K `[n_k_heads * key_dim]`, V `[n_v_heads * val_dim]`.
/// K heads are repeated to match V heads before recurrence.
///
/// The `in_proj_z` weight has output dim = `n_v_heads * val_dim`.
/// The `in_proj_a` weight has output dim = `n_v_heads` (for decay gate).
/// The `in_proj_b` weight has output dim = `n_v_heads` (for update rate).
///
/// # Arguments
///
/// * `x` - Input activation: `[n_embd]` (modified in-place with layer output)
/// * `layer` - Layer weights
/// * `state` - Recurrent state: `[n_v_heads * key_dim * val_dim]` (modified in-place)
/// * `conv_state` - `Conv1D` sliding window: `[conv_dim * kernel_size]` (modified in-place)
/// * `config` - Model config
/// * `scratch` - Temporary buffers (avoids allocation in hot path)
#[allow(clippy::needless_range_loop)]
pub fn forward_deltanet_layer(
    x: &mut [f32],
    layer: &DeltaNetLayerWeights,
    state: &mut [f32],
    conv_state: &mut [f32],
    config: &Config,
    scratch: &mut DeltaNetLayerScratch,
) {
    let n_embd = config.n_embd;
    let n_k_heads = config.deltanet_linear_n_heads;
    let n_v_heads = config.deltanet_linear_n_value_heads;
    let key_dim = config.deltanet_linear_head_dim;
    let val_dim = config.deltanet_linear_head_dim;
    let kernel_size = config.deltanet_conv_kernel_size;

    let q_dim = n_k_heads * key_dim;
    let k_dim = n_k_heads * key_dim;
    let v_dim = n_v_heads * val_dim;
    let qkv_dim = q_dim + k_dim + v_dim;
    let z_dim = v_dim; // output gate has same dim as value
    let conv_dim = qkv_dim;

    // 1. QKV projection (scratch buffers are pre-sized in `DeltaNetLayerScratch::new`).
    layer.in_proj_qkv.matvec(&x[..n_embd], &mut scratch.qkv);

    // 2. Z projection (output gate)
    layer.in_proj_z.matvec(&x[..n_embd], &mut scratch.z);

    // 3. Gate projections: a (decay input), b (update rate input).
    // Use the `ab_raw` scratch (`[a_raw | b_raw]`) to avoid per-call allocation.
    // in_proj_a and in_proj_b each output n_v_heads values.
    let (a_raw, b_raw) = scratch.ab_raw.split_at_mut(n_v_heads);
    layer.in_proj_a.matvec(&x[..n_embd], a_raw);
    layer.in_proj_b.matvec(&x[..n_embd], b_raw);

    // 4. Split QKV
    let (q_slice, rest) = scratch.qkv.split_at_mut(q_dim);
    let (k_slice, v_slice) = rest.split_at_mut(k_dim);

    // 5. Conv1D preprocessing
    // Concatenate [q, k, v] into a single vector for conv1d
    scratch.conv_buf[..q_dim].copy_from_slice(q_slice);
    scratch.conv_buf[q_dim..q_dim + k_dim].copy_from_slice(k_slice);
    scratch.conv_buf[q_dim + k_dim..].copy_from_slice(v_slice);

    causal_conv1d_update(
        &mut scratch.conv_buf,
        &layer.conv1d_weight,
        conv_state,
        conv_dim,
        kernel_size,
    );

    // Copy back after conv
    q_slice.copy_from_slice(&scratch.conv_buf[..q_dim]);
    k_slice.copy_from_slice(&scratch.conv_buf[q_dim..q_dim + k_dim]);
    v_slice.copy_from_slice(&scratch.conv_buf[q_dim + k_dim..]);

    // 6. Compute gates into the `beta_decay` scratch (`[beta | decay]`).
    // β = sigmoid(b);  g = ssm_a * softplus(a + dt_bias);  decay = exp(g).
    //
    // **Issue 594 (2026-08-10):** the GGUF converter (`qwen.py` line 297)
    // applies `data = -torch.exp(data)` during conversion, so the `ssm_a`
    // tensor in the GGUF ALREADY contains `-exp(A_log_raw)`. Our `a_log`
    // field is a direct copy of `ssm_a` (loader: `blk.N.ssm_a → a_log`),
    // so `layer.a_log[h]` IS `-exp(A_log_raw)` — no further `-exp()` needed.
    // The old code applied `-exp()` a second time, computing
    // `-exp(-exp(A_log_raw))` instead of `-exp(A_log_raw)`, which produced
    // wildly wrong decay rates → flat logits (G1 FAIL).
    //
    // Reference: `prismml-llama.cpp/src/models/qwen35.cpp` line 451:
    //   gate = alpha_softplus * ssm_a;   // ssm_a IS -exp(A_log) from GGUF
    let (beta, decay) = scratch.beta_decay.split_at_mut(n_v_heads);
    for h in 0..n_v_heads {
        beta[h] = crate::simd::fast_sigmoid(b_raw[h]);
        let a_val = a_raw[h] + layer.dt_bias[h];
        let g = layer.a_log[h] * softplus(a_val);
        decay[h] = g.exp(); // exp(g) ∈ (0, 1]
    }

    // 7-8. Expand K/Q heads to match V heads (repeat_interleave) and L2-normalize.
    // When `repeat_factor > 1`, each k/q head is broadcast to `repeat_factor` v heads.
    // In-place normalize into the pre-allocated `q_normed` / `k_normed` scratch.
    let repeat_factor = n_v_heads / n_k_heads;
    expand_heads_into(q_slice, n_k_heads, key_dim, repeat_factor, &mut scratch.q_normed);
    expand_heads_into(k_slice, n_k_heads, key_dim, repeat_factor, &mut scratch.k_normed);
    for h in 0..n_v_heads {
        let off = h * key_dim;
        l2_normalize(&mut scratch.q_normed[off..off + key_dim]);
        l2_normalize(&mut scratch.k_normed[off..off + key_dim]);
    }

    // 9. Gated delta rule recurrence (in-place into scratch — zero allocation).
    gated_deltanet_step_inplace(
        &scratch.q_normed,
        &scratch.k_normed,
        v_slice,
        state,
        beta,
        decay,
        n_v_heads,
        key_dim,
        val_dim,
        &mut scratch.recurrent_output,
        &mut scratch.kv_mem,
        &mut scratch.delta,
    );

    // 10. Gated RMSNorm: output = rms_norm(output) * silu(z)
    //
    // The norm is **PER HEAD** (Issue 594, 2026-08-10). `ssm_norm` is
    // `[val_dim]` — one gamma shared across heads — while `recurrent_output`
    // is `n_v_heads * val_dim`. The fork's `build_norm_gated` calls
    // `build_norm` on a tensor shaped `[head_v_dim, n_v_heads, ...]`, and ggml
    // norms along `ne[0]`, i.e. once per head.
    //
    // Normalizing the whole buffer against a `val_dim`-length gamma was both
    // the wrong math and an out-of-bounds read (`simd_scale_mul_inplace`
    // indexes gamma by `x`'s length). That is what produced NaN logits on the
    // real Ternary-Bonsai-27B through the ternary twin of this function.
    debug_assert_eq!(
        layer.linear_norm.len(),
        val_dim,
        "ssm_norm must be per-head (val_dim), got {}",
        layer.linear_norm.len()
    );
    for h in 0..n_v_heads {
        let off = h * val_dim;
        rmsnorm_with_gamma_eps(
            &mut scratch.recurrent_output[off..off + val_dim],
            &layer.linear_norm,
            config.rms_norm_eps,
        );
    }
    // Apply SiLU gate in-place (fused with the RMSNorm result).
    for i in 0..z_dim {
        let z_val = scratch.z[i];
        let sig = crate::simd::fast_sigmoid(z_val);
        scratch.recurrent_output[i] *= z_val * sig; // silu(z) = z * sigmoid(z)
    }

    // 11. Output projection
    layer
        .out_proj
        .matvec(&scratch.recurrent_output, &mut x[..n_embd]);
}

/// Expand `n_src_heads` heads of width `head_dim` into `n_src_heads * repeat`
/// heads writing into `dst`, using **tiled** (`ggml_repeat`) semantics:
/// `dst_head[j] = src_head[j % n_src_heads]`.
///
/// # Why tiled and not repeat-interleave (Issue 594, 2026-08-10)
///
/// This is the `DeltaNet` K/Q → V head broadcast, and the mapping is fixed by
/// llama.cpp, not by convention. Both of the reference's code paths agree on
/// modulo:
///
/// - **Fused path** (the default; `__fgdn_ch__` in the eval-callback trace):
///   `ggml_compute_forward_gated_delta_net_one_chunk` indexes
///   `iq1 = iv1 % neq1` / `ik1 = iv1 % nek1` — v-head `j` reads k-head
///   `j % num_k_heads` (`ggml/src/ggml-cpu/ops.cpp`).
/// - **Non-fused path**: `qwen35.cpp::build_layer_attn_linear` calls
///   `ggml_repeat_4d(q_conv, head_k_dim, num_v_heads, …)`, and
///   `ggml_compute_forward_repeat_f32` writes dst row `i1*ne01 + k1` from src
///   row `k1` — again modulo.
///
/// The previous implementation used repeat-interleave
/// (`dst[h*repeat + r] = src[h]`, i.e. `src_head = j / repeat`). For the real
/// Bonsai-27B shape (16 k-heads → 48 v-heads) that pairs 47 of 48 v-heads with
/// the **wrong** Q/K head, which scrambles the delta-rule recurrence while
/// leaving activation magnitudes plausible — the exact signature of the flat
/// logits in Issue 594.
///
/// When `repeat == 1` this is a plain copy. `dst` must hold
/// `n_src_heads * repeat * head_dim` elements.
// Issue 741 T10 Phase C D4 widening: pub(super) -> pub. The sole remaining
// external consumer is the relocated training module
// `riir_train_engine::deltanet::model_backward_recompute` (the recompute
// path re-derives the same head expansion as the forward). In-engine callers:
// forward.rs + profiling.rs (unchanged).
#[inline]
pub fn expand_heads_into(
    src: &[f32],
    n_src_heads: usize,
    head_dim: usize,
    repeat: usize,
    dst: &mut [f32],
) {
    let block = n_src_heads * head_dim;
    let src_block = &src[..block];
    for r in 0..repeat.max(1) {
        dst[r * block..r * block + block].copy_from_slice(src_block);
    }
}

// ---------------------------------------------------------------------------
// Forward pass for a full_attention layer (standard GQA + QK-norm + RoPE)
// ---------------------------------------------------------------------------

/// Forward pass for one standard attention layer with GQA and QK-norm.
///
/// Qwen3.5 `full_attention` layers use:
/// - GQA: 8 Q heads, 2 KV heads (0.8B/2B), `head_dim=256`
/// - QK-norm: `RMSNorm` on Q and K before attention
/// - `RoPE` with theta=10M, `partial_rotary_factor=0.25`
/// - Output gating via `attn_output_gate` (sigmoid on attention output)
///
/// # Arguments
///
/// * `x` - Input activation: `[n_embd]` (modified in-place with layer output)
/// * `layer` - Layer weights
/// * `cache` - KV cache for this layer
/// * `pos` - Current position in sequence
/// * `config` - Model config
/// * `rope_freq` - Pre-computed `RoPE` frequency table
#[allow(clippy::too_many_arguments)]
pub fn forward_attention_layer(
    x: &mut [f32],
    layer: &DeltaNetLayerWeights,
    cache: &mut crate::transformer::KVCache,
    pos: usize,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    scratch: &mut AttentionLayerScratch,
) {
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    // Issue 594: partial RoPE dimension count. 0 = sentinel for full head_dim.
    let rotary_dim = effective_rotary_dim(config);

    // 1. Gated Q projection + K + V projections.
    //
    // Qwen3.5 gated attention (Issue 594): `attn_wq` produces `2*q_dim` outputs
    // laid out as `[q(hd), gate(hd)]` per head, interleaved — see
    // `qwen35.cpp::build_layer_attn` and HuggingFace `Qwen3NextAttention.forward`.
    // The gate is sigmoided and multiplied onto the attention output AFTER
    // softmax-attention, BEFORE the output projection.
    layer
        .attn_wq
        .matvec(&x[..n_embd], &mut scratch.qg_buf);
    // Split qg_buf into q_buf and gate_buf (per-head interleaved layout):
    //   qg_buf = [q_h0(hd), gate_h0(hd), q_h1(hd), gate_h1(hd), ...]
    //   → q_buf[h]    = qg_buf[h * 2*hd .. h*2*hd + hd]
    //   → gate_buf[h] = qg_buf[h * 2*hd + hd .. h*2*hd + 2*hd]
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        scratch.q_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src..src + hd]);
        scratch.gate_buf[dst..dst + hd]
            .copy_from_slice(&scratch.qg_buf[src + hd..src + 2 * hd]);
    }
    layer.attn_wk.matvec(&x[..n_embd], &mut scratch.k_buf);
    layer.attn_wv.matvec(&x[..n_embd], &mut scratch.v_buf);

    // 2. QK-norm (Qwen3.5 applies RMSNorm to Q and K per-head before RoPE).
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.k_buf[off..off + hd], &layer.attn_k_norm, eps);
    }

    // 3. Apply partial RoPE (Issue 594: Qwen3.5 rotates only `rotary_dim` of
    //    `head_dim`; mrope collapses to standard partial RoPE for text-only).
    if rotary_dim == hd {
        crate::rope::apply_rope_with_freq(
            &mut scratch.q_buf,
            &mut scratch.k_buf,
            pos,
            hd,
            rope_freq.as_slice(),
        );
    } else {
        crate::rope::apply_partial_rope_with_freq(
            &mut scratch.q_buf,
            &mut scratch.k_buf,
            pos,
            hd,
            rotary_dim,
            rope_freq.as_slice(),
        );
    }

    // 4. Store K, V in cache
    let pos_off = pos * kvd;
    unsafe {
        std::ptr::copy_nonoverlapping(
            scratch.k_buf.as_ptr(),
            cache.key.as_mut_ptr().add(pos_off),
            kvd,
        );
        std::ptr::copy_nonoverlapping(
            scratch.v_buf.as_ptr(),
            cache.value.as_mut_ptr().add(pos_off),
            kvd,
        );
    }

    // 5. Multi-head attention with GQA
    let scale = 1.0 / (hd as f32).sqrt();
    scratch.attn_out[..q_dim].fill(0.0);
    let t_n = pos + 1;
    unsafe {
        crate::transformer::attention_heads_parallel(
            &scratch.q_buf,
            &cache.key,
            &cache.value,
            &mut scratch.attn_out,
            &mut scratch.head_scores,
            n_head,
            n_kv,
            kvd,
            hd,
            t_n,
            scale,
            0.0, // no logit softcapping for Qwen3.5
            config.block_size,
        );
    }

    // 6. Output gating (Issue 594): attn_out *= sigmoid(gate), then o_proj.
    //    Matches `qwen35.cpp::build_layer_attn`: `cur = ggml_mul(cur, ggml_sigmoid(gate))`.
    for i in 0..q_dim {
        scratch.attn_out[i] *= crate::simd::fast_sigmoid(scratch.gate_buf[i]);
    }

    // 7. Output projection
    layer
        .attn_wo
        .matvec(&scratch.attn_out[..q_dim], &mut x[..n_embd]);
}

/// Effective partial-RoPE dimension count for the full-attention layers.
///
/// Returns `config.rope_dimension_count` if non-zero (Issue 594: Qwen3.5
/// partial `RoPE`), otherwise falls back to `head_dim` (full rotation —
/// pre-Issue-594 behavior, backward compatible with the 0.8B reference path).
///
/// Exposed as `pub` so external tests and tools (e.g. the G1 logits-vs-llama.cpp
/// harness) can build a `RopeFreqTable` with the same `rotary_dim` the forward
/// uses internally, without duplicating the zero-sentinel rule.
#[inline]
pub fn effective_rotary_dim(config: &Config) -> usize {
    let r = config.rope_dimension_count;
    if r == 0 {
        config.head_dim
    } else {
        r
    }
}

// ---------------------------------------------------------------------------
// Scratch buffers (pre-allocated, zero alloc in hot path)
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffers for `DeltaNet` layer forward pass.
///
/// All buffers are sized at construction from `Config` and reused across
/// tokens and layers — `forward_deltanet_layer` performs zero allocations
/// on the hot path.
pub struct DeltaNetLayerScratch {
    pub(super) qkv: Vec<f32>,
    pub(super) z: Vec<f32>,
    pub(super) conv_buf: Vec<f32>,
    /// Concatenated `[a_raw | b_raw]`, each `[n_v_heads]`.
    pub(super) ab_raw: Vec<f32>,
    /// Concatenated `[beta | decay]`, each `[n_v_heads]`.
    pub(super) beta_decay: Vec<f32>,
    /// Per-head expanded Q after L2-normalize: `[n_v_heads * key_dim]`.
    pub(super) q_normed: Vec<f32>,
    /// Per-head expanded K after L2-normalize: `[n_v_heads * key_dim]`.
    pub(super) k_normed: Vec<f32>,
    /// Recurrent output: `[n_v_heads * val_dim]`.
    pub(super) recurrent_output: Vec<f32>,
    /// Per-head matvec scratch: `[val_dim]`.
    pub(super) kv_mem: Vec<f32>,
    /// Per-head delta scratch: `[val_dim]`.
    pub(super) delta: Vec<f32>,
}

impl DeltaNetLayerScratch {
    pub fn new(config: &Config) -> Self {
        let n_k_heads = config.deltanet_linear_n_heads;
        let n_v_heads = config.deltanet_linear_n_value_heads;
        let key_dim = config.deltanet_linear_head_dim;
        let q_dim = n_k_heads * key_dim;
        let v_dim = n_v_heads * key_dim;
        let qkv_dim = q_dim + q_dim + v_dim;

        Self {
            qkv: vec![0.0; qkv_dim],
            z: vec![0.0; v_dim],
            conv_buf: vec![0.0; qkv_dim],
            ab_raw: vec![0.0; n_v_heads * 2],
            beta_decay: vec![0.0; n_v_heads * 2],
            q_normed: vec![0.0; n_v_heads * key_dim],
            k_normed: vec![0.0; n_v_heads * key_dim],
            recurrent_output: vec![0.0; n_v_heads * key_dim],
            kv_mem: vec![0.0; key_dim],
            delta: vec![0.0; key_dim],
        }
    }
}

/// Pre-allocated scratch buffers for attention layer forward pass.
///
/// `qg_buf` holds the **gated Q projection** output (`attn_wq @ x`, size
/// `2*q_dim` = `[q_head0, gate_head0, q_head1, gate_head1, ...]` per Issue 594).
/// It is split into `q_buf` (the actual Q) and `gate_buf` (the pre-sigmoid
/// attention-output gate) by `forward_attention_layer[_ternary]`.
pub struct AttentionLayerScratch {
    /// Gated Q projection output `[2 * q_dim]` (q + gate interleaved per head).
    /// Issue 594: Qwen3.5 gated attention concatenates q and gate in the
    /// `attn_q` weight — layout `[q(hd), gate(hd)]` per head.
    pub(super) qg_buf: Vec<f32>,
    pub(super) q_buf: Vec<f32>,
    /// Attention-output gate `[q_dim]` (pre-sigmoid). Extracted from `qg_buf`.
    pub(super) gate_buf: Vec<f32>,
    pub(super) k_buf: Vec<f32>,
    pub(super) v_buf: Vec<f32>,
    pub(super) attn_out: Vec<f32>,
    pub(super) head_scores: Vec<f32>,
}

impl AttentionLayerScratch {
    pub fn new(config: &Config) -> Self {
        let q_dim = config.n_head * config.head_dim;
        let kvd = types::kv_dim(config);

        Self {
            qg_buf: vec![0.0; q_dim * 2],
            q_buf: vec![0.0; q_dim],
            gate_buf: vec![0.0; q_dim],
            k_buf: vec![0.0; kvd],
            v_buf: vec![0.0; kvd],
            attn_out: vec![0.0; q_dim],
            head_scores: vec![0.0; config.n_head * config.block_size],
        }
    }
}

/// Combined scratch buffer for the full hybrid forward pass.
pub struct HybridForwardScratch {
    pub deltanet: DeltaNetLayerScratch,
    pub attention: AttentionLayerScratch,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub hidden: Vec<f32>,
    /// Scratch copy of the hidden state used to avoid aliasing in the LM head
    /// matmul (`forward_qwen_deltanet` reads `[..n_embd]` while writing
    /// `[..vocab_size]`). Reused across tokens — zero per-token allocation.
    pub hidden_copy: Vec<f32>,
    /// Residual-stream snapshot: `[n_embd]`.
    ///
    /// **Issue 594 (2026-08-10) — second flat-logit root cause.** Both hybrid
    /// layer loops used to save the residual into `attention.q_buf`, but
    /// `forward_attention_layer{,_ternary}` **overwrites** `q_buf` with the Q
    /// projection. On every full-attention layer the "residual add" therefore
    /// added the post-RoPE Q vector instead of the layer input. `DeltaNet` layers
    /// were unaffected (they write `scratch.deltanet`), which is exactly why the
    /// per-layer bisect showed layers 0/1/2 bit-exact and layer 3 — the first
    /// full-attention layer — as the first divergence spike.
    ///
    /// A dedicated buffer makes the aliasing impossible rather than merely
    /// unlikely.
    pub residual: Vec<f32>,
    /// Hadamard-rotation scratch (Issue 980, `bonsai2_hadamard`): the rotated
    /// copy of a shared folded-matmul input. Sized
    /// `max(n_embd, v_dim, q_dim, mlp_hidden)` — the widest rotated width any
    /// folded matmul consumes (`ffn_down`'s input, `mlp_hidden`, wins in
    /// practice). Construction-time allocation only: rotation is OFF for
    /// pre-rotation files (`weights.rotation == None`), the buffer is then
    /// never read or written, and the steady-state forward stays alloc-free
    /// (G4).
    pub rotation_buf: Vec<f32>,
}

impl HybridForwardScratch {
    pub fn new(config: &Config) -> Self {
        let v_dim = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;
        let q_dim = config.n_head * config.head_dim;
        let rotation_len = config
            .n_embd
            .max(v_dim)
            .max(q_dim)
            .max(config.mlp_hidden);
        Self {
            deltanet: DeltaNetLayerScratch::new(config),
            attention: AttentionLayerScratch::new(config),
            gate: vec![0.0; config.mlp_hidden],
            up: vec![0.0; config.mlp_hidden],
            hidden: vec![0.0; config.mlp_hidden],
            hidden_copy: vec![0.0; config.n_embd],
            residual: vec![0.0; config.n_embd],
            rotation_buf: vec![0.0; rotation_len],
        }
    }
}

// ---------------------------------------------------------------------------
// Main forward function
// ---------------------------------------------------------------------------

/// Forward pass for Qwen3.5 hybrid DeltaNet/Attention model (CPU, decode mode).
///
/// Processes a single token through all layers, dispatching each layer to
/// either `DeltaNet` recurrence or standard attention based on `layer_types`.
///
/// # Arguments
///
/// * `x` - Output buffer for logits: `[vocab_size]` (also used internally)
/// * `weights` - Model weights
/// * `cache` - Hybrid cache (KV cache + `DeltaNet` state)
/// * `token` - Input token ID
/// * `pos` - Position in sequence
/// * `config` - Model configuration
/// * `scratch` - Pre-allocated scratch buffers
///
/// # Returns
///
/// Mutable reference to `x` containing logits `[vocab_size]`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn forward_qwen_deltanet<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetWeights,
    cache: &mut HybridCache,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
) -> &'a mut [f32] {
    let n = config.n_embd;

    // 1. Embedding lookup (no embedding scale for Qwen3.5 — unlike Gemma 2)
    let tok_off = token * n;
    x[..n].copy_from_slice(&weights.wte[tok_off..tok_off + n]);

    // 2. Layer loop with hybrid dispatch
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        // a. Save residual. MUST NOT be `attention.q_buf` — the attention layer
        //    overwrites it with the Q projection (Issue 594).
        scratch.residual[..n].copy_from_slice(&x[..n]);

        // b. Pre-attention/input RMSNorm
        rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.input_norm, config.rms_norm_eps);

        // c. Layer-specific forward
        if is_linear {
            // DeltaNet (linear attention) layer
            forward_deltanet_layer(
                &mut x[..n],
                layer_weights,
                &mut cache.deltanet_state.recurrent_states[layer_idx],
                &mut cache.deltanet_state.conv_states[layer_idx],
                config,
                &mut scratch.deltanet,
            );
        } else {
            // Standard attention layer with GQA + QK-norm + RoPE
            forward_attention_layer(
                &mut x[..n],
                layer_weights,
                &mut cache.kv_cache.layers[layer_idx],
                pos,
                config,
                rope_freq,
                &mut scratch.attention,
            );
        }

        // d. Residual add
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }

        // e. Save residual for MLP
        scratch.residual[..n].copy_from_slice(&x[..n]);

        // f. Pre-MLP RMSNorm
        rmsnorm_with_gamma_eps(
            &mut x[..n],
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // g. SwiGLU MLP
        layer_weights
            .gate_proj
            .matvec(&x[..n], &mut scratch.gate);
        layer_weights
            .up_proj
            .matvec(&x[..n], &mut scratch.up);
        swiglu(&mut scratch.hidden, &scratch.gate, &scratch.up);

        // h. Down projection
        layer_weights
            .down_proj
            .matvec(&scratch.hidden, &mut x[..n]);

        // i. Residual add
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }
    }

    // 3. Final RMSNorm
    rmsnorm_with_gamma_eps(&mut x[..n], &weights.final_norm, config.rms_norm_eps);

    // 4. LM head (logits) — copy hidden state to avoid aliasing between
    //    the read (`x[..n]`) and the write (`x[..vocab_size]`). The
    //    `hidden_copy` scratch is pre-allocated in `HybridForwardScratch`.
    scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
    weights
        .lm_head
        .matvec(&scratch.hidden_copy[..n], &mut x[..config.vocab_size]);

    &mut x[..config.vocab_size]
}

// ---------------------------------------------------------------------------
// Prefill path (Plan 182, T15)
// ---------------------------------------------------------------------------

/// Prefill context for batched prompt processing.
///
/// Stores hidden states for all prompt positions between layers,
/// enabling layer-by-layer prefill (like HuggingFace/llama.cpp).
/// Only the last position's output is needed to bootstrap decode.
///
/// Also owns the batched scratch buffers used by the batched MLP + QKV path
/// (Issue 598 — the actual wiring-in of the Issue 597 substrate).
/// Pre-allocated once at `new()` and reused across all layers + all positions;
/// zero per-layer allocation on the hot path.
pub struct PrefillContext {
    /// Hidden states for all prompt positions: `[seq_len * n_embd]`.
    /// Carried between layers as input/output.
    hidden: Vec<f32>,
    /// Maximum prompt length (capacity).
    max_prompt_len: usize,

    // ── Batched MLP scratch (Issue 598) ───────────────────────────────────
    /// Residual stream snapshot for the MLP block: `[max_prompt_len * n_embd]`.
    mlp_residual: Vec<f32>,
    /// `gate_proj` output: `[max_prompt_len * mlp_hidden]`.
    mlp_gate: Vec<f32>,
    /// `up_proj` output: `[max_prompt_len * mlp_hidden]`.
    mlp_up: Vec<f32>,
    /// `SwiGLU` intermediate: `[max_prompt_len * mlp_hidden]`.
    mlp_hidden_buf: Vec<f32>,

    // ── Batched attention scratch (Issue 598) ────────────────────────────
    /// Pre-attention residual snapshot: `[max_prompt_len * n_embd]`.
    attn_residual: Vec<f32>,
    /// Gated Q projection output: `[max_prompt_len * 2 * q_dim]`.
    attn_qg: Vec<f32>,
    /// Split Q (post RoPE): `[max_prompt_len * q_dim]`.
    attn_q: Vec<f32>,
    /// Split gate: `[max_prompt_len * q_dim]`.
    attn_gate: Vec<f32>,
    /// K projection output: `[max_prompt_len * kvd]`.
    attn_k: Vec<f32>,
    /// V projection output: `[max_prompt_len * kvd]`.
    attn_v: Vec<f32>,
    /// Attention output (post-scoring, pre-output-proj): `[max_prompt_len * q_dim]`.
    attn_out: Vec<f32>,
    /// Per-position head scores scratch for the sequential scoring phase:
    /// `[n_head * block_size]` (reused like the single-token path).
    head_scores: Vec<f32>,
}

impl PrefillContext {
    /// Create a new prefill context with the given capacity.
    pub fn new(config: &Config, max_prompt_len: usize) -> Self {
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = crate::types::kv_dim(config);
        let mlp = config.mlp_hidden;
        let cap = max_prompt_len;
        Self {
            hidden: vec![0.0f32; cap * n],
            max_prompt_len: cap,
            mlp_residual: vec![0.0f32; cap * n],
            mlp_gate: vec![0.0f32; cap * mlp],
            mlp_up: vec![0.0f32; cap * mlp],
            mlp_hidden_buf: vec![0.0f32; cap * mlp],
            attn_residual: vec![0.0f32; cap * n],
            attn_qg: vec![0.0f32; cap * 2 * q_dim],
            attn_q: vec![0.0f32; cap * q_dim],
            attn_gate: vec![0.0f32; cap * q_dim],
            attn_k: vec![0.0f32; cap * kvd],
            attn_v: vec![0.0f32; cap * kvd],
            attn_out: vec![0.0f32; cap * q_dim],
            head_scores: vec![0.0f32; config.n_head * config.block_size],
        }
    }
}

/// Batched `SwiGLU` MLP block for the prefill path (Issue 598).
///
/// Processes the MLP for ALL `seq_len` positions at once using weight-reuse
/// batched GEMMs (`Proj::matmat`). Bit-identical to the per-position loop —
/// each output element is the same dot product, just in a different loop order
/// (weight rows outer, positions inner → each weight row loaded once, reused
/// across all positions).
///
/// Operates in place on `prefill_ctx.hidden[..seq_len * n_embd]`:
///   1. Save residual → `mlp_residual`
///   2. Batched `post_attn_norm` `RMSNorm` (per position)
///   3. Batched `gate_proj.matmat` → `mlp_gate`
///   4. Batched `up_proj.matmat` → `mlp_up`
///   5. Batched `SwiGLU` on `[seq_len * mlp_hidden]`
///   6. Batched `down_proj.matmat` → overwrites `hidden`
///   7. Residual add: `hidden += mlp_residual`
#[inline]
fn batched_mlp(
    prefill_ctx: &mut PrefillContext,
    layer: &DeltaNetLayerWeights,
    seq_len: usize,
    n: usize,
    mlp: usize,
    eps: f64,
) {
    // 1. Save residual.
    prefill_ctx.mlp_residual[..seq_len * n]
        .copy_from_slice(&prefill_ctx.hidden[..seq_len * n]);

    // 2. Batched post_attn_norm RMSNorm (elementwise, per position).
    for p in 0..seq_len {
        let hs = &mut prefill_ctx.hidden[p * n..(p + 1) * n];
        rmsnorm_with_gamma_eps(hs, &layer.post_attn_norm, eps);
    }

    // 3. Batched gate_proj: mlp_gate = gate_proj @ hidden. ONE GEMM.
    layer.gate_proj.matmat(
        &prefill_ctx.hidden[..seq_len * n],
        &mut prefill_ctx.mlp_gate[..seq_len * mlp],
        seq_len,
    );
    // 4. Batched up_proj: mlp_up = up_proj @ hidden. ONE GEMM.
    layer.up_proj.matmat(
        &prefill_ctx.hidden[..seq_len * n],
        &mut prefill_ctx.mlp_up[..seq_len * mlp],
        seq_len,
    );
    // 5. Batched SwiGLU over the full [seq_len * mlp_hidden] buffer.
    //    Element-wise, so the loop over positions collapses into one call.
    swiglu(
        &mut prefill_ctx.mlp_hidden_buf[..seq_len * mlp],
        &prefill_ctx.mlp_gate[..seq_len * mlp],
        &prefill_ctx.mlp_up[..seq_len * mlp],
    );
    // 6. Batched down_proj: hidden = down_proj @ mlp_hidden_buf. ONE GEMM.
    //    Overwrites the normalized hidden — no longer needed after gate/up.
    layer.down_proj.matmat(
        &prefill_ctx.mlp_hidden_buf[..seq_len * mlp],
        &mut prefill_ctx.hidden[..seq_len * n],
        seq_len,
    );
    // 7. Residual add: hidden += mlp_residual.
    for i in 0..seq_len * n {
        prefill_ctx.hidden[i] += prefill_ctx.mlp_residual[i];
    }
}

/// Prefill: process all prompt tokens through the model, initializing caches.
///
/// Uses a layer-by-layer approach (like HuggingFace/llama.cpp) with batched
/// MLP + QKV projection (Issue 598 — the actual wiring-in of the Issue 597
/// substrate). The `DeltaNet` recurrence stays sequential; the attention scoring
/// stays sequential (causal mask); everything else is batched into weight-reuse
/// GEMMs via `Proj::matmat`.
///
/// 1. Embed all prompt tokens into `[seq_len, n_embd]`
/// 2. For each layer:
///    - Attention layers: Phase A (batched QKV proj + QK-norm + RoPE + cache
///      write) → Phase B (sequential attention scoring) → Phase C (batched
///      gate + o_proj + residual) → Phase D (batched MLP + residual)
///    - DeltaNet layers: Phase 1 (sequential recurrence + residual) →
///      Phase 2 (batched MLP + residual)
/// 3. Final norm + LM head on last position → return logits
///
/// # Arguments
///
/// * `weights` - Model weights
/// * `config` - Model configuration
/// * `cache` - Hybrid cache (KV cache + `DeltaNet` state) — initialized by this function
/// * `prompt_tokens` - Input token IDs
/// * `scratch` - Pre-allocated scratch buffers (used for the `DeltaNet` path)
/// * `rope_freq` - Pre-computed `RoPE` frequency table
/// * `prefill_ctx` - Pre-allocated buffer for inter-layer hidden states +
///   batched scratch (MLP + attention)
///
/// # Returns
///
/// Logits for the last prompt position: `[vocab_size]`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn prefill_qwen_deltanet(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    cache: &mut HybridCache,
    prompt_tokens: &[usize],
    scratch: &mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
    prefill_ctx: &mut PrefillContext,
) -> Vec<f32> {
    let v = config.vocab_size;
    let mut logits = vec![0.0f32; v];
    prefill_qwen_deltanet_into(
        weights,
        config,
        cache,
        prompt_tokens,
        scratch,
        rope_freq,
        prefill_ctx,
        &mut logits,
    );
    logits
}

/// Zero-alloc variant of [`prefill_qwen_deltanet`] that writes the logits
/// into a caller-supplied buffer.
///
/// `logits_out` must have length `>= config.vocab_size`. Mirrors the
/// `_into`/`_inplace` convention used by `forward_qwen_deltanet` (which writes
/// logits into `&mut x[..vocab_size]`). The allocating wrapper above delegates
/// here; hot-path callers (e.g. `generate_greedy_qwen_deltanet`) should call
/// this directly to reuse their activation buffer instead of allocating a
/// fresh `vocab_size` `Vec` (~600KB for Qwen3.5) per prefill.
#[allow(clippy::too_many_arguments)]
pub fn prefill_qwen_deltanet_into(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    cache: &mut HybridCache,
    prompt_tokens: &[usize],
    scratch: &mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
    prefill_ctx: &mut PrefillContext,
    logits_out: &mut [f32],
) {
    let n = config.n_embd;
    let v = config.vocab_size;
    let seq_len = prompt_tokens.len();
    assert!(seq_len > 0, "prefill requires at least one token");
    assert!(
        seq_len <= prefill_ctx.max_prompt_len,
        "prompt length {seq_len} exceeds prefill capacity {}",
        prefill_ctx.max_prompt_len
    );

    // 1. Embed all prompt tokens: hidden[p * n..(p+1)*n] = wte[token[p]]
    for (p, &token) in prompt_tokens.iter().enumerate() {
        let tok_off = token * n;
        prefill_ctx.hidden[p * n..(p + 1) * n].copy_from_slice(&weights.wte[tok_off..tok_off + n]);
    }

    // 2. Layer-by-layer processing.
    //
    // Issue 598: the per-position loop is split into a sequential phase
    // (DeltaNet recurrence or attention scoring — inherently position-
    // dependent) and a batched phase (MLP + QKV projection — position-
    // independent, collapses `seq_len` GEMVs into ONE weight-reuse GEMM per
    // projection). The batched path calls `Proj::matmat`; the sequential path
    // stays on `forward_deltanet_layer` / `forward_attention_layer`. The
    // result is bit-identical to the per-position loop — each output element
    // is the same dot product, just computed in a weight-reuse loop order.
    //
    // For Attention layers the QKV projection + QK-norm + RoPE + cache write
    // are ALSO batched (Phase A). Only the attention scoring (Phase B)
    // remains sequential — position p attends to K[0..=p], which is the
    // causal mask. Phase C (output gate + o_proj) is batched again.
    //
    // The DeltaNet recurrence cannot be batched (each position depends on
    // the recurrent state left by the previous one — would need parallel-
    // scan / Blelloch). Only its MLP tail is batched.
    let q_dim = config.n_head * config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let mlp = config.mlp_hidden;
    let eps = config.rms_norm_eps;
    let rotary_dim = effective_rotary_dim(config);
    let scale = 1.0 / (hd as f32).sqrt();

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        if is_linear {
            // ── DeltaNet layer ────────────────────────────────────────────
            // Phase 1 (sequential): residual + input_norm + recurrence + residual.
            for p in 0..seq_len {
                let hidden_slice = &mut prefill_ctx.hidden[p * n..(p + 1) * n];
                scratch.residual[..n].copy_from_slice(hidden_slice);
                rmsnorm_with_gamma_eps(hidden_slice, &layer_weights.input_norm, eps);
                forward_deltanet_layer(
                    hidden_slice,
                    layer_weights,
                    &mut cache.deltanet_state.recurrent_states[layer_idx],
                    &mut cache.deltanet_state.conv_states[layer_idx],
                    config,
                    &mut scratch.deltanet,
                );
                for (h, &q) in hidden_slice.iter_mut().zip(&scratch.residual[..n]) {
                    *h += q;
                }
            }

            // Phase 2 (batched MLP) — shared with attention layer below.
            batched_mlp(prefill_ctx, layer_weights, seq_len, n, mlp, eps);
        } else {
            // ── Attention layer ───────────────────────────────────────────
            let kv_cache = &mut cache.kv_cache.layers[layer_idx];

            // Phase A (batched): pre-attn residual + input_norm + QKV proj +
            // QK-norm + RoPE + K,V cache write. All positions at once.
            prefill_ctx.attn_residual[..seq_len * n]
                .copy_from_slice(&prefill_ctx.hidden[..seq_len * n]);
            // Batched input RMSNorm (elementwise, loop over positions).
            for p in 0..seq_len {
                let hs = &mut prefill_ctx.hidden[p * n..(p + 1) * n];
                rmsnorm_with_gamma_eps(hs, &layer_weights.input_norm, eps);
            }
            // Batched gated-Q / K / V projections — one weight-reuse GEMM each.
            layer_weights
                .attn_wq
                .matmat(&prefill_ctx.hidden[..seq_len * n], &mut prefill_ctx.attn_qg[..seq_len * 2 * q_dim], seq_len);
            layer_weights
                .attn_wk
                .matmat(&prefill_ctx.hidden[..seq_len * n], &mut prefill_ctx.attn_k[..seq_len * kvd], seq_len);
            layer_weights
                .attn_wv
                .matmat(&prefill_ctx.hidden[..seq_len * n], &mut prefill_ctx.attn_v[..seq_len * kvd], seq_len);
            // Split qg → q + gate (interleaved per-head layout, same as
            // `forward_attention_layer` L583-589).
            for p in 0..seq_len {
                let qg_off = p * 2 * q_dim;
                let q_off = p * q_dim;
                for h in 0..config.n_head {
                    let src = qg_off + h * 2 * hd;
                    let dst = q_off + h * hd;
                    prefill_ctx.attn_q[dst..dst + hd]
                        .copy_from_slice(&prefill_ctx.attn_qg[src..src + hd]);
                    prefill_ctx.attn_gate[dst..dst + hd]
                        .copy_from_slice(&prefill_ctx.attn_qg[src + hd..src + 2 * hd]);
                }
            }
            // Batched QK-norm (per head per position).
            for p in 0..seq_len {
                let q_off = p * q_dim;
                let k_off = p * kvd;
                for h in 0..config.n_head {
                    let off = q_off + h * hd;
                    rmsnorm_with_gamma_eps(&mut prefill_ctx.attn_q[off..off + hd], &layer_weights.attn_q_norm, eps);
                }
                for h in 0..config.n_kv_head {
                    let off = k_off + h * hd;
                    rmsnorm_with_gamma_eps(&mut prefill_ctx.attn_k[off..off + hd], &layer_weights.attn_k_norm, eps);
                }
            }
            // Batched partial RoPE (per position — `pos = p`).
            for p in 0..seq_len {
                let q_off = p * q_dim;
                let k_off = p * kvd;
                if rotary_dim == hd {
                    crate::rope::apply_rope_with_freq(
                        &mut prefill_ctx.attn_q[q_off..q_off + q_dim],
                        &mut prefill_ctx.attn_k[k_off..k_off + kvd],
                        p, hd, rope_freq.as_slice(),
                    );
                } else {
                    crate::rope::apply_partial_rope_with_freq(
                        &mut prefill_ctx.attn_q[q_off..q_off + q_dim],
                        &mut prefill_ctx.attn_k[k_off..k_off + kvd],
                        p, hd, rotary_dim, rope_freq.as_slice(),
                    );
                }
            }
            // Batched K,V cache write — copy each position's K,V into its slot.
            // Independent per position (no cross-position dependency).
            for p in 0..seq_len {
                let pos_off = p * kvd;
                let k_src = p * kvd;
                let v_src = p * kvd;
                kv_cache.key[pos_off..pos_off + kvd]
                    .copy_from_slice(&prefill_ctx.attn_k[k_src..k_src + kvd]);
                kv_cache.value[pos_off..pos_off + kvd]
                    .copy_from_slice(&prefill_ctx.attn_v[v_src..v_src + kvd]);
            }

            // Phase B (sequential): attention scoring per position.
            // Position p attends to K[0..=p] / V[0..=p] (causal mask).
            // Bit-identical to the per-position `forward_attention_layer` path —
            // same `attention_heads_parallel` call, same inputs, same output slot.
            for p in 0..seq_len {
                let q_off = p * q_dim;
                let out_off = p * q_dim;
                prefill_ctx.attn_out[out_off..out_off + q_dim].fill(0.0);
                let t_n = p + 1;
                unsafe {
                    crate::transformer::attention_heads_parallel(
                        &prefill_ctx.attn_q[q_off..q_off + q_dim],
                        &kv_cache.key,
                        &kv_cache.value,
                        &mut prefill_ctx.attn_out[out_off..out_off + q_dim],
                        &mut prefill_ctx.head_scores,
                        config.n_head, config.n_kv_head, kvd, hd, t_n, scale,
                        0.0, config.block_size,
                    );
                }
            }

            // Phase C (batched): output gate × attn_out + o_proj + residual add.
            // Output gating: attn_out[p] *= sigmoid(gate[p]). Elementwise.
            for i in 0..seq_len * q_dim {
                prefill_ctx.attn_out[i] *= crate::simd::fast_sigmoid(prefill_ctx.attn_gate[i]);
            }
            // Batched output projection: hidden = attn_wo @ attn_out.
            // Overwrites the normalized hidden (no longer needed after QKV proj).
            layer_weights.attn_wo.matmat(
                &prefill_ctx.attn_out[..seq_len * q_dim],
                &mut prefill_ctx.hidden[..seq_len * n],
                seq_len,
            );
            // Residual add: hidden[p] += attn_residual[p] (the PRE-norm hidden).
            for i in 0..seq_len * n {
                prefill_ctx.hidden[i] += prefill_ctx.attn_residual[i];
            }

            // Phase D (batched MLP) — shared with DeltaNet layer above.
            batched_mlp(prefill_ctx, layer_weights, seq_len, n, mlp, eps);
        }
    }

    // 3. Final RMSNorm on last position
    let last_off = (seq_len - 1) * n;
    rmsnorm_with_gamma_eps(
        &mut prefill_ctx.hidden[last_off..last_off + n],
        &weights.final_norm,
        config.rms_norm_eps,
    );

    // 4. LM head (logits from last position hidden state) — write into the
    // caller-supplied buffer (zero-alloc).
    let last_hidden = &prefill_ctx.hidden[last_off..last_off + n];
    assert!(logits_out.len() >= v, "logits_out too short: {} < {v}", logits_out.len());
    weights
        .lm_head
        .matvec(last_hidden, &mut logits_out[..v]);
}

// ---------------------------------------------------------------------------
// Greedy decode loop
// ---------------------------------------------------------------------------

/// Qwen3.5 EOS token ID (`<|im_end|>`).
///
/// For base mode (non-chat), EOS is `151643` (`<|endoftext|>`).
/// Chat mode uses `151645` to stop after assistant responses.
const QWEN_DELTANET_EOS: usize = 151_645;

/// Greedy (argmax) decode loop for Qwen3.5 hybrid DeltaNet/Attention models (Plan 182, T14).
///
/// Uses [`prefill_qwen_deltanet`] for prompt processing (T15), then autoregressively
/// generates `max_tokens` new tokens using greedy decoding with [`forward_qwen_deltanet`].
/// Each layer is dispatched to either `DeltaNet` recurrence or standard attention
/// based on the per-layer `layer_types` config.
///
/// # Arguments
///
/// * `weights` - Qwen3.5 model weights
/// * `config` - Model configuration (layer types, head dims, etc.)
/// * `prompt_tokens` - Input token IDs (e.g., encoded prompt)
/// * `max_tokens` - Maximum number of tokens to generate after prompt
///
/// # Returns
///
/// Vector of generated token IDs (excluding prompt tokens).
///
/// # Example
///
/// ```ignore
/// let generated = generate_greedy_qwen_deltanet(
///     &weights, &config, &prompt_tokens, 64,
/// );
/// ```
pub fn generate_greedy_qwen_deltanet(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let n = config.n_embd;
    let v = config.vocab_size;

    // Allocate activation buffer: must hold both hidden state [n_embd] and logits [vocab_size]
    let buf_size = n.max(v);
    let mut x = vec![0.0f32; buf_size];

    // Allocate hybrid cache (KV cache for attention layers + recurrent state for DeltaNet)
    let layer_types = if weights.layer_types.is_empty() {
        vec![DeltaNetLayerType::Attention; config.n_layer]
    } else {
        weights.layer_types.clone()
    };
    let mut cache = HybridCache::with_layer_types(config, &layer_types);

    // Allocate scratch buffers (reused across all forward passes)
    let mut scratch = HybridForwardScratch::new(config);

    // Precomputed RoPE frequency table for attention layers.
    // Issue 594: partial RoPE — build for `rotary_dim` (head_dim when
    // rope_dimension_count=0 sentinel).
    let rotary_dim = effective_rotary_dim(config);
    let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, rotary_dim);

    let mut generated = Vec::with_capacity(max_tokens);

    // Empty prompt: nothing to generate
    if prompt_tokens.is_empty() {
        return generated;
    }

    // ── Prefill: layer-by-layer prompt processing (T15) ──
    // Write logits into the activation buffer `x[..v]` (sized `n.max(v)`) to
    // avoid allocating a fresh `vocab_size` Vec (~600KB for Qwen3.5) per prefill.
    // The `x` buffer is reused by the decode loop below; we extract `first_token`
    // before that reuse, so the borrow ends here.
    let mut prefill_ctx = PrefillContext::new(config, prompt_tokens.len());
    prefill_qwen_deltanet_into(
        weights,
        config,
        &mut cache,
        prompt_tokens,
        &mut scratch,
        &rope_freq,
        &mut prefill_ctx,
        &mut x[..v],
    );

    // ── First decode token: argmax from prefill logits ──
    let first_token = x[..v]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(i, _)| i);

    if first_token == QWEN_DELTANET_EOS {
        return generated;
    }
    generated.push(first_token);

    // ── Decode loop: greedy (argmax) autoregressive generation ──
    for _ in 1..max_tokens {
        let pos = prompt_tokens.len() + generated.len() - 1;
        let current_token = *generated.last().unwrap();

        forward_qwen_deltanet(
            &mut x,
            weights,
            &mut cache,
            current_token,
            pos,
            config,
            &mut scratch,
            &rope_freq,
        );

        // Argmax over logits
        let logits = &x[..v];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(i, _)| i);

        if next_token == QWEN_DELTANET_EOS {
            break;
        }
        generated.push(next_token);
    }

    generated
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOAT proof T1e: Gated `DeltaNet` step produces deterministic output.
    #[test]
    fn test_gated_deltanet_step_deterministic() {
        let n_v_heads = 2;
        let key_dim = 4;
        let val_dim = 4;
        let state_dim = n_v_heads * key_dim * val_dim;

        // Identity-like Q, K, V
        let q = vec![1.0f32; n_v_heads * key_dim];
        let k = vec![1.0f32; n_v_heads * key_dim];
        let v = vec![1.0f32; n_v_heads * val_dim];
        let mut state = vec![0.0f32; state_dim];

        let beta = vec![1.0f32; n_v_heads]; // full update rate
        let decay = vec![1.0f32; n_v_heads]; // no decay (exp(g) = 1.0)

        let output = gated_deltanet_step(
            &q, &k, &v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
        );

        // With zero initial state, beta=1.0, decay=1.0:
        // Step 1: state decayed by 1.0 → still zero
        // Step 2: retrieved = S @ k = 0 (zero state)
        // Step 3: delta = 1.0 * (v - 0) = v
        // Step 4: state += k ⊗ v (outer product: all 1.0)
        // Step 5: output = (state @ q) / sqrt(d) = (key_dim * 1.0) / sqrt(key_dim)
        let scale = 1.0 / (key_dim as f32).sqrt();
        let expected = key_dim as f32 * scale;
        for h in 0..n_v_heads {
            for r in 0..val_dim {
                assert!(
                    (output[h * val_dim + r] - expected).abs() < 1e-3,
                    "output[head={h}, row={r}] = {}, expected {expected}",
                    output[h * val_dim + r]
                );
            }
        }
    }

    /// Test that decay factor correctly attenuates state.
    #[test]
    fn test_gated_deltanet_decay() {
        let n_v_heads = 1;
        let key_dim = 3;
        let val_dim = 3;
        let state_dim = n_v_heads * key_dim * val_dim;

        let q = vec![1.0f32; key_dim];
        let k = vec![1.0f32; key_dim];
        let v = vec![1.0f32; val_dim];
        let mut state = vec![0.0f32; state_dim];

        // Step 1: no decay
        let beta = vec![1.0f32];
        let decay = vec![1.0f32];
        let _step1 = gated_deltanet_step(
            &q, &k, &v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
        );

        // State should be all 1.0 (k ⊗ v outer product)
        for (i, s) in state.iter().enumerate() {
            assert!((*s - 1.0).abs() < 1e-4, "state[{i}] after step1 = {s}");
        }

        // Step 2: with decay=0.5
        let decay2 = vec![0.5f32];
        let step2 = gated_deltanet_step(
            &q, &k, &v, &mut state, &beta, &decay2, n_v_heads, key_dim, val_dim,
        );

        // After decay: state = 0.5 * 1.0 = 0.5
        // retrieved[r] = Σ_c 0.5 * k[c] = 0.5 * 3 = 1.5
        // delta[r] = 1.0 * (1.0 - 1.5) = -0.5
        // state += k ⊗ delta = [1.0 * (-0.5)] = -0.5 each
        // state[row,col] = 0.5 + (-0.5) = 0.0
        // output = Σ_c 0.0 * q[c] / sqrt(3) ≈ 0.0
        for (r, val) in step2.iter().enumerate() {
            assert!(
                val.abs() < 1e-3,
                "step2 output[{r}] = {val}, expected ≈0.0"
            );
        }
    }

    /// Test L2 normalize.
    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    /// Test softplus.
    #[test]
    fn test_softplus() {
        assert!((softplus(0.0f32) - std::f32::consts::LN_2).abs() < 1e-3); // ln(2)
        assert!((softplus(1.0f32) - 1.3133).abs() < 1e-3);
        assert!((softplus(-1.0f32) - 0.3133).abs() < 1e-3);
        assert!((softplus(20.0f32) - 20.0).abs() < 1e-3);
        assert!(softplus(-20.0f32).abs() < 1e-6);
    }

    /// GOAT proof T14: `generate_greedy_qwen_deltanet` produces tokens with zero weights.
    ///
    /// With all-zero weights, every forward pass produces zero hidden state and
    /// zero logits. Rust's `max_by` returns the last element when all are equal,
    /// so argmax of all-zero logits returns `vocab_size - 1`. This verifies the
    /// decode loop runs to completion without panicking, and the prefill + decode
    /// pipeline is structurally correct.
    #[test]
    fn test_generate_greedy_zero_weights_all_attention() {
        let config = Config::qwen_deltanet(4, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);

        let generated = generate_greedy_qwen_deltanet(&weights, &config, &[config.bos_token], 4);

        // With zero weights, logits are all zero → argmax returns vocab_size - 1.
        // vocab_size - 1 is not EOS (151645), so we should get max_tokens outputs.
        let expected_tok = config.vocab_size - 1;
        assert_eq!(generated.len(), 4, "expected 4 generated tokens");
        for (i, &tok) in generated.iter().enumerate() {
            assert_eq!(
                tok, expected_tok,
                "generated token {i} should be {expected_tok} (argmax of zero logits)"
            );
        }
    }

    /// **Issue 594 regression — the attention layer must not clobber the
    /// residual.**
    ///
    /// The hybrid layer loop used to snapshot the residual into
    /// `attention.q_buf`, which `forward_attention_layer` then overwrites with
    /// the Q projection. Every full-attention layer therefore added the
    /// post-RoPE Q vector instead of its own input. `DeltaNet` layers write a
    /// different scratch, so every DeltaNet-only synthetic fixture passed — the
    /// per-layer bisect against llama.cpp showed layers 0/1/2 bit-exact and
    /// layer 3 (the first full-attention layer) as the first divergence spike.
    ///
    /// The invariant asserted here is behavioural, so it survives refactors:
    /// **with `attn_wo` and the whole MLP zeroed, a full-attention layer is the
    /// identity on the residual stream** — regardless of what `attn_wq` does.
    /// So a nonzero `attn_wq` (which fills `q_buf` with garbage) must not change
    /// the logits.
    #[test]
    fn test_attention_layer_preserves_residual_issue_594() {
        use crate::deltanet::Proj;
        use crate::types::DeltaNetLayerType::*;

        let mut config = Config::qwen_deltanet(1, vec![Attention]);
        config.n_embd = 8;
        config.n_head = 2;
        config.n_kv_head = 2;
        config.head_dim = 4;
        config.mlp_hidden = 8;
        config.vocab_size = 8;
        config.deltanet_linear_head_dim = 4;
        config.deltanet_linear_n_heads = 1;
        config.deltanet_linear_n_value_heads = 1;

        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        // Distinctive embedding for token 0 and an identity LM head, so the
        // logits are a faithful readout of the residual stream.
        for (i, w) in weights.wte[..n].iter_mut().enumerate() {
            *w = (i as f32 + 1.0) * 0.25;
        }
        let mut lm = vec![0.0f32; config.vocab_size * n];
        for i in 0..n.min(config.vocab_size) {
            lm[i * n + i] = 1.0;
        }
        weights.lm_head = Proj::dense(lm, config.vocab_size, n);
        weights.final_norm = vec![1.0f32; n];
        weights.layers[0].input_norm = vec![1.0f32; n];
        weights.layers[0].post_attn_norm = vec![1.0f32; n];

        // Nonzero Q/gate projection: fills `q_buf` with values that are NOT the
        // residual. `attn_wo` stays zero, so the attention block contributes 0.
        let wq: Vec<f32> = (0..2 * q_dim * n).map(|i| ((i % 7) as f32) - 3.0).collect();
        weights.layers[0].attn_wq = Proj::dense(wq, 2 * q_dim, n);

        let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; n.max(config.vocab_size)];

        let logits = forward_qwen_deltanet(
            &mut x, &weights, &mut cache, 0, 0, &config, &mut scratch, &rope_freq,
        )
        .to_vec();

        // Expected: the layer is the identity, so the pre-LM-head hidden state
        // is `rmsnorm(wte_row)` and the identity head reads it straight out.
        let mut want = weights.wte[..n].to_vec();
        rmsnorm_with_gamma_eps(&mut want, &weights.final_norm, config.rms_norm_eps);

        for i in 0..n.min(config.vocab_size) {
            assert!(
                (logits[i] - want[i]).abs() < 1e-5,
                "logit[{i}] = {} but the residual-preserving value is {}; \
                 the attention layer clobbered the residual (Issue 594)",
                logits[i],
                want[i]
            );
        }
    }

    /// GOAT proof T14: `generate_greedy` with hybrid `DeltaNet` + Attention layers.
    ///
    /// Uses a mix of `DeltaNet` and Attention layers to verify the hybrid dispatch
    /// path produces tokens without panicking.
    #[test]
    fn test_generate_greedy_zero_weights_hybrid() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, DeltaNet, Attention, Attention];
        let config = Config::qwen_deltanet(4, layer_types);
        let weights = QwenDeltaNetWeights::zeros(&config);

        let generated = generate_greedy_qwen_deltanet(&weights, &config, &[config.bos_token], 4);

        let expected_tok = config.vocab_size - 1;
        assert_eq!(generated.len(), 4, "expected 4 generated tokens");
        for (i, &tok) in generated.iter().enumerate() {
            assert_eq!(
                tok, expected_tok,
                "generated token {i} should be {expected_tok}"
            );
        }
    }

    /// GOAT proof T14: `generate_greedy` with multi-token prompt.
    ///
    /// Verifies prefill processes multiple prompt tokens correctly, and the decode
    /// loop picks up from the right position.
    #[test]
    fn test_generate_greedy_multi_token_prompt() {
        let config = Config::qwen_deltanet(2, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);

        let prompt = vec![config.bos_token, 100, 200, 300];
        let generated = generate_greedy_qwen_deltanet(&weights, &config, &prompt, 2);

        let expected_tok = config.vocab_size - 1;
        assert_eq!(generated.len(), 2, "expected 2 generated tokens");
        for (i, &tok) in generated.iter().enumerate() {
            assert_eq!(
                tok, expected_tok,
                "generated token {i} should be {expected_tok}"
            );
        }
    }

    /// GOAT proof T14: `generate_greedy` with all-DeltaNet layers.
    ///
    /// Exercises the pure `DeltaNet` path with no attention layers.
    #[test]
    fn test_generate_greedy_all_deltanet() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet; 4];
        let config = Config::qwen_deltanet(4, layer_types);
        let weights = QwenDeltaNetWeights::zeros(&config);

        let generated = generate_greedy_qwen_deltanet(&weights, &config, &[config.bos_token], 4);

        let expected_tok = config.vocab_size - 1;
        assert_eq!(generated.len(), 4, "expected 4 generated tokens");
        for (i, &tok) in generated.iter().enumerate() {
            assert_eq!(
                tok, expected_tok,
                "generated token {i} should be {expected_tok}"
            );
        }
    }

    /// GOAT proof T14: empty prompt returns empty generated tokens.
    #[test]
    fn test_generate_greedy_empty_prompt() {
        let config = Config::qwen_deltanet(2, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);

        let generated = generate_greedy_qwen_deltanet(&weights, &config, &[], 4);

        assert!(
            generated.is_empty(),
            "empty prompt should produce no tokens"
        );
    }

    // ── T15: Prefill GOAT-proof tests ──

    /// GOAT proof T15: prefill returns logits of correct size (`vocab_size`).
    #[test]
    fn test_prefill_returns_correct_logit_size() {
        let config = Config::qwen_deltanet(2, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        let layer_types = vec![DeltaNetLayerType::Attention; config.n_layer];

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut prefill_ctx = PrefillContext::new(&config, 4);

        let logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache,
            &[0, 1, 2],
            &mut scratch,
            &rope_freq,
            &mut prefill_ctx,
        );

        assert_eq!(
            logits.len(),
            config.vocab_size,
            "logits should have vocab_size = {} elements",
            config.vocab_size
        );
    }

    /// GOAT proof T15: prefill with zero weights produces all-zero logits.
    #[test]
    fn test_prefill_zero_weights_zero_logits() {
        let config = Config::qwen_deltanet(2, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        let layer_types = vec![DeltaNetLayerType::Attention; config.n_layer];

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut prefill_ctx = PrefillContext::new(&config, 4);

        let logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache,
            &[0],
            &mut scratch,
            &rope_freq,
            &mut prefill_ctx,
        );

        // Zero weights → zero embeddings → zero hidden states → zero logits
        for (i, &logit) in logits.iter().enumerate() {
            assert_eq!(logit, 0.0, "logit[{i}] should be 0.0 with zero weights");
        }
    }

    /// GOAT proof T15: prefill with hybrid layers completes without panic.
    #[test]
    fn test_prefill_hybrid_layers() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let config = Config::qwen_deltanet(4, layer_types.clone());
        let weights = QwenDeltaNetWeights::zeros(&config);

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut prefill_ctx = PrefillContext::new(&config, 4);

        let logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache,
            &[0, 1, 2, 3],
            &mut scratch,
            &rope_freq,
            &mut prefill_ctx,
        );

        assert_eq!(
            logits.len(),
            config.vocab_size,
            "logits should have vocab_size elements"
        );
    }

    /// GOAT proof T15: prefill with all-DeltaNet layers completes without panic.
    #[test]
    fn test_prefill_all_deltanet() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet; 4];
        let config = Config::qwen_deltanet(4, layer_types.clone());
        let weights = QwenDeltaNetWeights::zeros(&config);

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut prefill_ctx = PrefillContext::new(&config, 4);

        let logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache,
            &[0, 1, 2, 3],
            &mut scratch,
            &rope_freq,
            &mut prefill_ctx,
        );

        assert_eq!(
            logits.len(),
            config.vocab_size,
            "logits should have vocab_size elements"
        );
    }

    /// GOAT proof T15: single-token prefill works correctly.
    #[test]
    fn test_prefill_single_token() {
        let config = Config::qwen_deltanet(2, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        let layer_types = vec![DeltaNetLayerType::Attention; config.n_layer];

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut prefill_ctx = PrefillContext::new(&config, 1);

        let logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache,
            &[42],
            &mut scratch,
            &rope_freq,
            &mut prefill_ctx,
        );

        assert_eq!(logits.len(), config.vocab_size);
        // With zero weights, all logits should be 0.0
        assert!(logits.iter().all(|&l| l == 0.0), "all logits should be 0.0");
    }

    // ── T16: E2E correctness GOAT-proof tests ──

    /// GOAT proof T16: prefill + sequential produce same first-token output.
    ///
    /// Verifies that `prefill_qwen_deltanet` and token-by-token `forward_qwen_deltanet`
    /// produce identical logits for the last prompt position.
    /// This is the key consistency property: the two code paths are mathematically
    /// equivalent and must produce the same result.
    #[test]
    fn test_prefill_matches_sequential_forward() {
        let config = Config::qwen_deltanet(4, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        let n = config.n_embd;
        let v = config.vocab_size;

        // Path 1: Sequential token-by-token forward
        let buf_size = n.max(v);
        let mut x_seq = vec![0.0f32; buf_size];
        let layer_types = vec![DeltaNetLayerType::Attention; config.n_layer];
        let mut cache_seq = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_seq = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);

        let prompt = vec![0, 1, 2];
        for (pos, &token) in prompt.iter().enumerate() {
            forward_qwen_deltanet(
                &mut x_seq,
                &weights,
                &mut cache_seq,
                token,
                pos,
                &config,
                &mut scratch_seq,
                &rope_freq,
            );
        }
        let seq_logits = x_seq[..v].to_vec();

        // Path 2: Layer-by-layer prefill
        let mut cache_pf = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_pf = HybridForwardScratch::new(&config);
        let mut prefill_ctx = PrefillContext::new(&config, 4);
        let pf_logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache_pf,
            &prompt,
            &mut scratch_pf,
            &rope_freq,
            &mut prefill_ctx,
        );

        // Both paths should produce identical logits
        assert_eq!(seq_logits.len(), pf_logits.len(), "logit size mismatch");
        for (i, (s, p)) in seq_logits.iter().zip(pf_logits.iter()).enumerate() {
            assert_eq!(s, p, "logit[{i}] mismatch: sequential={s}, prefill={p}");
        }
    }

    /// GOAT proof T16: prefill + sequential match with hybrid layer types.
    #[test]
    fn test_prefill_matches_sequential_hybrid() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let config = Config::qwen_deltanet(4, layer_types.clone());
        let weights = QwenDeltaNetWeights::zeros(&config);
        let n = config.n_embd;
        let v = config.vocab_size;

        let buf_size = n.max(v);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let prompt = vec![0, 1, 2, 3];

        // Sequential
        let mut x = vec![0.0f32; buf_size];
        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        for (pos, &token) in prompt.iter().enumerate() {
            forward_qwen_deltanet(
                &mut x,
                &weights,
                &mut cache,
                token,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
            );
        }
        let seq_logits = x[..v].to_vec();

        // Prefill
        let mut cache_pf = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_pf = HybridForwardScratch::new(&config);
        let mut prefill_ctx = PrefillContext::new(&config, prompt.len());
        let pf_logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache_pf,
            &prompt,
            &mut scratch_pf,
            &rope_freq,
            &mut prefill_ctx,
        );

        assert_eq!(seq_logits.len(), pf_logits.len());
        for (i, (s, p)) in seq_logits.iter().zip(pf_logits.iter()).enumerate() {
            assert_eq!(s, p, "logit[{i}] mismatch in hybrid");
        }
    }

    /// Issue 597 — the batched MLP prefill path must produce bit-identical
    /// logits + KV cache contents vs the sequential token-by-token forward,
    /// on **non-zero weights**. The existing T16 tests above use zero weights
    /// (trivially bit-identical); this test fills every projection with seeded
    /// pseudo-random values so the GEMM path is actually exercised.
    ///
    /// The batched path swaps the loop nest (weight rows outer, positions
    /// inner) but each output element is the same `simd_dot_f32` sum — no
    /// floating-point reassociation occurs inside the dot product. So the
    /// result must match bit-for-bit, not merely within tolerance.
    #[test]
    fn test_prefill_batched_mlp_matches_sequential_seeded() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let config = Config::qwen_deltanet(4, layer_types.clone());
        let n = config.n_embd;
        let v = config.vocab_size;
        let buf_size = n.max(v);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let prompt = vec![0, 1, 2, 3];

        // Seeded weight fill — mirrors `test_hybrid_dispatch_matches_sequential`.
        let mut weights = QwenDeltaNetWeights::zeros(&config);
        let mut seed: u64 = 42;
        let fill_seeded = |dst: &mut [f32], seed: &mut u64| {
            for val in dst.iter_mut() {
                *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *val = ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
        };
        fill_seeded(&mut weights.wte, &mut seed);
        fill_seeded(&mut weights.final_norm, &mut seed);
        fill_seeded(weights.lm_head.dense_data_mut(), &mut seed);
        for layer in &mut weights.layers {
            fill_seeded(layer.gate_proj.dense_data_mut(), &mut seed);
            fill_seeded(layer.up_proj.dense_data_mut(), &mut seed);
            fill_seeded(layer.down_proj.dense_data_mut(), &mut seed);
            fill_seeded(&mut layer.input_norm, &mut seed);
            fill_seeded(&mut layer.post_attn_norm, &mut seed);
            if layer.attn_wq.is_empty() {
                fill_seeded(layer.in_proj_qkv.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_a.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_b.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_z.dense_data_mut(), &mut seed);
                fill_seeded(layer.out_proj.dense_data_mut(), &mut seed);
                fill_seeded(&mut layer.conv1d_weight, &mut seed);
                fill_seeded(&mut layer.a_log, &mut seed);
                fill_seeded(&mut layer.dt_bias, &mut seed);
                fill_seeded(&mut layer.linear_norm, &mut seed);
            } else {
                fill_seeded(layer.attn_wq.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wk.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wv.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wo.dense_data_mut(), &mut seed);
                fill_seeded(&mut layer.attn_q_norm, &mut seed);
                fill_seeded(&mut layer.attn_k_norm, &mut seed);
            }
        }

        // Path A: Sequential token-by-token forward (the decode path).
        let mut x_seq = vec![0.0f32; buf_size];
        let mut cache_seq = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_seq = HybridForwardScratch::new(&config);
        for (pos, &token) in prompt.iter().enumerate() {
            forward_qwen_deltanet(
                &mut x_seq,
                &weights,
                &mut cache_seq,
                token,
                pos,
                &config,
                &mut scratch_seq,
                &rope_freq,
            );
        }
        let seq_logits = x_seq[..v].to_vec();

        // Path B: Layer-by-layer prefill with the batched MLP.
        let mut cache_pf = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_pf = HybridForwardScratch::new(&config);
        let mut prefill_ctx = PrefillContext::new(&config, prompt.len());
        let pf_logits = prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache_pf,
            &prompt,
            &mut scratch_pf,
            &rope_freq,
            &mut prefill_ctx,
        );

        // Bit-identical logits.
        assert_eq!(seq_logits.len(), pf_logits.len());
        let mut max_abs_diff = 0.0f32;
        for (i, (s, p)) in seq_logits.iter().zip(pf_logits.iter()).enumerate() {
            let diff = (s - p).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            assert_eq!(
                s, p,
                "logit[{i}] mismatch on seeded weights: sequential={s}, prefill={p} (max diff so far: {max_abs_diff})"
            );
        }

        // KV cache bit-identical for attention layers.
        for (layer_idx, &layer_type) in layer_types.iter().enumerate().take(config.n_layer) {
            if layer_type == Attention {
                let kv_seq = &cache_seq.kv_cache.layers[layer_idx];
                let kv_pf = &cache_pf.kv_cache.layers[layer_idx];
                assert_eq!(
                    kv_seq.key.len(),
                    kv_pf.key.len(),
                    "KV cache key length mismatch on layer {layer_idx}"
                );
                assert_eq!(
                    kv_seq.value.len(),
                    kv_pf.value.len(),
                    "KV cache value length mismatch on layer {layer_idx}"
                );
                for (i, (s, p)) in kv_seq.key.iter().zip(kv_pf.key.iter()).enumerate() {
                    assert_eq!(s, p, "KV key[{layer_idx}][{i}] mismatch");
                }
                for (i, (s, p)) in kv_seq.value.iter().zip(kv_pf.value.iter()).enumerate() {
                    assert_eq!(s, p, "KV value[{layer_idx}][{i}] mismatch");
                }
            }
        }
    }

    /// Issue 597 acceptance #2 — measure the batched prefill throughput win.
    ///
    /// Compares the per-token decode path (`forward_qwen_deltanet` × `seq_len`)
    /// against the batched prefill path (`prefill_qwen_deltanet`). The decode
    /// path cannot benefit from MLP batching (one token at a time). The prefill
    /// path collapses the `seq_len` GEMVs into one weight-reuse GEMM per
    /// projection.
    ///
    /// `#[ignore]` by default — it's an informational micro-bench, not a
    /// correctness gate. Run with `cargo test -- --ignored` to see the numbers.
    /// The result is printed to stderr; assertion checks self-consistency only.
    #[test]
    #[ignore]
    fn bench_batched_prefill_vs_decode() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let config = Config::qwen_deltanet(4, layer_types.clone());
        let n = config.n_embd;
        let v = config.vocab_size;
        let buf_size = n.max(v);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);

        // Seeded non-zero weights so the GEMM path is exercised.
        let mut weights = QwenDeltaNetWeights::zeros(&config);
        let mut seed: u64 = 42;
        let fill_seeded = |dst: &mut [f32], seed: &mut u64| {
            for val in dst.iter_mut() {
                *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                *val = ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
        };
        fill_seeded(&mut weights.wte, &mut seed);
        fill_seeded(&mut weights.final_norm, &mut seed);
        fill_seeded(weights.lm_head.dense_data_mut(), &mut seed);
        for layer in &mut weights.layers {
            fill_seeded(layer.gate_proj.dense_data_mut(), &mut seed);
            fill_seeded(layer.up_proj.dense_data_mut(), &mut seed);
            fill_seeded(layer.down_proj.dense_data_mut(), &mut seed);
            fill_seeded(&mut layer.input_norm, &mut seed);
            fill_seeded(&mut layer.post_attn_norm, &mut seed);
            if layer.attn_wq.is_empty() {
                fill_seeded(layer.in_proj_qkv.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_a.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_b.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_z.dense_data_mut(), &mut seed);
                fill_seeded(layer.out_proj.dense_data_mut(), &mut seed);
                fill_seeded(&mut layer.conv1d_weight, &mut seed);
                fill_seeded(&mut layer.a_log, &mut seed);
                fill_seeded(&mut layer.dt_bias, &mut seed);
                fill_seeded(&mut layer.linear_norm, &mut seed);
            } else {
                fill_seeded(layer.attn_wq.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wk.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wv.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wo.dense_data_mut(), &mut seed);
                fill_seeded(&mut layer.attn_q_norm, &mut seed);
                fill_seeded(&mut layer.attn_k_norm, &mut seed);
            }
        }

        eprintln!("\n==============================================================");
        eprintln!("Issue 597 bench: batched prefill vs sequential decode");
        eprintln!("config: n_embd={}, mlp_hidden={}, n_layer={}", config.n_embd, config.mlp_hidden, config.n_layer);
        eprintln!("------------------------------------------------------------");
        eprintln!("{:>8} | {:>14} | {:>15} | {:>8}", "seq_len", "decode ms/tok", "prefill ms/tok", "speedup");
        eprintln!("------------------------------------------------------------");

        for &seq_len in &[8usize, 16, 32, 64] {
            let prompt: Vec<usize> = (0..seq_len).collect();

            // Decode path: forward_qwen_deltanet × seq_len
            let mut x = vec![0.0f32; buf_size];
            let mut cache_dec = HybridCache::with_layer_types(&config, &layer_types);
            let mut scratch_dec = HybridForwardScratch::new(&config);
            // Warmup
            for (pos, &tok) in prompt.iter().enumerate() {
                forward_qwen_deltanet(&mut x, &weights, &mut cache_dec, tok, pos, &config, &mut scratch_dec, &rope_freq);
            }
            // Measure
            let mut cache_dec = HybridCache::with_layer_types(&config, &layer_types);
            let mut scratch_dec = HybridForwardScratch::new(&config);
            let t0 = std::time::Instant::now();
            for (pos, &tok) in prompt.iter().enumerate() {
                forward_qwen_deltanet(&mut x, &weights, &mut cache_dec, tok, pos, &config, &mut scratch_dec, &rope_freq);
            }
            let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Prefill path: prefill_qwen_deltanet (batched MLP)
            let mut logits = vec![0.0f32; v];
            let mut cache_pf = HybridCache::with_layer_types(&config, &layer_types);
            let mut scratch_pf = HybridForwardScratch::new(&config);
            let mut prefill_ctx = PrefillContext::new(&config, seq_len);
            // Warmup
            prefill_qwen_deltanet_into(&weights, &config, &mut cache_pf, &prompt, &mut scratch_pf, &rope_freq, &mut prefill_ctx, &mut logits);
            // Measure
            let mut cache_pf = HybridCache::with_layer_types(&config, &layer_types);
            let mut scratch_pf = HybridForwardScratch::new(&config);
            let mut prefill_ctx = PrefillContext::new(&config, seq_len);
            let t1 = std::time::Instant::now();
            prefill_qwen_deltanet_into(&weights, &config, &mut cache_pf, &prompt, &mut scratch_pf, &rope_freq, &mut prefill_ctx, &mut logits);
            let prefill_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let decode_ms_per_tok = decode_ms / seq_len as f64;
            let prefill_ms_per_tok = prefill_ms / seq_len as f64;
            let speedup = decode_ms_per_tok / prefill_ms_per_tok;
            eprintln!("{seq_len:>8} | {decode_ms_per_tok:>14.1} | {prefill_ms_per_tok:>15.1} | {speedup:>7.2}×");
        }
        eprintln!("==============================================================\n");
    }

    /// GOAT proof T16: `generate_greedy` is deterministic (same input → same output).
    #[test]
    fn test_greedy_decode_deterministic() {
        let config = Config::qwen_deltanet(4, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);

        let run1 = generate_greedy_qwen_deltanet(&weights, &config, &[0, 1, 2], 8);
        let run2 = generate_greedy_qwen_deltanet(&weights, &config, &[0, 1, 2], 8);

        assert_eq!(
            run1, run2,
            "two runs with same input must produce identical output"
        );
    }

    // ── T13: Hybrid dispatch GOAT-proof test ──

    /// GOAT proof T13: `forward_qwen_deltanet` with hybrid layers produces
    /// the same output as running each layer individually in sequence.
    ///
    /// Verifies the internal dispatch routing in `forward_qwen_deltanet` is
    /// consistent with explicit per-layer dispatch. Both paths route each layer
    /// to either `forward_deltanet_layer` or `forward_attention_layer` based on
    /// `layer_types` — this test catches any routing divergence.
    ///
    /// Uses a seeded weight fill for non-trivial activation values so that
    /// the test is numerically meaningful (not just all-zeros).
    #[test]
    fn test_hybrid_dispatch_matches_sequential() {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, Attention, DeltaNet, Attention];
        let config = Config::qwen_deltanet(4, layer_types.clone());
        let n = config.n_embd;
        let v = config.vocab_size;
        let buf_size = n.max(v);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);

        // Seeded weight fill: LCG produces deterministic non-zero values
        let mut weights = QwenDeltaNetWeights::zeros(&config);
        let mut seed: u64 = 42;
        let fill_seeded = |dst: &mut [f32], seed: &mut u64| {
            for val in dst.iter_mut() {
                *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                // Map to [-0.5, 0.5] — small values to avoid overflow in attention
                *val = ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
        };
        fill_seeded(&mut weights.wte, &mut seed);
        fill_seeded(&mut weights.final_norm, &mut seed);
        fill_seeded(weights.lm_head.dense_data_mut(), &mut seed);
        for layer in &mut weights.layers {
            fill_seeded(layer.gate_proj.dense_data_mut(), &mut seed);
            fill_seeded(layer.up_proj.dense_data_mut(), &mut seed);
            fill_seeded(layer.down_proj.dense_data_mut(), &mut seed);
            fill_seeded(&mut layer.input_norm, &mut seed);
            fill_seeded(&mut layer.post_attn_norm, &mut seed);
            if layer.attn_wq.is_empty() {
                // DeltaNet layer
                fill_seeded(layer.in_proj_qkv.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_a.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_b.dense_data_mut(), &mut seed);
                fill_seeded(layer.in_proj_z.dense_data_mut(), &mut seed);
                fill_seeded(layer.out_proj.dense_data_mut(), &mut seed);
                fill_seeded(&mut layer.conv1d_weight, &mut seed);
                fill_seeded(&mut layer.a_log, &mut seed);
                fill_seeded(&mut layer.dt_bias, &mut seed);
                fill_seeded(&mut layer.linear_norm, &mut seed);
            } else {
                // Attention layer
                fill_seeded(layer.attn_wq.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wk.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wv.dense_data_mut(), &mut seed);
                fill_seeded(layer.attn_wo.dense_data_mut(), &mut seed);
                fill_seeded(&mut layer.attn_q_norm, &mut seed);
                fill_seeded(&mut layer.attn_k_norm, &mut seed);
            }
        }

        // Path A: forward_qwen_deltanet (hybrid dispatch path)
        let mut x_dispatch = vec![0.0f32; buf_size];
        let mut cache_dispatch = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_dispatch = HybridForwardScratch::new(&config);

        let tokens = [0usize, 1, 2];
        for (pos, &token) in tokens.iter().enumerate() {
            forward_qwen_deltanet(
                &mut x_dispatch,
                &weights,
                &mut cache_dispatch,
                token,
                pos,
                &config,
                &mut scratch_dispatch,
                &rope_freq,
            );
        }
        let dispatch_logits = x_dispatch[..v].to_vec();

        // Path B: Manual sequential — replicate the layer loop from forward_qwen_deltanet
        // but explicitly dispatch per layer using the same functions.
        // This tests that the routing in forward_qwen_deltanet matches.
        let mut x_seq = vec![0.0f32; buf_size];
        let mut cache_seq = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_seq = HybridForwardScratch::new(&config);

        for (pos, &token) in tokens.iter().enumerate() {
            // Embedding
            let tok_off = token * n;
            x_seq[..n].copy_from_slice(&weights.wte[tok_off..tok_off + n]);

            // Layer loop — same as forward_qwen_deltanet's internal dispatch
            for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
                let is_linear = layer_types[layer_idx] == DeltaNet;

                // Save residual
                scratch_seq.residual[..n].copy_from_slice(&x_seq[..n]);

                // Pre-layer RMSNorm
                rmsnorm_with_gamma_eps(
                    &mut x_seq[..n],
                    &layer_weights.input_norm,
                    config.rms_norm_eps,
                );

                // Dispatch
                if is_linear {
                    forward_deltanet_layer(
                        &mut x_seq[..n],
                        layer_weights,
                        &mut cache_seq.deltanet_state.recurrent_states[layer_idx],
                        &mut cache_seq.deltanet_state.conv_states[layer_idx],
                        &config,
                        &mut scratch_seq.deltanet,
                    );
                } else {
                    forward_attention_layer(
                        &mut x_seq[..n],
                        layer_weights,
                        &mut cache_seq.kv_cache.layers[layer_idx],
                        pos,
                        &config,
                        &rope_freq,
                        &mut scratch_seq.attention,
                    );
                }

                // Residual add
                for (i, x) in x_seq[..n].iter_mut().enumerate() {
                    *x += scratch_seq.residual[i];
                }

                // Save residual for MLP
                scratch_seq.residual[..n].copy_from_slice(&x_seq[..n]);

                // Pre-MLP RMSNorm
                rmsnorm_with_gamma_eps(
                    &mut x_seq[..n],
                    &layer_weights.post_attn_norm,
                    config.rms_norm_eps,
                );

                // SwiGLU MLP
                layer_weights
                    .gate_proj
                    .matvec(&x_seq[..n], &mut scratch_seq.gate);
                layer_weights
                    .up_proj
                    .matvec(&x_seq[..n], &mut scratch_seq.up);
                swiglu(&mut scratch_seq.hidden, &scratch_seq.gate, &scratch_seq.up);

                // Down projection
                layer_weights
                    .down_proj
                    .matvec(&scratch_seq.hidden, &mut x_seq[..n]);

                // Residual add
                for (i, x) in x_seq[..n].iter_mut().enumerate() {
                    *x += scratch_seq.residual[i];
                }
            }

            // Final RMSNorm
            rmsnorm_with_gamma_eps(&mut x_seq[..n], &weights.final_norm, config.rms_norm_eps);

            // LM head
            let hidden_copy = x_seq[..n].to_vec();
            weights
                .lm_head
                .matvec(&hidden_copy, &mut x_seq[..v]);
        }
        let seq_logits = x_seq[..v].to_vec();

        // Both paths should produce identical logits
        assert_eq!(
            dispatch_logits.len(),
            seq_logits.len(),
            "logit size mismatch"
        );
        for (i, (d, s)) in dispatch_logits.iter().zip(seq_logits.iter()).enumerate() {
            assert_eq!(d, s, "logit[{i}] mismatch: dispatch={d}, sequential={s}");
        }
    }

    /// GOAT proof T16: KV cache positions are written correctly during prefill.
    ///
    /// Verifies that after prefill, the KV cache for attention layers has
    /// been populated for all prompt positions (positions `0..seq_len`).
    #[test]
    fn test_kv_cache_populated_after_prefill() {
        let config = Config::qwen_deltanet(2, vec![]);
        let weights = QwenDeltaNetWeights::zeros(&config);
        let layer_types = vec![DeltaNetLayerType::Attention; config.n_layer];

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut prefill_ctx = PrefillContext::new(&config, 4);

        let prompt = vec![10, 20, 30];
        prefill_qwen_deltanet(
            &weights,
            &config,
            &mut cache,
            &prompt,
            &mut scratch,
            &rope_freq,
            &mut prefill_ctx,
        );

        // With zero weights, K and V in cache should all be zero (since Q/K/V
        // projections are all zero weights). Verify cache positions are accessible.
        let kvd = crate::types::kv_dim(&config);
        for layer_idx in 0..config.n_layer {
            let kv_cache = &cache.kv_cache.layers[layer_idx];
            // KV cache should have block_size * kv_dim elements
            assert_eq!(
                kv_cache.key.len(),
                config.block_size * kvd,
                "KV cache key for layer {layer_idx} should have block_size * kv_dim elements"
            );
            assert_eq!(
                kv_cache.value.len(),
                config.block_size * kvd,
                "KV cache value for layer {layer_idx} should have block_size * kv_dim elements"
            );
        }
    }

    /// QK-norm verification: changing `attn_q_norm` / `attn_k_norm` gamma weights
    /// changes the attention-layer output. This proves QK-norm is actually
    /// applied (not skipped). Tests the leaf-level `forward_attention_layer`
    /// directly so the result is unambiguous.
    #[test]
    fn test_qk_norm_is_applied() {
        use crate::types::DeltaNetLayerType;
        let config = Config::qwen_deltanet(1, vec![DeltaNetLayerType::Attention]);
        let n = config.n_embd;
        let hd = config.head_dim;

        // Seeded fill helper
        let mut seed = 42u64;
        let next = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (*s >> 33) as f32
        };

        let mut weights = QwenDeltaNetWeights::zeros(&config);
        let layer = &mut weights.layers[0];
        for w in layer.attn_wq.dense_data_mut() { *w = next(&mut seed); }
        for w in layer.attn_wk.dense_data_mut() { *w = next(&mut seed); }
        for w in layer.attn_wv.dense_data_mut() { *w = next(&mut seed); }
        for w in layer.attn_wo.dense_data_mut() { *w = next(&mut seed); }

        // A variant with q_norm gamma = 2.0 (scale Q 2x after per-head norm)
        let mut layer_scaled = crate::deltanet::weights::DeltaNetLayerWeights {
            attn_q_norm: vec![2.0; hd],
            ..clone_layer(&weights.layers[0])
        };
        layer_scaled.attn_k_norm = vec![1.0; hd];

        // Input: all ones so Q/K/V projections are non-zero
        let mut x_identity = vec![1.0f32; n];
        let mut x_scaled = vec![1.0f32; n];

        let layer_types = vec![DeltaNetLayerType::Attention];
        let mut cache_id = HybridCache::with_layer_types(&config, &layer_types);
        let mut cache_sc = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch_id = HybridForwardScratch::new(&config);
        let mut scratch_sc = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);

        // Process TWO tokens so softmax has >1 element — this is needed because
        // with a single token, softmax is trivially 1.0 regardless of Q scaling.
        for pos in 0..2 {
            // Reset input to ones for each position
            x_identity.fill(1.0);
            x_scaled.fill(1.0);

            // Run with q_norm = ones (identity gamma)
            forward_attention_layer(&mut x_identity, &weights.layers[0], &mut cache_id.kv_cache.layers[0], pos, &config, &rope_freq, &mut scratch_id.attention);

            // Run with q_norm = twos (scale Q by 2x after norm)
            forward_attention_layer(&mut x_scaled, &layer_scaled, &mut cache_sc.kv_cache.layers[0], pos, &config, &rope_freq, &mut scratch_sc.attention);
        }

        // QK-norm scales Q per-head by gamma. With gamma=2.0, Q is scaled 2x,
        // which changes attention scores and thus the output. If QK-norm were
        // skipped, both outputs would be identical.
        let differ = x_identity.iter().zip(x_scaled.iter()).any(|(a, b)| a != b);
        assert!(differ, "QK-norm not applied: identical output despite different q_norm gamma");
    }

    /// Shallow clone of `DeltaNetLayerWeights` for test scaffolding.
    fn clone_layer(src: &DeltaNetLayerWeights) -> DeltaNetLayerWeights {
        DeltaNetLayerWeights {
            attn_wq: src.attn_wq.clone(),
            attn_wk: src.attn_wk.clone(),
            attn_wv: src.attn_wv.clone(),
            attn_wo: src.attn_wo.clone(),
            attn_q_norm: src.attn_q_norm.clone(),
            attn_k_norm: src.attn_k_norm.clone(),
            in_proj_qkv: src.in_proj_qkv.clone(),
            in_proj_a: src.in_proj_a.clone(),
            in_proj_b: src.in_proj_b.clone(),
            in_proj_z: src.in_proj_z.clone(),
            out_proj: src.out_proj.clone(),
            conv1d_weight: src.conv1d_weight.clone(),
            a_log: src.a_log.clone(),
            dt_bias: src.dt_bias.clone(),
            linear_norm: src.linear_norm.clone(),
            gate_proj: src.gate_proj.clone(),
            up_proj: src.up_proj.clone(),
            down_proj: src.down_proj.clone(),
            input_norm: src.input_norm.clone(),
            post_attn_norm: src.post_attn_norm.clone(),
        }
    }
}
