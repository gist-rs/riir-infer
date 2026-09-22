//! Ternary-weight forward pass for the hybrid DeltaNet/Attention model
//! (Issue 594).
//!
//! Mirrors [`super::forward::forward_qwen_deltanet`] — the architecture,
//! recurrence, and attention logic are identical. The only difference is that
//! the 12 projection matvecs per layer (plus the global `wte` and `lm_head`)
//! call [`simd_ternary_group_matvec_parallel`] on [`TernaryGroupWeights`] instead of
//! dense `matmul` on `Vec<f32>`.
//!
//! ## What is reused (not duplicated)
//!
//! The `DeltaNet` recurrence helpers ([`gated_deltanet_step_inplace`],
//! [`causal_conv1d_update`], [`expand_heads_into`], [`l2_normalize`],
//! [`softplus`]) and the attention machinery
//! ([`attention_heads_parallel`], [`apply_rope_with_freq`]) are weight-
//! container-agnostic — they operate on projection outputs and dense fields.
//! Only the orchestration that wires projections to recurrence/attention is
//! mirrored here.
//!
//! ## Update history
//!
//! - **Issue 594 (gated attention + partial RoPE)**: the full-attention layers
//!   now wire the Qwen3.5 gated-attention block — `attn_wq` produces
//!   `[q(hd), gate(hd)]` per head (interleaved), the gate is sigmoided and
//!   multiplied onto the attention output before `attn_wo`. Partial `RoPE`
//!   (`rope_dimension_count`) is applied via [`crate::rope::apply_partial_rope_with_freq`].
//!   Both match `llama.cpp::qwen35.cpp::build_layer_attn` bit-for-bit. For
//!   text-only inference, mrope (`rope.dimension_sections = [11, 11, 10, 0]`)
//!   collapses to standard partial `RoPE` (T=H=W=pos), so no section handling is
//!   needed.
//!
//! ## Remaining gate
//!
//! - **G1 gate** (logits vs llama.cpp ±1%) still needs the 7.1 GB model on the
//!   M3 to run. The gated-attention + partial-RoPE wiring is now complete;
//!   the remaining validation is empirical.

use super::forward::{
    AttentionLayerScratch, DeltaNetLayerScratch, HybridCache, HybridForwardScratch,
    causal_conv1d_update, effective_rotary_dim, expand_heads_into, gated_deltanet_step_inplace,
    l2_normalize, softplus,
};
use katgpt_core::TernaryMatvecHook;
use super::ternary_weights::{DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights};
use crate::types::{Config, DeltaNetLayerType, rmsnorm_with_gamma_eps, swiglu};
use katgpt_core::{
    TernaryFfnHook, TernaryGroupWeights, TernaryInputProjHook, simd_ternary_group_matvec_parallel,
};

use crate::deltanet::rotation::{
    TernaryRotationConfig, permute_gdn_v_grouped_inplace, permute_gdn_v_grouped_inverse_inplace,
    rotate_forward_inplace, rotate_inverse_inplace,
};

/// A `BitLinear` projection: `y[..w.rows] = W × x[..w.cols]`.
///
/// Same helper as [`crate::transformer::ternary::bitlinear`] — the kernel
/// asserts exact slice lengths, so slicing to `w.rows` / `w.cols` is the whole
/// job.
///
/// Uses the **row-parallel** kernel: on the real Bonsai-27B shapes this layer
/// actually runs, it is **7.21×** the serial kernel (0.25 → 1.80 tok/s,
/// [Bench 582](../../../.benchmarks/582_ternary_bonsai_decode_throughput_preflight.md)).
/// Bit-identical to serial (rows are independent) and allocation-free, so it
/// weakens neither the G1 nor the G4 gate. Below 256 rows it delegates to
/// serial — which is why `ssm_alpha`/`ssm_beta` (48 rows) are unaffected.
/// Ternary matvec dispatch: CPU SIMD (default) or GPU hook (Issue 599).
///
/// When `hook` is `Some`, dispatches to the GPU implementation (riir-gpu
/// `GemvTernaryCubeCL`). When `None`, uses the CPU SIMD path.
/// The GPU path uploads weights once (cached by pointer identity) and
/// dispatches a `CubeCL` kernel per call. The CPU path uses rayon-parallel
/// SIMD bit-plane extraction.
#[inline(always)]
fn bitlinear(y: &mut [f32], w: &TernaryGroupWeights, x: &[f32], hook: Option<&dyn TernaryMatvecHook>) {
    if let Some(h) = hook {
        h.matvec(w, &x[..w.cols], &mut y[..w.rows]);
    } else {
        simd_ternary_group_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
    }
}

/// Forward pass for a single `DeltaNet` (linear attention) layer with ternary
/// projections.
///
/// Mirrors [`super::forward::forward_deltanet_layer`]; see that function for
/// the recurrence math. Steps 1–3 and 11 use `bitlinear` instead of `matmul`;
/// steps 4–10 (conv1d, gates, recurrence, norm, `SiLU`) are identical and call
/// the same helpers.
#[allow(clippy::too_many_arguments)]
fn forward_deltanet_layer_ternary(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    state: &mut [f32],
    conv_state: &mut [f32],
    config: &Config,
    scratch: &mut DeltaNetLayerScratch,
    hook: Option<&dyn TernaryMatvecHook>,
    input_proj_hook: Option<&dyn TernaryInputProjHook>,
    rotation: Option<&TernaryRotationConfig>,
    rotation_buf: &mut [f32],
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
    let z_dim = v_dim;
    let conv_dim = qkv_dim;

    let x_in = &x[..n_embd];

    // 1-3. QKV + Z + A + B input projections.
    //
    // Issue 980: when the layer carries a folded (rotated) basis, qkv + z
    // consume the ROTATED input — computed ONCE into `rotation_buf` (the
    // fork's per-activation memoization, resolved structurally: these are
    // the only two folded matmuls on this input). a/b are NOT folded (the
    // dense escape set) and consume the primal `x_in` via the
    // GateProjWeights dispatch. The fused input-proj hook (Issue 602) is
    // bypassed on a rotated layer — its contract cannot apply the transform
    // — and per-matvec dispatch keeps the GPU hook for qkv/z.
    let rotated_qkvz = rotation.is_some();
    if rotated_qkvz {
        let rot = rotation.unwrap();
        let signs = rot.signs_for_width(n_embd);
        rotation_buf[..n_embd].copy_from_slice(x_in);
        rotate_forward_inplace(&mut rotation_buf[..n_embd], signs, rot.block_size);
    }
    let (a_raw, b_raw) = scratch.ab_raw.split_at_mut(n_v_heads);
    // Fused dispatch requires (a) no rotation and (b) ternary a/b — the
    // hook's signature passes a/b through to the GPU batch and cannot serve
    // the dense arm or apply the transform.
    let ab_ternary =
        layer.in_proj_a.as_ternary().is_some() && layer.in_proj_b.as_ternary().is_some();
    let use_fused_hook = input_proj_hook.is_some() && ab_ternary && !rotated_qkvz;
    if use_fused_hook {
        let iph = input_proj_hook.unwrap();
        iph.input_projections(
            &layer.in_proj_qkv,
            &layer.in_proj_z,
            layer.in_proj_a.as_ternary().unwrap(),
            layer.in_proj_b.as_ternary().unwrap(),
            x_in,
            &mut scratch.qkv[..qkv_dim],
            &mut scratch.z[..z_dim],
            a_raw,
            b_raw,
        );
    } else {
        // 1. QKV projection (ternary; folded ⇒ rotated input)
        let x_qkvz: &[f32] = if rotated_qkvz {
            &rotation_buf[..n_embd]
        } else {
            x_in
        };
        bitlinear(&mut scratch.qkv, &layer.in_proj_qkv, x_qkvz, hook);

        // 2. Z projection — output gate (ternary; folded ⇒ rotated input)
        bitlinear(&mut scratch.z, &layer.in_proj_z, x_qkvz, hook);

        // 3. Gate projections: a (decay), b (update rate) — never rotated
        // (Issue 980: the dense escape set consumes the primal input).
        layer.in_proj_a.matvec_into(a_raw, x_in);
        layer.in_proj_b.matvec_into(b_raw, x_in);
    }

    // 4. Split QKV
    let (q_slice, rest) = scratch.qkv.split_at_mut(q_dim);
    let (k_slice, v_slice) = rest.split_at_mut(k_dim);

    // 5. Conv1D preprocessing (dense weight — same as dense path)
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

    q_slice.copy_from_slice(&scratch.conv_buf[..q_dim]);
    k_slice.copy_from_slice(&scratch.conv_buf[q_dim..q_dim + k_dim]);
    v_slice.copy_from_slice(&scratch.conv_buf[q_dim + k_dim..]);

    // 6. Compute gates (dense fields: a_log, dt_bias)
    //
    // **Issue 594 (2026-08-10):** the GGUF converter (`qwen.py` line 297)
    // applies `data = -torch.exp(data)` to `A_log` during conversion, so the
    // `ssm_a` tensor in the GGUF ALREADY contains `-exp(A_log_raw)`. Our
    // `a_log` field is a direct copy of `ssm_a` (loader: `blk.N.ssm_a →
    // a_log`), so `layer.a_log[h]` IS `-exp(A_log_raw)` — do NOT re-apply
    // `-exp()`. The old code computed `-exp(-exp(A_log_raw))` (double exp),
    // producing wildly wrong decay rates → flat logits (G1 FAIL).
    //
    // Reference: `prismml-llama.cpp/src/models/qwen35.cpp` line 451:
    //   gate = alpha_softplus * ssm_a;   // ssm_a IS -exp(A_log) from GGUF
    let (beta, decay) = scratch.beta_decay.split_at_mut(n_v_heads);
    for h in 0..n_v_heads {
        beta[h] = crate::simd::fast_sigmoid(b_raw[h]);
        let a_val = a_raw[h] + layer.dt_bias[h];
        let g = layer.a_log[h] * softplus(a_val);
        decay[h] = g.exp();
    }

    // 7-8. Expand K/Q heads + L2-normalize
    let repeat_factor = n_v_heads / n_k_heads;
    expand_heads_into(
        q_slice,
        n_k_heads,
        key_dim,
        repeat_factor,
        &mut scratch.q_normed,
    );
    expand_heads_into(
        k_slice,
        n_k_heads,
        key_dim,
        repeat_factor,
        &mut scratch.k_normed,
    );
    for h in 0..n_v_heads {
        let off = h * key_dim;
        l2_normalize(&mut scratch.q_normed[off..off + key_dim]);
        l2_normalize(&mut scratch.k_normed[off..off + key_dim]);
    }

    // 9. Gated delta rule recurrence
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

    // 10. Gated RMSNorm + SiLU(z) (dense field: linear_norm)
    //
    // The norm is **PER HEAD** (Issue 594, 2026-08-10). `ssm_norm` in the real
    // GGUF is `[head_dim]` = 128, while `recurrent_output` is
    // `n_v_heads * val_dim` = 6144. In the fork, `build_norm_gated` calls
    // `build_norm(input, ssm_norm, ...)` on a tensor shaped
    // `[head_dim, n_heads, ...]`, and ggml norms along `ne[0]` — i.e. one RMS
    // per head, sharing the same 128-element gamma.
    //
    // Normalizing the whole 6144 buffer against a 128-element gamma was both
    // the wrong math AND an out-of-bounds read (`simd_scale_mul_inplace`
    // indexes gamma by x's length). That is what produced NaN logits on the
    // real model; the synthetic fixture hid it by sizing gamma at 6144.
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
    for i in 0..z_dim {
        let z_val = scratch.z[i];
        let sig = crate::simd::fast_sigmoid(z_val);
        scratch.recurrent_output[i] *= z_val * sig;
    }

    // 11. Output projection (ternary)
    //
    // Issue 980: on a folded model with `gdn_v_grouped`, the ssm_out input
    // arrives in tiled [hd, nk, rep] head order and the fold expects the
    // grouped [hd, rep, nk] order — the same feature permute the fork
    // applies (llama-model.cpp:2080) — then the shared sign+FWHT rotation.
    // Both run in place on `recurrent_output`; `rotation_buf` serves as the
    // permute temp. Linear norms above already ran (they are fold-invariant:
    // per-head RMSNorm + SiLU gate commute with the head-order permute).
    //
    // RIIR_B2_PERM / RIIR_B2_SSMOUT_ROT are G1-bisect knobs (default = the
    // fork contract): 0 forces the perm off, 1 on, 2 inverts the perm
    // direction; RIIR_B2_SSMOUT_ROT=0 skips the ssm_out rotation entirely.
    // Read ONCE (process-wide) — never in the per-layer hot path.
    static PERM_MODE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    let perm_mode = *PERM_MODE.get_or_init(|| {
        std::env::var("RIIR_B2_PERM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(-1) // -1 = follow the rotation config
    });
    static SSMOUT_ROT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ssmout_rot = *SSMOUT_ROT.get_or_init(|| {
        std::env::var("RIIR_B2_SSMOUT_ROT").map(|v| v != "0").unwrap_or(true)
    });
    if let Some(rot) = rotation {
        let perm_mode = if perm_mode >= 0 {
            perm_mode
        } else if rot.gdn_v_grouped {
            1
        } else {
            0
        };
        if perm_mode == 1 {
            permute_gdn_v_grouped_inplace(
                &mut scratch.recurrent_output,
                rotation_buf,
                rot.gdn_v_heads,
                rot.gdn_k_groups,
            );
        } else if perm_mode == 2 {
            // Inverted direction: grouped → tiled (the inverse permutation).
            permute_gdn_v_grouped_inverse_inplace(
                &mut scratch.recurrent_output,
                rotation_buf,
                rot.gdn_v_heads,
                rot.gdn_k_groups,
            );
        }
        if ssmout_rot {
            let signs = rot.signs_for_width(v_dim);
            rotate_forward_inplace(&mut scratch.recurrent_output, signs, rot.block_size);
        }
    }
    bitlinear(&mut x[..n_embd], &layer.out_proj, &scratch.recurrent_output, hook);
}

/// Forward pass for a single full-attention layer with ternary projections.
///
/// Mirrors [`super::forward::forward_attention_layer`]; see that function for
/// the attention math. The Q/K/V/O projections use `bitlinear`; QK-norm,
/// `RoPE`, and GQA attention are identical.
///
/// Issue 594: Qwen3.5 gated attention + partial `RoPE` are now wired —
/// `attn_wq` produces `[q(hd), gate(hd)]` per head (`2*q_dim` output), the
/// gate sigmoid-multiplies the attention output before `attn_wo`, and `RoPE`
/// rotates only `effective_rotary_dim(config)` of each head.
#[allow(clippy::too_many_arguments)]
fn forward_attention_layer_ternary(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    cache: &mut crate::transformer::KVCache,
    pos: usize,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    scratch: &mut AttentionLayerScratch,
    hook: Option<&dyn TernaryMatvecHook>,
    rotation: Option<&TernaryRotationConfig>,
    rotation_buf: &mut [f32],
) {
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = effective_rotary_dim(config);

    let x_in = &x[..n_embd];

    // 1. Gated Q projection + K + V (ternary).
    //
    // Issue 594: `attn_wq` is `[2*q_dim, n_embd]` — produces
    // `[q(hd), gate(hd)]` per head interleaved. Matches
    // `qwen35.cpp::build_layer_attn` + HuggingFace `Qwen3NextAttention.forward`.
    //
    // Issue 980: q/k/v are folded — rotate the shared input ONCE (the fork's
    // per-activation memoization; all three consume the same normed input).
    let rotated_qkv = rotation.is_some();
    if rotated_qkv {
        let rot = rotation.unwrap();
        let signs = rot.signs_for_width(n_embd);
        rotation_buf[..n_embd].copy_from_slice(x_in);
        rotate_forward_inplace(&mut rotation_buf[..n_embd], signs, rot.block_size);
    }
    let x_qkv: &[f32] = if rotated_qkv {
        &rotation_buf[..n_embd]
    } else {
        x_in
    };
    bitlinear(&mut scratch.qg_buf, &layer.attn_wq, x_qkv, hook);
    // Split qg_buf into q_buf and gate_buf (per-head interleaved layout).
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        scratch.q_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src..src + hd]);
        scratch.gate_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src + hd..src + 2 * hd]);
    }
    bitlinear(&mut scratch.k_buf, &layer.attn_wk, x_qkv, hook);
    bitlinear(&mut scratch.v_buf, &layer.attn_wv, x_qkv, hook);

    // 2. QK-norm (dense fields: attn_q_norm, attn_k_norm)
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.k_buf[off..off + hd], &layer.attn_k_norm, eps);
    }

    // 3. Apply partial RoPE (Issue 594: mrope collapses to standard partial
    //    RoPE for text-only — all sections share position `pos`).
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
            0.0,
            config.block_size,
        );
    }

    // 6. Output gating (Issue 594): attn_out *= sigmoid(gate).
    for i in 0..q_dim {
        scratch.attn_out[i] *= crate::simd::fast_sigmoid(scratch.gate_buf[i]);
    }

    // 7. Output projection (ternary)
    //
    // Issue 980: `attn_wo` is folded — its input (`attn_out`, consumed only
    // here) rotates IN PLACE before the matmul; the output is primal.
    if let Some(rot) = rotation {
        let q_dim_rot = q_dim;
        let signs = rot.signs_for_width(q_dim_rot);
        rotate_forward_inplace(&mut scratch.attn_out[..q_dim_rot], signs, rot.block_size);
    }
    bitlinear(&mut x[..n_embd], &layer.attn_wo, &scratch.attn_out[..q_dim], hook);
}

/// Ternary-weight forward pass for Qwen3.5 hybrid DeltaNet/Attention (decode,
/// single token).
///
/// Mirrors [`super::forward::forward_qwen_deltanet`]. Writes logits into
/// `x[..vocab_size]` and returns a mutable reference to them.
///
/// The embedding lookup uses [`QwenDeltaNetTernaryWeights::dequant_wte_row_into`]
/// (ternary bit-plane read), and the LM head uses
/// [`simd_ternary_group_matvec_parallel`] directly.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn forward_qwen_deltanet_ternary<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
) -> &'a mut [f32] {
    forward_qwen_deltanet_ternary_with_hook(
        x, weights, cache, token, pos, config, scratch, rope_freq, None, None, None, None,
    )
}

/// Forward pass with optional per-layer residual-stream capture (Issue 594 G1 bisect).
///
/// When `layer_capture` is `Some(buf)`, a copy of the post-layer residual stream
/// (`x[..n]` after the FFN residual add — the same tensor the `PrismML` fork's
/// capture API taps as `l_out`) is written to `buf[layer_idx][..n]` for every
/// layer. This is the diagnostic hook the divergence-from-reference bisect uses
/// to compute per-layer max-abs-diff against the fork's captured activations.
///
/// The buffer must be pre-sized: `buf.len() >= config.n_layer`, each
/// `buf[i].len() >= config.n_embd`. This avoids any allocation in the capture
/// path (the G4 invariant holds).
///
/// When `layer_capture` is `None`, this is bit-identical to
/// [`forward_qwen_deltanet_ternary`] — the public entry point delegates here.
#[allow(clippy::too_many_arguments)]
pub fn forward_qwen_deltanet_ternary_with_capture<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
    layer_capture: Option<&mut [Vec<f32>]>,
) -> &'a mut [f32] {
    forward_qwen_deltanet_ternary_with_hook(
        x, weights, cache, token, pos, config, scratch, rope_freq, layer_capture, None, None, None,
    )
}

/// Forward pass with optional GPU matvec hook (Issue 599).
///
/// This is the real implementation — all other variants delegate here.
/// When `hook` is `Some`, ternary projections dispatch to the GPU
/// (`GemvTernaryCubeCL`) instead of CPU SIMD. Everything else (`DeltaNet`
/// recurrence, attention scoring, `RMSNorm`, `SwiGLU`, `RoPE`) stays on CPU.
///
/// When `hook` is `None`, this is bit-identical to the CPU-only path.
/// G1 tolerance: GPU floating-point reduction order differs from CPU SIMD
/// (~1e-5 relative), which is acceptable for inference (greedy decode / argmax)
/// but NOT for bit-exact regression tests.
#[allow(clippy::too_many_arguments)]
pub fn forward_qwen_deltanet_ternary_with_hook<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
    mut layer_capture: Option<&mut [Vec<f32>]>,
    hook: Option<&dyn TernaryMatvecHook>,
    input_proj_hook: Option<&dyn TernaryInputProjHook>,
    ffn_hook: Option<&dyn TernaryFfnHook>,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let v_dim = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;

    // Rotation state (Issue 980). `None` for pre-rotation files — every
    // branch below then skips and the path is the exact pre-Bonsai-2 one.
    let rotation = weights.rotation.as_ref();
    let max_rot_len = n.max(v_dim).max(config.mlp_hidden);
    debug_assert!(
        scratch.rotation_buf.len() >= max_rot_len,
        "rotation scratch {} < required {max_rot_len} (HybridForwardScratch::new sizes it)",
        scratch.rotation_buf.len()
    );

    // 1. Embedding lookup — ternary wte row dequant
    weights.dequant_wte_row_into(token, &mut x[..n]);
    // Issue 980: a Hadamard-latent embedding table stores rotated rows;
    // restore the primal basis right after the lookup — Hadamard first, sign
    // second (the inverse of the folded-matmul transform; fork
    // build_inp_embd: `h = s * (H z)`).
    if let Some(rot) = rotation
        && rot.inverse_embedding
    {
        let signs = rot.signs_for_width(n);
        rotate_inverse_inplace(&mut x[..n], signs, rot.block_size);
    }

    // 2. Layer loop with hybrid dispatch
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        // a. Save residual. MUST NOT be `attention.q_buf` — the attention layer
        //    overwrites it with the Q projection (Issue 594).
        scratch.residual[..n].copy_from_slice(&x[..n]);

        // b. Pre-attention/input RMSNorm (dense field)
        rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.input_norm, config.rms_norm_eps);

        // c. Layer-specific forward (ternary projections)
        if is_linear {
            forward_deltanet_layer_ternary(
                &mut x[..n],
                layer_weights,
                &mut cache.deltanet_state.recurrent_states[layer_idx],
                &mut cache.deltanet_state.conv_states[layer_idx],
                config,
                &mut scratch.deltanet,
                hook,
                if rotation.is_some() { None } else { input_proj_hook },
                rotation,
                &mut scratch.rotation_buf,
            );
        } else {
            forward_attention_layer_ternary(
                &mut x[..n],
                layer_weights,
                &mut cache.kv_cache.layers[layer_idx],
                pos,
                config,
                rope_freq,
                &mut scratch.attention,
                hook,
                rotation,
                &mut scratch.rotation_buf,
            );
        }

        // d. Residual add
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }

        // e. Save residual for MLP
        scratch.residual[..n].copy_from_slice(&x[..n]);

        // f. Pre-MLP RMSNorm (dense field)
        rmsnorm_with_gamma_eps(
            &mut x[..n],
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // g-i. SwiGLU MLP (ternary projections) + residual add.
        //
        // Issue 601: when ffn_hook is present, the 3 matvecs + SwiGLU are fused
        // into a single GPU command buffer (2.574× faster than CPU parallel).
        // out aliases x (the residual was saved in step e).
        //
        // Issue 980: gate/up/down are all folded — the hook path cannot apply
        // the rotation, so it is bypassed on a rotated model (same posture as
        // the DeltaNet input-proj hook).
        if let Some(fh) = ffn_hook
            && rotation.is_none()
        {
            // Copy x to scratch.hidden_copy (the GPU dispatch uploads from the
            // input slice before writing the output, but Rust's borrow checker
            // can't prove non-aliasing of &x and &mut x).
            scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
            fh.ffn(
                &layer_weights.gate_proj,
                &layer_weights.up_proj,
                &layer_weights.down_proj,
                &scratch.hidden_copy[..n],
                &mut x[..n],
            );
        } else {
            // g. SwiGLU MLP (ternary projections)
            //
            // Issue 980: gate + up share the normed input — rotate ONCE into
            // the rotation scratch (fork memoization), then both matmuls read
            // it.
            let x_ffn: &[f32] = if let Some(rot) = rotation {
                let signs = rot.signs_for_width(n);
                scratch.rotation_buf[..n].copy_from_slice(&x[..n]);
                rotate_forward_inplace(&mut scratch.rotation_buf[..n], signs, rot.block_size);
                &scratch.rotation_buf[..n]
            } else {
                &x[..n]
            };
            bitlinear(&mut scratch.gate, &layer_weights.gate_proj, x_ffn, hook);
            bitlinear(&mut scratch.up, &layer_weights.up_proj, x_ffn, hook);
            swiglu(&mut scratch.hidden, &scratch.gate, &scratch.up);

            // h. Down projection (ternary) — folded: rotate `hidden` in place
            // (it is consumed only here; the output is primal).
            if let Some(rot) = rotation {
                let signs = rot.signs_for_width(config.mlp_hidden);
                rotate_forward_inplace(&mut scratch.hidden, signs, rot.block_size);
            }
            bitlinear(&mut x[..n], &layer_weights.down_proj, &scratch.hidden, hook);
        }

        // i. Residual add
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }

        // Diagnostic tap: snapshot the post-layer residual (matches the fork's
        // `l_out` capture tap). Only copies when a capture buffer was supplied.
        if let Some(buf) = layer_capture.as_deref_mut()
            && let Some(slot) = buf.get_mut(layer_idx)
        {
            slot[..n].copy_from_slice(&x[..n]);
        }
    }

    // 3. Final RMSNorm (dense field)
    rmsnorm_with_gamma_eps(&mut x[..n], &weights.final_norm, config.rms_norm_eps);

    // 4. LM head — ternary matvec (one call, produces all vocab_size logits)
    scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
    // Issue 980: output.weight is folded — rotate the final hidden in place
    // on the copy (the fork builds the head through the same rotated-matmul
    // path; logits come out in the primal basis).
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(n);
        rotate_forward_inplace(&mut scratch.hidden_copy[..n], signs, rot.block_size);
    }
    bitlinear(
        &mut x[..config.vocab_size],
        &weights.lm_head,
        &scratch.hidden_copy[..n],
        hook,
    );

    &mut x[..config.vocab_size]
}

/// Hidden-state extraction variant (Plan 524 Phase 7 / Issue 596).
///
/// Runs the same forward pass as [`forward_qwen_deltanet_ternary`] (through
/// the final `RMSNorm`), then returns `&mut scratch.hidden_copy[..n]` — the
/// post-final-norm residual stream — instead of the logits.
///
/// Used by embedding extractors (Bonsai dense embedder for `DenseEmbedIndex`
/// semantic rerank) where logits are not needed. The hidden state is the
/// pre-LM-head representation: a 12288-D (Bonsai-27B) or 2048-D (0.8B ref)
/// vector that captures the model's understanding of the input.
///
/// **Implementation note:** this delegates to the capture variant (which runs
/// the LM head as a side effect) then returns `scratch.hidden_copy`. The LM
/// head computation is wasted (~1 ternary matvec), but the alternative —
/// duplicating the ~75-line layer loop — violates DRY on a hot path. Embedding
/// extraction is not hot-path (once per chunk at index time), so the waste is
/// acceptable. Returns `&mut scratch.hidden_copy[..n]` (NOT `x[..n]`, which
/// was overwritten by the LM head).
#[allow(clippy::too_many_arguments)]
pub fn forward_qwen_deltanet_ternary_hidden<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
) -> &'a mut [f32] {
    // Run the full forward (produces logits + populates scratch.hidden_copy).
    let _logits = forward_qwen_deltanet_ternary_with_capture(
        x, weights, cache, token, pos, config, scratch, rope_freq, None,
    );
    // Return the hidden state copy (populated before the LM head).
    let n = config.n_embd;
    &mut scratch.hidden_copy[..n]
}

/// Greedy generation with ternary `DeltaNet` weights (decode-only, no prefill).
///
/// This is the minimal generate path for Issue 594 testing — it processes
/// the prompt token-by-token via [`forward_qwen_deltanet_ternary`] (no batched
/// prefill). A ternary prefill path can be added later by mirroring
/// [`super::forward::prefill_qwen_deltanet_into`] if decode-speed prefill is
/// needed for long prompts.
pub fn generate_greedy_qwen_deltanet_ternary(
    weights: &QwenDeltaNetTernaryWeights,
    config: &Config,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let n = config.n_embd;
    let v = config.vocab_size;
    let buf_size = n.max(v);
    let mut x = vec![0.0f32; buf_size];

    let layer_types = if weights.layer_types.is_empty() {
        vec![DeltaNetLayerType::Attention; config.n_layer]
    } else {
        weights.layer_types.clone()
    };
    let mut cache = HybridCache::with_layer_types(config, &layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    // Issue 594: partial RoPE — build the freq table for `rotary_dim`, not
    // `head_dim`. The inv_freq denominator is `rotary_dim` (matches HuggingFace
    // `compute_default_rope_parameters` where `dim = head_dim * partial_rotary_factor`).
    let rotary_dim = effective_rotary_dim(config);
    let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, rotary_dim);

    let mut generated = Vec::with_capacity(max_tokens);
    if prompt_tokens.is_empty() {
        return generated;
    }

    // Process prompt token-by-token (decode path, no batched prefill).
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            token,
            pos,
            config,
            &mut scratch,
            &rope_freq,
        );
    }

    // Argmax the prompt's last logits → first generated token
    let first_token = x[..v]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(i, _)| i);
    generated.push(first_token);

    // Decode loop
    for _ in 1..max_tokens {
        let pos = prompt_tokens.len() + generated.len() - 1;
        let current_token = *generated.last().unwrap();
        forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            current_token,
            pos,
            config,
            &mut scratch,
            &rope_freq,
        );
        let next_token = x[..v]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(i, _)| i);
        generated.push(next_token);
    }

    generated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;
    use crate::types::DeltaNetLayerType::*;

    /// Build a small Config for testing — all projection dims are multiples
    /// of 128 (the ternary group size). Uses 1 head to keep `n_v` = `n_k` so the
    /// dim conventions agree.
    fn small_config(layer_types: Vec<crate::types::DeltaNetLayerType>) -> Config {
        let mut config = Config::qwen_deltanet(2, layer_types);
        // Override to small multiples of 128.
        config.vocab_size = 256;
        config.n_embd = 128;
        config.n_head = 1;
        config.n_kv_head = 1;
        config.head_dim = 128;
        config.mlp_hidden = 256;
        config.deltanet_linear_head_dim = 128;
        config.deltanet_linear_n_heads = 1;
        config.deltanet_linear_n_value_heads = 1;
        config
    }

    /// The forward pass must run without panicking on a hybrid model (1
    /// `DeltaNet` + 1 Attention layer) with properly-sized ternary weights.
    /// This verifies the projection wiring, slice lengths, and cache/scratch
    /// sizing — NOT output correctness (weights are all-zero).
    #[test]
    fn test_forward_hybrid_no_panic() {
        let layer_types = vec![DeltaNet, Attention];
        let config = small_config(layer_types.clone());
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        assert!(
            weights.invariants_hold(),
            "zero weights must pass invariants"
        );

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);

        let buf_size = config.n_embd.max(config.vocab_size);
        let mut x = vec![0.0f32; buf_size];

        // Forward a single token — should not panic.
        let logits = forward_qwen_deltanet_ternary(
            &mut x,
            &weights,
            &mut cache,
            0, // token 0
            0, // pos 0
            &config,
            &mut scratch,
            &rope_freq,
        );

        assert_eq!(logits.len(), config.vocab_size);
    }

    /// All-attention model (no `DeltaNet` layers) — verifies the attention path.
    #[test]
    fn test_forward_all_attention_no_panic() {
        let layer_types = vec![Attention, Attention];
        let config = small_config(layer_types.clone());
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let buf_size = config.n_embd.max(config.vocab_size);
        let mut x = vec![0.0f32; buf_size];

        let logits = forward_qwen_deltanet_ternary(
            &mut x,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            &mut scratch,
            &rope_freq,
        );
        assert_eq!(logits.len(), config.vocab_size);
    }

    /// All-DeltaNet model (no attention layers) — verifies the recurrence path.
    #[test]
    fn test_forward_all_deltanet_no_panic() {
        let layer_types = vec![DeltaNet, DeltaNet];
        let config = small_config(layer_types.clone());
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let buf_size = config.n_embd.max(config.vocab_size);
        let mut x = vec![0.0f32; buf_size];

        let logits = forward_qwen_deltanet_ternary(
            &mut x,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            &mut scratch,
            &rope_freq,
        );
        assert_eq!(logits.len(), config.vocab_size);
    }

    /// Multi-token decode: run 3 tokens through the hybrid model to verify
    /// the cache + state transitions work across tokens.
    #[test]
    fn test_forward_multi_token_no_panic() {
        let layer_types = vec![DeltaNet, Attention];
        let config = small_config(layer_types.clone());
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);

        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let buf_size = config.n_embd.max(config.vocab_size);
        let mut x = vec![0.0f32; buf_size];

        for pos in 0..3 {
            let logits = forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                pos,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
            );
            assert_eq!(logits.len(), config.vocab_size);
        }
    }

    /// Gated attention gate-split wiring test (Issue 594).
    ///
    /// Verifies the per-head interleaved `[q(hd), gate(hd)]` layout in `attn_wq`
    /// is correctly sized at `2*q_dim` rows, and the forward runs without panic.
    #[test]
    fn test_gated_attention_wiring() {
        // Config: 1 attention layer, n_head=2, head_dim=128.
        // q_dim = 256, qg_dim = 512, n_embd = 128.
        let layer_types = vec![Attention];
        let mut config = Config::qwen_deltanet(1, layer_types.clone());
        config.vocab_size = 128;
        config.n_embd = 128;
        config.n_head = 2;
        config.n_kv_head = 2;
        config.head_dim = 128;
        config.mlp_hidden = 256;
        config.block_size = 16;
        config.deltanet_linear_head_dim = 128;
        config.deltanet_linear_n_heads = 1;
        config.deltanet_linear_n_value_heads = 1;

        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let layer = &weights.layers[0];

        // Structural property: attn_wq has 2*q_dim rows (q + gate concatenated per head).
        assert_eq!(
            layer.attn_wq.rows,
            2 * config.n_head * config.head_dim,
            "gated attn_wq must have 2*q_dim rows (q + gate concatenated per head)"
        );
        assert_eq!(
            layer.attn_wq.cols, config.n_embd,
            "attn_wq input dim must be n_embd"
        );

        // Forward should run without panic with the gated weight shape.
        let mut cache = HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let buf_size = config.n_embd.max(config.vocab_size);
        let mut x = vec![0.0f32; buf_size];

        let logits = forward_qwen_deltanet_ternary(
            &mut x,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            &mut scratch,
            &rope_freq,
        );
        assert_eq!(logits.len(), config.vocab_size);

        // Verify scratch sizing: qg_buf is 2*q_dim, gate_buf + q_buf are q_dim.
        assert_eq!(
            scratch.attention.qg_buf.len(),
            2 * config.n_head * config.head_dim
        );
        assert_eq!(
            scratch.attention.q_buf.len(),
            config.n_head * config.head_dim
        );
        assert_eq!(
            scratch.attention.gate_buf.len(),
            config.n_head * config.head_dim
        );
    }

    /// Partial `RoPE` test (Issue 594): `rope_dimension_count < head_dim` must
    /// leave the unrotated tail unchanged.
    ///
    /// With `rotary_dim = 64` and `head_dim = 128`, `RoPE` rotates only
    /// the first 64 dims (32 pairs) and dims [64..128) pass through. We verify
    /// by constructing a Q vector where the tail half is a known constant,
    /// running partial rope, and checking the tail is preserved.
    #[test]
    fn test_partial_rope_preserves_tail() {
        let head_dim = 128;
        let rotary_dim = 64; // Issue 594: partial rope
        let n_head = 2;
        let pos = 5; // nonzero so rotation is non-identity

        let freq = crate::rope::RopeFreqTable::new(10000.0, rotary_dim);

        // Build Q with distinguishable head + tail.
        let mut q = vec![0.0f32; n_head * head_dim];
        let mut k = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            let off = h * head_dim;
            // Rotated region [0..64): arbitrary non-zero
            for i in 0..rotary_dim {
                q[off + i] = (i as f32) * 0.1 + 1.0;
                k[off + i] = (i as f32) * 0.2 + 0.5;
            }
            // Pass-through tail [64..128): known sentinel values
            for i in rotary_dim..head_dim {
                q[off + i] = 42.0 + i as f32;
                k[off + i] = 99.0 + i as f32;
            }
        }

        // Snapshot tail values before rope.
        let q_tail_before: Vec<f32> = q
            .iter()
            .enumerate()
            .filter(|(i, _)| i % head_dim >= rotary_dim)
            .map(|(_, &v)| v)
            .collect();
        let k_tail_before: Vec<f32> = k
            .iter()
            .enumerate()
            .filter(|(i, _)| i % head_dim >= rotary_dim)
            .map(|(_, &v)| v)
            .collect();

        crate::rope::apply_partial_rope_with_freq(
            &mut q,
            &mut k,
            pos,
            head_dim,
            rotary_dim,
            freq.as_slice(),
        );

        // Tail must be UNCHANGED.
        let q_tail_after: Vec<f32> = q
            .iter()
            .enumerate()
            .filter(|(i, _)| i % head_dim >= rotary_dim)
            .map(|(_, &v)| v)
            .collect();
        let k_tail_after: Vec<f32> = k
            .iter()
            .enumerate()
            .filter(|(i, _)| i % head_dim >= rotary_dim)
            .map(|(_, &v)| v)
            .collect();
        assert_eq!(
            q_tail_before, q_tail_after,
            "partial rope must preserve Q tail"
        );
        assert_eq!(
            k_tail_before, k_tail_after,
            "partial rope must preserve K tail"
        );

        // Rotated region must have CHANGED (pos=5, non-zero input → non-trivial rotation).
        let q_head_changed = q
            .iter()
            .enumerate()
            .filter(|(i, _)| i % head_dim < rotary_dim)
            .any(|(i, &v)| (v - ((i % head_dim) as f32 * 0.1 + 1.0)).abs() > 1e-5);
        assert!(q_head_changed, "partial rope must rotate the head region");
    }
}
