//! Gemma 4 unified text model — weights + forward pass (Issue 577).
//!
//! Baseline loader for Plan 318 Phase F: run `gemma-4-12B-it.Q4_K_M.gguf`
//! in riir-engine so the 4B/2B MLA-MoE student has a real 12B baseline to
//! beat on Rust code quality.
//!
//! # Architecture (from `google/gemma-4-12B-it` config.json)
//!
//! - 48 layers alternating: 5 sliding-window + 1 full-attention (repeating)
//! - hidden=3840, intermediate=15360, vocab=262144
//! - sliding layers: 16 Q heads × `head_dim=256`, 8 KV heads (GQA 2:1),
//!   `RoPE` `theta=10_000` (full rotation), `sliding_window=1024`
//! - full-attention layers: 16 Q heads × `head_dim=512`, 1 KV head (MQA),
//!   `RoPE` `theta=1_000_000` with `partial_rotary_factor=0.25` (only first 128
//!   of 512 dims are rotated), no sliding window
//! - `attn_q_norm` + `attn_k_norm` `RMSNorm` applied AFTER projection, BEFORE
//!   `RoPE` (NEW vs Gemma 2)
//! - GELU (tanh approximation) gated FFN — matches existing `gegelu_tanh`
//! - post-norm (Gemma 2-style: norm BEFORE residual add)
//! - `attention_scale = 1.0` (NO `1/sqrt(head_dim)` pre-attn scaling)
//! - `final_logit_softcapping = 30.0` (same as Gemma 2)
//! - tied embeddings (`lm_head` = wte.T)
//! - `max_position_embeddings` = `262_144` (256K native context)
//!
//! # Reference
//!
//! Canonical GGUF tensor names + hparam layout come from llama.cpp's
//! `src/models/src/gemma4.cpp`. The SWA pattern is `set_swa_pattern(6, false)`:
//! layers at index `% 6 == 5` are full-attention, the rest are sliding.

use super::gemma2::{LoraApplier, NoLora};
use super::*;
use crate::rope::RopeFreqTable;
use katgpt_core::types::{self, Config, Gemma4LayerType};

/// Per-layer Gemma 4 weights.
///
/// Q/K/V/O projection sizes vary per layer (see `Gemma4LayerType`): sliding
/// layers use `head_dim=256` + 8 KV heads (`kv_dim=2048`), full-attention layers
/// use `global_head_dim=512` + 1 KV head (`kv_dim=512`). `attn_q_norm` and
/// `attn_k_norm` are NEW vs Gemma 2 — applied to the projected Q/K BEFORE
/// `RoPE`. Their length equals the layer's `head_dim` (256 sliding / 512 full).
#[derive(Clone)]
pub struct Gemma4LayerWeights {
    // Attention projections (sizes vary per layer type — see loader).
    pub attn_wq: Vec<f32>,
    pub attn_wk: Vec<f32>,
    pub attn_wv: Vec<f32>,
    pub attn_wo: Vec<f32>,
    // Q/K RMSNorm gammas (NEW vs Gemma 2). Length = head_dim of THIS layer.
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    // Gated GELU MLP (uniform across layers: gate/up=[mlp_hidden,n_embd],
    // down=[n_embd,mlp_hidden]).
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    // RMSNorm gammas (stored as gamma-1; the +1 offset is applied by the GGUF
    // converter, matching Gemma 2's convention).
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub pre_mlp_norm: Vec<f32>,
    pub post_mlp_norm: Vec<f32>,
    /// Per-layer output scale (Issue 397). Applied AFTER the second residual
    /// to prevent activation explosion. Gemma-4 GGUF stores this as
    /// `blk.N.layer_output_scale.weight` (1-element F32 tensor, ~0.053 for
    /// 12B). Without it, the large norm gammas (±143) cause the hidden state
    /// to explode to Inf → NaN after a few layers. Default 1.0 when absent
    /// (older converters / non-Gemma-4 models).
    pub layer_output_scale: f32,
    /// Cached layer type — drives Q/K size, `RoPE` base, sliding-window masking.
    pub layer_type: Gemma4LayerType,
}

/// All Gemma 4 transformer weights. Tied embeddings (`lm_head` = wte.T).
#[derive(Clone)]
pub struct Gemma4TransformerWeights {
    pub wte: Vec<f32>,        // [vocab_size, n_embd]
    pub final_norm: Vec<f32>, // [n_embd]
    pub layers: Vec<Gemma4LayerWeights>,
}

/// Compute the Q projection width for a given layer type.
#[inline]
pub fn q_dim_for(config: &Config, layer_type: Gemma4LayerType) -> usize {
    match layer_type {
        Gemma4LayerType::Sliding => config.n_head * config.head_dim,
        Gemma4LayerType::Full => config.n_head * config.global_head_dim,
    }
}

/// Compute the KV projection width for a given layer type.
#[inline]
pub fn kv_dim_for(config: &Config, layer_type: Gemma4LayerType) -> usize {
    match layer_type {
        Gemma4LayerType::Sliding => config.n_kv_head * config.head_dim,
        Gemma4LayerType::Full => config.n_global_kv_head * config.global_head_dim,
    }
}

/// Head dimension for a given layer type.
#[inline]
pub fn head_dim_for(config: &Config, layer_type: Gemma4LayerType) -> usize {
    match layer_type {
        Gemma4LayerType::Sliding => config.head_dim,
        Gemma4LayerType::Full => config.global_head_dim,
    }
}

/// Number of KV heads for a given layer type.
#[inline]
pub fn n_kv_head_for(config: &Config, layer_type: Gemma4LayerType) -> usize {
    match layer_type {
        Gemma4LayerType::Sliding => config.n_kv_head,
        Gemma4LayerType::Full => config.n_global_kv_head,
    }
}

/// Effective `RoPE` dimension (number of dims actually rotated) for a layer.
///
/// Sliding layers rotate the full `head_dim`. Full-attention layers rotate
/// only `head_dim * partial_rotary_factor` dims (Gemma-4 = 0.25 → 128 of 512).
#[inline]
pub fn rope_rot_dim_for(config: &Config, layer_type: Gemma4LayerType) -> usize {
    match layer_type {
        Gemma4LayerType::Sliding => config.head_dim,
        Gemma4LayerType::Full => {
            (config.global_head_dim as f32 * config.partial_rotary_factor) as usize
        }
    }
}

/// Apply partial rotary `RoPE` to a Q or K buffer in-place.
///
/// Only the first `n_rot` dims of each head are rotated (paired as
/// `(i, i + n_rot/2)` for `i in 0..n_rot/2`); the remaining
/// `head_dim - n_rot` dims pass through unchanged. This matches the
/// `partial_rotary_factor` convention used by Gemma-3 / Gemma-4 full-attention
/// layers and is the rotate-half variant.
///
/// For `partial_rotary_factor == 1.0` (sliding layers + pre-Gemma-4 behavior)
/// this degenerates to a full `head_dim` rotation.
pub fn apply_partial_rope(
    buf: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_rot: usize,
    freq_table: &[f32],
) {
    debug_assert!(n_rot <= head_dim, "n_rot must not exceed head_dim");
    debug_assert!(
        n_rot.is_multiple_of(2),
        "n_rot must be even (rotate-half convention)"
    );
    debug_assert!(
        freq_table.len() >= n_rot / 2,
        "freq_table too short for n_rot"
    );

    // Fast path: pos=0 is identity (all angles = 0).
    if pos == 0 {
        return;
    }

    let n_heads = buf.len() / head_dim;
    let half_rot = n_rot / 2;
    let pos_f = pos as f32;

    // Pre-compute cos/sin for the rotated half. The unrotated tail is left
    // untouched — the head is laid out as [rot_dim][pass_dim] and only the
    // first rot_dim entries are paired.
    let mut cos_sin = [0.0f32; 512];
    let use_stack = half_rot <= 256;
    let mut heap_buf;
    let cs_buf: &mut [f32] = if use_stack {
        &mut cos_sin[..half_rot * 2]
    } else {
        heap_buf = vec![0.0f32; half_rot * 2];
        &mut heap_buf
    };

    for i in 0..half_rot {
        let angle = pos_f * freq_table[i];
        let (sin_a, cos_a) = angle.sin_cos();
        cs_buf[i] = cos_a;
        cs_buf[half_rot + i] = sin_a;
    }
    let cos_table = &cs_buf[..half_rot];
    let sin_table = &cs_buf[half_rot..];

    for h in 0..n_heads {
        let off = h * head_dim;
        // Rotate pairs within the first n_rot dims: pair (off+i, off+i+half_rot).
        for i in 0..half_rot {
            let x0 = buf[off + i];
            let x1 = buf[off + i + half_rot];
            let c = cos_table[i];
            let s = sin_table[i];
            buf[off + i] = x0 * c - x1 * s;
            buf[off + i + half_rot] = x0 * s + x1 * c;
        }
        // Dims [n_rot..head_dim) pass through unchanged.
    }
}

/// Gemma 4 single-token forward pass (no `LoRA` — delegates to the generic
/// impl with [`NoLora`], which monomorphizes to zero overhead).
///
/// Drives both layer kinds (Sliding / Full) from the same loop using
/// per-layer shape derivation. Returns the logits slice (`&ctx.logits`).
///
/// Caller-owned `Gemma4Scratch` avoids per-call allocation; construct once
/// and reuse across decode steps.
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma4<'a>(
    ctx: &'a mut ForwardContext,
    weights: &Gemma4TransformerWeights,
    cache: &mut MultiLayerKVCache,
    scratch: &mut Gemma4Scratch,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    forward_gemma4_impl(
        ctx,
        weights,
        cache,
        scratch,
        token,
        pos,
        config,
        &mut NoLora,
    )
}

/// Gemma 4 single-token forward pass with `LoRA` adapters applied.
///
/// The `lora` argument provides per-layer low-rank deltas that are added
/// to each of the 7 projection outputs (Q/K/V/O + gate/up/down) after the
/// base matmul. See `gemma4_lora::Gemma4Lora` for the real impl.
#[cfg(feature = "gemma4_lora")]
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma4_with_lora<'a>(
    ctx: &'a mut ForwardContext,
    weights: &Gemma4TransformerWeights,
    cache: &mut MultiLayerKVCache,
    scratch: &mut Gemma4Scratch,
    token: usize,
    pos: usize,
    config: &Config,
    lora: &mut crate::transformer::gemma4_lora::Gemma4Lora,
) -> &'a mut [f32] {
    lora.reset();
    forward_gemma4_impl(ctx, weights, cache, scratch, token, pos, config, lora)
}

/// Generic Gemma 4 forward — shared layer loop + output head.
///
/// `L: LoraApplier` hooks are called after each base matmul. `NoLora`
/// (the default for `forward_gemma4`) monomorphizes to the same machine code
/// as a version with no hooks. The real `LoRA` impl (`Gemma4Lora`) is feature-
/// gated under `gemma4_lora`.
#[allow(clippy::too_many_arguments)]
fn forward_gemma4_impl<'a, L: LoraApplier>(
    ctx: &'a mut ForwardContext,
    weights: &Gemma4TransformerWeights,
    cache: &mut MultiLayerKVCache,
    scratch: &mut Gemma4Scratch,
    token: usize,
    pos: usize,
    config: &Config,
    lora: &mut L,
) -> &'a mut [f32] {
    let n = config.n_embd;

    // 1. Token embedding + sqrt(n_embd) scaling (Gemma family convention).
    // GGUF stores wte as [n_embd, vocab_size] (inner dim first) — same as
    // Gemma 2. The embedding row for `token` is `wte[token*n..(token+1)*n]`.
    let emb_off = token * n;
    debug_assert!(
        emb_off + n <= weights.wte.len(),
        "token {} out of vocab range (wte len {})",
        token,
        weights.wte.len()
    );
    let sqrt_n = (n as f32).sqrt();
    for i in 0..n {
        ctx.x[i] = weights.wte[emb_off + i] * sqrt_n;
    }

    // 2. Layer loop — per-layer type dispatch.
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let sw_cache = cache.sliding_capacity(layer_idx);
        let layer_cache = &mut cache.layers[layer_idx];
        let layer_type = layer_weights.layer_type;
        let hd = head_dim_for(config, layer_type);
        let n_kv = n_kv_head_for(config, layer_type);
        let kvd = kv_dim_for(config, layer_type);
        let q_dim = q_dim_for(config, layer_type);
        let n_head = config.n_head;
        let n_rot = rope_rot_dim_for(config, layer_type);
        let rope_theta = match layer_type {
            Gemma4LayerType::Sliding => config.rope_theta,
            Gemma4LayerType::Full => config.rope_theta_full,
        };

        // a. Save residual.
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. Pre-attention RMSNorm.
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV projections. Outputs land in ctx.q[..q_dim], ctx.k[..kvd], ctx.v[..kvd].
        let x_in = &ctx.x[..n];
        matmul(&mut ctx.q, &layer_weights.attn_wq, x_in, q_dim, n);
        lora.apply_q(&mut ctx.q, x_in);
        matmul(&mut ctx.k, &layer_weights.attn_wk, x_in, kvd, n);
        lora.apply_k(&mut ctx.k, x_in);
        // attention_k_eq_v: V is optional in the GGUF; if absent, V = K. The
        // loader materializes attn_wv as a copy of attn_wk when the tensor is
        // missing, so we always have a real V projection here.
        matmul(&mut ctx.v, &layer_weights.attn_wv, x_in, kvd, n);
        lora.apply_v(&mut ctx.v, x_in);
        // Issue 397: llama.cpp applies RMSNorm (without gamma) to V before
        // storing in cache. This normalizes the V values to unit RMS, which is
        // important for numerical stability when combined with the large norm
        // gammas in Gemma-4.
        {
            let v_slice = &mut ctx.v[..kvd];
            let sum_sq: f32 = v_slice.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / kvd as f32 + config.rms_norm_eps as f32).sqrt();
            for v in v_slice.iter_mut() {
                *v *= inv_rms;
            }
        }

        // d. Q/K RMSNorm (NEW vs Gemma 2). Applied per-head over head_dim.
        //    The norm gamma length == hd (256 sliding / 512 full).
        for h in 0..n_head {
            let off = h * hd;
            types::rmsnorm_with_gamma_eps(
                &mut ctx.q[off..off + hd],
                &layer_weights.attn_q_norm,
                config.rms_norm_eps,
            );
        }
        for h in 0..n_kv {
            let off = h * hd;
            types::rmsnorm_with_gamma_eps(
                &mut ctx.k[off..off + hd],
                &layer_weights.attn_k_norm,
                config.rms_norm_eps,
            );
        }

        // e. Apply partial rotary RoPE to Q and K (V is not rotated).
        //    Sliding: full rotation (n_rot == hd). Full: partial (n_rot = 128 of 512).
        //    Suppress unused-variable warning when rope_theta isn't read directly
        //    (the freq table is built from it once in Gemma4Scratch).
        let _ = rope_theta;
        let freq_table = match layer_type {
            Gemma4LayerType::Sliding => scratch.rope_freq_sliding.as_slice(),
            Gemma4LayerType::Full => scratch.rope_freq_full.as_slice(),
        };
        apply_partial_rope(&mut ctx.q, pos, hd, n_rot, freq_table);
        apply_partial_rope(&mut ctx.k, pos, hd, n_rot, freq_table);

        // f. Store K,V into per-layer cache at the current position. The cache
        //    for this layer was allocated with `kvd` matching THIS layer's
        //    KV width (the loader builds the cache with per-layer kv_dim).
        //
        //    Sliding-bounded layers (Plan 320 D3) use the PLAIN-MODULO ring
        //    convention (katgpt-rs Issue 683; migrated in riir-ai Issue 752):
        //    the layer's buffer is exactly `sw_cache * kvd` floats (1×) and K/V
        //    for logical position `pos` are written at the single slot
        //    `pos % sw_cache` — no mirror copy. Contiguity across a ring wrap
        //    is provided by the read-side two-slice gather below, not by the
        //    layout. (The former 2× mirrored layout was removed upstream.)
        if sw_cache > 0 {
            let ring_off = (pos % sw_cache) * kvd;
            debug_assert!(
                ring_off + kvd <= layer_cache.key.len()
                    && ring_off + kvd <= layer_cache.value.len(),
                "sliding-bounded layer {layer_idx}: ring write [{ring_off}..{}] overflows the \
                 1x ring buffer (key len {}, value len {})",
                ring_off + kvd,
                layer_cache.key.len(),
                layer_cache.value.len()
            );
            layer_cache.key[ring_off..ring_off + kvd].copy_from_slice(&ctx.k[..kvd]);
            layer_cache.value[ring_off..ring_off + kvd].copy_from_slice(&ctx.v[..kvd]);
        } else {
            let off = pos * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(off),
                    kvd,
                );
            }
        }

        // g. Multi-head attention. Gemma 4 uses attention_scale = 1.0 (NO
        //    1/sqrt(head_dim)); softcap is 50.0 (Gemma family default).
        //    For sliding layers, restrict the attention window to the last
        //    `sliding_window` positions.
        let scale = 1.0; // Gemma 4: no pre-attn scaling (per llama.cpp)
        let attn_softcap = config.attn_logit_softcapping;
        ctx.attn_out[..q_dim].fill(0.0);

        let (mut t_start, mut t_n_eff) = match layer_type {
            Gemma4LayerType::Sliding => {
                let sw = config.sliding_window;
                let start = pos.saturating_sub(sw.saturating_sub(1));
                (start, pos - start + 1)
            }
            Gemma4LayerType::Full => (0, pos + 1),
        };
        // Plain-modulo ring clamp (Issue 752): a sliding-bounded layer
        // physically holds only `sw_cache` slots, so the logical window can
        // never exceed the ring capacity. Without the clamp, `t % sw_cache`
        // residues would repeat across a window longer than the ring and the
        // same (stale) physical row would be attended twice.
        if sw_cache > 0 && t_n_eff > sw_cache {
            t_start = pos + 1 - sw_cache;
            t_n_eff = sw_cache;
        }

        // Slice the cache to the window: attention_head indexes
        // `key_cache[t * kv_dim + kv_group_offset]` for t in `0..t_n`, so we
        // pass a sub-slice starting at the window's first row and adjust t_n.
        //
        // Plain-modulo ring read (Issue 752, mirroring the katgpt-rs
        // `MultiLayerKVCache::sliding_capacity` contract): logical position t
        // lives at row `t % sw_cache`. The window `[t_start, pos]` is
        // contiguous in the ring while `t_start % sw_cache <= pos % sw_cache`;
        // a straddling window is gathered as two slices —
        // `[t_start % sw .. sw_cache)` then `[0 .. pos % sw + 1)` — into the
        // caller-owned grow-only gather buffers. The gather preserves logical
        // order, so attention consumes exactly the same rows in exactly the
        // same order as an unbounded-cache run.
        let (k_window, v_window): (&[f32], &[f32]) = if sw_cache > 0 {
            let rs = t_start % sw_cache;
            let re = pos % sw_cache;
            if rs <= re {
                // Contiguous within the ring: rows rs..=re.
                let start = rs * kvd;
                let end = (re + 1) * kvd;
                debug_assert!(
                    end <= layer_cache.key.len() && end <= layer_cache.value.len(),
                    "sliding-bounded layer {layer_idx}: contiguous ring read \
                     [{start}..{end}] overflows the 1x ring buffer"
                );
                (&layer_cache.key[start..end], &layer_cache.value[start..end])
            } else {
                // Window straddles the ring end: [rs..sw_cache) || [0..=re].
                let head_len = (sw_cache - rs) * kvd;
                let tail_len = (re + 1) * kvd;
                let total = head_len + tail_len;
                debug_assert_eq!(
                    total,
                    t_n_eff * kvd,
                    "two-slice gather length must equal the logical window length"
                );
                // Grow-only scratch: resized once to the ring window size on
                // the first straddling read, then reused — zero steady-state
                // allocation on the decode hot path.
                if scratch.ring_gather_k.len() < total {
                    scratch.ring_gather_k.resize(total, 0.0);
                }
                if scratch.ring_gather_v.len() < total {
                    scratch.ring_gather_v.resize(total, 0.0);
                }
                scratch.ring_gather_k[..head_len]
                    .copy_from_slice(&layer_cache.key[rs * kvd..rs * kvd + head_len]);
                scratch.ring_gather_k[head_len..total]
                    .copy_from_slice(&layer_cache.key[..tail_len]);
                scratch.ring_gather_v[..head_len]
                    .copy_from_slice(&layer_cache.value[rs * kvd..rs * kvd + head_len]);
                scratch.ring_gather_v[head_len..total]
                    .copy_from_slice(&layer_cache.value[..tail_len]);
                (
                    &scratch.ring_gather_k[..total],
                    &scratch.ring_gather_v[..total],
                )
            }
        } else {
            // sw_cache == 0 (full-size cache): the window runs from t_start to
            // the buffer end — byte-identical to the pre-Issue-752 path.
            (
                &layer_cache.key[t_start * kvd..],
                &layer_cache.value[t_start * kvd..],
            )
        };
        unsafe {
            attention_heads_parallel(
                &ctx.q,
                k_window,
                v_window,
                &mut ctx.attn_out,
                &mut scratch.window_scores,
                n_head,
                n_kv,
                kvd,
                hd,
                t_n_eff,
                scale,
                attn_softcap,
                config.sliding_window.max(config.block_size),
            );
        }

        // h. Output projection (Wo takes [q_dim] -> [n_embd]).
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);
        lora.apply_o(&mut ctx.x, &ctx.attn_out[..q_dim]);

        // i. Post-attention RMSNorm (Gemma 2-style post-norm: BEFORE residual).
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );
        // Residual add.
        for i in 0..n {
            ctx.x[i] += ctx.xr[i];
        }

        // j. Save residual 2 + pre-MLP RMSNorm.
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.pre_mlp_norm, config.rms_norm_eps);

        // k. Gated GELU FFN. `gegelu_tanh` computes hidden[i] = gelu_tanh(gate[i]) * up[i],
        //    which matches Gemma 4's `gelu_pytorch_tanh` activation.
        let mlp_hidden = config.mlp_hidden;
        matmul(
            &mut ctx.gate,
            &layer_weights.gate_proj,
            &ctx.x,
            mlp_hidden,
            n,
        );
        lora.apply_gate(&mut ctx.gate, &ctx.x[..n]);
        matmul(&mut ctx.up, &layer_weights.up_proj, &ctx.x, mlp_hidden, n);
        lora.apply_up(&mut ctx.up, &ctx.x[..n]);
        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // l. Down projection.
        matmul(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            mlp_hidden,
        );
        lora.apply_down(&mut ctx.x, &ctx.hidden);

        // m. Post-MLP RMSNorm (Gemma 2-style post-norm: BEFORE residual).
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_mlp_norm,
            config.rms_norm_eps,
        );
        // Residual add.
        for i in 0..n {
            ctx.x[i] += ctx.xr2[i];
        }
        // n. Layer output scale (Issue 397). Gemma-4 applies a per-layer
        //    scalar after the residual to prevent activation explosion.
        //    Without this, the large norm gammas cause the hidden state to
        //    grow unboundedly across layers → Inf → NaN.
        if layer_weights.layer_output_scale != 1.0 {
            for i in 0..n {
                ctx.x[i] *= layer_weights.layer_output_scale;
            }
        }
        lora.next_layer();
    }

    // Snapshot hidden state.
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm + tied lm_head (logits = wte @ x).
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);
    types::matmul_parallel(&mut ctx.logits, &weights.wte, &ctx.x, config.vocab_size, n);

    // 4. Final logit softcapping (same as Gemma 2): logits = cap * tanh(logits / cap).
    if config.final_logit_softcapping > 0.0 {
        let cap = config.final_logit_softcapping;
        let inv_cap = 1.0 / cap;
        for i in 0..config.vocab_size {
            ctx.logits[i] = cap * crate::simd::fast_tanh(ctx.logits[i] * inv_cap);
        }
    }

    &mut ctx.logits
}

/// Caller-owned scratch buffers for the Gemma 4 forward pass.
///
/// Construct once via [`Gemma4Scratch::new`] and reuse across decode steps —
/// zero per-call allocation on the steady-state decode path. The struct
/// caches per-layer-type `RoPE` frequency tables (sliding + full use different
/// `rope_theta`) and the per-layer attention-score window buffer.
pub struct Gemma4Scratch {
    /// `RoPE` frequency table for sliding layers (`theta=10_000`, `head_dim=256`).
    /// Length = `head_dim` / 2.
    ///
    /// `pub(crate)` so in-crate consumers can read it (historically the
    /// sibling `gemma4_train` module — Issue 741 T10 Phase A relocated that to
    /// riir-train-engine, which uses the pub `rope_freq_sliding()` accessor).
    pub(crate) rope_freq_sliding: RopeFreqTable,
    /// `RoPE` frequency table for full-attention layers (`theta=1_000_000`,
    /// `global_head_dim=512`). Length = `global_head_dim` / 2; the partial rotary
    /// pass reads only the first `rope_rot_dim_for(Full) / 2` entries.
    pub(crate) rope_freq_full: RopeFreqTable,
    /// Per-head attention score window. Length = `n_head * max_window` where
    /// `max_window = max(sliding_window, block_size)` — sized for the worst
    /// case so a single buffer serves both layer kinds.
    pub(crate) window_scores: Vec<f32>,
    /// Two-slice gather buffer for wrap-straddling ring K reads (Issue 752).
    /// Sliding-bounded layers read a plain-modulo ring; when the attention
    /// window wraps past the ring end, the two slices `[t_start % W .. W)` +
    /// `[0 .. pos % W + 1)` are gathered here in logical order so
    /// `attention_heads_parallel` sees a contiguous window. Grow-only:
    /// starts empty, resizes once to `window * kvd` on the first straddling
    /// read, then is reused — zero steady-state allocation.
    pub(crate) ring_gather_k: Vec<f32>,
    /// Value-side twin of [`Self::ring_gather_k`].
    pub(crate) ring_gather_v: Vec<f32>,
}

impl Gemma4Scratch {
    /// Construct scratch for the given config. Pre-allocates all buffers.
    pub fn new(config: &Config) -> Self {
        let rope_freq_sliding = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let rope_freq_full = RopeFreqTable::new(config.rope_theta_full, config.global_head_dim);
        let max_window = config.sliding_window.max(config.block_size);
        Self {
            rope_freq_sliding,
            rope_freq_full,
            window_scores: vec![0.0; config.n_head * max_window],
            ring_gather_k: Vec::new(),
            ring_gather_v: Vec::new(),
        }
    }

    /// `RoPE` frequency table for sliding layers (theta = `config.rope_theta`).
    pub fn rope_freq_sliding(&self) -> &RopeFreqTable {
        &self.rope_freq_sliding
    }

    /// `RoPE` frequency table for full-attention layers (theta = `config.rope_theta_full`).
    pub fn rope_freq_full(&self) -> &RopeFreqTable {
        &self.rope_freq_full
    }

    /// Per-head attention score window buffer (mutable, for `attention_heads_parallel`).
    pub fn window_scores_mut(&mut self) -> &mut Vec<f32> {
        &mut self.window_scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_gemma4_config() -> Config {
        // Minimal Gemma 4-shaped config for unit tests: 6 layers (1 full at
        // idx 5), tiny dims, tiny sliding window. Same per-layer shape rules
        // as the real 12B — just smaller for fast tests.
        let mut config = Config::gemma4_12b();
        config.vocab_size = 64;
        config.block_size = 32;
        config.n_embd = 32;
        config.n_head = 4;
        config.head_dim = 8; // sliding
        config.global_head_dim = 16; // full
        config.mlp_hidden = 64;
        config.n_layer = 6;
        config.n_kv_head = 2; // sliding GQA
        config.n_global_kv_head = 1; // full MQA
        config.sliding_window = 8;
        config.partial_rotary_factor = 0.5; // rotate 4 of 8 full-attn dims
        config.rope_theta = 10_000.0;
        config.rope_theta_full = 1_000_000.0;
        // Rebuild the per-layer type pattern for the new n_layer.
        config.gemma4_layer_types = (0..config.n_layer)
            .map(|i| {
                if i % 6 == 5 {
                    Gemma4LayerType::Full
                } else {
                    Gemma4LayerType::Sliding
                }
            })
            .collect();
        config
    }

    #[test]
    fn gemma4_12b_config_matches_real_model() {
        let config = Config::gemma4_12b();
        // Sanity: the preset encodes the real config.json values.
        assert_eq!(config.n_layer, 48, "n_layer");
        assert_eq!(config.n_embd, 3840, "hidden_size");
        assert_eq!(config.mlp_hidden, 15360, "intermediate_size");
        assert_eq!(config.vocab_size, 262_144, "vocab_size");
        assert_eq!(config.n_head, 16, "num_attention_heads");
        assert_eq!(config.head_dim, 256, "head_dim (sliding)");
        assert_eq!(config.global_head_dim, 512, "global_head_dim");
        assert_eq!(config.n_kv_head, 8, "num_key_value_heads (sliding GQA)");
        assert_eq!(
            config.n_global_kv_head, 1,
            "num_global_key_value_heads (MQA)"
        );
        assert_eq!(config.sliding_window, 1024, "sliding_window");
        assert_eq!(config.block_size, 262_144, "max_position_embeddings");
        assert_eq!(
            config.final_logit_softcapping, 30.0,
            "final_logit_softcapping"
        );
        assert_eq!(config.rope_theta, 10_000.0, "rope_theta sliding");
        assert_eq!(config.rope_theta_full, 1_000_000.0, "rope_theta full");
        assert_eq!(
            config.partial_rotary_factor, 0.25,
            "partial_rotary_factor full-attn"
        );

        // Layer pattern: 5 sliding + 1 full repeating (8 full-of-6 blocks).
        assert_eq!(
            config.gemma4_layer_types.len(),
            48,
            "layer_types length matches n_layer"
        );
        let full_count = config
            .gemma4_layer_types
            .iter()
            .filter(|t| **t == Gemma4LayerType::Full)
            .count();
        assert_eq!(full_count, 8, "8 full-attention layers (every 6th)");
        for &idx in &[5, 11, 17, 23, 29, 35, 41, 47] {
            assert_eq!(
                config.gemma4_layer_types[idx],
                Gemma4LayerType::Full,
                "layer {idx} should be Full"
            );
        }
    }

    #[test]
    fn per_layer_shape_derivation_matches_config() {
        let config = Config::gemma4_12b();

        // Sliding layers: 16 Q heads × 256 = 4096; 8 KV × 256 = 2048; rotate all 256.
        assert_eq!(q_dim_for(&config, Gemma4LayerType::Sliding), 4096);
        assert_eq!(kv_dim_for(&config, Gemma4LayerType::Sliding), 2048);
        assert_eq!(head_dim_for(&config, Gemma4LayerType::Sliding), 256);
        assert_eq!(n_kv_head_for(&config, Gemma4LayerType::Sliding), 8);
        assert_eq!(rope_rot_dim_for(&config, Gemma4LayerType::Sliding), 256);

        // Full-attention layers: 16 Q × 512 = 8192; 1 KV × 512 = 512; rotate 128 (= 512 × 0.25).
        assert_eq!(q_dim_for(&config, Gemma4LayerType::Full), 8192);
        assert_eq!(kv_dim_for(&config, Gemma4LayerType::Full), 512);
        assert_eq!(head_dim_for(&config, Gemma4LayerType::Full), 512);
        assert_eq!(n_kv_head_for(&config, Gemma4LayerType::Full), 1);
        assert_eq!(rope_rot_dim_for(&config, Gemma4LayerType::Full), 128);
    }

    #[test]
    fn partial_rope_leaves_unrotated_tail_unchanged() {
        // head_dim=8, n_rot=4: first 4 dims rotated in pairs (0,2) (1,3);
        // dims 4..8 are untouched.
        let head_dim = 8usize;
        let n_rot = 4usize;
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let freq = RopeFreqTable::new(10_000.0, n_rot);
        let original_tail = buf[n_rot..].to_vec();

        apply_partial_rope(&mut buf, 7, head_dim, n_rot, freq.as_slice());

        // First n_rot entries: rotated (not equal to original in general).
        assert_ne!(buf[0], 1.0, "first rotated dim should change at pos=7");
        // Tail unchanged.
        assert_eq!(
            &buf[n_rot..],
            &original_tail[..],
            "unrotated tail must pass through unchanged"
        );
    }

    #[test]
    fn partial_rope_at_pos_zero_is_identity() {
        // pos=0 is identity for both partial and full rope.
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = buf.clone();
        let freq = RopeFreqTable::new(10_000.0, 8);
        apply_partial_rope(&mut buf, 0, 8, 8, freq.as_slice());
        assert_eq!(buf, original, "pos=0 must be identity");
    }

    #[test]
    fn forward_gemma4_produces_finite_logits_on_synthetic_weights() {
        // G2 smoke: build a tiny config + non-zero weights and verify the
        // forward pass produces finite logits with the right shape. This
        // catches shape-mismatch panics in the per-layer branching.
        let config = tiny_gemma4_config();
        let n = config.n_embd;
        let n_layer = config.n_layer;
        let layer_types = config.gemma4_layer_types.clone();

        let mut layers = Vec::with_capacity(n_layer);
        for &layer_type in &layer_types {
            let q_dim = q_dim_for(&config, layer_type);
            let kvd = kv_dim_for(&config, layer_type);
            let hd = head_dim_for(&config, layer_type);
            let mlp = config.mlp_hidden;
            // Use tiny non-zero values so RMSNorm doesn't NaN out.
            let fill = |len: usize| vec![0.01; len];
            layers.push(Gemma4LayerWeights {
                attn_wq: fill(n * q_dim),
                attn_wk: fill(n * kvd),
                attn_wv: fill(n * kvd),
                attn_wo: fill(q_dim * n),
                attn_q_norm: fill(hd),
                attn_k_norm: fill(hd),
                gate_proj: fill(n * mlp),
                up_proj: fill(n * mlp),
                down_proj: fill(mlp * n),
                input_norm: fill(n),
                post_attn_norm: fill(n),
                pre_mlp_norm: fill(n),
                post_mlp_norm: fill(n),
                layer_output_scale: 1.0,
                layer_type,
            });
        }
        let weights = Gemma4TransformerWeights {
            wte: vec![0.1; config.vocab_size * n],
            final_norm: vec![0.01; n],
            layers,
        };

        // Build per-layer KV cache: each layer's kv_dim matches its layer type.
        let per_layer_kv_dim: Vec<usize> = layer_types
            .iter()
            .map(|&lt| kv_dim_for(&config, lt))
            .collect();
        let mut cache = MultiLayerKVCache::new_with_per_layer_kv_dim(&config, &per_layer_kv_dim);

        let mut ctx = ForwardContext::new(&config);
        let mut scratch = Gemma4Scratch::new(&config);

        let logits = forward_gemma4(&mut ctx, &weights, &mut cache, &mut scratch, 0, 0, &config);

        assert_eq!(logits.len(), config.vocab_size, "logits shape");
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit[{i}] = {l} is not finite");
        }
    }
}
