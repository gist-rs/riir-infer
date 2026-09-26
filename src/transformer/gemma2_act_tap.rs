//! Gemma-2 linear-INPUT tap forward — Issue 014 T2(b) (the dense-parent PTQ
//! lane's diagonal collector).
//!
//! The `forward_gemma2_f16_tapped` shape (Issue 883's V/K calibration fork of
//! `forward_gemma2_f16`) re-forked for the Issue-886 tap points: the INPUT of
//! every linear, not the K/V outputs. Four distinct input tensors per layer:
//!
//! | tap id (layer l) | linear inputs | width |
//! |---|---|---|
//! | `4l + 0` | `attn_wq` / `attn_wk` / `attn_wv` | `n_embd` |
//! | `4l + 1` | `attn_wo` (the attention output) | `n_head·head_dim` |
//! | `4l + 2` | `gate_proj` / `up_proj` | `n_embd` |
//! | `4l + 3` | `down_proj` (the GeGLU product) | `mlp_hidden` |
//!
//! `wte`/`lm_head` are the tied embedding — not a ternary-refit target in
//! this lane (the PTQ lane ternarizes the 7 per-layer linears only), so no
//! head tap. Structure correspondence to `forward_gemma2_f16` is the 883
//! fork's, verbatim, with the observes inserted at b / post-g / k / post-l.
//!
//! The held-out capture pools ride the same forward: at a deterministic
//! position stride (and never before `min_pos`), each tap vector is cloned
//! into its pool (capped) at the SAME point it is observed — the evaluation
//! set the PTQ arms score `E‖(W−Ŵ)x‖²` against. MEASUREMENT-ONLY: the
//! forward mutates nothing but the caller's accumulator/capture state.

use katgpt_core::act_channel_moments::ActChannelMoments;

use crate::gemma_layer::GemmaTransformerWeightsF16;
use crate::types;
use katgpt_transformer::MultiLayerKVCache;

use super::attention_heads_parallel;
use super::{ForwardContext, RAYON_MLP_THRESHOLD, RAYON_QKV_THRESHOLD};

/// Per-tap held-out activation pools: `pools[tap_id]` is a list of captured
/// vectors (each `width(tap_id)` long), capped by the collector.
#[derive(Default)]
pub struct GemmaActCapture {
    pub pools: Vec<Vec<Vec<f32>>>,
    pub cap_per_tap: usize,
    pub stride: usize,
    pub min_pos: usize,
}

impl GemmaActCapture {
    pub fn new(n_layer: usize, cap_per_tap: usize, stride: usize, min_pos: usize) -> Self {
        Self {
            pools: vec![Vec::new(); n_layer * 4],
            cap_per_tap,
            stride,
            min_pos,
        }
    }
}

/// One pool push (cap-checked, at the tap's own site).
#[inline]
fn push_capture(pool: &mut Vec<Vec<f32>>, x: &[f32], cap: usize) {
    if pool.len() < cap {
        pool.push(x.to_vec());
    }
}

/// The tap forward: `forward_gemma2_f16`'s layer stack (f16 weights, causal,
/// per-token) with the four linear-input observes per layer fed to `moments`
/// and — at the capture stride — cloned into `capture`'s pools.
///
/// Sequences must stay ≤ 4096 positions (the 883 fork's law: this stack does
/// not implement SWA rotation).
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma2_f16_act_tapped(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    moments: &mut ActChannelMoments,
    capture: &mut GemmaActCapture,
    token: usize,
    pos: usize,
    config: &crate::types::Config,
) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let mlp_hidden = config.mlp_hidden;

    // 1. Embedding: x = wte[token] * sqrt(n_embd)  (f16 → f32)
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) =
                (*weights.wte.get_unchecked(tok_off + i)).to_f32() * embed_scale;
        }
    }

    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;
    let do_capture = pos >= capture.min_pos && pos.is_multiple_of(capture.stride);

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];
        let base = layer_idx * 4;

        // a. residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. pre-attn RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // ── TAP 4l+0: the QKV input (post input_norm) ──
        if do_capture {
            push_capture(&mut capture.pools[base], &ctx.x[..n], capture.cap_per_tap);
        }
        moments.observe(base, &ctx.x[..n]);

        // d. QKV projections (rayon at threshold — the production shape)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| types::matmul_f16(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| types::matmul_f16(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| types::matmul_f16(v_buf, wv, x_in, kvd, n));
            });
        } else {
            types::matmul_f16(q_buf, wq, x_in, q_dim, n);
            types::matmul_f16(k_buf, wk, x_in, kvd, n);
            types::matmul_f16(v_buf, wv, x_in, kvd, n);
        }

        // e. RoPE
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // f. cache K,V
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // g. attention + softcapping (parallel heads — the production shape)
        let attn_softcap = config.attn_logit_softcapping;
        ctx.attn_out[..q_dim].fill(0.0);
        unsafe {
            attention_heads_parallel(
                &ctx.q,
                &layer_cache.key,
                &layer_cache.value,
                &mut ctx.attn_out,
                &mut ctx.head_scores,
                config.n_head,
                n_kv,
                kvd,
                hd,
                t_n,
                scale,
                attn_softcap,
                config.block_size,
            );
        }

        // ── TAP 4l+1: the `attn_wo` input (the attention output) ──
        if do_capture {
            push_capture(
                &mut capture.pools[base + 1],
                &ctx.attn_out[..q_dim],
                capture.cap_per_tap,
            );
        }
        moments.observe(base + 1, &ctx.attn_out[..q_dim]);

        // h. output projection
        types::matmul_f16(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // i. post-attn norm + residual
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // j. residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        // k. pre-MLP norm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.pre_mlp_norm, config.rms_norm_eps);

        // ── TAP 4l+2: the gate/up input (post pre_mlp_norm) ──
        if do_capture {
            push_capture(
                &mut capture.pools[base + 2],
                &ctx.x[..n],
                capture.cap_per_tap,
            );
        }
        moments.observe(base + 2, &ctx.x[..n]);

        // l. GeGLU
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| types::matmul_f16(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| types::matmul_f16(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            types::matmul_f16(gate_buf, wg, x_in, mlp_hidden, n);
            types::matmul_f16(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // ── TAP 4l+3: the `down_proj` input (the GeGLU product) ──
        if do_capture {
            push_capture(
                &mut capture.pools[base + 3],
                &ctx.hidden[..mlp_hidden],
                capture.cap_per_tap,
            );
        }
        moments.observe(base + 3, &ctx.hidden[..mlp_hidden]);

        // m. down projection
        types::matmul_f16_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            mlp_hidden,
        );

        // o. post-MLP norm + residual
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_mlp_norm,
            config.rms_norm_eps,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
}
