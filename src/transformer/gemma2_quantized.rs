//! Gemma-2 quantized-KV forward — Issue 013 T1/T2's serve path (the
//! model-side half of the fitted-value-table measurement; katgpt-rs Issue
//! 883 P1/P2, Research 587).
//!
//! The [`forward_gemma2_f16`](super::gemma2) layer stack with the KV cache
//! behind the [`QuantizedKVCache`] trait (katgpt-types): stores go through
//! `store_key`/`store_value` and attention reads a **dequant mirror** — a
//! caller-owned flat `[max_seq × kv_dim]` pair kept faithful to "what the
//! cache would serve now" by incremental per-position dequants plus a tile
//! refresh on the backend's tile boundaries (KVarN quantizes a tile when it
//! fills, or at the cache's last position — flipping those rows from exact
//! raw reads to dequantized reads; a mirror that skipped the refresh would
//! serve stale exact values). Everything else — norms, softcapping, tied
//! lm_head, the RoPE call — is byte-for-byte the f16 forward's structure
//! with steps (f) and (g) replaced; the correspondence is pinned by the
//! T1 bin's f16 control arm, which runs the unchanged forward over the
//! same chunks.
//!
//! Generic over the backend, so the P1 decoration
//! (`katgpt_core::fitted_value_table::MeanRemovedValueCache`) and the plain
//! backend share this one loop (the decorator's keys pass through — both
//! arms of T1 quantize K identically).
//!
//! G3 kill switch (bit-identity class): a mirror is not consulted by the
//! arithmetic the f16 forward doesn't have — swap the backend for a full
//! cache and the plain `forward_gemma2_f16` is the path (the bin's control
//! arm).

use super::attend::{attend_row, AttnShape};
use super::{ForwardContext, RAYON_MLP_THRESHOLD, RAYON_QKV_THRESHOLD};
use crate::gemma_layer::GemmaTransformerWeightsF16;
use crate::types::{self, Config};
use katgpt_types::QuantizedKVCache;

/// When the backend's tile boundaries flip rows from raw to dequantized
/// reads (KVarN semantics: a tile quantizes when it fills, or at the
/// cache's last position).
#[derive(Clone, Copy, Debug)]
pub struct MirrorRefresh {
    /// The backend's tile size (tokens per tile).
    pub tile_size: usize,
    /// The cache's `max_seq_len` — its final position quantizes early.
    pub cache_max_seq: usize,
}

/// Caller-owned dequant mirror: the attention-visible image of the
/// quantized cache. One flat buffer pair `[max_seq × kvd]`, plus the
/// per-layer highest dequantized position (`usize::MAX` = nothing yet —
/// the chunk-start sentinel, cleared by [`reset`](Self::reset)) and the
/// last layer that refreshed the buffer.
///
/// ⚠ The refresh is only incremental when the SAME layer refreshes
/// consecutively (`last_layer == layer_idx`): the buffer is shared across
/// layers, so at a layer boundary the rows below `pos` hold the PREVIOUS
/// layer's values and a full `0..=pos` re-dequant is REQUIRED (the
/// layer-major-within-position loop clobbers them every position). The
/// Issue 013 T1 lane ran this forward WITHOUT that guard — every layer
/// but the last attended over a foreign layer's K/V history — so that
/// lane's absolute PPLs carry the artifact (its relative gate deltas
/// compared two arms through the same path and stay apples-to-apples).
/// Found by the Issue 919 T3 G0 control (plain vs RawF32-mirror: 26.2 vs
/// 41,634 ppl — the mirror was not "what the cache would serve now").
/// The production alternative is a per-layer mirror (n_layer × the
/// buffers) which restores true per-layer incrementality.
pub struct QuantizedKvMirror {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    dequant_pos: Vec<usize>,
    last_layer: usize,
}

impl QuantizedKvMirror {
    #[must_use]
    pub fn new(config: &Config, max_seq_len: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![0.0; max_seq_len * kvd],
            value: vec![0.0; max_seq_len * kvd],
            dequant_pos: vec![usize::MAX; config.n_layer],
            last_layer: usize::MAX,
        }
    }

    /// Reset for a new chunk (the backend's own `reset` is the caller's).
    pub fn reset(&mut self) {
        self.dequant_pos.fill(usize::MAX);
        self.last_layer = usize::MAX;
    }
}

/// The `forward_gemma2_f16` stack over a generic quantized KV cache.
///
/// `refresh` names the backend's tile geometry (pass `None` for a backend
/// whose per-position reads never change after their step).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn forward_gemma2_f16_qkv<'a, C: QuantizedKVCache>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut C,
    mirror: &mut QuantizedKvMirror,
    token: usize,
    pos: usize,
    config: &Config,
    refresh: Option<&MirrorRefresh>,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] * sqrt(n_embd) -- wte is f16, convert to f32
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) =
                (*weights.wte.get_unchecked(tok_off + i)).to_f32() * embed_scale;
        }
    }

    // Loop-invariant attention scale and token count — hoisted out of per-layer loop.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. Pre-attention RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // d. QKV projections (Issue 053: serial for small models)
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

        // f'. Quantized store (the mirror is the attention-visible image).
        cache.store_key(layer_idx, pos, &ctx.k[..kvd]);
        cache.store_value(layer_idx, pos, &ctx.v[..kvd]);

        // g'. Mirror refresh: incremental (only the new position) unless a
        // tile just flipped — then re-read the whole flipped tile, because
        // its earlier rows changed from exact raw reads to dequantized
        // reads at this store.
        let last = mirror.dequant_pos[layer_idx];
        let start = if last != usize::MAX
            && last + 1 == pos
            && pos > 0
            && mirror.last_layer == layer_idx
        {
            let flipped = refresh.is_some_and(|r| {
                r.tile_size > 0
                    && (pos % r.tile_size == r.tile_size - 1 || pos == r.cache_max_seq - 1)
            });
            if flipped {
                let ts = refresh.map_or(0, |r| r.tile_size);
                pos + 1 - ts
            } else {
                pos
            }
        } else {
            0
        };
        for t in start..=pos {
            cache.dequantize_key_into(layer_idx, t, &mut mirror.key[t * kvd..(t + 1) * kvd]);
            cache.dequantize_value_into(layer_idx, t, &mut mirror.value[t * kvd..(t + 1) * kvd]);
        }
        mirror.dequant_pos[layer_idx] = pos;
        mirror.last_layer = layer_idx;

        // g''. Multi-head attention + softcapping over the mirror.
        unsafe {
            attend_row(
                ctx,
                &mirror.key,
                &mirror.value,
                layer_idx,
                AttnShape {
                    n_head: config.n_head,
                    n_kv_head: n_kv,
                    kv_dim: kvd,
                    head_dim: hd,
                    t_n,
                    scale,
                    softcap: config.attn_logit_softcapping,
                    block_size: config.block_size,
                },
            );
        }

        // h. Output projection (f16)
        types::matmul_f16(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // i. Post-attention RMSNorm + residual
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

        // j. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        // k. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.pre_mlp_norm, config.rms_norm_eps);

        // l. GeGLU (Issue 053: threshold-gated parallelism)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
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

        // m. Down projection (f16, Plan 096: parallel for 2304x9216)
        types::matmul_f16_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // o. Post-MLP RMSNorm + residual
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

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // Tied lm_head: logits = f16_wte @ x (Plan 096: parallel for 256K×2304)
    types::matmul_f16_parallel(&mut ctx.logits, &weights.wte, &ctx.x, config.vocab_size, n);

    // 4. Final logit softcapping
    if config.final_logit_softcapping > 0.0 {
        let cap = config.final_logit_softcapping;
        let inv_cap = 1.0 / cap;
        for i in 0..config.vocab_size {
            unsafe {
                *ctx.logits.get_unchecked_mut(i) =
                    cap * crate::simd::fast_tanh(*ctx.logits.get_unchecked(i) * inv_cap);
            }
        }
    }

    &mut ctx.logits
}
