//! Gemma 2 forward family — decode, trace, f16, block-causal variants.
//!
//! All variants share the same architecture: tied embeddings (scaled by
//! `sqrt(n_embd`)), `GeGLU` MLP, RMSNorm-with-gamma pre/post norms, attention
//! logit softcapping, and final logit softcapping. They differ in:
//!
//! - `forward_gemma2`         -- single-token causal decode (f32 weights)
//! - `forward_gemma2_f16`     -- single-token causal decode (f16 weights)
//! - `forward_gemma2_trace`   -- debug variant capturing intermediate tensors
//! - `forward_gemma2_block_causal` -- multi-token D2F block-causal forward
//! - `generate_gemma2`/`_f16` -- autoregressive generation wrappers

use super::*;
use crate::gemma_layer::{GemmaTransformerWeights, GemmaTransformerWeightsF16};

/// Gemma 2 forward pass: `GeGLU` MLP, `RoPE`, `RMSNorm` with gamma, post-norm, tied embeddings.
///
/// Key differences from `forward_base`:
/// - No wpe (uses `RoPE` for positional encoding)
/// - `GeGLU` MLP (gate/up/down instead of w1/w2)
/// - `RMSNorm` with learnable gamma (offset already baked in during weight load)
/// - Post-attention and post-MLP `RMSNorm` (Gemma 2 specific)
/// - Tied `lm_head` (reuses wte transposed via dot product)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_gemma2<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim; // Gemma 2: 8*256=2048 != n_embd=2304
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] * sqrt(n_embd) (Gemma 2 scales embeddings)
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    unsafe {
        load_embed_scale(&mut ctx.x, &weights.wte, tok_off, n, embed_scale);
    }

    // 2. Layer loop + output head (shared with `forward_gemma2_with_embedding`).
    forward_gemma2_layers(
        ctx,
        weights,
        cache,
        pos,
        config,
        n,
        hd,
        q_dim,
        kvd,
        n_kv,
        &mut NoHook,
        &mut NoLora,
    );

    &mut ctx.logits
}

/// Gemma 2 forward pass with a **caller-provided input embedding** (Plan 313 T0.1).
///
/// Identical to [`forward_gemma2`] in every respect except step 1: instead of
/// looking up `wte[token] * sqrt(n_embd)`, the caller supplies a precomputed
/// `embed` slice of length `n_embd`. This is the input the model sees.
///
/// The slice is scaled by `sqrt(n_embd)` to match Gemma 2's embedding scaling —
/// callers should pass the **un-scaled** vector (e.g. the probability-weighted
/// vocab mixture `ẽ = Σ_v p[v] · wte[v]`, or any other valid `n_embd` vector).
///
/// # Contract
///
/// - `embed.len()` MUST equal `config.n_embd`.
/// - The slice contents are interpreted as a hidden-state-space vector (not a
///   token distribution). Soft-embedding callers compute it as
///   `matmul(wte^T, probs)` where `probs` is the per-vocab probability.
/// - All downstream behavior (`RoPE`, KV cache update, residual adds, logits)
///   is identical to the concrete-token path.
///
/// # Use cases
///
/// - **`SwiR` Switch-Thinking** (Plan 275 / riir-ai Plan 313): Latent-mode
///   decode feeds the soft embedding `ẽ_t = Σ_v p_t[v] · e(v)` here instead
///   of a concrete token id.
/// - **LCLM soft-token injection** (Plan 264): compressed-segment soft
///   embeddings can be replayed through this path at inference time.
/// - **Any continuous-input decode** where the host has computed a custom
///   input vector in `n_embd` space.
///
/// # Equivalence with `forward_gemma2`
///
/// Calling `forward_gemma2_with_embedding(ctx, w, cache, &wte[token*n..(token+1)*n], pos, cfg)`
/// produces bit-identical logits to `forward_gemma2(ctx, w, cache, token, pos, cfg)`
/// (modulo FMA reordering from the different inner-loop structure of
/// `load_embed_scale` vs the direct copy here). The unit test in
/// `tests/gemma2_forward_with_embedding.rs` verifies this on the BOS token.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_gemma2_with_embedding<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    embed: &[f32],
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    debug_assert_eq!(embed.len(), n, "embed slice must have length n_embd");

    // 1. Embedding: x = embed * sqrt(n_embd) (caller provides un-scaled vector)
    let embed_scale = (n as f32).sqrt();
    unsafe {
        let mut i = 0;
        let chunk_end = n & !3;
        while i < chunk_end {
            *ctx.x.get_unchecked_mut(i) = *embed.get_unchecked(i) * embed_scale;
            *ctx.x.get_unchecked_mut(i + 1) = *embed.get_unchecked(i + 1) * embed_scale;
            *ctx.x.get_unchecked_mut(i + 2) = *embed.get_unchecked(i + 2) * embed_scale;
            *ctx.x.get_unchecked_mut(i + 3) = *embed.get_unchecked(i + 3) * embed_scale;
            i += 4;
        }
        while i < n {
            *ctx.x.get_unchecked_mut(i) = *embed.get_unchecked(i) * embed_scale;
            i += 1;
        }
    }

    // 2. Layer loop — identical to forward_gemma2 from here on.
    forward_gemma2_layers(
        ctx,
        weights,
        cache,
        pos,
        config,
        n,
        hd,
        q_dim,
        kvd,
        n_kv,
        &mut NoHook,
        &mut NoLora,
    );

    &mut ctx.logits
}

/// Gemma 2 single-token causal decode with `LoRA` deltas applied (Plan 410 Phase 1).
///
/// Identical to [`forward_gemma2`] except the 7 matmul insertion points per
/// layer receive `(α/r)·B(Ax)` `LoRA` deltas from `lora`. The base weights are
/// **frozen** — only the `LoRA` adapters are applied. When `lora` is
/// [`GemmaLora::empty`](crate::transformer::gemma2_lora::GemmaLora::empty),
/// the output is bit-identical to `forward_gemma2` (the G1 no-regression gate).
///
/// Requires the `gemma_lora` feature.
#[cfg(feature = "gemma_lora")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_gemma2_with_lora<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: &mut crate::transformer::gemma2_lora::GemmaLora,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // Reset layer cursor at the start of each forward pass.
    lora.reset();

    // 1. Embedding: x = wte[token] * sqrt(n_embd)
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    unsafe {
        load_embed_scale(&mut ctx.x, &weights.wte, tok_off, n, embed_scale);
    }

    // 2. Layer loop + output head with LoRA deltas.
    forward_gemma2_layers(
        ctx,
        weights,
        cache,
        pos,
        config,
        n,
        hd,
        q_dim,
        kvd,
        n_kv,
        &mut NoHook,
        lora,
    );

    &mut ctx.logits
}

/// Gemma 2 forward with **both** a caller-provided input embedding and `LoRA`
/// deltas (Plan 330 T1.1).
///
/// This is the composition of [`forward_gemma2_with_embedding`] (continuous
/// input) and [`forward_gemma2_with_lora`] (adapter applied). It exists because
/// latent-CoT training needs *both* at once: a latent thought slot must pass
/// through the same adapter the rest of the sequence does, or the adapter never
/// learns to use it.
///
/// Before this, `forward_gemma2_with_embedding` hardcoded `&mut NoLora`, so
/// continuous inputs silently bypassed the adapter — usable for frozen-base
/// latent decode (`SwiR`), useless for training one.
///
/// # Contract
///
/// Same as [`forward_gemma2_with_embedding`]: `embed.len() == config.n_embd`,
/// and the caller passes the **un-scaled** vector (this fn applies
/// `sqrt(n_embd)`).
///
/// # Equivalence
///
/// - With [`GemmaLora::empty`](crate::transformer::gemma2_lora::GemmaLora::empty),
///   bit-identical to `forward_gemma2_with_embedding` (gate G1-EQ).
/// - With `embed = wte[token]` (un-scaled), matches
///   `forward_gemma2_with_lora(token)` modulo FMA reordering (gate G1-TOK).
///
/// Requires the `gemma_lora` feature.
#[cfg(feature = "gemma_lora")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_gemma2_with_embedding_lora<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    embed: &[f32],
    pos: usize,
    config: &Config,
    lora: &mut crate::transformer::gemma2_lora::GemmaLora,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    debug_assert_eq!(embed.len(), n, "embed slice must have length n_embd");

    // Reset layer cursor at the start of each forward pass.
    lora.reset();

    // 1. Embedding: x = embed * sqrt(n_embd) (caller provides un-scaled vector).
    //    Same 4-wide unrolled scale as `forward_gemma2_with_embedding` — kept
    //    identical so G1-EQ is bit-exact rather than merely close.
    let embed_scale = (n as f32).sqrt();
    unsafe {
        let mut i = 0;
        let chunk_end = n & !3;
        while i < chunk_end {
            *ctx.x.get_unchecked_mut(i) = *embed.get_unchecked(i) * embed_scale;
            *ctx.x.get_unchecked_mut(i + 1) = *embed.get_unchecked(i + 1) * embed_scale;
            *ctx.x.get_unchecked_mut(i + 2) = *embed.get_unchecked(i + 2) * embed_scale;
            *ctx.x.get_unchecked_mut(i + 3) = *embed.get_unchecked(i + 3) * embed_scale;
            i += 4;
        }
        while i < n {
            *ctx.x.get_unchecked_mut(i) = *embed.get_unchecked(i) * embed_scale;
            i += 1;
        }
    }

    // 2. Layer loop + output head with LoRA deltas.
    forward_gemma2_layers(
        ctx,
        weights,
        cache,
        pos,
        config,
        n,
        hd,
        q_dim,
        kvd,
        n_kv,
        &mut NoHook,
        lora,
    );

    &mut ctx.logits
}

/// Post-layer injection hook (Issue 395 DRY refactor).
///
/// Called once per layer, AFTER the MLP residual add and delta-routing,
/// receiving `(layer_idx, &mut residual)` where `residual == &mut ctx.x[..n]`.
///
/// This is the single injection point that lets steering (Plan 391), residual
/// offset (FPCG causal test), and any future per-layer perturbation share the
/// **same compiled code** as the production `forward_gemma2` hot path —
/// eliminating the f32 divergence that arises when the layer loop is
/// copy-pasted into separate functions with different inline / FMA /
/// vectorization context.
///
/// `NoHook` is the zero-overhead no-op impl: `#[inline(always)]` empty body,
/// monomorphized away entirely. Production callers pass `&mut NoHook`.
///
/// **Public since Issue 673 Phase 2** (katgpt-rs / Research 492 — the
/// Recirculation PoC): the cross-step residual-mixture operator needs a
/// capture+mix hook that reads AND writes the residual per layer per token
/// — expressible only through this trait. The production hot path still
/// passes `&mut NoHook` (zero overhead unchanged); external users reach it
/// via `latent_steering_bridge::forward_one_token_hooked`.
pub trait PostLayerHook {
    /// Mutate the post-layer residual in-place. Called after the MLP residual
    /// add and after delta-routing (if enabled).
    fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]);
}

/// Zero-overhead no-op hook. The empty `after_layer` is `#[inline(always)]`,
/// so monomorphization + inlining eliminates the call site entirely.
pub(crate) struct NoHook;
impl PostLayerHook for NoHook {
    #[inline(always)]
    fn after_layer(&mut self, _layer_idx: usize, _residual: &mut [f32]) {}
}

// ─────────────────────────────────────────────────────────────────────────
// LoRA application trait (Plan 410 Phase 1) — mirrors the PostLayerHook
// monomorphization pattern so `NoLora` compiles to zero overhead while
// `GemmaLora` applies real LoRA deltas at the 7 matmul insertion points.
// ─────────────────────────────────────────────────────────────────────────

/// Applies `LoRA` weight-deltas at the 7 matmul insertion points within each
/// Gemma 2 layer: Q, K, V, O projections + gate, up, down MLP projections.
///
/// Each method is called AFTER the corresponding base matmul, receiving the
/// matmul's output (mutable) and input (read-only). The implementation adds
/// `(alpha/rank) × B @ (A @ input)` to the output in place. A no-op impl
/// ([`NoLora`]) monomorphizes to zero overhead.
///
/// Layer tracking: `next_layer` is called at the top of each layer iteration
/// (before any matmul), letting the impl select that layer's adapter set.
/// Issue 741 T10 Phase A: widened `pub` — the relocated riir-train-engine
/// `gemma2_train` calls these methods on `GemmaLora` cross-crate. The zero-overhead
/// `NoLora` impl stays crate-internal (a pub trait may be implemented for
/// crate-private types). D4 drift-row reversible.
pub trait LoraApplier {
    /// `q += ΔW_q @ x_in` — called after `matmul(q, wq, x_in)`.
    fn apply_q(&mut self, q: &mut [f32], x_in: &[f32]);
    /// `k += ΔW_k @ x_in` — called after `matmul(k, wk, x_in)`.
    fn apply_k(&mut self, k: &mut [f32], x_in: &[f32]);
    /// `v += ΔW_v @ x_in` — called after `matmul(v, wv, x_in)`.
    fn apply_v(&mut self, v: &mut [f32], x_in: &[f32]);
    /// `x += ΔW_o @ attn_out` — called after `matmul(x, wo, attn_out)`.
    fn apply_o(&mut self, x: &mut [f32], attn_out: &[f32]);
    /// `gate += ΔW_gate @ x_in` — called after `matmul(gate, w_gate, x_in)`.
    fn apply_gate(&mut self, gate: &mut [f32], x_in: &[f32]);
    /// `up += ΔW_up @ x_in` — called after `matmul(up, w_up, x_in)`.
    fn apply_up(&mut self, up: &mut [f32], x_in: &[f32]);
    /// `x += ΔW_down @ hidden` — called after `matmul(x, w_down, hidden)`.
    fn apply_down(&mut self, x: &mut [f32], hidden: &[f32]);
    /// Select the next layer's adapter set. Called at the top of each layer.
    fn next_layer(&mut self);
}

/// Zero-overhead no-op `LoRA` applier. All methods are `#[inline(always)]`
/// empty bodies — monomorphization + inlining eliminates every call site.
pub struct NoLora;
impl LoraApplier for NoLora {
    #[inline(always)]
    fn apply_q(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn apply_k(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn apply_v(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn apply_o(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn apply_gate(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn apply_up(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn apply_down(&mut self, _: &mut [f32], _: &[f32]) {}
    #[inline(always)]
    fn next_layer(&mut self) {}
}

/// Shared layer-loop + output head for the Gemma 2 forward pass.
///
/// Both [`forward_gemma2`] and [`forward_gemma2_with_embedding`] set up `ctx.x`
/// differently, then call this helper for the identical 26-layer transformer
/// body + final norm + tied `lm_head` + logit softcapping.
///
/// The `hook` parameter (Issue 395) lets steering / residual-offset paths run
/// the SAME compiled code as production, eliminating f32 divergence from
/// copy-pasted layer loops. Production callers pass `&mut NoHook`.
///
/// The `lora` parameter (Plan 410) applies `LoRA` weight-deltas at the 7 matmul
/// insertion points per layer. Production callers pass `&mut NoLora` which
/// monomorphizes to zero overhead.
///
/// Marked `#[inline(always)]` so the caller's embedding-setup path fuses with
/// the first layer's `RMSNorm` in the same code path — no function-call overhead.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_gemma2_layers<H: PostLayerHook + ?Sized, L: LoraApplier>(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    pos: usize,
    config: &Config,
    n: usize,
    hd: usize,
    q_dim: usize,
    kvd: usize,
    n_kv: usize,
    hook: &mut H,
    lora: &mut L,
) {
    // Loop-invariant attention scale (1/sqrt(hd)) and token count — pure
    // functions of loop-invariant params. Hoisted out of the per-layer loop.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual BEFORE normalization (un-normalized x)
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. Pre-attention RMSNorm (gamma already has +1 added during load)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // d. QKV projections — rayon parallel (same input ctx.x, independent outputs)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }
        // LoRA deltas on Q/K/V (Plan 410). No-op for `NoLora`.
        lora.apply_q(&mut ctx.q[..q_dim], x_in);
        lora.apply_k(&mut ctx.k[..kvd], x_in);
        lora.apply_v(&mut ctx.v[..kvd], x_in);

        // e. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e2. [RoVE] Rotate V by R_pos (Plan 557 T3.B, riir-ai commit cd62e6cd6).
        // Uses the SAME rotate-half convention as Q/K above — NOT katgpt-core's
        // adjacent-pair RopeAction (convention mismatch).
        #[cfg(feature = "rotary_value_embedding")]
        crate::rope::apply_rope_values(&mut ctx.v, pos, hd, ctx.rope_freq_table.as_slice());

        // f. Store K,V in per-layer cache
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

        // g. Multi-head attention with GQA + attention logit softcapping
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

        // g2. [RoVE] Inverse-rotate the attention output by R_{-pos}
        // (Plan 557 T3.B, riir-ai commit cd62e6cd6). De-rotates the aggregated
        // output back into the query's local frame.
        #[cfg(feature = "rotary_value_embedding")]
        crate::rope::apply_inverse_rope_output(
            &mut ctx.attn_out[..q_dim],
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // h. Output projection (Wo takes [q_dim] -> [n_embd])
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);
        // LoRA delta on O projection (Plan 410). No-op for `NoLora`.
        lora.apply_o(&mut ctx.x[..n], &ctx.attn_out[..q_dim]);

        // i. Post-attention RMSNorm (Gemma 2: norm BEFORE residual add)
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // Residual add (AFTER post-norm)
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
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        // LoRA deltas on gate/up projections (Plan 410). No-op for `NoLora`.
        lora.apply_gate(&mut ctx.gate[..mlp_hidden], x_in);
        lora.apply_up(&mut ctx.up[..mlp_hidden], x_in);
        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // m. x = down_proj @ hidden
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        // LoRA delta on down projection (Plan 410). No-op for `NoLora`.
        lora.apply_down(&mut ctx.x[..n], &ctx.hidden[..config.mlp_hidden]);

        // o. Post-MLP RMSNorm (Gemma 2: norm BEFORE residual add)
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_mlp_norm,
            config.rms_norm_eps,
        );

        // n. Residual add (AFTER post-norm)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(
            ctx,
            layer_idx,
            n,
            Some((
                &weights.delta_routing_query[layer_idx],
                &weights.delta_routing_norm[layer_idx],
            )),
        );

        // Post-layer injection hook (Issue 395). For `NoHook` this is
        // `#[inline(always)]` empty — eliminated entirely at monomorphization.
        // For steering/offset hooks this is the single injection point that
        // shares the production layer-loop's compiled code.
        hook.after_layer(layer_idx, &mut ctx.x[..n]);
        lora.next_layer();
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm with final_norm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // Tied lm_head: logits = wte @ x
    types::matmul_parallel(&mut ctx.logits, &weights.wte, &ctx.x, config.vocab_size, n);

    // 4. Final logit softcapping (Gemma 2): logits = cap * tanh(logits / cap)
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
}

/// Multi-token block-causal forward pass for Gemma 2 weights (Plan 108 T1).
///
/// Analogous to `forward_block_causal()` but uses `GemmaTransformerWeights`:
/// - Embedding scaling: `wte[token] * sqrt(n_embd)` (no wpe, uses `RoPE`)
/// - `RMSNorm` with gamma + eps (not plain rmsnorm)
/// - Q dim != `n_embd`: `q_dim = n_head * head_dim`
/// - `GeGLU` MLP (gate/up/down projections)
/// - Attention logit softcapping: `softcap * tanh(score / softcap)`
/// - Post-norm: `post_attn_norm`, `post_mlp_norm` (Gemma 2 specific)
/// - Tied `lm_head`: `logits = wte^T @ x`
/// - Final logit softcapping: `cap * tanh(logits / cap)`
///
/// Block-causal masking rules (via `block_causal_t_n`):
/// - Prompt positions attend to all prompt positions
/// - Within-block bidirectional, across-block causal
#[cfg(feature = "dllm")]
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma2_block_causal<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    prompt_len: usize,
    block_size: usize,
    config: &Config,
    all_logits: &'a mut [f32],
) -> &'a mut [f32] {
    let seq_len = tokens.len();
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim; // Gemma 2: q_dim != n_embd
    let kvd = types::kv_dim(config);

    assert!(
        seq_len > 0,
        "block-causal forward requires at least one token"
    );
    assert!(
        seq_len <= config.block_size,
        "seq_len {seq_len} exceeds block_size {}",
        config.block_size
    );
    assert!(
        all_logits.len() >= seq_len * config.vocab_size,
        "all_logits buffer too small"
    );

    // Gemma 2 embedding scaling factor
    let embed_scale = (n as f32).sqrt();

    // 1. Initialize hidden states with scaled embeddings (no positional encoding -- uses RoPE)
    for (p, &token) in tokens.iter().enumerate() {
        let tok_off = token * n;
        unsafe {
            load_embed_scale(
                &mut prefill.hidden[p * n..],
                &weights.wte,
                tok_off,
                n,
                embed_scale,
            );
        }
    }

    // 2. Layer loop
    let scale = 1.0 / (hd as f32).sqrt(); // loop-invariant attention scale
    for (_layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[_layer_idx];

        // Phase A: Compute K/V for ALL positions -> store in cache
        for (p, _token) in tokens.iter().enumerate() {
            // Load hidden state
            ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

            // Pre-attention RMSNorm
            types::rmsnorm_with_gamma_eps(
                &mut ctx.x,
                &layer_weights.input_norm,
                config.rms_norm_eps,
            );

            // K/V projections (Issue 053: threshold-gated parallelism)
            let x_in = &ctx.x[..n];
            let wk = &layer_weights.attn_wk;
            let wv = &layer_weights.attn_wv;
            let k_buf = &mut ctx.k;
            let v_buf = &mut ctx.v;
            if n >= RAYON_QKV_THRESHOLD {
                rayon::scope(|s| {
                    s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                    s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
                });
            } else {
                matmul(k_buf, wk, x_in, kvd, n);
                matmul(v_buf, wv, x_in, kvd, n);
            }

            // Apply RoPE to K only (empty Q slice -- not needed in Phase A)
            crate::rope::apply_rope_with_freq(
                &mut ctx.q[..0],
                &mut ctx.k,
                p,
                hd,
                ctx.rope_freq_table.as_slice(),
            );

            // [RoVE] Rotate V by R_p (Plan 557 T3.B, riir-ai commit cd62e6cd6).
            #[cfg(feature = "rotary_value_embedding")]
            crate::rope::apply_rope_values(&mut ctx.v, p, hd, ctx.rope_freq_table.as_slice());

            // Store K, V in per-layer cache
            let pos_off = p * kvd;
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
        }

        // Phase B+C: Attention + MLP for ALL positions
        for (p, _token) in tokens.iter().enumerate() {
            // Load hidden state
            ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

            // Save residual BEFORE normalization
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

            // Pre-attention RMSNorm
            types::rmsnorm_with_gamma_eps(
                &mut ctx.x,
                &layer_weights.input_norm,
                config.rms_norm_eps,
            );

            // Q projection
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, q_dim, n);

            // Apply RoPE to Q only (empty K slice -- already cached with RoPE)
            crate::rope::apply_rope_with_freq(
                &mut ctx.q,
                &mut ctx.k[..0],
                p,
                hd,
                ctx.rope_freq_table.as_slice(),
            );

            // Block-causal attention boundary
            let t_n = block_causal_t_n(p, prompt_len, block_size, seq_len);
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
                    config.n_kv_head,
                    kvd,
                    hd,
                    t_n,
                    scale,
                    attn_softcap,
                    config.block_size,
                );
            }

            // [RoVE] Inverse-rotate the attention output by R_{-p}
            // (Plan 557 T3.B, riir-ai commit cd62e6cd6).
            #[cfg(feature = "rotary_value_embedding")]
            crate::rope::apply_inverse_rope_output(
                &mut ctx.attn_out[..q_dim],
                p,
                hd,
                ctx.rope_freq_table.as_slice(),
            );

            // Output projection (Wo takes [q_dim] -> [n_embd])
            matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

            // Post-attention RMSNorm (Gemma 2: norm BEFORE residual add)
            types::rmsnorm_with_gamma_eps(
                &mut ctx.x,
                &layer_weights.post_attn_norm,
                config.rms_norm_eps,
            );

            // Residual add (AFTER post-norm)
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
                }
            }

            // --- MLP ---

            // Save residual2
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);

            // Pre-MLP RMSNorm
            types::rmsnorm_with_gamma_eps(
                &mut ctx.x,
                &layer_weights.pre_mlp_norm,
                config.rms_norm_eps,
            );

            // GeGLU (Issue 053: threshold-gated parallelism)
            let x_in = &ctx.x[..n];
            let wg = &layer_weights.gate_proj;
            let wu = &layer_weights.up_proj;
            let gate_buf = &mut ctx.gate;
            let up_buf = &mut ctx.up;
            let mlp_hidden = config.mlp_hidden;
            if n >= RAYON_MLP_THRESHOLD {
                rayon::scope(|s| {
                    s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                    s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
                });
            } else {
                matmul(gate_buf, wg, x_in, mlp_hidden, n);
                matmul(up_buf, wu, x_in, mlp_hidden, n);
            }
            types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

            // Down projection
            types::matmul_parallel(
                &mut ctx.x,
                &layer_weights.down_proj,
                &ctx.hidden,
                n,
                mlp_hidden,
            );

            // Post-MLP RMSNorm (Gemma 2: norm BEFORE residual add)
            types::rmsnorm_with_gamma_eps(
                &mut ctx.x,
                &layer_weights.post_mlp_norm,
                config.rms_norm_eps,
            );

            // Residual add (AFTER post-norm)
            for i in 0..n {
                unsafe {
                    *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
                }
            }

            // Store hidden state for next layer / LM head
            prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
        }
    }

    // Snapshot hidden state (last position, for backward compat)
    let last_p = seq_len - 1;
    ctx.hidden_state[..n].copy_from_slice(&prefill.hidden[last_p * n..(last_p + 1) * n]);

    // 3. LM Head for ALL positions
    for p in 0..seq_len {
        ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);

        // Final RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

        // Tied lm_head: logits = wte^T @ x
        let logits_offset = p * config.vocab_size;
        types::matmul_parallel(
            &mut all_logits[logits_offset..logits_offset + config.vocab_size],
            &weights.wte,
            &ctx.x,
            config.vocab_size,
            n,
        );

        // Final logit softcapping: cap * tanh(logits / cap)
        if config.final_logit_softcapping > 0.0 {
            let cap = config.final_logit_softcapping;
            let inv_cap = 1.0 / cap;
            for i in 0..config.vocab_size {
                unsafe {
                    let idx = logits_offset + i;
                    *all_logits.get_unchecked_mut(idx) =
                        cap * crate::simd::fast_tanh(*all_logits.get_unchecked(idx) * inv_cap);
                }
            }
        }
    }

    // Return last position's logits slice for backward compat
    let last_off = (seq_len - 1) * config.vocab_size;
    &mut all_logits[last_off..last_off + config.vocab_size]
}

/// Intermediate activations from a single Gemma 2 forward pass.
///
/// Used to compare CPU vs GPU outputs at each stage to isolate
/// numerical divergence.
#[derive(Debug)]
pub struct Gemma2ForwardTrace {
    /// Hidden state after embedding lookup (scaled by `sqrt(n_embd`)).
    pub after_embed: Vec<f32>,
    /// Hidden state after each layer (26 entries for Gemma 2 2B).
    pub after_layer: Vec<Vec<f32>>,
    /// Q projection after first layer (pre-RoPE).
    pub q_after_layer0: Vec<f32>,
    /// K projection after first layer (pre-RoPE).
    pub k_after_layer0: Vec<f32>,
    /// V projection after first layer.
    pub v_after_layer0: Vec<f32>,
    /// Q after `RoPE` in first layer.
    pub q_after_rope_layer0: Vec<f32>,
    /// K after `RoPE` in first layer.
    pub k_after_rope_layer0: Vec<f32>,
    /// Attention output after first layer.
    pub attn_out_layer0: Vec<f32>,
    /// Hidden state after final `RMSNorm`.
    pub after_final_norm: Vec<f32>,
    /// Logits after `lm_head` matmul (before softcapping).
    pub logits_before_softcap: Vec<f32>,
    /// Logits after final softcapping.
    pub logits: Vec<f32>,
}

/// Run a single Gemma 2 forward pass, capturing intermediate activations
/// after each layer for CPU/GPU comparison.
///
/// This is significantly slower than `forward_gemma2()` due to extra
/// allocations and copies. Use only for debugging.
pub fn forward_gemma2_trace(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> Gemma2ForwardTrace {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] * sqrt(n_embd)
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i) * embed_scale;
        }
    }
    let after_embed = ctx.x[..n].to_vec();

    let mut after_layer = Vec::with_capacity(config.n_layer);
    let mut q_after_layer0 = Vec::new();
    let mut k_after_layer0 = Vec::new();
    let mut v_after_layer0 = Vec::new();
    let mut q_after_rope_layer0 = Vec::new();
    let mut k_after_rope_layer0 = Vec::new();
    let mut attn_out_layer0 = Vec::new();

    // Loop-invariant attention scale and token count — hoisted out of per-layer loop.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual BEFORE normalization
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. Pre-attention RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // d. QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, q_dim, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Capture Q, K, V after projection (pre-RoPE) for layer 0
        if layer_idx == 0 {
            q_after_layer0 = ctx.q[..q_dim].to_vec();
            k_after_layer0 = ctx.k[..kvd].to_vec();
            v_after_layer0 = ctx.v[..kvd].to_vec();
        }

        // e. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // Capture Q, K after RoPE for layer 0
        if layer_idx == 0 {
            q_after_rope_layer0 = ctx.q[..q_dim].to_vec();
            k_after_rope_layer0 = ctx.k[..kvd].to_vec();
        }

        // f. Store K,V in per-layer cache
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

        // g. Multi-head attention with GQA + attention logit softcapping (Plan 096: parallel heads)
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

        // Capture attn_out for layer 0
        if layer_idx == 0 {
            attn_out_layer0 = ctx.attn_out[..q_dim].to_vec();
        }

        // h. Output projection
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // i. Post-attention RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // Residual add (AFTER post-norm)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // j. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        // k. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.pre_mlp_norm, config.rms_norm_eps);

        // l. GeGLU
        matmul(
            &mut ctx.gate,
            &layer_weights.gate_proj,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        matmul(
            &mut ctx.up,
            &layer_weights.up_proj,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // m. Down projection (Plan 096: parallel for 2304x9216)
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // o. Post-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_mlp_norm,
            config.rms_norm_eps,
        );

        // n. Residual add (AFTER post-norm)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(
            ctx,
            layer_idx,
            n,
            Some((
                &weights.delta_routing_query[layer_idx],
                &weights.delta_routing_norm[layer_idx],
            )),
        );

        // Capture hidden state after this layer
        after_layer.push(ctx.x[..n].to_vec());
    }

    // 3. Final RMSNorm with final_norm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);
    let after_final_norm = ctx.x[..n].to_vec();

    // Tied lm_head: logits = wte @ x (Plan 096: parallel for 256Kx2304)
    types::matmul_parallel(&mut ctx.logits, &weights.wte, &ctx.x, config.vocab_size, n);
    let logits_before_softcap = ctx.logits[..config.vocab_size].to_vec();

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
    let logits = ctx.logits[..config.vocab_size].to_vec();

    Gemma2ForwardTrace {
        after_embed,
        after_layer,
        q_after_layer0,
        k_after_layer0,
        v_after_layer0,
        q_after_rope_layer0,
        k_after_rope_layer0,
        attn_out_layer0,
        after_final_norm,
        logits_before_softcap,
        logits,
    }
}

/// Generate tokens autoregressively using Gemma 2 architecture.
/// Returns prompt tokens followed by generated tokens.
pub fn generate_gemma2(
    weights: &GemmaTransformerWeights,
    config: &Config,
    rng: &mut Rng,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    // Pre-allocate for prompt + worst-case generated tokens so the autoregressive
    // decode loop never reallocates (each push is O(1) amortized → O(1) worst-case).
    let mut tokens: Vec<usize> = Vec::with_capacity(prompt_tokens.len() + max_tokens);
    tokens.extend_from_slice(prompt_tokens);

    // Prefill: process all prompt tokens
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        forward_gemma2(&mut ctx, weights, &mut cache, token, pos, config);
    }

    // Generate: autoregressive decode
    for _ in 0..max_tokens {
        // Temperature scaling + softmax on last forward pass logits
        softmax_scaled(&mut ctx.logits, 1.0 / config.temperature);

        // Reuse pre-allocated CDF buffer -- avoids vocab_size allocation per token
        let next = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        if next == 1 {
            break;
        } // EOS
        tokens.push(next);

        // Forward pass for next position
        let pos = tokens.len() - 1;
        forward_gemma2(&mut ctx, weights, &mut cache, tokens[pos], pos, config);
    }

    tokens.shrink_to_fit();
    tokens
}

// ── f16 Weight Inference (Plan 095) ──────────────────────────

/// Gemma 2 forward pass with f16 weights.
///
/// Identical to [`forward_gemma2`] but uses `matmul_f16` for all projections.
/// Weights stored as `half::f16` halve memory bandwidth per token.
/// Activations remain f32 throughout -- only weights are f16.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_gemma2_f16<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
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
        let layer_cache = &mut cache.layers[layer_idx];

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

        // f. Cache K,V
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

        // g. Multi-head attention + softcapping (Plan 096: parallel heads)
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

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(
            ctx,
            layer_idx,
            n,
            Some((
                &weights.delta_routing_query[layer_idx],
                &weights.delta_routing_norm[layer_idx],
            )),
        );
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

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

/// Generate tokens using f16-weight Gemma 2 inference.
///
/// Same as [`generate_gemma2`] but uses [`forward_gemma2_f16`] for reduced
/// memory bandwidth (~5.2 GB/token vs ~10.4 GB/token).
pub fn generate_gemma2_f16(
    weights: &GemmaTransformerWeightsF16,
    config: &Config,
    rng: &mut Rng,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    // Pre-allocate for prompt + worst-case generated tokens so the autoregressive
    // decode loop never reallocates.
    let mut tokens: Vec<usize> = Vec::with_capacity(prompt_tokens.len() + max_tokens);
    tokens.extend_from_slice(prompt_tokens);

    // Prefill
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        forward_gemma2_f16(&mut ctx, weights, &mut cache, token, pos, config);
    }

    // Generate
    for _ in 0..max_tokens {
        softmax_scaled(&mut ctx.logits, 1.0 / config.temperature);
        // Reuse pre-allocated CDF buffer -- avoids vocab_size allocation per token
        let next = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        if next == 1 {
            break;
        }
        tokens.push(next);
        let pos = tokens.len() - 1;
        forward_gemma2_f16(&mut ctx, weights, &mut cache, tokens[pos], pos, config);
    }

    tokens.shrink_to_fit();
    tokens
}

// ── Attention Capture (Issue 380 Path B) ─────────────────────

/// Per-layer attention features captured at the last token position.
///
/// Computed by [`forward_gemma2_attn_capture`] at caller-specified layers.
/// All three scalars are means across `config.n_head` heads.
#[derive(Clone, Debug)]
pub struct AttnLayerFeatures {
    /// Layer index.
    pub layer_idx: usize,
    /// Mean Shannon entropy of the attention distribution across all heads,
    /// normalized by `ln(t_n)` to the `[0, 1]` range
    /// (0 = fully focused on one token, 1 = uniform attention).
    pub entropy_normalized: f32,
    /// Mean of the per-head maximum attention weight — how concentrated each
    /// head is on its single most-attended token. High = focused.
    pub mean_max_weight: f32,
    /// Mean attention weight on the last (current) token across all heads.
    /// High = the model is attending to its most recent output.
    pub mean_self_weight: f32,
}

/// Gemma 2 forward pass that captures attention pattern features at specified layers.
///
/// This is a **cold-path diagnostic function** (Issue 380 Path B) — it duplicates
/// the layer loop from [`forward_gemma2_layers`] to compute attention entropy at
/// the last token position for each layer in `capture_layers`. The entropy is
/// computed from `softmax(Q_h · K^T)` over all positions `[0..t_n]` for each
/// head `h`, after `RoPE` is applied to Q/K and after K is stored in the cache.
///
/// **Do NOT use in hot paths** — the entropy computation adds `O(n_head * t_n)`
/// work per captured layer, plus a `t_n`-sized allocation. Use
/// [`forward_gemma2`] for production decode.
///
/// The rest of the forward pass is identical to [`forward_gemma2`]: tied
/// embeddings scaled by `sqrt(n_embd)`, `GeGLU` MLP, RMSNorm-with-gamma
/// pre/post norms, attention logit softcapping, and final logit softcapping.
/// No hooks, no `LoRA` — this is a plain forward pass.
///
/// Returns `(&mut ctx.logits, Vec<AttnLayerFeatures>)`. The features vector
/// has one entry per layer in `capture_layers` (in ascending layer order, since
/// the layer loop is sequential). Layers not in `capture_layers` are skipped.
///
/// # Contract
///
/// - `capture_layers` entries must be `< config.n_layer`.
/// - `token < config.vocab_size`.
/// - `pos` is the 0-based position of `token` in the sequence; `t_n = pos + 1`.
/// - The KV cache for positions `[0..pos]` must already be populated (i.e. this
///   function is called after prefill or as part of the decode loop).
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma2_attn_capture<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    capture_layers: &[usize],
) -> (&'a mut [f32], Vec<AttnLayerFeatures>) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim; // Gemma 2: 8*256=2048 != n_embd=2304
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] * sqrt(n_embd) (Gemma 2 scales embeddings)
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    unsafe {
        load_embed_scale(&mut ctx.x, &weights.wte, tok_off, n, embed_scale);
    }

    // Loop-invariant attention scale and token count — hoisted out of per-layer loop.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;
    let softcap = config.attn_logit_softcapping;
    let scale_over_softcap = if softcap > 0.0 { scale / softcap } else { 0.0 };
    let ln_tn = (t_n as f32).ln().max(1e-12);

    let mut features: Vec<AttnLayerFeatures> = Vec::with_capacity(capture_layers.len());

    // 2. Layer loop (inlined from forward_gemma2_layers — no hooks, no LoRA).
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual BEFORE normalization (un-normalized x)
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. Pre-attention RMSNorm (gamma already has +1 added during load)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // d. QKV projections — rayon parallel (same input ctx.x, independent outputs)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }

        // e. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // f. Store K,V in per-layer cache
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

        // ── Attention capture (Issue 380 Path B) ──
        // After RoPE+cache-store, before the fused attention call: compute
        // per-head softmax(Q·K^T) at the last token position to extract
        // entropy / max-weight / self-weight. This duplicates the score
        // computation that `attention_heads_parallel` fuses into value
        // accumulation — we can't reuse its output because it doesn't expose
        // the per-position weights.
        if capture_layers.contains(&layer_idx) {
            let mut entropy_sum = 0.0_f32;
            let mut max_weight_sum = 0.0_f32;
            let mut self_weight_sum = 0.0_f32;

            // Reuse ctx.head_scores as scratch for the current head's scores.
            // Layout: head_scores[h * block_size .. (h+1) * block_size]. We
            // only need one head's worth at a time, so use [0..t_n].
            let scores_buf = &mut ctx.head_scores[..t_n];

            for h in 0..config.n_head {
                let q_off = h * hd;
                let kv_group = h * n_kv / config.n_head;
                let kv_group_off = kv_group * hd;

                // Compute Q·K scores for all positions [0..t_n].
                let mut max_score = f32::NEG_INFINITY;
                for (t, slot) in scores_buf.iter_mut().enumerate().take(t_n) {
                    let k_off = t * kvd + kv_group_off;
                    let dot = crate::simd::simd_dot_f32(
                        &ctx.q[q_off..q_off + hd],
                        &layer_cache.key[k_off..k_off + hd],
                        hd,
                    );
                    let score = if softcap > 0.0 {
                        softcap * crate::simd::fast_tanh(dot * scale_over_softcap)
                    } else {
                        dot * scale
                    };
                    *slot = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Softmax (stable: subtract max).
                let mut sum = 0.0_f32;
                for slot in scores_buf.iter_mut().take(t_n) {
                    let e = (*slot - max_score).exp();
                    *slot = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;

                // Entropy + max weight + self weight (last position).
                let mut entropy = 0.0_f32;
                let mut max_w = 0.0_f32;
                for &e in scores_buf.iter().take(t_n) {
                    let w = e * inv_sum;
                    if w > 1e-12 {
                        entropy -= w * w.ln();
                    }
                    if w > max_w {
                        max_w = w;
                    }
                }
                let self_w = scores_buf[t_n - 1] * inv_sum;

                entropy_sum += entropy / ln_tn; // normalize to [0, 1]
                max_weight_sum += max_w;
                self_weight_sum += self_w;
            }

            let n_head_f = config.n_head as f32;
            features.push(AttnLayerFeatures {
                layer_idx,
                entropy_normalized: entropy_sum / n_head_f,
                mean_max_weight: max_weight_sum / n_head_f,
                mean_self_weight: self_weight_sum / n_head_f,
            });
        }

        // g. Multi-head attention with GQA + attention logit softcapping
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

        // h. Output projection (Wo takes [q_dim] -> [n_embd])
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // i. Post-attention RMSNorm (Gemma 2: norm BEFORE residual add)
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // Residual add (AFTER post-norm)
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
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // m. x = down_proj @ hidden
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // o. Post-MLP RMSNorm (Gemma 2: norm BEFORE residual add)
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_mlp_norm,
            config.rms_norm_eps,
        );

        // n. Residual add (AFTER post-norm)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(
            ctx,
            layer_idx,
            n,
            Some((
                &weights.delta_routing_query[layer_idx],
                &weights.delta_routing_norm[layer_idx],
            )),
        );
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm with final_norm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // Tied lm_head: logits = wte @ x
    types::matmul_parallel(&mut ctx.logits, &weights.wte, &ctx.x, config.vocab_size, n);

    // 4. Final logit softcapping (Gemma 2): logits = cap * tanh(logits / cap)
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

    (&mut ctx.logits, features)
}
