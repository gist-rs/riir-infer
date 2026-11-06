//! Gemma-2 f16 forward with an armed KV evictor — riir-infer Issue 1180 T2b
//! (the SinkHeavy protection half of the sink-audit T4 cell).
//!
//! [`forward_gemma2_f16_evictable`] is [`super::forward_gemma2_f16`] with an
//! [`EvictorState`] driving every layer's KV cache. Deliberately a near-copy
//! rather than a parameterization of `forward_gemma2_f16_hk` — the
//! `forward_attention_layer_evictable` precedent (deltanet/forward.rs): the
//! unarmed body stays bit-frozen for every existing caller, and the three
//! deltas are visible at their call sites:
//!
//! 1. K/V write at the EVICTOR'S packed slot (`admit(pos)`), never at the
//!    logical position; RoPE still uses the logical `pos` (compaction moves
//!    storage, never phases);
//! 2. `t_n = slot + 1` per layer (the causal prefix the compacted cache
//!    holds), which equals `pos + 1` whenever no eviction has fired;
//! 3. post-attention `observe` → `maybe_evict` → `tick` on the same
//!    `ctx.head_scores` the attention just wrote — pure side state.
//!
//! So an armed state with `budget ≥` the whole sequence produces
//! `to_bits`-identical logits to the unarmed forward (T3; pinned by the
//! test below on a small synthetic gemma-2).
//!
//! The value-store seam (`ValueStoreHook`, the Issue-013 lossy-V lanes) is
//! NOT carried: those hooks index the cache by logical position and would
//! desynchronize under compaction. This body is the plain `NoVQuant` path.
//!
//! ⚠ Same caveat as the deltanet precedent: below `PARALLEL_HEADS_MIN_SEQ`
//! the attention kernel's sequential fallback reuses `head_scores[..block]`
//! for every head, so heads ≥ 1 observe a stale row for the first 511
//! positions. Under the deferred protocol (eviction at decode start) with a
//! recency window W ≪ prompt length those observations have aged out before
//! any selection reads them.
//!
//! Every gemma-2 layer is full attention in this body (no sliding window —
//! the plain forward has none either), so every layer index carries state.

use super::*;
use crate::deltanet::kv_evict::EvictorState;
use crate::gemma_layer::GemmaTransformerWeightsF16;

/// [`super::forward_gemma2_f16`] with an armed [`EvictorState`] (one
/// attention-layer entry per gemma-2 layer). See the module doc for the
/// three deltas and the T3 bit-identity contract.
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma2_f16_evictable<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    evictor: &mut EvictorState,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] * sqrt(n_embd) (identical).
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) =
                (*weights.wte.get_unchecked(tok_off + i)).to_f32() * embed_scale;
        }
    }

    let scale = 1.0 / (hd as f32).sqrt();

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];
        let ev = evictor.layer(layer_idx);

        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // QKV projections (identical).
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

        // RoPE at the LOGICAL position (never the slot).
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // Δ1 — admit the slot, store K, V there.
        let slot = ev.admit(pos as u64);
        let off = slot * kvd;
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

        // Δ2 — attention over the causal prefix the (compacted) cache holds.
        let shape = super::attend::AttnShape {
            n_head: config.n_head,
            n_kv_head: n_kv,
            kv_dim: kvd,
            head_dim: hd,
            t_n: slot + 1,
            scale,
            softcap: config.attn_logit_softcapping,
            block_size: config.block_size,
        };
        unsafe {
            super::attend::attend_row(ctx, &layer_cache.key, &layer_cache.value, layer_idx, shape);
        }

        // Δ3 — observe the post-softmax rows, then evict to budget when due.
        // Pure side state: neither reads nor writes the activation.
        ev.observe(&ctx.head_scores, config.block_size, pos as u64);
        ev.maybe_evict(layer_cache, kvd, pos as u64);
        ev.tick();

        // Output projection + post-attn norm + residual (identical).
        types::matmul_f16(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);
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

        // MLP (identical).
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.pre_mlp_norm, config.rms_norm_eps);
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
        types::matmul_f16_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
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

    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);
    types::matmul_f16_parallel(&mut ctx.logits, &weights.wte, &ctx.x, config.vocab_size, n);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::kv_evict::{EvictLayerConfig, EvictPolicy};
    use katgpt_core::kv_eviction::differential::DiffEvictConfig;
    use katgpt_core::kv_sink_window::SinkWindowPolicy;

    /// A small synthetic gemma-2 (deterministic pseudo-random f16 weights).
    /// `seq` crosses `PARALLEL_HEADS_MIN_SEQ` (512) so both attention-kernel
    /// arms are exercised.
    fn tiny() -> (Config, GemmaTransformerWeightsF16) {
        let config = Config {
            n_layer: 2,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 16,
            n_embd: 64,
            mlp_hidden: 96,
            vocab_size: 97,
            block_size: 600,
            attn_logit_softcapping: 50.0,
            final_logit_softcapping: 30.0,
            ..Config::default()
        };
        // Deterministic LCG in [-0.25, 0.25) — no RNG dependency.
        let mut state = 0x1180_u64;
        let mut draw = |len: usize| -> Vec<half::f16> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u = (state >> 40) as f32 / (1u64 << 24) as f32;
                    half::f16::from_f32((u - 0.5) * 0.5)
                })
                .collect()
        };
        let (n, kvd, q_dim, h) = (
            config.n_embd,
            config.n_kv_head * config.head_dim,
            config.n_head * config.head_dim,
            config.mlp_hidden,
        );
        let layers = (0..config.n_layer)
            .map(|_| crate::gemma_layer::GemmaLayerWeightsF16 {
                attn_wq: draw(q_dim * n),
                attn_wk: draw(kvd * n),
                attn_wv: draw(kvd * n),
                attn_wo: draw(n * q_dim),
                gate_proj: draw(h * n),
                up_proj: draw(h * n),
                down_proj: draw(n * h),
                input_norm: vec![1.0; n],
                post_attn_norm: vec![1.0; n],
                pre_mlp_norm: vec![1.0; n],
                post_mlp_norm: vec![1.0; n],
            })
            .collect();
        let w = GemmaTransformerWeightsF16 {
            wte: draw(config.vocab_size * n),
            final_norm: vec![1.0; n],
            layers,
            #[cfg(feature = "delta_routing")]
            delta_routing_query: vec![vec![0.0; n]; config.n_layer],
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: vec![vec![1.0; n]; config.n_layer],
        };
        (config, w)
    }

    fn evictor(config: &Config, budget: usize, cadence: usize, n_sink: usize) -> EvictorState {
        let cfg = EvictLayerConfig {
            policy: EvictPolicy::Differential(DiffEvictConfig::max_recent(32)),
            budget,
            cadence,
            defer_until: 0,
        };
        EvictorState::new(
            Some(&cfg),
            SinkWindowPolicy::new(n_sink, usize::MAX),
            &vec![true; config.n_layer],
            config.n_head,
            config.block_size,
        )
    }

    fn token_at(p: usize, vocab: usize) -> usize {
        (p * 31 + 7) % vocab
    }

    /// T3: armed with headroom (budget ≥ sequence) ⇒ `to_bits`-identical
    /// logits to the unarmed forward at EVERY position (both attention-kernel
    /// arms: sequential < 512, parallel ≥ 512).
    #[test]
    fn armed_headroom_is_bit_identical_to_unarmed() {
        let (config, w) = tiny();
        let seq = 560;
        let mut ctx_a = ForwardContext::new(&config);
        let mut ctx_b = ForwardContext::new(&config);
        let mut cache_a = MultiLayerKVCache::new(&config);
        let mut cache_b = MultiLayerKVCache::new(&config);
        let mut ev = evictor(&config, usize::MAX, 1, 4);
        for p in 0..seq {
            let t = token_at(p, config.vocab_size);
            let a: Vec<u32> = forward_gemma2_f16(&mut ctx_a, &w, &mut cache_a, t, p, &config)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let b: Vec<u32> =
                forward_gemma2_f16_evictable(&mut ctx_b, &w, &mut cache_b, t, p, &config, &mut ev)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
            assert_eq!(a, b, "logits diverge at pos {p}");
        }
        assert_eq!(ev.total_stats().evicted_rows, 0);
    }

    /// A tight budget actually evicts, keeps the cache bounded, keeps the
    /// pinned sinks resident, and the logits stay finite.
    #[test]
    fn tight_budget_evicts_and_pins_sinks() {
        let (config, w) = tiny();
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let (budget, cadence) = (40, 8);
        let mut ev = evictor(&config, budget, cadence, 4);
        for p in 0..300 {
            let t = token_at(p, config.vocab_size);
            let out =
                forward_gemma2_f16_evictable(&mut ctx, &w, &mut cache, t, p, &config, &mut ev);
            assert!(out.iter().all(|v| v.is_finite()), "non-finite at {p}");
        }
        let stats = ev.total_stats();
        assert!(stats.evicted_rows > 0, "tight budget never evicted");
        for li in 0..config.n_layer {
            let layer = ev.layer(li);
            assert!(
                layer.live() <= budget + cadence,
                "layer {li} live {}",
                layer.live()
            );
            for s in 0..4u64 {
                assert!(
                    layer.slot_of_logical(s).is_some(),
                    "sink {s} evicted in layer {li}"
                );
            }
        }
    }
}
