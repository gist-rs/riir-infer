//! katgpt-rs Issue 933 T4 — fixed-query modelless AttnRes on the Gemma-2 f16
//! forward (the E5 probe; Research 619 MSAR, motivated by the paper's
//! conv-removal ablation: source diversity retains most value under query
//! degradation).
//!
//! The stock [`forward_gemma2_f16`] has NO AttnRes — gemma-2 is the class of
//! checkpoint the probe targets ("without AttnRes weights"). This fork
//! re-implements the f16 layer stack verbatim (the Issue-886 / Issue-920
//! fork precedent: each measurement lane forks the f16 forward at its
//! observe points; the fork is measurement-scoped, the plain path untouched)
//! with the Block-Attention-Residual wiring from
//! [`katgpt_transformer::attn_res`] inserted at the three Kimi-semantics
//! points:
//!
//! | point | site | Kimi analogue |
//! |---|---|---|
//! | token start | `b0 = scaled embedding` pushed | "b0 = token embedding (always included as a source)" |
//! | layer half start ×2 | `x ← mix(x, [b0, b1..bn-1, x])` BEFORE each residual save | `hidden = apply_attn_res(hidden, ...)` before attention and before MLP |
//! | block boundary | `push(x)` after the block's last residual add | `block_residual.push(hidden)` |
//!
//! The mixing weights come from
//! [`apply_attn_res_depth_profile`](katgpt_transformer::attn_res::apply_attn_res_depth_profile):
//! `score[i] = -λ·age[i]` — a deterministic exponential-depth profile
//! replacing the learned `score_weight` query that gemma-2 does not carry.
//! `λ = f32::INFINITY` is the DISARMED posture: the mix is a byte-copy, the
//! whole forward is bit-identical to the stock f16 path (pinned by test —
//! a wiring drift indicts the instrument).
//!
//! The depth axis is PER TOKEN: `b0..bn` are snapshots of THIS token's
//! residual stream at its own layer-block boundaries; the state is cleared
//! and rebuilt at every forward call (decode shape: one token per call, KV
//! cache handles the sequence axis).
//!
//! MEASUREMENT LANE (`issue933_attn_res_probe` feature): no serving claim;
//! the issue's pre-registered ppl A/B decides GREEN/RED. Deterministic:
//! same weights + tokens + λ → byte-identical logits.

use katgpt_transformer::attn_res::{
    apply_attn_res_depth_profile, AttnResBlockState, AttnResConfig, AttnResScratch,
};
use katgpt_transformer::MultiLayerKVCache;

use crate::gemma_layer::GemmaTransformerWeightsF16;
use crate::transformer::gemma2::{NoVQuant, ValueStoreHook};
use crate::transformer::{ForwardContext, RAYON_MLP_THRESHOLD, RAYON_QKV_THRESHOLD};
use crate::types::{self, Config};

/// Probe posture: exponential-depth rate λ + layers-per-block S.
#[derive(Clone, Copy, Debug)]
pub struct AttnResDepthProbe {
    /// Layers per block (Kimi-K3-0.40B uses 4).
    pub block_size: usize,
    /// Exponential decay rate over block age. `f32::INFINITY` = disarmed
    /// (bit-identical to the stock forward — the α=0 law).
    pub lambda: f32,
}

/// Per-token depth-axis state, pre-allocated once and reused across calls.
pub struct GemmaAttnResState {
    block: AttnResBlockState,
    scratch: AttnResScratch,
}

impl GemmaAttnResState {
    /// Pre-allocate for `n_layer` layers at `block_size` layers per block.
    pub fn new(n_layer: usize, n_embd: usize, block_size: usize) -> Self {
        debug_assert!(block_size > 0, "block_size must be ≥ 1");
        // Full multiples push n_layer/block_size boundaries; +1 covers the
        // b0 push, +1 headroom for a non-multiple tail.
        let max_blocks = n_layer / block_size + 2;
        let config = AttnResConfig {
            hidden_size: n_embd,
            block_size,
            rms_eps: 1e-5,
        };
        Self {
            block: AttnResBlockState::new_with_capacity(n_embd, max_blocks),
            // scores sized for max_blocks residuals + 1 prefix entry.
            scratch: AttnResScratch::new(&config, max_blocks),
        }
    }

    /// Number of depth entries currently held (b0 + completed blocks).
    #[inline]
    pub fn entries(&self) -> usize {
        self.block.len()
    }
}

/// One depth-profile mix: `x ← out` (the mixed value becomes the residual
/// base, per the Kimi integration semantics).
#[inline]
fn mix_depth(
    ctx: &mut ForwardContext,
    state: &mut GemmaAttnResState,
    lambda: f32,
    n: usize,
) {
    let out = apply_attn_res_depth_profile(&state.block, &mut state.scratch, lambda, &ctx.x[..n]);
    ctx.x[..n].copy_from_slice(out);
}

/// [`crate::transformer::forward_gemma2_f16`] with the modelless AttnRes
/// wiring (see module docs). `pos` is the position of `token` within the
/// current chunk (KV rows `0..=pos` are this chunk's, exactly the stock
/// reuse contract — re-running a chunk from 0 overwrites exactly its rows).
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma2_f16_attn_res<'a>(
    ctx: &'a mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    probe: &AttnResDepthProbe,
    state: &mut GemmaAttnResState,
) -> &'a mut [f32] {
    // Verbatim body of forward_gemma2_f16_hk (NoVQuant posture) + the three
    // AttnRes insertion points, each marked `// T4:`.
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let mut vq = NoVQuant;

    // 1. Embedding: x = wte[token] * sqrt(n_embd) -- wte is f16, convert to f32
    let tok_off = token * n;
    let embed_scale = (n as f32).sqrt();
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) =
                (*weights.wte.get_unchecked(tok_off + i)).to_f32() * embed_scale;
        }
    }

    // T4: reset the depth-axis state for this token; b0 = the scaled
    // embedding (the always-included source).
    state.block.clear();
    state.block.push(&ctx.x[..n]);

    // Loop-invariant attention scale and token count — hoisted out of per-layer loop.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // T4: mix at the attention-half input (the mixed value becomes the
        // residual base).
        mix_depth(ctx, state, probe.lambda, n);

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

        // d2. [Issue 013 T2] Pre-RoPE K tap — `NoVQuant`: empty, inlined away.
        vq.keys_pre_rope(layer_idx, pos, &ctx.k[..kvd]);

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

        // f2. [Issue 013 T1] Lossy V-cache seam — `NoVQuant`: empty, inlined away.
        vq.value_stored(layer_idx, pos, &mut layer_cache.value);

        // f3. [Issue 013 T3] V read-path selector (verbatim from the stock
        // f16 loop — `NoVQuant` resolves to the plain path).
        let shape = super::attend::AttnShape {
            n_head: config.n_head,
            n_kv_head: n_kv,
            kv_dim: kvd,
            head_dim: hd,
            t_n,
            scale,
            softcap: config.attn_logit_softcapping,
            block_size: config.block_size,
        };
        #[cfg(feature = "row_logit_floor")]
        let fused_supported = config.attn_logit_softcapping > 0.0 && ctx.logit_floor.is_none();
        #[cfg(not(feature = "row_logit_floor"))]
        let fused_supported = config.attn_logit_softcapping > 0.0;
        let fused = if fused_supported {
            vq.fused_recon(layer_idx, t_n, &layer_cache.key)
        } else {
            None
        };
        if let Some(view) = fused {
            unsafe {
                super::attend::attend_row_fused(ctx, view, layer_idx, shape);
            }
        } else {
            let v_read: &[f32] = match vq.values_for_attention(layer_idx, pos, t_n, &layer_cache.key)
            {
                Some(v) => v,
                None => &layer_cache.value,
            };

            // g. Multi-head attention + softcapping (Plan 096: parallel heads)
            unsafe {
                super::attend::attend_row(ctx, &layer_cache.key, v_read, layer_idx, shape);
            }
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

        // T4: mix at the MLP-half input.
        mix_depth(ctx, state, probe.lambda, n);

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
            mlp_hidden,
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

        // Delta routing: accumulate per-sublayer deltas, route at block
        // boundaries (Plan 097) — verbatim from the stock f16 loop so the
        // fork stays faithful when both features are on.
        #[cfg(feature = "delta_routing")]
        super::apply_delta_routing_step(
            ctx,
            layer_idx,
            n,
            Some((
                &weights.delta_routing_query[layer_idx],
                &weights.delta_routing_norm[layer_idx],
            )),
        );

        // T4: block boundary — push the completed block sum. Kimi pushes at
        // block boundaries only (the trailing partial block never enters).
        if (layer_idx + 1) % probe.block_size == 0 {
            state.block.push(&ctx.x[..n]);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma_layer::{GemmaLayerWeights, GemmaTransformerWeights};
    use crate::transformer::forward_gemma2_f16;
    use crate::types::Rng;

    /// Small gemma-2-shaped config: post-norm, softcaps, GQA 2/1.
    fn test_config(n_layer: usize) -> Config {
        Config {
            n_layer,
            n_kv_head: 1,
            attn_logit_softcapping: 50.0,
            final_logit_softcapping: 30.0,
            ..Config::micro()
        }
    }

    /// Deterministic synthetic f32 weights → F16 (the tests_prefill
    /// `gemma2_test_weights` shape, converted field-wise).
    fn test_weights_f16(config: &Config, rng: &mut Rng) -> GemmaTransformerWeightsF16 {
        let n = config.n_embd;
        let kv_dim = config.n_kv_head * config.head_dim;
        let q_dim = config.n_head * config.head_dim;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;
        let scale = 0.01;
        let mut f32s = |len: usize| -> Vec<f32> { (0..len).map(|_| rng.normal() * scale).collect() };
        let f16s = |v: Vec<f32>| -> Vec<half::f16> { v.into_iter().map(half::f16::from_f32).collect() };

        let layers: Vec<_> = (0..config.n_layer)
            .map(|_| GemmaLayerWeights {
                attn_wq: f32s(q_dim * n),
                attn_wk: f32s(kv_dim * n),
                attn_wv: f32s(kv_dim * n),
                attn_wo: f32s(n * q_dim),
                gate_proj: f32s(mlp * n),
                up_proj: f32s(mlp * n),
                down_proj: f32s(n * mlp),
                input_norm: vec![1.0; n],
                post_attn_norm: vec![1.0; n],
                pre_mlp_norm: vec![1.0; n],
                post_mlp_norm: vec![1.0; n],
            })
            .collect();
        let f32_weights = GemmaTransformerWeights {
            wte: f32s(vocab * n),
            final_norm: vec![1.0; n],
            layers,
            #[cfg(feature = "delta_routing")]
            delta_routing_query: vec![vec![0.0f32; config.n_embd]; config.n_layer],
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: vec![vec![1.0f32; config.n_embd]; config.n_layer],
        };

        GemmaTransformerWeightsF16 {
            wte: f16s(f32_weights.wte),
            final_norm: f32_weights.final_norm.clone(),
            layers: f32_weights
                .layers
                .into_iter()
                .map(|l| crate::gemma_layer::GemmaLayerWeightsF16 {
                    attn_wq: f16s(l.attn_wq),
                    attn_wk: f16s(l.attn_wk),
                    attn_wv: f16s(l.attn_wv),
                    attn_wo: f16s(l.attn_wo),
                    gate_proj: f16s(l.gate_proj),
                    up_proj: f16s(l.up_proj),
                    down_proj: f16s(l.down_proj),
                    input_norm: l.input_norm,
                    post_attn_norm: l.post_attn_norm,
                    pre_mlp_norm: l.pre_mlp_norm,
                    post_mlp_norm: l.post_mlp_norm,
                })
                .collect(),
            #[cfg(feature = "delta_routing")]
            delta_routing_query: vec![vec![0.0f32; config.n_embd]; config.n_layer],
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: vec![vec![1.0f32; config.n_embd]; config.n_layer],
        }
    }

    /// λ=∞ (the disarmed posture) must be BIT-IDENTICAL to the stock
    /// `forward_gemma2_f16` at every position — a wiring drift indicts the
    /// instrument, not the mechanism (the 929 K=1 pin precedent).
    #[test]
    fn lambda_inf_is_bitwise_identical_to_stock() {
        let config = test_config(5);
        let mut rng = Rng::new(1234);
        let weights = test_weights_f16(&config, &mut rng);
        let probe = AttnResDepthProbe { block_size: 2, lambda: f32::INFINITY };

        let mut stock_ctx = crate::transformer::ForwardContext::new(&config);
        let mut stock_cache = MultiLayerKVCache::new(&config);
        let mut fork_ctx = crate::transformer::ForwardContext::new(&config);
        let mut fork_cache = MultiLayerKVCache::new(&config);
        let mut state = GemmaAttnResState::new(config.n_layer, config.n_embd, probe.block_size);

        let tokens = [3usize, 1, 4, 1, 5, 9, 2, 6];
        for (pos, &tok) in tokens.iter().enumerate() {
            let stock_logits =
                forward_gemma2_f16(&mut stock_ctx, &weights, &mut stock_cache, tok, pos, &config);
            let stock_bits: Vec<u32> = stock_logits[..config.vocab_size].iter().map(|v| v.to_bits()).collect();
            let fork_logits = forward_gemma2_f16_attn_res(
                &mut fork_ctx, &weights, &mut fork_cache, tok, pos, &config, &probe, &mut state,
            );
            for (v, &b) in fork_logits[..config.vocab_size].iter().zip(&stock_bits) {
                assert_eq!(v.to_bits(), b, "λ=∞ drifted from stock at pos {pos}");
            }
        }
    }

    /// Depth-entry accounting: b0 + one push per completed block; the
    /// trailing partial block never pushes (Kimi boundary semantics).
    #[test]
    fn block_boundaries_push_b0_plus_completed_blocks() {
        let config = test_config(5); // 5 layers
        let probe = AttnResDepthProbe { block_size: 2, lambda: 1.0 };
        let mut rng = Rng::new(99);
        let weights = test_weights_f16(&config, &mut rng);

        let mut ctx = crate::transformer::ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut state = GemmaAttnResState::new(config.n_layer, config.n_embd, probe.block_size);
        let _ = forward_gemma2_f16_attn_res(
            &mut ctx, &weights, &mut cache, 1, 0, &config, &probe, &mut state,
        );
        // 5 layers / block 2 → boundaries after layers 1 and 3 → 2 pushes + b0.
        assert_eq!(state.entries(), 3, "b0 + 2 completed blocks");
    }

    /// Determinism: same inputs → byte-identical logits across runs.
    #[test]
    fn deterministic_across_runs() {
        let config = test_config(4);
        let probe = AttnResDepthProbe { block_size: 2, lambda: 1.0 };
        let mut rng = Rng::new(7);
        let weights = test_weights_f16(&config, &mut rng);

        let run = || {
            let mut ctx = crate::transformer::ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut state = GemmaAttnResState::new(config.n_layer, config.n_embd, probe.block_size);
            let mut bits = Vec::new();
            for (pos, &tok) in [2usize, 7, 1].iter().enumerate() {
                let logits = forward_gemma2_f16_attn_res(
                    &mut ctx, &weights, &mut cache, tok, pos, &config, &probe, &mut state,
                );
                bits.extend(logits[..config.vocab_size].iter().map(|v| v.to_bits()));
            }
            bits
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "same inputs must give byte-identical logits");
    }

    /// A finite λ must actually change the logits once a block boundary has
    /// passed (guards against the wiring silently no-op-ing).
    #[test]
    fn finite_lambda_changes_logits_after_first_boundary() {
        let config = test_config(5);
        let mut rng = Rng::new(55);
        let weights = test_weights_f16(&config, &mut rng);
        let tokens = [1usize, 2, 3];

        let logits_for = |lambda: f32| -> Vec<u32> {
            let probe = AttnResDepthProbe { block_size: 2, lambda };
            let mut ctx = crate::transformer::ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut state = GemmaAttnResState::new(config.n_layer, config.n_embd, probe.block_size);
            let mut bits = Vec::new();
            for (pos, &tok) in tokens.iter().enumerate() {
                let logits = forward_gemma2_f16_attn_res(
                    &mut ctx, &weights, &mut cache, tok, pos, &config, &probe, &mut state,
                );
                bits.extend(logits[..config.vocab_size].iter().map(|v| v.to_bits()));
            }
            bits
        };

        let disarmed = logits_for(f32::INFINITY);
        let armed = logits_for(1.0);
        assert_ne!(
            disarmed, armed,
            "finite λ must change logits once a boundary has passed"
        );
    }
}
