//! Gemma-2 f16 projection-OUTPUT hook — katgpt-rs Issue 920 T1 (the
//! HyperThink modelless delta-overlay lane's capture/apply seam).
//!
//! [`forward_gemma2_f16_hk`] hooks the KV-cache rows (Issue 013); the
//! Issue-886 fork hooks linear INPUTS. This fork hooks the six
//! projection-OUTPUT windows the bias overlay attaches to — the insertion
//! points a trained bias tensor would occupy, immediately after each base
//! matmul:
//!
//! | site | after | width |
//! |---|---|---|
//! | `Q` | `attn_wq` (pre-RoPE) | `n_head·head_dim` |
//! | `V` | `attn_wv` (pre-cache) | `kv_dim` |
//! | `O` | `attn_wo` (pre post-attn norm) | `n_embd` |
//! | `Gate` | `gate_proj` (pre GeGLU) | `mlp_hidden` |
//! | `Up` | `up_proj` (pre GeGLU) | `mlp_hidden` |
//! | `Down` | `down_proj` (pre post-MLP norm) | `n_embd` |
//!
//! `K` is never hooked — a constant added to every key cancels under exact
//! softmax (the paper's exclusion; [`katgpt_pruners::BiasSite`] carries no K
//! variant either).
//!
//! The hook receives `&mut [f32]`: a CAPTURE impl copies (issue 920 T1's
//! paired `out_with_c − out_plain` sweeps); an OVERLAY impl adds a frozen
//! per-window constant (T3's serving simulation — the same insertion points,
//! zero weight mutation). The default no-op monomorphizes away exactly like
//! [`NoVQuant`]/[`NoHook`]. Structure is [`forward_gemma2_f16_hk`]'s layer
//! stack verbatim with the six hook calls inserted — the Issue-886 fork
//! precedent (each measurement lane forks the f16 forward at its observe
//! points; the fork is measurement-scoped, the plain path untouched).
//!
//! MEASUREMENT LANE (`hyperthink_t1` feature): no serving claim; the
//! issue's pre-registered gates decide. Deterministic: same binary + same
//! model + same probes → byte-identical captures (fixed-order f64
//! accumulation in the builder; matmul row partitioning is fixed-order).

use katgpt_transformer::MultiLayerKVCache;

use crate::gemma_layer::GemmaTransformerWeightsF16;
use crate::types;
use crate::transformer::ForwardContext;
use crate::transformer::RAYON_QKV_THRESHOLD;

/// The projection-output windows the bias overlay attaches to (K excluded —
/// softmax shift-invariance; see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiasDeltaSite {
    Q,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl BiasDeltaSite {
    /// Canonical slot order (serialization order; matches
    /// `katgpt_pruners::BiasSite::ALL`).
    pub const ALL: [BiasDeltaSite; 6] = [
        BiasDeltaSite::Q,
        BiasDeltaSite::V,
        BiasDeltaSite::O,
        BiasDeltaSite::Gate,
        BiasDeltaSite::Up,
        BiasDeltaSite::Down,
    ];

    /// Stable name (matches `katgpt_pruners::BiasSite::name`).
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            BiasDeltaSite::Q => "q",
            BiasDeltaSite::V => "v",
            BiasDeltaSite::O => "o",
            BiasDeltaSite::Gate => "gate",
            BiasDeltaSite::Up => "up",
            BiasDeltaSite::Down => "down",
        }
    }
}

/// The projection-output hook. Called once per site per layer per token,
/// AFTER the base matmul, BEFORE the next op (RoPE / cache / norm / GeGLU).
/// The hook may read or mutate `out`; mutations ARE the overlay.
pub trait BiasDeltaHook {
    /// `site`'s output window for `layer_idx` at `pos`.
    fn proj_out(
        &mut self,
        layer_idx: usize,
        pos: usize,
        site: BiasDeltaSite,
        out: &mut [f32],
    ) {
        let _ = (layer_idx, pos, site, out);
    }
}

/// Zero-overhead no-op [`BiasDeltaHook`] — the unarmed posture.
pub struct NoBiasDelta;
impl BiasDeltaHook for NoBiasDelta {}

/// [`forward_gemma2_f16_hk`]'s layer stack with the six projection-output
/// hook calls. Structure verbatim (embed scale, norms, RoPE, cache, softcap
/// attention, GeGLU, residuals); the only additions are the hook calls.
/// HEADLESS like the Issue-886 fork — ends at `ctx.hidden_state`, no lm_head
/// (see the tail comment). Sequences ≤ 4096 positions (the 883 fork's law:
/// no SWA rotation here).
#[allow(clippy::too_many_arguments)]
pub fn forward_gemma2_f16_bias_hook<H: BiasDeltaHook + ?Sized>(
    ctx: &mut ForwardContext,
    weights: &GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    hook: &mut H,
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

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        // b. pre-attn RMSNorm
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

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

        // ── HOOK Q: post-wq, pre-RoPE ──
        hook.proj_out(layer_idx, pos, BiasDeltaSite::Q, &mut ctx.q[..q_dim]);

        // e. RoPE
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // f. cache K,V — AFTER the V hook (an overlay mutation must land in
        //    the cached row attention reads; a capture reads the same
        //    pre-store projection output either way).
        // ── HOOK V: post-wv, pre-cache ──
        hook.proj_out(layer_idx, pos, BiasDeltaSite::V, &mut ctx.v[..kvd]);
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
            crate::transformer::attention_heads_parallel(
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

        // h. output projection
        types::matmul_f16(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // ── HOOK O: post-wo, pre post-attn norm ──
        hook.proj_out(layer_idx, pos, BiasDeltaSite::O, &mut ctx.x[..n]);

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

        // l. GeGLU
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        if n >= crate::transformer::RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| types::matmul_f16(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| types::matmul_f16(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            types::matmul_f16(gate_buf, wg, x_in, mlp_hidden, n);
            types::matmul_f16(up_buf, wu, x_in, mlp_hidden, n);
        }

        // ── HOOKS Gate + Up: post-projection, pre GeGLU ──
        hook.proj_out(layer_idx, pos, BiasDeltaSite::Gate, &mut ctx.gate[..mlp_hidden]);
        hook.proj_out(layer_idx, pos, BiasDeltaSite::Up, &mut ctx.up[..mlp_hidden]);

        types::gegelu_tanh(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // m. down projection
        types::matmul_f16_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            mlp_hidden,
        );

        // ── HOOK Down: post-down, pre post-MLP norm ──
        hook.proj_out(layer_idx, pos, BiasDeltaSite::Down, &mut ctx.x[..n]);

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

    // HEADLESS (the act_tapped fork's shape): the lane's T1 capture needs no
    // logits, and the tied-lm_head matmul over the 256K vocab would dominate
    // the capture run for nothing. T4's generation+overlay posture extends
    // this fork (a head flag) in its own commit — never a second fork.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Site names are the vocabulary's (the bin maps them 1:1).
    #[test]
    fn site_names_match_vocabulary() {
        let expected = ["q", "v", "o", "gate", "up", "down"];
        for (s, name) in BiasDeltaSite::ALL.iter().zip(expected) {
            assert_eq!(s.name(), name);
        }
    }

    /// The no-op hook compiles to nothing observable — a smoke, the
    /// monomorphization law is structural (same shape as NoVQuant).
    #[test]
    fn no_op_hook_is_zero_state() {
        let _ = NoBiasDelta;
        // Size sanity: a unit struct carries no capture state.
        assert_eq!(std::mem::size_of::<NoBiasDelta>(), 0);
    }
}
