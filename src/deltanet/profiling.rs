//! Per-component profiling for the ternary `DeltaNet` forward path.
//!
//! This module is an **instrumented copy** of
//! [`super::ternary_forward::forward_qwen_deltanet_ternary_with_hook`] with
//! `Instant::now()` timing around each section. It exists to localize the
//! ~370 ms/token gap not covered by the GPU FFN (Issue 600) and GPU input
//! projection (Issue 599/602) paths.
//!
//! ## Why a copy, not a hook?
//!
//! The production forward is a single monolithic function. Adding timing
//! instrumentation to it would either:
//! - Pollute the hot path with `Instant::now()` calls behind a cfg (fragile)
//! - Require a callback/hook mechanism (over-engineered for a one-shot
//!   measurement)
//!
//! A copy in a profiling-only module is the DRY-acceptable tradeoff: it will
//! drift from the production forward, but drift is caught by the G1 check
//! (the profiled forward must produce identical logits).
//!
//! ## Usage
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/p603 cargo test -p riir-engine \
//!     --features forward_profiling --release \
//!     --test bench_603_forward_profiling -- --nocapture --ignored
//! ```

use std::time::{Duration, Instant};

use super::forward::{
    AttentionLayerScratch, DeltaNetLayerScratch, HybridCache, HybridForwardScratch,
    causal_conv1d_update, effective_rotary_dim, expand_heads_into, gated_deltanet_step_inplace,
    l2_normalize, softplus,
};
use super::ternary_weights::{DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights};
use crate::rope::RopeFreqTable;
use crate::simd::fast_sigmoid;
use crate::types::{Config, DeltaNetLayerType, rmsnorm_with_gamma_eps, swiglu};
use katgpt_core::{
    TernaryFfnHook, TernaryGroupWeights, TernaryInputProjHook, TernaryMatvecHook,
    simd_ternary_group_matvec_parallel,
};

/// Per-component timing accumulator.
///
/// Each field is the total `Duration` spent in that component across ALL
/// layers and ALL tokens in the profiling run. Divide by `n_tokens` for
/// per-token averages.
#[derive(Default, Debug)]
pub struct ForwardProfiler {
    // ── Once-per-token components ──
    pub embed: Duration,
    pub final_norm: Duration,
    pub lm_head: Duration,

    // ── Per-layer components (accumulated across all 64 layers) ──
    pub input_norm: Duration,
    pub post_attn_norm: Duration,
    pub residual_add: Duration,

    // ── DeltaNet layer internals (accumulated across DeltaNet layers only) ──
    pub dn_input_proj: Duration,
    pub dn_conv1d: Duration,
    pub dn_gates: Duration,
    pub dn_expand_norm: Duration,
    pub dn_recurrence: Duration,
    pub dn_norm_silu: Duration,
    pub dn_out_proj: Duration,

    // ── Attention layer internals (accumulated across attention layers only) ──
    pub attn_qkv: Duration,
    pub attn_qk_norm: Duration,
    pub attn_rope: Duration,
    pub attn_cache_store: Duration,
    pub attn_scores: Duration,
    pub attn_gate: Duration,
    pub attn_out_proj: Duration,

    // ── FFN (accumulated across all layers) ──
    pub ffn: Duration,

    // ── Counters ──
    pub n_deltanet_layers: u64,
    pub n_attention_layers: u64,
    pub n_tokens: u64,
}

impl ForwardProfiler {
    /// Record a timing into a field.
    #[inline(always)]
    fn record(field: &mut Duration, start: Instant) {
        *field += start.elapsed();
    }

    /// Print the per-token breakdown table.
    pub fn report(&self) {
        let n_tok = self.n_tokens.max(1) as f64;
        let per = |d: &Duration| d.as_secs_f64() * 1000.0 / n_tok;

        let total_dn_layers = per(&self.dn_input_proj)
            + per(&self.dn_conv1d)
            + per(&self.dn_gates)
            + per(&self.dn_expand_norm)
            + per(&self.dn_recurrence)
            + per(&self.dn_norm_silu)
            + per(&self.dn_out_proj);

        let total_attn_layers = per(&self.attn_qkv)
            + per(&self.attn_qk_norm)
            + per(&self.attn_rope)
            + per(&self.attn_cache_store)
            + per(&self.attn_scores)
            + per(&self.attn_gate)
            + per(&self.attn_out_proj);

        let total_norms_residuals =
            per(&self.input_norm) + per(&self.post_attn_norm) + per(&self.residual_add);

        let total = per(&self.embed)
            + total_dn_layers
            + total_attn_layers
            + total_norms_residuals
            + per(&self.ffn)
            + per(&self.final_norm)
            + per(&self.lm_head);

        let dn_count = self.n_deltanet_layers.max(1) as f64 / n_tok;
        let attn_count = self.n_attention_layers.max(1) as f64 / n_tok;

        let sep30 = "─".repeat(30);
        let sep10 = "─".repeat(10);
        let sep6 = "─".repeat(6);
        let eq30 = "═".repeat(30);
        let eq10 = "═".repeat(10);
        let eq6 = "═".repeat(6);

        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!("  Forward Profiling Breakdown (per token, {n_tok:.0} tokens avg)");
        eprintln!("  DeltaNet layers/token: {dn_count:.0}  Attention layers/token: {attn_count:.0}");
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!("  {:<30} {:>10} {:>6}", "Component", "ms/token", "%");
        eprintln!("  {sep30:<30} {sep10:>10} {sep6:>6}");

        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        let row = |name: &str, ms: f64| {
            eprintln!("  {name:<30} {ms:>10.2} {:>5.1}%", pct(ms));
        };

        row("Embedding lookup", per(&self.embed));
        eprintln!("  {sep30:<30} {sep10:>10} {sep6:>6}");
        eprintln!("  DeltaNet layer internals:");
        row("  dn_input_proj", per(&self.dn_input_proj));
        row("  dn_conv1d", per(&self.dn_conv1d));
        row("  dn_gates", per(&self.dn_gates));
        row("  dn_expand_norm", per(&self.dn_expand_norm));
        row("  dn_recurrence", per(&self.dn_recurrence));
        row("  dn_norm_silu", per(&self.dn_norm_silu));
        row("  dn_out_proj", per(&self.dn_out_proj));
        row("  ΔN subtotal", total_dn_layers);
        eprintln!("  {sep30:<30} {sep10:>10} {sep6:>6}");
        eprintln!("  Attention layer internals:");
        row("  attn_qkv", per(&self.attn_qkv));
        row("  attn_qk_norm", per(&self.attn_qk_norm));
        row("  attn_rope", per(&self.attn_rope));
        row("  attn_cache_store", per(&self.attn_cache_store));
        row("  attn_scores", per(&self.attn_scores));
        row("  attn_gate", per(&self.attn_gate));
        row("  attn_out_proj", per(&self.attn_out_proj));
        row("  Attn subtotal", total_attn_layers);
        eprintln!("  {sep30:<30} {sep10:>10} {sep6:>6}");
        eprintln!("  Per-layer overhead:");
        row("  input_norm", per(&self.input_norm));
        row("  post_attn_norm", per(&self.post_attn_norm));
        row("  residual_add", per(&self.residual_add));
        row("  Overhead subtotal", total_norms_residuals);
        eprintln!("  {sep30:<30} {sep10:>10} {sep6:>6}");
        row("FFN (SwiGLU)", per(&self.ffn));
        row("Final RMSNorm", per(&self.final_norm));
        row("LM head", per(&self.lm_head));
        eprintln!("  {eq30:<30} {eq10:>10} {eq6:>6}");
        row("TOTAL", total);
        eprintln!("  Estimated throughput: {:.2} tok/s", 1000.0 / total.max(0.001));
        eprintln!("═══════════════════════════════════════════════════════════════");
    }
}

/// Ternary matvec dispatch: CPU SIMD (default) or GPU hook.
///
/// Copy of `ternary_forward::bitlinear` (private in production).
#[inline(always)]
fn bitlinear(y: &mut [f32], w: &TernaryGroupWeights, x: &[f32], hook: Option<&dyn TernaryMatvecHook>) {
    if let Some(h) = hook {
        h.matvec(w, &x[..w.cols], &mut y[..w.rows]);
    } else {
        simd_ternary_group_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
    }
}

/// Instrumented `DeltaNet` layer forward (mirrors production exactly + timing).
#[allow(clippy::too_many_arguments)]
fn profiled_deltanet_layer(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    state: &mut [f32],
    conv_state: &mut [f32],
    config: &Config,
    scratch: &mut DeltaNetLayerScratch,
    prof: &mut ForwardProfiler,
    hook: Option<&dyn TernaryMatvecHook>,
    input_proj_hook: Option<&dyn TernaryInputProjHook>,
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

    // 1-3. QKV + Z + A + B input projections
    // Issue 980: a/b are GateProjWeights — the fused hook (ternary a/b only)
    // or GateProjWeights dispatch serves them; this profiler lane serves the
    // pre-rotation Bonsai files, so a/b are ternary here in practice.
    let t = Instant::now();
    let (a_raw, b_raw) = scratch.ab_raw.split_at_mut(n_v_heads);
    if let Some(iph) = input_proj_hook {
        iph.input_projections(
            &layer.in_proj_qkv,
            &layer.in_proj_z,
            layer.in_proj_a.as_ternary().expect("profiler fused-hook lane requires ternary in_proj_a"),
            layer.in_proj_b.as_ternary().expect("profiler fused-hook lane requires ternary in_proj_b"),
            x_in,
            &mut scratch.qkv[..qkv_dim],
            &mut scratch.z[..z_dim],
            a_raw,
            b_raw,
        );
    } else {
        bitlinear(&mut scratch.qkv, &layer.in_proj_qkv, x_in, hook);
        bitlinear(&mut scratch.z, &layer.in_proj_z, x_in, hook);
        layer.in_proj_a.matvec_into(a_raw, x_in);
        layer.in_proj_b.matvec_into(b_raw, x_in);
    }
    ForwardProfiler::record(&mut prof.dn_input_proj, t);

    // 4. Split QKV
    let (q_slice, rest) = scratch.qkv.split_at_mut(q_dim);
    let (k_slice, v_slice) = rest.split_at_mut(k_dim);

    // 5. Conv1D
    let t = Instant::now();
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
    ForwardProfiler::record(&mut prof.dn_conv1d, t);

    // 6. Gates
    let t = Instant::now();
    let (beta, decay) = scratch.beta_decay.split_at_mut(n_v_heads);
    for h in 0..n_v_heads {
        beta[h] = fast_sigmoid(b_raw[h]);
        let a_val = a_raw[h] + layer.dt_bias[h];
        let g = layer.a_log[h] * softplus(a_val);
        decay[h] = g.exp();
    }
    ForwardProfiler::record(&mut prof.dn_gates, t);

    // 7-8. Expand + L2 normalize
    let t = Instant::now();
    let repeat_factor = n_v_heads / n_k_heads;
    expand_heads_into(q_slice, n_k_heads, key_dim, repeat_factor, &mut scratch.q_normed);
    expand_heads_into(k_slice, n_k_heads, key_dim, repeat_factor, &mut scratch.k_normed);
    for h in 0..n_v_heads {
        let off = h * key_dim;
        l2_normalize(&mut scratch.q_normed[off..off + key_dim]);
        l2_normalize(&mut scratch.k_normed[off..off + key_dim]);
    }
    ForwardProfiler::record(&mut prof.dn_expand_norm, t);

    // 9. Recurrence
    let t = Instant::now();
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
    ForwardProfiler::record(&mut prof.dn_recurrence, t);

    // 10. Norm + SiLU
    let t = Instant::now();
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
        let sig = fast_sigmoid(z_val);
        scratch.recurrent_output[i] *= z_val * sig;
    }
    ForwardProfiler::record(&mut prof.dn_norm_silu, t);

    // 11. Output projection
    let t = Instant::now();
    bitlinear(&mut x[..n_embd], &layer.out_proj, &scratch.recurrent_output, hook);
    ForwardProfiler::record(&mut prof.dn_out_proj, t);
}

/// Instrumented attention layer forward (mirrors production exactly + timing).
#[allow(clippy::too_many_arguments)]
fn profiled_attention_layer(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    cache: &mut crate::transformer::KVCache,
    pos: usize,
    config: &Config,
    rope_freq: &RopeFreqTable,
    scratch: &mut AttentionLayerScratch,
    prof: &mut ForwardProfiler,
    hook: Option<&dyn TernaryMatvecHook>,
) {
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = effective_rotary_dim(config);

    let x_in = &x[..n_embd];

    // 1. Q/K/V projections
    let t = Instant::now();
    bitlinear(&mut scratch.qg_buf, &layer.attn_wq, x_in, hook);
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        scratch.q_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src..src + hd]);
        scratch.gate_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src + hd..src + 2 * hd]);
    }
    bitlinear(&mut scratch.k_buf, &layer.attn_wk, x_in, hook);
    bitlinear(&mut scratch.v_buf, &layer.attn_wv, x_in, hook);
    ForwardProfiler::record(&mut prof.attn_qkv, t);

    // 2. QK-norm
    let t = Instant::now();
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.k_buf[off..off + hd], &layer.attn_k_norm, eps);
    }
    ForwardProfiler::record(&mut prof.attn_qk_norm, t);

    // 3. RoPE
    let t = Instant::now();
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
    ForwardProfiler::record(&mut prof.attn_rope, t);

    // 4. Store K, V in cache
    let t = Instant::now();
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
    ForwardProfiler::record(&mut prof.attn_cache_store, t);

    // 5. Attention scores
    let t = Instant::now();
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
    ForwardProfiler::record(&mut prof.attn_scores, t);

    // 6. Gate
    let t = Instant::now();
    for i in 0..q_dim {
        scratch.attn_out[i] *= fast_sigmoid(scratch.gate_buf[i]);
    }
    ForwardProfiler::record(&mut prof.attn_gate, t);

    // 7. Output projection
    let t = Instant::now();
    bitlinear(&mut x[..n_embd], &layer.attn_wo, &scratch.attn_out[..q_dim], hook);
    ForwardProfiler::record(&mut prof.attn_out_proj, t);
}

/// Instrumented full ternary forward — mirrors
/// `forward_qwen_deltanet_ternary_with_hook` exactly, with per-component
/// timing accumulated into `prof`.
///
/// Returns the logits slice (same as production) and updates `prof` with
/// timings. Call `prof.report()` after the run to see the breakdown.
#[allow(clippy::too_many_arguments)]
pub fn profiled_forward_ternary<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    prof: &mut ForwardProfiler,
    hook: Option<&dyn TernaryMatvecHook>,
    input_proj_hook: Option<&dyn TernaryInputProjHook>,
    ffn_hook: Option<&dyn TernaryFfnHook>,
) -> &'a mut [f32] {
    let n = config.n_embd;

    // 1. Embedding
    let t = Instant::now();
    weights.dequant_wte_row_into(token, &mut x[..n]);
    ForwardProfiler::record(&mut prof.embed, t);

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
        if is_linear {
            prof.n_deltanet_layers += 1;
        } else {
            prof.n_attention_layers += 1;
        }

        // a. Save residual
        let t = Instant::now();
        scratch.residual[..n].copy_from_slice(&x[..n]);
        ForwardProfiler::record(&mut prof.residual_add, t);

        // b. Input RMSNorm
        let t = Instant::now();
        rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.input_norm, config.rms_norm_eps);
        ForwardProfiler::record(&mut prof.input_norm, t);

        // c. Layer-specific forward
        if is_linear {
            profiled_deltanet_layer(
                &mut x[..n],
                layer_weights,
                &mut cache.deltanet_state.recurrent_states[layer_idx],
                &mut cache.deltanet_state.conv_states[layer_idx],
                config,
                &mut scratch.deltanet,
                prof,
                hook,
                input_proj_hook,
            );
        } else {
            profiled_attention_layer(
                &mut x[..n],
                layer_weights,
                &mut cache.kv_cache.layers[layer_idx],
                pos,
                config,
                rope_freq,
                &mut scratch.attention,
                prof,
                hook,
            );
        }

        // d. Residual add
        let t = Instant::now();
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }
        ForwardProfiler::record(&mut prof.residual_add, t);

        // e. Save residual for MLP
        let t = Instant::now();
        scratch.residual[..n].copy_from_slice(&x[..n]);
        ForwardProfiler::record(&mut prof.residual_add, t);

        // f. Post-attention RMSNorm
        let t = Instant::now();
        rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.post_attn_norm, config.rms_norm_eps);
        ForwardProfiler::record(&mut prof.post_attn_norm, t);

        // g-i. FFN
        let t = Instant::now();
        if let Some(fh) = ffn_hook {
            scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
            fh.ffn(
                &layer_weights.gate_proj,
                &layer_weights.up_proj,
                &layer_weights.down_proj,
                &scratch.hidden_copy[..n],
                &mut x[..n],
            );
        } else {
            bitlinear(&mut scratch.gate, &layer_weights.gate_proj, &x[..n], hook);
            bitlinear(&mut scratch.up, &layer_weights.up_proj, &x[..n], hook);
            swiglu(&mut scratch.hidden, &scratch.gate, &scratch.up);
            bitlinear(&mut x[..n], &layer_weights.down_proj, &scratch.hidden, hook);
        }
        ForwardProfiler::record(&mut prof.ffn, t);

        // i. Residual add
        let t = Instant::now();
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }
        ForwardProfiler::record(&mut prof.residual_add, t);
    }

    // 3. Final RMSNorm
    let t = Instant::now();
    rmsnorm_with_gamma_eps(&mut x[..n], &weights.final_norm, config.rms_norm_eps);
    ForwardProfiler::record(&mut prof.final_norm, t);

    // 4. LM head
    let t = Instant::now();
    scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
    bitlinear(
        &mut x[..config.vocab_size],
        &weights.lm_head,
        &scratch.hidden_copy,
        hook,
    );
    ForwardProfiler::record(&mut prof.lm_head, t);

    prof.n_tokens += 1;

    &mut x[..config.vocab_size]
}
