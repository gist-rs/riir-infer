//! Ternary/GDN forward over a quantized KV cache — Issue 919 T3 cell 2
//! (the strong-MA Bonsai-27B cell the gemma cell-1 verdict deferred to).
//!
//! The [`forward_qwen_deltanet_ternary`](super::ternary_forward) stack with
//! the full-attention layers' KV cache behind the
//! [`QuantizedKVCache`](katgpt_types::QuantizedKVCache) trait: stores go
//! through `store_key`/`store_value` and attention reads a **per-layer dequant
//! mirror** ([`TernaryKvMirror`]). The `DeltaNet` layers carry no KV by
//! construction (a fixed recurrent state, not a per-token cache) — they run
//! the unmodified
//! [`forward_deltanet_layer_ternary`](super::ternary_forward::forward_deltanet_layer_ternary)
//! and never touch the backend.
//!
//! ## Correspondence law (the `gemma2_quantized` precedent)
//!
//! Everything else — gated Q projection, QK-norm, partial RoPE, output
//! gating, Hadamard rotation, SwiGLU MLP, LM head — is byte-for-byte the
//! plain ternary forward's structure with steps (4) and (5) of
//! `forward_attention_layer_ternary` replaced. The correspondence is pinned
//! by the G0 control in [`mod@tests`] (synthetic tiny model: the mirror path
//! over [`RawF32KvCache`](crate::quant::kvq_ab::RawF32KvCache) must reproduce
//! the plain forward's logits bit-identically) and by the harness bin's G0
//! gate on the real checkpoint.
//!
//! ## Mirror semantics (store-time quantization only)
//!
//! The kvq_ab backends quantize AT STORE time (RawF32 copies, Diag
//! accumulates raw, Q8/Exempt quantize the row) — no KVarN-style tile-time
//! requantization — so a mirror row is final the moment it is written and
//! the refresh is exactly one row per store. A tile-time-quantizing backend
//! would need the gemma-side tile-flip refresh and is NOT supported by this
//! lane. Per-layer mirrors (not one shared buffer + a `last_layer` guard):
//! the hybrid's attention layers are never consecutive, so the shared-buffer
//! guard would force a full `0..=pos` re-dequant at every attention layer —
//! the exact defect the gemma lane measured before the Issue 919 fix. This
//! is the module doc's own named "production alternative" shape.
//!
//! ## Divergences from the plain forward (both documented, both harmless
//! to the A/B)
//!
//! - No matvec/FFN hooks: the A/B passes none; the hook plumbing stays in
//!   the plain forward.
//! - No layer capture: same reason.
//!
//! MEASUREMENT LANE (the vk_p1_g1 P0 posture): no serving claim; the A/B
//! bin's pre-registered gates decide quality.

use super::forward::{
    AttentionLayerScratch, DeltaNetState, HybridForwardScratch, effective_rotary_dim,
};
use super::rotation::{TernaryRotationConfig, rotate_forward_inplace, rotate_inverse_inplace};
use super::ternary_forward::{bitlinear, forward_deltanet_layer_ternary};
use super::ternary_weights::{DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights};
use crate::types::{self, Config, DeltaNetLayerType, rmsnorm_with_gamma_eps, swiglu};
use katgpt_core::simd_ternary_group_matvec_parallel;
use katgpt_types::QuantizedKVCache;

/// Caller-owned per-layer dequant mirror: the attention-visible image of the
/// quantized cache, one flat `[max_seq × kvd]` pair per layer (only the
/// full-attention layers are written; the `DeltaNet` slots stay zero).
///
/// `reset` zeroes every row — hygiene, not correctness: the per-token path
/// writes each row before attention ever reads it, so a stale read would be
/// a logic bug; zeroing turns it into a loud G0 failure instead of a silent
/// one. Cheap: `[n_layer × max_seq × kvd]` memsets once per chunk.
pub struct TernaryKvMirror {
    /// Per-layer dequantized K rows `[n_layer][max_seq × kvd]`.
    pub key: Vec<Vec<f32>>,
    /// Per-layer dequantized V rows `[n_layer][max_seq × kvd]`.
    pub value: Vec<Vec<f32>>,
}

impl TernaryKvMirror {
    #[must_use]
    pub fn new(config: &Config, max_seq_len: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![vec![0.0; max_seq_len * kvd]; config.n_layer],
            value: vec![vec![0.0; max_seq_len * kvd]; config.n_layer],
        }
    }

    /// Zero every mirror row. Call beside the backend's own `reset()` per
    /// chunk (write-before-read makes this hygiene, not correctness).
    pub fn reset(&mut self) {
        for row in &mut self.key {
            row.fill(0.0);
        }
        for row in &mut self.value {
            row.fill(0.0);
        }
    }
}

/// One full-attention layer of the ternary forward, K/V behind a generic
/// quantized backend + its mirror. Steps 1–3 and 6–7 are byte-for-byte
/// `forward_attention_layer_ternary` (gated Q projection → QK-norm → partial
/// RoPE; output gating → rotated output projection); steps 4–5 are the
/// quantized store + mirror attention.
///
/// No hook: the A/B lane passes none (see the module doc).
#[allow(clippy::too_many_arguments)]
pub fn forward_attention_layer_ternary_qkv<C: QuantizedKVCache>(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    cache: &mut C,
    mirror: &mut TernaryKvMirror,
    layer_idx: usize,
    pos: usize,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    scratch: &mut AttentionLayerScratch,
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

    // 1. Gated Q projection + K + V (ternary). Issue 594/980 — verbatim.
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
    bitlinear(&mut scratch.qg_buf, &layer.attn_wq, x_qkv, None);
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        scratch.q_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src..src + hd]);
        scratch.gate_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src + hd..src + 2 * hd]);
    }
    bitlinear(&mut scratch.k_buf, &layer.attn_wk, x_qkv, None);
    bitlinear(&mut scratch.v_buf, &layer.attn_wv, x_qkv, None);

    // 2. QK-norm (verbatim).
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.k_buf[off..off + hd], &layer.attn_k_norm, eps);
    }

    // 3. Partial RoPE (verbatim).
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

    // 4'. Quantized store + mirror refresh (row `pos` only — store-time
    //     quantization, see the module doc).
    cache.store_key(layer_idx, pos, &scratch.k_buf[..kvd]);
    cache.store_value(layer_idx, pos, &scratch.v_buf[..kvd]);
    {
        let key_row = &mut mirror.key[layer_idx][pos * kvd..(pos + 1) * kvd];
        cache.dequantize_key_into(layer_idx, pos, key_row);
    }
    {
        let val_row = &mut mirror.value[layer_idx][pos * kvd..(pos + 1) * kvd];
        cache.dequantize_value_into(layer_idx, pos, val_row);
    }

    // 5'. Multi-head attention with GQA over the mirror.
    let scale = 1.0 / (hd as f32).sqrt();
    scratch.attn_out[..q_dim].fill(0.0);
    let t_n = pos + 1;
    unsafe {
        crate::transformer::attention_heads_parallel(
            &scratch.q_buf,
            &mirror.key[layer_idx],
            &mirror.value[layer_idx],
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

    // 6. Output gating (verbatim).
    for i in 0..q_dim {
        scratch.attn_out[i] *= crate::simd::fast_sigmoid(scratch.gate_buf[i]);
    }

    // 7. Output projection — folded: rotate `attn_out` in place (verbatim).
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(q_dim);
        rotate_forward_inplace(&mut scratch.attn_out[..q_dim], signs, rot.block_size);
    }
    bitlinear(
        &mut x[..n_embd],
        &layer.attn_wo,
        &scratch.attn_out[..q_dim],
        None,
    );
}

/// The ternary hybrid forward with the full-attention layers' KV behind a
/// generic quantized backend. The `DeltaNet` layers run the unmodified
/// `forward_deltanet_layer_ternary` over `state`; only attention layers touch
/// `cache`/`mirror`.
///
/// Byte-for-byte `forward_qwen_deltanet_ternary_with_hook`'s structure
/// (embedding → inverse-embedding rotation → hybrid layer loop → final norm →
/// rotated LM head) with the cache plumbing swapped. No hooks, no capture.
#[allow(clippy::too_many_arguments)]
pub fn forward_qwen_deltanet_ternary_qkv<'a, C: QuantizedKVCache>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    state: &mut DeltaNetState,
    cache: &mut C,
    mirror: &mut TernaryKvMirror,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let v_dim = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;

    // Rotation state (Issue 980) — verbatim.
    let rotation = weights.rotation.as_ref();
    let max_rot_len = n.max(v_dim).max(config.mlp_hidden);
    debug_assert!(
        scratch.rotation_buf.len() >= max_rot_len,
        "rotation scratch {} < required {max_rot_len} (HybridForwardScratch::new sizes it)",
        scratch.rotation_buf.len()
    );

    // 1. Embedding lookup + inverse rotation — verbatim.
    weights.dequant_wte_row_into(token, &mut x[..n]);
    if let Some(rot) = rotation
        && rot.inverse_embedding
    {
        let signs = rot.signs_for_width(n);
        rotate_inverse_inplace(&mut x[..n], signs, rot.block_size);
    }

    // 2. Layer loop with hybrid dispatch.
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        // a. Save residual + pre-attention RMSNorm (the layer body's steps a-b).
        scratch.residual[..n].copy_from_slice(&x[..n]);
        rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.input_norm, config.rms_norm_eps);

        // c. Layer-specific forward.
        if is_linear {
            forward_deltanet_layer_ternary(
                &mut x[..n],
                layer_weights,
                &mut state.recurrent_states[layer_idx],
                &mut state.conv_states[layer_idx],
                config,
                &mut scratch.deltanet,
                None,
                None,
                rotation,
                &mut scratch.rotation_buf,
            );
        } else {
            forward_attention_layer_ternary_qkv(
                &mut x[..n],
                layer_weights,
                cache,
                mirror,
                layer_idx,
                pos,
                config,
                rope_freq,
                &mut scratch.attention,
                rotation,
                &mut scratch.rotation_buf,
            );
        }

        // d. Residual add.
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }

        // e-i. MLP tail (the layer body's steps e-i, plain branch — no FFN
        // hook in this lane): pre-MLP norm → rotated shared input → gate+up →
        // SwiGLU → rotated hidden → down → residual add.
        scratch.residual[..n].copy_from_slice(&x[..n]);
        rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.post_attn_norm, config.rms_norm_eps);
        let x_ffn: &[f32] = if let Some(rot) = rotation {
            let signs = rot.signs_for_width(n);
            scratch.rotation_buf[..n].copy_from_slice(&x[..n]);
            rotate_forward_inplace(&mut scratch.rotation_buf[..n], signs, rot.block_size);
            &scratch.rotation_buf[..n]
        } else {
            &x[..n]
        };
        bitlinear(&mut scratch.gate, &layer_weights.gate_proj, x_ffn, None);
        bitlinear(&mut scratch.up, &layer_weights.up_proj, x_ffn, None);
        swiglu(&mut scratch.hidden, &scratch.gate, &scratch.up);
        if let Some(rot) = rotation {
            let signs = rot.signs_for_width(config.mlp_hidden);
            rotate_forward_inplace(&mut scratch.hidden, signs, rot.block_size);
        }
        bitlinear(&mut x[..n], &layer_weights.down_proj, &scratch.hidden, None);
        for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
            *xi += *r;
        }
    }

    // 3. Final RMSNorm — verbatim.
    rmsnorm_with_gamma_eps(&mut x[..n], &weights.final_norm, config.rms_norm_eps);

    // 4. LM head — rotated copy, direct kernel (no hook in this lane).
    scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(n);
        rotate_forward_inplace(&mut scratch.hidden_copy[..n], signs, rot.block_size);
    }
    simd_ternary_group_matvec_parallel(
        &weights.lm_head,
        &scratch.hidden_copy[..n],
        &mut x[..config.vocab_size],
    );

    &mut x[..config.vocab_size]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::forward::HybridCache;
    use crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
    use crate::quant::kvq_ab::{Q8AbsmaxKvCache, RawF32KvCache};
    use katgpt_types::TernaryGroupWeights;

    /// A tiny but REAL qwen35 hybrid: 2 layers (GDN, attention), every
    /// projection a real ternary group-quant of a deterministic pattern.
    /// Shapes are the minimum every scratch sizing path accepts.
    fn tiny_config() -> Config {
        Config {
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 16,
            mlp_hidden: 96,
            vocab_size: 128,
            block_size: 16,
            deltanet_linear_n_heads: 4,
            deltanet_linear_n_value_heads: 4,
            deltanet_linear_head_dim: 16,
            deltanet_conv_kernel_size: 4,
            rms_norm_eps: 1e-5,
            ..Config::default()
        }
    }

    fn pat(i: usize) -> f32 {
        (((i % 13) as f32) - 6.0) * 0.05
    }

    fn tiny_proj(rows: usize, cols: usize) -> crate::deltanet::ternary_weights::ProjWeights {
        let w: Vec<f32> = (0..rows * cols).map(pat).collect();
        crate::deltanet::ternary_weights::ProjWeights::Ternary(
            TernaryGroupWeights::quantize_from_f32(&w, rows, cols),
        )
    }

    fn tiny_gate_proj(
        rows: usize,
        cols: usize,
    ) -> crate::deltanet::ternary_weights::GateProjWeights {
        let w: Vec<f32> = (0..rows * cols).map(pat).collect();
        crate::deltanet::ternary_weights::GateProjWeights::Ternary(
            TernaryGroupWeights::quantize_from_f32(&w, rows, cols),
        )
    }

    fn tiny_weights(config: &Config) -> QwenDeltaNetTernaryWeights {
        use crate::deltanet::ternary_weights::DeltaNetTernaryLayerWeights;
        use crate::types::DeltaNetLayerType;
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = types::kv_dim(config);
        let hd = config.head_dim;
        let n_k = config.deltanet_linear_n_heads;
        let n_v = config.deltanet_linear_n_value_heads;
        let v_dim = n_v * config.deltanet_linear_head_dim;
        let qkv_dim = 2 * n_k * config.deltanet_linear_head_dim + v_dim;
        // The conv is DEPTHWISE over the concatenated q|k|v channels:
        // conv_dim = (n_k + n_k + n_v) · head_dim (causal_conv1d_update
        // indexes conv_weight[ch * kernel + k] for ch in 0..conv_dim).
        let conv_dim = qkv_dim;
        let kernel = config.deltanet_conv_kernel_size;
        // GDN-layer dense fields (a_log < 0 → decay in (0,1); dt_bias 0).
        let gdn_layer = || DeltaNetTernaryLayerWeights {
            attn_wq: crate::deltanet::ternary_weights::ProjWeights::empty(),
            attn_wk: crate::deltanet::ternary_weights::ProjWeights::empty(),
            attn_wv: crate::deltanet::ternary_weights::ProjWeights::empty(),
            attn_wo: crate::deltanet::ternary_weights::ProjWeights::empty(),
            in_proj_qkv: tiny_proj(qkv_dim, n),
            in_proj_a: tiny_gate_proj(n_v, n),
            in_proj_b: tiny_gate_proj(n_v, n),
            in_proj_z: tiny_proj(v_dim, n),
            out_proj: tiny_proj(n, v_dim),
            gate_proj: tiny_proj(config.mlp_hidden, n),
            up_proj: tiny_proj(config.mlp_hidden, n),
            down_proj: tiny_proj(n, config.mlp_hidden),
            attn_q_norm: Vec::new(),
            attn_k_norm: Vec::new(),
            conv1d_weight: (0..conv_dim * kernel).map(pat).collect(),
            // a_log is PRE-NEGATED (loader copies ssm_a = -exp(A_log) verbatim);
            // negative values keep decay = exp(a_log·softplus(·)) in (0,1).
            a_log: (0..n_v).map(|i| -0.5 - pat(i).abs()).collect(),
            dt_bias: (0..n_v).map(pat).collect(),
            // ssm_norm is the SHARED per-head gamma [head_dim] (the ternary
            // forward's debug_assert pins len == val_dim == head_dim).
            linear_norm: vec![1.0; config.deltanet_linear_head_dim],
            input_norm: vec![1.0; n],
            post_attn_norm: vec![1.0; n],
        };
        // Attention-layer dense fields (GDN fields empty — unused).
        let attn_layer = || DeltaNetTernaryLayerWeights {
            attn_wq: tiny_proj(2 * q_dim, n),
            attn_wk: tiny_proj(kvd, n),
            attn_wv: tiny_proj(kvd, n),
            attn_wo: tiny_proj(n, q_dim),
            in_proj_qkv: crate::deltanet::ternary_weights::ProjWeights::empty(),
            in_proj_a: crate::deltanet::ternary_weights::GateProjWeights::empty(),
            in_proj_b: crate::deltanet::ternary_weights::GateProjWeights::empty(),
            in_proj_z: crate::deltanet::ternary_weights::ProjWeights::empty(),
            out_proj: crate::deltanet::ternary_weights::ProjWeights::empty(),
            gate_proj: tiny_proj(config.mlp_hidden, n),
            up_proj: tiny_proj(config.mlp_hidden, n),
            down_proj: tiny_proj(n, config.mlp_hidden),
            attn_q_norm: vec![1.0; hd],
            attn_k_norm: vec![1.0; hd],
            conv1d_weight: Vec::new(),
            a_log: Vec::new(),
            dt_bias: Vec::new(),
            linear_norm: Vec::new(),
            input_norm: vec![1.0; n],
            post_attn_norm: vec![1.0; n],
        };
        let wte_w: Vec<f32> = (0..config.vocab_size * n).map(pat).collect();
        QwenDeltaNetTernaryWeights {
            wte: TernaryGroupWeights::quantize_from_f32(&wte_w, config.vocab_size, n),
            final_norm: vec![1.0; n],
            lm_head: TernaryGroupWeights::quantize_from_f32(&wte_w, config.vocab_size, n),
            layers: vec![gdn_layer(), attn_layer()],
            layer_types: vec![DeltaNetLayerType::DeltaNet, DeltaNetLayerType::Attention],
            rotation: None,
        }
    }

    /// G0 (the harness law, unit-level): the mirror path over RawF32 must
    /// reproduce the plain forward's logits BIT-IDENTICALLY, per token, with
    /// the GDN state carried identically across tokens (the paired reset
    /// before each sequence is exercised by running two chunks).
    #[test]
    fn g0_rawf32_mirror_is_bit_identical_to_plain() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::forward::effective_rotary_dim(&config),
        );
        let seq = [3usize, 17, 91, 42, 7, 100, 55, 63];

        // Plain arm.
        let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut x_plain = vec![0.0f32; config.vocab_size];
        let mut plain: Vec<Vec<f32>> = Vec::new();
        for chunk in seq.chunks(4) {
            cache.reset();
            for (pos, &t) in chunk.iter().enumerate() {
                let logits = forward_qwen_deltanet_ternary(
                    &mut x_plain,
                    &weights,
                    &mut cache,
                    t,
                    pos,
                    &config,
                    &mut scratch,
                    &rope_freq,
                );
                plain.push(logits.to_vec());
            }
        }

        // Mirror arm (RawF32 — full precision).
        let kvd = types::kv_dim(&config);
        let mut state = DeltaNetState::new(&config, &weights.layer_types);
        let mut raw = RawF32KvCache::new(config.n_layer, config.block_size, kvd);
        let mut mirror = TernaryKvMirror::new(&config, config.block_size);
        let mut scratch_q = HybridForwardScratch::new(&config);
        let mut x_q = vec![0.0f32; config.vocab_size];
        let mut mir: Vec<Vec<f32>> = Vec::new();
        for chunk in seq.chunks(4) {
            raw.reset();
            mirror.reset();
            for s in state.recurrent_states.iter_mut() {
                s.fill(0.0);
            }
            for s in state.conv_states.iter_mut() {
                s.fill(0.0);
            }
            for (pos, &t) in chunk.iter().enumerate() {
                let logits = forward_qwen_deltanet_ternary_qkv(
                    &mut x_q,
                    &weights,
                    &mut state,
                    &mut raw,
                    &mut mirror,
                    t,
                    pos,
                    &config,
                    &mut scratch_q,
                    &rope_freq,
                );
                mir.push(logits.to_vec());
            }
        }

        assert_eq!(plain.len(), mir.len());
        for (i, (a, b)) in plain.iter().zip(&mir).enumerate() {
            assert_eq!(a, b, "token {i}: mirror path diverged from plain");
        }
    }

    /// The seam actually engages: the Q8 arm through the same mirror path
    /// must DIFFER from plain (a bit-identical Q8 arm would mean the
    /// quantized store never fed attention — a silently-broken harness).
    #[test]
    fn q8_arm_through_the_mirror_differs_from_plain() {
        let config = tiny_config();
        let weights = tiny_weights(&config);
        let rope_freq = crate::rope::RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::forward::effective_rotary_dim(&config),
        );
        let seq = [3usize, 17, 91, 42, 7, 100, 55, 63];
        let kvd = types::kv_dim(&config);

        let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut x_plain = vec![0.0f32; config.vocab_size];
        let mut plain: Vec<Vec<f32>> = Vec::new();
        for chunk in seq.chunks(4) {
            cache.reset();
            for (pos, &t) in chunk.iter().enumerate() {
                let logits = forward_qwen_deltanet_ternary(
                    &mut x_plain,
                    &weights,
                    &mut cache,
                    t,
                    pos,
                    &config,
                    &mut scratch,
                    &rope_freq,
                );
                plain.push(logits.to_vec());
            }
        }

        let mut state = DeltaNetState::new(&config, &weights.layer_types);
        let mut q8 = Q8AbsmaxKvCache::new(config.n_layer, config.block_size, kvd);
        let mut mirror = TernaryKvMirror::new(&config, config.block_size);
        let mut scratch_q = HybridForwardScratch::new(&config);
        let mut x_q = vec![0.0f32; config.vocab_size];
        let mut diffs = 0usize;
        let mut i = 0usize;
        for chunk in seq.chunks(4) {
            q8.reset();
            mirror.reset();
            for s in state.recurrent_states.iter_mut() {
                s.fill(0.0);
            }
            for s in state.conv_states.iter_mut() {
                s.fill(0.0);
            }
            for (pos, &t) in chunk.iter().enumerate() {
                let logits = forward_qwen_deltanet_ternary_qkv(
                    &mut x_q,
                    &weights,
                    &mut state,
                    &mut q8,
                    &mut mirror,
                    t,
                    pos,
                    &config,
                    &mut scratch_q,
                    &rope_freq,
                );
                if logits != plain[i].as_slice() {
                    diffs += 1;
                }
                i += 1;
            }
        }
        assert!(diffs > 0, "Q8 arm read bit-identical — the quantized store never fed attention");
    }

    /// The mirror resets clean: after `reset`, rows read back zero, so a
    /// consumer that forgot to store before reading fails loudly instead of
    /// carrying the previous sequence's K/V.
    #[test]
    fn mirror_reset_zeroes_every_layer() {
        let config = tiny_config();
        let mut mirror = TernaryKvMirror::new(&config, 8);
        mirror.key[1][0] = 1.5;
        mirror.value[0][7] = -2.0;
        mirror.reset();
        for (l, k) in mirror.key.iter().enumerate() {
            assert!(k.iter().all(|&v| v == 0.0), "layer {l} key not zeroed");
        }
        for (l, v) in mirror.value.iter().enumerate() {
            assert!(v.iter().all(|&v| v == 0.0), "layer {l} value not zeroed");
        }
    }
}
