//! Loop-alignment probe at a looped answer position (katgpt-rs Issue 929 /
//! Research 614 — DiscoLoop, arXiv:2607.00341). The Bonsai-2 capable-checkpoint
//! leg's shared machinery: the runner bin and the machinery gates consume the
//! SAME code (the Issue-022 T3.1 parity law — a forward-body drift indicts
//! the instrument, not the checkpoint).
//!
//! Mechanism (the lt2 weight-shared-loop semantics, caller-side on the hybrid
//! qwen35 stack): the caller embeds the answer-position token ONCE (the stock
//! embed glue: bit-plane row read + inverse-rotate a folded table), then this
//! module re-enters the FULL layer stack K times — the same public
//! `qwen_deltanet_ternary_layer_body` the stock forward calls, in the same
//! order. Each pass advances the GDN recurrent+conv states once and
//! overwrites the position's KV: that IS the loop on a hybrid model, and the
//! probe measures OUR looped path.
//!
//! Probe per loop k, in the decode basis (the same readout the stock final
//! head applies — final RMSNorm, then the folded head's forward rotate, then
//! the ternary head matvec):
//! - `cos(H, W[v̂])` + top1−top2 margin via `probe_alignment_with_row`
//!   (the `W[v̂]` row dequantized on demand; the packed head is never
//!   materialized — a whole-table f32 dequant would need ~5.1 GB);
//! - `cos(H, W[bridge])` — the paper's true-bridge quantity (analysis-only;
//!   the fixture knows the bridge, the runtime never reads it).
//!
//! The K-th loop's probe logits ARE the decode readout of the looped state —
//! identical computation to the stock final head on the same state — so the
//! returned argmax is the greedy decode's first token (one head GEMV saved
//! per query; pinned bit-identical to the stock forward at K=1 by test).

use katgpt_core::loop_alignment_probe::probe_alignment_with_row;
use katgpt_core::simd::{simd_dot_f32, simd_sum_sq};
use katgpt_core::simd_ternary_group_matvec_parallel;

use super::forward::{HybridCache, HybridForwardScratch};
use super::rotation::{rotate_forward_inplace, rotate_inverse_inplace};
use super::ternary_forward::qwen_deltanet_ternary_layer_body;
use super::ternary_weights::QwenDeltaNetTernaryWeights;
use crate::rope::RopeFreqTable;
use crate::types::{Config, DeltaNetLayerType, rmsnorm_with_gamma_eps};

/// Probe scratch, allocated once per run (the hot-loop rule: no per-call
/// alloc inside the loop body).
pub struct LoopProbeScratch {
    /// Final-normed (+ folded-head rotated) state copy — the decode basis.
    pub xh: Vec<f32>,
    /// One ternary head matvec's logits.
    pub logits: Vec<f32>,
    /// The dequantized `W[v̂]` row.
    pub row: Vec<f32>,
    /// The dequantized `W[bridge]` row.
    pub bridge_row: Vec<f32>,
}

impl LoopProbeScratch {
    pub fn new(config: &Config) -> Self {
        Self {
            xh: vec![0.0; config.n_embd],
            logits: vec![0.0; config.vocab_size],
            row: vec![0.0; config.n_embd],
            bridge_row: vec![0.0; config.n_embd],
        }
    }
}

/// One probe read at one loop of the answer position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopProbeRow {
    pub argmax: usize,
    pub margin: f32,
    /// `cos(H, W[v̂])` — the alignment leg (decode basis).
    pub cos_alignment: f32,
    /// `cos(H, W[bridge])` — the paper's true-bridge quantity (`0.0` when
    /// `bridge == 0`, the no-bridge sentinel).
    pub cos_bridge: f32,
}

/// Embed the answer-position token once — the stock embed glue, verbatim
/// (bit-plane row read + inverse-rotate a Hadamard-folded embedding table).
/// The loop re-enters the stack over THIS state; re-embedding per loop would
/// reset it (the loop is the caller's, per the forward's `x` in/out
/// contract).
pub fn embed_answer_token(x: &mut [f32], weights: &QwenDeltaNetTernaryWeights, token: usize) {
    let n = x.len().min(weights.wte.cols);
    weights.dequant_wte_row_into(token, &mut x[..n]);
    if let Some(rot) = weights.rotation.as_ref()
        && rot.inverse_embedding
    {
        let signs = rot.signs_for_width(n);
        rotate_inverse_inplace(&mut x[..n], signs, rot.block_size);
    }
}

/// K weight-shared stack passes over the embedded state + one probe read per
/// pass. `out_rows` is cleared and extended to `loops` rows (caller-sized,
/// no alloc here beyond the rows themselves). Returns the greedy decode's
/// first token — the argmax of the FINAL loop's readout (the K-th probe's
/// logits are the decode readout: the stock final head applies the same
/// norm+rotate+matvec to the same state).
#[allow(clippy::too_many_arguments)]
pub fn looped_answer_position_probe(
    x: &mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    config: &Config,
    answer_pos: usize,
    loops: usize,
    bridge_token: usize,
    probe_scratch: &mut LoopProbeScratch,
    out_rows: &mut Vec<LoopProbeRow>,
) -> usize {
    let n = config.n_embd;
    out_rows.clear();
    out_rows.reserve(loops);
    let mut first_token = 0usize;
    for _k in 0..loops {
        // One full stack pass — the stock per-layer body, same order.
        for (li, layer_weights) in weights.layers.iter().enumerate() {
            let is_linear = weights.layer_types[li] == DeltaNetLayerType::DeltaNet;
            qwen_deltanet_ternary_layer_body(
                x,
                layer_weights,
                is_linear,
                &mut cache.deltanet_state.recurrent_states[li],
                &mut cache.deltanet_state.conv_states[li],
                &mut cache.kv_cache.layers[li],
                answer_pos,
                config,
                scratch,
                rope_freq,
                weights.rotation.as_ref(),
                None,
                None,
                None,
            );
        }
        // Probe read in the decode basis.
        probe_scratch.xh.copy_from_slice(&x[..n]);
        rmsnorm_with_gamma_eps(
            &mut probe_scratch.xh,
            &weights.final_norm,
            config.rms_norm_eps,
        );
        if let Some(rot) = weights.rotation.as_ref() {
            let signs = rot.signs_for_width(n);
            rotate_forward_inplace(&mut probe_scratch.xh, signs, rot.block_size);
        }
        simd_ternary_group_matvec_parallel(
            &weights.lm_head,
            &probe_scratch.xh,
            &mut probe_scratch.logits,
        );
        let probe = probe_alignment_with_row(
            &probe_scratch.xh,
            &probe_scratch.logits,
            n,
            &mut probe_scratch.row,
            |v, buf| weights.dequant_lm_head_row_into(v, buf),
        );
        let cos_bridge = if bridge_token != 0 {
            weights.dequant_lm_head_row_into(bridge_token, &mut probe_scratch.bridge_row);
            let denom = (simd_sum_sq(&probe_scratch.xh, n)
                * simd_sum_sq(&probe_scratch.bridge_row, n))
            .sqrt();
            if denom > 0.0 && denom.is_finite() {
                simd_dot_f32(&probe_scratch.xh, &probe_scratch.bridge_row, n) / denom
            } else {
                0.0
            }
        } else {
            0.0
        };
        first_token = argmax_of(&probe_scratch.logits);
        out_rows.push(LoopProbeRow {
            argmax: probe.argmax,
            margin: probe.margin,
            cos_alignment: probe.cos_alignment,
            cos_bridge,
        });
    }
    first_token
}

/// Strict `>` argmax → lowest index on ties (deterministic).
pub fn argmax_of(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
