use crate::types::{self, *};
use rayon::prelude::*;

// LoRA-Still compaction wiring (Plan 267 T7). Proposal 041 T1.1: the
// `#[path = "../transformer_still.rs"]` child-module declaration moved OUT
// of this file — `transformer_still.rs` is a lora_still consumer that STAYS
// in riir-engine (it imports `crate::lora_still`, an engine-local module,
// so it cannot cross the crate line). riir-engine now declares it top-level
// under the same `lora_still` gate; its imports of `crate::transformer` /
// `crate::types` resolve through the same-path re-exports.

// Submodule split (Plan 302): each forward-variant family lives in its
// own file. Items are `pub use`-d here to preserve the historical
// `crate::transformer::*` / `riir_engine::transformer::*` import paths.
// Issue 741 T10 Phase A: widened `pub` — the training halves (gemma2_train /
// gemma4_train / train_shared) relocated to riir-train-engine and consume
// `attention_heads_parallel` cross-crate from there. D4 drift-row reversible.
pub mod attention;
#[cfg(feature = "dllm")]
mod dllm;
mod gemma2;
// Gemma 4 unified text model — sliding+full attention baseline loader
// (Issue 577). Opt-in via `gemma4_inference` feature.
#[cfg(feature = "gemma4_inference")]
pub mod gemma4;

// Issue 883 P0 (katgpt-rs) — the V/K calibration lane: gemma-2 f16
// forward + taps feeding the shared fitted-anchor-table substrate.
// Opt-in (`vk_calibration`).
#[cfg(feature = "vk_calibration")]
pub mod gemma2_calibration;
// Plan 320 Phase C1: Gemma 4 LoRA forward wiring (weight-delta application
// at the 7 matmul insertion points). Feature-gated — opt-in.
#[cfg(feature = "gemma4_lora")]
pub mod gemma4_lora;

// Plan 410 Phase 1: Gemma 2 LoRA forward wiring (weight-delta application
// at the 7 matmul insertion points). Feature-gated — opt-in.
#[cfg(feature = "gemma_lora")]
pub mod gemma2_lora;

mod llama;
// Plan 333 T2.2: ternary-weight forward pass. Same shape as `llama`, with the
// 7 projections running through the group-scale ternary kernel.
mod mtp;
mod prefill;
#[cfg(feature = "raven")]
mod raven;
#[cfg(feature = "ternary_inference")]
mod ternary;

pub use attention::attention_head;
// The sole consumer is `causal_validation::gemma2`, which is itself gated
// `not(target_arch = "wasm32")` (depends on the SentencePiece C++ tokenizer).
// Match that gate here so the re-export isn't dead on wasm32.
// Proposal 041 T1.1: widened `pub` (was pub(crate)) — causal_validation lives
// in riir-engine and consumes these cross-crate through the re-export.
#[cfg(all(not(target_arch = "wasm32"), feature = "causal_validation"))]
pub use attention::attention_head_softcap;
pub use attention::attention_heads_parallel;
// Set-causal attention head (Research 376 Phase 0 T0.3, 2026-07-04).
#[cfg(feature = "set_diffusion")]
pub use attention::attention_head_set_causal;
#[cfg(feature = "dllm")]
pub use attention::masked_cross_entropy;
// block_causal_t_n is always available — forward_block_causal (which uses it)
// is always compiled (gated only at the re-export level in mod.rs).
pub use attention::block_causal_t_n;
pub use gemma2::{
    AttnLayerFeatures, Gemma2ForwardTrace, forward_gemma2, forward_gemma2_attn_capture,
    forward_gemma2_f16, forward_gemma2_trace, forward_gemma2_with_embedding, generate_gemma2,
    generate_gemma2_f16,
};
// Plan 410 Phase 1: LoRA-aware Gemma 2 forward entry point.
#[cfg(feature = "gemma_lora")]
pub use gemma2::{forward_gemma2_with_embedding_lora, forward_gemma2_with_lora};
// Issue 395: shared layer loop + hook trait (pub(crate) — internal to riir-engine).
// Plan 410: NoLora is always-on (all forward callers pass it). LoraApplier is
// only consumed by the feature-gated gemma2_lora module.
// Gated behind causal_validation: the only consumers are causal_validation
// and latent_steering_bridge (which itself depends on causal_validation).
// Issue 741 T10 Phase A: widened `pub` — the relocated riir-train-engine
// gemma2_train calls these trait methods on GemmaLora cross-crate (method
// resolution needs the trait nameable). D4 drift-row reversible.
#[cfg(feature = "gemma_lora")]
pub use gemma2::LoraApplier;
#[cfg(feature = "dllm")]
pub use gemma2::forward_gemma2_block_causal;
// Same `not(target_arch = "wasm32")` gate as the `causal_validation` module
// itself (lib.rs) — the only consumers live there and don't compile on wasm32.
// `PostLayerHook` is `pub` since Issue 673 Phase 2 (the Recirculation PoC's
// capture+mix hook — see gemma2.rs); the other two stay crate-internal.
#[cfg(all(not(target_arch = "wasm32"), feature = "causal_validation"))]
pub use gemma2::PostLayerHook;
#[cfg(all(not(target_arch = "wasm32"), feature = "causal_validation"))]
pub use gemma2::{NoLora, forward_gemma2_layers};
pub use llama::{forward_llama, forward_llama_attn_capture, generate_llama};
pub use mtp::{
    MtpProjection, cluster_map_from_embeddings, cluster_map_round_robin, load_mtp_projection,
    project_target_activation, select_topk_indices_into_buf,
};
#[cfg(feature = "ternary_inference")]
pub use ternary::{forward_ternary, generate_ternary};
// Issue 019 Phase B.3: `select_topk_indices_into` (no `_buf` suffix) is the
// historical riir-engine name. The canonical katgpt-forward name is
// `select_topk_indices_into_buf`; the bare-`_into` form is kept as a
// deprecated delegation alias (defined in `mtp.rs`) so any stray downstream
// caller keeps compiling while it migrates. Silence the deprecation warning
// on this re-export until all call sites move to the `_buf` name.
#[allow(deprecated)]
pub use mtp::select_topk_indices_into;
// Deprecated allocating variant — same pattern as above.
#[allow(deprecated)]
pub use mtp::select_topk_indices;
pub use mtp::{clustered_lm_head, standard_lm_head};
#[cfg(feature = "domain_latent")]
pub use prefill::generate_with_prefill_and_domain_latent;
pub use prefill::{forward_prefill, generate_with_prefill};
// Issue 019 F2 / Plan 406 Phase 2 T2.3: `RavenKVCache` adopted from
// katgpt-transformer (canonical superset). Forward path + router/readout/
// update kernels stay local in `raven.rs`.
#[cfg(feature = "set_diffusion")]
pub use dllm::forward_set_causal;
#[cfg(feature = "dllm")]
pub use dllm::{forward_bidirectional, forward_block_causal};
#[cfg(feature = "raven")]
pub use katgpt_transformer::RavenKVCache;
#[cfg(feature = "raven")]
pub use raven::{
    forward_raven, raven_compute_router, raven_compute_router_into, raven_readout,
    raven_readout_into, raven_update,
};

// Issue 019 F2 / Plan 406 Phase 1+2: KV-cache substrate types de-forked
// to katgpt-transformer. Local forward paths + ForwardContext stay.
// `DflashCache` impl moved to a `CacheAdapter` borrowing wrapper (Issue 373)
// so `MultiLayerKVCache` can be the canonical foreign type without
// violating the orphan rule.
pub use katgpt_transformer::{
    KVCache, KVLayerSnapshot, KVSnapshot, LayerWeights, MultiLayerKVCache, PAGE_SIZE, PagedKVCache,
    PrefillContext, TransformerWeights, preload_kv_cache,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_prefill;

/// Minimum `hidden_dim` for QKV projections before using Rayon parallelism.
/// Below this threshold, serial matmul is faster due to Rayon overhead (~5μs).
/// Guideline: "rayon wins only when per-iteration work > 10μs or count > 1000" (Issue 053).
pub(super) const RAYON_QKV_THRESHOLD: usize = 1024;
/// Minimum `n_embd` before using Rayon for GeGLU/SwiGLU gate+up projections.
/// Same rationale as QKV threshold: avoid thread spawn overhead for small models.
pub(super) const RAYON_MLP_THRESHOLD: usize = 1024;
/// Chunked embedding load: `dst[i] = a[a_off + i] + b[b_off + i]` for `i in 0..n`.
///
/// Processes 4 elements per iteration to help LLVM auto-vectorize the addition.
/// Falls back to a scalar tail loop for the remaining 0..3 elements.
///
/// # Safety
/// Caller must ensure `a_off + n <= a.len()` and `b_off + n <= b.len()`
/// and `n <= dst.len()`.
#[inline(always)]
pub(super) unsafe fn load_embed_add(
    dst: &mut [f32],
    a: &[f32],
    a_off: usize,
    b: &[f32],
    b_off: usize,
    n: usize,
) {
    unsafe {
        let mut i = 0;
        let chunk_end = n & !3;
        while i < chunk_end {
            let ai = a_off + i;
            let bi = b_off + i;
            *dst.get_unchecked_mut(i) = *a.get_unchecked(ai) + *b.get_unchecked(bi);
            *dst.get_unchecked_mut(i + 1) = *a.get_unchecked(ai + 1) + *b.get_unchecked(bi + 1);
            *dst.get_unchecked_mut(i + 2) = *a.get_unchecked(ai + 2) + *b.get_unchecked(bi + 2);
            *dst.get_unchecked_mut(i + 3) = *a.get_unchecked(ai + 3) + *b.get_unchecked(bi + 3);
            i += 4;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(a_off + i) + *b.get_unchecked(b_off + i);
            i += 1;
        }
    }
}

/// Chunked embedding load: `dst[i] = a[a_off + i]` for `i in 0..n`.
///
/// 4-wide chunked copy for auto-vectorization.
///
/// # Safety
/// Caller must ensure `a_off + n <= a.len()` and `n <= dst.len()`.
#[inline(always)]
pub(super) unsafe fn load_embed(dst: &mut [f32], a: &[f32], a_off: usize, n: usize) {
    unsafe {
        let mut i = 0;
        let chunk_end = n & !3;
        while i < chunk_end {
            let ai = a_off + i;
            *dst.get_unchecked_mut(i) = *a.get_unchecked(ai);
            *dst.get_unchecked_mut(i + 1) = *a.get_unchecked(ai + 1);
            *dst.get_unchecked_mut(i + 2) = *a.get_unchecked(ai + 2);
            *dst.get_unchecked_mut(i + 3) = *a.get_unchecked(ai + 3);
            i += 4;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(a_off + i);
            i += 1;
        }
    }
}

/// Chunked embedding load: `dst[i] = a[a_off + i] * scale` for `i in 0..n`.
///
/// 4-wide chunked multiply for auto-vectorization.
///
/// # Safety
/// Caller must ensure `a_off + n <= a.len()` and `n <= dst.len()`.
#[inline(always)]
pub(super) unsafe fn load_embed_scale(
    dst: &mut [f32],
    a: &[f32],
    a_off: usize,
    n: usize,
    scale: f32,
) {
    unsafe {
        let mut i = 0;
        let chunk_end = n & !3;
        while i < chunk_end {
            let ai = a_off + i;
            *dst.get_unchecked_mut(i) = *a.get_unchecked(ai) * scale;
            *dst.get_unchecked_mut(i + 1) = *a.get_unchecked(ai + 1) * scale;
            *dst.get_unchecked_mut(i + 2) = *a.get_unchecked(ai + 2) * scale;
            *dst.get_unchecked_mut(i + 3) = *a.get_unchecked(ai + 3) * scale;
            i += 4;
        }
        while i < n {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(a_off + i) * scale;
            i += 1;
        }
    }
}
// Issue 374 / Plan 406 Phase 3 T3.1: `LayerWeights` and `TransformerWeights`
// adopted from `katgpt-transformer` (canonical). The local definitions were
// deleted — they were a strict subset of the canonical structs (same 6
// LayerWeights fields; same TransformerWeights fields minus the cfg-gated
// `delta_routing_*` pair, which is dead weight in the plain forward path but
// harmless). `#[derive(Clone)]` was pushed upstream (Issue 374 B1, katgpt-rs
// commit 222b34ba) so the CPU LoRA fallback trainer in `riir-train-gpu` can
// still fork frozen base weights via `.clone()`.
//
// RNG-consumption order is identical: the canonical `new()` initializes
// `delta_routing_query`/`delta_routing_norm` with literal 0.0/1.0 (no `rng`
// calls), so the same seed produces bit-identical shared-field weights.

/// Pre-allocated buffers for zero-alloc forward passes.
/// Create once, reuse across calls.
pub struct ForwardContext {
    pub x: Vec<f32>,        // [n_embd] main activation
    pub xr: Vec<f32>,       // [n_embd] residual
    pub xr2: Vec<f32>,      // [n_embd] residual 2
    pub q: Vec<f32>,        // [n_embd] query
    pub k: Vec<f32>,        // [kv_dim] key (kv_dim = n_kv_head * head_dim)
    pub v: Vec<f32>,        // [kv_dim] value
    pub attn_out: Vec<f32>, // [n_embd] attention output
    pub scores: Vec<f32>,   // [block_size] attention scores (max possible)
    // [n_head * block_size] per-head scores for parallel attention.
    // Issue 741 T10 Phase A: widened `pub` (was pub(crate)) — the relocated
    // riir-train-engine gemma2_train passes it alongside `&ctx.q` in one call
    // (split field borrows; an accessor cannot express that). D4 drift-row
    // reversible.
    pub head_scores: Vec<f32>,
    pub hidden: Vec<f32>,       // [mlp_hidden] MLP hidden
    pub gate: Vec<f32>,         // [mlp_hidden] GeGLU gate buffer (Plan 087: Gemma 2)
    pub up: Vec<f32>,           // [mlp_hidden] GeGLU up buffer (Plan 087: Gemma 2)
    pub logits: Vec<f32>,       // [vocab_size] output logits
    pub hidden_state: Vec<f32>, // [n_embd] final hidden state (Plan 009 compat)
    /// `LoRA` intermediate buffer `[lora_rank]`. Pre-allocated, zero alloc in hot path.
    pub lora_buf: Vec<f32>,
    // Sparse MLP buffers (Plan 022: TwELL-inspired unstructured sparsity)
    #[cfg(feature = "sparse_mlp")]
    pub active_indices: Vec<usize>, // [mlp_hidden] pre-allocated index buffer
    #[cfg(feature = "sparse_mlp")]
    pub active_values: Vec<f32>, // [mlp_hidden] pre-allocated value buffer
    // Paged KV cache: pre-allocated flat buffers for attention computation
    pub paged_flat_key: Vec<f32>,   // [block_size * kv_dim]
    pub paged_flat_value: Vec<f32>, // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    #[cfg(feature = "raven")]
    pub raven_query_buf: Vec<f32>, // [max(kv_dim, max_num_slots)]
    // Raven: pre-allocated readout buffers (Issue 020)
    #[cfg(feature = "raven")]
    pub raven_scores_buf: Vec<f32>, // [max_num_slots]
    #[cfg(feature = "raven")]
    pub raven_output_buf: Vec<f32>, // [max_kv_dim]
    // Clustered LM head: pre-allocated scratch buffers (Issues 021, 025)
    cluster_scores_buf: Vec<f32>,           // [max_num_clusters]
    cluster_indexed_buf: Vec<(usize, f32)>, // [max_num_clusters]
    cluster_selected_buf: Vec<usize>,       // [max_topk]
    // MTP Drafter: pre-allocated projection buffer [n_embd] for target activation conditioning (Plan 055)
    pub mtp_context_buf: Vec<f32>,
    // Pre-allocated HLA temp buffers (eliminates per-token allocation, Issue 004 P0-1)
    #[cfg(feature = "hla")]
    pub hla_tmp_k_cqv: Vec<f32>, // [head_dim]
    #[cfg(feature = "hla")]
    pub hla_tmp_u: Vec<f32>, // [head_dim]
    #[cfg(feature = "hla")]
    pub ahla_tmp_r: Vec<f32>, // [head_dim]
    // Role-aware HLA scratch: caller-owned buffer for role-transported keys,
    // replacing the per-head `Cow::Owned(Vec)` allocation (one Vec alloc per
    // Q head per token when a role is assigned). Sized to head_dim.
    //
    // Gated on `hla_role_aware` (not just `hla`) because only the role-aware
    // kernel path consumes it. Under `hla` alone the field would be dead —
    // warning under `cargo check --workspace` (default features).
    #[cfg(feature = "hla_role_aware")]
    pub hla_tmp_k_transport: Vec<f32>, // [head_dim]
    // Delta routing: block delta accumulation buffers (Plan 097)
    #[cfg(feature = "delta_routing")]
    pub block_deltas: Vec<Vec<f32>>, // [n_blocks][n_embd] accumulated deltas per block
    #[cfg(feature = "delta_routing")]
    pub delta_routing_logits: Vec<f32>, // [n_layer + 1] routing logits temp buffer
    #[cfg(feature = "delta_routing")]
    pub delta_qn_buf: Vec<f32>, // [n_embd] pre-allocated query·norm product buffer
    // Pre-computed RoPE frequency table (Issue 024)
    pub rope_freq_table: crate::rope::RopeFreqTable,
    // Pre-allocated CDF buffer for zero-alloc token sampling
    pub cdf_buf: Vec<f32>,
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        let block_kv = config.block_size * kvd;
        // Q and attention output buffers need n_head * head_dim elements.
        // For most models n_head * head_dim == n_embd, but some (e.g. MiniCPM5-1B)
        // have q_dim > n_embd (16*128=2048 vs n_embd=1536).
        let q_dim = config.n_head * config.head_dim;
        let buf_dim = q_dim.max(config.n_embd);
        // Gemma 4 full-attention layers have a larger head_dim + Q projection
        // (16 × 512 = 8192 vs sliding's 16 × 256 = 4096). Size the Q / attn_out
        // buffers for the worst case across the two layer kinds so the same
        // ForwardContext can drive both. K/V buffers also vary per layer; the
        // gemma4 forward pass writes K/V directly into the per-layer cache
        // (which has its own correctly-sized kv_dim), so the context's k/v
        // buffers only need to hold the largest single-layer projection.
        #[cfg(feature = "gemma4_inference")]
        let buf_dim = {
            let full_q_dim = config.n_head * config.global_head_dim;
            buf_dim.max(full_q_dim)
        };
        #[cfg(feature = "gemma4_inference")]
        let kvd = {
            let sliding_kvd = kvd;
            let full_kvd = config.n_global_kv_head * config.global_head_dim;
            sliding_kvd.max(full_kvd)
        };
        Self {
            x: vec![0.0; config.n_embd],
            xr: vec![0.0; config.n_embd],
            xr2: vec![0.0; config.n_embd],
            q: vec![0.0; buf_dim],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; buf_dim],
            scores: vec![0.0; config.block_size],
            head_scores: vec![0.0; config.n_head * config.block_size],
            hidden: vec![0.0; config.mlp_hidden],
            gate: vec![0.0; config.mlp_hidden],
            up: vec![0.0; config.mlp_hidden],
            logits: vec![0.0; config.vocab_size],
            hidden_state: vec![0.0; config.n_embd],
            lora_buf: vec![0.0; config.lora_rank],
            #[cfg(feature = "sparse_mlp")]
            active_indices: vec![0; config.mlp_hidden],
            #[cfg(feature = "sparse_mlp")]
            active_values: vec![0.0; config.mlp_hidden],
            paged_flat_key: vec![0.0; block_kv],
            paged_flat_value: vec![0.0; block_kv],
            #[cfg(feature = "raven")]
            raven_query_buf: vec![0.0; kvd.max(64)], // max(kv_dim, max_num_slots) — Issue 022
            #[cfg(feature = "raven")]
            raven_scores_buf: vec![0.0; 64], // max_num_slots
            #[cfg(feature = "raven")]
            raven_output_buf: vec![0.0; kvd], // kv_dim
            cluster_scores_buf: vec![0.0; 64], // max_num_clusters (reasonable default)
            cluster_indexed_buf: vec![(0usize, 0.0f32); 64], // max_num_clusters — pre-filled to avoid first-use alloc
            cluster_selected_buf: vec![0; 4], // max_topk — pre-filled to avoid first-use alloc
            mtp_context_buf: vec![0.0; config.n_embd],
            #[cfg(feature = "hla")]
            hla_tmp_k_cqv: vec![0.0; config.head_dim],
            #[cfg(feature = "hla")]
            hla_tmp_u: vec![0.0; config.head_dim],
            #[cfg(feature = "hla")]
            ahla_tmp_r: vec![0.0; config.head_dim],
            #[cfg(feature = "hla_role_aware")]
            hla_tmp_k_transport: vec![0.0; config.head_dim],
            #[cfg(feature = "delta_routing")]
            block_deltas: {
                let block_size = 4; // Default B=4
                let n_blocks = config.n_layer.div_ceil(block_size);
                (0..n_blocks).map(|_| vec![0.0; config.n_embd]).collect()
            },
            #[cfg(feature = "delta_routing")]
            delta_routing_logits: vec![0.0; config.n_layer + 1],
            #[cfg(feature = "delta_routing")]
            delta_qn_buf: vec![0.0; config.n_embd],
            // Pre-allocated CDF buffer for sample_token_into — avoids vocab_size allocation per token
            cdf_buf: vec![0.0; config.vocab_size],
            rope_freq_table: crate::rope::RopeFreqTable::new(config.rope_theta, config.head_dim),
        }
    }
}

// ---------------------------------------------------------------------------
// Delta routing: softmax over delta sources, additive to residual (Plan 097)
// ---------------------------------------------------------------------------

/// Depth routing: cross-layer delta aggregation via softmax-weighted additive sum.
///
/// `depth_route(residual`, sources, query, norm):
///   V = stack(sources)            // [N, D]
///   K = norm(V)                    // `RMSNorm`
///   logits = `dot(query_weight`, K)  // per-source score
///   weights = softmax(logits)      // routing weights
///   residual += `weighted_sum(weights`, V)  // additive
///
/// Issue 053: takes `block_deltas` slice directly with index range to avoid
/// per-token `Vec<&[f32]>` allocation at call sites.
#[cfg(feature = "delta_routing")]
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn depth_route(
    residual: &mut [f32],
    block_deltas: &[Vec<f32>], // [n_blocks][n_embd] all delta buffers
    source_range: std::ops::RangeInclusive<usize>, // indices [0..=block_idx]
    query_weight: &[f32],      // [n_embd] per-layer query
    norm_weight: &[f32],       // [n_embd] RMSNorm gamma
    logits_buf: &mut [f32],    // [N] temp buffer
    qn_buf: &mut [f32],        // [n_embd] pre-allocated query·norm product buffer
    n_embd: usize,
) {
    // Clamp the source range up front so the inner loops can iterate
    // branch-free. `RangeInclusive` is not `Copy`, so without this we'd be
    // forced to either `.clone()` it three times (as the previous version did)
    // or re-check `src_idx >= block_deltas.len()` on every iteration.
    let start = *source_range.start();
    let end = (*source_range.end()).min(block_deltas.len().saturating_sub(1));
    if end < start {
        return;
    }
    let n_sources = end - start + 1;

    // 1. RMSNorm each source and compute dot product with query
    let eps = 1e-5f32;
    let mut max_logit = f32::NEG_INFINITY;

    // Pre-compute query·norm product once (avoids redundant multiply per source)
    for d in 0..n_embd {
        qn_buf[d] = query_weight[d] * norm_weight[d];
    }

    for i in 0..n_sources {
        let src_idx = start + i;
        let src = &block_deltas[src_idx];

        // RMSNorm using SIMD self-dot (O(n/8) vs scalar O(n))
        let sum_sq = crate::simd::simd_dot_f32(src, src, n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Fused dot: qn_buf · src * inv_rms (qn_buf = query_weight * norm_weight, pre-computed)
        // Process 4 elements at a time to help LLVM auto-vectorize the fused multiply-accumulate.
        let scale = inv_rms;
        let mut logit = 0.0f32;
        let chunks = n_embd / 4;
        for c in 0..chunks {
            let d = c * 4;
            let qn0 = qn_buf[d];
            let qn1 = qn_buf[d + 1];
            let qn2 = qn_buf[d + 2];
            let qn3 = qn_buf[d + 3];
            logit += (qn0 * src[d] + qn1 * src[d + 1]) * scale;
            logit += (qn2 * src[d + 2] + qn3 * src[d + 3]) * scale;
        }
        for d in (chunks * 4)..n_embd {
            logit += qn_buf[d] * src[d] * scale;
        }

        logits_buf[i] = logit;
        if logit > max_logit {
            max_logit = logit;
        }
    }

    // 2. Softmax (numerically stable)
    let mut sum_exp = 0.0f32;
    for i in 0..n_sources {
        let exp_val = (logits_buf[i] - max_logit).exp();
        logits_buf[i] = exp_val;
        sum_exp += exp_val;
    }
    let inv_sum = 1.0 / sum_exp;

    // 3. Weighted sum of sources, added to residual (additive routing)
    // Process 4 elements at a time to help LLVM auto-vectorize.
    for i in 0..n_sources {
        let src_idx = start + i;
        let w = logits_buf[i] * inv_sum;
        let src = &block_deltas[src_idx];
        let chunks = n_embd / 4;
        let remainder = n_embd % 4;
        for c in 0..chunks {
            let d = c * 4;
            residual[d] += w * src[d];
            residual[d + 1] += w * src[d + 1];
            residual[d + 2] += w * src[d + 2];
            residual[d + 3] += w * src[d + 3];
        }
        for d in (chunks * 4)..(chunks * 4 + remainder) {
            residual[d] += w * src[d];
        }
    }
}

/// Apply the per-sublayer delta-routing step used by Gemma 2 / `LLaMA` forwards.
///
/// Accumulates `x - xr` (current hidden state minus pre-layer residual) into
/// `ctx.block_deltas[block_idx]`, and at every block boundary (every 4 layers)
/// optionally routes accumulated deltas via [`depth_route`] into the residual.
/// After routing (or when no routing weights are provided, as for `LLaMA`), the
/// block's delta buffer is cleared.
///
/// Pass `Some((query_weight, norm_weight))` for Gemma 2 weights that ship
/// `delta_routing_query` / `delta_routing_norm` matrices. Pass `None` for
/// LLaMA-family weights that lack them -- accumulation still happens so the
/// buffers don't grow unbounded, but no cross-layer routing is performed.
#[cfg(feature = "delta_routing")]
#[inline(always)]
pub(super) fn apply_delta_routing_step(
    ctx: &mut ForwardContext,
    layer_idx: usize,
    n_embd: usize,
    routing_weights: Option<(&[f32], &[f32])>,
) {
    const BLOCK_SIZE: usize = 4;
    let block_idx = layer_idx / BLOCK_SIZE;
    let pos_in_block = layer_idx % BLOCK_SIZE;

    // Accumulate delta: current x minus pre-layer residual.
    // The bounds check (`block_idx < block_deltas.len()`) is loop-invariant and
    // was hoisted out of the per-d loop so it doesn't inhibit auto-vectorization
    // of the accumulation.
    if block_idx < ctx.block_deltas.len() {
        let bd = &mut ctx.block_deltas[block_idx];
        for ((bd, &x), &xr) in bd.iter_mut().zip(&ctx.x).zip(&ctx.xr).take(n_embd) {
            *bd += x - xr;
        }
    }

    // At block boundary: route accumulated deltas from all completed blocks
    if pos_in_block == BLOCK_SIZE - 1 && block_idx < ctx.block_deltas.len() {
        if let Some((query, norm)) = routing_weights {
            depth_route(
                &mut ctx.x[..n_embd],
                &ctx.block_deltas,
                0..=block_idx,
                query,
                norm,
                &mut ctx.delta_routing_logits,
                &mut ctx.delta_qn_buf,
                n_embd,
            );
        }
        ctx.block_deltas[block_idx].fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// PrefillContext — Pre-allocated buffers for bidirectional prefill (Plan 025)
// ---------------------------------------------------------------------------

// Issue 375 / Plan 406 Phase 3 T3.2: `PrefillContext` adopted from
// `katgpt-transformer` (canonical). The local definition was deleted — it
// used `normed_x` (double-norm cache); the canonical uses fused
// `queries` + `residuals` (pre-computed Q projection + attention residual
// from Phase A, reused in Phase B). The fused path is strictly better:
// bit-identical result, 1 fewer `rmsnorm` per position per layer (the Q
// matmul moves from Phase B to Phase A, and the residual `xr = rmsnorm(hidden)`
// is saved once instead of recomputed).

/// Causal decode: single token forward with optional `LoRA` adapter.
/// Backward-compatible wrapper that passes `None` for `LoRA`.
#[inline(always)]
pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    forward_with_lora(ctx, weights, cache, token, pos, config, None)
}

/// Causal decode: single token forward with the caller's `LoRA` adapter
/// EXPLICITLY threaded (riir-ai Issue 938: the caller that owns BOTH the
/// adapter and the KV cache is the weight-epoch seam — it must stamp/compare
/// [`crate::types::WeightEpoch`] before appending to a non-empty cache;
/// this function does no epoch checking itself).
///
/// Only `ModelArchitecture::Generic` takes the `forward_base` path — the
/// arch-specific weight types panic here exactly as in [`forward`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_with_lora<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
) -> &'a mut [f32] {
    match config.model_arch {
        ModelArchitecture::Generic => {
            #[cfg(not(feature = "domain_latent"))]
            {
                forward_base(ctx, weights, cache, token, pos, config, lora)
            }
            #[cfg(feature = "domain_latent")]
            {
                forward_base(ctx, weights, cache, token, pos, config, lora, None)
            }
        }
        ModelArchitecture::Gemma2 => {
            panic!("Use forward_gemma2 with GemmaTransformerWeights for Gemma2 architecture");
        }
        #[cfg(feature = "gemma4_inference")]
        ModelArchitecture::Gemma4 => {
            panic!("Use forward_gemma4 with Gemma4TransformerWeights for Gemma4 architecture");
        }
        ModelArchitecture::Llama => {
            panic!("Use forward_llama with LlamaTransformerWeights for Llama architecture");
        }
        #[cfg(feature = "deltanet_inference")]
        ModelArchitecture::QwenDeltaNet => {
            panic!(
                "Use forward_qwen_deltanet with QwenDeltaNetWeights for QwenDeltaNet architecture"
            );
        }
        #[cfg(feature = "ternary_inference")]
        ModelArchitecture::Ternary => {
            panic!("Use forward_ternary with TernaryTransformerWeights for Ternary architecture");
        }
    }
}

/// Forward with optional `LoRA` and domain latent (Plan 038).
/// Convenience wrapper for callers that need both conditioning signals.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn forward_with_domain_latent<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    forward_base(ctx, weights, cache, token, pos, config, lora, domain_latent)
}

/// Internal forward with optional `LoRA` and domain latent (writer `LoRA` during decode).
/// Zero-alloc forward pass. Writes logits into `ctx.logits` and returns &mut to it.
/// Multi-layer: `RMSNorm` → Attn → Res → `RMSNorm` → MLP → Res per layer, then LM Head.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_base<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let attn_scale = 1.0 / (hd as f32).sqrt();

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    unsafe {
        load_embed_add(
            &mut ctx.x,
            &weights.wte,
            tok_off,
            &weights.wpe,
            pos_off_emb,
            n,
        );
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            for i in 0..kvd {
                unsafe {
                    *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                }
            }
        }

        // Store K,V in per-layer cache (kv_dim elements per position)
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

        // Multi-head attention with GQA (Issue 042: parallel for long sequences)
        ctx.attn_out[..n].fill(0.0);
        let t_n = pos + 1;

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
                attn_scale,
                0.0,
                config.block_size,
            );
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.up,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: clustered when vocab >= threshold AND cluster weights present
    if config.vocab_size >= config.mtp_cluster_vocab_threshold
        && let Some(classifier) = weights.mtp_cluster_classifier.as_ref()
        && let Some(cluster_map) = weights.mtp_cluster_map.as_ref()
    {
        clustered_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            classifier,
            cluster_map,
            config.vocab_size,
            n,
            config.mtp_cluster_topk,
            &mut ctx.cluster_scores_buf,
            &mut ctx.cluster_indexed_buf,
            &mut ctx.cluster_selected_buf,
        );
    } else {
        standard_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            config.vocab_size,
            n,
        );
    }

    &mut ctx.logits
}

/// Forward pass using `PagedKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` but stores KV in paged memory,
/// enabling copy-on-write fork for `DDTree` branch exploration.
/// Builds a temporary flat KV buffer per layer for attention computation.
#[inline(always)]
pub fn forward_paged<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    paged_cache: &mut PagedKVCache,
    seq_idx: usize,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let attn_scale = 1.0 / (hd as f32).sqrt();

    // Ensure pages allocated for this sequence up to pos
    paged_cache.ensure_pages(seq_idx, pos);

    // Flat KV cache for attention computation (pre-allocated, reused from ForwardContext)
    let t_n = pos + 1;
    let flat_kv_len = t_n * kvd;
    let flat_key = &mut ctx.paged_flat_key[..flat_kv_len];
    let flat_value = &mut ctx.paged_flat_value[..flat_kv_len];
    flat_key.fill(0.0);
    flat_value.fill(0.0);

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    unsafe {
        load_embed_add(
            &mut ctx.x,
            &weights.wte,
            tok_off,
            &weights.wpe,
            pos_off_emb,
            n,
        );
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Write K,V to paged cache
        paged_cache.write_kv(layer_idx, seq_idx, pos, &ctx.k, &ctx.v);

        // Build flat KV from paged cache for attention
        for t in 0..t_n {
            let k_slice = &mut flat_key[t * kvd..(t + 1) * kvd];
            let v_slice = &mut flat_value[t * kvd..(t + 1) * kvd];
            paged_cache.read_kv(layer_idx, seq_idx, t, k_slice, v_slice);
        }

        // Multi-head attention with GQA (reuse existing attention_head)
        ctx.attn_out[..n].fill(0.0);

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                attention_head(
                    &ctx.q,
                    flat_key,
                    flat_value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    attn_scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.up,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Zero-alloc generation: `ctx`, `cache`, `tokens` all provided by caller.
///
/// `tokens` is cleared and filled with generated token ids.
/// `ctx` and `cache` are reused across calls.
pub fn generate_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    let mut token = config.bos_token;
    let mut pos = 0;

    for _ in 0..n_tokens {
        if pos >= config.block_size {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        }

        let logits = forward(ctx, weights, cache, token, pos, config);

        softmax_scaled(logits, 1.0 / config.temperature);

        // Reuse pre-allocated CDF buffer — avoids vocab_size allocation per token.
        // Re-borrow ctx.logits directly to avoid conflicting with cdf_buf borrow.
        let next_token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        tokens.push(next_token);

        if next_token == config.bos_token {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        } else {
            token = next_token;
            pos += 1;
        }
    }
}

/// Generate tokens autoregressively. Returns generated token ids.
pub fn generate(
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
) -> Vec<usize> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    // Pre-allocate to expected output size — generate_into pushes n_tokens.
    let mut tokens = Vec::with_capacity(n_tokens);
    generate_into(
        &mut ctx,
        &mut cache,
        weights,
        config,
        rng,
        n_tokens,
        &mut tokens,
    );
    tokens
}

/// Generate multiple samples in parallel using rayon.
///
/// Each sample gets its own `ForwardContext` + `MultiLayerKVCache` via `map_init`,
/// so there's no contention. The `seeds` slice provides one seed per sample.
/// Returns `Vec<Vec<usize>>` with one token sequence per sample.
pub fn generate_batch(
    weights: &TransformerWeights,
    config: &Config,
    seeds: &[u64],
    n_tokens: usize,
) -> Vec<Vec<usize>> {
    seeds
        .par_iter()
        .map_init(
            || (ForwardContext::new(config), MultiLayerKVCache::new(config)),
            |(ctx, cache), &seed| {
                let mut rng = Rng::new(seed);
                let mut tokens = Vec::with_capacity(n_tokens);
                generate_into(ctx, cache, weights, config, &mut rng, n_tokens, &mut tokens);
                tokens
            },
        )
        .collect()
}

/// Convert token ids to readable characters (a-z, _ for BOS).
pub fn tokens_to_string(tokens: &[usize]) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    tokens
        .iter()
        .map(|&t| if t < 26 { CHARS[t] as char } else { '_' })
        .collect()
}

/// Forward pass using `TurboQuant` compressed KV cache (Plan 043).
///
/// Mirrors `forward_base` but stores K/V into a compressed cache and
/// dequantizes on-the-fly during attention scoring. The rest of the
/// transformer (embedding, QKV projection, MLP, LM head) is unchanged.
///
/// **Trade-off**: ~8× KV cache memory savings at the cost of dequantization
/// overhead during attention. Best for long sequences where cache memory
/// dominates.
#[cfg(feature = "turboquant")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_turboquant<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut crate::turboquant::TurboQuantKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let attn_scale = 1.0 / (hd as f32).sqrt();

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    unsafe {
        load_embed_add(
            &mut ctx.x,
            &weights.wte,
            tok_off,
            &weights.wpe,
            pos_off_emb,
            n,
        );
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Store compressed K,V
        cache.store_key(layer_idx, pos, &ctx.k[..kvd]);
        cache.store_value(layer_idx, pos, &ctx.v[..kvd]);

        // Dequantize only the NEW position (Issue 043: incremental, not O(pos))
        // Previous positions were already dequantized on prior calls and persist in the flat buffers.
        cache.dequantize_key_into(
            layer_idx,
            pos,
            &mut ctx.paged_flat_key[pos * kvd..(pos + 1) * kvd],
        );
        cache.dequantize_value_into(
            layer_idx,
            pos,
            &mut ctx.paged_flat_value[pos * kvd..(pos + 1) * kvd],
        );
        let t_n = pos + 1;

        // Multi-head attention with GQA (Issue 042: parallel for long sequences)
        ctx.attn_out[..n].fill(0.0);

        unsafe {
            attention_heads_parallel(
                &ctx.q,
                &ctx.paged_flat_key,
                &ctx.paged_flat_value,
                &mut ctx.attn_out,
                &mut ctx.head_scores,
                config.n_head,
                n_kv,
                kvd,
                hd,
                t_n,
                attn_scale,
                0.0,
                config.block_size,
            );
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.up,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::swiglu_inplace(&mut ctx.hidden, &ctx.up);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}
