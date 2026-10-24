//! `FlashMemory` periodic sparse attention for Bonsai/Qwen3.5 GQA layers.
//!
//! Issue 584 Phase 2 (2026-08-14). Implements the FlashMemory-DeepSeek-V4
//! periodic sigmoid-threshold sparse selection (arXiv:2606.09079, Research 436)
//! for standard GQA attention layers — the attention-layer variant found in
//! Bonsai-27B's hybrid DeltaNet/Attention architecture.
//!
//! # Why a separate module
//!
//! Phase 1 landed `FlashMemory` for MLA layers (Kimi-K3) in katgpt-attn
//! (`mla_forward_token_flashmemory`). Bonsai-27B uses GQA (not MLA) in its
//! attention layers, so the MLA forward doesn't apply. The GQA block cache +
//! selector substrate ships in katgpt-attn (`GqaFlashMemoryBlockCache` +
//! `GqaFlashMemorySelector`); this module provides the riir-engine forward
//! integration that wires those into Bonsai's `forward_attention_layer` path.
//!
//! # Sparse forward flow
//!
//! 1. Q/K/V projections + QK-norm + `RoPE` — identical to dense `forward_attention_layer`.
//! 2. Store K, V in cache — identical to dense.
//! 3. **Rebuild GQA block centroids** from the key cache (mean per KV head per block).
//! 4. **Select blocks** via `GqaFlashMemorySelector` (sigmoid threshold, periodic refresh).
//! 5. **Sparse attention**: for each query head, attend only to tokens in blocks
//!    selected for its KV group. Softmax is computed over the selected tokens only.
//! 6. Output gating + output projection — identical to dense.
//!
//! # Alloc-free steady state (G4)
//!
//! The sparse forward reuses the existing `AttentionLayerScratch` buffers. The
//! block selection uses the pre-allocated `PerHeadSelection` inside the selector.
//! No heap allocation occurs during steady-state decode (between block-cache rebuilds).

#[cfg(feature = "flashmemory_trained_indexer")]
use katgpt_attn::dash_attn::flashmemory_sparse::DualEncoderIndexer;
use katgpt_attn::dash_attn::flashmemory_sparse::{
    GqaFlashMemoryBlockCache, GqaFlashMemorySelector, PerHeadSelection,
};
use katgpt_core::simd::{fast_sigmoid, simd_dot_f32};
use katgpt_transformer::KVCache;

use crate::deltanet::forward::{AttentionLayerScratch, effective_rotary_dim};
use crate::deltanet::weights::DeltaNetLayerWeights;
use crate::rope::RopeFreqTable;
use crate::types::{self, Config};

/// Block-selection strategy seam for the GQA sparse forward.
///
/// Implemented by the modelless [`GqaFlashMemorySelector`] (sigmoid over
/// dot-product — the production default) and, under the
/// `flashmemory_trained_indexer` feature, by [`TrainedIndexerSelector`]
/// (a `DualEncoderIndexer` checkpoint served through `select_gqa`). Static
/// dispatch — the forward monomorphizes per selector with zero dyn cost.
pub trait GqaBlockSelector {
    /// Select blocks per KV head at KV-head resolution.
    ///
    /// * `query_keys` — `[n_kv_head * head_dim]`, the block-contiguous
    ///   group-mean query (see [`build_kv_resolution_query`]).
    /// * `attn_scale` — `1/sqrt(head_dim)`; modelless selectors scale the
    ///   raw dot by it. Trained encoders learn their own scale and ignore it.
    fn select_blocks(
        &mut self,
        query_keys: &[f32],
        block_cache: &GqaFlashMemoryBlockCache,
        attn_scale: f32,
        step: usize,
    ) -> &PerHeadSelection;

    /// Invalidate the cached selection so the next [`select_blocks`](Self::select_blocks)
    /// re-ranks. The decode lane calls this when a NEW block opens (Issue
    /// 1117's block-open law): the budget rule forces the CURRENT block in
    /// at refresh time, so a refresh at the open puts the fresh block into
    /// the selection — without it, tokens landing in the new block before
    /// the next periodic refresh would be stranded outside the selection.
    /// Default: no-op (a stateless selector).
    fn force_refresh(&mut self) {}
}

impl GqaBlockSelector for GqaFlashMemorySelector {
    fn select_blocks(
        &mut self,
        query_keys: &[f32],
        block_cache: &GqaFlashMemoryBlockCache,
        attn_scale: f32,
        step: usize,
    ) -> &PerHeadSelection {
        self.select(query_keys, block_cache, attn_scale, step)
    }

    fn force_refresh(&mut self) {
        // Spell the inherent method — `self.force_refresh()` would recurse
        // into this trait impl.
        GqaFlashMemorySelector::force_refresh(self);
    }
}

/// Trained-indexer adapter for the GQA sparse forward (Plan 337 D1 serving
/// path, Issue 452 GAP 2).
///
/// Wraps a [`DualEncoderIndexer`] loaded from a riir-train checkpoint
/// (`to_bytes` format — the `.ckpt` files produced by `plan337_train_indexer`).
/// Selection goes through `select_gqa` — the raw per-KV-head centroid feature
/// space the checkpoints were trained on. Opt-in via the
/// `flashmemory_trained_indexer` feature, never default (modelless-first:
/// the modelless `GqaFlashMemorySelector` remains the production default).
#[cfg(feature = "flashmemory_trained_indexer")]
pub struct TrainedIndexerSelector {
    /// The wrapped trained indexer. Public for construction via
    /// `DualEncoderIndexer::from_weights` / `from_bytes`.
    pub indexer: DualEncoderIndexer,
}

#[cfg(feature = "flashmemory_trained_indexer")]
impl TrainedIndexerSelector {
    /// Load from a checkpoint's `to_bytes()` buffer.
    ///
    /// * `fm_config` — serving config (threshold here is the trained
    ///   selector's operating point, e.g. the D3-gated value).
    /// * `n_kv_head` — number of KV heads (selection is per KV head).
    /// * `max_blocks` — capacity; ≥ the maximum blocks the forward will see.
    pub fn from_ckpt_bytes(
        bytes: &[u8],
        fm_config: katgpt_attn::dash_attn::flashmemory_sparse::FlashMemoryConfig,
        n_kv_head: usize,
        max_blocks: usize,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            indexer: DualEncoderIndexer::from_bytes(fm_config, n_kv_head, max_blocks, bytes)?,
        })
    }
}

#[cfg(feature = "flashmemory_trained_indexer")]
impl GqaBlockSelector for TrainedIndexerSelector {
    fn select_blocks(
        &mut self,
        query_keys: &[f32],
        block_cache: &GqaFlashMemoryBlockCache,
        _attn_scale: f32,
        step: usize,
    ) -> &PerHeadSelection {
        // No attn_scale: the trained encoders learn their own scale (same
        // contract as DualEncoderIndexer::select).
        self.indexer.select_gqa(query_keys, block_cache, step)
    }

    fn force_refresh(&mut self) {
        self.indexer.force_refresh();
    }
}

/// Sparse GQA attention forward with `FlashMemory` block selection.
///
/// Drop-in replacement for `forward_attention_layer` that restricts attention
/// to FlashMemory-selected blocks. The Q/K/V projection, QK-norm, `RoPE`, cache
/// append, output gating, and output projection are identical to the dense
/// path — only the attention scoring + value aggregation steps are sparsified.
///
/// # Arguments
///
/// * `x` — Input activation `[n_embd]`, modified in-place with layer output.
/// * `layer` — Layer weights (`attn_wq`, `attn_wk`, `attn_wv`, `attn_wo`, norms, gate).
/// * `cache` — GQA KV cache (`key` + `value`, each `[block_size * kv_dim]`).
/// * `pos` — Current position in sequence.
/// * `config` — Model config.
/// * `rope_freq` — Pre-computed `RoPE` frequency table.
/// * `scratch` — Pre-allocated attention scratch buffers.
/// * `block_cache` — GQA block centroid cache (rebuilt each call).
/// * `selector` — GQA `FlashMemory` selector (periodic refresh). Generic over
///   [`GqaBlockSelector`]: the modelless `GqaFlashMemorySelector` by default,
///   or [`TrainedIndexerSelector`] under `flashmemory_trained_indexer`.
/// * `step` — Decode step (for refresh scheduling; typically equals `pos`).
#[allow(clippy::too_many_arguments)]
pub fn forward_attention_layer_flashmemory<S: GqaBlockSelector>(
    x: &mut [f32],
    layer: &DeltaNetLayerWeights,
    cache: &mut KVCache,
    pos: usize,
    config: &Config,
    rope_freq: &RopeFreqTable,
    scratch: &mut AttentionLayerScratch,
    block_cache: &mut GqaFlashMemoryBlockCache,
    selector: &mut S,
    step: usize,
) {
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = effective_rotary_dim(config);

    // ── 1-4: Q/K/V projections + QK-norm + RoPE (identical to dense) ───────
    layer.attn_wq.matvec(&x[..n_embd], &mut scratch.qg_buf);
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        scratch.q_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src..src + hd]);
        scratch.gate_buf[dst..dst + hd].copy_from_slice(&scratch.qg_buf[src + hd..src + 2 * hd]);
    }
    layer.attn_wk.matvec(&x[..n_embd], &mut scratch.k_buf);
    layer.attn_wv.matvec(&x[..n_embd], &mut scratch.v_buf);

    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        types::rmsnorm_with_gamma_eps(&mut scratch.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        types::rmsnorm_with_gamma_eps(&mut scratch.k_buf[off..off + hd], &layer.attn_k_norm, eps);
    }

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

    // ── 5: Store K, V in cache (identical to dense) ────────────────────────
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

    // ── 6-8: refresh → select → sparse attention (shared core, Issue 1117) ─
    // The centroid refresh, the KV-resolution query, the selection, and the
    // per-head sparse accumulation live in ONE home (refresh_and_select +
    // sparse_attention_accumulate) shared with the ternary decode lane — the
    // Rebuild posture here is this research forward's byte-identical shape.
    let seq_len = pos + 1;
    let scale = 1.0 / (hd as f32).sqrt();
    build_kv_resolution_query(&scratch.q_buf, n_head, n_kv, hd, &mut scratch.qg_buf[..kvd]);
    let mut last_n_blocks_scratch = 0usize;
    let selection = refresh_and_select(
        &cache.key,
        block_cache,
        selector,
        &scratch.qg_buf[..kvd],
        scale,
        seq_len,
        step,
        CacheUpdate::Rebuild,
        &mut last_n_blocks_scratch,
    );
    {
        let s = &mut *scratch;
        let AttentionLayerScratch {
            q_buf,
            attn_out,
            head_scores,
            ..
        } = s;
        sparse_attention_accumulate(
            q_buf,
            &*cache,
            attn_out,
            head_scores,
            selection,
            block_cache,
            n_head,
            n_kv,
            hd,
            kvd,
            seq_len,
            scale,
        );
    }

    // ── 9: Output gating (identical to dense) ─────────────────────────────
    for i in 0..q_dim {
        scratch.attn_out[i] *= fast_sigmoid(scratch.gate_buf[i]);
    }

    // ── 10: Output projection (identical to dense) ─────────────────────────
    layer
        .attn_wo
        .matvec(&scratch.attn_out[..q_dim], &mut x[..n_embd]);
}

/// How the sparse forward refreshes the block cache before selecting
/// (Issue 1117 — the two postures the decode lane measures against each
/// other; the single-layer f32 research forward is always Rebuild).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheUpdate {
    /// Full `rebuild_from_keys` every call — O(seq_len × kv_dim).
    Rebuild,
    /// Append-only decode: `update_last_block_from_keys` — O(block_size ×
    /// kv_dim); a block OPEN force-refreshes the selector the same step so
    /// the fresh block is the budget rule's "current block" immediately.
    /// `last_n_blocks` is the per-layer scratch carrying the previous step's
    /// active-block count.
    Incremental,
}

/// Cache-refresh + selection half of the sparse forward — shared by the f32
/// research forward ([`CacheUpdate::Rebuild`]) and the ternary decode lane
/// ([`CacheUpdate::Incremental`]). Returns the selection (borrowed from the
/// selector; the caller must not touch the selector until it is done with
/// it). `kv_query` is the KV-resolution query `[n_kv * head_dim]` (the
/// caller builds it with [`build_kv_resolution_query`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn refresh_and_select<'a, S: GqaBlockSelector>(
    cache_key: &[f32],
    block_cache: &mut GqaFlashMemoryBlockCache,
    selector: &'a mut S,
    kv_query: &[f32],
    scale: f32,
    seq_len: usize,
    step: usize,
    update: CacheUpdate,
    last_n_blocks: &mut usize,
) -> &'a PerHeadSelection {
    match update {
        CacheUpdate::Rebuild => block_cache.rebuild_from_keys(cache_key, seq_len),
        CacheUpdate::Incremental => {
            block_cache.update_last_block_from_keys(cache_key, seq_len);
            let n_blocks = block_cache.n_active_blocks();
            if n_blocks != *last_n_blocks {
                selector.force_refresh();
                *last_n_blocks = n_blocks;
            }
        }
    }
    selector.select_blocks(kv_query, block_cache, scale, step)
}

/// Per-head sparse attention accumulation — verbatim from the original
/// `forward_attention_layer_flashmemory` (one home for the score/softmax/
/// V-accumulation form; the dense kernel's identical per-head order is what
/// makes the all-blocks arm bit-comparable to dense). `q_buf` is the
/// post-norm post-RoPE queries `[n_head * head_dim]`; `attn_out` receives
/// `[.. n_head * head_dim]`; `scores` is per-head softmax scratch with
/// capacity ≥ the largest selected-token count (the decode lane passes the
/// attention scratch's `head_scores`, which is `n_head × cache capacity`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_attention_accumulate(
    q_buf: &[f32],
    cache: &KVCache,
    attn_out: &mut [f32],
    scores: &mut [f32],
    selection: &PerHeadSelection,
    block_cache: &GqaFlashMemoryBlockCache,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    kvd: usize,
    seq_len: usize,
    scale: f32,
) {
    let q_dim = n_head * hd;
    attn_out[..q_dim].fill(0.0);

    for q_h in 0..n_head {
        let kv_group = q_h * n_kv / n_head;
        let q_slice = &q_buf[q_h * hd..(q_h + 1) * hd];
        let out_slice = &mut attn_out[q_h * hd..(q_h + 1) * hd];
        out_slice.fill(0.0);

        let selected = &selection.blocks_per_head[kv_group];
        let mut fallback = [0usize; 1];
        let blocks_to_attend: &[usize] = if selected.is_empty() {
            fallback[0] = block_cache.n_active_blocks().saturating_sub(1);
            &fallback[..]
        } else {
            selected
        };

        // First pass: compute scores over selected tokens.
        let mut n_scored = 0usize;
        let mut max_score = f32::NEG_INFINITY;

        for &block_idx in blocks_to_attend {
            let (tok_start, tok_end) = block_cache.block_token_range(block_idx, seq_len);
            for tok in tok_start..tok_end {
                let key_off = tok * kvd + kv_group * hd;
                let k_slice = &cache.key[key_off..key_off + hd];
                let score = simd_dot_f32(q_slice, k_slice, hd) * scale;
                scores[n_scored] = score;
                n_scored += 1;
                if score > max_score {
                    max_score = score;
                }
            }
        }

        // Softmax (numerically stable, over selected tokens only).
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut().take(n_scored) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        // Second pass: weighted value sum.
        let mut score_idx = 0usize;
        for &block_idx in blocks_to_attend {
            let (tok_start, tok_end) = block_cache.block_token_range(block_idx, seq_len);
            for tok in tok_start..tok_end {
                let val_off = tok * kvd + kv_group * hd;
                let weight = scores[score_idx] * inv_sum;
                score_idx += 1;
                let v_slice = &cache.value[val_off..val_off + hd];
                for (o, &v) in out_slice.iter_mut().zip(v_slice.iter()) {
                    *o += weight * v;
                }
            }
        }
    }
}

/// Average query heads into a KV-head-resolution query for the selector.
///
/// GQA block-contiguous mapping (identical to the dense scoring path's
/// `kv_group = q_h * n_kv / n_head`): KV head g owns query heads
/// `[g * group_size, (g+1) * group_size)` where `group_size = n_head / n_kv`.
///
/// `q_heads` is `n_head * head_dim`; `out` must be exactly `n_kv * head_dim`.
///
/// (Bench 457 regression: averaging the interleaved set {g, g+n_kv, g+2*n_kv,
/// ...} instead mixes heads from different KV groups — quantified on real
/// Bonsai at −5.1pp recall @ thr 0.5.)
pub(crate) fn build_kv_resolution_query(
    q_heads: &[f32],
    n_head: usize,
    n_kv: usize,
    hd: usize,
    out: &mut [f32],
) {
    let group_size = n_head / n_kv;
    let inv = 1.0 / group_size as f32;
    out.fill(0.0);
    for kv_h in 0..n_kv {
        let dst_off = kv_h * hd;
        for g in 0..group_size {
            let src_off = (kv_h * group_size + g) * hd;
            for d in 0..hd {
                out[dst_off + d] += q_heads[src_off + d];
            }
        }
        for d in 0..hd {
            out[dst_off + d] *= inv;
        }
    }
}

/// Compute the KV-reduction ratio for the current selection.
///
/// Returns `(selected_tokens, total_tokens, reduction_pct)` where
/// `reduction_pct = (1 - selected/total) * 100`. This is the G5 metric.
pub fn kv_reduction_ratio(
    selection: &PerHeadSelection,
    block_cache: &GqaFlashMemoryBlockCache,
    seq_len: usize,
) -> (usize, usize, f32) {
    let total = seq_len;
    let n_kv = selection.blocks_per_head.len();
    if n_kv == 0 || total == 0 {
        return (0, total, 0.0);
    }
    let mut total_selected = 0usize;
    for blocks in &selection.blocks_per_head {
        for &block_idx in blocks {
            let (start, end) = block_cache.block_token_range(block_idx, seq_len);
            total_selected += end - start;
        }
    }
    let avg_selected = total_selected / n_kv;
    let reduction = (1.0 - avg_selected as f32 / total as f32) * 100.0;
    (avg_selected, total, reduction)
}

/// Per-attention-layer decode state for the FlashMemory budget lane
/// (Issue 1117 T2): the block centroid cache, the budget selector, and the
/// block-open detector scratch.
pub struct FlashMemoryLayerState {
    pub block_cache: GqaFlashMemoryBlockCache,
    pub selector: GqaFlashMemorySelector,
    /// The previous decode step's active-block count — a changed count means
    /// a new block opened and [`refresh_and_select`] force-refreshes the
    /// selector (the block-open law on [`GqaBlockSelector::force_refresh`]).
    pub last_n_blocks: usize,
}

/// Per-layer decode state for the FlashMemory budget decode lane
/// (Issue 1117 T2). One entry per model layer: `Some` on full-attention
/// layers, `None` on GDN/DeltaNet layers (they have no KV cache to select
/// over). Built once at boot, never per tick.
pub struct FlashMemoryDecodeState {
    pub layers: Vec<Option<FlashMemoryLayerState>>,
    /// `false` = full [`GqaFlashMemoryBlockCache::rebuild_from_keys`] every
    /// decode step — the reference posture the G1 equivalence test and the
    /// T4 cost comparison measure the incremental path against. Selection
    /// refreshes IDENTICALLY in both modes (same schedule, bit-identical
    /// centroids) — only the centroid update cost differs.
    pub incremental: bool,
}

impl FlashMemoryDecodeState {
    /// Build one state entry per layer. `fm_config.block_size` should match
    /// the serving lane's operating point (the Bench-671 fidelity numbers
    /// were measured at block size 64). Gated on `flashmemory_budget` — the
    /// selector is built `.with_budget(rho)` (threshold unused by that rule).
    #[cfg(feature = "flashmemory_budget")]
    pub fn new(
        config: &Config,
        layer_types: &[crate::types::DeltaNetLayerType],
        max_positions: usize,
        fm_config: katgpt_attn::dash_attn::flashmemory_sparse::FlashMemoryConfig,
        rho: f32,
    ) -> Self {
        use katgpt_attn::dash_attn::flashmemory_sparse::GqaFlashMemorySelector;
        assert!(rho > 0.0 && rho <= 1.0, "budget must be in (0, 1]");
        let max_blocks = max_positions.div_ceil(fm_config.block_size).max(1);
        let mut layers = Vec::with_capacity(layer_types.len());
        let mut attention_layers = 0usize;
        for lt in layer_types {
            if *lt == crate::types::DeltaNetLayerType::Attention {
                layers.push(Some(FlashMemoryLayerState {
                    block_cache: GqaFlashMemoryBlockCache::new(
                        config.n_kv_head,
                        config.head_dim,
                        &fm_config,
                        max_positions,
                    ),
                    selector: GqaFlashMemorySelector::new(
                        fm_config.clone(),
                        config.n_kv_head,
                        config.head_dim,
                        max_blocks,
                    )
                    .with_budget(rho),
                    last_n_blocks: 0,
                }));
                attention_layers += 1;
            } else {
                layers.push(None);
            }
        }
        assert!(
            attention_layers > 0,
            "flashmemory decode state on a model with no full-attention layers"
        );
        Self {
            layers,
            incremental: true,
        }
    }

    /// Number of full-attention layers carrying state.
    pub fn attention_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::weights::{DeltaNetLayerWeights, Proj};
    use crate::types::{Config, DeltaNetLayerType};
    use katgpt_attn::dash_attn::flashmemory_sparse::FlashMemoryConfig;

    /// Build a small GQA config: `n_head=4`, `n_kv_head=2`, `head_dim=8`, `n_embd=32`.
    fn small_gqa_config() -> Config {
        let mut config = Config::qwen_deltanet(1, vec![DeltaNetLayerType::Attention]);
        config.vocab_size = 64;
        config.n_embd = 32;
        config.n_head = 4;
        config.n_kv_head = 2;
        config.head_dim = 8;
        config.mlp_hidden = 64;
        config.block_size = 128;
        config.rms_norm_eps = 1e-6;
        config.rope_dimension_count = 0; // full rotation
        config
    }

    /// Synthetic attention layer weights for the small config.
    fn make_layer_weights(config: &Config) -> DeltaNetLayerWeights {
        let q_dim = config.n_head * config.head_dim;
        let kvd = config.n_kv_head * config.head_dim;
        let n_embd = config.n_embd;

        // Diagonal-ish init: data[r * cols + (r % cols)] = 0.1.
        let mk = |rows: usize, cols: usize| -> Proj {
            let mut data = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let c = r % cols;
                data[r * cols + c] = 0.1;
            }
            Proj::dense(data, rows, cols)
        };

        DeltaNetLayerWeights {
            attn_wq: mk(q_dim * 2, n_embd),
            attn_wk: mk(kvd, n_embd),
            attn_wv: mk(kvd, n_embd),
            attn_wo: mk(n_embd, q_dim),
            attn_q_norm: vec![1.0; config.head_dim],
            attn_k_norm: vec![1.0; config.head_dim],
            in_proj_qkv: Proj::empty(),
            in_proj_a: Proj::empty(),
            in_proj_b: Proj::empty(),
            in_proj_z: Proj::empty(),
            out_proj: Proj::empty(),
            conv1d_weight: Vec::new(),
            a_log: Vec::new(),
            dt_bias: Vec::new(),
            linear_norm: Vec::new(),
            gate_proj: Proj::empty(),
            up_proj: Proj::empty(),
            down_proj: Proj::empty(),
            input_norm: vec![1.0; n_embd],
            post_attn_norm: vec![1.0; n_embd],
        }
    }

    /// Smoke test: sparse forward runs without panic at small scale.
    #[test]
    fn smoke_sparse_forward_no_panic() {
        let config = small_gqa_config();
        let layer = make_layer_weights(&config);
        let n_embd = config.n_embd;

        let mut x = vec![0.5f32; n_embd];
        let mut cache = KVCache::new(&config);
        let mut scratch = AttentionLayerScratch::new(&config);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);

        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let max_seq = 32;
        let mut block_cache =
            GqaFlashMemoryBlockCache::new(config.n_kv_head, config.head_dim, &fm_config, max_seq);
        let max_blocks = max_seq.div_ceil(fm_config.block_size);
        let mut selector =
            GqaFlashMemorySelector::new(fm_config, config.n_kv_head, config.head_dim, max_blocks);

        for pos in 0..16 {
            forward_attention_layer_flashmemory(
                &mut x,
                &layer,
                &mut cache,
                pos,
                &config,
                &rope_freq,
                &mut scratch,
                &mut block_cache,
                &mut selector,
                pos,
            );
        }
        for &v in &x {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// G1: with threshold=0 (select ALL blocks), sparse ≈ dense.
    #[test]
    fn g1_sparse_all_blocks_matches_dense() {
        let config = small_gqa_config();
        let layer = make_layer_weights(&config);
        let n_embd = config.n_embd;
        let seq_len = 16;

        // Dense forward.
        let mut x_dense = vec![0.5f32; n_embd];
        let mut cache_dense = KVCache::new(&config);
        let mut scratch_dense = AttentionLayerScratch::new(&config);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);

        for pos in 0..seq_len {
            crate::deltanet::forward::forward_attention_layer(
                &mut x_dense,
                &layer,
                &mut cache_dense,
                pos,
                &config,
                &rope_freq,
                &mut scratch_dense,
                None,
            );
        }

        // Sparse forward with threshold=0 (select ALL blocks).
        let mut x_sparse = vec![0.5f32; n_embd];
        let mut cache_sparse = KVCache::new(&config);
        let mut scratch_sparse = AttentionLayerScratch::new(&config);

        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.0,
        };
        let mut block_cache =
            GqaFlashMemoryBlockCache::new(config.n_kv_head, config.head_dim, &fm_config, seq_len);
        let max_blocks = seq_len.div_ceil(fm_config.block_size);
        let mut selector =
            GqaFlashMemorySelector::new(fm_config, config.n_kv_head, config.head_dim, max_blocks);

        for pos in 0..seq_len {
            forward_attention_layer_flashmemory(
                &mut x_sparse,
                &layer,
                &mut cache_sparse,
                pos,
                &config,
                &rope_freq,
                &mut scratch_sparse,
                &mut block_cache,
                &mut selector,
                pos,
            );
        }

        let dot: f32 = x_dense.iter().zip(&x_sparse).map(|(a, b)| a * b).sum();
        let norm_dense = x_dense.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_sparse = x_sparse.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (norm_dense * norm_sparse + 1e-10);
        assert!(
            cos > 0.95,
            "cosine with threshold=0 (all blocks) should be > 0.95, got {cos}"
        );
    }

    /// G5: KV-reduction ratio computes correctly.
    #[test]
    fn g5_kv_reduction_ratio() {
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let seq_len = 16;
        let mut block_cache = GqaFlashMemoryBlockCache::new(2, 8, &fm_config, seq_len);
        let keys = vec![1.0f32; seq_len * 2 * 8];
        block_cache.rebuild_from_keys(&keys, seq_len);

        let mut selection = PerHeadSelection::new(2, 4);
        selection.blocks_per_head[0].push(0);
        selection.blocks_per_head[0].push(1);

        let (sel, total, red) = kv_reduction_ratio(&selection, &block_cache, seq_len);
        assert_eq!(total, 16);
        assert_eq!(sel, 4); // avg(8, 0) = 4
        assert!((red - 75.0).abs() < 0.1, "reduction = {red}");
    }

    /// Regression (Bench 457): the KV-resolution query must average the query
    /// heads that actually attend to each KV head under the block-contiguous
    /// GQA mapping (`kv_group` = `q_h` * `n_kv` / `n_head`) — NOT the interleaved set
    /// {g, `g+n_kv`, ...}. With `n_head=4`, `n_kv=2`: KV0 owns {q0,q1}, KV1 owns
    /// {q2,q3}. (The interleaved build would average {q0,q2} and {q1,q3}.)
    #[test]
    fn kv_resolution_query_uses_block_contiguous_gqa_groups() {
        let n_head = 4;
        let n_kv = 2;
        let hd = 8;
        // Head h carries the constant (h+1) * 0.25 in every dim — distinct
        // per head, so the two groupings average different values.
        let q_heads: Vec<f32> = (0..n_head * hd)
            .map(|i| ((i / hd) + 1) as f32 * 0.25)
            .collect();
        let mut out = vec![0.0f32; n_kv * hd];
        build_kv_resolution_query(&q_heads, n_head, n_kv, hd, &mut out);
        // KV0 = avg(q0, q1) = (0.25 + 0.50) / 2 = 0.375 (interleaved: 0.50);
        // KV1 = avg(q2, q3) = (0.75 + 1.00) / 2 = 0.875 (interleaved: 0.75).
        let expected = [0.375f32, 0.875f32];
        for (kv_h, chunk) in out.chunks(hd).enumerate() {
            for (d, &v) in chunk.iter().enumerate() {
                assert!((v - expected[kv_h]).abs() < 1e-6, "KV{kv_h}[{d}] = {v}");
            }
        }
    }

    /// Bonsai-like dims smoke test.
    #[test]
    fn bonsai_like_dims_smoke() {
        let fm_config = FlashMemoryConfig::test_config();
        let n_kv_head = 8;
        let head_dim = 256;
        let seq_len = 128;
        let mut block_cache =
            GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm_config, seq_len);
        let kv_dim = n_kv_head * head_dim;
        let keys = vec![0.01f32; seq_len * kv_dim];
        block_cache.rebuild_from_keys(&keys, seq_len);
        assert_eq!(block_cache.n_active_blocks(), 8);

        let max_blocks = seq_len.div_ceil(fm_config.block_size);
        let mut sel = GqaFlashMemorySelector::new(fm_config, n_kv_head, head_dim, max_blocks);
        let query = vec![0.01; n_kv_head * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let selection = sel.select(&query, &block_cache, scale, 0);
        assert!(selection.total_selections() > 0);
    }

    /// Trained-path G1 parity (Plan 337 D1 / Issue 452 GAP 2): a zero-weight
    /// `DualEncoderIndexer` at threshold 0.5 scores every block σ(0)=0.5 ≥ 0.5 →
    /// selects ALL blocks → the trained sparse forward matches dense (same
    /// contract as `g1_sparse_all_blocks_matches_dense`, exercised through
    /// `TrainedIndexerSelector` + the generic forward seam).
    #[cfg(feature = "flashmemory_trained_indexer")]
    #[test]
    fn g1_trained_zero_weight_all_blocks_matches_dense() {
        let config = small_gqa_config();
        let layer = make_layer_weights(&config);
        let n_embd = config.n_embd;
        let seq_len = 16;

        // Dense reference.
        let mut x_dense = vec![0.5f32; n_embd];
        let mut cache_dense = KVCache::new(&config);
        let mut scratch_dense = AttentionLayerScratch::new(&config);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        for pos in 0..seq_len {
            crate::deltanet::forward::forward_attention_layer(
                &mut x_dense,
                &layer,
                &mut cache_dense,
                pos,
                &config,
                &rope_freq,
                &mut scratch_dense,
                None,
            );
        }

        // Trained sparse forward: zero weights (all scores σ(0)=0.5),
        // threshold 0.5 → every block selected.
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let head_dim = config.head_dim;
        let n_kv = config.n_kv_head;
        let hidden = (head_dim / 4).max(4);
        let max_blocks = seq_len.div_ceil(fm_config.block_size);
        let indexer = DualEncoderIndexer::from_weights(
            fm_config.clone(),
            head_dim,
            n_kv,
            max_blocks,
            vec![0.0; hidden * head_dim],
            vec![0.0; hidden],
            vec![0.0; hidden],
            0.0,
            vec![0.0; hidden * head_dim],
            vec![0.0; hidden],
            vec![0.0; hidden],
            0.0,
        );
        let mut selector = TrainedIndexerSelector { indexer };

        let mut x_sparse = vec![0.5f32; n_embd];
        let mut cache_sparse = KVCache::new(&config);
        let mut scratch_sparse = AttentionLayerScratch::new(&config);
        let mut block_cache = GqaFlashMemoryBlockCache::new(n_kv, head_dim, &fm_config, seq_len);

        for pos in 0..seq_len {
            forward_attention_layer_flashmemory(
                &mut x_sparse,
                &layer,
                &mut cache_sparse,
                pos,
                &config,
                &rope_freq,
                &mut scratch_sparse,
                &mut block_cache,
                &mut selector,
                pos,
            );
        }

        let dot: f32 = x_dense.iter().zip(&x_sparse).map(|(a, b)| a * b).sum();
        let norm_dense = x_dense.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_sparse = x_sparse.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (norm_dense * norm_sparse + 1e-10);
        assert!(
            cos > 0.95,
            "trained zero-weight all-blocks cos should be > 0.95, got {cos}"
        );
    }

    /// `from_ckpt_bytes` round-trips a real checkpoint buffer (the D1 loading
    /// path: riir-train `to_bytes()` ckpt → served selector).
    #[cfg(feature = "flashmemory_trained_indexer")]
    #[test]
    fn trained_selector_loads_from_ckpt_bytes() {
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let head_dim = 8;
        let n_kv = 2;
        let max_blocks = 4;
        let hidden = (head_dim / 4).max(4);
        let indexer = DualEncoderIndexer::from_weights(
            fm_config.clone(),
            head_dim,
            n_kv,
            max_blocks,
            vec![0.1; hidden * head_dim],
            vec![0.0; hidden],
            vec![0.1; hidden],
            0.0,
            vec![0.1; hidden * head_dim],
            vec![0.0; hidden],
            vec![0.1; hidden],
            0.0,
        );
        let bytes = indexer.to_bytes();

        let loaded =
            TrainedIndexerSelector::from_ckpt_bytes(&bytes, fm_config.clone(), n_kv, max_blocks)
                .expect("checkpoint loads");
        assert_eq!(loaded.indexer.param_count(), indexer.param_count());

        // Truncated buffer must be rejected, not panic.
        let short = &bytes[..bytes.len() - 4];
        assert!(
            TrainedIndexerSelector::from_ckpt_bytes(short, fm_config, n_kv, max_blocks).is_err()
        );
    }
}
