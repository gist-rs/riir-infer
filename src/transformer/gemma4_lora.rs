//! Gemma 4 `LoRA` application during forward (Plan 320 Phase C1).
//!
//! Implements [`LoraApplier`] for Gemma 4 12B — applies weight-delta `LoRA`
//! adapters at the 7 matmul insertion points per layer (Q/K/V/O + gate/up/down).
//! The base weights are frozen; only the `LoRA` delta `(α/r)·B(Ax)` is added.
//!
//! ## Architecture
//!
//! Mirrors [`crate::transformer::gemma2_lora::GemmaLora`] (Plan 410 Phase 1):
//! each layer can have up to 7 optional [`LoraAdapter`]s. Missing adapters are
//! skipped (no-op). The struct owns a single scratch buffer of size `max_rank`,
//! reused across all 7 points per layer — zero allocation in the hot path.
//!
//! ## Gemma 4 vs Gemma 2 differences
//!
//! Gemma 4 has **per-layer-type** Q/K/V dimensions: sliding layers use
//! `head_dim=256` with 8 KV heads (`kv_dim=2048`), full-attention layers use
//! `global_head_dim=512` with 1 KV head (`kv_dim=512`). The `LoraAdapter` carries
//! its own `in_dim`/`out_dim`, so the hook trait interface works unchanged —
//! the adapter dimensions just differ per layer. The constructor
//! [`Gemma4Lora::new`] derives each adapter's dimensions from the per-layer
//! type via `q_dim_for` / `kv_dim_for`.
//!
//! ## Feature gate
//!
//! Behind `gemma4_lora` (implies `gemma4_inference`). When the feature is off,
//! callers pass `&mut NoLora` which monomorphizes to zero overhead.

use crate::transformer::gemma2::LoraApplier;
use crate::transformer::gemma4::{kv_dim_for, q_dim_for};
use crate::types::{Config, Gemma4LayerType, LoraAdapter, Rng, lora_apply};

/// Per-layer `LoRA` adapters for Gemma 4. Each field is `None` if that projection
/// has no `LoRA` for this layer (the projection runs without any delta).
#[derive(Clone)]
pub struct Gemma4LayerLora {
    pub q: Option<LoraAdapter>,
    pub k: Option<LoraAdapter>,
    pub v: Option<LoraAdapter>,
    pub o: Option<LoraAdapter>,
    pub gate: Option<LoraAdapter>,
    pub up: Option<LoraAdapter>,
    pub down: Option<LoraAdapter>,
}

impl Gemma4LayerLora {
    /// All-none layer (no `LoRA` on any projection). Used for layers that the
    /// loaded `LoRA` file does not cover.
    pub fn empty() -> Self {
        Self {
            q: None,
            k: None,
            v: None,
            o: None,
            gate: None,
            up: None,
            down: None,
        }
    }
}

/// Real `LoRA` applier for Gemma 4 — holds per-layer adapter sets + a scratch
/// buffer. Created once per model load, passed by `&mut` to
/// `forward_gemma4_with_lora`.
///
/// The `current_layer` index is advanced by `LoraApplier::next_layer` at the
/// top of each layer iteration, selecting that layer's adapter set.
#[derive(Clone)]
pub struct Gemma4Lora {
    /// One entry per model layer. `layers[i]` is applied during layer `i`.
    pub layers: Vec<Gemma4LayerLora>,
    /// Scratch buffer for the rank-dim intermediate `A @ input`. Reused across
    /// all 7 projection points — no allocation in the hot path.
    buf: Vec<f32>,
    /// Current layer index (advanced by `next_layer`).
    current_layer: usize,
}

impl Gemma4Lora {
    /// Construct from per-layer adapter sets. The scratch buffer is sized to
    /// `max_rank` across all adapters.
    pub fn new(layers: Vec<Gemma4LayerLora>) -> Self {
        let max_rank = layers
            .iter()
            .flat_map(|l: &Gemma4LayerLora| {
                [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]
                    .into_iter()
                    .filter_map(|a: &Option<LoraAdapter>| a.as_ref().map(|adp| adp.rank))
            })
            .max()
            .unwrap_or(0);
        Self {
            layers,
            buf: vec![0.0; max_rank],
            current_layer: 0,
        }
    }

    /// Construct an all-empty `Gemma4Lora` for `n_layers` layers. Equivalent to
    /// `NoLora` in effect, but typed as `Gemma4Lora` — useful as a placeholder
    /// when the `LoRA` file covers only some layers.
    pub fn empty(n_layers: usize) -> Self {
        Self::new((0..n_layers).map(|_| Gemma4LayerLora::empty()).collect())
    }

    /// Reset the layer cursor to 0. Called at the start of each forward pass
    /// (the layer loop increments via `next_layer`).
    #[inline(always)]
    pub fn reset(&mut self) {
        self.current_layer = 0;
    }

    /// Helper: apply `adapter` to `output` given `input`, using `buf` as scratch.
    /// Returns early if the adapter is `None`.
    #[inline(always)]
    fn apply_one(
        output: &mut [f32],
        input: &[f32],
        adapter: Option<&LoraAdapter>,
        buf: &mut [f32],
    ) {
        if let Some(adp) = adapter {
            // Safety: buf is sized to max_rank >= adp.rank at construction.
            let r = adp.rank;
            debug_assert!(buf.len() >= r, "scratch buf too small for rank {r}");
            let scratch = &mut buf[..r];
            lora_apply(output, adp, input, scratch);
        }
    }
}

impl LoraApplier for Gemma4Lora {
    #[inline(always)]
    fn apply_q(&mut self, q: &mut [f32], x_in: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.q.as_ref();
        Self::apply_one(q, x_in, adp, &mut self.buf);
    }

    #[inline(always)]
    fn apply_k(&mut self, k: &mut [f32], x_in: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.k.as_ref();
        Self::apply_one(k, x_in, adp, &mut self.buf);
    }

    #[inline(always)]
    fn apply_v(&mut self, v: &mut [f32], x_in: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.v.as_ref();
        Self::apply_one(v, x_in, adp, &mut self.buf);
    }

    #[inline(always)]
    fn apply_o(&mut self, x: &mut [f32], attn_out: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.o.as_ref();
        Self::apply_one(x, attn_out, adp, &mut self.buf);
    }

    #[inline(always)]
    fn apply_gate(&mut self, gate: &mut [f32], x_in: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.gate.as_ref();
        Self::apply_one(gate, x_in, adp, &mut self.buf);
    }

    #[inline(always)]
    fn apply_up(&mut self, up: &mut [f32], x_in: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.up.as_ref();
        Self::apply_one(up, x_in, adp, &mut self.buf);
    }

    #[inline(always)]
    fn apply_down(&mut self, x: &mut [f32], hidden: &[f32]) {
        let idx = self.current_layer;
        let layer = &self.layers[idx];
        let adp = layer.down.as_ref();
        Self::apply_one(x, hidden, adp, &mut self.buf);
    }

    #[inline(always)]
    fn next_layer(&mut self) {
        self.current_layer += 1;
    }
}

// ─── Factory: init adapters for all layers at a given rank ──────────────────

/// Configuration for Gemma 4 `LoRA` fine-tuning (Plan 320 Phase C1).
#[derive(Clone, Debug)]
pub struct Gemma4LoraConfig {
    /// `LoRA` rank `r` (typically 8, 16, 32, 64, or 128).
    pub rank: usize,
    /// `LoRA` alpha (scaling = alpha / rank; typically 2 × rank).
    pub alpha: f32,
    /// Target attention projections (Q/K/V/O).
    pub target_attention: bool,
    /// Target FFN projections (gate/up/down).
    pub target_ffn: bool,
}

impl Default for Gemma4LoraConfig {
    fn default() -> Self {
        Self::rank_64()
    }
}

impl Gemma4LoraConfig {
    /// Rank-64, alpha=128 (scaling=2.0), all 7 projections.
    /// The "Strand-Rust-Coder" recipe from Plan 320 Phase C.
    pub fn rank_64() -> Self {
        Self {
            rank: 64,
            alpha: 128.0,
            target_attention: true,
            target_ffn: true,
        }
    }

    /// Rank-16, alpha=32 (scaling=2.0), all 7 projections.
    /// Lighter config for quick experiments.
    pub fn rank_16() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target_attention: true,
            target_ffn: true,
        }
    }

    /// Attention-only (Q/K/V/O), no FFN. Matches the common "attention-only
    /// `LoRA`" recipe — fewer params, faster training.
    pub fn attention_only(mut self) -> Self {
        self.target_ffn = false;
        self
    }
}

/// Create a zero-initialized `LoRA` adapter (B=0 → ΔW=0 at init → identity).
fn make_lora_zeros(rank: usize, alpha: f32, in_dim: usize, out_dim: usize) -> LoraAdapter {
    LoraAdapter {
        rank,
        in_dim,
        out_dim,
        a: vec![0.0; rank * in_dim],
        b: vec![0.0; out_dim * rank],
        alpha,
    }
}

impl Gemma4Lora {
    /// Initialize all layers' adapters for a given config, with standard `LoRA`
    /// init (A=0, B=0 → ΔW=0 → forward is identical to base at init).
    ///
    /// The caller is expected to re-initialize A with Kaiming-like random
    /// values before training (B stays zero so the initial delta is zero).
    /// The adapter dimensions are derived from the per-layer type (sliding vs
    /// full attention have different Q/K/V sizes).
    pub fn zeros(config: &Config, lora_config: &Gemma4LoraConfig) -> Self {
        let n = config.n_embd;
        let mlp = config.mlp_hidden;
        let rank = lora_config.rank;
        let alpha = lora_config.alpha;
        let tgt_attn = lora_config.target_attention;
        let tgt_ffn = lora_config.target_ffn;

        // Use the config's layer types (set by the loader) rather than deriving
        // from `layer_idx % 6 == 5`. The loader sets `gemma4_layer_types` to
        // match the actual model's SWA pattern.
        let layers = (0..config.n_layer)
            .map(|layer_idx| {
                let layer_type = config.gemma4_layer_types.get(layer_idx).copied().unwrap_or(
                    if layer_idx % 6 == 5 {
                        Gemma4LayerType::Full
                    } else {
                        Gemma4LayerType::Sliding
                    },
                );
                let q_dim = q_dim_for(config, layer_type);
                let kvd = kv_dim_for(config, layer_type);

                Gemma4LayerLora {
                    q: (tgt_attn).then(|| make_lora_zeros(rank, alpha, n, q_dim)),
                    k: (tgt_attn).then(|| make_lora_zeros(rank, alpha, n, kvd)),
                    v: (tgt_attn).then(|| make_lora_zeros(rank, alpha, n, kvd)),
                    o: (tgt_attn).then(|| make_lora_zeros(rank, alpha, q_dim, n)),
                    gate: (tgt_ffn).then(|| make_lora_zeros(rank, alpha, n, mlp)),
                    up: (tgt_ffn).then(|| make_lora_zeros(rank, alpha, n, mlp)),
                    down: (tgt_ffn).then(|| make_lora_zeros(rank, alpha, mlp, n)),
                }
            })
            .collect();
        Self::new(layers)
    }

    /// Initialize with random A (Kaiming-like) and zero B (standard `LoRA` init).
    /// ΔW = B @ A ≈ 0 at init, preserving base model behavior.
    pub fn new_random(config: &Config, lora_config: &Gemma4LoraConfig, rng: &mut Rng) -> Self {
        let mut lora = Self::zeros(config, lora_config);
        // Re-init A with Kaiming-like random values per adapter.
        for layer in &mut lora.layers {
            for adapter in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.o,
                &mut layer.gate,
                &mut layer.up,
                &mut layer.down,
            ] {
                if let Some(adp) = adapter.as_mut() {
                    let scale = (2.0 / adp.in_dim as f32).sqrt();
                    for v in adp.a.iter_mut() {
                        *v = rng.normal() * scale;
                    }
                }
            }
        }
        lora
    }

    /// Total number of trainable `LoRA` parameters across all adapters.
    pub fn total_params(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| {
                [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]
                    .into_iter()
                    .filter_map(|a| a.as_ref())
            })
            .map(|adp| adp.a.len() + adp.b.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter(rank: usize, in_dim: usize, out_dim: usize, fill: f32) -> LoraAdapter {
        LoraAdapter {
            rank,
            in_dim,
            out_dim,
            a: vec![fill; rank * in_dim],
            b: vec![fill; out_dim * rank],
            alpha: rank as f32, // scale = alpha/rank = 1.0
        }
    }

    #[test]
    fn empty_gemma4_lora_is_no_op() {
        // An all-empty Gemma4Lora should not modify any output — same as NoLora.
        let mut lora = Gemma4Lora::empty(4);
        let mut output = vec![1.0, 2.0, 3.0];
        let input = vec![0.5; 3];
        lora.next_layer();
        lora.apply_q(&mut output, &input);
        assert_eq!(
            output,
            vec![1.0, 2.0, 3.0],
            "empty LoRA must not change output"
        );
    }

    #[test]
    fn real_lora_modifies_output() {
        // rank=1, in_dim=2, out_dim=2, fill=1.0, scale=1.0.
        // delta = 1.0 * B @ (A @ input) where B=[[1],[1]], A=[[1,1]], input=[1,1].
        // A @ input = [2], B @ [2] = [2, 2]. So output += [2, 2].
        let adapter = make_adapter(1, 2, 2, 1.0);
        let layer = Gemma4LayerLora {
            q: Some(adapter),
            ..Gemma4LayerLora::empty()
        };
        let mut lora = Gemma4Lora::new(vec![layer]);
        let mut output = vec![0.0, 0.0];
        let input = vec![1.0, 1.0];
        lora.apply_q(&mut output, &input);
        assert_eq!(
            output,
            vec![2.0, 2.0],
            "LoRA delta should be B@(A@input) = [2,2]"
        );
    }

    #[test]
    fn layer_advancement_selects_correct_adapters() {
        // Layer 0 has a Q adapter, layer 1 does not.
        let layers = vec![
            Gemma4LayerLora {
                q: Some(make_adapter(1, 1, 1, 1.0)),
                ..Gemma4LayerLora::empty()
            },
            Gemma4LayerLora::empty(),
        ];
        let mut lora = Gemma4Lora::new(layers);

        // Layer 0: Q adapter active
        let mut out0 = vec![0.0];
        lora.apply_q(&mut out0, &[1.0]);
        assert_eq!(
            out0,
            vec![1.0],
            "layer 0 Q delta = 1.0 * 1.0 * 1.0 * 1.0 = 1.0"
        );

        // Advance to layer 1: no Q adapter
        lora.next_layer();
        let mut out1 = vec![5.0];
        lora.apply_q(&mut out1, &[1.0]);
        assert_eq!(out1, vec![5.0], "layer 1 has no Q adapter — no change");
    }

    #[test]
    fn max_rank_sizing() {
        // Adapters of different ranks — buf must be sized to max.
        let layers = vec![Gemma4LayerLora {
            q: Some(make_adapter(4, 2, 2, 1.0)),
            k: Some(make_adapter(8, 2, 2, 1.0)),
            ..Gemma4LayerLora::empty()
        }];
        let lora = Gemma4Lora::new(layers);
        assert_eq!(
            lora.buf.len(),
            8,
            "buf must be sized to max rank across adapters"
        );
    }
}
