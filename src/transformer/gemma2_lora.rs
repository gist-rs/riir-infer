//! Gemma 2 `LoRA` application during inference (Plan 410 Phase 1).
//!
//! Implements `LoraApplier` for real Gemma 2 2B — applies weight-delta `LoRA`
//! adapters at the 7 matmul insertion points per layer (Q/K/V/O + gate/up/down).
//! The base weights are frozen; only the `LoRA` delta `(α/r)·B(Ax)` is added.
//!
//! ## Architecture
//!
//! Each layer can have up to 7 optional [`LoraAdapter`]s. A loaded Go `LoRA`
//! typically targets a subset (e.g., Q/V only, or all 7). Missing adapters are
//! skipped (no-op). The struct owns a single scratch buffer of size `rank`,
//! reused across all 7 points per layer — zero allocation in the hot path.
//!
//! ## Feature gate
//!
//! Behind `gemma_lora`. When the feature is off, callers pass `&mut NoLora`
//! which monomorphizes to zero overhead.

use crate::transformer::LoraApplier;
use crate::types::{LoraAdapter, lora_apply};

/// Per-layer `LoRA` adapters for Gemma 2. Each field is `None` if that projection
/// has no `LoRA` for this layer (the projection runs without any delta).
#[derive(Clone)]
pub struct GemmaLayerLora {
    pub q: Option<LoraAdapter>,
    pub k: Option<LoraAdapter>,
    pub v: Option<LoraAdapter>,
    pub o: Option<LoraAdapter>,
    pub gate: Option<LoraAdapter>,
    pub up: Option<LoraAdapter>,
    pub down: Option<LoraAdapter>,
}

impl GemmaLayerLora {
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

/// Real `LoRA` applier for Gemma 2 — holds per-layer adapter sets + a scratch
/// buffer. Created once per model load, passed by `&mut` to
/// `forward_gemma2_layers` (`crate::transformer::forward_gemma2_layers`).
///
/// The `current_layer` index is advanced by `LoraApplier::next_layer` at the
/// top of each layer iteration, selecting that layer's adapter set.
#[derive(Clone)]
pub struct GemmaLora {
    /// One entry per model layer. `layers[i]` is applied during layer `i`.
    pub layers: Vec<GemmaLayerLora>,
    /// Scratch buffer for the rank-dim intermediate `A @ input`. Reused across
    /// all 7 projection points — no allocation in the hot path.
    buf: Vec<f32>,
    /// Current layer index (advanced by `next_layer`).
    current_layer: usize,
}

impl GemmaLora {
    /// Construct from per-layer adapter sets. The scratch buffer is sized to
    /// `max_rank` across all adapters.
    pub fn new(layers: Vec<GemmaLayerLora>) -> Self {
        let max_rank = layers
            .iter()
            .flat_map(|l: &GemmaLayerLora| {
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

    /// Construct an all-empty `GemmaLora` for `n_layers` layers. Equivalent to
    /// `NoLora` in effect, but typed as `GemmaLora` — useful as a placeholder
    /// when the `LoRA` file covers only some layers.
    pub fn empty(n_layers: usize) -> Self {
        Self::new((0..n_layers).map(|_| GemmaLayerLora::empty()).collect())
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

impl LoraApplier for GemmaLora {
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
    fn empty_gemma_lora_is_no_op() {
        // An all-empty GemmaLora should not modify any output — same as NoLora.
        let mut lora = GemmaLora::empty(4);
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
        let layer = GemmaLayerLora {
            q: Some(adapter),
            ..GemmaLayerLora::empty()
        };
        let mut lora = GemmaLora::new(vec![layer]);
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
            GemmaLayerLora {
                q: Some(make_adapter(1, 1, 1, 1.0)),
                ..GemmaLayerLora::empty()
            },
            GemmaLayerLora::empty(),
        ];
        let mut lora = GemmaLora::new(layers);

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
        let layers = vec![GemmaLayerLora {
            q: Some(make_adapter(4, 2, 2, 1.0)),
            k: Some(make_adapter(8, 2, 2, 1.0)),
            ..GemmaLayerLora::empty()
        }];
        let lora = GemmaLora::new(layers);
        assert_eq!(
            lora.buf.len(),
            8,
            "buf must be sized to max rank across adapters"
        );
    }
}
