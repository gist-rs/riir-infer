//! GPU stage (Metal / CubeCL) over the Plan 623 T1.5 range seams of
//! [`TernaryDeltanetGpuForward`] — `forward_range_dispatch` +
//! `set_hidden_state` / `read_hidden_state` / `set_pos`. The range funnel
//! dispatches the same kernels in the same order as the 1-box funnel, so a
//! GPU-GPU pipeline is bit-identical to the 1-box GPU forward (T1.5 G1).
//!
//! PoC limits (tracked in Plan 623 T2.2c): the forwarder uploads the WHOLE
//! model on every stage (no range-scoped weight upload yet — so the lane
//! does not yet buy capacity), and each handoff round-trips the hidden via
//! a host `Vec` (T1.5's documented per-call upload cost).

use std::sync::Arc;

use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::multi_host::{LayerRange, StageHandshake};
use riir_infer_core::types::Config;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

use crate::stage::{ArtifactIdentity, StageCompute, ternary_handshake};

pub struct TernaryGpuStage {
    fwd: TernaryDeltanetGpuForward,
    weights: Arc<QwenDeltaNetTernaryWeights>,
    range: LayerRange,
    n_layer: u32,
    n_embd: usize,
    vocab: usize,
    max_positions: usize,
    handshake: StageHandshake,
    out: Vec<f32>,
}

impl TernaryGpuStage {
    /// `config.block_size` is the position capacity: clamp it to the
    /// sequence bound BEFORE calling (the Issue-994 VRAM gate refuses the
    /// loader's context-sized default).
    pub fn new(
        config: &Config,
        weights: Arc<QwenDeltaNetTernaryWeights>,
        range: LayerRange,
        identity: ArtifactIdentity,
    ) -> Result<Self, String> {
        range.validate(config.n_layer as u32)?;
        let ctx = CubeCLContext::new().map_err(|e| format!("GPU init: {e:?}"))?;
        let fwd = TernaryDeltanetGpuForward::new(&ctx, config, &weights);
        Ok(Self {
            fwd,
            weights,
            range,
            n_layer: config.n_layer as u32,
            n_embd: config.n_embd,
            vocab: config.vocab_size,
            max_positions: config.block_size,
            handshake: ternary_handshake(identity, range),
            out: Vec::new(),
        })
    }
}

impl StageCompute for TernaryGpuStage {
    fn handshake(&self) -> &StageHandshake {
        &self.handshake
    }
    fn n_layer(&self) -> u32 {
        self.n_layer
    }
    fn hidden_dim(&self) -> usize {
        self.n_embd
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn max_positions(&self) -> usize {
        self.max_positions
    }

    fn step(&mut self, token: u32, pos: usize, input: &[f32]) -> &[f32] {
        self.fwd.set_pos(pos);
        match self.range.is_first() {
            true => self.fwd.set_input_token(&self.weights, token as usize),
            false => self.fwd.set_hidden_state(&input[..self.n_embd]),
        }
        self.fwd.forward_range_dispatch(self.range);
        let (out, width) = match self.range.is_last(self.n_layer) {
            true => (self.fwd.read_logits(), self.vocab),
            false => (self.fwd.read_hidden_state(), self.n_embd),
        };
        self.out = out;
        &self.out[..width]
    }
}
