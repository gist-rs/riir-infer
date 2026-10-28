//! GPU stage (Metal / CubeCL) over the Plan 623 T1.5 range seams of
//! [`TernaryDeltanetGpuForward`] — `forward_range_dispatch` +
//! `set_hidden_state` / `read_hidden_state` / `set_pos`. The range funnel
//! dispatches the same kernels in the same order as the 1-box funnel, so a
//! GPU-GPU pipeline is bit-identical to the 1-box GPU forward (T1.5 G1).
//!
//! Capacity (Plan 623 T2.2c): hand it a SCOPED load
//! ([`crate::stage::stage_scope`]) and the forwarder uploads — and
//! allocates recurrent/KV state for — only this stage's layers (+ embed on
//! the first stage, `lm_head` on the last). Remaining PoC cost: each
//! handoff round-trips the hidden via a host `Vec`.

use std::sync::Arc;

use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::multi_host::{LayerRange, StageHandshake};
use riir_infer_core::types::Config;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

use crate::stage::{ArtifactIdentity, StageCompute, require_coverage, ternary_handshake};

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
    tok_scratch: Vec<usize>,
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
        require_coverage(&weights, range, config.n_layer as u32)?;
        let ctx = CubeCLContext::new().map_err(|e| format!("GPU init: {e:?}"))?;
        // Issue 1146 rung 2a: route the prefill GEMMs through the tensor
        // zero-copy path (falls through to simdgroup per site on failure).
        #[cfg(all(feature = "gpu_metal_tensor", target_os = "macos"))]
        riir_infer_gpu::set_prefill_use_metal_tensor_zerocopy(true);
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
            tok_scratch: Vec::new(),
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

    /// Submit + finish — the fused batch split ([`StageCompute::step_submit`])
    /// composes back into exactly this call order, so the solo path and the
    /// batch path are the same kernel sequence by construction.
    fn step(&mut self, token: u32, pos: usize, input: &[f32]) -> &[f32] {
        self.step_submit(token, pos, input);
        self.step_finish()
    }

    fn step_submit(&mut self, token: u32, pos: usize, input: &[f32]) {
        self.fwd.set_pos(pos);
        match self.range.is_first() {
            true => self.fwd.set_input_token(&self.weights, token as usize),
            false => self.fwd.set_hidden_state(&input[..self.n_embd]),
        }
        // Enqueue only — the readback happens in `step_finish`. The fused
        // batch submits every stream's chain before any readback, so the
        // shared queue runs the chains back to back with no inter-chain
        // idle (T2.4).
        self.fwd.forward_range_dispatch(self.range);
    }

    fn step_finish(&mut self) -> &[f32] {
        let (out, width) = match self.range.is_last(self.n_layer) {
            true => (self.fwd.read_logits(), self.vocab),
            false => (self.fwd.read_hidden_state(), self.n_embd),
        };
        self.out = out;
        &self.out[..width]
    }

    /// The batched range prefill (`prefill_range_chunk`): one dispatch
    /// chain per chunk instead of one decode step per position — the
    /// weights stream once per chunk, not once per token.
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        input: &[f32],
        base_pos: usize,
        prompt_len: usize,
        want_logits: bool,
        out: &mut Vec<f32>,
    ) {
        self.tok_scratch.clear();
        self.tok_scratch.extend(tokens.iter().map(|&t| t as usize));
        let hidden_in = match self.range.is_first() {
            true => None,
            false => Some(input),
        };
        // The forwarder hands back an owned readback — move it, no copy.
        *out = self.fwd.prefill_range_chunk(
            &self.tok_scratch,
            hidden_in,
            base_pos,
            prompt_len,
            self.range,
            want_logits,
        );
    }
}
