//! The stage compute seam + the CPU reference stage.

use std::path::Path;
use std::sync::Arc;

use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_range;
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, QwenDeltaNetTernaryWeights, WeightScope,
    effective_rotary_dim,
};
use riir_infer_core::disaggregated::geometry_fingerprint;
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::multi_host::{FrameDtype, LayerRange, StageHandshake};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;

/// One layer window of one model. The runner owns the wire; the stage
/// owns its weights view, cache and scratch.
pub trait StageCompute {
    fn handshake(&self) -> &StageHandshake;
    fn n_layer(&self) -> u32;
    /// Width of the hidden activation crossing a stage boundary.
    fn hidden_dim(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// Position capacity (`config.block_size` at construction): a step at
    /// `pos >= max_positions()` refuses in the runner, never overruns a
    /// KV row.
    fn max_positions(&self) -> usize;
    /// Run this stage's window at `pos`. The first stage reads `token`
    /// (embedding) and ignores `input`; later stages read `input`
    /// (`hidden_dim` wide). Returns the hidden (non-last stage) or the
    /// logits (last stage).
    fn step(&mut self, token: u32, pos: usize, input: &[f32]) -> &[f32];

    /// Enqueue this stage's decode step WITHOUT waiting for the result
    /// (T2.4 fused batch step). Async backends (GPU) enqueue the kernel
    /// chain and return; the CPU default computes it now into the stage's
    /// own output buffer. Paired with EXACTLY ONE [`Self::step_finish`],
    /// read back in submission order — the fused batch submits every
    /// stream's chain first so the queue runs them back to back with no
    /// inter-chain idle, then finishes each in FIFO order. Per-stream
    /// arithmetic (kernel order, operands, fold order) is unchanged, so
    /// `submit`+`finish` == `step` bit-for-bit by construction.
    fn step_submit(&mut self, token: u32, pos: usize, input: &[f32]);

    /// Complete the step enqueued by [`Self::step_submit`] and return its
    /// output slice (hidden for a non-last stage, logits for the last),
    /// exactly as [`Self::step`] would have returned. The slice borrows
    /// the stage — copy/argmax before the next submit on this stage.
    fn step_finish(&mut self) -> &[f32];

    /// Run this stage's window over one prefill CHUNK of consecutive
    /// positions `base_pos..base_pos + n` (T2.3b). The first stage reads
    /// `tokens` (`n` ids); later stages read `input` (`n` rows of
    /// `hidden_dim`). Into `out`: every row's hidden (non-last stage), or
    /// the final row's logits (last stage, only when `want_logits` — else
    /// `out` is left empty). `prompt_len` is the whole prompt's length.
    ///
    /// Default: [`Self::step`] per position (bit-identical to the
    /// per-position protocol by construction). A backend with a batched
    /// prefill overrides it — the GPU stage (`prefill_range_chunk`).
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        input: &[f32],
        base_pos: usize,
        prompt_len: usize,
        want_logits: bool,
        out: &mut Vec<f32>,
    ) {
        let _ = prompt_len;
        let range = self.handshake().range;
        let (first, last) = (range.is_first(), range.is_last(self.n_layer()));
        let h = self.hidden_dim();
        let n = match first {
            true => tokens.len(),
            false => input.len() / h,
        };
        out.clear();
        for i in 0..n {
            let tok = tokens.get(i).copied().unwrap_or(0);
            let row = match first {
                true => &input[..0],
                false => &input[i * h..(i + 1) * h],
            };
            let r = self.step(tok, base_pos + i, row);
            match (last, want_logits && i + 1 == n) {
                (false, _) => out.extend_from_slice(r),
                (true, true) => out.extend_from_slice(r),
                (true, false) => {}
            }
        }
    }
}

/// Arch tag for the ternary DeltaNet hybrid (Bonsai-2 family).
pub const ARCH_QWEN35_DELTANET_TERNARY: &str = "qwen35-deltanet-ternary";

/// The handshake a ternary-DeltaNet stage presents. Backend-agnostic: a
/// CPU and a GPU stage of one artifact tile legally (the pipeline is then
/// token-correct but not bit-identical to either 1-box backend — the G1
/// bit bar holds per backend).
pub fn ternary_handshake(identity: ArtifactIdentity, range: LayerRange) -> StageHandshake {
    StageHandshake {
        arch: ARCH_QWEN35_DELTANET_TERNARY.into(),
        geometry_fingerprint: identity.geometry,
        artifact_blake3: identity.artifact,
        range,
        dtype: FrameDtype::F32,
    }
}

/// The weight scope a stage over `range` needs (Plan 623 T2.2c): its layer
/// window, the embedding iff first, `final_norm` + `lm_head` iff last.
/// Feed it to `load_qwen_deltanet_ternary_weights_gguf_scoped` so a stage
/// holds only its own slice of the model.
pub fn stage_scope(range: LayerRange, n_layer: u32) -> WeightScope {
    WeightScope::stage(range.lo as usize, range.hi as usize, n_layer as usize)
}

/// Refuse a weights view that does not hold every tensor `range` reads —
/// a scoped load for a different window would otherwise dispatch empty
/// layers and compute plausible garbage.
pub fn require_coverage(
    weights: &QwenDeltaNetTernaryWeights,
    range: LayerRange,
    n_layer: u32,
) -> Result<(), String> {
    let scope = stage_scope(range, n_layer);
    match weights.covers(&scope) {
        true => Ok(()),
        false => Err(format!(
            "weights do not cover stage {range:?} (needs {scope:?}) — load with that scope or the full model"
        )),
    }
}

/// The two identity digests a stage handshake carries, for one artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactIdentity {
    /// Metadata-plane geometry digest (`disaggregated::geometry_fingerprint`
    /// — the Issue-028 substrate, every key the config derivation reads).
    pub geometry: [u8; 32],
    /// BLAKE3 of the artifact bytes, streamed.
    pub artifact: [u8; 32],
}

impl ArtifactIdentity {
    pub fn of_gguf(path: &Path) -> Result<Self, String> {
        let gguf = GgufFile::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let geometry = geometry_fingerprint(&gguf);
        let mut h = blake3::Hasher::new();
        let f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        h.update_reader(f)
            .map_err(|e| format!("hash {}: {e}", path.display()))?;
        Ok(Self {
            geometry,
            artifact: *h.finalize().as_bytes(),
        })
    }
}

/// CPU reference stage over `forward_qwen_deltanet_ternary_range` — the
/// same compiled loop the 1-box forward delegates to (Plan 623 T1.1).
pub struct TernaryCpuStage {
    weights: Arc<QwenDeltaNetTernaryWeights>,
    config: Config,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    rope: RopeFreqTable,
    x: Vec<f32>,
    range: LayerRange,
    handshake: StageHandshake,
}

impl TernaryCpuStage {
    pub fn new(
        config: Config,
        weights: Arc<QwenDeltaNetTernaryWeights>,
        range: LayerRange,
        identity: ArtifactIdentity,
    ) -> Result<Self, String> {
        range.validate(config.n_layer as u32)?;
        require_coverage(&weights, range, config.n_layer as u32)?;
        let cache = HybridCache::with_layer_types(&config, &weights.layer_types);
        let scratch = HybridForwardScratch::new(&config);
        let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
        let x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        let handshake = ternary_handshake(identity, range);
        Ok(Self {
            weights,
            config,
            cache,
            scratch,
            rope,
            x,
            range,
            handshake,
        })
    }
}

impl StageCompute for TernaryCpuStage {
    fn handshake(&self) -> &StageHandshake {
        &self.handshake
    }
    fn n_layer(&self) -> u32 {
        self.config.n_layer as u32
    }
    fn hidden_dim(&self) -> usize {
        self.config.n_embd
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn max_positions(&self) -> usize {
        self.config.block_size
    }

    /// Submit + finish — the fused batch split ([`StageCompute::step_submit`])
    /// composes back into exactly this call order, so the solo path and the
    /// batch path are the same computation by construction.
    fn step(&mut self, token: u32, pos: usize, input: &[f32]) -> &[f32] {
        self.step_submit(token, pos, input);
        self.step_finish()
    }

    fn step_submit(&mut self, token: u32, pos: usize, input: &[f32]) {
        let n = self.config.n_embd;
        if !self.range.is_first() {
            self.x[..n].copy_from_slice(&input[..n]);
        }
        forward_qwen_deltanet_ternary_range(
            &mut self.x,
            &self.weights,
            &mut self.cache,
            token as usize,
            pos,
            &self.config,
            &mut self.scratch,
            &self.rope,
            self.range,
        );
    }

    fn step_finish(&mut self) -> &[f32] {
        let width = if self.range.is_last(self.config.n_layer as u32) {
            self.config.vocab_size
        } else {
            self.config.n_embd
        };
        &self.x[..width]
    }
}

/// Byte-exact state snapshot/restore over the Issue-028 handoff segment
/// (`HybridStateSegment`): the T5.1 v0 rewind. The GDN journal's rank-1
/// unwind is the FAST path for a later rung — this is the CORRECT path,
/// existing substrate, zero forward surgery. Unbounded (non-sliding) KV
/// only: the segment captures rows `[0..fill_pos)` and restores the fill
/// marker, so a snapshot at the pre-verify cursor + restore is the
/// rewind, bit-exact.
impl crate::spec_fill::SpecStage for TernaryCpuStage {
    fn snapshot_state(&self, out: &mut Vec<u8>) -> Result<(), String> {
        out.clear();
        riir_infer_core::multi_host::HybridStateSegment::encode_into(
            &self.cache,
            &self.weights.layer_types,
            self.range,
            out,
        )
    }

    fn restore_state(&mut self, bytes: &[u8]) -> Result<(), String> {
        riir_infer_core::multi_host::HybridStateSegment::restore_into(
            bytes,
            &mut self.cache,
            &self.weights.layer_types,
            self.range,
        )
    }
}
