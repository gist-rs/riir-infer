//! The stage compute seam + the CPU reference stage.

use std::path::Path;
use std::sync::Arc;

use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_range;
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, QwenDeltaNetTernaryWeights, effective_rotary_dim,
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

    fn step(&mut self, token: u32, pos: usize, input: &[f32]) -> &[f32] {
        let n = self.config.n_embd;
        if !self.range.is_first() {
            self.x[..n].copy_from_slice(&input[..n]);
        }
        let width = match self.range.is_last(self.config.n_layer as u32) {
            true => self.config.vocab_size,
            false => n,
        };
        let out = forward_qwen_deltanet_ternary_range(
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
        &out[..width]
    }
}
