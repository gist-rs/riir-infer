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
use riir_infer_core::types::DeltaNetLayerType;

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

    /// Plan-623 T3.4: arm the signal-driven readback — the async backend
    /// creates its staged-copy completion future NOW (right after this
    /// stage's chain enqueue, so the copy rides FIFO-after this chain and
    /// the future's resolution is the "chain done" signal value); the
    /// later [`Self::step_finish`] resolves it instead of a synchronous
    /// read, and the GPU runs the NEXT stream's chain under the host's
    /// convert+send work. No-op by default (the CPU stage has nothing to
    /// defer); pairing with [`Self::step_finish`] stays EXACTLY ONE — the
    /// same bit-for-bit law as `submit`+`finish`.
    fn schedule_async_finish(&mut self) {}

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

    /// In-process exact checkpoint (Issue 1148 rung 2): live attention rows
    /// `[0..cursor * kv_dim)` + the full GDN recurrence/conv state, copied
    /// into `ck`'s retained buffers. Byte-exact by construction (pure
    /// copies); zero allocation once the buffers are warm. `cursor` is the
    /// runner's count of written KV positions — the ternary forward never
    /// advances the cache's `fill_pos` tracker, so the tracker is NOT a
    /// live bound; the runner's cursor is.
    fn checkpoint_state(
        &self,
        cursor: usize,
        ck: &mut crate::spec_fill::StageCheckpoint,
    ) -> Result<(), String> {
        let kvd = self.config.n_kv_head * self.config.head_dim;
        let live = cursor
            .checked_mul(kvd)
            .ok_or("checkpoint: cursor * kv_dim overflow")?;
        let window = (self.range.hi - self.range.lo) as usize;
        if ck.layers.is_empty() {
            // First call: build the kind-matched slots (their buffers grow
            // into retained capacity on this and every later round).
            ck.layers.reserve(window);
            ck.layers.extend(
                self.weights.layer_types[self.range.lo as usize..self.range.hi as usize]
                    .iter()
                    .map(|lt| match lt {
                        DeltaNetLayerType::Attention => {
                            crate::spec_fill::CheckpointLayer::Attention {
                                key: Vec::new(),
                                value: Vec::new(),
                            }
                        }
                        DeltaNetLayerType::DeltaNet => {
                            crate::spec_fill::CheckpointLayer::DeltaNet {
                                rec: Vec::new(),
                                conv: Vec::new(),
                            }
                        }
                    }),
            );
        } else if ck.layers.len() != window {
            return Err(format!(
                "checkpoint: ck has {} layers, stage window has {} — a checkpoint cannot move between stages",
                ck.layers.len(),
                self.weights.layer_types.len()
            ));
        }
        for (i, lt) in self.weights.layer_types[self.range.lo as usize..self.range.hi as usize]
            .iter()
            .enumerate()
        {
            let gi = self.range.lo as usize + i;
            let slot = &mut ck.layers[i];
            match (*lt, slot) {
                (
                    DeltaNetLayerType::Attention,
                    crate::spec_fill::CheckpointLayer::Attention { key, value },
                ) => {
                    if self.cache.kv_cache.sliding_capacity[gi] != 0 {
                        return Err(format!(
                            "checkpoint: layer {gi} is sliding-window (ring layout) — the spec rewind is unbounded-only"
                        ));
                    }
                    let layer = &self.cache.kv_cache.layers[gi];
                    if layer.key.len() < live || layer.value.len() < live {
                        return Err(format!(
                            "checkpoint: cursor {cursor} × kv_dim {kvd} = {live} rows > layer {gi} buffer ({})",
                            layer.key.len()
                        ));
                    }
                    copy_retained(key, &layer.key[..live]);
                    copy_retained(value, &layer.value[..live]);
                }
                (
                    DeltaNetLayerType::DeltaNet,
                    crate::spec_fill::CheckpointLayer::DeltaNet { rec, conv },
                ) => {
                    copy_retained(rec, &self.cache.deltanet_state.recurrent_states[gi]);
                    copy_retained(conv, &self.cache.deltanet_state.conv_states[gi]);
                }
                (lt, slot) => {
                    return Err(format!(
                        "checkpoint: layer {gi} is {lt:?} but the checkpoint slot is {:?} — kinds never renegotiate",
                        slot.kind_name()
                    ));
                }
            }
        }
        ck.cursor = cursor;
        Ok(())
    }

    /// Copy a checkpoint back (byte-exact on every restored row; rows
    /// beyond `ck.cursor` stay stale — the forward reads/writes rows
    /// positionally and never touches them before rewrite). The `fill_pos`
    /// tracker is re-armed to `ck.cursor` for consistency with the codec
    /// path's observable state (the ternary forward itself ignores it).
    fn restore_checkpoint(&mut self, ck: &crate::spec_fill::StageCheckpoint) -> Result<(), String> {
        let window = (self.range.hi - self.range.lo) as usize;
        if ck.layers.len() != window {
            return Err(format!(
                "restore_checkpoint: ck has {} layers, stage window has {window}",
                ck.layers.len()
            ));
        }
        let kvd = self.config.n_kv_head * self.config.head_dim;
        let live = ck
            .cursor
            .checked_mul(kvd)
            .ok_or("restore_checkpoint: cursor * kv_dim overflow")?;
        for (i, lt) in self.weights.layer_types[self.range.lo as usize..self.range.hi as usize]
            .iter()
            .enumerate()
        {
            let gi = self.range.lo as usize + i;
            match (&ck.layers[i], *lt) {
                (
                    crate::spec_fill::CheckpointLayer::Attention { key, value },
                    DeltaNetLayerType::Attention,
                ) => {
                    let layer = &mut self.cache.kv_cache.layers[gi];
                    if key.len() != live || value.len() != live {
                        return Err(format!(
                            "restore_checkpoint: ck rows ({},{}) != cursor {} × kv_dim {kvd}",
                            key.len(),
                            value.len(),
                            ck.cursor
                        ));
                    }
                    if layer.key.len() < live || layer.value.len() < live {
                        return Err(format!(
                            "restore_checkpoint: layer {gi} buffer too small for {} live rows",
                            live
                        ));
                    }
                    layer.key[..live].copy_from_slice(key);
                    layer.value[..live].copy_from_slice(value);
                }
                (
                    crate::spec_fill::CheckpointLayer::DeltaNet { rec, conv },
                    DeltaNetLayerType::DeltaNet,
                ) => {
                    let dst_rec = &mut self.cache.deltanet_state.recurrent_states[gi];
                    let dst_conv = &mut self.cache.deltanet_state.conv_states[gi];
                    if dst_rec.len() != rec.len() || dst_conv.len() != conv.len() {
                        return Err(format!(
                            "restore_checkpoint: layer {gi} gdn shape ({},{}) != ck ({},{})",
                            dst_rec.len(),
                            dst_conv.len(),
                            rec.len(),
                            conv.len()
                        ));
                    }
                    dst_rec.copy_from_slice(rec);
                    dst_conv.copy_from_slice(conv);
                }
                (slot, lt) => {
                    return Err(format!(
                        "restore_checkpoint: layer {gi} is {lt:?} but the checkpoint slot is {:?}",
                        slot.kind_name()
                    ));
                }
            }
        }
        self.cache.kv_cache.set_fill_pos(ck.cursor);
        Ok(())
    }
}

/// Copy `src` into a retained buffer: one realloc-free memcpy once warm
/// (clear keeps the capacity; extend only grows on a genuine shape
/// change). This is the whole checkpoint cost — Issue 1148's point.
fn copy_retained(dst: &mut Vec<f32>, src: &[f32]) {
    dst.clear();
    dst.extend_from_slice(src);
}

impl TernaryCpuStage {
    /// TEST SEAM (#[doc(hidden)]): NaN-poison every attention layer's KV
    /// rows `[from_pos..to_pos)` — the instrument behind the
    /// cursor-overwrite truncate-law pin (the G1 battery's
    /// `kv_cursor_overwrite_truncate_law_nan_poison`). Never called by
    /// production code.
    #[doc(hidden)]
    pub fn poison_kv_rows_beyond(&mut self, from_pos: usize, to_pos: usize) {
        let kvd = self.config.n_kv_head * self.config.head_dim;
        for (i, lt) in self.weights.layer_types[self.range.lo as usize..self.range.hi as usize]
            .iter()
            .enumerate()
        {
            let gi = self.range.lo as usize + i;
            if *lt != DeltaNetLayerType::Attention {
                continue;
            }
            let layer = &mut self.cache.kv_cache.layers[gi];
            let lo = from_pos * kvd;
            let hi = (to_pos * kvd).min(layer.key.len());
            for v in layer.key[lo..hi].iter_mut() {
                *v = f32::NAN;
            }
            for v in layer.value[lo..hi].iter_mut() {
                *v = f32::NAN;
            }
        }
    }
}
