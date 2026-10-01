//! The Q8 artifact lane (instinct issue 018 Lane D2a — the STORAGE tier):
//! convert a checkpoint's F16 `model.safetensors` into
//! `derived/model.q8.safetensors` (same container, >=2D tensors as Q8_0
//! blocks, 1D norms/biases carried F16 untouched) plus a BLAKE3 sidecar,
//! and load it back with byte-identical numerics.
//!
//! Why this tier first: the fake-quant probe (Bench 0046) proved the Q8
//! weight error is retention-safe; an artifact decoded by
//! [`crate::laya::riir::weights`]'s Q8_0 widen produces the SAME f32
//! values the probe computed in memory (the decode arithmetic is the
//! probe's own — `d · q`), so adoption moves the file/cache footprint
//! (842 MB → ~424 MB per checkpoint) with **no numerics change at all**
//! — the cell re-seat is satisfied by the byte-equality proof below, and
//! the probe's reads remain the reads. Device-resident quantized buffers
//! (the memory tier, D2b) are the follow-up with real kernel work.
//!
//! The converter REFUSES to emit an artifact whose read-back is not
//! byte-identical to the in-memory fake-quant of the same weights — the
//! proof runs on the real bytes at conversion time, not in a test. The
//! container/proof/commit scaffolding is the shared
//! [`super::blocked_artifact`] machinery (Plan 616 Phase 3); this module
//! owns the Q8_0 encode loop and the paths.

use std::path::{Path, PathBuf};

use super::super::{LayaError, Result};
use super::blocked_artifact::{
    ContainerEntry, commit_artifact, prove_readback, resolve_weights_variant, serialize_container,
};
use super::fake_quant::{BLOCK, WeightPosture, q8_quant_of, q8_scale_bits, q8_scale_f32};
use super::weights::{self, f32_to_f16_bits};

/// The derived artifact's path under a checkpoint dir.
#[must_use]
pub fn q8_artifact_path(dir: &Path) -> PathBuf {
    dir.join("derived").join("model.q8.safetensors")
}

/// The sidecar's path (BLAKE3 hex, the converter writes, the loader
/// verifies — a derived artifact has no upstream pin, so this is its
/// tamper-evidence).
#[must_use]
pub fn q8_sidecar_path(dir: &Path) -> PathBuf {
    dir.join("derived").join("model.q8.safetensors.blake3")
}

/// The converter (`laya-quant8 <checkpoint-dir>`): F16 map in, Q8_0
/// artifact + sidecar out. Round-trip verified before the file commits
/// (the shared machinery's atomic write + proof — a failed conversion
/// never leaves a half artifact behind).
pub fn convert_checkpoint(dir: &Path) -> Result<Q8ConvertReport> {
    let src = dir.join("model.safetensors");
    let f16_map = weights::load(&src, "q8-convert")?;

    // Encode: every >=2D tensor to Q8_0 blocks; 1D tensors carried F16
    // (widen→narrow is bit-exact for f16-sourced data — weights.rs's
    // roundtrip law — so passthrough through the widened map is safe).
    let mut entries: Vec<ContainerEntry> = Vec::new();
    let mut names: Vec<&String> = f16_map.keys().collect();
    names.sort_unstable();
    let mut quantized_tensors = 0usize;
    let mut quantized_elements = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut f16_bytes = 0u64;
    let mut q8_bytes = 0u64;
    for tensor_name in &names {
        let w = &f16_map[tensor_name.as_str()];
        let numel: usize = w.shape.iter().product();
        f16_bytes += numel as u64 * 2;
        if w.shape.len() < 2 {
            skipped.push((*tensor_name).clone());
            let mut data = Vec::with_capacity(numel * 2);
            for &v in w.wide_f32() {
                data.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            q8_bytes += data.len() as u64;
            entries.push(((*tensor_name).clone(), w.shape.clone(), "F16".into(), data));
            continue;
        }
        quantized_tensors += 1;
        quantized_elements += numel;
        let mut data = Vec::with_capacity(numel / BLOCK * 34 + 34);
        for block in w.wide_f32().chunks(BLOCK) {
            let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let bits = q8_scale_bits(amax);
            data.extend_from_slice(&bits.to_le_bytes());
            let d = q8_scale_f32(bits);
            for &x in block {
                data.push(q8_quant_of(x, d) as u8);
            }
        }
        q8_bytes += data.len() as u64;
        entries.push(((*tensor_name).clone(), w.shape.clone(), "Q8_0".into(), data));
    }

    // Serialize → PROOF → commit (the shared machinery; the proof reads
    // the serialized artifact back through the LOADER's own path and
    // requires byte-identity with the in-memory fake-quant — the
    // adoption's no-numerics-change evidence on the real bytes).
    let out_bytes = serialize_container(&entries);
    let mut expect_map = f16_map;
    let _expect_rep =
        super::fake_quant::fake_quant_q8_map(&mut expect_map).map_err(LayaError::Runtime)?;
    prove_readback(&out_bytes, &expect_map, "model.q8.safetensors")?;
    let digest = commit_artifact(dir, "model.q8.safetensors", &out_bytes)?;

    Ok(Q8ConvertReport {
        tensors: entries.len(),
        quantized_tensors,
        quantized_elements,
        skipped_tensors: skipped,
        f16_bytes,
        q8_bytes,
        digest,
    })
}

/// What a conversion did (printed by the converter, kept with the
/// artifact's record).
#[derive(Debug, serde::Serialize)]
pub struct Q8ConvertReport {
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub quantized_elements: usize,
    pub skipped_tensors: Vec<String>,
    pub f16_bytes: u64,
    pub q8_bytes: u64,
    pub digest: String,
}

/// Resolve the weights file for a checkpoint dir under the
/// `LAYA_WEIGHTS_VARIANT` env: unset/`f16` → the canonical F16 file, a
/// derived variant → its artifact (sidecar-verified). The variant table
/// carries BOTH formats (q8 — Plan 616 Phase 1; q4 — Phase 3); the
/// shared resolution lives in [`super::blocked_artifact`]. The bool is
/// true when the loaded file is a derived artifact (either format).
pub fn resolve_weights_file(dir: &Path, ckpt: &'static str) -> Result<(PathBuf, bool)> {
    let (path, posture) = resolve_weights_variant(
        dir,
        ckpt,
        &[
            (
                "q8",
                WeightPosture::Q8Artifact,
                q8_artifact_path(dir),
                q8_sidecar_path(dir),
            ),
            (
                "q4",
                WeightPosture::Q4Artifact,
                super::q4_artifact::q4_artifact_path(dir),
                super::q4_artifact::q4_sidecar_path(dir),
            ),
        ],
    )?;
    Ok((path, posture.is_some()))
}

/// [`resolve_weights_file`]'s posture-bearing twin: the SAME resolution
/// with the resolved [`WeightPosture`] (the artifact's own posture —
/// `q8-artifact` or `q4-artifact`) instead of a bare bool, for callers
/// that label the record per format.
pub fn resolve_weights_posture(
    dir: &Path,
    ckpt: &'static str,
) -> Result<(PathBuf, Option<WeightPosture>)> {
    resolve_weights_variant(
        dir,
        ckpt,
        &[
            (
                "q8",
                WeightPosture::Q8Artifact,
                q8_artifact_path(dir),
                q8_sidecar_path(dir),
            ),
            (
                "q4",
                WeightPosture::Q4Artifact,
                super::q4_artifact::q4_artifact_path(dir),
                super::q4_artifact::q4_sidecar_path(dir),
            ),
        ],
    )
}
