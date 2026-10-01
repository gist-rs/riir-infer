//! The blocked-artifact container machinery shared by every derived
//! weights artifact (Plan 616 Phase 3 — extracted from the Q8 lane so a
//! new format lands as a decoder + an encode loop, never a second
//! transcription of the container/proof/commit scaffolding). The per-
//! format modules ([`super::q8_artifact`] / [`super::q4_artifact`]) own
//! their encode loops + paths; everything format-agnostic lives here:
//! the safetensors container layout, the read-back byte-identity proof,
//! the atomic commit + BLAKE3 sidecar, and the `LAYA_WEIGHTS_VARIANT`
//! resolution.
//!
//! The PROOF is the adoption's no-numerics-change evidence and runs on
//! the real bytes at conversion time, not in a test: the serialized
//! artifact is read back through the LOADER's own path and required
//! byte-identical to the in-memory fake-quant of the same weights (the
//! retained payloads resolve both sides through the same widen
//! arithmetic).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::super::{LayaError, Result};
use super::fake_quant::WeightPosture;
use super::weights::{self, Weights};

/// One container entry: tensor name, shape, dtype word, blocked/flat
/// bytes (the exact data-section span).
pub type ContainerEntry = (String, Vec<usize>, String, Vec<u8>);

/// The safetensors container layout: 8-byte LE header length + JSON
/// header (`{name: {dtype, shape, data_offsets}}`, offsets relative to
/// the data section) + the data blob, entries in the given order.
#[must_use]
pub fn serialize_container(entries: &[ContainerEntry]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data_blob: Vec<u8> = Vec::new();
    for (tensor_name, shape, dtype, data) in entries {
        let begin = data_blob.len();
        data_blob.extend_from_slice(data);
        header.insert(
            tensor_name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, data_blob.len()],
            }),
        );
    }
    let header_json = serde_json::to_string(&serde_json::Value::Object(header))
        .expect("a header built from String keys and JSON values serializes");
    let mut out_bytes = Vec::with_capacity(8 + header_json.len() + data_blob.len());
    out_bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out_bytes.extend_from_slice(header_json.as_bytes());
    out_bytes.extend_from_slice(&data_blob);
    out_bytes
}

/// THE PROOF: read the serialized artifact back through the LOADER's own
/// path and require byte-identity with the in-memory transform of the
/// same weights — tensor count, shapes, and every element's f32 BITS.
/// A mismatch refuses to emit (the caller names the artifact).
pub fn prove_readback(
    out_bytes: &[u8],
    expect_map: &HashMap<String, Weights>,
    artifact: &str,
) -> Result<()> {
    let bad = |detail: String| LayaError::Pin {
        checkpoint: "blocked-convert",
        file: artifact.to_string(),
        detail,
    };
    let got_map = weights::from_bytes(out_bytes, "blocked-convert-verify")?;
    if got_map.len() != expect_map.len() {
        return Err(bad(format!(
            "tensor count {} != {} — refusing to emit",
            got_map.len(),
            expect_map.len()
        )));
    }
    for (tensor_name, expect_w) in expect_map {
        let Some(got_w) = got_map.get(tensor_name) else {
            return Err(bad(format!(
                "tensor {tensor_name} missing from the read-back"
            )));
        };
        if got_w.shape != expect_w.shape || got_w.numel() != expect_w.numel() {
            return Err(bad(format!("tensor {tensor_name} shape drift")));
        }
        for (i, (g, e)) in got_w
            .wide_f32()
            .iter()
            .zip(expect_w.wide_f32().iter())
            .enumerate()
        {
            if g.to_bits() != e.to_bits() {
                return Err(bad(format!(
                    "tensor {tensor_name}[{i}] NOT byte-identical to the in-memory transform \
                     ({g} vs {e}) — refusing to emit"
                )));
            }
        }
    }
    Ok(())
}

/// Commit atomically: `.tmp` → rename, then the BLAKE3 sidecar (written
/// after the artifact — a sidecar without its artifact never exists; a
/// truncated artifact fails the loader's verify). Returns the digest.
pub fn commit_artifact(dir: &Path, artifact_name: &str, out_bytes: &[u8]) -> Result<String> {
    let out_path = dir.join("derived").join(artifact_name);
    std::fs::create_dir_all(out_path.parent().expect("has parent"))
        .map_err(|e| LayaError::Runtime(format!("create derived/: {e}")))?;
    let tmp = out_path.with_extension("tmp");
    std::fs::write(&tmp, out_bytes)
        .map_err(|e| LayaError::Runtime(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &out_path)
        .map_err(|e| LayaError::Runtime(format!("rename {}: {e}", out_path.display())))?;
    let digest = blake3::hash(out_bytes).to_hex().to_string();
    let side = dir.join("derived").join(format!("{artifact_name}.blake3"));
    std::fs::write(&side, format!("{digest}\n"))
        .map_err(|e| LayaError::Runtime(format!("write sidecar: {e}")))?;
    Ok(digest)
}

/// Resolve the weights file for a checkpoint dir under the
/// `LAYA_WEIGHTS_VARIANT` env: unset/`f16` → the canonical
/// `model.safetensors` (upstream-pinned, no sidecar), a derived variant
/// → its artifact + BLAKE3 sidecar (verified here — a derived artifact
/// has no upstream pin). Anything else fails loud. A derived variant is
/// NEVER auto-derived at load (a 450 MB read+write inside somebody's
/// request) — a missing artifact names the converter command.
pub fn resolve_weights_variant(
    dir: &Path,
    ckpt: &'static str,
    variants: &[(&'static str, WeightPosture, PathBuf, PathBuf)],
) -> Result<(PathBuf, Option<WeightPosture>)> {
    let requested = match std::env::var("LAYA_WEIGHTS_VARIANT") {
        Ok(v) if v.is_empty() => None,
        Ok(v) => Some(v),
        Err(_) => None,
    };
    let Some(variant) = requested else {
        return Ok((dir.join("model.safetensors"), None));
    };
    if variant == "f16" {
        return Ok((dir.join("model.safetensors"), None));
    }
    let Some((_, posture, path, side)) = variants.iter().find(|(name, _, _, _)| *name == variant)
    else {
        let names: Vec<&str> = variants.iter().map(|(n, _, _, _)| *n).collect();
        return Err(LayaError::Config {
            checkpoint: ckpt,
            detail: format!(
                "unknown LAYA_WEIGHTS_VARIANT {variant:?} — expected unset, \"f16\" or one of \
                 {names:?} (an env typo must fail loud, never fall back)"
            ),
        });
    };
    if !path.is_file() || !side.is_file() {
        return Err(LayaError::Config {
            checkpoint: ckpt,
            detail: format!(
                "LAYA_WEIGHTS_VARIANT={variant} but {} (or its .blake3 sidecar) is \
                 missing — derive it first with the variant's converter bin \
                 (see this checkpoint's derived/ dir)",
                path.display()
            ),
        });
    }
    let want = std::fs::read_to_string(side)
        .map_err(|e| LayaError::Runtime(format!("read {}: {e}", side.display())))?
        .trim()
        .to_string();
    let bytes = std::fs::read(path)
        .map_err(|e| LayaError::Runtime(format!("read {}: {e}", path.display())))?;
    let got = blake3::hash(&bytes).to_hex().to_string();
    if got != want {
        return Err(LayaError::Pin {
            checkpoint: ckpt,
            file: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            detail: format!("BLAKE3 mismatch (want {want}, got {got}) — re-derive"),
        });
    }
    Ok((path.clone(), Some(*posture)))
}
