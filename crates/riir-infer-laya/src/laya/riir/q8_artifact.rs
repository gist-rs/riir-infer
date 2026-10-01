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
//! proof runs on the real bytes at conversion time, not in a test.

use std::path::{Path, PathBuf};

use super::super::{LayaError, Result};
use super::fake_quant::{BLOCK, q8_quant_of, q8_scale_bits, q8_scale_f32};
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
/// (written to `.tmp`, proven, then renamed — a failed conversion never
/// leaves a half artifact behind).
pub fn convert_checkpoint(dir: &Path) -> Result<Q8ConvertReport> {
    let src = dir.join("model.safetensors");
    let f16_map = weights::load(&src, "q8-convert")?;

    // Encode: every >=2D tensor to Q8_0 blocks; 1D tensors carried F16
    // (widen→narrow is bit-exact for f16-sourced data — weights.rs's
    // roundtrip law — so passthrough through the widened map is safe).
    let mut entries: Vec<(String, Vec<usize>, String, Vec<u8>)> = Vec::new();
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

    // Serialize the container (the safetensors layout: 8-byte LE header
    // length + JSON header + data section).
    let mut header = serde_json::Map::new();
    let mut data_blob: Vec<u8> = Vec::with_capacity(q8_bytes as usize);
    for (tensor_name, shape, dtype, data) in &entries {
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
        .map_err(|e| LayaError::Runtime(format!("header serialize: {e}")))?;
    let mut out_bytes = Vec::with_capacity(8 + header_json.len() + data_blob.len());
    out_bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out_bytes.extend_from_slice(header_json.as_bytes());
    out_bytes.extend_from_slice(&data_blob);

    // PROOF before commit: read the serialized artifact back through the
    // LOADER's own path and require byte-identity with the in-memory
    // fake-quant of the same weights. This is the adoption's
    // no-numerics-change evidence, produced on the real bytes. The
    // read-back's >=2D payloads are RETAINED Q8 (Plan 616 Phase 1) — the
    // comparison resolves both sides through the same widen arithmetic.
    let mut expect_map = f16_map;
    let _expect_rep =
        super::fake_quant::fake_quant_q8_map(&mut expect_map).map_err(LayaError::Runtime)?;
    let got_map = weights::from_bytes(&out_bytes, "q8-convert-verify")?;
    if got_map.len() != expect_map.len() {
        return Err(LayaError::Pin {
            checkpoint: "q8-convert",
            file: "model.q8.safetensors".into(),
            detail: format!(
                "tensor count {} != {} — refusing to emit",
                got_map.len(),
                expect_map.len()
            ),
        });
    }
    for (tensor_name, expect_w) in &expect_map {
        let Some(got_w) = got_map.get(tensor_name) else {
            return Err(LayaError::Pin {
                checkpoint: "q8-convert",
                file: "model.q8.safetensors".into(),
                detail: format!("tensor {tensor_name} missing from the read-back"),
            });
        };
        if got_w.shape != expect_w.shape || got_w.numel() != expect_w.numel() {
            return Err(LayaError::Pin {
                checkpoint: "q8-convert",
                file: "model.q8.safetensors".into(),
                detail: format!("tensor {tensor_name} shape drift"),
            });
        }
        for (i, (g, e)) in got_w
            .wide_f32()
            .iter()
            .zip(expect_w.wide_f32().iter())
            .enumerate()
        {
            if g.to_bits() != e.to_bits() {
                return Err(LayaError::Pin {
                    checkpoint: "q8-convert",
                    file: "model.q8.safetensors".into(),
                    detail: format!(
                        "tensor {tensor_name}[{i}] NOT byte-identical to the fake-quant \
                         path ({g} vs {e}) — refusing to emit"
                    ),
                });
            }
        }
    }

    // Commit atomically: .tmp → rename, sidecar after the artifact.
    let out_path = q8_artifact_path(dir);
    std::fs::create_dir_all(out_path.parent().expect("has parent"))
        .map_err(|e| LayaError::Runtime(format!("create derived/: {e}")))?;
    let tmp = out_path.with_extension("safetensors.tmp");
    std::fs::write(&tmp, &out_bytes)
        .map_err(|e| LayaError::Runtime(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, &out_path)
        .map_err(|e| LayaError::Runtime(format!("rename {}: {e}", out_path.display())))?;
    let digest = blake3::hash(&out_bytes).to_hex().to_string();
    std::fs::write(q8_sidecar_path(dir), format!("{digest}\n"))
        .map_err(|e| LayaError::Runtime(format!("write sidecar: {e}")))?;

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
/// `LAYA_WEIGHTS_VARIANT` env: unset/`f16` → the canonical
/// `model.safetensors` (upstream-pinned), `q8` → the derived artifact
/// (BLAKE3 sidecar verified here — it has no upstream pin). Anything
/// else fails loud. The q8 variant is NEVER auto-derived at load (a
/// 850 MB read+write inside somebody's request) — a missing artifact
/// names the converter command instead.
pub fn resolve_weights_file(dir: &Path, ckpt: &'static str) -> Result<(PathBuf, bool)> {
    match std::env::var("LAYA_WEIGHTS_VARIANT").as_deref() {
        Ok("") | Err(_) => Ok((dir.join("model.safetensors"), false)),
        Ok("f16") => Ok((dir.join("model.safetensors"), false)),
        Ok("q8") => {
            let path = q8_artifact_path(dir);
            let side = q8_sidecar_path(dir);
            if !path.is_file() || !side.is_file() {
                return Err(LayaError::Config {
                    checkpoint: ckpt,
                    detail: format!(
                        "LAYA_WEIGHTS_VARIANT=q8 but {} (or its .blake3 sidecar) is \
                         missing — derive it first: cargo run --release --features \
                         laya-riir -p riir-infer-laya --bin laya-quant8 -- <checkpoint-dir>",
                        path.display()
                    ),
                });
            }
            let want = std::fs::read_to_string(&side)
                .map_err(|e| LayaError::Runtime(format!("read {}: {e}", side.display())))?
                .trim()
                .to_string();
            let bytes = std::fs::read(&path)
                .map_err(|e| LayaError::Runtime(format!("read {}: {e}", path.display())))?;
            let got = blake3::hash(&bytes).to_hex().to_string();
            if got != want {
                return Err(LayaError::Pin {
                    checkpoint: ckpt,
                    file: "derived/model.q8.safetensors".into(),
                    detail: format!("BLAKE3 mismatch (want {want}, got {got}) — re-derive"),
                });
            }
            Ok((path, true))
        }
        Ok(other) => Err(LayaError::Config {
            checkpoint: ckpt,
            detail: format!(
                "unknown LAYA_WEIGHTS_VARIANT {other:?} — expected unset, \"f16\" or \"q8\" \
                 (an env typo must fail loud, never fall back)"
            ),
        }),
    }
}
