//! The Q4 artifact lane (Plan 616 Phase 3 — the house blocked family's
//! 4-bit rung, the [`super::q8_artifact`] storage tier's twin at half the
//! byte rate): convert a checkpoint's F16 `model.safetensors` into
//! `derived/model.q4.safetensors` (same container, >=2D tensors as Q4_0
//! blocks — one f16 scale + 16 packed-nibble bytes per 32 weights, 1D
//! norms/biases carried F16 untouched) plus a BLAKE3 sidecar, and load
//! it back with byte-identical numerics AT THE Q4 GRID.
//!
//! The format is the house Q8_0 law's own 4-bit shape (NOT GGML's
//! unsigned `d·(q−8)`): `d = f16(amax/7)`, signed nibbles in [-7, +7]
//! (both ±amax on the grid ends up to the f16 scale's own rounding, zero
//! exact), packed GGML-style (even element low nibble, odd high). The
//! decode sign-extends, so GGML-produced `-8` nibbles interop.
//!
//! ⚠ ADOPTION IS NOT THIS MODULE: the Q4 grid's weight error is ~15× the
//! Q8 grid's (`amax/7` steps), and its retention is D1-PRICED
//! SEPARATELY (instinct issue 018's own law) before
//! `LAYA_WEIGHTS_VARIANT=q4` serves anything. What lands here is the
//! format seam — decoder ([`super::weights::widen_q4_0`]), converter
//! (this module), and the same bit-identity battery at Q4's OWN
//! fidelity (resident == host-widen, byte-for-byte; never vs F16).
//!
//! The container/proof/commit scaffolding is the shared
//! [`super::blocked_artifact`] machinery; this module owns the Q4_0
//! encode loop and the paths.

use std::path::{Path, PathBuf};

use super::super::{LayaError, Result};
use super::blocked_artifact::{
    ContainerEntry, commit_artifact, prove_readback, serialize_container,
};
use super::fake_quant::{BLOCK, q4_quant_of, q4_scale_bits, q4_scale_f32};
use super::weights::{self, f32_to_f16_bits};

/// The derived artifact's path under a checkpoint dir.
#[must_use]
pub fn q4_artifact_path(dir: &Path) -> PathBuf {
    dir.join("derived").join("model.q4.safetensors")
}

/// The sidecar's path (BLAKE3 hex, the converter writes, the loader
/// verifies).
#[must_use]
pub fn q4_sidecar_path(dir: &Path) -> PathBuf {
    dir.join("derived").join("model.q4.safetensors.blake3")
}

/// The converter (`laya-quant4 <checkpoint-dir>`): F16 map in, Q4_0
/// artifact + sidecar out. Round-trip verified before the file commits
/// (the shared machinery's atomic write + proof — a failed conversion
/// never leaves a half artifact behind).
pub fn convert_checkpoint(dir: &Path) -> Result<Q4ConvertReport> {
    let src = dir.join("model.safetensors");
    let f16_map = weights::load(&src, "q4-convert")?;

    // Encode: every >=2D tensor to Q4_0 blocks (GGML nibble order: the
    // even element in a byte's LOW nibble, the odd element in the HIGH
    // nibble; an odd tail's final half-byte is present, unused); 1D
    // tensors carried F16 (the house norm law, the q8 lane's own rule).
    let mut entries: Vec<ContainerEntry> = Vec::new();
    let mut names: Vec<&String> = f16_map.keys().collect();
    names.sort_unstable();
    let mut quantized_tensors = 0usize;
    let mut quantized_elements = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut f16_bytes = 0u64;
    let mut q4_bytes = 0u64;
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
            q4_bytes += data.len() as u64;
            entries.push(((*tensor_name).clone(), w.shape.clone(), "F16".into(), data));
            continue;
        }
        quantized_tensors += 1;
        quantized_elements += numel;
        let mut data = Vec::with_capacity(numel.div_ceil(BLOCK) * 18 + 18);
        for block in w.wide_f32().chunks(BLOCK) {
            let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let bits = q4_scale_bits(amax);
            data.extend_from_slice(&bits.to_le_bytes());
            let d = q4_scale_f32(bits);
            for pair in block.chunks(2) {
                let lo = q4_quant_of(pair[0], d) as u8;
                let hi = pair.get(1).map(|&v| q4_quant_of(v, d) as u8).unwrap_or(0);
                data.push((hi & 0x0F) << 4 | (lo & 0x0F));
            }
        }
        q4_bytes += data.len() as u64;
        entries.push(((*tensor_name).clone(), w.shape.clone(), "Q4_0".into(), data));
    }

    // Serialize → PROOF → commit (the shared machinery; the proof target
    // is the in-memory FAKE-QUANT Q4 grid — the artifact's decode values
    // must be the grid's values bit-for-bit).
    let out_bytes = serialize_container(&entries);
    let mut expect_map = f16_map;
    let _expect_rep =
        super::fake_quant::fake_quant_q4_map(&mut expect_map).map_err(LayaError::Runtime)?;
    prove_readback(&out_bytes, &expect_map, "model.q4.safetensors")?;
    let digest = commit_artifact(dir, "model.q4.safetensors", &out_bytes)?;

    Ok(Q4ConvertReport {
        tensors: entries.len(),
        quantized_tensors,
        quantized_elements,
        skipped_tensors: skipped,
        f16_bytes,
        q4_bytes,
        digest,
    })
}

/// What a conversion did (printed by the converter, kept with the
/// artifact's record).
#[derive(Debug, serde::Serialize)]
pub struct Q4ConvertReport {
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub quantized_elements: usize,
    pub skipped_tensors: Vec<String>,
    pub f16_bytes: u64,
    pub q4_bytes: u64,
    pub digest: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The converter's encode loop, on a small synthetic checkpoint:
    /// build the F16 container bytes, run `convert_checkpoint` against a
    /// temp dir, and require the emitted artifact's decode values to
    /// equal the in-memory fake-quant-Q4 grid bit-for-bit (the proof's
    /// own assertion), with the byte budget exactly
    /// `q4_blocked_len` per >=2D tensor + F16 for the 1D tensors.
    #[test]
    fn q4_converter_round_trips_through_the_proof() {
        use super::super::fake_quant::fake_quant_q4;
        use super::super::weights::{f32_to_f16_bits, q4_blocked_len};

        // A deterministic 2×32 matrix + one 1D norm, hand-serialized as
        // the F16 checkpoint (the container layout weights.rs parses).
        let mut s = 0x5EEDu32;
        let mut w = Vec::with_capacity(64);
        for _ in 0..64 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / 16_777_216.0;
            w.push((u - 0.5) * 0.5);
        }
        let mut expect = w.clone();
        let (max_err, _) = fake_quant_q4(&mut expect);
        assert!(max_err > 0.0, "the synthetic block must carry grid error");

        let mut data: Vec<u8> = Vec::new();
        for &v in &w {
            data.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        let mut norm_data = Vec::new();
        for &v in &[1.0f32, 0.5] {
            norm_data.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        let header = format!(
            concat!(
                r#"{{"w":{{"dtype":"F16","shape":[2,32],"data_offsets":[0,"#,
                "{}",
                r#"]}},"n":{{"dtype":"F16","shape":[2],"data_offsets":["#,
                "{}",
                ",",
                "{}",
                r#"]}}}}"#
            ),
            data.len(),
            data.len(),
            data.len() + norm_data.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&data);
        buf.extend_from_slice(&norm_data);

        let dir = std::env::temp_dir().join(format!("q4_convert_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("model.safetensors"), &buf).expect("write f16");

        let rep = convert_checkpoint(&dir).expect("the proof passes");
        assert_eq!(rep.quantized_tensors, 1);
        assert_eq!(rep.quantized_elements, 64);
        assert_eq!(rep.skipped_tensors, vec!["n".to_string()]);
        // 2×32 = one full block: 2 + 16 bytes.
        assert_eq!(rep.q4_bytes, q4_blocked_len(64) as u64 + 4);

        // The artifact on disk decodes (through the loader) to the grid's
        // exact values.
        let bytes = std::fs::read(q4_artifact_path(&dir)).expect("artifact");
        let map = weights::from_bytes(&bytes, "test").expect("parses");
        let got = map["w"].wide_f32();
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "element {i}: {g} vs {e}");
        }
        // Re-converting over an existing artifact is idempotent (same
        // bytes → same digest).
        let rep2 = convert_checkpoint(&dir).expect("re-convert");
        assert_eq!(rep.digest, rep2.digest);
        drop(map);
        std::fs::remove_dir_all(&dir).ok();
    }
}
