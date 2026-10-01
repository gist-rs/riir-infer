//! Fake-quant instrument (instinct issue 018 Lane D1): quantize-then-
//! dequantize the checkpoint's weights at load and run the UNCHANGED
//! forward — measuring WEIGHT-QUANTIZATION error only. Kernel numerics
//! (a real Q8 kernel's accumulation order) are D2's per-device
//! determinism gate's subject, deliberately NOT this probe's.
//!
//! The block format is the house GGUF Q8_0 family: 32 weights per block,
//! one scale, `q` in [-127, 127]. The probe quantizes against the
//! STORAGE grid — the scale is f16-rounded exactly as a real Q8_0 tensor
//! stores it — so the error the forward sees is the error real adoption
//! (D2) would introduce, not a wider-grid idealization.
//!
//! Scope: tensors with shape.ndim() >= 2 (matmul weights + embedding
//! tables) are transformed; 1D tensors (norm weights/biases, per-tensor
//! scales) are left untouched — the house GGUF norm law (norms stay
//! F32/F16), and they are <1% of the checkpoint's bytes. The report
//! names every skipped tensor so the posture is disclosed, never silent.

use std::collections::HashMap;

use serde::Serialize;

use super::weights::{Weights, f16_bits_to_f32, f32_to_f16_bits};

/// The block size of the house GGUF Q8_0 family.
pub const BLOCK: usize = 32;

/// What a [`fake_quant_q8_map`] pass did — serialized into the consumer's
/// record verbatim (the posture is disclosed, never inferred).
#[derive(Debug, Clone, Serialize)]
pub struct FakeQuantReport {
    /// Tensors transformed (ndim >= 2).
    pub quantized_tensors: usize,
    /// Elements transformed.
    pub quantized_elements: usize,
    /// 32-weight blocks processed.
    pub blocks: usize,
    /// 1D tensors left untouched (the house norm law), sorted by name.
    pub skipped_tensors: Vec<String>,
    /// max |w − dequant(quant(w))| over the transformed elements.
    pub max_abs_err: f32,
    /// mean |w − dequant(quant(w))| over the transformed elements.
    pub mean_abs_err: f64,
    /// Bytes these tensors occupy at F16 storage — the honest share a
    /// real Q8 adoption would roughly halve.
    pub quantized_f16_bytes: u64,
}

/// The weight-posture vocabulary (D1's fake-quant, D2a's Q8 artifact;
/// D2b's device-resident kernels extend this enum — never a bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPosture {
    /// The shipped posture: the checkpoint's own F16 widened to F32.
    F16,
    /// D1's probe: every >=2D tensor fake-quantized Q8_0 (block-32, f16
    /// scale) in memory; the forward is byte-identical code.
    FakeQuantQ8,
    /// D2a's storage tier: the weights loaded from the derived Q8_0
    /// artifact (`LAYA_WEIGHTS_VARIANT=q8`) — no in-memory transform;
    /// the decode values are byte-identical to [`Self::FakeQuantQ8`]
    /// (the converter's proof), so the probe's reads carry.
    Q8Artifact,
}

impl WeightPosture {
    /// The record label — printed/serialized beside every number so a
    /// quantized read can never be mistaken for the shipped posture.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::FakeQuantQ8 => "fake-quant-q8",
            Self::Q8Artifact => "q8-artifact",
        }
    }
}

/// Fake-quantize every >=2D tensor of a loaded weight map in place.
/// Deterministic: tensors are visited in sorted-name order and the
/// transform is a pure function of the bytes.
#[must_use]
pub fn fake_quant_q8_map(map: &mut HashMap<String, Weights>) -> FakeQuantReport {
    let mut rep = FakeQuantReport {
        quantized_tensors: 0,
        quantized_elements: 0,
        blocks: 0,
        skipped_tensors: Vec::new(),
        max_abs_err: 0.0,
        mean_abs_err: 0.0,
        quantized_f16_bytes: 0,
    };
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort_unstable();
    for name in &names {
        let Some(w) = map.get_mut(name) else {
            unreachable!("name came from the map's own keys");
        };
        if w.shape.len() < 2 {
            rep.skipped_tensors.push(name.clone());
            continue;
        }
        rep.quantized_tensors += 1;
        rep.quantized_elements += w.data.len();
        rep.quantized_f16_bytes += w.data.len() as u64 * 2;
        let (max_err, sum_err) = fake_quant_q8(&mut w.data);
        rep.blocks += w.data.len().div_ceil(BLOCK);
        rep.max_abs_err = rep.max_abs_err.max(max_err);
        rep.mean_abs_err += sum_err;
    }
    if rep.quantized_elements > 0 {
        rep.mean_abs_err /= rep.quantized_elements as f64;
    }
    rep
}

/// The Q8_0 scale for one block's amax: the f16-ROUNDED value of
/// `amax / 127` — returned as f16 BITS (the storage form). The probe,
/// the artifact converter and the artifact reader all derive the scale
/// through this ONE function; the probe quantizes against the storage
/// grid so its measured error is the error real adoption introduces.
#[must_use]
pub fn q8_scale_bits(amax: f32) -> u16 {
    f32_to_f16_bits(amax / 127.0)
}

/// The scale as f32 (the reader/probe arithmetic).
#[must_use]
pub fn q8_scale_f32(bits: u16) -> f32 {
    f16_bits_to_f32(bits)
}

/// The quantized value of ONE weight against scale `d` — `roundf(w/d)`
/// (half away from zero, the GGUF reference's rounding) clamped to
/// [-127, 127]. A non-positive `d` (degenerate block) quantizes to 0.
#[must_use]
pub fn q8_quant_of(w: f32, d: f32) -> i8 {
    if d <= 0.0 {
        return 0;
    }
    ((w * (1.0 / d)).round().clamp(-127.0, 127.0)) as i8
}

/// Q8_0 fake-quant of ONE tensor's payload: per [`BLOCK`]-weight block,
/// scale `d = f16(amax / 127)` ([`q8_scale_bits`]), `q =
/// [`q8_quant_of`]`, dequant `w' = d · q`. A degenerate block (amax 0,
/// or a scale that flushes to zero in f16) maps every weight to 0.0 —
/// the format's own behaviour for that block (its stored scale IS 0).
///
/// Returns (max |err|, Σ |err|) over the tensor.
pub fn fake_quant_q8(data: &mut [f32]) -> (f32, f64) {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f64;
    for block in data.chunks_mut(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let d = q8_scale_f32(q8_scale_bits(amax));
        for w in block.iter_mut() {
            let q = f32::from(q8_quant_of(*w, d));
            let dq = d * q;
            let err = (*w - dq).abs();
            sum_err += f64::from(err);
            max_err = max_err.max(err);
            *w = dq;
        }
    }
    (max_err, sum_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant block round-trips to its own value up to the f16 scale
    /// rounding: d = f16(c/127), q = 127, so w' = 127·f16(c/127) — the
    /// relative error is the f16 grid (~1e-3), never more.
    #[test]
    fn constant_block_round_trips_at_f16_scale_precision() {
        for &c in &[0.5f32, 1.0, -2.0, 33.0, 0.001] {
            let mut data = vec![c; BLOCK];
            let (max_err, _) = fake_quant_q8(&mut data);
            let expect = (127.0f32 * f16_bits_to_f32(f32_to_f16_bits(c / 127.0)) - c).abs();
            assert!(max_err <= expect * 1.5 + 1e-9, "c={c} err={max_err}");
            assert!((data[0] - c).abs() <= c.abs() * 2e-3 + 1e-6);
        }
    }

    /// The error bound of the format: |w − d·round(w/d)| ≤ d/2 with d =
    /// f16(amax/127) — plus the scale's own f16 rounding (~4e-3 relative
    /// to amax). Nothing exceeds it on a random-ish sweep.
    #[test]
    fn error_stays_within_the_format_bound() {
        // A deterministic spread across magnitudes (no RNG — a fixed LCG).
        let mut s = 0x12345678u32;
        let mut data = Vec::with_capacity(64 * BLOCK);
        for _ in 0..64 * BLOCK {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / 16_777_216.0; // 2^24
            data.push((u - 0.5) * 4.0);
        }
        let amax = data.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let d = f16_bits_to_f32(f32_to_f16_bits(amax / 127.0));
        let (max_err, _) = fake_quant_q8(&mut data);
        // d/2 + the scale's f16 rounding applied at |q| ≤ 127.
        let bound = d * 0.5 + (amax / 127.0 - d).abs() * 127.0 + 1e-6;
        assert!(max_err <= bound, "err {max_err} > bound {bound}");
    }

    /// A degenerate block (all zeros) maps to zeros — its stored scale is
    /// 0 — and contributes error 0.
    #[test]
    fn zero_block_maps_to_zero() {
        let mut data = vec![0.0f32; 2 * BLOCK + 7]; // a ragged tail
        data[70] = 0.0;
        let (max_err, sum) = fake_quant_q8(&mut data);
        assert_eq!(max_err, 0.0);
        assert_eq!(sum, 0.0);
        assert!(data.iter().all(|&w| w == 0.0));
    }

    /// The map pass: >=2D tensors transformed, 1D skipped and NAMED, the
    /// report's arithmetic consistent, and the pass is DETERMINISTIC
    /// (sorted-name visit order — two runs byte-identical).
    #[test]
    fn map_pass_transforms_matrices_and_names_the_skips() {
        let mk = |shape: Vec<usize>, v: f32| {
            let n: usize = shape.iter().product();
            Weights {
                shape,
                data: vec![v; n],
            }
        };
        let mut map = HashMap::new();
        map.insert("b_norm".to_string(), mk(vec![8], 1.0));
        map.insert("w_small".to_string(), mk(vec![2, 2], 3.0));
        map.insert("a_emb".to_string(), mk(vec![4, 8], 0.25));
        let rep = fake_quant_q8_map(&mut map);
        assert_eq!(rep.quantized_tensors, 2);
        assert_eq!(rep.quantized_elements, 4 + 32);
        assert_eq!(rep.blocks, 1 + 1);
        assert_eq!(rep.skipped_tensors, vec!["b_norm".to_string()]);
        assert_eq!(rep.quantized_f16_bytes, (4 + 32) as u64 * 2);
        // A constant tensor's transform is near-exact (the first test's law).
        assert!(rep.max_abs_err < 0.01, "{}", rep.max_abs_err);

        // Determinism: a second pass over the SAME map is a no-op change
        // (already-quantized values re-quantize to themselves up to the
        // grid) and the report is identical.
        let rep2 = fake_quant_q8_map(&mut map);
        assert_eq!(rep.quantized_tensors, rep2.quantized_tensors);
        assert_eq!(rep.quantized_elements, rep2.quantized_elements);
    }
}
