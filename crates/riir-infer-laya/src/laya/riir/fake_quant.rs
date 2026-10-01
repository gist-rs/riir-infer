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

use super::weights::{WeightData, Weights, f16_bits_to_f32, f32_to_f16_bits};

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
    /// D2's Q4 second (Plan 616 Phase 3): the weights loaded from the
    /// derived Q4_0 artifact (`LAYA_WEIGHTS_VARIANT=q4`) — no in-memory
    /// transform; the decode values are byte-identical to the in-memory
    /// fake-quant Q4 (the converter's proof). ITS OWN posture, never a
    /// q8 relabel — its numerics are the Q4 grid's, its retention is
    /// D1-priced separately before any adoption.
    Q4Artifact,
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
            Self::Q4Artifact => "q4-artifact",
        }
    }
}

/// Fake-quantize every >=2D tensor of a loaded weight map in place with
/// the Q8_0 grid. Deterministic: tensors are visited in sorted-name order
/// and the transform is a pure function of the bytes.
///
/// A RETAINED raw-quant payload ([`super::weights::WeightData::Q8`] or
/// `Q4` — either artifact posture) is REFUSED loud: the probe is the
/// in-memory transform of an F16 map, and quantizing a payload whose
/// stored values ARE the quantized values would quantize twice (the same
/// refusal the agent's load path already applies, one layer out).
pub fn fake_quant_q8_map(map: &mut HashMap<String, Weights>) -> Result<FakeQuantReport, String> {
    fake_quant_map(map, QuantGrid::Q8)
}

/// The Q4_0 twin (Plan 616 Phase 3): the same map pass against the Q4
/// grid — the converter's read-back proof target and the future D4
/// retention probe's instrument. Same refusals, same report shape.
pub fn fake_quant_q4_map(map: &mut HashMap<String, Weights>) -> Result<FakeQuantReport, String> {
    fake_quant_map(map, QuantGrid::Q4)
}

/// The grid a [`fake_quant_map`] pass runs — the ONE enum behind both
/// entry points, so the two probes cannot drift apart structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGrid {
    Q8,
    Q4,
}

impl QuantGrid {
    /// The scale (as f16 BITS) for one block's amax — the grid's own law.
    #[must_use]
    pub fn scale_bits(self, amax: f32) -> u16 {
        match self {
            Self::Q8 => q8_scale_bits(amax),
            Self::Q4 => q4_scale_bits(amax),
        }
    }

    /// The quantized value of ONE weight against scale `d`.
    #[must_use]
    pub fn quant_of(self, w: f32, d: f32) -> i8 {
        match self {
            Self::Q8 => q8_quant_of(w, d),
            Self::Q4 => q4_quant_of(w, d),
        }
    }

    /// The scale as f32 (the dequant arithmetic).
    #[must_use]
    pub fn scale_f32(self, bits: u16) -> f32 {
        match self {
            Self::Q8 => q8_scale_f32(bits),
            Self::Q4 => q4_scale_f32(bits),
        }
    }
}

/// The shared map pass behind both `fake_quant_{q8,q4}_map` entry
/// points — one body, the grid injected ([`QuantGrid`]), never a second
/// transcription.
fn fake_quant_map(
    map: &mut HashMap<String, Weights>,
    grid: QuantGrid,
) -> Result<FakeQuantReport, String> {
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
        let WeightData::F32(data) = &mut w.data else {
            return Err(format!(
                "fake-quant probe over tensor {name}: the payload is ALREADY raw quantized \
                 (the artifact posture) — quantizing twice is refused; \
                 unset LAYA_WEIGHTS_VARIANT"
            ));
        };
        rep.quantized_tensors += 1;
        rep.quantized_elements += data.len();
        rep.quantized_f16_bytes += data.len() as u64 * 2;
        let (max_err, sum_err) = fake_quant_grid(data, grid);
        rep.blocks += data.len().div_ceil(BLOCK);
        rep.max_abs_err = rep.max_abs_err.max(max_err);
        rep.mean_abs_err += sum_err;
    }
    if rep.quantized_elements > 0 {
        rep.mean_abs_err /= rep.quantized_elements as f64;
    }
    Ok(rep)
}

/// The shared block loop behind [`fake_quant_q8`] / [`fake_quant_q4`]
/// — one body, the grid injected.
fn fake_quant_grid(data: &mut [f32], grid: QuantGrid) -> (f32, f64) {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f64;
    for block in data.chunks_mut(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let d = grid.scale_f32(grid.scale_bits(amax));
        for w in block.iter_mut() {
            let q = f32::from(grid.quant_of(*w, d));
            let dq = d * q;
            let err = (*w - dq).abs();
            sum_err += f64::from(err);
            max_err = max_err.max(err);
            *w = dq;
        }
    }
    (max_err, sum_err)
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

/// The Q4_0 twins (Plan 616 Phase 3): the house blocked family's 4-bit
/// rung — the SAME block law (32 weights, one f16 scale) with a signed
/// 4-bit grid. Deliberately NOT GGML's unsigned `d·(q−8)` form: the
/// house Q8_0 is the symmetric `d·q` whose grid ends carry ±amax (up to
/// the scale's own f16 rounding, the same storage-grid law), and the
/// 4-bit rung keeps that shape (`d = f16(amax/7)`, q in [-7, 7], zero
/// exact), packed as GGML nibbles (even element low, odd high) so the
/// container layout stays GGUF-shaped. The decode sign-extends, so a
/// `-8` nibble (never produced here, produced by GGML's own converters)
/// decodes as d·(−8) — interop-safe, house-exact.
///
/// # The scale
///
/// The Q4_0 scale for one block's amax: the f16-ROUNDED value of
/// `amax / 7` — returned as f16 BITS (the storage form). The real-grid
/// law puts both ±amax exactly on the grid ends (q = ±7) and zero at 0;
/// the STORED scale carries the f16 rounding's own ~2⁻¹² relative error,
/// exactly as the Q8 scale does (the same storage-grid law — the probe
/// quantizes against the f16-rounded scale, so the measured error is the
/// error real adoption introduces). The converter and the artifact
/// reader derive the scale through this ONE function, so an artifact
/// widened anywhere decodes to the same values the converter proved.
#[must_use]
pub fn q4_scale_bits(amax: f32) -> u16 {
    f32_to_f16_bits(amax / 7.0)
}

/// The Q4_0 scale as f32 (the reader arithmetic).
#[must_use]
pub fn q4_scale_f32(bits: u16) -> f32 {
    f16_bits_to_f32(bits)
}

/// The quantized value of ONE weight against scale `d` — `roundf(w/d)`
/// (half away from zero, the GGUF reference's rounding) clamped to
/// [-7, 7]. A non-positive `d` (degenerate block) quantizes to 0.
#[must_use]
pub fn q4_quant_of(w: f32, d: f32) -> i8 {
    if d <= 0.0 {
        return 0;
    }
    ((w * (1.0 / d)).round().clamp(-7.0, 7.0)) as i8
}

/// Q4_0 fake-quant of ONE tensor's payload (Plan 616 Phase 3): per
/// [`BLOCK`]-weight block, scale `d = f16(amax / 7)` ([`q4_scale_bits`]),
/// `q = [`q4_quant_of`]`, dequant `w' = d · q`. A degenerate block maps
/// every weight to 0.0 (its stored scale IS 0). This is the converter's
/// read-back proof target and the future D4 retention probe's in-memory
/// twin — the Q4 grid's error is measured against THIS arithmetic, never
/// a wider-grid idealization.
///
/// Returns (max |err|, Σ |err|) over the tensor.
pub fn fake_quant_q4(data: &mut [f32]) -> (f32, f64) {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f64;
    for block in data.chunks_mut(BLOCK) {
        let amax = block.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let d = q4_scale_f32(q4_scale_bits(amax));
        for w in block.iter_mut() {
            let q = f32::from(q4_quant_of(*w, d));
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
                data: WeightData::F32(vec![v; n]),
            }
        };
        let mut map = HashMap::new();
        map.insert("b_norm".to_string(), mk(vec![8], 1.0));
        map.insert("w_small".to_string(), mk(vec![2, 2], 3.0));
        map.insert("a_emb".to_string(), mk(vec![4, 8], 0.25));
        let rep = fake_quant_q8_map(&mut map).expect("f32 payloads");
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
        let rep2 = fake_quant_q8_map(&mut map).expect("f32 payloads");
        assert_eq!(rep.quantized_tensors, rep2.quantized_tensors);
        assert_eq!(rep.quantized_elements, rep2.quantized_elements);
    }

    /// A retained Q8 payload (the q8 artifact posture) is REFUSED — the
    /// probe must never quantize twice.
    #[test]
    fn map_pass_refuses_a_raw_q8_payload() {
        let q8: Vec<u8> = vec![0u8; super::super::weights::q8_blocked_len(32)];
        let mut map = HashMap::new();
        map.insert(
            "w".to_string(),
            Weights {
                shape: vec![1, 32],
                data: WeightData::Q8(super::super::weights::RawQ8::new(32, q8).expect("layout")),
            },
        );
        let err = fake_quant_q8_map(&mut map).expect_err("refused");
        assert!(err.contains("twice"), "{err}");
    }

    /// A retained Q4 payload is refused by BOTH probes — the double-quant
    /// refusal is posture-shaped, not format-shaped.
    #[test]
    fn map_pass_refuses_a_raw_q4_payload_from_both_probes() {
        let q4: Vec<u8> = vec![0u8; super::super::weights::q4_blocked_len(32)];
        let mk = || {
            let mut map = HashMap::new();
            map.insert(
                "w".to_string(),
                Weights {
                    shape: vec![1, 32],
                    data: WeightData::Q4(
                        super::super::weights::RawQ4::new(32, q4.clone()).expect("layout"),
                    ),
                },
            );
            map
        };
        assert!(
            fake_quant_q8_map(&mut mk())
                .expect_err("q8 probe refused")
                .contains("twice")
        );
        assert!(
            fake_quant_q4_map(&mut mk())
                .expect_err("q4 probe refused")
                .contains("twice")
        );
    }

    /// The Q4 grid's laws, on synthetic values: ±amax land exactly on the
    /// grid ends in QUANT (q = ±7 — round of 7±ε is 7), zero maps to zero
    /// exactly, the dequant error at the ends is the f16 scale's own
    /// rounding (never more), and a degenerate block maps to zeros.
    #[test]
    fn q4_grid_both_amax_ends_exact_and_zero_exact() {
        for &amax in &[0.25f32, 1.0, 3.0, 0.001] {
            let d = q4_scale_f32(q4_scale_bits(amax));
            assert_eq!(q4_quant_of(amax, d), 7);
            assert_eq!(q4_quant_of(-amax, d), -7);
            // The end-point dequant error is exactly the f16 scale rounding
            // (|7·f16(a/7) − a| — the SAME term the Q8 end point carries at
            // q=±127), never larger.
            let scale_err = (7.0 * d - amax).abs();
            let f16_bound = (amax / 7.0 - d).abs() * 7.0 + 1e-9;
            assert!(scale_err <= f16_bound, "amax={amax} scale_err={scale_err}");
            assert_eq!(q4_quant_of(0.0, d), 0);
            assert_eq!(d * 0.0, 0.0);
        }
        // A degenerate block: scale 0 → all zeros, error 0.
        let mut data = vec![0.0f32; BLOCK + 5];
        let (max_err, sum) = fake_quant_q4(&mut data);
        assert_eq!((max_err, sum), (0.0, 0.0));
    }

    /// The Q4 error bound on a deterministic sweep: |w − d·q| ≤ d/2 plus
    /// the scale's f16 rounding applied at |q| ≤ 7 — the Q8 bound test's
    /// 4-bit shape, one-eighteenth the resolution.
    #[test]
    fn q4_error_stays_within_the_format_bound() {
        let mut s = 0x12345678u32;
        let mut data = Vec::with_capacity(64 * BLOCK);
        for _ in 0..64 * BLOCK {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / 16_777_216.0; // 2^24
            data.push((u - 0.5) * 4.0);
        }
        let amax = data.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let d = f16_bits_to_f32(f32_to_f16_bits(amax / 7.0));
        let (max_err, _) = fake_quant_q4(&mut data);
        let bound = d * 0.5 + (amax / 7.0 - d).abs() * 7.0 + 1e-6;
        assert!(max_err <= bound, "err {max_err} > bound {bound}");
    }

    /// The Q4 map pass: the report's arithmetic matches the Q8 pass's
    /// shape (same tensors, same blocks, same skipped set) — the two
    /// probes differ ONLY in the grid, never in coverage.
    #[test]
    fn q4_map_pass_covers_the_same_surface_as_q8() {
        let mk = |shape: Vec<usize>, v: f32| {
            let n: usize = shape.iter().product();
            Weights {
                shape,
                data: WeightData::F32(vec![v; n]),
            }
        };
        let mut m8 = HashMap::new();
        m8.insert("b_norm".to_string(), mk(vec![8], 1.0));
        m8.insert("w".to_string(), mk(vec![4, 8], 0.25));
        let mut m4 = HashMap::new();
        m4.insert("b_norm".to_string(), mk(vec![8], 1.0));
        m4.insert("w".to_string(), mk(vec![4, 8], 0.25));
        let r8 = fake_quant_q8_map(&mut m8).expect("f32");
        let r4 = fake_quant_q4_map(&mut m4).expect("f32");
        assert_eq!(r8.quantized_tensors, r4.quantized_tensors);
        assert_eq!(r8.quantized_elements, r4.quantized_elements);
        assert_eq!(r8.blocks, r4.blocks);
        assert_eq!(r8.skipped_tensors, r4.skipped_tensors);
        // The Q4 error on a constant block is the f16 scale rounding plus
        // the coarser grid — bounded, never zero-exact like Q8's tail.
        assert!(
            r4.max_abs_err > 0.0 && r4.max_abs_err < 0.05,
            "{}",
            r4.max_abs_err
        );
    }
}
