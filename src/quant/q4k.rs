//! `Q4_K` (k-quant 4-bit medium) quantization format.
//!
//! Ported from llama.cpp's `ggml-quants.c` and `ggml-common.h`.
//! Reference: <https://github.com/ggml-org/llama.cpp>
//!
//! # Block Layout (144 bytes for 256 elements = 4.5 bpw)
//!
//! ```text
//! Offset  Size   Field
//! 0       2      d      — f16 super-block scale for quantized sub-scales
//! 2       2      dmin   — f16 super-block scale for quantized sub-mins
//! 4       12     scales — 8 × (6-bit scale, 6-bit min) packed into 12 bytes
//! 16      128    qs     — 4-bit packed quantized values (2 per byte)
//! ```
//!
//! # Sub-block Layout
//!
//! Each super-block contains 8 sub-blocks of 32 elements.
//! Sub-blocks are paired into the `qs` array:
//! - Pair k: sub-blocks (2k, 2k+1) share `qs[k*32..k*32+32]`
//! - Sub-block 2k → low nibble, sub-block 2k+1 → high nibble
//!
//! Dequant formula: `value = d * sc * nibble - dmin * m`

use half::f16;

/// Super-block size: 256 elements per block.
pub const QK_K: usize = 256;

/// Number of sub-blocks per super-block.
const N_SUB_BLOCKS: usize = 8;

/// Elements per sub-block (256 / 8 = 32).
const SUB_BLOCK_SIZE: usize = QK_K / N_SUB_BLOCKS;

/// Size of the packed scales array (bytes).
pub(crate) const K_SCALE_SIZE: usize = 12;

/// A `Q4_K` quantization block.
///
/// Represents 256 weight values in 144 bytes (4.5 bits per weight).
/// Layout matches llama.cpp's `block_q4_K` for GPU upload compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4K {
    /// Super-block scale for quantized sub-scales (f16 bits).
    pub d: u16,
    /// Super-block scale for quantized sub-mins (f16 bits).
    pub dmin: u16,
    /// Packed per-sub-block scales and mins (6-bit each, 8 pairs in 12 bytes).
    pub scales: [u8; K_SCALE_SIZE],
    /// 4-bit packed quantized values (2 values per byte, 128 bytes for 256 values).
    pub qs: [u8; QK_K / 2],
}

// Static assert: BlockQ4K must be exactly 144 bytes.
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);

// ── Scale encoding/decoding ─────────────────────────────────────

/// Decode (scale, min) pair from packed scales array for sub-block `j` (0..8).
///
/// Groups 0-3: `scales[j] & 63` = scale, `scales[j+4] & 63` = min.
/// Groups 4-7: packed into remaining nibbles with 6-bit fields.
#[inline]
pub fn get_scale_min_k4(j: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    debug_assert!(j < N_SUB_BLOCKS);
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Encode (scale, min) pair into packed scales array for sub-block `j` (0..8).
///
/// Both values must be in [0, 63] (6-bit range).
fn set_scale_min_k4(j: usize, ls: u8, lm: u8, scales: &mut [u8; K_SCALE_SIZE]) {
    debug_assert!(j < N_SUB_BLOCKS);
    debug_assert!(ls <= 63, "scale {ls} exceeds 6-bit range");
    debug_assert!(lm <= 63, "min {lm} exceeds 6-bit range");
    if j < 4 {
        scales[j] = ls;
        scales[j + 4] = lm;
    } else {
        // Low 4 bits of scale and min go into byte j+4
        scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
        // High 2 bits of scale go into top of byte j-4
        scales[j - 4] |= (ls >> 4) << 6;
        // High 2 bits of min go into top of byte j
        scales[j] |= (lm >> 4) << 6;
    }
}

// ── Quantize ────────────────────────────────────────────────────

/// Quantize a row of f32 values to `Q4_K` blocks.
///
/// `src` length must be a multiple of `QK_K` (256).
/// `dst` length must be `src.len() / QK_K`.
pub fn quantize_row_q4_k(src: &[f32], dst: &mut [BlockQ4K]) {
    assert!(
        src.len().is_multiple_of(QK_K),
        "src length must be multiple of {QK_K}, got {}",
        src.len()
    );
    let nb = src.len() / QK_K;
    assert!(
        dst.len() >= nb,
        "dst too short: need {nb} blocks, got {}",
        dst.len()
    );

    for i in 0..nb {
        quantize_block(&src[i * QK_K..(i + 1) * QK_K], &mut dst[i]);
    }
}

/// Quantize one super-block of 256 f32 values.
fn quantize_block(src: &[f32], block: &mut BlockQ4K) {
    debug_assert_eq!(src.len(), QK_K);

    // Zero out the block
    *block = BlockQ4K {
        d: 0,
        dmin: 0,
        scales: [0u8; K_SCALE_SIZE],
        qs: [0u8; QK_K / 2],
    };

    // Phase 1: Find per-sub-block scale and min offset.
    let mut sub_scales = [0.0f32; N_SUB_BLOCKS];
    let mut sub_mins = [0.0f32; N_SUB_BLOCKS]; // always >= 0 (offset added before quant)
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;

    for j in 0..N_SUB_BLOCKS {
        let base = j * SUB_BLOCK_SIZE;

        // 4-wide min/max scan: independent accumulators break the loop-carried
        // f32::min/f32::max dependency (non-associative → not auto-vectorizable).
        let mut min0 = f32::INFINITY;
        let mut min1 = f32::INFINITY;
        let mut min2 = f32::INFINITY;
        let mut min3 = f32::INFINITY;
        let mut max0 = f32::NEG_INFINITY;
        let mut max1 = f32::NEG_INFINITY;
        let mut max2 = f32::NEG_INFINITY;
        let mut max3 = f32::NEG_INFINITY;
        let chunks = SUB_BLOCK_SIZE / 4;
        for c in 0..chunks {
            let k = c * 4;
            let v0 = src[base + k];
            let v1 = src[base + k + 1];
            let v2 = src[base + k + 2];
            let v3 = src[base + k + 3];
            min0 = min0.min(v0);
            max0 = max0.max(v0);
            min1 = min1.min(v1);
            max1 = max1.max(v1);
            min2 = min2.min(v2);
            max2 = max2.max(v2);
            min3 = min3.min(v3);
            max3 = max3.max(v3);
        }
        let min_val = min0.min(min1).min(min2).min(min3);
        let max_val = max0.max(max1).max(max2).max(max3);

        // Min offset shifts values so minimum maps to 0.
        // dequant: x̂ = d*sc*q - dmin*m, so q = round((x + dmin*m) / (d*sc))
        let min_offset = (-min_val).max(0.0);
        let range = max_val + min_offset; // total range after offset

        if range > 0.0 {
            sub_scales[j] = range / 15.0;
            sub_mins[j] = min_offset;
        }

        max_scale = max_scale.max(sub_scales[j]);
        max_min = max_min.max(sub_mins[j]);
    }

    // Phase 2: Quantize sub-block scales/mins to 6-bit.
    let d_super = if max_scale > 0.0 {
        max_scale / 63.0
    } else {
        0.0
    };
    let dmin_super = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    for j in 0..N_SUB_BLOCKS {
        let ls = ((inv_scale * sub_scales[j]).round() as u8).min(63);
        let lm = ((inv_min * sub_mins[j]).round() as u8).min(63);
        set_scale_min_k4(j, ls, lm, &mut block.scales);
    }

    block.d = f16::from_f32(d_super).to_bits();
    block.dmin = f16::from_f32(dmin_super).to_bits();

    // Phase 3: Quantize values to 4-bit.
    // Sub-blocks paired: (2k, 2k+1) share qs[k*32..k*32+32]
    // Sub-block 2k → low nibble, sub-block 2k+1 → high nibble
    // Use pre-computed d/dmin instead of re-decoding from f16
    let d = d_super;
    let dmin = dmin_super;

    for pair in 0..4 {
        let qs_base = pair * 32;

        for sub_in_pair in 0..2 {
            let sub_idx = pair * 2 + sub_in_pair;
            let (sc, m) = get_scale_min_k4(sub_idx, &block.scales);
            let d_sub = d * sc as f32;
            let m_sub = dmin * m as f32;

            let val_base = sub_idx * SUB_BLOCK_SIZE;

            if d_sub == 0.0 {
                continue; // qs already zeroed
            }

            // Hoist loop invariants: shift and mask depend only on sub_in_pair.
            let shift = sub_in_pair * 4;
            let mask = !(0x0Fu8 << shift);
            // Fused multiply-add: (src[k] + m_sub) * inv_d_sub == src[k] * inv_d_sub + bias
            // where bias = m_sub * inv_d_sub. mul_add emits a single FMA instruction.
            let inv_d_sub = 1.0 / d_sub;
            let bias = m_sub * inv_d_sub;
            for k in 0..SUB_BLOCK_SIZE {
                let v = src[val_base + k].mul_add(inv_d_sub, bias).round() as i32;
                let q = v.clamp(0, 15) as u8;

                block.qs[qs_base + k] = (block.qs[qs_base + k] & mask) | (q << shift);
            }
        }
    }
}

// ── Dequantize ──────────────────────────────────────────────────

/// Dequantize `Q4_K` blocks to f32 values.
///
/// `src` contains the quantized blocks, `dst` must have length `n`
/// where `n` is a multiple of `QK_K` and `src.len() >= n / QK_K`.
///
/// Ported from llama.cpp's `dequantize_row_q4_K`.
pub fn dequantize_row_q4_k(src: &[BlockQ4K], dst: &mut [f32]) {
    let n = dst.len();
    assert!(
        n.is_multiple_of(QK_K),
        "dst length must be multiple of {QK_K}"
    );
    let nb = n / QK_K;
    assert!(
        src.len() >= nb,
        "src too short: need {nb} blocks, got {}",
        src.len()
    );

    for (i, block) in src.iter().take(nb).enumerate() {
        let d = f32::from(f16::from_bits(block.d));
        let dmin = f32::from(f16::from_bits(block.dmin));
        let qs = &block.qs;
        let scales = &block.scales;

        // 4 pairs of sub-blocks (64 values per iteration)
        // Direct index computation: pair p covers sub-blocks [2p, 2p+1]
        // and qs bytes [32p .. 32p+32]
        for pair in 0..4 {
            let (sc0, m0) = get_scale_min_k4(pair * 2, scales);
            let d_sc0 = d * sc0 as f32;
            let m0_val = dmin * m0 as f32;

            let (sc1, m1) = get_scale_min_k4(pair * 2 + 1, scales);
            let d_sc1 = d * sc1 as f32;
            let m1_val = dmin * m1 as f32;

            let qs_base = pair * 32;
            let out_base = i * QK_K + pair * 64;

            // Low nibbles → sub-block 2p (4 elements at a time for auto-vectorization)
            let chunks = 32 / 4;
            for c in 0..chunks {
                let l = c * 4;
                dst[out_base + l] = d_sc0 * (qs[qs_base + l] & 0x0F) as f32 - m0_val;
                dst[out_base + l + 1] = d_sc0 * (qs[qs_base + l + 1] & 0x0F) as f32 - m0_val;
                dst[out_base + l + 2] = d_sc0 * (qs[qs_base + l + 2] & 0x0F) as f32 - m0_val;
                dst[out_base + l + 3] = d_sc0 * (qs[qs_base + l + 3] & 0x0F) as f32 - m0_val;
            }
            // High nibbles → sub-block 2p+1
            for c in 0..chunks {
                let l = c * 4;
                dst[out_base + 32 + l] = d_sc1 * (qs[qs_base + l] >> 4) as f32 - m1_val;
                dst[out_base + 32 + l + 1] = d_sc1 * (qs[qs_base + l + 1] >> 4) as f32 - m1_val;
                dst[out_base + 32 + l + 2] = d_sc1 * (qs[qs_base + l + 2] >> 4) as f32 - m1_val;
                dst[out_base + 32 + l + 3] = d_sc1 * (qs[qs_base + l + 3] >> 4) as f32 - m1_val;
            }
        }
    }
}

// ── Fused DeQuant + GEMV (Plan 486) ────────────────────────────
//
// CPU-side fused dequant+dot for Q4_K blocks. When the `simd_lut_q4k` feature
// is enabled, the inner loop dispatches to `katgpt_core::simd_lut_dequant::
// dequant_dot_via_lut` (Plan 431 Phase 3, 4.58× win on the raw primitive).
// When disabled, falls back to fused arithmetic (compute dequant value +
// immediate mul-add, no scratch buffer — the fairest non-LUT baseline).
//
// The standalone `dequantize_row_q4_k` above is intentionally NOT touched —
// Plan 431 Phase 4 proved plain LUT dequant is 0.286× on NEON (no gather).
// Only the fused dequant+dot slot uses the LUT path here.

/// Fused dot product of one row of `Q4_K` blocks with an f32 vector.
///
/// Computes `sum(dequant(blocks) .* x)` without materializing the dequantized
/// row to a buffer. `blocks` covers `blocks.len() * QK_K` consecutive elements;
/// `x` must be at least that long.
///
/// When `simd_lut_q4k` is enabled, the inner sub-block dot uses the katgpt-core
/// LUT-fused kernel (`dequant_dot_via_lut`). Otherwise it uses fused arithmetic
/// (same per-element formula as `dequantize_row_q4_k`, but accumulated directly
/// into the result with no scratch allocation).
#[inline]
pub fn gemv_q4_k_row(blocks: &[BlockQ4K], x: &[f32]) -> f32 {
    #[cfg(feature = "simd_lut_q4k")]
    {
        gemv_q4_k_row_lut(blocks, x)
    }
    #[cfg(not(feature = "simd_lut_q4k"))]
    {
        gemv_q4_k_row_arithmetic(blocks, x)
    }
}

/// Fused arithmetic reference — the G1 correctness oracle and G2 baseline.
///
/// Computes each dequant value via the same `d*sc*nibble - dmin*m` formula as
/// `dequantize_row_q4_k`, then immediately multiply-adds into the accumulator.
/// No scratch buffer allocation (unlike the old `cpu_q4k_gemv` test helper which
/// allocated a full-row buffer). This is the fairest non-LUT comparator: it
/// isolates the LUT benefit from the fusion benefit.
#[inline]
pub fn gemv_q4_k_row_arithmetic(blocks: &[BlockQ4K], x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (b, block) in blocks.iter().enumerate() {
        let d = f32::from(f16::from_bits(block.d));
        let dmin = f32::from(f16::from_bits(block.dmin));
        let qs = &block.qs;
        let scales = &block.scales;
        for pair in 0..4 {
            let (sc0, m0) = get_scale_min_k4(pair * 2, scales);
            let d_sc0 = d * sc0 as f32;
            let m0_val = dmin * m0 as f32;
            let (sc1, m1) = get_scale_min_k4(pair * 2 + 1, scales);
            let d_sc1 = d * sc1 as f32;
            let m1_val = dmin * m1 as f32;
            let qs_base = pair * 32;
            let x_base = b * QK_K + pair * 64;
            for l in 0..32 {
                let lo = d_sc0 * (qs[qs_base + l] & 0x0F) as f32 - m0_val;
                acc += lo * x[x_base + l];
                let hi = d_sc1 * (qs[qs_base + l] >> 4) as f32 - m1_val;
                acc += hi * x[x_base + 32 + l];
            }
        }
    }
    acc
}

/// Fused LUT path — uses `dequant_dot_via_lut` for the inner sub-block dot.
///
/// Gated on `simd_lut_q4k`. Builds a `UInt4Lut` per sub-block pair (8 pairs per
/// block = 16 LUT builds per 256-element block), then calls the katgpt-core
/// fused kernel which dequants in registers and FMAs into the accumulator.
///
/// # Zero-scale guard
///
/// When `d_sc == 0.0` (sub-block scale is zero), every code maps to the
/// constant `-m_val`. The LUT form `(i - zero) * scale` cannot represent this
/// (`zero = m_val / 0 = inf`). We handle it as a direct sum fallback. This
/// branch is rarely taken for real model weights (sub-scales are typically
/// non-zero) but is needed for correctness on degenerate inputs.
#[cfg(feature = "simd_lut_q4k")]
#[inline]
pub fn gemv_q4_k_row_lut(blocks: &[BlockQ4K], x: &[f32]) -> f32 {
    use katgpt_core::simd_lut_dequant::{dequant_dot_via_lut, QuantLut, UInt4Lut};

    let mut acc = 0.0_f32;
    for (b, block) in blocks.iter().enumerate() {
        let d = f32::from(f16::from_bits(block.d));
        let dmin = f32::from(f16::from_bits(block.dmin));
        let qs = &block.qs;
        let scales = &block.scales;
        for pair in 0..4 {
            let (sc0, m0) = get_scale_min_k4(pair * 2, scales);
            let d_sc0 = d * sc0 as f32;
            let m0_val = dmin * m0 as f32;
            let (sc1, m1) = get_scale_min_k4(pair * 2 + 1, scales);
            let d_sc1 = d * sc1 as f32;
            let m1_val = dmin * m1 as f32;
            let qs_base = pair * 32;
            let x_base = b * QK_K + pair * 64;
            let qs_slice = &qs[qs_base..qs_base + 32];

            // Low nibble sub-block 2p
            if d_sc0 != 0.0 {
                let lut0 = UInt4Lut::build(d_sc0, m0_val / d_sc0);
                acc += dequant_dot_via_lut(qs_slice, &lut0, &x[x_base..x_base + 32], 0, 0x0F);
            } else {
                // d_sc0 == 0: every code maps to -m0_val (constant offset)
                acc += -m0_val * x[x_base..x_base + 32].iter().sum::<f32>();
            }

            // High nibble sub-block 2p+1
            if d_sc1 != 0.0 {
                let lut1 = UInt4Lut::build(d_sc1, m1_val / d_sc1);
                acc += dequant_dot_via_lut(
                    qs_slice,
                    &lut1,
                    &x[x_base + 32..x_base + 64],
                    4,
                    0x0F,
                );
            } else {
                acc += -m1_val * x[x_base + 32..x_base + 64].iter().sum::<f32>();
            }
        }
    }
    acc
}

/// Full GEMV: `output[i] = dequant(blocks_row_i) · x`.
///
/// `blocks` is laid out as `m` rows of `nb = n / QK_K` blocks each (row-major).
/// `x` has length `n`. Returns a length-`m` vector.
///
/// This is the canonical CPU reference for `Q4_K` GEMV — replaces the duplicated
/// inline `cpu_q4k_gemv` test helper in `riir-gpu/src/gemv_q4k_cubecl.rs`.
#[inline]
pub fn gemv_q4_k(blocks: &[BlockQ4K], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert!(
        n.is_multiple_of(QK_K),
        "n must be multiple of {QK_K}, got {n}"
    );
    let nb = n / QK_K;
    assert!(
        blocks.len() >= m * nb,
        "blocks too short: need {}, got {}",
        m * nb,
        blocks.len()
    );
    assert!(x.len() >= n, "x too short: need {n}, got {}", x.len());

    let mut out = vec![0.0_f32; m];
    for i in 0..m {
        out[i] = gemv_q4_k_row(&blocks[i * nb..(i + 1) * nb], &x[..n]);
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn test_block_q4k_size() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::align_of::<BlockQ4K>(), 2); // u16 alignment
    }

    #[test]
    fn test_scale_encode_decode_roundtrip() {
        let test_values: [(u8, u8); 8] = [
            (0, 0),
            (63, 0),
            (0, 63),
            (63, 63),
            (32, 16),
            (1, 62),
            (62, 1),
            (31, 31),
        ];

        let mut scales = [0u8; K_SCALE_SIZE];
        for (j, &(sc, mn)) in test_values.iter().enumerate().take(N_SUB_BLOCKS) {
            set_scale_min_k4(j, sc, mn, &mut scales);
        }

        for (j, &expected) in test_values.iter().enumerate().take(N_SUB_BLOCKS) {
            let actual = get_scale_min_k4(j, &scales);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_quantize_dequantize_zeros() {
        let src = vec![0.0f32; QK_K];
        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            assert!(
                v.abs() < f32::EPSILON,
                "Zero roundtrip failed at index {i}: {v}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_uniform() {
        let src = vec![1.5f32; QK_K];
        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            let err = (v - 1.5).abs();
            assert!(
                err < 0.1,
                "Uniform roundtrip error too large at index {i}: {v} (error {err})"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_sine() {
        // Sine wave — tests asymmetric quantization with both positive and negative values
        let src: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 / 32.0) * std::f32::consts::PI * 2.0).sin())
            .collect();

        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dst);

        let mut max_err = 0.0f32;
        let mut mean_err = 0.0f32;
        for (i, (&orig, &deq)) in src.iter().zip(dst.iter()).enumerate() {
            let err = (orig - deq).abs();
            mean_err += err;
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.15,
                "Sine roundtrip error too large at index {i}: orig={orig}, deq={deq}, err={err}"
            );
        }
        mean_err /= QK_K as f32;
        assert!(mean_err < 0.05, "Mean error too large: {mean_err}");
    }

    #[test]
    fn test_quantize_dequantize_negative_range() {
        let src: Vec<f32> = (0..QK_K)
            .map(|i| -(1.0 + (i as f32 % 10.0) / 10.0))
            .collect();
        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dst);

        let mut max_err = 0.0f32;
        for (i, (&orig, &deq)) in src.iter().zip(dst.iter()).enumerate() {
            let err = (orig - deq).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 0.15,
                "Negative range error at index {i}: orig={orig}, deq={deq}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_multi_block() {
        let mut src = Vec::with_capacity(QK_K * 2);
        // Block 0: small values around zero
        for i in 0..QK_K {
            src.push(((i as f32 - 128.0) / 128.0).sin() * 0.1);
        }
        // Block 1: larger values
        for i in 0..QK_K {
            src.push(((i as f32 - 128.0) / 128.0).sin() * 5.0);
        }

        let mut blocks = [BlockQ4K::zeroed(); 2];
        quantize_row_q4_k(&src, &mut blocks);

        let mut dst = vec![0.0f32; QK_K * 2];
        dequantize_row_q4_k(&blocks, &mut dst);

        let mut max_err = 0.0f32;
        for (orig, deq) in src.iter().zip(dst.iter()) {
            let err = (orig - deq).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.5, "Multi-block max error too large: {max_err}");
    }

    #[test]
    fn test_quantize_weight_distribution() {
        // Simulate typical transformer weight distribution: N(0, sigma)
        // Q4_K works best when values share a similar scale within each sub-block.
        let mut src = vec![0.0f32; QK_K];
        let sigma = 0.02f32;
        for (i, val) in src.iter_mut().enumerate() {
            // Deterministic pseudo-normal via sine mixing
            let x = ((i as f32 * 12.989_8 + 78.233).sin() * 43758.545).fract();
            *val = (x - 0.5) * 2.0 * sigma;
        }

        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dst);

        let mut max_err = 0.0f32;
        let mut mse = 0.0f32;
        for (&orig, &deq) in src.iter().zip(dst.iter()) {
            let err = (orig - deq).abs();
            if err > max_err {
                max_err = err;
            }
            mse += (orig - deq) * (orig - deq);
        }
        let rmse = (mse / QK_K as f32).sqrt();
        // Q4_K at 4.5 bpw typically achieves RMSE < 10% of value range
        assert!(
            max_err < sigma,
            "Max error too large: {max_err} (sigma={sigma})"
        );
        assert!(rmse < sigma * 0.5, "RMSE too large: {rmse} (sigma={sigma})");
    }

    #[test]
    fn test_dequantize_matches_llama_cpp_pattern() {
        // Verify our dequantize matches the llama.cpp pattern:
        // Each 64-element group: 32 from low nibbles, 32 from high nibbles
        let src: Vec<f32> = (0..QK_K)
            .map(|i| (i as f32 * 0.123 - 15.0) / 10.0)
            .collect();

        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);
        let block = &blocks[0];

        // Manually dequantize using the llama.cpp pattern
        let d = f32::from(f16::from_bits(block.d));
        let dmin = f32::from(f16::from_bits(block.dmin));
        let mut expected = vec![0.0f32; QK_K];

        let mut q_idx = 0usize;
        let mut is = 0usize;
        for _ in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, &block.scales);
            let d0 = d * sc0 as f32;
            let m0_val = dmin * m0 as f32;

            let (sc1, m1) = get_scale_min_k4(is + 1, &block.scales);
            let d1 = d * sc1 as f32;
            let m1_val = dmin * m1 as f32;

            let out_base = is * SUB_BLOCK_SIZE;
            for l in 0..32 {
                expected[out_base + l] = d0 * (block.qs[q_idx + l] & 0x0F) as f32 - m0_val;
            }
            for l in 0..32 {
                expected[out_base + 32 + l] = d1 * (block.qs[q_idx + l] >> 4) as f32 - m1_val;
            }

            q_idx += 32;
            is += 2;
        }

        // Verify dequantize_row_q4_k matches manual result
        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dst);

        for (i, (&a, &b)) in expected.iter().zip(dst.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "Manual vs function mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        // 256 f32 values = 1024 bytes → 1 block = 144 bytes
        let src = vec![1.0f32; QK_K];
        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let original_bytes = src.len() * 4; // 1024
        let compressed_bytes = std::mem::size_of::<BlockQ4K>(); // 144
        let ratio = original_bytes as f64 / compressed_bytes as f64;

        assert!(
            ratio > 7.0,
            "Compression ratio too low: {ratio:.1}x (expected ~7.1x)"
        );
    }

    // ── Plan 486: fused gemv_q4_k tests ────────────────────────────

    /// Helper: quantize a row of f32 values to `Q4_K` blocks and return them.
    fn quantize_row(src: &[f32]) -> Vec<BlockQ4K> {
        assert!(src.len().is_multiple_of(QK_K));
        let mut blocks = vec![BlockQ4K::zeroed(); src.len() / QK_K];
        quantize_row_q4_k(src, &mut blocks);
        blocks
    }

    #[test]
    fn test_gemv_q4_k_row_arithmetic_matches_dequant_then_dot() {
        // The fused arithmetic path must match the explicit two-step:
        // dequantize_row_q4_k into a buffer, then scalar dot with x.
        let src: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 / 32.0) * std::f32::consts::PI * 2.0).sin())
            .collect();
        let blocks = quantize_row(&src);
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.1).cos()).collect();

        // Two-step reference
        let mut dequant = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut dequant);
        let two_step: f32 = dequant.iter().zip(&x).map(|(a, b)| a * b).sum();

        // Fused arithmetic
        let fused = gemv_q4_k_row_arithmetic(&blocks, &x);
        assert!(
            (fused - two_step).abs() < 1e-4,
            "fused arithmetic {fused} != two-step {two_step}"
        );
    }

    #[test]
    fn test_gemv_q4_k_row_zero_input() {
        // Zero weights → zero output (dot of anything with zeros is zero)
        let src = vec![0.0_f32; QK_K];
        let blocks = quantize_row(&src);
        let x = vec![1.0_f32; QK_K];

        let result = gemv_q4_k_row_arithmetic(&blocks, &x);
        assert!(
            result.abs() < 0.5,
            "zero-weight dot should be ~0, got {result}"
        );
    }

    #[test]
    fn test_gemv_q4_k_row_dispatcher_uses_arithmetic_without_feature() {
        // Without simd_lut_q4k, the dispatcher must use the arithmetic path.
        // We verify by checking the dispatcher matches the arithmetic oracle.
        let src: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.05).sin()).collect();
        let blocks = quantize_row(&src);
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.07).cos()).collect();

        let via_dispatcher = gemv_q4_k_row(&blocks, &x);
        let via_arithmetic = gemv_q4_k_row_arithmetic(&blocks, &x);

        // When feature is off, these must be bit-identical (same code path).
        // When feature is on, they may differ by FMA reordering — so we use
        // a relative tolerance that works for both cases.
        #[cfg(feature = "simd_lut_q4k")]
        {
            let rel = (via_dispatcher - via_arithmetic).abs()
                / via_arithmetic.abs().max(1e-10);
            assert!(rel < 1e-5, "LUT vs arithmetic rel diff {rel} too large");
        }
        #[cfg(not(feature = "simd_lut_q4k"))]
        {
            assert_eq!(
                via_dispatcher.to_bits(),
                via_arithmetic.to_bits(),
                "dispatcher must be bit-identical to arithmetic when feature off"
            );
        }
    }

    #[test]
    fn test_gemv_q4_k_row_matches_arithmetic_single_block() {
        // T2.2: LUT path matches arithmetic oracle on single block.
        // Only meaningful when the feature is on.
        let src: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 / 16.0) * std::f32::consts::PI).sin() * 2.0)
            .collect();
        let blocks = quantize_row(&src);
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.03).cos() * 0.5).collect();

        let arithmetic = gemv_q4_k_row_arithmetic(&blocks, &x);

        #[cfg(feature = "simd_lut_q4k")]
        {
            let lut = gemv_q4_k_row_lut(&blocks, &x);
            let rel = (lut - arithmetic).abs() / arithmetic.abs().max(1e-10);
            assert!(
                rel < 1e-5,
                "LUT {lut} vs arithmetic {arithmetic}: rel diff {rel} >= 1e-5"
            );
        }
        #[cfg(not(feature = "simd_lut_q4k"))]
        {
            // Sanity: arithmetic oracle is finite.
            assert!(arithmetic.is_finite(), "arithmetic oracle not finite");
        }
    }

    #[test]
    fn test_gemv_q4_k_row_matches_arithmetic_multi_block() {
        // T2.3: multi-block row (4 blocks = 1024 elements) — per-block LUT rebuild.
        let n = QK_K * 4;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).sin() * 1.5).collect();
        let blocks = quantize_row(&src);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();

        let arithmetic = gemv_q4_k_row_arithmetic(&blocks, &x);

        #[cfg(feature = "simd_lut_q4k")]
        {
            let lut = gemv_q4_k_row_lut(&blocks, &x);
            let rel = (lut - arithmetic).abs() / arithmetic.abs().max(1e-10);
            assert!(
                rel < 1e-5,
                "multi-block LUT {lut} vs arithmetic {arithmetic}: rel diff {rel} >= 1e-5"
            );
        }
        #[cfg(not(feature = "simd_lut_q4k"))]
        {
            assert!(arithmetic.is_finite());
        }
    }

    #[test]
    fn test_gemv_q4_k_full_gemv() {
        // T2.4: full GEMV (2×256) produces sensible output.
        let m = 2;
        let n = QK_K;
        let src: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let blocks = quantize_row(&src);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();

        let result = gemv_q4_k(&blocks, &x, m, n);
        assert_eq!(result.len(), m);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} not finite");
        }

        // Cross-check: row-by-row arithmetic matches the GEMV.
        let nb = n / QK_K;
        for i in 0..m {
            let row_arith = gemv_q4_k_row_arithmetic(&blocks[i * nb..(i + 1) * nb], &x);
            assert!(
                (result[i] - row_arith).abs() < 1e-4,
                "GEMV row {i}: {} vs arithmetic {}",
                result[i],
                row_arith
            );
        }
    }

    #[test]
    fn test_gemv_q4_k_row_determinism() {
        // Same input → same output across 100 calls (G6 determinism precondition).
        let src: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.1).sin()).collect();
        let blocks = quantize_row(&src);
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.05).cos()).collect();

        let first = gemv_q4_k_row(&blocks, &x);
        for _ in 0..100 {
            let v = gemv_q4_k_row(&blocks, &x);
            assert_eq!(v.to_bits(), first.to_bits(), "non-deterministic gemv_q4_k_row");
        }
    }
}
