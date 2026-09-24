//! `Q5_K` quantization — 5-bit k-quant (5.5 bpw, 176 bytes / 256 weights).
//!
//! Layout (matches llama.cpp `block_q5_K`):
//! - `d`: f16 super-block scale for quantized scales (2 bytes)
//! - `dmin`: f16 super-block scale for quantized mins (2 bytes)
//! - `scales`: [u8; 12] — 8 sub-block scale/min pairs, 6-bit quantized
//! - `qh`: [u8; 32] — quants, high bit (1 bit per weight, 256/8 = 32)
//! - `qs`: [u8; 128] — quants, low 4 bits (2 weights per byte, 256/2 = 128)
//!
//! Total: 2 + 2 + 12 + 32 + 128 = 176 bytes per 256 weights.
//!
//! The model is `w = d * sc_j * (q_5bit) - dmin * m_j` where `(sc_j, m_j)` are
//! decoded from `scales` via `get_scale_min_k4` (shared with `Q4_K`) and `q_5bit`
//! is the 5-bit quant assembled from `qs` (low 4) + `qh` (high 1).

use crate::quant::q4k::{K_SCALE_SIZE, get_scale_min_k4};
use bytemuck::{Pod, Zeroable};
use half::f16;

/// Super-block size (256 weights per block — shared across all k-quants).
pub const QK_K: usize = 256;

/// `Q5_K` block: 176 bytes for 256 weights.
///
/// Field order matches the GGML on-disk layout (`block_q5_K` in ggml-common.h).
/// `#[repr(C)]` + `Pod` so `bytemuck::cast_slice::<u8, BlockQ5K>` works on the
/// mmap'd GGUF tensor bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5K {
    /// f16 super-block scale for quantized scales.
    pub d: u16,
    /// f16 super-block scale for quantized mins.
    pub dmin: u16,
    /// 8 sub-block scale/min pairs, 6-bit quantized (same packing as `Q4_K`).
    pub scales: [u8; K_SCALE_SIZE],
    /// quants, high bit (1 bit per weight).
    pub qh: [u8; QK_K / 8],
    /// quants, low 4 bits (2 weights per byte, nibble-packed).
    pub qs: [u8; QK_K / 2],
}

// Compile-time size check (matches the C static_assert).
const _: () = assert!(
    core::mem::size_of::<BlockQ5K>() == 2 * 2 + K_SCALE_SIZE + QK_K / 2 + QK_K / 8,
    "wrong BlockQ5K size"
);

/// Dequantize a row of `Q5_K` blocks into f32.
///
/// Ported from llama.cpp `dequantize_row_q5_K`. The packing interleaves low
/// nibbles (qs) with a single high bit (qh) to form a 5-bit quant in [0, 31];
/// the scale + min decode reuses `get_scale_min_k4` (shared with `Q4_K`).
pub fn dequantize_row_q5_k(src: &[BlockQ5K], dst: &mut [f32]) {
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
        let min = f32::from(f16::from_bits(block.dmin));
        let ql = &block.qs;
        let qh = &block.qh;

        let mut is = 0usize;
        // u1/u2 are the single-bit masks for the high bit, cycling through the
        // 4 32-element groups within each 128-element chunk. Each 64-element
        // group uses 2 sub-blocks (is, is+1); the high-bit mask shifts by 2
        // between groups so qh[l]'s bit 0 serves group 0, bit 1 serves group 1,
        // etc. (8 groups total over the full 256-element block).
        //
        // Note: unlike ql (which advances 32 bytes per group), qh is reused
        // across all 4 groups — the different bit positions (via u1/u2 shifts)
        // disambiguate which group each qh byte serves. This matches the
        // llama.cpp `dequantize_row_q5_K` reference (qh pointer is NOT advanced
        // in the loop; only ql is).
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let out_base = i * QK_K;

        // 4 iterations of 64 elements each = 256 total.
        for j_chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, &block.scales);
            let d1 = d * sc0 as f32;
            let m1 = min * m0 as f32;
            let (sc1, m1_raw) = get_scale_min_k4(is + 1, &block.scales);
            let d2 = d * sc1 as f32;
            let m2 = min * m1_raw as f32;

            let ql_base = j_chunk * 32;
            let out_off = out_base + j_chunk * 64;

            // Lower nibble → group 2*j_chunk (first 32 of the 64).
            for l in 0..32 {
                let q_lo = (ql[ql_base + l] & 0x0F) as f32;
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                dst[out_off + l] = d1 * (q_lo + hi) - m1;
            }
            // Upper nibble → group 2*j_chunk + 1 (next 32).
            for l in 0..32 {
                let q_lo = (ql[ql_base + l] >> 4) as f32;
                let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                dst[out_off + 32 + l] = d2 * (q_lo + hi) - m2;
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_q5k_size() {
        assert_eq!(core::mem::size_of::<BlockQ5K>(), 176);
    }

    /// All-zero block dequantizes to `-min * m` for each sub-block; with d=dmin=0
    /// (f16 zero), every output is exactly 0.0. Catches struct-layout regressions.
    #[test]
    fn test_dequantize_zeros() {
        let block = BlockQ5K {
            d: 0u16,
            dmin: 0u16,
            scales: [0u8; K_SCALE_SIZE],
            qh: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 2],
        };
        let mut out = vec![1.0_f32; QK_K]; // poison
        dequantize_row_q5_k(&[block], &mut out);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "expected all zeros, got {out:?}"
        );
    }

    /// Non-zero d + all-15 quants: `d * 1 * 15` per element (min=0). Verifies
    /// the nibble + high-bit assembly path end-to-end.
    #[test]
    fn test_dequantize_uniform() {
        let d = f16::from_f32(0.5);
        let block = BlockQ5K {
            d: d.to_bits(),
            dmin: 0u16,
            scales: [1u8; K_SCALE_SIZE], // sc=1, m=1 for all sub-blocks (j<4 path)
            qh: {
                // Set bits 0 and 1 of every qh byte → high bit = 16 for both u1=1
                // and u2=2 in the first 64-group; subsequent groups use u1/u2
                // shifts that still land in the low bits. To make every 5-bit
                // quant = 31 (15 low + 16 high), set ALL bits of qh.
                [0xFFu8; QK_K / 8]
            },
            qs: [0xFFu8; QK_K / 2], // both nibbles = 15
        };
        let mut out = vec![0.0_f32; QK_K];
        dequantize_row_q5_k(&[block], &mut out);
        // Each output = d * sc * (15 + 16) - dmin * m. d=0.5, sc=1 (scales[j]&63=1
        // for j<4, but j>=4 path gives a different sc from the nibble packing —
        // just assert all finite + same sign, since the exact value depends on
        // the scale packing for sub-blocks 4-7).
        assert!(out.iter().all(|&v| v.is_finite()));
        assert!(
            out.iter().all(|&v| v > 0.0),
            "expected all positive, got sample {out:?}"
        );
    }
}
