//! `Q2_K` k-quant format (2.625 bits/weight) — 2-bit values with per-sub-block
//! 4-bit scale/min pairs and f16 super-block scales.
//!
//! Issue 780 U3 wall-1 lever (a): the `DFlash2` chat drafter loads the
//! analogalok `Q2_K` GGUF quant-resident (`F:/models/Qwen3.8-27B-DFlash2-Q2_K.gguf`,
//! 705 MB vs the 3.85 GB BF16 safetensors). This module is the host-side
//! reference dequant; the in-kernel dequant (`dflash2_gemv_q2k_rows` in
//! riir-gpu) mirrors it op-for-op and is pinned against it by the
//! kernel-vs-host cosine gate.
//!
//! Block layout — **verbatim** `ggml-common.h` `block_q2_K` (`QK_K` = 256),
//! field order included (scales FIRST, then qs, then the f16 pair at the
//! end; total 84 bytes):
//!
//! ```text
//! struct block_q2_K {          // 84 bytes / 256 weights
//!     uint8_t  scales[16];     // 16 sub-blocks × (4-bit scale | 4-bit min<<4)
//!     uint8_t  qs[64];         // 2-bit codes, 4 per byte, LSB-first
//!     ggml_half d;             // super-block scale for quantized scales
//!     ggml_half dmin;          // super-block scale for quantized mins
//! }
//! ```
//!
//! Dequant reference: `dequantize_row_q2_K` in `ggml-quants.c`
//! (llama.cpp @ the PR-27342 fork) — `y = d*(sc&0xF)*q − dmin*(sc>>4)`
//! per sub-block of 16, where the byte holding the 2-bit code for flat
//! element `(half*128 + (2j+b)*16 + l)` is `qs[half*32 + b*16 + l]` and
//! the shift is `2*j` (the reference's `n/j/shift` triple loop flattens to
//! exactly this mapping — pinned by the module tests).

/// Super-block size: 256 elements per block (the k-quant convention).
pub const QK_K: usize = 256;

/// `Q2_K` block: 84 bytes per 256 weights (2.625 bits/weight).
///
/// `#[repr(C)]` with no padding — byte-identical to ggml's `block_q2_K`,
/// so `bytemuck::cast_slice` over a GGUF tensor slice is sound.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ2K {
    /// Per-sub-block scales and mins, 4 bits each, one byte per sub-block:
    /// `scale | (min << 4)` for sub-block `i` (16 sub-blocks of 16 elements).
    pub scales: [u8; QK_K / 16],
    /// 2-bit quantized values, 4 per byte, LSB-first within each byte.
    pub qs: [u8; QK_K / 4],
    /// Super-block scale for the quantized sub-scales (f16 bits).
    pub d: u16,
    /// Super-block scale for the quantized sub-mins (f16 bits).
    pub dmin: u16,
}

const _: () = assert!(std::mem::size_of::<BlockQ2K>() == 84);
// NOTE: align is 2 (the u16 f16 fields) — bytemuck::cast_slice checks the
// DATA pointer at runtime, and every block address is even (data_start is
// 32-aligned, strides 84/110/144 are even), so the casts are sound.

use bytemuck::Zeroable as _;

/// Dequantize a row of `Q2_K` blocks to f32.
///
/// `src.len()` blocks cover `src.len() * QK_K` elements; `dst` must be
/// that long. Element mapping is the flat form of the ggml reference loop
/// (see the module doc).
pub fn dequantize_row_q2_k(src: &[BlockQ2K], dst: &mut [f32]) {
    let nb = src.len();
    assert!(
        dst.len() >= nb * QK_K,
        "dst too short: need {} got {}",
        nb * QK_K,
        dst.len()
    );
    for (i, blk) in src.iter().enumerate() {
        let d = half::f16::from_bits(blk.d).to_f32();
        let dmin = half::f16::from_bits(blk.dmin).to_f32();
        for sb16 in 0..16usize {
            let sc = blk.scales[sb16];
            let dl = d * (sc & 0xF) as f32;
            let ml = dmin * (sc >> 4) as f32;
            // half = sb16/8 selects which 128-element half (qs base +32 per
            // half); b = sb16&1 selects the 16-byte lane within it; j =
            // (sb16%8)/2 sets the 2-bit shift. Flat element = sb16*16 + l.
            let half = sb16 >> 3;
            let b = sb16 & 1;
            let shift = 2 * ((sb16 & 7) >> 1);
            let qbase = half * 32 + b * 16;
            let base = i * QK_K + sb16 * 16;
            for l in 0..16usize {
                let q = ((blk.qs[qbase + l] >> shift) & 3) as f32;
                dst[base + l] = dl * q - ml;
            }
        }
    }
}

/// Quantize a row to `Q2_K` blocks — the ROUND-TRIP FIXTURE encoder (NOT the
/// ggml-exact quantizer; any valid encoding round-trips through the dequant).
///
/// Per sub-block: 4-bit min offset `ml`, 4-bit scale `dl`, 2-bit codes
/// `q = round((v + ml) / dl)` — the dequant's own inverse.
pub fn quantize_row_q2_k_fixture(src: &[f32], dst: &mut [BlockQ2K]) {
    let nb = src.len() / QK_K;
    assert!(
        src.len() >= nb * QK_K,
        "src not a multiple of {QK_K}"
    );
    assert!(dst.len() >= nb, "dst too short for {nb} blocks");
    for i in 0..nb {
        let mut blk = BlockQ2K::zeroed();
        blk.d = half::f16::from_f32(1.0).to_bits();
        blk.dmin = half::f16::from_f32(1.0).to_bits();
        for sb in 0..16usize {
            let seg = &src[i * QK_K + sb * 16..i * QK_K + sb * 16 + 16];
            let mn = seg.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = seg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ml = (-mn).clamp(0.0, 15.0).round();
            let dl = ((mx + ml) / 3.0).clamp(1.0, 15.0).round();
            blk.scales[sb] = (dl as u8 & 0xF) | ((ml as u8 & 0xF) << 4);
            let half = sb >> 3;
            let b = sb & 1;
            let shift = 2 * ((sb & 7) >> 1);
            let qbase = half * 32 + b * 16;
            for (l, &v) in seg.iter().enumerate() {
                let q = (((v + ml) / dl).round()).clamp(0.0, 3.0) as u8;
                blk.qs[qbase + l] |= (q & 3) << shift;
            }
        }
        dst[i] = blk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16b(v: f32) -> u16 {
        half::f16::from_f32(v).to_bits()
    }

    #[test]
    fn hand_block_dequants_to_reference_mapping() {
        // One block, hand-set fields; assertions pin the EXACT byte/shift
        // mapping. The qs lanes are INTERLEAVED: element l of sub-block
        // (half, b, j) = BYTE qs[half*32 + b*16 + l] at shift 2*j (one code
        // per byte position — the reference's `q[l] >> shift`).
        let mut blk = BlockQ2K::zeroed();
        blk.d = f16b(2.0);
        blk.dmin = f16b(0.5);
        blk.scales[0] = 3 | (2 << 4); // sub-block 0: sc=3, min=2 → dl=6, ml=1.0
        blk.scales[1] = 1; //  sub-block 1: sc=1, min=0 → dl=2, ml=0
        blk.scales[2] = 1; //  sub-block 2: sc=1, min=0 → dl=2, ml=0
        blk.scales[8] = 1; //  sub-block 8 (half 1): sc=1, min=0 → dl=2, ml=0
        // sub-block 0 (half0, b0, j0 → bytes qs[0..16], shift 0):
        blk.qs[0] = 0;
        blk.qs[1] = 1;
        blk.qs[2] = 2;
        blk.qs[3] = 3; // elements 0..4 → q = 0,1,2,3
        // sub-block 1 (b=1 → bytes qs[16..32], shift 0): elem 16 → q=2
        blk.qs[16] = 2;
        // sub-block 2 (j=1 → shift 2, bytes qs[0..16] AGAIN — the interleave):
        // elem 32 reads qs[0] bits 2..3
        blk.qs[0] |= 1 << 2;
        // sub-block 8 (half1 → bytes qs[32..48], shift 0): elem 128 → q=1
        blk.qs[32] = 1;
        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q2_k(&[blk], &mut dst);
        // Sub-block 0 (elements 0..16):
        assert_eq!(dst[0], 6.0 * 0.0 - 1.0);
        assert_eq!(dst[1], 6.0 * 1.0 - 1.0);
        assert_eq!(dst[2], 6.0 * 2.0 - 1.0);
        assert_eq!(dst[3], 6.0 * 3.0 - 1.0);
        // Sub-block 1 (elements 16..32):
        assert_eq!(dst[16], 2.0 * 2.0);
        // Sub-block 2 (elements 32..48, j=1 → shift 2, bytes qs[0..16]):
        assert_eq!(dst[32], 2.0 * 1.0); // qs[0] bits 2..3 = 1
        assert_eq!(dst[33], 2.0 * 0.0); // qs[1] bits 2..3 = 0
        // Sub-block 8 (elements 128..144, half 1 → bytes qs[32..48]):
        assert_eq!(dst[128], 2.0 * 1.0);
        // Unset sub-blocks (sc=0, min=0) decode to exactly 0:
        assert_eq!(dst[48], 0.0); // sub-block 3
        assert_eq!(dst[255], 0.0); // sub-block 15
    }

    #[test]
    fn fixture_encoder_round_trips_within_code_step() {
        let src: Vec<f32> = (0..QK_K * 3)
            .map(|i| ((i as f32 / 17.0) * std::f32::consts::PI).sin() * 3.0)
            .collect();
        let mut blocks = vec![BlockQ2K::zeroed(); 3];
        quantize_row_q2_k_fixture(&src, &mut blocks);
        let mut dst = vec![0.0f32; src.len()];
        dequantize_row_q2_k(&blocks, &mut dst);
        // Error budget: q rounding ≤ dl/2 (dl ≤ (6+ml)/3 ≤ 7 → ≤ 3.5) + the
        // ml/dl 4-bit rounding residue ≤ ~1. A MAPPING bug produces errors of
        // the full sub-block magnitude (≥ 6 here) and fails this bound.
        for (i, (a, b)) in src.iter().zip(dst.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(err <= 4.5, "idx {i}: {a} vs {b} (err {err})");
        }
    }
}
