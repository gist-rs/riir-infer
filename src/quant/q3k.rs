//! `Q3_K` k-quant format (3.4375 bits/weight) — 3-bit values with 6-bit
//! sub-block scales (offset −32) and an f16 super-block scale.
//!
//! Issue 780 U3 wall-1 lever (a): the `DFlash2` chat drafter loads the
//! analogalok `Q2_K` GGUF quant-resident, whose mixed-quant artifact carries
//! `attn_output` + `ffn_down` as `Q3_K`. Host-side reference dequant here;
//! the in-kernel twin (`dflash2_gemv_q3k_rows` in riir-gpu) mirrors it and
//! is pinned against it by the kernel-vs-host cosine gate.
//!
//! Block layout — **verbatim** `ggml-common.h` `block_q3_K` (`QK_K` = 256),
//! field order included (hmask FIRST, then qs, scales, d; total 110 bytes):
//!
//! ```text
//! struct block_q3_K {          // 110 bytes / 256 weights
//!     uint8_t  hmask[32];      // high bit of each code (1 bit/element)
//!     uint8_t  qs[64];         // low 2 bits of each code (2 bits/element)
//!     uint8_t  scales[12];     // 16 sub-block scales, 6 bits each (packed)
//!     ggml_half d;             // super-block scale
//! }
//! ```
//!
//! Dequant reference: `dequantize_row_q3_K` in `ggml-quants.c` — the 12
//! scale bytes unpack into 16 signed 6-bit values via the `kmask1/kmask2`
//! u32 interleave (`sc = scales[i] − 32`), and
//! `y = d*sc*((q & 3) − (hbit ? 0 : 4))` per element, where the code byte
//! for flat element `(half*128 + (2j+b)*16 + l)` is `qs[half*32 + b*16 + l]`
//! at shift `2*j` (same interleaved layout as `Q2_K`) and the hmask byte is
//! `hmask[b*16 + l]` at bit `half*4 + j`.

/// Super-block size: 256 elements per block (the k-quant convention).
pub const QK_K: usize = 256;

/// `Q3_K` block: 110 bytes per 256 weights (3.4375 bits/weight).
///
/// `#[repr(C)]` with no padding — byte-identical to ggml's `block_q3_K`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ3K {
    /// High bit of each element's code (bit `half*4 + j` of byte `b*16 + l`).
    pub hmask: [u8; QK_K / 8],
    /// Low 2 bits of each element's code, 4 per byte, LSB-first.
    pub qs: [u8; QK_K / 4],
    /// 16 sub-block scales, 6 bits each, packed via the kmask interleave.
    pub scales: [u8; 12],
    /// Super-block scale (f16 bits).
    pub d: u16,
}

const _: () = assert!(std::mem::size_of::<BlockQ3K>() == 110);
// NOTE: align is 2 (the u16 d field) — same cast soundness note as q2k.

use bytemuck::Zeroable as _;

const KMASK1: u32 = 0x0303_0303;
const KMASK2: u32 = 0x0F0F_0F0F;

/// Unpack the 12 packed scale bytes into 16 signed 6-bit scale values
/// (still offset by +32 — callers subtract). Verbatim ggml interleave.
#[inline]
fn unpack_scales(scales: &[u8; 12]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    aux[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    aux[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    // Order matters: aux[2]/aux[3] consume the ORIGINAL aux[0]/aux[1]/tmp
    // before aux[0]/aux[1] are overwritten (the C reference's sequence).
    let tmp = aux[2];
    let a2 = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    let a3 = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    // Verbatim ggml interleave — the `>> 0` lane is kept for reference
    // symmetry (>> 0 / >> 2 / >> 4 / >> 6 map the four a-words).
    #[allow(clippy::identity_op)]
    let a0 = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    let a1 = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut out = [0i8; 16];
    let words = [a0, a1, a2, a3];
    for i in 0..16usize {
        let b = words[i >> 2].to_le_bytes()[i & 3];
        out[i] = b as i8;
    }
    out
}

/// Dequantize a row of `Q3_K` blocks to f32.
///
/// `src.len()` blocks cover `src.len() * QK_K` elements; `dst` must be
/// that long. Element mapping is the flat form of the ggml reference loop
/// (see the module doc).
pub fn dequantize_row_q3_k(src: &[BlockQ3K], dst: &mut [f32]) {
    let nb = src.len();
    assert!(
        dst.len() >= nb * QK_K,
        "dst too short: need {} got {}",
        nb * QK_K,
        dst.len()
    );
    for (i, blk) in src.iter().enumerate() {
        let d_all = half::f16::from_bits(blk.d).to_f32();
        let sc = unpack_scales(&blk.scales);
        for (sb16, &sc_v) in sc.iter().enumerate() {
            let dl = d_all * (sc_v - 32) as f32;
            let half = sb16 >> 3;
            let b = sb16 & 1;
            let j = (sb16 & 7) >> 1;
            let shift = 2 * j;
            let qbase = half * 32 + b * 16;
            let hbase = b * 16;
            let hbit = (half * 4 + j) as u8;
            let base = i * QK_K + sb16 * 16;
            for l in 0..16usize {
                let q = ((blk.qs[qbase + l] >> shift) & 3) as i32;
                let hi = if (blk.hmask[hbase + l] >> hbit) & 1 == 1 {
                    0
                } else {
                    4
                };
                dst[base + l] = dl * (q - hi) as f32;
            }
        }
    }
}

/// Quantize a row to `Q3_K` blocks — the ROUND-TRIP FIXTURE encoder (NOT the
/// ggml-exact quantizer). Per sub-block: a 6-bit scale (offset +32) and
/// 3-bit signed codes centered at −4..3 via the hmask bit.
pub fn quantize_row_q3_k_fixture(src: &[f32], dst: &mut [BlockQ3K]) {
    let nb = src.len() / QK_K;
    assert!(src.len() >= nb * QK_K, "src not a multiple of {QK_K}");
    assert!(dst.len() >= nb, "dst too short for {nb} blocks");
    for i in 0..nb {
        let mut blk = BlockQ3K::zeroed();
        // Simple per-super-block scale: fit max |v|/4 into the 6-bit range.
        let amax: f32 = src[i * QK_K..(i + 1) * QK_K]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        // codes span −4..3 (8 levels); scale s.t. 4*dl_max ≥ amax, dl in
        // [−32,31]−32 = we encode sc = round(31 * amax / (4*amax_4))… keep it
        // simple: d = 1.0, sc chosen so dl·4 covers amax: dl = amax/4 → sc =
        // dl (offset removed on dequant). Clamp to the 6-bit range.
        let dl = (amax / 4.0).clamp(1.0, 31.0);
        let sc_i = (dl.round() as i32).clamp(1, 31) + 32; // offset +32
        let dl = sc_i as f32 - 32.0;
        // pack sc into all 16 sub-block slots via the inverse interleave:
        // easiest correct path — build the 16 bytes then re-pack. The
        // interleave is its own structure; emulate the reference packing by
        // constructing the pre-interleave "aux" the unpack inverts:
        // the unpack produces words a0..a3 from tmp = aux[2]; invert by
        // assigning every byte the same value v = sc_i (all sub-blocks share
        // the scale in this fixture) and solving: if all 16 output bytes are
        // v, then a0..a3 all have bytes = v. Work backwards:
        //   a0 = (x0 & kmask2) | ((t>>0 & kmask1)<<4)  with bytes v ⇒
        //   x0 bytes = v&0x0F, t bits supply v>>4 in the 2-bit lanes.
        // Constant-fill packing: every unpacked byte = v. From the inverse
        // interleave — scales[0..8] carry the low nibble in BOTH nibbles (a0/a1
        // read the low nibble, a2/a3 read the HIGH nibble of the same bytes);
        // each byte of scales[8..12] (= tmp) carries the SAME 2-bit high field
        // `h` in all four of its lanes (h*0x55), so a0..a3 all receive h<<4.
        // Pinned by the round-trip test.
        let v = sc_i as u8 & 0x3F;
        let lo = v & 0x0F;
        for j in 0..8 {
            blk.scales[j] = lo | (lo << 4);
        }
        let h = (v >> 4) & 0x3;
        let tmp = u32::from_le_bytes([h.wrapping_mul(0x55); 4]);
        blk.scales[8..12].copy_from_slice(&tmp.to_le_bytes());
        blk.d = half::f16::from_f32(1.0).to_bits();
        for sb in 0..16usize {
            let half = sb >> 3;
            let b = sb & 1;
            let j = (sb & 7) >> 1;
            let shift = 2 * j;
            let qbase = half * 32 + b * 16;
            let hbase = b * 16;
            let hbit = (half * 4 + j) as u8;
            for l in 0..16usize {
                let x = src[i * QK_K + sb * 16 + l] / dl;
                // signed code −4..3: q3 = round(x) clamped; hbit carries the
                // sign class (h=1 → 0..3, h=0 → −4..−1).
                let c = x.round().clamp(-4.0, 3.0) as i32;
                let (ql, hi) = if c >= 0 {
                    (c, true)
                } else {
                    (c + 4, false)
                };
                blk.qs[qbase + l] |= ((ql as u8) & 3) << shift;
                if hi {
                    blk.hmask[hbase + l] |= 1 << hbit;
                }
            }
        }
        dst[i] = blk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_unpack_matches_the_reference_interleave() {
        // Distinctive 12 bytes; assert the FULL 16-byte unpack against the
        // C reference computed by hand (aux words → byte interleave).
        let mut s = [0u8; 12];
        for (i, b) in s.iter_mut().enumerate() {
            *b = 0x11 * (i as u8 + 1); // 0x11,0x22,...,0xCC
        }
        // Reference (little-endian u32 loads):
        let aux0 = u32::from_le_bytes([0x11, 0x22, 0x33, 0x44]);
        let aux1 = u32::from_le_bytes([0x55, 0x66, 0x77, 0x88]);
        let aux2 = u32::from_le_bytes([0x99, 0xAA, 0xBB, 0xCC]);
        let tmp = aux2;
        let r2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        let r3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        let r0 = (aux0 & KMASK2) | ((tmp & KMASK1) << 4);
        let r1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let want: Vec<i8> = [r0, r1, r2, r3]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(|b| b as i8)
            .collect();
        let got = unpack_scales(&s);
        assert_eq!(got.as_slice(), want.as_slice());
        // sanity: the values must stay in the 6-bit domain
        for g in got {
            assert!((-32..=63).contains(&g), "scale {g} outside 6-bit domain");
        }
    }

    #[test]
    fn hand_block_dequants_to_reference_mapping() {
        // Constant-fill scales (the fixture's own construction — all 16 slots
        // unpack to v) with v = 38 → dl = 1.0*(38−32) = 6 everywhere. Codes:
        // element l of sub-block (half, b, j) = BYTE qs[half*32 + b*16 + l] at
        // shift 2*j; hmask byte = b*16 + l at bit half*4 + j (hi → no −4).
        let mut blk = BlockQ3K::zeroed();
        blk.d = half::f16::from_f32(1.0).to_bits();
        let v: u8 = 38;
        let lo = v & 0x0F;
        for j in 0..8 {
            blk.scales[j] = lo | (lo << 4);
        }
        let h = (v >> 4) & 0x3;
        let tmp = u32::from_le_bytes([h.wrapping_mul(0x55); 4]);
        blk.scales[8..12].copy_from_slice(&tmp.to_le_bytes());
        // sub-block 0 (bytes qs[0..16], shift 0, hmask[0..16] bit 0):
        blk.qs[0] = 0;
        blk.qs[1] = 1;
        blk.qs[2] = 2;
        blk.qs[3] = 3; // elems 0..4 → q = 0,1,2,3
        blk.hmask[0] = 1; // elem 0 hi (bit 0)
        // sub-block 2 (j=1 → shift 2, bytes qs[0..16] again, hbit = 0*4+1 = 1):
        blk.qs[0] |= 1 << 2; // elem 32 → q = 1 at shift 2
        blk.hmask[0] |= 1 << 1; // …and hi (hmask bit 1)
        // sub-block 8 (half1 → bytes qs[32..48], shift 0, hbit 4):
        blk.qs[32] = 2;
        blk.hmask[0] |= 1 << 4; // hmask byte 0 (b=0), bit 4 → hi
        let mut dst = vec![0.0f32; QK_K];
        dequantize_row_q3_k(&[blk], &mut dst);
        // Sub-block 0 (dl = 6):
        assert_eq!(dst[0], 6.0 * (0.0 - 0.0)); // hi → no −4
        assert_eq!(dst[1], 6.0 * (1.0 - 4.0)); // lo → −3
        assert_eq!(dst[2], 6.0 * (2.0 - 4.0));
        assert_eq!(dst[3], 6.0 * (3.0 - 4.0));
        // Sub-block 1 (bytes qs[16..32]): unset → code 0, lo → −4
        assert_eq!(dst[16], 6.0 * (0.0 - 4.0));
        // Sub-block 2 (shift 2 re-reading qs[0..16]): elem 32 → q=1, hi
        assert_eq!(dst[32], 6.0 * 1.0);
        // Sub-block 8 (half1, bytes qs[32..48], hbit 4): elem 128 → q=2, hi
        assert_eq!(dst[128], 6.0 * 2.0);
        // Sub-block 15 (half1, b=1, j=3 → bytes qs[48..64], shift 6, hbit 7):
        // unset → 6*(0−4)
        assert_eq!(dst[255], 6.0 * (0.0 - 4.0));
    }

    #[test]
    fn fixture_encoder_round_trips_within_code_step() {
        let src: Vec<f32> = (0..QK_K * 3)
            .map(|i| ((i as f32 / 13.0) * std::f32::consts::PI).cos() * 4.0)
            .collect();
        let mut blocks = vec![BlockQ3K::zeroed(); 3];
        quantize_row_q3_k_fixture(&src, &mut blocks);
        let mut dst = vec![0.0f32; src.len()];
        dequantize_row_q3_k(&blocks, &mut dst);
        for (i, (a, b)) in src.iter().zip(dst.iter()).enumerate() {
            let err = (a - b).abs();
            // dl ≤ 31/4·… amax ≤ 4 → dl ≤ 1 → code step ≤ 1; allow scale
            // rounding + clamp residue.
            assert!(err <= 2.5, "idx {i}: {a} vs {b} (err {err})");
        }
    }
}
