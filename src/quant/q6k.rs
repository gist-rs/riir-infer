//! `Q6_K` quantization — 6-bit k-quant (6.5625 bpw, 210 bytes / 256 weights).
//!
//! Layout (matches llama.cpp `block_q6_K`):
//! - `ql`: [u8; 128] — quants, lower 4 bits (2 weights per byte)
//! - `qh`: [u8; 64] — quants, upper 2 bits (4 weights per byte... actually
//!   packed 2 bits per weight into 64 bytes via the specific shift pattern in
//!   `dequantize_row_q6_k`)
//! - `scales`: [i8; 16] — per-sub-block int8 scales (16 sub-blocks of 16)
//! - `d`: f16 super-block scale (2 bytes)
//!
//! Total: 128 + 64 + 16 + 2 = 210 bytes per 256 weights.
//!
//! The model is `w = d * sc_j * q_6bit` where `sc_j` is a direct int8 scale
//! (no min offset, unlike `Q4_K/Q5_K`) and `q_6bit` is the 6-bit quant in [0, 63]
//! re-centered to [-32, 31] by subtracting 32.

use half::f16;

/// Super-block size (256 weights per block — shared across all k-quants).
pub const QK_K: usize = 256;

/// `Q6_K` block: 210 bytes for 256 weights.
///
/// Field order matches the GGML on-disk layout (`block_q6_K` in ggml-common.h).
/// `#[repr(C)]` + `Pod` so `bytemuck::cast_slice::<u8, BlockQ6K>` works on the
/// mmap'd GGUF tensor bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ6K {
    /// quants, lower 4 bits (2 weights per byte, nibble-packed).
    pub ql: [u8; QK_K / 2],
    /// quants, upper 2 bits (2 bits per weight, packed 4-per-byte via the
    /// shift pattern in the dequant loop).
    pub qh: [u8; QK_K / 4],
    /// per-sub-block int8 scales (16 sub-blocks of 16 weights each).
    pub scales: [i8; QK_K / 16],
    /// f16 super-block scale.
    pub d: u16,
}

// Compile-time size check (matches the C static_assert).
const _: () = assert!(
    core::mem::size_of::<BlockQ6K>() == core::mem::size_of::<u16>() + QK_K / 16 + 3 * QK_K / 4,
    "wrong BlockQ6K size"
);

/// Dequantize a row of `Q6_K` blocks into f32.
///
/// Ported from llama.cpp `dequantize_row_q6_K`. The 256 weights are processed
/// in two 128-element halves; within each half, 32 bytes of `ql` + `qh` yield
/// 128 outputs via the 4-way interleaving pattern (q1/q2/q3/q4 per `l`).
pub fn dequantize_row_q6_k(src: &[BlockQ6K], dst: &mut [f32]) {
    let n = dst.len();
    assert!(
        n.is_multiple_of(QK_K),
        "dst length must be multiple of {QK_K}"
    );
    let nb = n / QK_K;
    assert!(src.len() >= nb, "src too short: need {nb} blocks, got {}", src.len());

    for (i, block) in src.iter().take(nb).enumerate() {
        let d = f32::from(f16::from_bits(block.d));

        let ql = &block.ql;
        let qh = &block.qh;
        let sc = &block.scales;

        let block_base = i * QK_K;
        // Two 128-element halves; per llama.cpp the loop advances ql/qh/sc pointers
        // by 64/32/8 between halves. We track byte offsets instead of raw pointers.
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for n_half in 0..2 {
            let out_base = block_base + n_half * 128;
            for l in 0..32 {
                let is = l / 16;
                // q1: ql[l+0] low nibble + qh[l] bits [1:0]
                let q1 = ((ql[ql_off + l] & 0x0F) | (((qh[qh_off + l]) & 3) << 4)) as i32 - 32;
                // q2: ql[l+32] low nibble + qh[l] bits [3:2]
                let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                // q3: ql[l+0] high nibble + qh[l] bits [5:4]
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                // q4: ql[l+32] high nibble + qh[l] bits [7:6]
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                dst[out_base + l] = d * sc[sc_off + is] as f32 * q1 as f32;
                dst[out_base + l + 32] = d * sc[sc_off + is + 2] as f32 * q2 as f32;
                dst[out_base + l + 64] = d * sc[sc_off + is + 4] as f32 * q3 as f32;
                dst[out_base + l + 96] = d * sc[sc_off + is + 6] as f32 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_q6k_size() {
        assert_eq!(core::mem::size_of::<BlockQ6K>(), 210);
    }

    /// All-zero block with d=0 → all zeros.
    #[test]
    fn test_dequantize_zeros() {
        let block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
            d: 0u16,
        };
        let mut out = vec![1.0_f32; QK_K];
        dequantize_row_q6_k(&[block], &mut out);
        assert!(out.iter().all(|&v| v == 0.0), "expected all zeros, got {out:?}");
    }

    /// d=1, scales=1, all-zero quants → q = 0-32 = -32 for every element, so
    /// every output = 1 * 1 * (-32) = -32. Verifies the 6-bit assembly + scale
    /// multiply end-to-end with a known value.
    #[test]
    fn test_dequantize_known_value() {
        let block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [1i8; QK_K / 16],
            d: f16::from_f32(1.0).to_bits(),
        };
        let mut out = vec![0.0_f32; QK_K];
        dequantize_row_q6_k(&[block], &mut out);
        // Every q_6bit = 0 (all-zero ql/qh), re-centered = 0 - 32 = -32.
        // w = d * sc * q = 1.0 * 1.0 * (-32.0) = -32.0.
        assert!(
            out.iter().all(|&v| v == -32.0),
            "expected all -32.0, got sample {:?}",
            &out[..8]
        );
    }
}
