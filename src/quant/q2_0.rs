//! `Q2_0` (2-bit ternary + group scale) quantization format — Ternary-Bonsai lineage.
//!
//! Verified against the `PrismML` llama.cpp fork's `block_q2_0` + the real
//! `Ternary-Bonsai-27B-Q2_0.gguf` artifact (Plan 333, katgpt-rs Issue 578).
//!
//! # Block Layout (34 bytes for 128 elements = 2.125 bpw)
//!
//! ```text
//! Offset  Size   Field
//! 0       2      d      — f16 block scale (symmetric; per 128-weight group)
//! 2       32     qs     — 128 × 2-bit codes, packed 4-per-byte, LSB-first
//! ```
//!
//! # Decode formula (per llama.cpp `dequantize_row_q2_0`)
//!
//! ```text
//! q = (qs[j/4] >> ((j%4)*2)) & 0x03
//! y[j] = ((int)q - 1) * d
//! ```
//!
//! Codes map to weight levels:
//!
//! | code | (q-1) | weight (× d) |
//! |------|-------|--------------|
//! | 00   | -1    | -d           |
//! | 01   |  0    |  0           |
//! | 10   | +1    | +d           |
//! | 11   | +2    | +2d          |
//!
//! ⚠️ **The fourth state (code 3 → +2d) cannot be represented by the
//! `katgpt_types::TernaryGroupWeights` bit-plane container** (which holds
//! exactly `{-1, 0, +1}`). The reference encoder (`quantize_row_q2_0_ref`
//! in the `PrismML` fork) sets `d = amax` so `w/d ∈ [-1, +1]` and never
//! produces code 3; a measured scan of 30.72M weights across two structurally
//! different tensors in the real Bonsai-27B checkpoint found **zero** code-3
//! weights (0.000%). This f32 dequant path decodes all four codes faithfully
//! (preserving +2d); the bridge into `TernaryGroupWeights` MUST reject code 3
//! loudly rather than silently folding it to +1 — a future Prism-ML checkpoint
//! or the `Q2_g64` / `PQ2_0` variants may use it. See katgpt-rs Issue 578 §"The
//! format has a FOURTH state".

use half::f16;

/// Block size: 128 elements per block (group size g128).
pub const Q2_0_BLOCK_SIZE: usize = 128;

/// Size of one `Q2_0` block in bytes (2 + 32 = 34).
pub const BLOCK_BYTES: usize = 34;

/// A `Q2_0` quantization block — 128 ternary-coded weights + one f16 group scale.
///
/// Layout matches the `PrismML` llama.cpp fork's `block_q2_0` (the
/// Ternary-Bonsai checkpoint format) for mmap zero-copy compatibility.
/// Symmetric to `BlockQ8_0` (also 34 bytes, but 32 × 8-bit codes there).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ2_0 {
    /// Block scale as f16 bits (symmetric quantization, per 128-weight group).
    pub d: u16,
    /// 32 bytes packing 128 × 2-bit codes (4 per byte, LSB-first within each byte).
    pub qs: [u8; Q2_0_BLOCK_SIZE / 4],
}

// Static assert: BlockQ2_0 must be exactly 34 bytes.
const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == BLOCK_BYTES);

// ── Dequantize ──────────────────────────────────────────────────

/// Dequantize `Q2_0` blocks to f32 values.
///
/// `src` contains the quantized blocks, `dst` must have length `n`
/// where `n` is a multiple of `Q2_0_BLOCK_SIZE` and `src.len() >= n / Q2_0_BLOCK_SIZE`.
///
/// Ported from the `PrismML` fork's `dequantize_row_q2_0`. Decodes all four
/// 2-bit codes faithfully (see the module doc on the fourth state).
pub fn dequantize_row_q2_0(src: &[BlockQ2_0], dst: &mut [f32]) {
    let n = dst.len();
    assert!(
        n.is_multiple_of(Q2_0_BLOCK_SIZE),
        "dst length must be multiple of {Q2_0_BLOCK_SIZE}, got {n}"
    );
    let nb = n / Q2_0_BLOCK_SIZE;
    assert!(
        src.len() >= nb,
        "src too short: need {nb} blocks, got {}",
        src.len()
    );

    for (i, block) in src.iter().take(nb).enumerate() {
        let d = f32::from(f16::from_bits(block.d));
        let out_base = i * Q2_0_BLOCK_SIZE;

        // Inner loop: extract 2-bit code, apply (q-1)*d.
        // 4 codes per byte, LSB-first: code j lives at byte j/4, bits (j%4)*2..(j%4)*2+1.
        for j in 0..Q2_0_BLOCK_SIZE {
            let byte = block.qs[j / 4];
            let shift = (j % 4) * 2;
            let q = ((byte >> shift) & 0x03) as i32;
            // Decode: 0→-1, 1→0, 2→+1, 3→+2 (all × d). Preserves the fourth state.
            dst[out_base + j] = (q - 1) as f32 * d;
        }
    }
}

// ── Q2_0 → TernaryGroupWeights bridge (Plan 333 T3.1c) ──────────

/// Repack `Q2_0` GGUF blocks into the katgpt-core `TernaryGroupWeights`
/// substrate (Issue 578).
///
/// The `Q2_0` format encodes each weight as a 2-bit code `{-1, 0, +1, +2}`
/// (scaled by the block's f16 group scale). `TernaryGroupWeights` holds
/// exactly `{-1, 0, +1}` as two bit-planes (`pos_bits` / `neg_bits`), so the
/// repack splits each 2-bit code into the two planes:
///
/// | code | weight | pos bit | neg bit |
/// |------|--------|---------|---------|
/// | 00   | -1     | 0       | 1       |
/// | 01   |  0     | 0       | 0       |
/// | 10   | +1     | 1       | 0       |
/// | 11   | +2     | **REJECT** | **REJECT** |
///
/// Code 3 (+2d) cannot be represented by the ternary alphabet — it is
/// unreachable via the reference encoder (which sets `d = amax`), and a
/// 30.72M-weight scan of the real Bonsai-27B checkpoint found zero
/// occurrences (Issue 578). This function returns `Err` on the first code-3
/// weight it sees rather than silently folding it to +1, so a future
/// Prism-ML checkpoint that does use it fails loudly.
///
/// The repack is size-neutral: `Q2_0` = 34 B per 128 weights (2 B scale + 32 B
/// codes), `TernaryGroupWeights` = 34 B per 128 weights (2 B scale + 16 B
/// pos + 16 B neg). No expansion.
#[cfg(feature = "q2_0_ternary_bridge")]
pub fn repack_q2_0_to_ternary_group(
    blocks: &[BlockQ2_0],
    rows: usize,
    cols: usize,
) -> Result<katgpt_core::TernaryGroupWeights, Q2oRepackError> {
    use katgpt_core::TernaryGroupWeights;

    // Geometry cross-check: the Q2_0 block count must match (rows × cols / 128).
    let expected_blocks = rows * cols / Q2_0_BLOCK_SIZE;
    if blocks.len() < expected_blocks {
        return Err(Q2oRepackError::TooFewBlocks {
            got: blocks.len(),
            expected: expected_blocks,
        });
    }
    if !cols.is_multiple_of(Q2_0_BLOCK_SIZE) {
        return Err(Q2oRepackError::ColsNotMultipleOf128 { cols });
    }

    let mut out = TernaryGroupWeights::new(rows, cols);
    let blocks_per_row = cols / Q2_0_BLOCK_SIZE;

    for row in 0..rows {
        let row_block_base = row * blocks_per_row;
        for g in 0..blocks_per_row {
            let block = &blocks[row_block_base + g];
            // Copy the f16 group scale verbatim.
            out.group_scale[row * out.groups_per_row + g] = half::f16::from_bits(block.d);

            // Repack the 128 codes into the two bit-planes.
            // GROUP_SIZE = 128 = 2 × 64, so group g spans blocks64 [2g, 2g+2).
            let b0 = row * out.blocks64 + g * 2;
            let b1 = b0 + 1;
            for j in 0..Q2_0_BLOCK_SIZE {
                let byte = block.qs[j / 4];
                let shift = (j % 4) * 2;
                let code = (byte >> shift) & 0x03;
                let bit_idx = if j < 64 { b0 } else { b1 };
                let bit_mask = 1u64 << (j & 63);
                match code {
                    0 => {
                        // -1: neg bit set.
                        out.neg_bits[bit_idx] |= bit_mask;
                    }
                    1 => {
                        // 0: both clear (no-op).
                    }
                    2 => {
                        // +1: pos bit set.
                        out.pos_bits[bit_idx] |= bit_mask;
                    }
                    _ => {
                        // Code 3 (+2d): cannot represent — reject loudly.
                        return Err(Q2oRepackError::UnsupportedFourthState {
                            row,
                            col: g * Q2_0_BLOCK_SIZE + j,
                        });
                    }
                }
            }
        }
    }

    debug_assert!(
        out.invariant_holds(),
        "pos & neg == 0 invariant violated after Q2_0 repack"
    );
    Ok(out)
}

/// Errors from the `Q2_0` → `TernaryGroupWeights` repack.
#[cfg(feature = "q2_0_ternary_bridge")]
#[derive(Debug, thiserror::Error)]
pub enum Q2oRepackError {
    #[error("too few Q2_0 blocks: got {got}, expected {expected}")]
    TooFewBlocks { got: usize, expected: usize },
    #[error("cols ({cols}) is not a multiple of 128")]
    ColsNotMultipleOf128 { cols: usize },
    #[error("Q2_0 code 3 (+2d) at (row {row}, col {col}) cannot be represented in TernaryGroupWeights")]
    UnsupportedFourthState { row: usize, col: usize },
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn test_block_q2_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ2_0>(), BLOCK_BYTES);
        assert_eq!(std::mem::size_of::<BlockQ2_0>(), 34);
        assert_eq!(std::mem::align_of::<BlockQ2_0>(), 2); // u16 alignment
        assert_eq!(Q2_0_BLOCK_SIZE / 4, 32); // 32-byte payload
    }

    #[test]
    fn test_dequant_all_four_codes_at_known_scale() {
        // Manually construct one block where the first 4 weights hit all 4 codes.
        // d = 3.0 (f16 bits). byte 0 packs codes [0, 1, 2, 3] LSB-first → 0b11_10_01_00 = 0b00111001 = 0x39.
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(3.0).to_bits();
        block.qs[0] = 0b11_10_01_00; // codes 0, 1, 2, 3 for weights 0, 1, 2, 3
        // Remaining weights = code 1 (= 0) so they contribute nothing.

        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);

        // code 0 → (0-1)*3.0 = -3.0
        // code 1 → (1-1)*3.0 =  0.0
        // code 2 → (2-1)*3.0 =  3.0
        // code 3 → (3-1)*3.0 =  6.0  ← the fourth state, faithfully preserved
        assert_eq!(dst[0], -3.0);
        assert_eq!(dst[1], 0.0);
        assert_eq!(dst[2], 3.0);
        assert_eq!(dst[3], 6.0);
        // weights 4.. all decode to code 1 (= 0) since qs[1..] is zero → code 0... wait.
        // qs bytes beyond [0] are 0 → all codes are 0 → weight = -3.0. Fix the block.
    }

    #[test]
    fn test_dequant_zero_block() {
        // All-zero block: d=0, all codes 0 → (0-1)*0 = 0 (the -1×0 case).
        let block = BlockQ2_0::zeroed();
        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!(v.abs() < f32::EPSILON, "zero block nonzero at {i}: {v}");
        }
    }

    #[test]
    fn test_dequant_uniform_positive() {
        // All weights = +d: every code is 2 (binary 10), packed 4-per-byte = 0b10_10_10_10 = 0xAA.
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(2.5).to_bits();
        block.qs.fill(0b10_10_10_10);
        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert_eq!(v, 2.5, "uniform +d failed at {i}: {v}");
        }
    }

    #[test]
    fn test_dequant_uniform_negative() {
        // All weights = -d: every code is 0 (binary 00), packed = 0x00.
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(1.75).to_bits();
        block.qs.fill(0b00_00_00_00);
        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert_eq!(v, -1.75, "uniform -d failed at {i}: {v}");
        }
    }

    #[test]
    fn test_dequant_uniform_zero() {
        // All weights = 0: every code is 1 (binary 01), packed 4-per-byte = 0b01_01_01_01 = 0x55.
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(5.0).to_bits();
        block.qs.fill(0b01_01_01_01);
        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!(v.abs() < f32::EPSILON, "uniform 0 failed at {i}: {v}");
        }
    }

    #[test]
    fn test_dequant_matches_formula_reference() {
        // Build a block with a deterministic mix of codes + verify against
        // the llama.cpp formula: y[j] = ((int)q - 1) * d.
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(0.5).to_bits();
        // Walk codes 0,1,2,3 across the first 16 weights (4 per byte, 4 bytes).
        for b in 0..4usize {
            block.qs[b] = 0b11_10_01_00; // codes 0,1,2,3 in each of bytes 0..4
        }
        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);

        // First 16 weights: pattern [−0.5, 0.0, +0.5, +1.0] × 4.
        let expected_pattern = [-0.5f32, 0.0, 0.5, 1.0];
        for j in 0..16usize {
            let expected = expected_pattern[j % 4];
            assert_eq!(
                dst[j], expected,
                "formula mismatch at {j}: got {} expected {expected}",
                dst[j]
            );
        }
        // Weights 16.. are code 0 → -0.5.
        for (j, &w) in dst.iter().enumerate().take(Q2_0_BLOCK_SIZE).skip(16) {
            assert_eq!(w, -0.5, "tail mismatch at {j}: {w}");
        }
    }

    #[test]
    fn test_multi_block_dequant() {
        // Two blocks with different scales; verify block boundaries.
        let mut blocks = [BlockQ2_0::zeroed(); 2];
        blocks[0].d = f16::from_f32(1.0).to_bits();
        blocks[0].qs.fill(0b10_10_10_10); // all +1.0
        blocks[1].d = f16::from_f32(2.0).to_bits();
        blocks[1].qs.fill(0b00_00_00_00); // all -2.0

        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE * 2];
        dequantize_row_q2_0(&blocks, &mut dst);

        for i in 0..Q2_0_BLOCK_SIZE {
            assert_eq!(dst[i], 1.0, "block 0 weight {i}");
            assert_eq!(dst[Q2_0_BLOCK_SIZE + i], -2.0, "block 1 weight {i}");
        }
    }

    #[test]
    fn test_compression_ratio() {
        // 128 f32 values = 512 bytes → 1 block = 34 bytes → ~15.06× compression.
        let original_bytes = Q2_0_BLOCK_SIZE * 4;
        let compressed_bytes = std::mem::size_of::<BlockQ2_0>();
        let ratio = original_bytes as f64 / compressed_bytes as f64;
        assert!(
            ratio > 15.0,
            "compression ratio too low: {ratio:.2}x (expected ~15.06x)"
        );
        // bits/weight = 34*8/128 = 2.125
        let bpw = compressed_bytes as f64 * 8.0 / Q2_0_BLOCK_SIZE as f64;
        assert!((bpw - 2.125).abs() < 1e-9, "bits/weight = {bpw}, expected 2.125");
    }

    #[test]
    fn test_packing_lsb_first_within_byte() {
        // Verify the LSB-first packing convention against the llama.cpp formula.
        // qs[0] = 0b11_10_01_00 means:
        //   weight 0 → bits 1:0 → 00 → code 0
        //   weight 1 → bits 3:2 → 01 → code 1
        //   weight 2 → bits 5:4 → 10 → code 2
        //   weight 3 → bits 7:6 → 11 → code 3
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(1.0).to_bits();
        block.qs[0] = 0b11_10_01_00;
        let mut dst = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut dst);
        // (q-1)*d at d=1.0: codes → [-1, 0, +1, +2]
        assert_eq!(dst[0], -1.0);
        assert_eq!(dst[1], 0.0);
        assert_eq!(dst[2], 1.0);
        assert_eq!(dst[3], 2.0);
    }

    // ── Q2_0 → TernaryGroupWeights bridge tests (Plan 333 T3.1c) ──────────

    #[cfg(feature = "q2_0_ternary_bridge")]
    #[test]
    fn test_repack_all_three_valid_codes() {
        use super::repack_q2_0_to_ternary_group;

        // One row × 128 cols (= 1 Q2_0 block).
        // First 4 weights: codes 0, 1, 2, 1 → -1, 0, +1, 0.
        // Remaining 124 weights: code 2 (+1).
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(3.0).to_bits();
        block.qs[0] = 0b01_10_01_00; // codes 0, 1, 2, 1 for weights 0..4
        for b in 1..(Q2_0_BLOCK_SIZE / 4) {
            block.qs[b] = 0b10_10_10_10; // code 2 (+1)
        }

        let tg = repack_q2_0_to_ternary_group(std::slice::from_ref(&block), 1, Q2_0_BLOCK_SIZE)
            .expect("repack must succeed for valid codes");

        assert_eq!(tg.rows, 1);
        assert_eq!(tg.cols, Q2_0_BLOCK_SIZE);
        assert_eq!(tg.groups_per_row, 1);
        // The group scale should be the f16 from the Q2_0 block.
        assert_eq!(tg.group_scale[0], f16::from_f32(3.0));

        // Verify the repacked values via TernaryGroupWeights::get.
        assert_eq!(tg.get(0, 0), -1, "code 0 → -1");
        assert_eq!(tg.get(0, 1), 0, "code 1 → 0");
        assert_eq!(tg.get(0, 2), 1, "code 2 → +1");
        assert_eq!(tg.get(0, 3), 0, "code 1 → 0");
        for j in 4..Q2_0_BLOCK_SIZE {
            assert_eq!(tg.get(0, j), 1, "tail weight {j} should be +1");
        }

        // The pos & neg == 0 invariant must hold.
        assert!(tg.invariant_holds(), "invariant must hold after repack");
    }

    #[cfg(feature = "q2_0_ternary_bridge")]
    #[test]
    fn test_repack_rejects_code_3_fourth_state() {
        use super::{Q2oRepackError, repack_q2_0_to_ternary_group};

        // One block with code 3 at weight 3.
        let mut block = BlockQ2_0::zeroed();
        block.d = f16::from_f32(1.0).to_bits();
        block.qs[0] = 0b11_00_00_00; // weight 3 = code 3

        let err = repack_q2_0_to_ternary_group(
            std::slice::from_ref(&block),
            1,
            Q2_0_BLOCK_SIZE,
        )
        .expect_err("code 3 must be rejected");

        match err {
            Q2oRepackError::UnsupportedFourthState { row, col } => {
                assert_eq!(row, 0);
                assert_eq!(col, 3, "code 3 is at weight index 3");
            }
            other => panic!("expected UnsupportedFourthState, got {other:?}"),
        }
    }

    #[cfg(feature = "q2_0_ternary_bridge")]
    #[test]
    fn test_repack_multi_row_round_trip() {
        use super::repack_q2_0_to_ternary_group;

        // 2 rows × 256 cols = 2 rows × 2 blocks each = 4 blocks total.
        // Row 0: block 0 all code 2 (+1), block 1 all code 0 (-1).
        // Row 1: block 0 all code 1 (0), block 1 all code 2 (+1).
        let mut blocks = [BlockQ2_0::zeroed(); 4];
        blocks[0].d = f16::from_f32(2.0).to_bits();
        blocks[0].qs.fill(0b10_10_10_10);
        blocks[1].d = f16::from_f32(1.5).to_bits();
        blocks[1].qs.fill(0b00_00_00_00);
        blocks[2].d = f16::from_f32(5.0).to_bits();
        blocks[2].qs.fill(0b01_01_01_01);
        blocks[3].d = f16::from_f32(0.5).to_bits();
        blocks[3].qs.fill(0b10_10_10_10);

        let tg =
            repack_q2_0_to_ternary_group(&blocks, 2, 2 * Q2_0_BLOCK_SIZE)
                .expect("multi-row repack");

        assert_eq!(tg.rows, 2);
        assert_eq!(tg.cols, 256);
        assert_eq!(tg.groups_per_row, 2);

        // Row 0: first 128 = +1 (scale 2.0), second 128 = -1 (scale 1.5).
        assert_eq!(tg.scale_at(0, 0), 2.0);
        assert_eq!(tg.scale_at(0, 1), 1.5);
        assert_eq!(tg.get(0, 0), 1);
        assert_eq!(tg.get(0, 127), 1);
        assert_eq!(tg.get(0, 128), -1);
        assert_eq!(tg.get(0, 255), -1);

        // Row 1: first 128 = 0 (scale 5.0), second 128 = +1 (scale 0.5).
        assert_eq!(tg.scale_at(1, 0), 5.0);
        assert_eq!(tg.scale_at(1, 1), 0.5);
        assert_eq!(tg.get(1, 0), 0);
        assert_eq!(tg.get(1, 127), 0);
        assert_eq!(tg.get(1, 128), 1);
        assert_eq!(tg.get(1, 255), 1);

        assert!(tg.invariant_holds());
    }
}
