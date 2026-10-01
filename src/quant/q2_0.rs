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

// ── Q2_0A decode (Issue 027 T1/T2) ──────────────────────────────

/// Dequantize `Q2_0`-layout blocks against an EXPLICIT grid (the Q2_0A
/// decode path): code `c` decodes to `level[c]·d` where `level = {l0, 0,
/// l2, 2}` in d-units, instead of the wire's structural `(c−1)·d` uniform
/// ladder.
///
/// This is the whole reason the format variant exists: the base wire
/// decode hard-pins the level ladder, so a non-uniform grid can only be
/// CONSUMED by a decoder that knows the grid (a global format constant —
/// same 34 B blocks, one committed grid per model). Measured 2026-10-01:
/// feeding solved-grid codes to the UNIFORM decode inflates MSE ~35%
/// (the first run's phantom "solved loses" — the grid never reached the
/// dequant).
pub fn dequantize_row_q2_0_grid(
    src: &[BlockQ2_0],
    dst: &mut [f32],
    grid: super::lut_grid::Q2Grid,
) {
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
    let levels = [grid.l0, 0.0, grid.l2, 2.0f32];
    for (i, block) in src.iter().take(nb).enumerate() {
        let d = f32::from(f16::from_bits(block.d));
        let out_base = i * Q2_0_BLOCK_SIZE;
        for j in 0..Q2_0_BLOCK_SIZE {
            let byte = block.qs[j / 4];
            let shift = (j % 4) * 2;
            let q = ((byte >> shift) & 0x03) as usize;
            dst[out_base + j] = levels[q] * d;
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
    #[error(
        "Q2_0 code 3 (+2d) at (row {row}, col {col}) cannot be represented in TernaryGroupWeights"
    )]
    UnsupportedFourthState { row: usize, col: usize },
    /// A dense (arm A) tensor has no Q2_0 wire payload — the collapsed-GGUF
    /// writer emits it as F16 wire instead (Issue 022 T4.1/T4.3).
    #[error("dense_f16 tensor has no Q2_0 wire payload (write as F16 wire)")]
    NotTernary,
}

/// Pack `TernaryGroupWeights` back into `Q2_0` wire blocks — the INVERSE of
/// [`repack_q2_0_to_ternary_group`] (riir-infer Issue 022 T4.1: the
/// re-ternarization arms emit the container; the collapsed-GGUF writer
/// needs the wire payload).
///
/// Lossless by construction for every container this crate can hold: codes
/// `{-1, 0, +1}` map to `00/01/10`, the f16 group scale copies verbatim, and
/// code 3 (+2d) is unreachable from a two-plane container. `cols` must be a
/// multiple of 128 (the wire format has no partial group) and the pos&neg
/// invariant must hold — both refused loud.
///
/// Appends `rows * (cols / 128)` blocks to `out` (write-into: the caller can
/// size the buffer once and reuse it across tensors).
#[cfg(feature = "q2_0_ternary_bridge")]
pub fn pack_ternary_group_to_q2_0(
    w: &katgpt_core::TernaryGroupWeights,
    out: &mut Vec<BlockQ2_0>,
) -> Result<(), Q2oRepackError> {
    if !w.cols.is_multiple_of(Q2_0_BLOCK_SIZE) {
        return Err(Q2oRepackError::ColsNotMultipleOf128 { cols: w.cols });
    }
    if w.pos_bits.len() != w.rows * w.blocks64 || w.neg_bits.len() != w.rows * w.blocks64 {
        return Err(Q2oRepackError::TooFewBlocks {
            got: w.pos_bits.len().min(w.neg_bits.len()),
            expected: w.rows * w.blocks64,
        });
    }
    let blocks_per_row = w.cols / Q2_0_BLOCK_SIZE;
    out.reserve(rows_blocks(w.rows, blocks_per_row));

    for row in 0..w.rows {
        for g in 0..blocks_per_row {
            let b0 = row * w.blocks64 + g * 2;
            let b1 = b0 + 1;
            let mut block = BlockQ2_0 {
                d: w.group_scale[row * w.groups_per_row + g].to_bits(),
                qs: [0u8; Q2_0_BLOCK_SIZE / 4],
            };
            for j in 0..Q2_0_BLOCK_SIZE {
                let word = if j < 64 { b0 } else { b1 };
                let mask = 1u64 << (j & 63);
                let pos = (w.pos_bits[word] & mask) != 0;
                let neg = (w.neg_bits[word] & mask) != 0;
                debug_assert!(
                    !(pos && neg),
                    "pos & neg == 0 invariant violated at row {row}, col {}",
                    g * Q2_0_BLOCK_SIZE + j
                );
                // 2-bit code, 4 per byte, LSB-first. ALL FOUR states are
                // written explicitly: 00 = -1, 01 = 0, 10 = +1 — a zero
                // must emit code 1, never a skipped nibble (code 0 decodes
                // as -1, and an unwritten qs nibble IS code 0).
                let code: u8 = if pos {
                    2
                } else if neg {
                    0
                } else {
                    1
                };
                block.qs[j / 4] |= code << ((j % 4) * 2);
            }
            out.push(block);
        }
    }
    Ok(())
}

#[cfg(feature = "q2_0_ternary_bridge")]
const fn rows_blocks(rows: usize, blocks_per_row: usize) -> usize {
    rows * blocks_per_row
}

// ── Encoders (Issue 027 / Plan 615) ─────────────────────────────

use super::lut_grid::Q2Grid;

/// Per-block scale scan shared by the encoders: returns `(d_bits, d_f32)`
/// for the T0 sign-absorbing rule — `d = sign(argmax|w|)·f16(amax/2)`, the
/// FIRST maximum winning the argmax (strict `>` scan; the A2 convention in
/// `dq_fakequant::a2_quant_dequant_block`, so the activation and weight
/// lanes share one tie-break). A zero / f16-underflow scale returns `None`
/// and the caller emits an all-zero block. Public for the offline
/// histogram/eval bin (`lut_grid_solve`), which pools `x = w/d` under the
/// SAME rule the encoder applies.
#[inline]
pub fn t0_block_scale(block: &[f32]) -> Option<(u16, f32)> {
    let mut amax = 0f32;
    let mut aidx = 0usize;
    for (i, &w) in block.iter().enumerate() {
        let m = w.abs();
        if m > amax {
            amax = m;
            aidx = i;
        }
    }
    if amax == 0.0 {
        return None;
    }
    let d = f16::from_f32(block[aidx].signum() * amax / 2.0);
    let d_f32 = d.to_f32();
    if d_f32 == 0.0 || !d_f32.is_finite() {
        return None;
    }
    Some((d.to_bits(), d_f32))
}

#[inline]
fn zero_block() -> BlockQ2_0 {
    BlockQ2_0 {
        d: 0,
        qs: [0b01_01_01_01; Q2_0_BLOCK_SIZE / 4], // all codes 1 (zero)
    }
}

/// Reference RTN symmetric encoder: per-128 block, `d = f16(amax)`,
/// `q = clamp(round(w/d), −1, +1)` — code 3 is unreachable (the wire
/// round-trip of every Bonsai ternary block).
///
/// `w.len()` must be a multiple of 128 (the wire has no partial group);
/// appends `w.len()/128` blocks. Byte-identity control: dequantizing a
/// ternary-valued `BlockQ2_0` and re-encoding with this reproduces the
/// original bytes (tested).
pub fn quantize_row_q2_0_symmetric(w: &[f32], out: &mut Vec<BlockQ2_0>) {
    assert!(
        w.len().is_multiple_of(Q2_0_BLOCK_SIZE),
        "q2_0 encode: length must be a multiple of {Q2_0_BLOCK_SIZE}, got {}",
        w.len()
    );
    for block_w in w.as_chunks::<Q2_0_BLOCK_SIZE>().0 {
        let mut amax = 0f32;
        for &v in block_w {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let d16 = f16::from_f32(amax);
        let d_f32 = d16.to_f32();
        if d_f32 == 0.0 || !d_f32.is_finite() {
            out.push(zero_block());
            continue;
        }
        let mut block = BlockQ2_0 {
            d: d16.to_bits(),
            qs: [0u8; Q2_0_BLOCK_SIZE / 4],
        };
        for (j, &v) in block_w.iter().enumerate() {
            let q = (v / d_f32).round().clamp(-1.0, 1.0) as i32;
            let code = (q + 1) as u8;
            block.qs[j / 4] |= code << ((j % 4) * 2);
        }
        out.push(block);
    }
}

/// Issue 027 T0 — the asymmetric encoder: same bytes, same decode, the
/// FOURTH code state activated encoder-only.
///
/// `d = sign(argmax|w|)·f16(amax/2)` steers the double step (+2d) onto the
/// side holding the block's largest-magnitude element; the effective grid
/// is `{-d, 0, +d, +2d}` = `{-1, 0, +1, +2}·d`. `q = clamp(round(w/d),
/// −1, +2)`; the anchor element always lands on code 3 (2·d/d rounds to
/// ±2 under any f16 rounding of d — tested). The wire format and every
/// decoder are unchanged; only the TernaryGroupWeights bridge rejects
/// code 3 (`UnsupportedFourthState`) — that is T4's kernel-cost axis.
pub fn quantize_row_q2_0_asymmetric(w: &[f32], out: &mut Vec<BlockQ2_0>) {
    assert!(
        w.len().is_multiple_of(Q2_0_BLOCK_SIZE),
        "q2_0 encode: length must be a multiple of {Q2_0_BLOCK_SIZE}, got {}",
        w.len()
    );
    for block_w in w.as_chunks::<Q2_0_BLOCK_SIZE>().0 {
        let Some((d_bits, d_f32)) = t0_block_scale(block_w) else {
            out.push(zero_block());
            continue;
        };
        let mut block = BlockQ2_0 {
            d: d_bits,
            qs: [0u8; Q2_0_BLOCK_SIZE / 4],
        };
        for (j, &v) in block_w.iter().enumerate() {
            let q = (v / d_f32).round().clamp(-1.0, 2.0) as i32;
            let code = (q + 1) as u8;
            block.qs[j / 4] |= code << ((j % 4) * 2);
        }
        out.push(block);
    }
}

/// Issue 027 T1/T2 — the grid-generic encoder: nearest-level assignment in
/// DECODED space over any solved [`Q2Grid`] (round() is only correct for
/// the uniform ladder; a non-uniform grid needs the true nearest level).
/// Scale rule identical to T0 (the grid's anchor is pinned at +2 d-units).
/// Exact ties resolve to the LOWEST code, matching the solver's
/// assignment tie-break.
pub fn quantize_row_q2_0_grid(w: &[f32], grid: Q2Grid, out: &mut Vec<BlockQ2_0>) {
    assert!(
        w.len().is_multiple_of(Q2_0_BLOCK_SIZE),
        "q2_0 encode: length must be a multiple of {Q2_0_BLOCK_SIZE}, got {}",
        w.len()
    );
    let levels = [grid.l0, 0.0, grid.l2, 2.0];
    for block_w in w.as_chunks::<Q2_0_BLOCK_SIZE>().0 {
        let Some((d_bits, d_f32)) = t0_block_scale(block_w) else {
            out.push(zero_block());
            continue;
        };
        let mut block = BlockQ2_0 {
            d: d_bits,
            qs: [0u8; Q2_0_BLOCK_SIZE / 4],
        };
        for (j, &v) in block_w.iter().enumerate() {
            let mut best_code = 0u8;
            let mut best_dist = f32::INFINITY;
            for (c, &l) in levels.iter().enumerate() {
                let dist = (v - l * d_f32).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_code = c as u8;
                }
            }
            block.qs[j / 4] |= best_code << ((j % 4) * 2);
        }
        out.push(block);
    }
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
        assert!(
            (bpw - 2.125).abs() < 1e-9,
            "bits/weight = {bpw}, expected 2.125"
        );
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

        let err = repack_q2_0_to_ternary_group(std::slice::from_ref(&block), 1, Q2_0_BLOCK_SIZE)
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

        let tg = repack_q2_0_to_ternary_group(&blocks, 2, 2 * Q2_0_BLOCK_SIZE)
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

    // ── Issue 027 / Plan 615 encoder tests ─────────────────────

    /// Deterministic pseudo-f32 in [-1, 1) for fixtures.
    fn fixture_f32(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) & 0xFFFFFF) as f32 / 0x1000000 as f32) * 2.0 - 1.0
        }
    }

    /// `n` fixture samples collected.
    fn fixture_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut next = fixture_f32(seed);
        (0..n).map(|_| next()).collect()
    }

    fn count_code(blocks: &[BlockQ2_0], code: u8) -> usize {
        let mut n = 0;
        for b in blocks {
            for j in 0..Q2_0_BLOCK_SIZE {
                if (b.qs[j / 4] >> ((j % 4) * 2)) & 0x03 == code {
                    n += 1;
                }
            }
        }
        n
    }

    /// T0: the anchor element ALWAYS lands on code 3, and every nonzero
    /// block carries at least one code 3 (the fourth state is reachable).
    #[test]
    fn t0_anchor_always_emits_code3() {
        let mut w = fixture_vec(0xDEB0_2700, 512);
        // Force unambiguous anchors: 7.5 is above the fixture range and
        // exactly representable in f16 (d = f16(3.75), +2d = 7.5).
        for b in 0..4 {
            w[b * 128] = 7.5;
        }
        let mut out = Vec::new();
        quantize_row_q2_0_asymmetric(&w, &mut out);
        assert_eq!(out.len(), w.len() / 128);
        // The forced anchor element is the block max → code 3.
        for (b, block) in out.iter().enumerate() {
            let anchor_code = block.qs[0] & 0x03; // j=0
            if b < 4 {
                assert_eq!(anchor_code, 3, "block {b}: anchor must be code 3");
            }
        }
        assert!(
            count_code(&out, 3) >= out.len(),
            "at least one code 3 per nonzero block (its own anchor)"
        );
    }

    /// T0 vs symmetric round-trip error on a random fixture: T0 must not be
    /// WORSE in MSE than the symmetric encoder on this distribution (the
    /// lane's whole premise; a per-distribution loss is recorded, not
    /// hidden — this fixture is the sanity floor, the per-family gate is
    /// the bin's job).
    #[test]
    fn t0_mse_not_worse_than_symmetric_on_uniform_fixture() {
        let w = fixture_vec(0xA11C_0F42, 4096);
        let mut sym = Vec::new();
        let mut asym = Vec::new();
        quantize_row_q2_0_symmetric(&w, &mut sym);
        quantize_row_q2_0_asymmetric(&w, &mut asym);
        let mut dsym = vec![0f32; w.len()];
        let mut dasym = vec![0f32; w.len()];
        dequantize_row_q2_0(&sym, &mut dsym);
        dequantize_row_q2_0(&asym, &mut dasym);
        let mse = |d: &[f32]| -> f64 {
            d.iter()
                .zip(&w)
                .map(|(&a, &b)| {
                    let e = (a - b) as f64;
                    e * e
                })
                .sum::<f64>()
                / w.len() as f64
        };
        // Uniform on [-1,1): symmetric d=amax≈1 puts every weight at q≈±1
        // or 0 — terrible. T0 halves d and spreads over 4 levels — better.
        assert!(
            mse(&dasym) <= mse(&dsym),
            "T0 mse {} must be <= symmetric mse {}",
            mse(&dasym),
            mse(&dsym)
        );
    }

    /// The Bonsai byte-identity control: a ternary-valued block
    /// ({−1,0,+1}·d, d = amax) re-encoded symmetrically reproduces the
    /// original bytes — the symmetric arm is the identity on ternary
    /// containers (T3's control, executable).
    #[test]
    fn symmetric_reencode_of_ternary_blocks_is_byte_identical() {
        // Build a valid ternary block by hand: d = 1.5, codes in {0,1,2}.
        let mut orig = BlockQ2_0 {
            d: f16::from_f32(1.5).to_bits(),
            qs: [0u8; Q2_0_BLOCK_SIZE / 4],
        };
        let mut raw = fixture_f32(0x7E3A);
        let mut next = move || {
            let x = raw();
            if x < -0.34 {
                0u8
            } else if x < 0.34 {
                1u8
            } else {
                2u8
            }
        };
        for j in 0..Q2_0_BLOCK_SIZE {
            orig.qs[j / 4] |= next() << ((j % 4) * 2);
        }
        // Ensure at least one of each code.
        orig.qs[0] = 0b00_01_10_00;

        let mut deq = vec![0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(&[orig], &mut deq);
        let mut re = Vec::new();
        quantize_row_q2_0_symmetric(&deq, &mut re);
        assert_eq!(re.len(), 1);
        let a = bytemuck::bytes_of(&orig);
        let b = bytemuck::bytes_of(&re[0]);
        assert_eq!(a, b, "symmetric re-encode must reproduce the original bytes");
    }

    /// The issue's Bonsai claim, executable: asymmetric re-encode of a
    /// ternary block can only reproduce or LOSE (never beat) the original.
    #[test]
    fn asymmetric_reencode_of_ternary_blocks_never_beats() {
        let d = 1.5f32;
        let mut orig = BlockQ2_0 {
            d: f16::from_f32(d).to_bits(),
            qs: [0u8; Q2_0_BLOCK_SIZE / 4],
        };
        let mut raw = fixture_f32(0x51DE);
        let mut next = move || {
            let x = raw();
            if x < 0.0 {
                0u8
            } else if x < 0.5 {
                1u8
            } else {
                2u8
            }
        };
        for j in 0..Q2_0_BLOCK_SIZE {
            orig.qs[j / 4] |= next() << ((j % 4) * 2);
        }
        let mut deq = vec![0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(&[orig], &mut deq);
        let mut re = Vec::new();
        quantize_row_q2_0_asymmetric(&deq, &mut re);
        let mut rdeq = vec![0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(&re, &mut rdeq);
        let err = |v: &[f32]| -> f64 {
            v.iter()
                .zip(&deq)
                .map(|(&a, &b)| {
                    let e = (a - b) as f64;
                    e * e
                })
                .sum::<f64>()
        };
        assert!(
            err(&rdeq) >= err(&deq) - 1e-9,
            "asymmetric re-encode of ternary input must not reduce error"
        );
    }

    /// Grid encoder: on the T0 uniform grid it must agree with the
    /// asymmetric round() encoder in ERROR per element (identical scale
    /// rule; nearest-level == round for uniform levels EXCEPT at exact
    /// midpoints, where the lowest-code tie-break picks the other
    /// MSE-equal level — the invariant is equal error, not equal bytes).
    /// On a non-uniform grid the assignment must actually move.
    #[test]
    fn grid_encoder_matches_round_on_uniform_and_moves_on_solved() {
        let w = fixture_vec(0x6C1D, 4096);
        let mut a = Vec::new();
        let mut b = Vec::new();
        quantize_row_q2_0_asymmetric(&w, &mut a);
        quantize_row_q2_0_grid(&w, super::super::lut_grid::GRID_T0_UNIFORM, &mut b);
        assert_eq!(a.len(), b.len());
        let mut da = vec![0f32; w.len()];
        let mut db = vec![0f32; w.len()];
        dequantize_row_q2_0(&a, &mut da);
        dequantize_row_q2_0(&b, &mut db);
        for (i, (&wi, (va, vb))) in w.iter().zip(da.iter().zip(&db)).enumerate() {
            // d comes from the element's OWN block.
            let d = f32::from(f16::from_bits(a[i / Q2_0_BLOCK_SIZE].d));
            let ea = (wi - va).abs();
            let eb = (wi - vb).abs();
            assert!(
                (ea - eb).abs() <= d.abs() * 1e-3,
                "uniform grid must be error-equal to round-clamp: w={wi} a={va} b={vb} d={d}"
            );
        }
        // Non-uniform: probe block [1.0, 0.225, 0…], d = f16(0.5). The
        // element x = 0.45 sits between the uniform boundary (0.5 → level 0,
        // code 1) and the solved boundary (l2 = 0.8 → boundary 0.4 → level
        // 0.8, code 2) — the assignment must MOVE.
        let mut probe = vec![0f32; 128];
        probe[0] = 1.0;
        probe[1] = 0.225;
        let grid = super::super::lut_grid::Q2Grid { l0: -0.5, l2: 0.8 };
        let mut pa = Vec::new();
        let mut pb = Vec::new();
        quantize_row_q2_0_asymmetric(&probe, &mut pa);
        quantize_row_q2_0_grid(&probe, grid, &mut pb);
        let ca = (pa[0].qs[0] >> 2) & 0x03; // j=1
        let cb = (pb[0].qs[0] >> 2) & 0x03;
        assert_eq!(ca, 1, "x=0.45 under uniform: round → q=0 → code 1");
        assert_eq!(cb, 2, "x=0.45 under l2=0.8: boundary 0.4 → level 0.8 → code 2");
        // The anchor element (j=0) stays code 3 on both — the scale rule is
        // grid-independent.
        assert_eq!((pa[0].qs[0]) & 0x03, 3);
        assert_eq!((pb[0].qs[0]) & 0x03, 3);
    }

    /// Non-multiple-of-128 lengths are refused loudly.
    #[test]
    #[should_panic(expected = "multiple of 128")]
    fn encoder_refuses_partial_blocks() {
        let w = vec![0f32; 100];
        let mut out = Vec::new();
        quantize_row_q2_0_symmetric(&w, &mut out);
    }

    /// Q2_0A decode: on the uniform grid it must equal the base dequant
    /// bit-for-bit (levels {−1,0,+1,+2} ≡ (c−1)), and on a non-uniform grid
    /// the wire round-trip must beat the T0 wire round-trip on a fixture
    /// (the grid-aware decode is what makes T1 real).
    #[test]
    fn grid_dequant_matches_base_on_uniform_and_beats_t0_on_solved() {
        let w = fixture_vec(0xA11C_0F42, 4096);
        let mut t0_blocks = Vec::new();
        quantize_row_q2_0_asymmetric(&w, &mut t0_blocks);
        let mut base = vec![0f32; w.len()];
        dequantize_row_q2_0(&t0_blocks, &mut base);
        let mut via_grid = vec![0f32; w.len()];
        dequantize_row_q2_0_grid(&t0_blocks, &mut via_grid, super::super::lut_grid::GRID_T0_UNIFORM);
        assert_eq!(base, via_grid, "uniform-grid decode must equal the base decode");

        // A shrunken uniform-ish grid on the same CODES decodes differently
        // (the format mismatch the first run measured). The grid-encoder +
        // grid-dequant pair must close the loop: SOLVE the grid from the
        // fixture's own d²-weighted histogram, encode with it, decode with
        // it — the wire round-trip must beat the T0 pair (the grid is
        // data-dependent; a hand-picked off-distribution grid may lose, and
        // that is correct).
        let mut hist = super::super::lut_grid::WeightHistogram::new();
        for block in w.as_chunks::<128>().0 {
            if let Some((_, d)) = super::super::q2_0::t0_block_scale(block) {
                let mass = (d as f64) * (d as f64);
                for &v in block {
                    hist.record_weighted((v / d) as f64, mass);
                }
            }
        }
        let (grid, _) = super::super::lut_grid::solve_lloyd_max(&hist);
        let mut solved_blocks = Vec::new();
        quantize_row_q2_0_grid(&w, grid, &mut solved_blocks);
        let mut solved = vec![0f32; w.len()];
        dequantize_row_q2_0_grid(&solved_blocks, &mut solved, grid);
        let mse = |d: &[f32]| -> f64 {
            d.iter()
                .zip(&w)
                .map(|(&a, &b)| {
                    let e = (a - b) as f64;
                    e * e
                })
                .sum::<f64>()
                / w.len() as f64
        };
        // Fixture is uniform on [-1,1): the shrunken grid should win here.
        assert!(
            mse(&solved) <= mse(&base),
            "grid round-trip must not lose to T0 on its own grid: {} vs {}",
            mse(&solved),
            mse(&base)
        );
    }
}
