//! CubeCL Q4_K fused dequant+GEMV kernel for decode (Plan 106 T2.7).
//!
//! Performs `output[M] = dequant_q4k(weight) @ input[N]` with inline
//! Q4_K dequantization during the dot product — no intermediate f32
//! weight materialization.
//!
//! # Weight Buffers
//!
//! Two GPU buffers per projection:
//!
//! | Buffer | Type | Layout per block | Purpose |
//! |--------|------|-----------------|---------|
//! | `weight_q4k` | `Array<u32>` | 36 u32 | Scales + qs (zero-copy from GGUF) |
//! | `d_dmin` | `Array<f32>` | 2 f32 | Pre-decoded d and dmin (CPU f16→f32) |
//!
//! `weight_q4k` layout per block (matches WGSL `gemv_q4k.wgsl`):
//! - Word 0: d|dmin packed as u32 (f16 pair, **skipped** by this kernel)
//! - Words 1–3: scales (12 bytes, 8×6-bit scale+min pairs)
//! - Words 4–35: qs (128 bytes, 4-bit packed nibbles)
//!
//! # Dequantization (inline per element)
//!
//! For element in sub-block j (0..7) at position p (0..31):
//! ```text
//! sc, m = decode_6bit_scales(j, s0, s1, s2)
//! nibble = extract_nibble(j/2, p, j%2, qs_data)
//! value = d × sc × nibble − dmin × m
//! ```
//!
//! # Dispatch
//!
//! Uses plane (subgroup) cooperative dot product with `plane_sum()` reduction.
//! Each plane handles one output row. Lanes cooperatively compute the dot
//! product, reading contiguous weight elements for coalesced access.
//!
//! | Variant | CubeDim | CubeCount | Rows/cube |
//! |---------|---------|-----------|-----------|
//! | Plane | `new_1d(256)` | `ceil(m/8)` | 8 (Metal subgroup=32) |
//! | Tiled | `new_1d(256)` | `ceil(m/256)` | 256 |

#[cfg(feature = "cubecl_runtime")]
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use riir_infer_core::quant::q4k::BlockQ4K;

/// Q4_K super-block size: 256 elements per block.
pub(crate) const Q4K_BLOCK_SIZE: u32 = 256;

/// Q4_K block size in u32 words: 144 bytes / 4 = 36 words.
pub(crate) const Q4K_WORDS_PER_BLOCK: u32 = 36;

// ── CPU helpers ────────────────────────────────────────────────────

/// Convert IEEE 754 f16 bit pattern to f32.
///
/// Uses the `half` crate for correct conversion including subnormals (Issue 593).
/// Inf/NaN not expected for valid Q4_K data.
#[cfg(feature = "cubecl_runtime")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    // Use the `half` crate for correct f16→f32 conversion, including
    // subnormal (denormal) values. The prior manual implementation flushed
    // subnormals to zero (exp==0 → 0.0), which corrupted Q4_K blocks whose
    // super-block scale `d` fell in the f16 subnormal range (Issue 593).
    // On real Gemma-4-12B FFN weights, ~1.5% of blocks have subnormal d,
    // affecting ~8% of rows → 15% relative error in the GEMV output.
    half::f16::from_bits(bits).to_f32()
}

/// Extract d and dmin from each BlockQ4K as f32 values.
///
/// Returns a flat Vec where each block contributes 2 f32 values: [d, dmin].
/// Total length: `blocks.len() * 2`.
#[cfg(feature = "cubecl_runtime")]
pub fn prepare_q4k_d_dmin(blocks: &[BlockQ4K]) -> Vec<f32> {
    let mut result = Vec::with_capacity(blocks.len() * 2);
    for block in blocks {
        result.push(f16_bits_to_f32(block.d));
        result.push(f16_bits_to_f32(block.dmin));
    }
    result.shrink_to_fit();
    result
}

// ── CubeCL helper functions ────────────────────────────────────────

/// Extract byte at index (0–3) from a u32 word (little-endian).
#[cfg(feature = "cubecl_runtime")]
#[cube]
pub(crate) fn get_byte(word: u32, idx: u32) -> u32 {
    (word >> (idx * 8u32)) & 0xFFu32
}

/// Decode 6-bit scale for sub-block j (0..7) from packed scale words.
///
/// Groups 0–3: `sc = byte(s0, j) & 63`
/// Groups 4–7: `sc = (byte(s2, j-4) & 0x0F) | ((byte(s0, j-4) >> 6) << 4)`
///
/// Returns f32 in range [0, 63].
#[cfg(feature = "cubecl_runtime")]
#[cube]
pub(crate) fn get_scale_k4(j: u32, s0: u32, _s1: u32, s2: u32) -> f32 {
    // Compute for j < 4 path
    let sc_low = get_byte(s0, j) & 63u32;

    // Compute for j >= 4 path (j2 = j - 4, wrapping is safe since we guard)
    let j2 = j - 4u32;
    let sj4 = get_byte(s2, j2);
    let sjm4 = get_byte(s0, j2);
    let sc_high = (sj4 & 0x0Fu32) | ((sjm4 >> 6u32) << 4u32);

    // Select based on j (two if-blocks to avoid conditional expression macro bug)
    let mut sc_u = sc_low;
    if j >= 4u32 {
        sc_u = sc_high;
    }

    sc_u as f32
}

/// Decode 6-bit min for sub-block j (0..7) from packed scale words.
///
/// Groups 0–3: `m = byte(s1, j) & 63`
/// Groups 4–7: `m = (byte(s2, j-4) >> 4) | ((byte(s1, j-4) >> 6) << 4)`
///
/// Returns f32 in range [0, 63].
#[cfg(feature = "cubecl_runtime")]
#[cube]
pub(crate) fn get_min_k4(j: u32, _s0: u32, s1: u32, s2: u32) -> f32 {
    // Compute for j < 4 path
    let m_low = get_byte(s1, j) & 63u32;

    // Compute for j >= 4 path
    let j2 = j - 4u32;
    let sj4 = get_byte(s2, j2);
    let sj = get_byte(s1, j2);
    let m_high = (sj4 >> 4u32) | ((sj >> 6u32) << 4u32);

    // Select based on j
    let mut m_u = m_low;
    if j >= 4u32 {
        m_u = m_high;
    }

    m_u as f32
}

// ---------------------------------------------------------------------------
// Plane (subgroup) Q4_K dequant+GEMV — primary kernel
// ---------------------------------------------------------------------------

/// CubeCL Q4_K dequant+GEMV kernel using plane (subgroup) cooperative dot product.
///
/// Each plane handles one output row. Lanes cooperatively compute the dot product
/// by iterating over Q4_K blocks, dequantizing on-the-fly:
///
/// ```text
/// for each block of 256 elements:
///   load block header (d, dmin from d_dmin buffer; s0, s1, s2 from weight_q4k)
///   for each element this lane handles (stride = PLANE_DIM):
///     decode scale/min from 6-bit packed scales
///     extract 4-bit nibble from qs data
///     dequantize: value = d × scale × nibble − dmin × min
///     accumulate: partial += value × input[col]
/// reduce: plane_sum(partial) → output[row]
/// ```
///
/// # Buffer sizes
///
/// - `weight_q4k`: `m × (n/256) × 36` u32 elements
/// - `d_dmin`: `m × (n/256) × 2` f32 elements
/// - `input`: `n` f32 elements (must be multiple of 256)
/// - `output`: `m` f32 elements
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_q4k_plane(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;
    let blocks_per_row = n / Q4K_BLOCK_SIZE;
    let q4k_stride = blocks_per_row * Q4K_WORDS_PER_BLOCK; // u32 words per row
    let dd_stride = blocks_per_row * 2u32; // f32 elements per row

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let mut partial = f32::new(0.0f32);

    let mut block_idx = 0u32;
    while block_idx < blocks_per_row {
        let col_base = block_idx * Q4K_BLOCK_SIZE;
        let bo_q4k = row * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;
        let bo_dd = row * dd_stride + block_idx * 2u32;

        // Read pre-decoded d and dmin as f32 (converted from f16 on CPU)
        let d = d_dmin[bo_dd as usize];
        let dmin = d_dmin[(bo_dd + 1u32) as usize];

        // Read scale words (3 u32 = 12 bytes encoding 8×6-bit scale+min pairs)
        let s0 = weight_q4k[(bo_q4k + 1u32) as usize]; // scales[0..3]
        let s1 = weight_q4k[(bo_q4k + 2u32) as usize]; // scales[4..7]
        let s2 = weight_q4k[(bo_q4k + 3u32) as usize]; // scales[8..11]

        // Each lane processes elements at stride PLANE_DIM within this 256-element block.
        // For Metal (PLANE_DIM=32): 256/32 = 8 iterations per block.
        let mut k = lane;
        while k < Q4K_BLOCK_SIZE {
            let col = col_base + k;

            // Determine sub-block and position within sub-block
            let sub_block = k / 32u32; // 0..7
            let pos = k % 32u32; // 0..31

            // Decode scale and min for this sub-block
            let sc = get_scale_k4(sub_block, s0, s1, s2);
            let min_val = get_min_k4(sub_block, s0, s1, s2);

            // Extract nibble from packed qs data
            // Pair layout: sub-blocks (2k, 2k+1) share qs[k*32..k*32+32]
            // Sub-block 2k → low nibble, sub-block 2k+1 → high nibble
            let pair = sub_block / 2u32; // 0..3
            let is_high = sub_block % 2u32; // 0 or 1

            let qs_byte_idx = pair * 32u32 + pos;
            let qs_word_idx = bo_q4k + 4u32 + qs_byte_idx / 4u32;
            let byte_in_word = qs_byte_idx % 4u32;
            let shift_bits = byte_in_word * 8u32;
            let qs_byte = (weight_q4k[qs_word_idx as usize] >> shift_bits) & 0xFFu32;

            // Select low or high nibble (two if-blocks to avoid conditional expression)
            let mut nibble = qs_byte & 0x0Fu32;
            if is_high == 1u32 {
                nibble = (qs_byte >> 4u32) & 0x0Fu32;
            }

            // Dequantize: value = d × scale × nibble − dmin × min
            let dequant = d * sc * (nibble as f32) - dmin * min_val;

            partial += dequant * input[col as usize];

            k += PLANE_DIM;
        }

        block_idx += 1u32;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane
    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row
    if lane == 0u32 {
        output[row as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Row-tiled Q4_K dequant+GEMV (Issue 609)
// ---------------------------------------------------------------------------

/// Output rows processed per plane by [`gemv_q4k_plane_rowtiled`].
///
/// Must equal that kernel's hand-unrolled accumulator count — a mismatch leaves
/// output rows unwritten, which a relative-error check cannot see because CubeCL
/// pools buffers (the bug that produced a bogus 2.80× in Issue 606 T3). The G1
/// gate asserts full coverage via a sentinel pre-fill for exactly this reason.
#[cfg(all(feature = "cubecl_runtime", feature = "q4k_rowtiled_gemv"))]
pub(crate) const Q4K_ROWS_PER_PLANE: u32 = 4;

/// Row-tiled Q4_K GEMV — one input load feeds 4 output rows.
///
/// # Why (Issue 609, from the Issue 606 T4 audit)
///
/// The T4 audit measured `gemv_q4k_plane` at **4.5–11.9% of the M3 Max
/// roofline** — worse than the ternary kernel it was the structural template
/// for, which reaches ~23% on identical geometry (47.5 vs 92.4 GB/s at
/// 17408×5120, a 1.95× deficit at equal bytes).
///
/// This applies the change that bought the ternary path 1.95×: give each plane
/// `Q4K_ROWS_PER_PLANE` consecutive rows and hold one accumulator per row in
/// registers, so each `input[col]` is loaded **once** and reused across all four
/// rows instead of being re-read per row.
///
/// # Honest expectation — smaller than the ternary win
///
/// Row tiling saves input loads, and Q4_K has proportionally fewer of them to
/// save. Per element the ternary kernel did ~9 ALU ops against 1 input load;
/// Q4_K does ~15 (6-bit scale decode, nibble extract, `d*sc*nib − dmin*min`)
/// against the same 1 load. Saving 3 of 4 loads is therefore a much smaller
/// fraction of total cost here, so this is expected to land nearer 1.1–1.3× than
/// 2×. The G2 gate (≥1.5×) may well fail; see the module note on the remaining
/// lever if it does.
///
/// # The bigger lever this does NOT take (recorded so it is not lost)
///
/// Within one sub-block the dequant is affine in the nibble, so it factors:
///
/// ```text
/// Σ_j (d·sc·nib_j − dmin·min) · x_j  =  d·sc·(Σ_j nib_j·x_j) − dmin·min·(Σ_j x_j)
/// ```
///
/// That would replace ~4 ops per element (2 multiplies, a subtract, an FMA) with
/// ~2 (one FMA for `nib·x`, one add for `x`), applying `d·sc` and `dmin·min` once
/// per sub-block — and it would make the 6-bit scale decode once per lane per
/// block instead of once per iteration. It is **not** done here because it
/// requires changing the lane→element map: today `k = lane + i·32` makes
/// `sub_block = i`, so every lane visits all 8 sub-blocks and `sc` changes every
/// iteration. Factoring needs each lane pinned to one sub-block (lane `L` →
/// sub-block `L/4`, elements `(L%4)·8 .. +8`), which is a different kernel, not a
/// tweak. Issue 609 scoped the row-tiling port; this stays a follow-up so the two
/// effects are measured separately rather than confounded.
#[cfg(all(feature = "cubecl_runtime", feature = "q4k_rowtiled_gemv"))]
#[cube(launch_unchecked)]
fn gemv_q4k_plane_rowtiled(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;
    let blocks_per_row = n / Q4K_BLOCK_SIZE;
    let q4k_stride = blocks_per_row * Q4K_WORDS_PER_BLOCK;
    let dd_stride = blocks_per_row * 2u32;

    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let row_base = plane_id * Q4K_ROWS_PER_PLANE;

    if row_base >= m {
        terminate!();
    }

    // Rows past `m` alias `m-1` so every load stays in bounds; discarded at
    // write time.
    let r0 = row_base;
    let mut r1 = m - 1u32;
    let mut r2 = m - 1u32;
    let mut r3 = m - 1u32;
    if row_base + 1u32 < m {
        r1 = row_base + 1u32;
    }
    if row_base + 2u32 < m {
        r2 = row_base + 2u32;
    }
    if row_base + 3u32 < m {
        r3 = row_base + 3u32;
    }

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);

    let mut block_idx = 0u32;
    while block_idx < blocks_per_row {
        let col_base = block_idx * Q4K_BLOCK_SIZE;

        let b0 = r0 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;
        let b1 = r1 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;
        let b2 = r2 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;
        let b3 = r3 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;

        let e0 = r0 * dd_stride + block_idx * 2u32;
        let e1 = r1 * dd_stride + block_idx * 2u32;
        let e2 = r2 * dd_stride + block_idx * 2u32;
        let e3 = r3 * dd_stride + block_idx * 2u32;

        let d0 = d_dmin[e0 as usize];
        let dmin0 = d_dmin[(e0 + 1u32) as usize];
        let d1 = d_dmin[e1 as usize];
        let dmin1 = d_dmin[(e1 + 1u32) as usize];
        let d2 = d_dmin[e2 as usize];
        let dmin2 = d_dmin[(e2 + 1u32) as usize];
        let d3 = d_dmin[e3 as usize];
        let dmin3 = d_dmin[(e3 + 1u32) as usize];

        let s00 = weight_q4k[(b0 + 1u32) as usize];
        let s01 = weight_q4k[(b0 + 2u32) as usize];
        let s02 = weight_q4k[(b0 + 3u32) as usize];
        let s10 = weight_q4k[(b1 + 1u32) as usize];
        let s11 = weight_q4k[(b1 + 2u32) as usize];
        let s12 = weight_q4k[(b1 + 3u32) as usize];
        let s20 = weight_q4k[(b2 + 1u32) as usize];
        let s21 = weight_q4k[(b2 + 2u32) as usize];
        let s22 = weight_q4k[(b2 + 3u32) as usize];
        let s30 = weight_q4k[(b3 + 1u32) as usize];
        let s31 = weight_q4k[(b3 + 2u32) as usize];
        let s32 = weight_q4k[(b3 + 3u32) as usize];

        // `k = lane + i*32`, so `sub_block == i` and `pos == lane`.
        let mut k = lane;
        while k < Q4K_BLOCK_SIZE {
            let col = col_base + k;
            let sub_block = k / 32u32;
            let pos = k % 32u32;

            // The one load this tile exists to amortize.
            let x = input[col as usize];

            let pair = sub_block / 2u32;
            let is_high = sub_block % 2u32;
            let qs_byte_idx = pair * 32u32 + pos;
            let word_off = 4u32 + qs_byte_idx / 4u32;
            let shift_bits = (qs_byte_idx % 4u32) * 8u32;

            let byte0 = (weight_q4k[(b0 + word_off) as usize] >> shift_bits) & 0xFFu32;
            let byte1 = (weight_q4k[(b1 + word_off) as usize] >> shift_bits) & 0xFFu32;
            let byte2 = (weight_q4k[(b2 + word_off) as usize] >> shift_bits) & 0xFFu32;
            let byte3 = (weight_q4k[(b3 + word_off) as usize] >> shift_bits) & 0xFFu32;

            let mut nib0 = byte0 & 0x0Fu32;
            let mut nib1 = byte1 & 0x0Fu32;
            let mut nib2 = byte2 & 0x0Fu32;
            let mut nib3 = byte3 & 0x0Fu32;
            if is_high == 1u32 {
                nib0 = (byte0 >> 4u32) & 0x0Fu32;
                nib1 = (byte1 >> 4u32) & 0x0Fu32;
                nib2 = (byte2 >> 4u32) & 0x0Fu32;
                nib3 = (byte3 >> 4u32) & 0x0Fu32;
            }

            acc0 += (d0 * get_scale_k4(sub_block, s00, s01, s02) * (nib0 as f32)
                - dmin0 * get_min_k4(sub_block, s00, s01, s02))
                * x;
            acc1 += (d1 * get_scale_k4(sub_block, s10, s11, s12) * (nib1 as f32)
                - dmin1 * get_min_k4(sub_block, s10, s11, s12))
                * x;
            acc2 += (d2 * get_scale_k4(sub_block, s20, s21, s22) * (nib2 as f32)
                - dmin2 * get_min_k4(sub_block, s20, s21, s22))
                * x;
            acc3 += (d3 * get_scale_k4(sub_block, s30, s31, s32) * (nib3 as f32)
                - dmin3 * get_min_k4(sub_block, s30, s31, s32))
                * x;

            k += PLANE_DIM;
        }

        block_idx += 1u32;
    }

    let t0 = plane_sum(acc0);
    let t1 = plane_sum(acc1);
    let t2 = plane_sum(acc2);
    let t3 = plane_sum(acc3);

    if lane == 0u32 {
        output[r0 as usize] = t0;
        if row_base + 1u32 < m {
            output[r1 as usize] = t1;
        }
        if row_base + 2u32 < m {
            output[r2 as usize] = t2;
        }
        if row_base + 3u32 < m {
            output[r3 as usize] = t3;
        }
    }
}

/// Sub-block-factored, row-tiled Q4_K GEMV (Issue 609 T2).
///
/// # The factoring
///
/// Within one 32-element sub-block, `sc` and `min` are constant, so the dequant
/// is affine in the nibble and the scales lift out of the element loop:
///
/// ```text
/// Σ_j (d·sc·nib_j − dmin·min) · x_j  =  d·sc·(Σ_j nib_j·x_j) − dmin·min·(Σ_j x_j)
/// ```
///
/// [`gemv_q4k_plane_rowtiled`] cannot use this because its lane→element map
/// (`k = lane + i*32`) makes `sub_block == i`: every lane visits all 8
/// sub-blocks, so `sc`/`min` change every iteration and nothing hoists. This
/// kernel pins each lane to **one** sub-block, which is what unlocks the
/// factoring — and, as a side effect, fixes two other findings from the Issue
/// 606 T4 audit at the same time.
///
/// # Lane map
///
/// `lane → sub_block = lane/4`, `chunk = lane%4`, covering the 8 consecutive
/// elements `pos = chunk*8 .. +8`. Across 32 lanes that is `8 × 4 × 8 = 256`
/// elements — an exact partition of the block, so the strided inner loop
/// disappears entirely.
///
/// # What this buys, per lane per block per row (8 elements)
///
/// | | rowtiled (T1) | factored (T2) |
/// |---|---|---|
/// | weight word loads | 8 | **2** |
/// | 6-bit scale/min decodes | 8 pairs | **1 pair** |
/// | multiplies for scale application | 8×(2 mul + 1 sub) | **2 mul + 1 sub** |
/// | per-element work | ~22 ops | **~7 ops + FMA** |
///
/// The 8 consecutive bytes a lane needs live in exactly **2 u32 words**, so the
/// nibbles come from two loads instead of eight — which also removes the 4×
/// redundant-load pattern the T4 audit flagged (lanes 0–3 previously all read
/// word `bo+4`).
///
/// `Σ_j x_j` is **row-independent**, so it is accumulated once and reused across
/// all four rows; only `Σ_j nib_j·x_j` is per-row.
///
/// # Preconditions
///
/// `n` must be a multiple of `Q4K_BLOCK_SIZE` (256), which `Q4KHandle::from_blocks`
/// already asserts — so the element partition is exact and the kernel needs no
/// bounds checks on `col`.
#[cfg(all(feature = "cubecl_runtime", feature = "q4k_rowtiled_gemv"))]
#[cube(launch_unchecked)]
fn gemv_q4k_plane_factored(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;
    let blocks_per_row = n / Q4K_BLOCK_SIZE;
    let q4k_stride = blocks_per_row * Q4K_WORDS_PER_BLOCK;
    let dd_stride = blocks_per_row * 2u32;

    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let row_base = plane_id * Q4K_ROWS_PER_PLANE;

    if row_base >= m {
        terminate!();
    }

    let r0 = row_base;
    let mut r1 = m - 1u32;
    let mut r2 = m - 1u32;
    let mut r3 = m - 1u32;
    if row_base + 1u32 < m {
        r1 = row_base + 1u32;
    }
    if row_base + 2u32 < m {
        r2 = row_base + 2u32;
    }
    if row_base + 3u32 < m {
        r3 = row_base + 3u32;
    }

    // Lane-fixed geometry — all loop-invariant, computed once.
    let sub_block = lane / 4u32;
    let chunk = lane % 4u32;
    let pair = sub_block / 2u32;
    let is_high = sub_block % 2u32;
    // 8 consecutive qs bytes at `pair*32 + chunk*8` → exactly 2 u32 words.
    let word0 = pair * 8u32 + chunk * 2u32;
    // Nibble select folds into the shift: low nibble at +0, high at +4.
    let sh = is_high * 4u32;
    let elem_off = sub_block * 32u32 + chunk * 8u32;

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);

    let mut block_idx = 0u32;
    while block_idx < blocks_per_row {
        let col = block_idx * Q4K_BLOCK_SIZE + elem_off;

        // Row-independent: loaded once, reused by all four rows.
        let x0 = input[col as usize];
        let x1 = input[(col + 1u32) as usize];
        let x2 = input[(col + 2u32) as usize];
        let x3 = input[(col + 3u32) as usize];
        let x4 = input[(col + 4u32) as usize];
        let x5 = input[(col + 5u32) as usize];
        let x6 = input[(col + 6u32) as usize];
        let x7 = input[(col + 7u32) as usize];
        let sum_x = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

        let wb = block_idx * Q4K_WORDS_PER_BLOCK + 4u32 + word0;
        let eb = block_idx * 2u32;

        // ── row 0 ──
        let b0 = r0 * q4k_stride + wb;
        let w00 = weight_q4k[b0 as usize];
        let w01 = weight_q4k[(b0 + 1u32) as usize];
        let s00 = weight_q4k[(r0 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 1u32) as usize];
        let s01 = weight_q4k[(r0 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 2u32) as usize];
        let s02 = weight_q4k[(r0 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 3u32) as usize];
        let nib_dot0 = (((w00 >> sh) & 15u32) as f32) * x0
            + (((w00 >> (8u32 + sh)) & 15u32) as f32) * x1
            + (((w00 >> (16u32 + sh)) & 15u32) as f32) * x2
            + (((w00 >> (24u32 + sh)) & 15u32) as f32) * x3
            + (((w01 >> sh) & 15u32) as f32) * x4
            + (((w01 >> (8u32 + sh)) & 15u32) as f32) * x5
            + (((w01 >> (16u32 + sh)) & 15u32) as f32) * x6
            + (((w01 >> (24u32 + sh)) & 15u32) as f32) * x7;
        let e0 = r0 * dd_stride + eb;
        acc0 += d_dmin[e0 as usize] * get_scale_k4(sub_block, s00, s01, s02) * nib_dot0
            - d_dmin[(e0 + 1u32) as usize] * get_min_k4(sub_block, s00, s01, s02) * sum_x;

        // ── row 1 ──
        let b1 = r1 * q4k_stride + wb;
        let w10 = weight_q4k[b1 as usize];
        let w11 = weight_q4k[(b1 + 1u32) as usize];
        let s10 = weight_q4k[(r1 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 1u32) as usize];
        let s11 = weight_q4k[(r1 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 2u32) as usize];
        let s12 = weight_q4k[(r1 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 3u32) as usize];
        let nib_dot1 = (((w10 >> sh) & 15u32) as f32) * x0
            + (((w10 >> (8u32 + sh)) & 15u32) as f32) * x1
            + (((w10 >> (16u32 + sh)) & 15u32) as f32) * x2
            + (((w10 >> (24u32 + sh)) & 15u32) as f32) * x3
            + (((w11 >> sh) & 15u32) as f32) * x4
            + (((w11 >> (8u32 + sh)) & 15u32) as f32) * x5
            + (((w11 >> (16u32 + sh)) & 15u32) as f32) * x6
            + (((w11 >> (24u32 + sh)) & 15u32) as f32) * x7;
        let e1 = r1 * dd_stride + eb;
        acc1 += d_dmin[e1 as usize] * get_scale_k4(sub_block, s10, s11, s12) * nib_dot1
            - d_dmin[(e1 + 1u32) as usize] * get_min_k4(sub_block, s10, s11, s12) * sum_x;

        // ── row 2 ──
        let b2 = r2 * q4k_stride + wb;
        let w20 = weight_q4k[b2 as usize];
        let w21 = weight_q4k[(b2 + 1u32) as usize];
        let s20 = weight_q4k[(r2 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 1u32) as usize];
        let s21 = weight_q4k[(r2 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 2u32) as usize];
        let s22 = weight_q4k[(r2 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 3u32) as usize];
        let nib_dot2 = (((w20 >> sh) & 15u32) as f32) * x0
            + (((w20 >> (8u32 + sh)) & 15u32) as f32) * x1
            + (((w20 >> (16u32 + sh)) & 15u32) as f32) * x2
            + (((w20 >> (24u32 + sh)) & 15u32) as f32) * x3
            + (((w21 >> sh) & 15u32) as f32) * x4
            + (((w21 >> (8u32 + sh)) & 15u32) as f32) * x5
            + (((w21 >> (16u32 + sh)) & 15u32) as f32) * x6
            + (((w21 >> (24u32 + sh)) & 15u32) as f32) * x7;
        let e2 = r2 * dd_stride + eb;
        acc2 += d_dmin[e2 as usize] * get_scale_k4(sub_block, s20, s21, s22) * nib_dot2
            - d_dmin[(e2 + 1u32) as usize] * get_min_k4(sub_block, s20, s21, s22) * sum_x;

        // ── row 3 ──
        let b3 = r3 * q4k_stride + wb;
        let w30 = weight_q4k[b3 as usize];
        let w31 = weight_q4k[(b3 + 1u32) as usize];
        let s30 = weight_q4k[(r3 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 1u32) as usize];
        let s31 = weight_q4k[(r3 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 2u32) as usize];
        let s32 = weight_q4k[(r3 * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK + 3u32) as usize];
        let nib_dot3 = (((w30 >> sh) & 15u32) as f32) * x0
            + (((w30 >> (8u32 + sh)) & 15u32) as f32) * x1
            + (((w30 >> (16u32 + sh)) & 15u32) as f32) * x2
            + (((w30 >> (24u32 + sh)) & 15u32) as f32) * x3
            + (((w31 >> sh) & 15u32) as f32) * x4
            + (((w31 >> (8u32 + sh)) & 15u32) as f32) * x5
            + (((w31 >> (16u32 + sh)) & 15u32) as f32) * x6
            + (((w31 >> (24u32 + sh)) & 15u32) as f32) * x7;
        let e3 = r3 * dd_stride + eb;
        acc3 += d_dmin[e3 as usize] * get_scale_k4(sub_block, s30, s31, s32) * nib_dot3
            - d_dmin[(e3 + 1u32) as usize] * get_min_k4(sub_block, s30, s31, s32) * sum_x;

        block_idx += 1u32;
    }

    let t0 = plane_sum(acc0);
    let t1 = plane_sum(acc1);
    let t2 = plane_sum(acc2);
    let t3 = plane_sum(acc3);

    if lane == 0u32 {
        output[r0 as usize] = t0;
        if row_base + 1u32 < m {
            output[r1 as usize] = t1;
        }
        if row_base + 2u32 < m {
            output[r2 as usize] = t2;
        }
        if row_base + 3u32 < m {
            output[r3 as usize] = t3;
        }
    }
}

// ---------------------------------------------------------------------------
// Shared memory tiled Q4_K dequant+GEMV — fallback kernel
// ---------------------------------------------------------------------------

/// CubeCL Q4_K dequant+GEMV kernel using shared memory tiling (no subgroups).
///
/// Each thread computes one output row. Input vector tiles (256 elements)
/// are loaded into shared memory, matching Q4_K block size for natural
/// alignment. The tiling matches `gemv_tile_f32` but with inline dequant.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_q4k_tiled(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;
    let blocks_per_row = n / Q4K_BLOCK_SIZE;
    let q4k_stride = blocks_per_row * Q4K_WORDS_PER_BLOCK;
    let dd_stride = blocks_per_row * 2u32;

    // Each thread handles one output row
    let row = ABSOLUTE_POS as u32;

    if row >= m {
        terminate!();
    }

    let mut sum = f32::new(0.0f32);

    // Shared memory for input tile: 256 f32 = 1 KB
    // Tile size matches Q4_K block size for natural alignment
    let mut tile_input = Shared::<[f32]>::new_slice(Q4K_BLOCK_SIZE as usize);

    let t = UNIT_POS;
    let mut block_idx = 0u32;

    while block_idx < blocks_per_row {
        let col_base = block_idx * Q4K_BLOCK_SIZE;

        // Cooperatively load 256 input values into shared memory
        let input_idx = col_base + t;
        if input_idx < n {
            tile_input[t as usize] = input[input_idx as usize];
        } else {
            tile_input[t as usize] = f32::new(0.0f32);
        }

        sync_cube();

        // Read block header for this row's block
        let bo_q4k = row * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;
        let bo_dd = row * dd_stride + block_idx * 2u32;

        let d = d_dmin[bo_dd as usize];
        let dmin = d_dmin[(bo_dd + 1u32) as usize];
        let s0 = weight_q4k[(bo_q4k + 1u32) as usize];
        let s1 = weight_q4k[(bo_q4k + 2u32) as usize];
        let s2 = weight_q4k[(bo_q4k + 3u32) as usize];

        // Process all 256 elements in this block
        let mut k = 0u32;
        while k < Q4K_BLOCK_SIZE {
            let sub_block = k / 32u32;
            let pos = k % 32u32;

            let sc = get_scale_k4(sub_block, s0, s1, s2);
            let min_val = get_min_k4(sub_block, s0, s1, s2);

            let pair = sub_block / 2u32;
            let is_high = sub_block % 2u32;

            let qs_byte_idx = pair * 32u32 + pos;
            let qs_word_idx = bo_q4k + 4u32 + qs_byte_idx / 4u32;
            let byte_in_word = qs_byte_idx % 4u32;
            let shift_bits = byte_in_word * 8u32;
            let qs_byte = (weight_q4k[qs_word_idx as usize] >> shift_bits) & 0xFFu32;

            let mut nibble = qs_byte & 0x0Fu32;
            if is_high == 1u32 {
                nibble = (qs_byte >> 4u32) & 0x0Fu32;
            }

            let dequant = d * sc * (nibble as f32) - dmin * min_val;
            sum += dequant * tile_input[k as usize];

            k += 1u32;
        }

        sync_cube();
        block_idx += 1u32;
    }

    output[row as usize] = sum;
}

// ---------------------------------------------------------------------------
// Q4K Handle wrapper
// ---------------------------------------------------------------------------

/// Paired GPU handles for one Q4_K quantized projection.
///
/// Stores the packed Q4_K block data (scales + qs) and pre-decoded
/// d/dmin f32 values as separate CubeCL handles.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct Q4KHandle {
    /// Packed Q4_K blocks as `Array<u32>` (36 u32 per block per row).
    /// Layout matches WGSL `gemv_q4k.wgsl` for zero-copy GGUF compatibility.
    pub weight_q4k: Handle,
    /// Pre-decoded d and dmin as `Array<f32>` (2 f32 per block per row).
    /// Converted from f16 on CPU to avoid GPU-side f16→f32 conversion.
    pub d_dmin: Handle,
    /// Output dimension (number of rows).
    pub m: usize,
    /// Input dimension (must be multiple of 256).
    pub n: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl Q4KHandle {
    /// Upload a quantized projection matrix to CubeCL GPU buffers.
    ///
    /// `blocks` contains the packed `BlockQ4K` array for the full `[m, n]` matrix.
    /// Each row has `n / 256` blocks. Total blocks: `m * (n / 256)`.
    ///
    /// Creates two GPU buffers:
    /// 1. `weight_q4k`: raw block bytes as u32 array (zero-copy compatible)
    /// 2. `d_dmin`: d and dmin pre-decoded from f16 to f32
    pub fn from_blocks(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        blocks: &[BlockQ4K],
        m: usize,
        n: usize,
    ) -> Self {
        assert!(
            n.is_multiple_of(Q4K_BLOCK_SIZE as usize),
            "n ({n}) must be multiple of {Q4K_BLOCK_SIZE}"
        );

        // Upload raw block bytes as u32 array
        let weight_bytes = bytemuck::cast_slice::<BlockQ4K, u8>(blocks);
        let weight_q4k = client.create_from_slice(weight_bytes);

        // Pre-decode d and dmin from f16 to f32
        let d_dmin_data = prepare_q4k_d_dmin(blocks);
        let d_dmin = client.create_from_slice(f32::as_bytes(&d_dmin_data));

        Self {
            weight_q4k,
            d_dmin,
            m,
            n,
        }
    }

    /// Block count per row.
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.n / Q4K_BLOCK_SIZE as usize
    }
}

// ---------------------------------------------------------------------------
// Public API: auto-selecting launcher
// ---------------------------------------------------------------------------

/// CubeCL Q4_K dequant+GEMV launcher with automatic plane/tiled selection.
///
/// Selects the plane (subgroup) kernel when the device supports it,
/// otherwise falls back to the shared-memory tiled kernel.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = GpuContext::new()?;
/// let client = ctx.cubecl_client();
///
/// // Quantize and upload weights
/// let blocks = quantize_projection(&weights, rows, cols);
/// let handle = Q4KHandle::from_blocks(&client, &blocks, rows, cols);
///
/// // Launch GEMV
/// let input_handle = client.create_from_slice(f32::as_bytes(&input));
/// let output_handle = client.empty(rows * 4);
///
/// unsafe {
///     GemvQ4KCubeCL::launch::<ActiveRuntime>(
///         &client, &handle, input_handle, output_handle,
///     );
/// }
///
/// let result = f32::from_bytes(&client.read_one(output_handle).unwrap());
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct GemvQ4KCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvQ4KCubeCL {
    /// Launch Q4_K dequant+GEMV: `output[M] = dequant_q4k(weight) @ input[N]`.
    ///
    /// Auto-selects plane or tiled kernel based on device capabilities.
    ///
    /// # Safety
    ///
    /// - `handle.weight_q4k` must have `m × (n/256) × 36` u32 elements
    /// - `handle.d_dmin` must have `m × (n/256) × 2` f32 elements
    /// - `input_handle` must have `n` f32 elements (n must be multiple of 256)
    /// - `output_handle` must have `m` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let has_plane = client.features().plane.contains(Plane::Ops);

        unsafe {
            match has_plane {
                // Issue 609 T2: the sub-block-factored kernel measures 3.27–3.37×
                // over `launch_plane` on M3 Metal, lifting the dominant shapes
                // from 11.9% to 41–45% of the 400 GB/s roofline. Gated on NOT
                // `cuda_backend` (except macOS, where the feature is inert —
                // Issue 949) because Issue 607 G5 found a 1.48× Metal win on
                // the ternary path *reverse* to 0.48× on CUDA; Issue 609 G5 must
                // sweep this kernel there before any CUDA promotion.
                #[cfg(all(feature = "q4k_rowtiled_gemv", any(not(feature = "cuda_backend"), target_os = "macos")))]
                true => Self::launch_factored::<R>(client, handle, input_handle, output_handle),
                #[cfg(not(all(feature = "q4k_rowtiled_gemv", any(not(feature = "cuda_backend"), target_os = "macos"))))]
                true => Self::launch_plane::<R>(client, handle, input_handle, output_handle),
                false => Self::launch_tiled::<R>(client, handle, input_handle, output_handle),
            }
        }
    }

    /// Launch plane (subgroup) Q4_K dequant+GEMV kernel.
    ///
    /// Each plane handles one output row with cooperative dot product + `plane_sum()`.
    /// Workgroup: 256 threads → 8 planes (with plane_dim=32 on Metal).
    /// Dispatch: `ceil(m / 8)` workgroups.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_plane<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Conservative: Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg = (handle.m as u32).div_ceil(rows_per_wg).max(1);

        let m = handle.m;
        let n = handle.n;
        let blocks_per_row = handle.blocks_per_row();
        let weight_len = m * blocks_per_row * (Q4K_WORDS_PER_BLOCK as usize);
        let dd_len = m * blocks_per_row * 2;

        unsafe {
            gemv_q4k_plane::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }

    /// Launch the row-tiled Q4_K dequant+GEMV kernel (Issue 609).
    ///
    /// Each plane owns `Q4K_ROWS_PER_PLANE` consecutive rows, so a workgroup
    /// covers `(wg_size / plane_size) * ROWS_PER_PLANE` = 32 rows. See
    /// [`gemv_q4k_plane_rowtiled`].
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    #[cfg(feature = "q4k_rowtiled_gemv")]
    pub unsafe fn launch_rowtiled<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * Q4K_ROWS_PER_PLANE; // 32
        let num_wg = (handle.m as u32).div_ceil(rows_per_wg).max(1);

        let m = handle.m;
        let n = handle.n;
        let blocks_per_row = handle.blocks_per_row();
        let weight_len = m * blocks_per_row * (Q4K_WORDS_PER_BLOCK as usize);
        let dd_len = m * blocks_per_row * 2;

        unsafe {
            gemv_q4k_plane_rowtiled::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }

    /// Launch the sub-block-factored row-tiled Q4_K GEMV (Issue 609 T2).
    ///
    /// Same 32-rows-per-workgroup geometry as [`Self::launch_rowtiled`]; the
    /// difference is entirely inside the kernel (lanes pinned to one sub-block,
    /// scales lifted out of the element loop). See [`gemv_q4k_plane_factored`].
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`, plus `handle.n` must be a multiple of
    /// `Q4K_BLOCK_SIZE` (guaranteed by `Q4KHandle::from_blocks`).
    #[cfg(feature = "q4k_rowtiled_gemv")]
    pub unsafe fn launch_factored<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * Q4K_ROWS_PER_PLANE; // 32
        let num_wg = (handle.m as u32).div_ceil(rows_per_wg).max(1);

        let m = handle.m;
        let n = handle.n;
        let blocks_per_row = handle.blocks_per_row();
        let weight_len = m * blocks_per_row * (Q4K_WORDS_PER_BLOCK as usize);
        let dd_len = m * blocks_per_row * 2;

        unsafe {
            gemv_q4k_plane_factored::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }

    /// Launch shared-memory tiled Q4_K dequant+GEMV kernel (fallback, no subgroups).
    ///
    /// Each thread computes one output row. Input tiled into shared memory.
    /// Workgroup: 256 threads → 256 rows per workgroup.
    /// Dispatch: `ceil(m / 256)` workgroups.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_tiled<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let num_wg = (handle.m as u32).div_ceil(wg_size).max(1);

        let m = handle.m;
        let n = handle.n;
        let blocks_per_row = handle.blocks_per_row();
        let weight_len = m * blocks_per_row * (Q4K_WORDS_PER_BLOCK as usize);
        let dd_len = m * blocks_per_row * 2;

        unsafe {
            gemv_q4k_tiled::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::context::GpuContext;
    use bytemuck::Zeroable;
    use crate::cubecl_runtime::ActiveRuntime;
    use riir_infer_core::quant::q4k::{
        BlockQ4K, QK_K, gemv_q4_k, quantize_row_q4_k,
    };

    /// Helper: quantize a projection, run CubeCL GEMV, return result.
    ///
    /// 1. Quantize f32 weight data to Q4_K blocks
    /// 2. Upload to CubeCL
    /// 3. Run dequant+GEMV
    /// 4. Read result
    fn run_q4k_gemv(weight: &[f32], input: &[f32], m: usize, n: usize) -> Vec<f32> {
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();

        // Quantize weight to Q4_K
        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;
        let mut all_blocks = Vec::new();

        for row in 0..m {
            let mut padded_row = vec![0.0f32; padded_n];
            let src = &weight[row * n..(row + 1) * n];
            padded_row[..n].copy_from_slice(src);

            let row_blocks_start = all_blocks.len();
            all_blocks.resize(row_blocks_start + blocks_per_row, BlockQ4K::zeroed());
            quantize_row_q4_k(&padded_row, &mut all_blocks[row_blocks_start..]);
        }

        // For non-aligned n, use padded_n for kernel (must be multiple of 256)
        let effective_n = padded_n;

        // Upload
        let q4k_handle = Q4KHandle::from_blocks(&client, &all_blocks, m, effective_n);

        // Pad input to effective_n if needed
        let mut padded_input = input.to_vec();
        if padded_input.len() < effective_n {
            padded_input.resize(effective_n, 0.0);
        }

        let input_handle = client.create_from_slice(f32::as_bytes(&padded_input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        // SAFETY: buffer sizes match the kernel requirements
        unsafe {
            GemvQ4KCubeCL::launch::<ActiveRuntime>(
                &client,
                &q4k_handle,
                input_handle,
                output_handle.clone(),
            );
        }

        // Read result
        let bytes = client.read_one(output_handle).expect("should read output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Helper: CPU reference dequant+GEMV.
    ///
    /// Uses the canonical `riir_infer_core::quant::q4k::gemv_q4_k` (Plan 486). This
    /// replaces the former inline two-step (`dequantize_row_q4_k` + scalar dot)
    /// with a single call to the fused CPU GEMV, removing code duplication.
    fn cpu_q4k_gemv(weight: &[f32], input: &[f32], m: usize, n: usize) -> Vec<f32> {
        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;

        // Quantize all rows into contiguous block array.
        let mut all_blocks = vec![BlockQ4K::zeroed(); m * blocks_per_row];
        for row in 0..m {
            let mut padded_row = vec![0.0_f32; padded_n];
            padded_row[..n].copy_from_slice(&weight[row * n..(row + 1) * n]);
            quantize_row_q4_k(
                &padded_row,
                &mut all_blocks[row * blocks_per_row..(row + 1) * blocks_per_row],
            );
        }

        // Pad input to padded_n to match the GEMV's n contract.
        let mut padded_input = input.to_vec();
        if padded_input.len() < padded_n {
            padded_input.resize(padded_n, 0.0);
        }

        // Fused CPU GEMV (arithmetic path by default; LUT path if simd_lut_q4k).
        gemv_q4_k(&all_blocks, &padded_input[..padded_n], m, padded_n)
    }

    #[test]
    fn test_q4k_prepare_d_dmin() {
        // Create a simple block with known d and dmin
        let src = vec![0.5f32; QK_K];
        let mut blocks = [BlockQ4K::zeroed(); 1];
        quantize_row_q4_k(&src, &mut blocks);

        let d_dmin = prepare_q4k_d_dmin(&blocks);
        assert_eq!(d_dmin.len(), 2, "Should have 2 f32 per block");

        // d should be positive (it's a scale factor)
        assert!(d_dmin[0] > 0.0, "d should be positive, got {}", d_dmin[0]);
        // dmin should be non-negative
        assert!(
            d_dmin[1] >= 0.0,
            "dmin should be non-negative, got {}",
            d_dmin[1]
        );
    }

    #[test]
    fn test_q4k_gemv_zeros() {
        // Zero weights → zero output
        let m = 4;
        let n = 256;
        let weight = vec![0.0f32; m * n];
        let input = vec![1.0f32; n];

        let result = run_q4k_gemv(&weight, &input, m, n);

        assert_eq!(result.len(), m, "Output should have {m} elements");
        for (i, &v) in result.iter().enumerate() {
            assert!(
                v.abs() < 0.5,
                "Zero weight GEMV output[{i}] should be ~0, got {v}"
            );
        }
    }

    #[test]
    fn test_q4k_gemv_ones_input() {
        // Simple 2×256 matrix, all-ones input
        let m = 2;
        let n = 256;
        let weight: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let input = vec![1.0f32; n];

        let gpu_result = run_q4k_gemv(&weight, &input, m, n);
        let cpu_result = cpu_q4k_gemv(&weight, &input, m, n);

        assert_eq!(gpu_result.len(), m, "Output should have {m} elements");
        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            let err = (gpu - cpu).abs();
            assert!(
                err < 1.0,
                "Output[{i}]: GPU={gpu}, CPU={cpu}, err={err} (Q4_K quantization tolerance)"
            );
        }
    }

    #[test]
    fn test_q4k_gemv_matches_cpu() {
        // 4×256 matrix with sine wave weights
        let m = 4;
        let n = 256;
        let weight: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.1).sin() * 2.0).collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        let gpu_result = run_q4k_gemv(&weight, &input, m, n);
        let cpu_result = cpu_q4k_gemv(&weight, &input, m, n);

        let mut max_err = 0.0f32;
        for (&gpu, &cpu) in gpu_result.iter().zip(cpu_result.iter()) {
            let err = (gpu - cpu).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // Q4_K quantization error: ~0.5 per element × 256 = ~128 total possible error
        // But with random-ish data, errors cancel → expect < 5.0
        assert!(
            max_err < 10.0,
            "Max error too large: {max_err} (Q4_K quantization tolerance)"
        );
    }

    #[test]
    fn test_q4k_gemv_large() {
        // Larger matrix: 8×512 (2 blocks per row)
        let m = 8;
        let n = 512;
        let weight: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.05).sin() * 0.5)
            .collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos() * 0.3).collect();

        let gpu_result = run_q4k_gemv(&weight, &input, m, n);
        let cpu_result = cpu_q4k_gemv(&weight, &input, m, n);

        assert_eq!(gpu_result.len(), m);
        let mut max_err = 0.0f32;
        for (&gpu, &cpu) in gpu_result.iter().zip(cpu_result.iter()) {
            let err = (gpu - cpu).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // Larger matrix → more accumulated quantization error
        assert!(
            max_err < 15.0,
            "Max error too large: {max_err} (multi-block Q4_K tolerance)"
        );
    }

    #[test]
    fn test_q4k_gemv_identity_like() {
        // Diagonal-like matrix: each row has one dominant element
        let m = 4;
        let n = 256;
        let mut weight = vec![0.01f32; m * n];
        for row in 0..m {
            weight[row * n + row * 64] = 10.0; // Dominant element at different positions
        }
        let input = vec![1.0f32; n];

        let gpu_result = run_q4k_gemv(&weight, &input, m, n);
        let cpu_result = cpu_q4k_gemv(&weight, &input, m, n);

        // Each output should be roughly the sum of one dominant element + small background
        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            let err = (gpu - cpu).abs();
            assert!(err < 2.0, "Output[{i}]: GPU={gpu}, CPU={cpu}, err={err}");
        }
    }

    #[test]
    fn test_q4k_gemv_negative_weights() {
        // Matrix with negative weights
        let m = 2;
        let n = 256;
        let weight: Vec<f32> = (0..m * n)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

        let gpu_result = run_q4k_gemv(&weight, &input, m, n);
        let cpu_result = cpu_q4k_gemv(&weight, &input, m, n);

        let mut max_err = 0.0f32;
        for (&gpu, &cpu) in gpu_result.iter().zip(cpu_result.iter()) {
            let err = (gpu - cpu).abs();
            if err > max_err {
                max_err = err;
            }
        }

        assert!(max_err < 5.0, "Max error with negative weights: {max_err}");
    }
}
