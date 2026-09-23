//! CubeCL ternary GEMV kernel reading from a **block-contiguous** GPU buffer
//! (Issue 650 Phase 2).
//!
//! This is the AoS (array-of-structures) counterpart to the SoA (structure-of-
//! arrays) kernel in [`crate::gemv_ternary_cubecl`]. Both compute the same
//! `output = dequant_ternary(weights) @ input`, but this kernel reads from a
//! single GPU buffer where each 128-weight group stores its scale + pos_bits +
//! neg_bits **contiguously** in one 9-u32 (36-byte) block.
//!
//! # Why a separate kernel
//!
//! The SoA kernel reads from 3 separate GPU buffers (`pos_bits_u32`,
//! `neg_bits_u32`, `group_scale_f32`), requiring 3 independent global-memory
//! transactions per group. The block-contiguous layout co-locates all three
//! components in one cache-line-friendly block, reducing this to 1 transaction.
//!
//! Bench 645 follow-up #5 identified this 3× memory-access overhead as the
//! structural explanation for the 1.89× vs llama.cpp's 9.4× gap — the SoA
//! kernel's tiling cannot fix the memory-access pattern.
//!
//! # Block layout (GPU-friendly, 36 bytes = 9 × u32)
//!
//! ```text
//! u32[0]     = f32 scale bits (pre-decoded from f16)
//! u32[1..5]  = pos bit-plane (4 × u32 = 128 bits)
//! u32[5..9]  = neg bit-plane (4 × u32 = 128 bits)
//! ```
//!
//! This is the GPU-friendly counterpart to `TernaryBlockAoS` (34 bytes, CPU).
//! The 2-byte difference (34→36) is the u32 alignment cost — negligible
//! (1.5% overhead) and worth it for clean GPU addressing.
//!
//! # Dispatch
//!
//! Same geometry as `gemv_ternary_plane_rowtiled8`: `CubeDim::new_1d(256)`,
//! 8 planes per workgroup, 8 rows per plane (`TERNARY_ROWS_PER_PLANE_8`).
//! The only difference from the SoA kernel is the buffer indexing inside the
//! inner loop.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Plane trait needed for plane_sum() resolution
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::gemv_ternary_cubecl::{prepare_block_contiguous_u32, U32_PER_BLOCK_GROUP};
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::{TERNARY_GROUP_SIZE, TERNARY_ROWS_PER_PLANE_8};
use katgpt_core::TernaryGroupWeights;

// ── Handle ──────────────────────────────────────────────────────────

/// GPU handle for block-contiguous ternary weights (Issue 650).
///
/// Single GPU buffer holding all groups packed as 9-u32 blocks:
/// `{f32_scale, [u32;4] pos, [u32;4] neg}` per 128-weight group.
///
/// Contrast with [`crate::TernaryHandle`] which uses 3 separate buffers
/// (SoA layout). This handle uses 1 buffer (AoS layout), eliminating the
/// 3× memory-access overhead.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct TernaryHandleBlockContiguous {
    /// Block-contiguous buffer: `[rows * groups_per_row * 9]` u32 elements.
    /// Each 9-u32 block = `{f32 scale bits, 4 pos words, 4 neg words}`.
    pub block_buf: Handle,
    /// Output dimension (number of rows).
    pub m: usize,
    /// Input dimension (number of columns).
    pub n: usize,
    /// Groups per row (= `n.div_ceil(128)`).
    pub groups_per_row: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryHandleBlockContiguous {
    /// Upload a ternary projection matrix as a block-contiguous GPU buffer.
    ///
    /// Calls `prepare_block_contiguous_u32` to pack the SoA weight data into
    /// 9-u32 blocks (scale + pos + neg per group), then uploads as a single
    /// `Array<u32>` buffer.
    pub fn from_weights(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w: &TernaryGroupWeights,
    ) -> Self {
        let buf = prepare_block_contiguous_u32(w);
        let block_buf = client.create_from_slice(bytemuck::cast_slice(&buf));
        Self {
            block_buf,
            m: w.rows,
            n: w.cols,
            groups_per_row: w.groups_per_row,
        }
    }

    /// Upload from a pre-packed block-contiguous buffer (skip the SoA→AoS
    /// conversion if the caller already has the packed data).
    pub fn from_block_buf(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        block_buf: &[u32],
        rows: usize,
        cols: usize,
    ) -> Self {
        let groups_per_row = cols.div_ceil(TERNARY_GROUP_SIZE as usize);
        debug_assert_eq!(
            block_buf.len(),
            rows * groups_per_row * U32_PER_BLOCK_GROUP,
            "block buffer size mismatch"
        );
        let block_buf = client.create_from_slice(bytemuck::cast_slice(block_buf));
        Self {
            block_buf,
            m: rows,
            n: cols,
            groups_per_row,
        }
    }
}

// ── CubeCL kernel ───────────────────────────────────────────────────

/// Block-contiguous ternary GEMV kernel — row-tiled at width 8 (Issue 650).
///
/// Mirrors `gemv_ternary_plane_rowtiled8` from the SoA path, but reads from a
/// single `block_buf` instead of 3 separate arrays. For each row `r` and word
/// index `w`:
///
/// - group `g = w / 4`, word-in-group `wi = w % 4`
/// - block base `= (r * groups_per_row + g) * 9`
/// - scale `= f32::from_bits(block_buf[base + 0])`
/// - pos word `= block_buf[base + 1 + wi]`
/// - neg word `= block_buf[base + 5 + wi]`
///
/// The scale is shared across all 4 words in a group, so the kernel caches it
/// and only reloads when the group changes (every 4 word iterations).
///
/// # Memory-access benefit
///
/// Each 9-u32 block (36 bytes) fits in one GPU cache line. Reading scale + pos
/// + neg for one group is one cache-line fetch, vs three separate fetches from
/// three different buffers in the SoA layout. This eliminates the 3× overhead
/// identified in Bench 645 follow-up #5.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_block_contiguous_rowtiled8(
    block_buf: &[u32],
    input: &[f32],
    output: &mut [f32],
    groups_per_row: u32,
    n: u32,
    m: u32,
) {
    // Each group = 4 u32 words per bit-plane = 128 columns.
    // words_per_row = groups_per_row * 4.
    let words_per_row = groups_per_row * 4u32;

    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let row_base = plane_id * TERNARY_ROWS_PER_PLANE_8;

    if row_base >= m {
        terminate!();
    }

    // 8 rows per plane — bounds-clamp each to m-1 (sentinel rows contribute 0).
    let r0 = row_base;
    let mut r1 = m - 1u32;
    let mut r2 = m - 1u32;
    let mut r3 = m - 1u32;
    let mut r4 = m - 1u32;
    let mut r5 = m - 1u32;
    let mut r6 = m - 1u32;
    let mut r7 = m - 1u32;
    if row_base + 1u32 < m { r1 = row_base + 1u32; }
    if row_base + 2u32 < m { r2 = row_base + 2u32; }
    if row_base + 3u32 < m { r3 = row_base + 3u32; }
    if row_base + 4u32 < m { r4 = row_base + 4u32; }
    if row_base + 5u32 < m { r5 = row_base + 5u32; }
    if row_base + 6u32 < m { r6 = row_base + 6u32; }
    if row_base + 7u32 < m { r7 = row_base + 7u32; }

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);
    let mut acc4 = f32::new(0.0f32);
    let mut acc5 = f32::new(0.0f32);
    let mut acc6 = f32::new(0.0f32);
    let mut acc7 = f32::new(0.0f32);

    let mut w = lane;
    while w < words_per_row {
        // Group index and word-in-group.
        let g = w >> 2u32; // w / 4
        let wi = w & 3u32; // w % 4
        let col_base = w * 32u32;

        // Block base for each row: (row * groups_per_row + g) * 9.
        let blk0 = (r0 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk1 = (r1 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk2 = (r2 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk3 = (r3 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk4 = (r4 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk5 = (r5 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk6 = (r6 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;
        let blk7 = (r7 * groups_per_row + g) * U32_PER_BLOCK_GROUP_U32;

        // Scale (u32 f32 bits) — shared across all 4 words in this group.
        let s0 = f32::from_bits(block_buf[(blk0) as usize]);
        let s1 = f32::from_bits(block_buf[(blk1) as usize]);
        let s2 = f32::from_bits(block_buf[(blk2) as usize]);
        let s3 = f32::from_bits(block_buf[(blk3) as usize]);
        let s4 = f32::from_bits(block_buf[(blk4) as usize]);
        let s5 = f32::from_bits(block_buf[(blk5) as usize]);
        let s6 = f32::from_bits(block_buf[(blk6) as usize]);
        let s7 = f32::from_bits(block_buf[(blk7) as usize]);

        // Pos word: block_buf[base + 1 + wi].
        let p0 = block_buf[(blk0 + 1u32 + wi) as usize];
        let p1 = block_buf[(blk1 + 1u32 + wi) as usize];
        let p2 = block_buf[(blk2 + 1u32 + wi) as usize];
        let p3 = block_buf[(blk3 + 1u32 + wi) as usize];
        let p4 = block_buf[(blk4 + 1u32 + wi) as usize];
        let p5 = block_buf[(blk5 + 1u32 + wi) as usize];
        let p6 = block_buf[(blk6 + 1u32 + wi) as usize];
        let p7 = block_buf[(blk7 + 1u32 + wi) as usize];

        // Neg word: block_buf[base + 5 + wi].
        let q0 = block_buf[(blk0 + 5u32 + wi) as usize];
        let q1 = block_buf[(blk1 + 5u32 + wi) as usize];
        let q2 = block_buf[(blk2 + 5u32 + wi) as usize];
        let q3 = block_buf[(blk3 + 5u32 + wi) as usize];
        let q4 = block_buf[(blk4 + 5u32 + wi) as usize];
        let q5 = block_buf[(blk5 + 5u32 + wi) as usize];
        let q6 = block_buf[(blk6 + 5u32 + wi) as usize];
        let q7 = block_buf[(blk7 + 5u32 + wi) as usize];

        // Two-accumulator select-form ternary dot (matches SoA rowtiled8).
        let mut a0p = f32::new(0.0f32);
        let mut a0n = f32::new(0.0f32);
        let mut a1p = f32::new(0.0f32);
        let mut a1n = f32::new(0.0f32);
        let mut a2p = f32::new(0.0f32);
        let mut a2n = f32::new(0.0f32);
        let mut a3p = f32::new(0.0f32);
        let mut a3n = f32::new(0.0f32);
        let mut a4p = f32::new(0.0f32);
        let mut a4n = f32::new(0.0f32);
        let mut a5p = f32::new(0.0f32);
        let mut a5n = f32::new(0.0f32);
        let mut a6p = f32::new(0.0f32);
        let mut a6n = f32::new(0.0f32);
        let mut a7p = f32::new(0.0f32);
        let mut a7n = f32::new(0.0f32);

        if col_base + 32u32 <= n {
            // Fast path: no per-bit bounds check (uniform branch, all lanes).
            #[unroll]
            for b in 0u32..32u32 {
                let x = input[(col_base + b) as usize];
                let pb0 = (p0 >> b) & 1u32 != 0u32;
                let nb0 = (q0 >> b) & 1u32 != 0u32;
                let pb1 = (p1 >> b) & 1u32 != 0u32;
                let nb1 = (q1 >> b) & 1u32 != 0u32;
                let pb2 = (p2 >> b) & 1u32 != 0u32;
                let nb2 = (q2 >> b) & 1u32 != 0u32;
                let pb3 = (p3 >> b) & 1u32 != 0u32;
                let nb3 = (q3 >> b) & 1u32 != 0u32;
                let pb4 = (p4 >> b) & 1u32 != 0u32;
                let nb4 = (q4 >> b) & 1u32 != 0u32;
                let pb5 = (p5 >> b) & 1u32 != 0u32;
                let nb5 = (q5 >> b) & 1u32 != 0u32;
                let pb6 = (p6 >> b) & 1u32 != 0u32;
                let nb6 = (q6 >> b) & 1u32 != 0u32;
                let pb7 = (p7 >> b) & 1u32 != 0u32;
                let nb7 = (q7 >> b) & 1u32 != 0u32;
                a0p += select(pb0, x, f32::new(0.0f32));
                a0n += select(nb0, x, f32::new(0.0f32));
                a1p += select(pb1, x, f32::new(0.0f32));
                a1n += select(nb1, x, f32::new(0.0f32));
                a2p += select(pb2, x, f32::new(0.0f32));
                a2n += select(nb2, x, f32::new(0.0f32));
                a3p += select(pb3, x, f32::new(0.0f32));
                a3n += select(nb3, x, f32::new(0.0f32));
                a4p += select(pb4, x, f32::new(0.0f32));
                a4n += select(nb4, x, f32::new(0.0f32));
                a5p += select(pb5, x, f32::new(0.0f32));
                a5n += select(nb5, x, f32::new(0.0f32));
                a6p += select(pb6, x, f32::new(0.0f32));
                a6n += select(nb6, x, f32::new(0.0f32));
                a7p += select(pb7, x, f32::new(0.0f32));
                a7n += select(nb7, x, f32::new(0.0f32));
            }
        } else {
            // Slow path: ragged tail (non-multiple-of-32 n). Per-bit guard.
            #[unroll]
            for b in 0u32..32u32 {
                let col = col_base + b;
                if col < n {
                    let x = input[col as usize];
                    let pb0 = (p0 >> b) & 1u32 != 0u32;
                    let nb0 = (q0 >> b) & 1u32 != 0u32;
                    let pb1 = (p1 >> b) & 1u32 != 0u32;
                    let nb1 = (q1 >> b) & 1u32 != 0u32;
                    let pb2 = (p2 >> b) & 1u32 != 0u32;
                    let nb2 = (q2 >> b) & 1u32 != 0u32;
                    let pb3 = (p3 >> b) & 1u32 != 0u32;
                    let nb3 = (q3 >> b) & 1u32 != 0u32;
                    let pb4 = (p4 >> b) & 1u32 != 0u32;
                    let nb4 = (q4 >> b) & 1u32 != 0u32;
                    let pb5 = (p5 >> b) & 1u32 != 0u32;
                    let nb5 = (q5 >> b) & 1u32 != 0u32;
                    let pb6 = (p6 >> b) & 1u32 != 0u32;
                    let nb6 = (q6 >> b) & 1u32 != 0u32;
                    let pb7 = (p7 >> b) & 1u32 != 0u32;
                    let nb7 = (q7 >> b) & 1u32 != 0u32;
                    a0p += select(pb0, x, f32::new(0.0f32));
                    a0n += select(nb0, x, f32::new(0.0f32));
                    a1p += select(pb1, x, f32::new(0.0f32));
                    a1n += select(nb1, x, f32::new(0.0f32));
                    a2p += select(pb2, x, f32::new(0.0f32));
                    a2n += select(nb2, x, f32::new(0.0f32));
                    a3p += select(pb3, x, f32::new(0.0f32));
                    a3n += select(nb3, x, f32::new(0.0f32));
                    a4p += select(pb4, x, f32::new(0.0f32));
                    a4n += select(nb4, x, f32::new(0.0f32));
                    a5p += select(pb5, x, f32::new(0.0f32));
                    a5n += select(nb5, x, f32::new(0.0f32));
                    a6p += select(pb6, x, f32::new(0.0f32));
                    a6n += select(nb6, x, f32::new(0.0f32));
                    a7p += select(pb7, x, f32::new(0.0f32));
                    a7n += select(nb7, x, f32::new(0.0f32));
                }
            }
        }

        let a0 = a0p - a0n;
        let a1 = a1p - a1n;
        let a2 = a2p - a2n;
        let a3 = a3p - a3n;
        let a4 = a4p - a4n;
        let a5 = a5p - a5n;
        let a6 = a6p - a6n;
        let a7 = a7p - a7n;

        acc0 += a0 * s0;
        acc1 += a1 * s1;
        acc2 += a2 * s2;
        acc3 += a3 * s3;
        acc4 += a4 * s4;
        acc5 += a5 * s5;
        acc6 += a6 * s6;
        acc7 += a7 * s7;

        w += PLANE_DIM;
    }

    let t0 = plane_sum(acc0);
    let t1 = plane_sum(acc1);
    let t2 = plane_sum(acc2);
    let t3 = plane_sum(acc3);
    let t4 = plane_sum(acc4);
    let t5 = plane_sum(acc5);
    let t6 = plane_sum(acc6);
    let t7 = plane_sum(acc7);

    if lane == 0u32 {
        output[r0 as usize] = t0;
        if row_base + 1u32 < m { output[r1 as usize] = t1; }
        if row_base + 2u32 < m { output[r2 as usize] = t2; }
        if row_base + 3u32 < m { output[r3 as usize] = t3; }
        if row_base + 4u32 < m { output[r4 as usize] = t4; }
        if row_base + 5u32 < m { output[r5 as usize] = t5; }
        if row_base + 6u32 < m { output[r6 as usize] = t6; }
        if row_base + 7u32 < m { output[r7 as usize] = t7; }
    }
}

/// U32_PER_BLOCK_GROUP as a compile-time u32 for CubeCL kernel use.
#[cfg(feature = "cubecl_runtime")]
const U32_PER_BLOCK_GROUP_U32: u32 = U32_PER_BLOCK_GROUP as u32;

// ── Launcher ────────────────────────────────────────────────────────

/// Block-contiguous ternary GEMV kernel launcher (Issue 650 Phase 2).
///
/// Wraps the CubeCL kernel with the same dispatch geometry as
/// [`crate::GemvTernaryCubeCL::launch_rowtiled8`], but reads from a single
/// block-contiguous buffer instead of 3 separate arrays.
///
/// # Safety
///
/// - `input_handle` must point to `n` f32 elements
/// - `output_handle` must point to `m` f32 elements
/// - `handle.block_buf` must have been created from the same `client`
/// - `m > 0`
#[cfg(feature = "cubecl_runtime")]
pub struct GemvTernaryBlockContiguousCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvTernaryBlockContiguousCubeCL {
    /// Launch the block-contiguous ternary GEMV.
    ///
    /// Dispatch: `CubeDim::new_1d(256)`, `ceil(m / 64)` workgroups.
    /// Same row-tile width as the SoA `launch_rowtiled8` (8 rows per plane).
    ///
    /// # Safety
    ///
    /// See struct-level docs.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandleBlockContiguous,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE_8; // 64

        let m = handle.m as u32;
        let n = handle.n as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let block_len = handle.m * handle.groups_per_row * U32_PER_BLOCK_GROUP;

        unsafe {
            gemv_ternary_block_contiguous_rowtiled8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.block_buf.clone(), block_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(output_handle, m as usize),
                groups_per_row,
                n,
                m,
            );
        }
    }
}
