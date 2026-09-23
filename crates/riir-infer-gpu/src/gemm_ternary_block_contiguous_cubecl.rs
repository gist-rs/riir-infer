//! Block-contiguous simdgroup-matrix ternary GEMM kernel (Issue 650 Phase 2).
//!
//! This is the block-contiguous variant of [`crate::gemm_ternary_simdgroup_cubecl`].
//! Instead of reading from 3 separate GPU buffers (pos_bits, neg_bits, group_scale),
//! it reads from a single block-contiguous buffer where each 128-weight group is
//! packed into 9 × u32 = 36 bytes.
//!
//! ## Why block-contiguous
//!
//! The 3-buffer layout requires 3 global-memory accesses per K-tile per row.
//! This single-buffer layout requires 1 access per group. For large weight
//! matrices (ffn_down at N=17408), that's a 3× reduction in global-memory
//! accesses — the structural explanation for the 1.89× vs llama.cpp 9.4× gap
//! (Bench 645 follow-up #5).
//!
//! ## Buffer layout (from `prepare_block_contiguous_u32`)
//!
//! Each 128-weight group = 9 × u32:
//! ```text
//! u32[0]     = f32 scale bits
//! u32[1..5]  = pos bit-plane (4 × u32 = 128 bits)
//! u32[5..9]  = neg bit-plane (4 × u32 = 128 bits)
//! ```
//!
//! Block index for (row, group): `block_base = (row * groups_per_row + group) * 9`
//!
//! ## Status
//!
//! **PROTOTYPE** — compiles but NOT GPU-validated. The CPU reference
//! (`block_contiguous_gemm_cpu_ref`) validates the indexing logic. When GPU
//! is available, run G1 (correctness vs CPU ref) + G2 (≥3× speedup).

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::cmma;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::U32_PER_BLOCK_GROUP;

/// The cmma tile dimension — Metal `simdgroup_matrix_8x8` is 8×8×8.
#[cfg(feature = "cubecl_runtime")]
const CMMA_DIM: u32 = 8;

/// u32 words per bit-plane per group (128 bits / 32 = 4).
#[cfg(feature = "cubecl_runtime")]
const U32_PER_PLANE_PER_GROUP: u32 = 4;

/// Block-contiguous simdgroup-matrix ternary GEMM kernel.
///
/// Each 32-thread workgroup (one simdgroup) computes one 8×8 output tile of
/// `output_batch[P × m]`. Identical dispatch to `gemm_ternary_simdgroup` —
/// the ONLY difference is the weight buffer layout.
///
/// See [`crate::gemm_ternary_simdgroup_cubecl`] for the full design rationale
/// (cmma, shared-memory dequant, SWAP dispatch). This file documents only
/// what's different: the weight read pattern.
///
/// # Weight dequant (block-contiguous)
///
/// For weight at (row=r, col=c):
/// 1. Compute group `g = c / 128`
/// 2. Compute block base: `blk = (r * groups_per_row + g) * U32_PER_BLOCK_GROUP`
/// 3. Scale: `scale = f32::from_bits(block_buf[blk])`
/// 4. Local col within group: `lc = c % 128`
/// 5. Pos word: `block_buf[blk + 1 + lc / 32]`, bit `lc % 32`
/// 6. Neg word: `block_buf[blk + 1 + 4 + lc / 32]`, bit `lc % 32`
/// 7. `sign = pos - neg`, weight = `scale * sign`
///
/// This matches `block_contiguous_gemm_cpu_ref` exactly.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_block_contiguous(
    block_buf: &[u32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * CMMA_DIM;
    let base_m = wg_m * CMMA_DIM;

    let acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    let mut tile_w = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x = Shared::<[f32]>::new_slice(64usize);
    let mut result_tile = Shared::<[f32]>::new_slice(64usize);

    let tid = UNIT_POS;
    let e0 = tid * 2u32;
    let e1 = tid * 2u32 + 1u32;
    let row0 = e0 / CMMA_DIM;
    let col0 = e0 % CMMA_DIM;
    let row1 = e1 / CMMA_DIM;
    let col1 = e1 % CMMA_DIM;

    let num_k_tiles = n.div_ceil(CMMA_DIM);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * CMMA_DIM;

        // ── Dequantize weight tile from block-contiguous buffer ──
        //
        // Element 0 of this thread's 2 assigned elements:
        let wr0 = base_m + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            // Block-contiguous read: one base index per (row, group).
            let g0 = wc0 / 128u32;
            let blk_base0 = (wr0 * groups_per_row + g0) * U32_PER_BLOCK_GROUP as u32;
            let scale0 = f32::from_bits(block_buf[blk_base0 as usize]);

            let lc0 = wc0 % 128u32;
            let pos_word_idx0 = blk_base0 + 1u32 + (lc0 / 32u32);
            let neg_word_idx0 = blk_base0 + 1u32 + U32_PER_PLANE_PER_GROUP + (lc0 / 32u32);
            let bit_pos0 = lc0 % 32u32;

            let pos_bit0 = (block_buf[pos_word_idx0 as usize] >> bit_pos0) & 1u32;
            let neg_bit0 = (block_buf[neg_word_idx0 as usize] >> bit_pos0) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let pos_val = select(pos_bit0 != 0u32, one, zero);
            let neg_val = select(neg_bit0 != 0u32, one, zero);
            let sign = pos_val - neg_val;
            tile_w[e0 as usize] = sign * scale0;
        } else {
            tile_w[e0 as usize] = f32::new(0.0f32);
        }

        // Element 1:
        let wr1 = base_m + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let g1 = wc1 / 128u32;
            let blk_base1 = (wr1 * groups_per_row + g1) * U32_PER_BLOCK_GROUP as u32;
            let scale1 = f32::from_bits(block_buf[blk_base1 as usize]);

            let lc1 = wc1 % 128u32;
            let pos_word_idx1 = blk_base1 + 1u32 + (lc1 / 32u32);
            let neg_word_idx1 = blk_base1 + 1u32 + U32_PER_PLANE_PER_GROUP + (lc1 / 32u32);
            let bit_pos1 = lc1 % 32u32;

            let pos_bit1 = (block_buf[pos_word_idx1 as usize] >> bit_pos1) & 1u32;
            let neg_bit1 = (block_buf[neg_word_idx1 as usize] >> bit_pos1) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let pos_val = select(pos_bit1 != 0u32, one, zero);
            let neg_val = select(neg_bit1 != 0u32, one, zero);
            let sign = pos_val - neg_val;
            tile_w[e1 as usize] = sign * scale1;
        } else {
            tile_w[e1 as usize] = f32::new(0.0f32);
        }

        // ── Load input tile (identical to the SoA kernel) ──
        let xr0 = base_p + row0;
        let xk0 = k_base + col0;
        if xr0 < p_tokens && xk0 < n {
            tile_x[e0 as usize] = input_batch[(xr0 * n + xk0) as usize];
        } else {
            tile_x[e0 as usize] = f32::new(0.0f32);
        }

        let xr1 = base_p + row1;
        let xk1 = k_base + col1;
        if xr1 < p_tokens && xk1 < n {
            tile_x[e1 as usize] = input_batch[(xr1 * n + xk1) as usize];
        } else {
            tile_x[e1 as usize] = f32::new(0.0f32);
        }

        sync_cube();

        let mat_a = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A,
            8usize,
            8usize,
            8usize,
            cmma::MatrixLayout::RowMajor,
            &tile_w,
            8,
        );
        let mat_b = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B,
            8usize,
            8usize,
            8usize,
            cmma::MatrixLayout::ColMajor,
            &tile_x,
            8,
        );

        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a, &mat_b, &acc, &acc);

        sync_cube();
        k_tile += 1u32;
    }

    // Store accumulator → staging → global output (identical to SoA kernel).
    //
    // cmma convention: acc[row][col] = Σ_k A[row][k] * B[k][col]
    //   where A = tile_w (RowMajor, weight tile), B = tile_x (ColMajor, input tile)
    // After derivation (see gemm_ternary_simdgroup_cubecl.rs lines 257-264):
    //   acc[row][col] = output(token=base_p+col, row=base_m+row)
    // So the token index comes from col, the m-row comes from row — NOT the
    // other way around. (An earlier draft of this kernel had them swapped,
    // producing a transposed output — caught by the pre-GPU audit 2026-08-13.)
    cmma::store(
        &mut result_tile,
        &acc,
        8,
        cmma::MatrixLayout::RowMajor,
    );

    sync_cube();

    let out_tok0 = base_p + col0;
    let out_row0 = base_m + row0;
    let out_tok1 = base_p + col1;
    let out_row1 = base_m + row1;
    if out_tok0 < p_tokens && out_row0 < m {
        output_batch[(out_tok0 * m + out_row0) as usize] = result_tile[e0 as usize];
    }
    if out_tok1 < p_tokens && out_row1 < m {
        output_batch[(out_tok1 * m + out_row1) as usize] = result_tile[e1 as usize];
    }
}

/// Launcher struct for the block-contiguous simdgroup ternary GEMM.
///
/// Mirrors [`crate::gemm_ternary_simdgroup_cubecl::GemmTernarySimdgroupCubeCL`]
/// but takes a single block-contiguous buffer instead of 3 separate buffers.
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryBlockContiguousCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmTernaryBlockContiguousCubeCL {
    /// Launch the block-contiguous ternary GEMM kernel.
    ///
    /// - `block_buf`: GPU handle from `prepare_block_contiguous_u32` uploaded
    ///   via `client.create_from_slice(bytemuck::cast_slice(&buf))`
    /// - `input_batch`: `[p_tokens, n]` row-major f32
    /// - `output_batch`: `[p_tokens, m]` row-major f32
    /// - Dispatch: `CubeCount::Static(ceil(P/8), ceil(M/8), 1)`, `CubeDim::new_1d(32)`
    ///
    /// # Safety
    ///
    /// CubeCL launch — the caller must ensure buffer sizes match the declared
    /// dimensions. The kernel bounds-checks all accesses, but malformed buffers
    /// can still cause GPU faults.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        block_buf: Handle,
        input_batch: Handle,
        output_batch: Handle,
        m: usize,
        n: usize,
        groups_per_row: usize,
        p_tokens: usize,
    ) {
        let cube_count_p = p_tokens.div_ceil(CMMA_DIM as usize) as u32;
        let cube_count_m = m.div_ceil(CMMA_DIM as usize) as u32;
        let cube_count = CubeCount::Static(cube_count_p, cube_count_m, 1);

        unsafe {
            gemm_ternary_block_contiguous::launch_unchecked::<R>(
                client,
                cube_count,
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(
                    block_buf,
                    m * groups_per_row * U32_PER_BLOCK_GROUP,
                ),
                BufferArg::from_raw_parts(input_batch, p_tokens * n),
                BufferArg::from_raw_parts(output_batch, p_tokens * m),
                groups_per_row as u32,
                n as u32,
                m as u32,
                p_tokens as u32,
            );
        }
    }
}
