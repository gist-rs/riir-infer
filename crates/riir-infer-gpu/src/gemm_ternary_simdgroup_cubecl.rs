//! Simdgroup-matrix ternary GEMM kernel (Issue 641 / Issue 637 T5 follow-up).
//!
//! This is the **real lever** for the 8.38× prefill gap measured in
//! [riir-clippy Bench 010](../../riir-clippy/.benchmarks/010_llamacpp_vs_katgpt_ternary_clippy_tps.md):
//! llama.cpp's `kernel_mul_mm_q2_0_f32` uses Metal `simdgroup_matrix_8x8`
//! cooperative matrix intrinsics, which is a different hardware path from the
//! plane-cooperative kernel in [`crate::gemm_ternary_batched_cubecl`]. This
//! kernel mirrors that approach via CubeCL's `cmma` API.
//!
//! # What changed vs the plane-cooperative kernel
//!
//! [`GemmTernaryBatchedCubeCL`] (the 1.08× kernel from Bench 641) does the
//! ternary dot product in **scalar ALU** inside a plane — each lane holds
//! accumulators and does `select(pos, x, 0) - select(neg, x, 0)` per bit,
//! then `plane_sum()` reduces. The sweep showed this is **occupancy-bound**:
//! past ~8 accumulators the lost parallelism outweighs the loads saved, and
//! no tile reaches 3×. The mechanism is launch-overhead amortization, not
//! weight/ALU amortization.
//!
//! This kernel uses Metal's **cooperative matrix multiply-accumulate unit**
//! (`simdgroup_matrix_8x8<float>`) instead. The 8×8×8 f32 multiply-accumulate
//! runs on a dedicated hardware unit that does 64 FMAs per simdgroup per
//! cycle — far beyond scalar ALU throughput. The ternary weights are
//! **dequantized once per K-tile** into a shared-memory staging buffer, then
//! the cmma unit consumes them at hardware rate.
//!
//! # Design (adapted from [`crate::matmul_swap_ab_cubecl::matmul_swap_ab_cmma_f32`])
//!
//! Each workgroup (32 threads = 1 simdgroup) computes one 8×8 output tile of
//! `output_batch[P × m]`. The K-dimension (weight columns / input features)
//! is tiled into 8-element blocks. For each K-tile:
//!
//! 1. **Dequantize** 8 rows × 8 cols of the ternary weight into `tile_w`
//!    (shared memory, cooperative: each of 32 threads fills 2 of 64 elements).
//!    Per element: extract the pos/neg bit for (row, col), compute
//!    `sign = pos - neg`, multiply by `group_scale[row, col/128]`.
//! 2. **Load** 8 tokens × 8 cols of the input activation into `tile_x`
//!    (shared memory, cooperative, f32 directly — no conversion needed).
//! 3. **cmma::execute** — `acc += tile_w @ tile_x` via the hardware unit.
//!
//! The dequant cost is **amortized across all 64 cmma FMAs** in the tile,
//! which is the structural difference from the plane-cooperative kernel.
//!
//! # Layout
//!
//! - `input_batch`: row-major `[p_tokens, n]` — `input[tok * n + col]`
//! - `output_batch`: row-major `[p_tokens, m]` — `output[tok * m + row]`
//! - Weight: the standard ternary bit-plane format (see [`crate::gemv_ternary_cubecl`]):
//!   - `pos_bits_u32`: `m × blocks64 × 2` u32 elements
//!   - `neg_bits_u32`: `m × blocks64 × 2` u32 elements
//!   - `group_scale_f32`: `m × groups_per_row` f32 elements
//!
//! # Dispatch
//!
//! - `CubeDim::new_1d(32)` — one simdgroup per workgroup (the cmma plane size)
//! - `CubeCount::Static(ceil(P/8), ceil(M/8), 1)` — swapped so the large P-axis
//!   gets X-tiles for better workgroup distribution

#![allow(clippy::too_many_arguments)]

// Issue 771 T1: the smem-staged multi-simdgroup 128×64 family lives in its
// own file (this one is at the 2048-line budget); its launch + toggle hang
// off the same [`GemmTernarySimdgroupCubeCL`] type via an inherent impl in
// the child module (zero lib.rs surface change).
#[cfg(feature = "ternary_gemm_simdgroup")]
#[path = "gemm_ternary_simdgroup_smem_cubecl.rs"]
pub mod smem;

#[cfg(feature = "cubecl_runtime")]
use cubecl::cmma;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::TernaryHandle;

// Issue 655: half::f16 is the CubeCL primitive type for f16 — same type
// cubecl-core implements CubeType/CubePrimitive/Cast for. Must be in scope
// for #[cube] kernels to resolve f16 correctly.
#[cfg(feature = "cubecl_runtime")]
use half::f16;

/// The cmma tile dimension — Metal `simdgroup_matrix_8x8` is 8×8×8.
#[cfg(feature = "cubecl_runtime")]
const CMMA_DIM: u32 = 8;

// ---------------------------------------------------------------------------
// Simdgroup-matrix ternary GEMM kernel
// ---------------------------------------------------------------------------

/// Simdgroup-matrix ternary bit-plane GEMM.
///
/// Computes `output_batch[P × m] = dequant_ternary(weight[m × n]) @ input_batch[P × n]^T`
/// using Metal's `simdgroup_matrix_8x8` cooperative matrix multiply-accumulate
/// (CubeCL `cmma`). Each 32-thread workgroup (one simdgroup) computes one 8×8
/// output tile. See the module docs for the full design.
///
/// # Why cmma vs plane-cooperative
///
/// The plane-cooperative kernel ([`crate::gemm_ternary_batched_cubecl`]) does
/// the dot product in scalar ALU and is occupancy-bound at ~1.08×. This kernel
/// offloads the multiply-accumulate to the hardware cmma unit, which does 64
/// FMAs per cycle per simdgroup. The dequantized weight tile lives in shared
/// memory and is consumed at hardware rate.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    // SWAP: X = P-tile (tokens, large axis), Y = M-tile (rows).
    // Gives better workgroup distribution when M is small (common for attn
    // projections) — mirrors matmul_swap_ab_cmma_f32's proven dispatch.
    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * CMMA_DIM; // first token index this workgroup covers
    let base_m = wg_m * CMMA_DIM; // first row index this workgroup covers

    // CMMA accumulator: 8×8 f32, initialized to 0. Created ONCE before the
    // K-loop; each cmma::execute accumulates into it in-place.
    #[allow(unused_mut)]
    let mut acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    // Shared memory for the 8×8 weight tile (dequantized f32) and input tile.
    // 64 f32 = 256 bytes each. Loaded cooperatively by 32 threads (2 elem/thread).
    //
    // Follow-up: a K_BLOCK=32 variant (dequant 8×32, 4 cmma per sync) would
    // amortize the sync + dequant 4×. Requires shared-memory sub-slicing
    // support in cmma::from_slice that this CubeCL version doesn't expose
    // cleanly — see Bench 645 §"Follow-ups".
    let mut tile_w = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x = Shared::<[f32]>::new_slice(64usize);

    // Shared-memory staging buffer for the cmma store (bounds-safe write path).
    let mut result_tile = Shared::<[f32]>::new_slice(64usize);

    // Per-thread tile-element mapping. 32 threads × 2 elements = 64 (8×8 tile).
    let tid = UNIT_POS;
    let e0 = tid * 2u32;
    let e1 = tid * 2u32 + 1u32;
    let row0 = e0 / CMMA_DIM;
    let col0 = e0 % CMMA_DIM;
    let row1 = e1 / CMMA_DIM;
    let col1 = e1 % CMMA_DIM;

    // K-loop: tile over the n dimension in 8-element blocks.
    let num_k_tiles = n.div_ceil(CMMA_DIM);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * CMMA_DIM;

        // ── Dequantize weight tile: tile_w[r*8 + c] = dequant(W[base_m+r, k_base+c]) ──
        // Out-of-bounds rows/cols are zero-padded (safe for cmma).
        let wr0 = base_m + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let neg_word_idx = pos_word_idx;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[neg_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let pos_val = select(pos_bit != 0u32, one, zero);
            let neg_val = select(neg_bit != 0u32, one, zero);
            let sign = pos_val - neg_val;
            let scale_idx = (wr0 * groups_per_row + (wc0 / 128u32)) as usize;
            let scale = group_scale_f32[scale_idx];
            tile_w[e0 as usize] = sign * scale;
        } else {
            tile_w[e0 as usize] = f32::new(0.0f32);
        }

        let wr1 = base_m + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let neg_word_idx = pos_word_idx;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[neg_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let pos_val = select(pos_bit != 0u32, one, zero);
            let neg_val = select(neg_bit != 0u32, one, zero);
            let sign = pos_val - neg_val;
            let scale_idx = (wr1 * groups_per_row + (wc1 / 128u32)) as usize;
            let scale = group_scale_f32[scale_idx];
            tile_w[e1 as usize] = sign * scale;
        } else {
            tile_w[e1 as usize] = f32::new(0.0f32);
        }

        // ── Load input tile: tile_x[tok*8 + col] = input[(base_p+tok) * n + (k_base+col)] ──
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

    // Store the 8×8 accumulator to the shared-memory staging buffer, then
    // conditionally copy valid elements to the global output. Same bounds-safe
    // pattern as matmul_swap_ab_cmma_f32 — cmma::store writes the full 8×8
    // unconditionally, so we stage + bounds-check.
    cmma::store(
        &mut result_tile,
        &acc,
        8,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();

    // Each thread copies its 2 elements from staging to the output.
    // result_tile[row*8 + col] = acc[row][col]
    //   = dot(W[base_m+row], X[base_p+col])
    //   = output for token (base_p+col), row (base_m+row).
    // Output layout is [P × m] row-major: output[token * m + row].
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

// ---------------------------------------------------------------------------
// 8×32 output-tile variant — amortizes weight dequant 4×
// ---------------------------------------------------------------------------

/// The number of cmma sub-tiles in the P (token) dimension per workgroup.
/// Each workgroup computes an 8-row × (8×BN_SUB) -token output tile.
const BN_SUB: u32 = 4; // 8 × 4 = 32-token output width

/// Simdgroup-matrix ternary GEMM with 8×32 output tiling.
///
/// Each workgroup computes 8 rows × 32 tokens of output. The weight tile
/// (8×8 per K-step) is dequantized **once** and reused across 4 cmma
/// executes — amortizing the dequant cost (the binding constraint per Bench
/// 645) over 4× more output elements. Total dequant work drops 4× vs the
/// 8×8 variant because there are 4× fewer workgroups in the P-dimension.
///
/// Requires `p_tokens >= 8` (ideally `p_tokens >= 32` for full utilization).
/// For `p_tokens < 32`, the extra sub-tiles are zero-padded (correct but
/// wasteful — the 8×8 variant is preferred for small P).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup_8x32(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * (CMMA_DIM * BN_SUB); // first token index (stride = 32)
    let base_m = wg_m * CMMA_DIM; // first row index

    // 4 accumulators — one per 8-token sub-tile.
    #[allow(unused_mut)]
    let mut acc0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );

    // Shared memory — 4 separate input sub-tiles + 4 staging tiles + 1 weight tile.
    // CubeCL's cmma::from_slice reads from the START of a Shared allocation (no
    // offset param), so we use separate allocations rather than subslicing.
    // Total: 9 × 64 f32 = 576 f32 = 2304 bytes — well within Metal's 32 KB.
    let mut tile_w = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x0 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x1 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x2 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x3 = Shared::<[f32]>::new_slice(64usize);
    let mut result0 = Shared::<[f32]>::new_slice(64usize);
    let mut result1 = Shared::<[f32]>::new_slice(64usize);
    let mut result2 = Shared::<[f32]>::new_slice(64usize);
    let mut result3 = Shared::<[f32]>::new_slice(64usize);

    // Per-thread element mapping: 32 threads × 2 elements = 64 per 8×8 tile.
    // Element e maps to (row=e/8, col=e%8) in the tile — matches the 8×8 kernel.
    let tid = UNIT_POS;
    let e0 = tid * 2u32;
    let e1 = tid * 2u32 + 1u32;
    let row0 = e0 / CMMA_DIM;
    let col0 = e0 % CMMA_DIM;
    let row1 = e1 / CMMA_DIM;
    let col1 = e1 % CMMA_DIM;

    // K-loop: tile over the n dimension in 8-element blocks.
    let num_k_tiles = n.div_ceil(CMMA_DIM);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * CMMA_DIM;

        // ── Dequantize weight tile (ONCE per K-step, reused 4×) ──
        // tile_w[row*8 + col] = dequant(W[base_m+row, k_base+col]).
        // RowMajor layout matches the A-matrix in cmma::from_slice.
        let wr0 = base_m + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr0 * groups_per_row + (wc0 / 128u32)) as usize];
            tile_w[e0 as usize] = sign * scale;
        } else {
            tile_w[e0 as usize] = f32::new(0.0f32);
        }

        let wr1 = base_m + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr1 * groups_per_row + (wc1 / 128u32)) as usize];
            tile_w[e1 as usize] = sign * scale;
        } else {
            tile_w[e1 as usize] = f32::new(0.0f32);
        }

        // ── Load 4 input sub-tiles cooperatively ──
        // B-matrix is ColMajor: B[k][tok] at index tok*8 + k.
        // With e = tok*8 + k: tok = e/8 = row, k = e%8 = col (same as 8×8 kernel).
        // So tile_xN[e] = input[(base_p + N*8 + row) * n + (k_base + col)].
        let tok0_s0 = base_p + row0;
        let tok0_s1 = base_p + CMMA_DIM + row0;
        let tok0_s2 = base_p + 2u32 * CMMA_DIM + row0;
        let tok0_s3 = base_p + 3u32 * CMMA_DIM + row0;
        let xk0 = k_base + col0;
        if tok0_s0 < p_tokens && xk0 < n {
            tile_x0[e0 as usize] = input_batch[(tok0_s0 * n + xk0) as usize];
        } else {
            tile_x0[e0 as usize] = f32::new(0.0f32);
        }
        if tok0_s1 < p_tokens && xk0 < n {
            tile_x1[e0 as usize] = input_batch[(tok0_s1 * n + xk0) as usize];
        } else {
            tile_x1[e0 as usize] = f32::new(0.0f32);
        }
        if tok0_s2 < p_tokens && xk0 < n {
            tile_x2[e0 as usize] = input_batch[(tok0_s2 * n + xk0) as usize];
        } else {
            tile_x2[e0 as usize] = f32::new(0.0f32);
        }
        if tok0_s3 < p_tokens && xk0 < n {
            tile_x3[e0 as usize] = input_batch[(tok0_s3 * n + xk0) as usize];
        } else {
            tile_x3[e0 as usize] = f32::new(0.0f32);
        }

        let tok1_s0 = base_p + row1;
        let tok1_s1 = base_p + CMMA_DIM + row1;
        let tok1_s2 = base_p + 2u32 * CMMA_DIM + row1;
        let tok1_s3 = base_p + 3u32 * CMMA_DIM + row1;
        let xk1 = k_base + col1;
        if tok1_s0 < p_tokens && xk1 < n {
            tile_x0[e1 as usize] = input_batch[(tok1_s0 * n + xk1) as usize];
        } else {
            tile_x0[e1 as usize] = f32::new(0.0f32);
        }
        if tok1_s1 < p_tokens && xk1 < n {
            tile_x1[e1 as usize] = input_batch[(tok1_s1 * n + xk1) as usize];
        } else {
            tile_x1[e1 as usize] = f32::new(0.0f32);
        }
        if tok1_s2 < p_tokens && xk1 < n {
            tile_x2[e1 as usize] = input_batch[(tok1_s2 * n + xk1) as usize];
        } else {
            tile_x2[e1 as usize] = f32::new(0.0f32);
        }
        if tok1_s3 < p_tokens && xk1 < n {
            tile_x3[e1 as usize] = input_batch[(tok1_s3 * n + xk1) as usize];
        } else {
            tile_x3[e1 as usize] = f32::new(0.0f32);
        }

        sync_cube(); // weight + all input tiles ready

        // mat_a is the same for all 4 sub-tiles (same weight, same K-slice).
        let mat_a = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor, &tile_w, 8,
        );

        // 4 cmma executes — one per sub-tile accumulator.
        let mat_b0 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x0, 8,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a, &mat_b0, &acc0, &acc0);

        let mat_b1 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x1, 8,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a, &mat_b1, &acc1, &acc1);

        let mat_b2 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x2, 8,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a, &mat_b2, &acc2, &acc2);

        let mat_b3 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x3, 8,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a, &mat_b3, &acc3, &acc3);

        sync_cube(); // safe to overwrite shared memory
        k_tile += 1u32;
    }

    // Store 4 accumulators to staging, then bounds-checked copy to output.
    cmma::store(&mut result0, &acc0, 8, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut result1, &acc1, 8, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut result2, &acc2, 8, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut result3, &acc3, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();

    // Each thread copies its 2 elements per sub-tile from staging to output.
    // resultN[row*8 + col] = accN[row][col] = output for token (base_p+N*8+col), row (base_m+row).
    // Output layout is [P × m] row-major: output[token * m + row].
    let out_row0 = base_m + row0;
    let out_row1 = base_m + row1;

    let out_tok0_s0 = base_p + col0;
    if out_tok0_s0 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s0 * m + out_row0) as usize] = result0[e0 as usize];
    }
    let out_tok0_s1 = base_p + CMMA_DIM + col0;
    if out_tok0_s1 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s1 * m + out_row0) as usize] = result1[e0 as usize];
    }
    let out_tok0_s2 = base_p + 2u32 * CMMA_DIM + col0;
    if out_tok0_s2 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s2 * m + out_row0) as usize] = result2[e0 as usize];
    }
    let out_tok0_s3 = base_p + 3u32 * CMMA_DIM + col0;
    if out_tok0_s3 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s3 * m + out_row0) as usize] = result3[e0 as usize];
    }

    let out_tok1_s0 = base_p + col1;
    if out_tok1_s0 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s0 * m + out_row1) as usize] = result0[e1 as usize];
    }
    let out_tok1_s1 = base_p + CMMA_DIM + col1;
    if out_tok1_s1 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s1 * m + out_row1) as usize] = result1[e1 as usize];
    }
    let out_tok1_s2 = base_p + 2u32 * CMMA_DIM + col1;
    if out_tok1_s2 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s2 * m + out_row1) as usize] = result2[e1 as usize];
    }
    let out_tok1_s3 = base_p + 3u32 * CMMA_DIM + col1;
    if out_tok1_s3 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s3 * m + out_row1) as usize] = result3[e1 as usize];
    }
}

// ---------------------------------------------------------------------------
// Issue 768: 32×32 output-tile variant — the input-reuse lever.
//
// The Bench 774 phase decomposition measured the 8×32 kernel's #1 cost as
// the INPUT activation global loads (42–63% marginal at every production
// shape): every input element is re-loaded once per M-row-tile workgroup —
// M/8 = 2176 re-loads at the FFN shape = 91.2 GB/GEMM @ p=2048, which alone
// accounts for essentially the whole kernel wall at ~400 GB/s. Every prior
// variant widened P (8×32, 8×64) or restructured K (K_BLOCK, scale-deferred);
// none increased M-rows-per-workgroup.
//
// This kernel computes a 32(M) × 32(P) output tile: the input tile is loaded
// once per K-step and reused across 4 M-subtiles (input traffic ÷4), and the
// weight tile is dequantized once and reused across 4 P-subtiles (same as
// 8×32 — but 4× fewer workgroups re-dequant the same weights). Per-element
// accumulation order is IDENTICAL to the 8×32 kernel (same k-tile sequence,
// same A/B bits per execute), so outputs are bit-identical by construction.
// ---------------------------------------------------------------------------

/// Simdgroup-matrix ternary GEMM with a 32×32 output tile (Issue 768).
///
/// 16 accumulators (4 M-subtiles × 4 P-subtiles, each 8×8). Best for
/// `m >= 64` and `p_tokens >= 32`; ragged shapes are zero-padded (correct,
/// wasteful — the 8×32 kernel is preferred for small m).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup_32x32(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * (CMMA_DIM * BN_SUB); // token stride = 32
    let base_m = wg_m * (CMMA_DIM * 4u32); // row stride = 32

    // 16 accumulators — one per (M-subtile, P-subtile) pair.
    #[allow(unused_mut)]
    let mut acc_0_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_0_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_0_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_0_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_1_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_1_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_1_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_1_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_2_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_2_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_2_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_2_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_3_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_3_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_3_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc_3_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );

    // Shared memory — 4 weight sub-tiles + 4 input sub-tiles (64 f32 each)
    // + ONE result staging tile reused 16× in the epilogue (sync-separated).
    let mut tile_w0 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_w1 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_w2 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_w3 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x0 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x1 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x2 = Shared::<[f32]>::new_slice(64usize);
    let mut tile_x3 = Shared::<[f32]>::new_slice(64usize);
    let mut result = Shared::<[f32]>::new_slice(64usize);

    // Per-thread element mapping (same as 8×8/8×32): 32 threads × 2 elements
    // = 64 per 8×8 tile. Element e ↦ (row=e/8, col=e%8).
    let tid = UNIT_POS;
    let e0 = tid * 2u32;
    let e1 = tid * 2u32 + 1u32;
    let row0 = e0 / CMMA_DIM;
    let col0 = e0 % CMMA_DIM;
    let row1 = e1 / CMMA_DIM;
    let col1 = e1 % CMMA_DIM;

    // K-loop: tile over the n dimension in 8-element blocks.
    let num_k_tiles = n.div_ceil(CMMA_DIM);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * CMMA_DIM;

        // ── Dequantize 4 weight sub-tiles (rows base_m + wi*8 + r) ──
        // tile_w0 (wi=0)
        let wr0 = base_m + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr0 * groups_per_row + (wc0 / 128u32)) as usize];
            tile_w0[e0 as usize] = sign * scale;
        } else {
            tile_w0[e0 as usize] = f32::new(0.0f32);
        }
        let wr1 = base_m + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr1 * groups_per_row + (wc1 / 128u32)) as usize];
            tile_w0[e1 as usize] = sign * scale;
        } else {
            tile_w0[e1 as usize] = f32::new(0.0f32);
        }

        // tile_w1 (wi=1)
        let wr0 = base_m + CMMA_DIM + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr0 * groups_per_row + (wc0 / 128u32)) as usize];
            tile_w1[e0 as usize] = sign * scale;
        } else {
            tile_w1[e0 as usize] = f32::new(0.0f32);
        }
        let wr1 = base_m + CMMA_DIM + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr1 * groups_per_row + (wc1 / 128u32)) as usize];
            tile_w1[e1 as usize] = sign * scale;
        } else {
            tile_w1[e1 as usize] = f32::new(0.0f32);
        }

        // tile_w2 (wi=2)
        let wr0 = base_m + 2u32 * CMMA_DIM + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr0 * groups_per_row + (wc0 / 128u32)) as usize];
            tile_w2[e0 as usize] = sign * scale;
        } else {
            tile_w2[e0 as usize] = f32::new(0.0f32);
        }
        let wr1 = base_m + 2u32 * CMMA_DIM + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr1 * groups_per_row + (wc1 / 128u32)) as usize];
            tile_w2[e1 as usize] = sign * scale;
        } else {
            tile_w2[e1 as usize] = f32::new(0.0f32);
        }

        // tile_w3 (wi=3)
        let wr0 = base_m + 3u32 * CMMA_DIM + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr0 * groups_per_row + (wc0 / 128u32)) as usize];
            tile_w3[e0 as usize] = sign * scale;
        } else {
            tile_w3[e0 as usize] = f32::new(0.0f32);
        }
        let wr1 = base_m + 3u32 * CMMA_DIM + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr1 * groups_per_row + (wc1 / 128u32)) as usize];
            tile_w3[e1 as usize] = sign * scale;
        } else {
            tile_w3[e1 as usize] = f32::new(0.0f32);
        }

        // ── Load 4 input sub-tiles (tokens base_p + pi*8 + r) ──
        // B-matrix ColMajor: tile_x[e] = input[(base_p + pi*8 + row) * n + (k_base + col)].
        let xr0 = base_p + row0;
        let xk0 = k_base + col0;
        if xr0 < p_tokens && xk0 < n {
            tile_x0[e0 as usize] = input_batch[(xr0 * n + xk0) as usize];
        } else {
            tile_x0[e0 as usize] = f32::new(0.0f32);
        }
        let xr1 = base_p + row1;
        let xk1 = k_base + col1;
        if xr1 < p_tokens && xk1 < n {
            tile_x0[e1 as usize] = input_batch[(xr1 * n + xk1) as usize];
        } else {
            tile_x0[e1 as usize] = f32::new(0.0f32);
        }

        let xr0 = base_p + CMMA_DIM + row0;
        let xk0 = k_base + col0;
        if xr0 < p_tokens && xk0 < n {
            tile_x1[e0 as usize] = input_batch[(xr0 * n + xk0) as usize];
        } else {
            tile_x1[e0 as usize] = f32::new(0.0f32);
        }
        let xr1 = base_p + CMMA_DIM + row1;
        let xk1 = k_base + col1;
        if xr1 < p_tokens && xk1 < n {
            tile_x1[e1 as usize] = input_batch[(xr1 * n + xk1) as usize];
        } else {
            tile_x1[e1 as usize] = f32::new(0.0f32);
        }

        let xr0 = base_p + 2u32 * CMMA_DIM + row0;
        let xk0 = k_base + col0;
        if xr0 < p_tokens && xk0 < n {
            tile_x2[e0 as usize] = input_batch[(xr0 * n + xk0) as usize];
        } else {
            tile_x2[e0 as usize] = f32::new(0.0f32);
        }
        let xr1 = base_p + 2u32 * CMMA_DIM + row1;
        let xk1 = k_base + col1;
        if xr1 < p_tokens && xk1 < n {
            tile_x2[e1 as usize] = input_batch[(xr1 * n + xk1) as usize];
        } else {
            tile_x2[e1 as usize] = f32::new(0.0f32);
        }

        let xr0 = base_p + 3u32 * CMMA_DIM + row0;
        let xk0 = k_base + col0;
        if xr0 < p_tokens && xk0 < n {
            tile_x3[e0 as usize] = input_batch[(xr0 * n + xk0) as usize];
        } else {
            tile_x3[e0 as usize] = f32::new(0.0f32);
        }
        let xr1 = base_p + 3u32 * CMMA_DIM + row1;
        let xk1 = k_base + col1;
        if xr1 < p_tokens && xk1 < n {
            tile_x3[e1 as usize] = input_batch[(xr1 * n + xk1) as usize];
        } else {
            tile_x3[e1 as usize] = f32::new(0.0f32);
        }

        sync_cube(); // all tiles ready

        // ── 8 fragment loads + 16 executes (every A × every B) ──
        let mat_a0 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor, &tile_w0, 8,
        );
        let mat_a1 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor, &tile_w1, 8,
        );
        let mat_a2 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor, &tile_w2, 8,
        );
        let mat_a3 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor, &tile_w3, 8,
        );
        let mat_b0 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x0, 8,
        );
        let mat_b1 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x1, 8,
        );
        let mat_b2 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x2, 8,
        );
        let mat_b3 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x3, 8,
        );

        // NOTE: execute ORDER within a k_tile does not matter (each writes a
        // distinct accumulator); per-accumulator order across k_tiles is the
        // same sequence as the 8×32 kernel → bit-identical outputs.
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a0, &mat_b0, &acc_0_0, &acc_0_0);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a0, &mat_b1, &acc_0_1, &acc_0_1);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a0, &mat_b2, &acc_0_2, &acc_0_2);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a0, &mat_b3, &acc_0_3, &acc_0_3);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a1, &mat_b0, &acc_1_0, &acc_1_0);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a1, &mat_b1, &acc_1_1, &acc_1_1);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a1, &mat_b2, &acc_1_2, &acc_1_2);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a1, &mat_b3, &acc_1_3, &acc_1_3);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a2, &mat_b0, &acc_2_0, &acc_2_0);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a2, &mat_b1, &acc_2_1, &acc_2_1);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a2, &mat_b2, &acc_2_2, &acc_2_2);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a2, &mat_b3, &acc_2_3, &acc_2_3);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a3, &mat_b0, &acc_3_0, &acc_3_0);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a3, &mat_b1, &acc_3_1, &acc_3_1);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a3, &mat_b2, &acc_3_2, &acc_3_2);
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_a3, &mat_b3, &acc_3_3, &acc_3_3);

        sync_cube(); // safe to overwrite shared memory
        k_tile += 1u32;
    }

    // ── Epilogue: 16 (store → sync → bounds-checked copy) cycles over ONE
    // staging tile (sync-separated; runs once per workgroup — negligible).
    // result[r*8 + c] = acc[r][c]: output token (base_p + pi*8 + c), row
    // (base_m + wi*8 + r). Output layout [P × m]: output[token * m + row].
    let out_row0 = base_m + row0;
    let out_row1 = base_m + row1;

    // wi=0
    cmma::store(&mut result, &acc_0_0, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + col0) < p_tokens && out_row0 < m {
        output_batch[((base_p + col0) * m + out_row0) as usize] = result[e0 as usize];
    }
    if (base_p + col1) < p_tokens && out_row1 < m {
        output_batch[((base_p + col1) * m + out_row1) as usize] = result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_0_1, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + CMMA_DIM + col0) < p_tokens && out_row0 < m {
        output_batch[((base_p + CMMA_DIM + col0) * m + out_row0) as usize] = result[e0 as usize];
    }
    if (base_p + CMMA_DIM + col1) < p_tokens && out_row1 < m {
        output_batch[((base_p + CMMA_DIM + col1) * m + out_row1) as usize] = result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_0_2, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 2u32 * CMMA_DIM + col0) < p_tokens && out_row0 < m {
        output_batch[((base_p + 2u32 * CMMA_DIM + col0) * m + out_row0) as usize] =
            result[e0 as usize];
    }
    if (base_p + 2u32 * CMMA_DIM + col1) < p_tokens && out_row1 < m {
        output_batch[((base_p + 2u32 * CMMA_DIM + col1) * m + out_row1) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_0_3, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 3u32 * CMMA_DIM + col0) < p_tokens && out_row0 < m {
        output_batch[((base_p + 3u32 * CMMA_DIM + col0) * m + out_row0) as usize] =
            result[e0 as usize];
    }
    if (base_p + 3u32 * CMMA_DIM + col1) < p_tokens && out_row1 < m {
        output_batch[((base_p + 3u32 * CMMA_DIM + col1) * m + out_row1) as usize] =
            result[e1 as usize];
    }
    sync_cube();

    // wi=1
    cmma::store(&mut result, &acc_1_0, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + col0) < p_tokens && (out_row0 + CMMA_DIM) < m {
        output_batch[((base_p + col0) * m + out_row0 + CMMA_DIM) as usize] = result[e0 as usize];
    }
    if (base_p + col1) < p_tokens && (out_row1 + CMMA_DIM) < m {
        output_batch[((base_p + col1) * m + out_row1 + CMMA_DIM) as usize] = result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_1_1, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + CMMA_DIM + col0) < p_tokens && (out_row0 + CMMA_DIM) < m {
        output_batch[((base_p + CMMA_DIM + col0) * m + out_row0 + CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + CMMA_DIM + col1) < p_tokens && (out_row1 + CMMA_DIM) < m {
        output_batch[((base_p + CMMA_DIM + col1) * m + out_row1 + CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_1_2, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 2u32 * CMMA_DIM + col0) < p_tokens && (out_row0 + CMMA_DIM) < m {
        output_batch[((base_p + 2u32 * CMMA_DIM + col0) * m + out_row0 + CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + 2u32 * CMMA_DIM + col1) < p_tokens && (out_row1 + CMMA_DIM) < m {
        output_batch[((base_p + 2u32 * CMMA_DIM + col1) * m + out_row1 + CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_1_3, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 3u32 * CMMA_DIM + col0) < p_tokens && (out_row0 + CMMA_DIM) < m {
        output_batch[((base_p + 3u32 * CMMA_DIM + col0) * m + out_row0 + CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + 3u32 * CMMA_DIM + col1) < p_tokens && (out_row1 + CMMA_DIM) < m {
        output_batch[((base_p + 3u32 * CMMA_DIM + col1) * m + out_row1 + CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();

    // wi=2
    cmma::store(&mut result, &acc_2_0, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + col0) < p_tokens && (out_row0 + 2u32 * CMMA_DIM) < m {
        output_batch[((base_p + col0) * m + out_row0 + 2u32 * CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + col1) < p_tokens && (out_row1 + 2u32 * CMMA_DIM) < m {
        output_batch[((base_p + col1) * m + out_row1 + 2u32 * CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_2_1, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + CMMA_DIM + col0) < p_tokens && (out_row0 + 2u32 * CMMA_DIM) < m {
        output_batch[((base_p + CMMA_DIM + col0) * m + out_row0 + 2u32 * CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + CMMA_DIM + col1) < p_tokens && (out_row1 + 2u32 * CMMA_DIM) < m {
        output_batch[((base_p + CMMA_DIM + col1) * m + out_row1 + 2u32 * CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_2_2, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 2u32 * CMMA_DIM + col0) < p_tokens && (out_row0 + 2u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 2u32 * CMMA_DIM + col0) * m + out_row0 + 2u32 * CMMA_DIM) as usize
        ] = result[e0 as usize];
    }
    if (base_p + 2u32 * CMMA_DIM + col1) < p_tokens && (out_row1 + 2u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 2u32 * CMMA_DIM + col1) * m + out_row1 + 2u32 * CMMA_DIM) as usize
        ] = result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_2_3, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 3u32 * CMMA_DIM + col0) < p_tokens && (out_row0 + 2u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 3u32 * CMMA_DIM + col0) * m + out_row0 + 2u32 * CMMA_DIM) as usize
        ] = result[e0 as usize];
    }
    if (base_p + 3u32 * CMMA_DIM + col1) < p_tokens && (out_row1 + 2u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 3u32 * CMMA_DIM + col1) * m + out_row1 + 2u32 * CMMA_DIM) as usize
        ] = result[e1 as usize];
    }
    sync_cube();

    // wi=3
    cmma::store(&mut result, &acc_3_0, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + col0) < p_tokens && (out_row0 + 3u32 * CMMA_DIM) < m {
        output_batch[((base_p + col0) * m + out_row0 + 3u32 * CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + col1) < p_tokens && (out_row1 + 3u32 * CMMA_DIM) < m {
        output_batch[((base_p + col1) * m + out_row1 + 3u32 * CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_3_1, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + CMMA_DIM + col0) < p_tokens && (out_row0 + 3u32 * CMMA_DIM) < m {
        output_batch[((base_p + CMMA_DIM + col0) * m + out_row0 + 3u32 * CMMA_DIM) as usize] =
            result[e0 as usize];
    }
    if (base_p + CMMA_DIM + col1) < p_tokens && (out_row1 + 3u32 * CMMA_DIM) < m {
        output_batch[((base_p + CMMA_DIM + col1) * m + out_row1 + 3u32 * CMMA_DIM) as usize] =
            result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_3_2, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 2u32 * CMMA_DIM + col0) < p_tokens && (out_row0 + 3u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 2u32 * CMMA_DIM + col0) * m + out_row0 + 3u32 * CMMA_DIM) as usize
        ] = result[e0 as usize];
    }
    if (base_p + 2u32 * CMMA_DIM + col1) < p_tokens && (out_row1 + 3u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 2u32 * CMMA_DIM + col1) * m + out_row1 + 3u32 * CMMA_DIM) as usize
        ] = result[e1 as usize];
    }
    sync_cube();
    cmma::store(&mut result, &acc_3_3, 8, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if (base_p + 3u32 * CMMA_DIM + col0) < p_tokens && (out_row0 + 3u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 3u32 * CMMA_DIM + col0) * m + out_row0 + 3u32 * CMMA_DIM) as usize
        ] = result[e0 as usize];
    }
    if (base_p + 3u32 * CMMA_DIM + col1) < p_tokens && (out_row1 + 3u32 * CMMA_DIM) < m {
        output_batch[
            ((base_p + 3u32 * CMMA_DIM + col1) * m + out_row1 + 3u32 * CMMA_DIM) as usize
        ] = result[e1 as usize];
    }
}

// ---------------------------------------------------------------------------
// Issue 655: F16 cmma variants — dequant weights to f16, use f16 input tiles,
// mixed-precision cmma <f16, f16, f32>. Metal's simdgroup_matrix_8x8<half>
// has 2× instruction throughput vs <float>.
// ---------------------------------------------------------------------------

/// Simdgroup-matrix ternary GEMM with **f16 input tiles** (Issue 655).
///
/// Identical structure to `gemm_ternary_simdgroup` but dequantizes the weight
/// tile and loads the input tile as `f16` instead of `f32`. The cmma uses
/// `<f16, f16, f32>` mixed precision — f16 inputs, f32 accumulator. On Apple
/// Silicon, `simdgroup_matrix_8x8<half>` has 2× instruction issue throughput
/// vs `<float>`, so this variant aims for ~2× cmma-bound speedup.
///
/// The weight values are exact in f16 (ternary sign × f16 group scale). The
/// input activation is truncated from f32 to f16 — the precision cost is the
/// G1 gate's concern.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup_f16(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * CMMA_DIM;
    let base_m = wg_m * CMMA_DIM;

    // f32 accumulator — no precision loss in the accumulation.
    #[allow(unused_mut)]
    let mut acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    // f16 shared-memory tiles — half the bandwidth of the f32 variants.
    let mut tile_w = Shared::<[f16]>::new_slice(64usize);
    let mut tile_x = Shared::<[f16]>::new_slice(64usize);
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

        // Dequantize weight tile to f16: sign × f16 group scale.
        let wr0 = base_m + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr0 * groups_per_row + (wc0 / 128u32)) as usize];
            tile_w[e0 as usize] = f16::cast_from(sign * scale);
        } else {
            tile_w[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        let wr1 = base_m + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            let scale = group_scale_f32[(wr1 * groups_per_row + (wc1 / 128u32)) as usize];
            tile_w[e1 as usize] = f16::cast_from(sign * scale);
        } else {
            tile_w[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        // Load input tile to f16 (truncate from f32).
        let xr0 = base_p + row0;
        let xk0 = k_base + col0;
        if xr0 < p_tokens && xk0 < n {
            tile_x[e0 as usize] = f16::cast_from(input_batch[(xr0 * n + xk0) as usize]);
        } else {
            tile_x[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        let xr1 = base_p + row1;
        let xk1 = k_base + col1;
        if xr1 < p_tokens && xk1 < n {
            tile_x[e1 as usize] = f16::cast_from(input_batch[(xr1 * n + xk1) as usize]);
        } else {
            tile_x[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        sync_cube();

        // Mixed-precision cmma: f16 inputs, f32 accumulator.
        let mat_a = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            8usize,
            8usize,
            8usize,
            cmma::MatrixLayout::RowMajor,
            &tile_w,
            8,
        );
        let mat_b = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            8usize,
            8usize,
            8usize,
            cmma::MatrixLayout::ColMajor,
            &tile_x,
            8,
        );

        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&mat_a, &mat_b, &acc, &acc);

        sync_cube();
        k_tile += 1u32;
    }

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

// ---------------------------------------------------------------------------
// Issue 767: scale-deferred simdgroup GEMM — the f16 {-1,0,1} weight rides
// the matrix unit UNSCALED; group scales apply to the f32 accumulator at
// group boundaries (every 128 K-elements = 16 K-tiles). The per-element
// scale-mul — the measured ~4.5 cycles/element dequant binder (Bench 645
// §"Why it doesn't reach 3×", follow-up #4) — leaves the inner loop entirely.
// The weight side is EXACT in f16 (ternary signs; no f16 rounding, no
// subnormal-flush hazard), so the failure shape of the refuted in-kernel-
// scaled f16 variant (Bench 760 T3, max_rel 70 from `f16(sign*scale)`) is
// structurally impossible here. The activation side is f16 (10-bit
// mantissa) — the remaining G1 risk, gated FIRST at P=128 (Issue 767 T2).
//
// The math (algebraically identical, fp-rounding differs):
//   scaled:   out[j] = Σ_g Σ_{k∈g} (w[k,j]·s_g)·x[k]   // per-element mul
//   deferred: out[j] = Σ_g s_g·(Σ_{k∈g} w[k,j]·x[k])  // one mul per group-partial
//
// The 4090's cmma_i8 kernel (Bench 709) shipped this structure on
// CoopMma/Vulkan; this is the Metal simdgroup twin.
// ---------------------------------------------------------------------------

/// K-elements per quant group — MUST match the `wc / 128u32` scale indexing
/// of the scaled variants and `TernaryHandle::groups_per_row` semantics
/// (`groups_per_row = ceil(n / 128)`).
const K_PER_GROUP: u32 = 128;
/// K-tiles (of 8) per quant group.
const KTILES_PER_GROUP: u32 = K_PER_GROUP / CMMA_DIM;

/// Scale-deferred simdgroup ternary GEMM, 8×32 output tile (Issue 767 T1).
///
/// Same launch surface + tile shape as [`gemm_ternary_simdgroup_8x32`], so
/// the A/B (`bench_767_scale_deferred_g1`) compares the implementations as
/// shipped. Structural differences:
/// - weight tile dequantizes to **f16 signs only** ({-1,0,1}, exact);
/// - input tiles cast to f16 (mixed `<f16,f16,f32>` cmma — 2× issue rate);
/// - the 4 accumulators hold **unscaled group partials**; at every group
///   boundary (16 K-tiles) each thread merges its 2 elements × 4 sub-tiles
///   into per-thread f32 totals with ONE scale load (`row0 == row1`: a
///   thread's two staging elements share a tile row);
/// - `cmma::fill` zeroes the partials (register op — no extra sync).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup_scale_deferred(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let wg_p = CUBE_POS_X;
    let wg_m = CUBE_POS_Y;
    let base_p = wg_p * (CMMA_DIM * BN_SUB);
    let base_m = wg_m * CMMA_DIM;

    // 4 partial accumulators — per-group UNSCALED partials.
    #[allow(unused_mut)]
    let mut acc0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    #[allow(unused_mut)]
    let mut acc3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );

    // f16 tiles — half the smem of the f32 8×32 variant (9 × 64 f16 ≈ 1.1 KB).
    let mut tile_w = Shared::<[f16]>::new_slice(64usize);
    let mut tile_x0 = Shared::<[f16]>::new_slice(64usize);
    let mut tile_x1 = Shared::<[f16]>::new_slice(64usize);
    let mut tile_x2 = Shared::<[f16]>::new_slice(64usize);
    let mut tile_x3 = Shared::<[f16]>::new_slice(64usize);
    // f32 staging for the group-boundary merge (partials store here).
    let mut result0 = Shared::<[f32]>::new_slice(64usize);
    let mut result1 = Shared::<[f32]>::new_slice(64usize);
    let mut result2 = Shared::<[f32]>::new_slice(64usize);
    let mut result3 = Shared::<[f32]>::new_slice(64usize);

    let tid = UNIT_POS;
    let e0 = tid * 2u32;
    let e1 = tid * 2u32 + 1u32;
    let row0 = e0 / CMMA_DIM;
    let col0 = e0 % CMMA_DIM;
    let row1 = e1 / CMMA_DIM;
    let col1 = e1 % CMMA_DIM;

    // Per-thread scaled totals — 4 sub-tiles × 2 elements (8 f32 registers).
    // The epilogue writes these to the output (bounds-checked, same mapping
    // as the 8×32 variant).
    let mut tot0e0 = f32::new(0.0f32);
    let mut tot0e1 = f32::new(0.0f32);
    let mut tot1e0 = f32::new(0.0f32);
    let mut tot1e1 = f32::new(0.0f32);
    let mut tot2e0 = f32::new(0.0f32);
    let mut tot2e1 = f32::new(0.0f32);
    let mut tot3e0 = f32::new(0.0f32);
    let mut tot3e1 = f32::new(0.0f32);

    // The thread's two staging elements share a tile row (e0, e1 consecutive
    // within an 8-wide row) → ONE scale load per boundary covers both.
    let merge_row = base_m + row0;
    let merge_row_valid = merge_row < m;

    let num_k_tiles = n.div_ceil(CMMA_DIM);
    let mut k_tile = 0u32;
    while k_tile < num_k_tiles {
        let k_base = k_tile * CMMA_DIM;

        // ── Dequant weight tile: SIGN ONLY, f16, exact {-1,0,1}. No scale. ──
        let wr0 = base_m + row0;
        let wc0 = k_base + col0;
        if wr0 < m && wc0 < n {
            let pos_word_idx = (wr0 * words_per_row + (wc0 / 32u32)) as usize;
            let bit_pos = wc0 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            tile_w[e0 as usize] = f16::cast_from(sign);
        } else {
            tile_w[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        let wr1 = base_m + row1;
        let wc1 = k_base + col1;
        if wr1 < m && wc1 < n {
            let pos_word_idx = (wr1 * words_per_row + (wc1 / 32u32)) as usize;
            let bit_pos = wc1 % 32u32;
            let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
            let one = f32::new(1.0f32);
            let zero = f32::new(0.0f32);
            let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
            tile_w[e1 as usize] = f16::cast_from(sign);
        } else {
            tile_w[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        // ── Load 4 input sub-tiles, f32 → f16 cast (the activation-side
        //    precision point; Issue 767 T2 gates this FIRST). ──
        let tok0_s0 = base_p + row0;
        let tok0_s1 = base_p + CMMA_DIM + row0;
        let tok0_s2 = base_p + 2u32 * CMMA_DIM + row0;
        let tok0_s3 = base_p + 3u32 * CMMA_DIM + row0;
        let xk0 = k_base + col0;
        if tok0_s0 < p_tokens && xk0 < n {
            tile_x0[e0 as usize] = f16::cast_from(input_batch[(tok0_s0 * n + xk0) as usize]);
        } else {
            tile_x0[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }
        if tok0_s1 < p_tokens && xk0 < n {
            tile_x1[e0 as usize] = f16::cast_from(input_batch[(tok0_s1 * n + xk0) as usize]);
        } else {
            tile_x1[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }
        if tok0_s2 < p_tokens && xk0 < n {
            tile_x2[e0 as usize] = f16::cast_from(input_batch[(tok0_s2 * n + xk0) as usize]);
        } else {
            tile_x2[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }
        if tok0_s3 < p_tokens && xk0 < n {
            tile_x3[e0 as usize] = f16::cast_from(input_batch[(tok0_s3 * n + xk0) as usize]);
        } else {
            tile_x3[e0 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        let tok1_s0 = base_p + row1;
        let tok1_s1 = base_p + CMMA_DIM + row1;
        let tok1_s2 = base_p + 2u32 * CMMA_DIM + row1;
        let tok1_s3 = base_p + 3u32 * CMMA_DIM + row1;
        let xk1 = k_base + col1;
        if tok1_s0 < p_tokens && xk1 < n {
            tile_x0[e1 as usize] = f16::cast_from(input_batch[(tok1_s0 * n + xk1) as usize]);
        } else {
            tile_x0[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }
        if tok1_s1 < p_tokens && xk1 < n {
            tile_x1[e1 as usize] = f16::cast_from(input_batch[(tok1_s1 * n + xk1) as usize]);
        } else {
            tile_x1[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }
        if tok1_s2 < p_tokens && xk1 < n {
            tile_x2[e1 as usize] = f16::cast_from(input_batch[(tok1_s2 * n + xk1) as usize]);
        } else {
            tile_x2[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }
        if tok1_s3 < p_tokens && xk1 < n {
            tile_x3[e1 as usize] = f16::cast_from(input_batch[(tok1_s3 * n + xk1) as usize]);
        } else {
            tile_x3[e1 as usize] = f16::cast_from(f32::new(0.0f32));
        }

        sync_cube();

        let mat_a = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor, &tile_w, 8,
        );

        let mat_b0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x0, 8,
        );
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&mat_a, &mat_b0, &acc0, &acc0);

        let mat_b1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x1, 8,
        );
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&mat_a, &mat_b1, &acc1, &acc1);

        let mat_b2 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x2, 8,
        );
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&mat_a, &mat_b2, &acc2, &acc2);

        let mat_b3 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::ColMajor, &tile_x3, 8,
        );
        cmma::execute::<f16, f16, f32, f32, cmma::Plane>(&mat_a, &mat_b3, &acc3, &acc3);

        sync_cube();

        // ── Group boundary: merge the unscaled partials into the totals. ──
        // Fires every KTILES_PER_GROUP tiles (exact group end) OR at the last
        // tile (tail group — covers n % 128 != 0 and the zero-padded n % 8
        // tail; zero-padded elements contribute exactly 0 to the partial).
        let k_end = (k_tile + 1u32) * CMMA_DIM; // exclusive K end covered so far
        if (k_tile + 1u32).is_multiple_of(KTILES_PER_GROUP) || k_tile + 1u32 == num_k_tiles {
            cmma::store(&mut result0, &acc0, 8, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut result1, &acc1, 8, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut result2, &acc2, 8, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut result3, &acc3, 8, cmma::MatrixLayout::RowMajor);
            sync_cube();

            if merge_row_valid {
                // Group index covering k_end-1 (the merge's own scale row).
                // For the forced-final boundary with a padded tail this can
                // index the last group's scale while partial elements past n
                // are zero — merging them adds exactly 0. Bounds: g ≤ (n-1)/128
                // < groups_per_row by construction.
                let g = (k_end - 1u32) / K_PER_GROUP;
                let s = group_scale_f32[(merge_row * groups_per_row + g) as usize];
                tot0e0 += s * result0[e0 as usize];
                tot0e1 += s * result0[e1 as usize];
                tot1e0 += s * result1[e0 as usize];
                tot1e1 += s * result1[e1 as usize];
                tot2e0 += s * result2[e0 as usize];
                tot2e1 += s * result2[e1 as usize];
                tot3e0 += s * result3[e0 as usize];
                tot3e1 += s * result3[e1 as usize];
            }

            // Zero the partials (register op; no extra sync). The next staging
            // write is ≥1 full K-tile away, behind the loop's own syncs.
            cmma::fill(&mut acc0, 0.0f32);
            cmma::fill(&mut acc1, 0.0f32);
            cmma::fill(&mut acc2, 0.0f32);
            cmma::fill(&mut acc3, 0.0f32);
            sync_cube();
        }

        k_tile += 1u32;
    }

    // Epilogue: totals → output (bounds-checked; same mapping as the 8×32
    // variant — resultN[row*8+col] ↔ output[(base_p+N*8+col)*m + base_m+row]).
    let out_row0 = base_m + row0;
    let out_row1 = base_m + row1;

    let out_tok0_s0 = base_p + col0;
    if out_tok0_s0 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s0 * m + out_row0) as usize] = tot0e0;
    }
    let out_tok0_s1 = base_p + CMMA_DIM + col0;
    if out_tok0_s1 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s1 * m + out_row0) as usize] = tot1e0;
    }
    let out_tok0_s2 = base_p + 2u32 * CMMA_DIM + col0;
    if out_tok0_s2 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s2 * m + out_row0) as usize] = tot2e0;
    }
    let out_tok0_s3 = base_p + 3u32 * CMMA_DIM + col0;
    if out_tok0_s3 < p_tokens && out_row0 < m {
        output_batch[(out_tok0_s3 * m + out_row0) as usize] = tot3e0;
    }

    let out_tok1_s0 = base_p + col1;
    if out_tok1_s0 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s0 * m + out_row1) as usize] = tot0e1;
    }
    let out_tok1_s1 = base_p + CMMA_DIM + col1;
    if out_tok1_s1 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s1 * m + out_row1) as usize] = tot1e1;
    }
    let out_tok1_s2 = base_p + 2u32 * CMMA_DIM + col1;
    if out_tok1_s2 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s2 * m + out_row1) as usize] = tot2e1;
    }
    let out_tok1_s3 = base_p + 3u32 * CMMA_DIM + col1;
    if out_tok1_s3 < p_tokens && out_row1 < m {
        output_batch[(out_tok1_s3 * m + out_row1) as usize] = tot3e1;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Simdgroup-matrix ternary GEMM launcher.
///
/// Computes `output_batch[P × m] = dequant_ternary(weight) @ input_batch[P × n]^T`
/// in a single dispatch using Metal `simdgroup_matrix_8x8` (CubeCL `cmma`).
///
/// This is the **simdgroup-matrix path** — the hardware cooperative matrix
/// approach that Bench 641 identified as the real lever for the 8.38× prefill
/// gap. The plane-cooperative [`GemmTernaryBatchedCubeCL`] caps at ~1.08×
/// because it is occupancy-bound in scalar ALU; this kernel offloads the
/// multiply-accumulate to the hardware cmma unit.
///
/// # Layout
///
/// - `input_handle`: row-major `[p_tokens, n]` — `input[tok * n + col]`
/// - `output_handle`: row-major `[p_tokens, m]` — `output[tok * m + row]`
/// - `handle`: the ternary projection `[m × n]`, same handle the GEMV path uses
///
/// # Safety
///
/// - `input_handle` must point to `p_tokens × handle.n` f32 elements
/// - `output_handle` must point to `p_tokens × handle.m` f32 elements
/// - `p_tokens > 0`
/// - buffers in `handle` must have been created from the same `client`
/// - the device must support cmma `(f32, f32, f32)` at 8×8×8 — call
///   [`Self::cmma_available`] to check
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernarySimdgroupCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemmTernarySimdgroupCubeCL {
    /// Launch the simdgroup-matrix ternary GEMM.
    ///
    /// # Safety
    ///
    /// - `input_handle` must point to `p_tokens × handle.n` f32 elements
    /// - `output_handle` must point to `p_tokens × handle.m` f32 elements
    /// - `p_tokens > 0`
    /// - buffers in `handle` must have been created from the same `client`
    /// - the device must support cmma `(f32, f32, f32)` at 8×8×8
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        // SWAP: X = P-tiles (tokens), Y = M-tiles (rows). Each workgroup
        // computes one 8×8 output tile. CubeDim=32 = one simdgroup (the cmma
        // plane size on Metal).
        let num_wg_x = p.div_ceil(CMMA_DIM).max(1);
        let num_wg_y = m.div_ceil(CMMA_DIM).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_simdgroup::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32), // one simdgroup per workgroup
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Launch the 8×32 output-tile simdgroup-matrix ternary GEMM.
    ///
    /// Variant of [`Self::launch`] with 4× larger output tiles (8 rows × 32
    /// tokens per workgroup). Dequantizes the weight tile ONCE per K-step and
    /// reuses it across 4 cmma executes — reducing total dequant work 4× vs
    /// the 8×8 variant. Best for `p_tokens >= 32`.
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`Self::launch`].
    pub unsafe fn launch_8x32<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        // X = P-tiles (stride 32), Y = M-tiles (stride 8).
        let num_wg_x = p.div_ceil(CMMA_DIM * BN_SUB).max(1);
        let num_wg_y = m.div_ceil(CMMA_DIM).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_simdgroup_8x32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Launch the scale-deferred simdgroup ternary GEMM (Issue 767 T1).
    ///
    /// 8×32-tile variant of [`Self::launch_8x32`] where the weight rides the
    /// cmma unit as **f16 signs only** ({-1,0,1}, exact) and the group scales
    /// apply to the f32 accumulator at group boundaries (every 128 K) — the
    /// per-element scale-mul leaves the inner loop entirely. The 4090's
    /// cmma_i8 kernel (Bench 709) shipped this structure; this is the Metal
    /// simdgroup twin.
    ///
    /// # Safety
    ///
    /// Same as [`Self::launch_8x32`] + the device must support cmma
    /// `(f16, f16, f32)` at 8×8×8 — call [`Self::cmma_available_f16`].
    #[cfg(feature = "ternary_gemm_simdgroup_f16")]
    pub unsafe fn launch_scale_deferred<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        // Same 8×32 dispatch shape as launch_8x32.
        let num_wg_x = p.div_ceil(CMMA_DIM * BN_SUB).max(1);
        let num_wg_y = m.div_ceil(CMMA_DIM).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_simdgroup_scale_deferred::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Launch the 32×32 output-tile simdgroup-matrix ternary GEMM (Issue 768).
    ///
    /// The input-reuse variant: 4× more M-rows per workgroup than
    /// [`Self::launch_8x32`] → input activation traffic ÷4 (the measured #1
    /// cost, Bench 774: 42–63% marginal) and weight re-dequant ÷4. Per-element
    /// accumulation order is identical to the 8×32 kernel — outputs are
    /// bit-identical by construction.
    ///
    /// Best for `m >= 64` and `p_tokens >= 32` (ragged shapes zero-padded;
    /// small-m shapes prefer the 8×32 kernel's finer M-tiling).
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`Self::launch_8x32`].
    pub unsafe fn launch_32x32<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        // X = P-tiles (stride 32), Y = M-tiles (stride 32).
        let num_wg_x = p.div_ceil(CMMA_DIM * BN_SUB).max(1);
        let num_wg_y = m.div_ceil(CMMA_DIM * 4u32).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_simdgroup_32x32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Check whether the device supports cmma `(f32, f32, f32)` at 8×8×8.
    ///
    /// On Metal3 this returns `true` (4 wmma configs registered including
    /// `(f32,f32,f32)`). On backends without cmma support, returns `false`.
    pub fn cmma_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
        use cubecl::ir::{ElemType, FloatKind};
        use cubecl::ir::features::MmaConfig;

        client.features().matmul.cmma.contains(&MmaConfig {
            a_type: ElemType::Float(FloatKind::F32).into(),
            b_type: ElemType::Float(FloatKind::F32).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 8,
            n: 8,
            k: 8,
        })
    }

    /// Launch the f16-input simdgroup-matrix ternary GEMM (Issue 655).
    ///
    /// Mixed-precision variant of [`Self::launch`]: dequantizes weights to f16
    /// and uses f16 input tiles with `<f16, f16, f32>` cmma. On Apple Silicon,
    /// `simdgroup_matrix_8x8<half>` has 2× instruction throughput. The
    /// accumulator stays f32 (no precision loss in the dot-product sum).
    ///
    /// # Safety
    ///
    /// Same as [`Self::launch`] + the device must support cmma `(f16, f16, f32)`
    /// at 8×8×8 — call [`Self::cmma_available_f16`] to check.
    #[cfg(feature = "ternary_gemm_simdgroup_f16")]
    pub unsafe fn launch_f16<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = p.div_ceil(CMMA_DIM).max(1);
        let num_wg_y = m.div_ceil(CMMA_DIM).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_simdgroup_f16::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, p_tokens * handle.n),
                BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
                blocks64,
                groups_per_row,
                n,
                m,
                p,
            );
        }
    }

    /// Check whether the device supports cmma `(f16, f16, f32)` at 8×8×8.
    #[cfg(feature = "ternary_gemm_simdgroup_f16")]
    pub fn cmma_available_f16<R: Runtime>(client: &ComputeClient<R>) -> bool {
        use cubecl::ir::{ElemType, FloatKind};
        use cubecl::ir::features::MmaConfig;

        client.features().matmul.cmma.contains(&MmaConfig {
            a_type: ElemType::Float(FloatKind::F16).into(),
            b_type: ElemType::Float(FloatKind::F16).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 8,
            n: 8,
            k: 8,
        })
    }
}
