//! Issue 771 T1 — smem-staged **multi-simdgroup** ternary GEMM (the M3
//! prefill lane's next input-reuse mechanism).
//!
//! The 32×32 tile (Bench 774) cut input-activation traffic 4× by giving one
//! simdgroup a 32×32 output tile, but the workgroup is still ONE simdgroup:
//! every workgroup re-reads its X rows from global and re-dequants its W rows
//! alone. At the FFN shape (m=17408, n=5120, p=2048) the X matrix is still
//! re-read once per M-tile — 272… no: `ceil(m/32) = 544` times ≈ 22.8 GB/GEMM
//! — and the whole mma path saturates near 2.9 TF (gate: ≥5.4 TF, llama.cpp
//! proves ~6.8 TF on this silicon).
//!
//! This kernel is the "new mechanism" Issue 771 T1 asks for (BM=64 registers
//! and f16-input carry measured refutations at matched assumptions): a
//! **256-thread workgroup = 8 simdgroups** computing one **128(M) × 64(P)**
//! output tile, with the W tile (128×32) and X tile (64×32) staged in
//! **threadgroup memory once and shared across all 8 simdgroups** — the
//! llama.cpp MMQ family structure.
//!
//! # Per-execute overhead vs the 32×32 kernel (the design arithmetic)
//!
//! | cost | 32×32 | smem 128×64 | cut |
//! |---|---|---|---|
//! | X global traffic | 22.8 GB (÷32 re-read) | 5.7 GB (÷128 re-read) | **÷4** |
//! | W bit traffic | 2.86 GB (÷64 re-read) | 1.43 GB (÷32 re-read) | **÷2** |
//! | staging writes /execute | 16/thread per 16 exec | 24/thread per 64 exec | **÷2.7** |
//! | syncs | 2 per K-tile (640) | 2 per 4 K-tiles (160) | **÷4** |
//! | from_slice | 32 per K-tile | 8 per 4 K-tiles per sg | **÷4** |
//!
//! smem budget: W 16 KB + X 8 KB + 8 per-sg result tiles 2 KB = **26 KB** of
//! the 32 KB Apple threadgroup limit.
//!
//! # Issue 843 T3 — the staged epilogue (default ON)
//!
//! The epilogue originally issued 16× (store → sync → scatter → sync) = 32
//! workgroup barriers. The staged twin (`gemm_ternary_simdgroup_smem_estage`)
//! stores all 16 accumulator tiles into the dead W staging area in two
//! 8-tile halves — 4 barriers — and is now the [`PREFILL_SMEM_STAGED_EPILOGUE`]
//! default: cooled-window kernel A/B +2.5–2.8% @16384 (×2 samples) and
//! +4.5% @8192 (×2), tie @2048, bit-identical on every ragged path
//! (`tests/probe_843_t3_smem_longm.rs`). The same probe REFUTED the
//! dispatch-order axis swap (−6…−7.5% everywhere) — the W-slab-sharing
//! order is correct.
//!
//! # Bit-identity (by construction)
//!
//! The K loop runs 32-wide outer tiles, each executing its four 8-wide
//! sub-tiles **in ascending order and skipping only fully-out-of-range
//! sub-tiles** — the per-element accumulation sequence over k is exactly the
//! 32×32 kernel's k-tile sequence (`ceil(n/8)` executed 8-blocks, in order).
//! Operands carry identical bits (same dequant formula; OOB elements staged
//! as 0.0 exactly like the guard-padded writes). Zero-tail executes add
//! exact ±0 products, never flipping a nonzero accumulator — and the skip
//! removes the only case where a fresh +0.0 could flip a −0.0. Outputs are
//! bit-identical to [`super::GemmTernarySimdgroupCubeCL::launch_32x32`].

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "ternary_gemm_simdgroup")]
use cubecl::cmma;

#[cfg(feature = "ternary_gemm_simdgroup")]
use cubecl::prelude::*;

#[cfg(feature = "ternary_gemm_simdgroup")]
use cubecl::server::Handle;

// Sub-slicing shared memory: `tile.slice(start, end)` — the range-index path
// (`&tile[off..]`) panics in cubecl 0.11.0-pre.2's expansion (len_static
// assumes the raw Array type; new_slice wraps it in a Slice aggregate).
#[cfg(feature = "ternary_gemm_simdgroup")]
use cubecl::frontend::SliceOperator;

#[cfg(feature = "ternary_gemm_simdgroup")]
use super::GemmTernarySimdgroupCubeCL;

#[cfg(feature = "ternary_gemm_simdgroup")]
use crate::gemv_ternary_cubecl::TernaryHandle;

/// M rows per workgroup (4 simdgroups along M).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_BM: u32 = 128;

/// Tokens per workgroup (2 simdgroups along P).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_BP: u32 = 64;

/// K columns staged per outer iteration (4 cmma sub-tiles of 8).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_KT: u32 = 32;

/// Simdgroups along P.
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_SG_P: u32 = 2;

/// Threads per workgroup (8 simdgroups).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_THREADS: u32 = 256;

/// Comptime end index for `tile_w.slice` spans (the full tile length).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_W_LEN: usize = (SMEM_BM * SMEM_KT) as usize;

/// Dequant elements per thread per outer iteration (comptime-unrolled).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_W_J: u32 = SMEM_BM * SMEM_KT / SMEM_THREADS;

/// X-load elements per thread per outer iteration (comptime-unrolled).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_X_J: u32 = SMEM_BP * SMEM_KT / SMEM_THREADS;

/// Comptime end index for `tile_x.slice` spans (the full tile length).
#[cfg(feature = "ternary_gemm_simdgroup")]
const SMEM_X_LEN: usize = (SMEM_BP * SMEM_KT) as usize;

/// Simdgroup-matrix ternary bit-plane GEMM, 128×64 tile staged in threadgroup
/// memory and shared across 8 simdgroups.
///
/// Computes `output_batch[P × m] = dequant_ternary(weight[m × n]) @ input_batch[P × n]^T`.
/// Dispatch heuristic keeps `m < 128 || p < 64` on the 32×32/8×32 arms (the
/// wider tile pads small shapes).
#[cfg(feature = "ternary_gemm_simdgroup")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup_smem(
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
    axis_swap: u32,
) {
    let words_per_row = blocks64 * 2u32;

    // Issue 843 T3 probe: dispatch-order knob. Hardware threadgroup dispatch
    // order is undocumented on Apple GPUs; swapping the CubeCount axes (with
    // this flag) moves which workgroup axis iterates fastest — the
    // co-resident wave then shares W slabs (swap=0) vs X tiles (swap=1).
    let wg_p = if axis_swap != 0u32 {
        CUBE_POS_Y
    } else {
        CUBE_POS_X
    };
    let wg_m = if axis_swap != 0u32 {
        CUBE_POS_X
    } else {
        CUBE_POS_Y
    };
    let base_p = wg_p * SMEM_BP;
    let base_m = wg_m * SMEM_BM;

    let tid = UNIT_POS;
    let sg = tid / 32u32;
    let sg_m = sg / SMEM_SG_P; // 0..4
    let sg_p = sg % SMEM_SG_P; // 0..2

    // 16 accumulators — (mi, pj) 8×8 sub-tiles of this simdgroup's 32×32.
    let acc_0_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_0_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_0_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_0_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );

    // Shared memory: the W tile (128 rows × 32 k) and X tile (64 tokens ×
    // 32 k) are staged ONCE per outer iteration and consumed by ALL 8
    // simdgroups; each simdgroup owns a PRIVATE 64-element result region
    // (8 × 64 = 512 f32) for the epilogue — one shared tile would race
    // across the 8 sgs' stores.
    let mut tile_w = Shared::<[f32]>::new_slice((SMEM_BM * SMEM_KT) as usize);
    let mut tile_x = Shared::<[f32]>::new_slice((SMEM_BP * SMEM_KT) as usize);
    let mut result = Shared::<[f32]>::new_slice(512usize);
    let res_base = sg * 64u32;

    // Per-thread element mapping inside each staged 8×8 tile (same as the
    // 32×32 kernel): lane ↦ (row = e/8, col = e%8).
    let lane = tid % 32u32;
    let e0 = lane * 2u32;
    let e1 = e0 + 1u32;
    let lr0 = e0 / 8u32;
    let lc0 = e0 % 8u32;
    let lr1 = e1 / 8u32;
    let lc1 = e1 % 8u32;

    let num_k_outer = n.div_ceil(SMEM_KT);
    let mut it = 0u32;
    while it < num_k_outer {
        let k_base = it * SMEM_KT;

        // ── Cooperative dequant: 128×32 weight tile, 16 elements/thread ──
        for j in 0..SMEM_W_J
        {
            let idx = tid + j * SMEM_THREADS;
            let r = idx / SMEM_KT;
            let kk = idx % SMEM_KT;
            let grow = base_m + r;
            let gcol = k_base + kk;
            if grow < m && gcol < n {
                let pos_word_idx = (grow * words_per_row + (gcol / 32u32)) as usize;
                let bit_pos = gcol % 32u32;
                let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
                let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
                let one = f32::new(1.0f32);
                let zero = f32::new(0.0f32);
                let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
                let scale = group_scale_f32[(grow * groups_per_row + (gcol / 128u32)) as usize];
                tile_w[idx as usize] = sign * scale;
            } else {
                tile_w[idx as usize] = f32::new(0.0f32);
            }
        }

        // ── Cooperative X load: 64×32 token tile, 8 elements/thread ──
        for j in 0..SMEM_X_J
        {
            let idx = tid + j * SMEM_THREADS;
            let t = idx / SMEM_KT;
            let kk = idx % SMEM_KT;
            let gtok = base_p + t;
            let gcol = k_base + kk;
            if gtok < p_tokens && gcol < n {
                tile_x[idx as usize] = input_batch[(gtok * n + gcol) as usize];
            } else {
                tile_x[idx as usize] = f32::new(0.0f32);
            }
        }

        sync_cube(); // both tiles ready for all 8 simdgroups

        // ── Executes: 4 sub-k × 16 (A,B) pairs per simdgroup ──
        // A view (mi, ki): rows sg_m*32 + mi*8, cols ki*8, stride KT.
        // B view (pj, ki): tokens sg_p*32 + pj*8, cols ki*8, stride KT.
        // Sub-tiles fully past n are SKIPPED so the executed k-tile sequence
        // matches the 32×32 kernel exactly (bit-identity — see module docs).
        let mut ki = 0u32;
        while ki < 4u32 {
            let k_lo = k_base + ki * 8u32;
            if k_lo < n {
                let a_base = sg_m * 32u32 * SMEM_KT + ki * 8u32;
                let b_base = sg_p * 32u32 * SMEM_KT + ki * 8u32;
                let a0 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let a1 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base + 8u32 * SMEM_KT) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let a2 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base + 2u32 * 8u32 * SMEM_KT) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let a3 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base + 3u32 * 8u32 * SMEM_KT) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let b0 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base) as usize, SMEM_X_LEN), SMEM_KT,
                );
                let b1 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base + 8u32 * SMEM_KT) as usize, SMEM_X_LEN), SMEM_KT,
                );
                let b2 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base + 2u32 * 8u32 * SMEM_KT) as usize, SMEM_X_LEN), SMEM_KT,
                );
                let b3 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base + 3u32 * 8u32 * SMEM_KT) as usize, SMEM_X_LEN), SMEM_KT,
                );
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b0, &acc_0_0, &acc_0_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b1, &acc_0_1, &acc_0_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b2, &acc_0_2, &acc_0_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b3, &acc_0_3, &acc_0_3);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b0, &acc_1_0, &acc_1_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b1, &acc_1_1, &acc_1_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b2, &acc_1_2, &acc_1_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b3, &acc_1_3, &acc_1_3);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b0, &acc_2_0, &acc_2_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b1, &acc_2_1, &acc_2_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b2, &acc_2_2, &acc_2_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b3, &acc_2_3, &acc_2_3);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b0, &acc_3_0, &acc_3_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b1, &acc_3_1, &acc_3_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b2, &acc_3_2, &acc_3_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b3, &acc_3_3, &acc_3_3);
            }
            ki += 1u32;
        }

        sync_cube(); // tiles consumed before the next staging pass overwrites
        it += 1u32;
    }

    // ── Epilogue: 16 accumulator tiles per simdgroup ──
    // store → sync → guarded scatter → sync (the 32×32 pattern; the result
    // tile is sg-private so the barriers only order intra-sg store/scatter).
    let m_row_base = base_m + sg_m * 32u32;
    let t_tok_base = base_p + sg_p * 32u32;

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_0_0, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + lr0 < m && t_tok_base + lc0 < p_tokens {
        output_batch[((t_tok_base + lc0) * m + m_row_base + lr0) as usize] = result[(res_base + e0) as usize];
    }
    if m_row_base + lr1 < m && t_tok_base + lc1 < p_tokens {
        output_batch[((t_tok_base + lc1) * m + m_row_base + lr1) as usize] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_0_1, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + lr0 < m && t_tok_base + 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc0) * m + m_row_base + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + lr1 < m && t_tok_base + 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc1) * m + m_row_base + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_0_2, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + lr0 < m && t_tok_base + 2u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc0) * m + m_row_base + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + lr1 < m && t_tok_base + 2u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc1) * m + m_row_base + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_0_3, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + lr0 < m && t_tok_base + 3u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc0) * m + m_row_base + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + lr1 < m && t_tok_base + 3u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc1) * m + m_row_base + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_1_0, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 8u32 + lr0 < m && t_tok_base + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + lc0) * m + m_row_base + 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 8u32 + lr1 < m && t_tok_base + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + lc1) * m + m_row_base + 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_1_1, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 8u32 + lr0 < m && t_tok_base + 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc0) * m + m_row_base + 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 8u32 + lr1 < m && t_tok_base + 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc1) * m + m_row_base + 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_1_2, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 8u32 + lr0 < m && t_tok_base + 2u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc0) * m + m_row_base + 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 8u32 + lr1 < m && t_tok_base + 2u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc1) * m + m_row_base + 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_1_3, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 8u32 + lr0 < m && t_tok_base + 3u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc0) * m + m_row_base + 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 8u32 + lr1 < m && t_tok_base + 3u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc1) * m + m_row_base + 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_2_0, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 2u32 * 8u32 + lr0 < m && t_tok_base + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + lc0) * m + m_row_base + 2u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 2u32 * 8u32 + lr1 < m && t_tok_base + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + lc1) * m + m_row_base + 2u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_2_1, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 2u32 * 8u32 + lr0 < m && t_tok_base + 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc0) * m + m_row_base + 2u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 2u32 * 8u32 + lr1 < m && t_tok_base + 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc1) * m + m_row_base + 2u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_2_2, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 2u32 * 8u32 + lr0 < m && t_tok_base + 2u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc0) * m + m_row_base + 2u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 2u32 * 8u32 + lr1 < m && t_tok_base + 2u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc1) * m + m_row_base + 2u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_2_3, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 2u32 * 8u32 + lr0 < m && t_tok_base + 3u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc0) * m + m_row_base + 2u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 2u32 * 8u32 + lr1 < m && t_tok_base + 3u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc1) * m + m_row_base + 2u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_3_0, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 3u32 * 8u32 + lr0 < m && t_tok_base + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + lc0) * m + m_row_base + 3u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 3u32 * 8u32 + lr1 < m && t_tok_base + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + lc1) * m + m_row_base + 3u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_3_1, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 3u32 * 8u32 + lr0 < m && t_tok_base + 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc0) * m + m_row_base + 3u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 3u32 * 8u32 + lr1 < m && t_tok_base + 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 8u32 + lc1) * m + m_row_base + 3u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_3_2, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 3u32 * 8u32 + lr0 < m && t_tok_base + 2u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc0) * m + m_row_base + 3u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 3u32 * 8u32 + lr1 < m && t_tok_base + 2u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 2u32 * 8u32 + lc1) * m + m_row_base + 3u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();

    cmma::store(result.slice_mut(res_base as usize, res_base as usize + 64usize), &acc_3_3, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    if m_row_base + 3u32 * 8u32 + lr0 < m && t_tok_base + 3u32 * 8u32 + lc0 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc0) * m + m_row_base + 3u32 * 8u32 + lr0) as usize
        ] = result[(res_base + e0) as usize];
    }
    if m_row_base + 3u32 * 8u32 + lr1 < m && t_tok_base + 3u32 * 8u32 + lc1 < p_tokens {
        output_batch[
            ((t_tok_base + 3u32 * 8u32 + lc1) * m + m_row_base + 3u32 * 8u32 + lr1) as usize
        ] = result[(res_base + e1) as usize];
    }
    sync_cube();
}

/// Issue 843 T3 — the staged-epilogue twin of [`gemm_ternary_simdgroup_smem`].
///
/// The head + K loop are byte-for-byte the base kernel's (bit-identity by
/// construction). The epilogue changes SHAPE, not values: the 16 accumulator
/// tiles are stored into the (post-loop dead) W staging area — each
/// simdgroup owns the private 512-element slice
/// `tile_w[sg*512 .. sg*512+512)` — in TWO 8-tile halves with a
/// store→sync→scatter→sync per half: **4 workgroup barriers instead of 32**,
/// with the 8 per-simdgroup `cmma::store`s issued back-to-back. The
/// cross-lane smem round-trip keeps exactly the base kernel's ordering
/// guarantees (a full barrier between every store batch and the scatter that
/// reads it, and between the scatter and the next half's overwriting stores;
/// the per-sg regions stay disjoint, so no cross-sg hazard exists).
#[cfg(feature = "ternary_gemm_simdgroup")]
#[cube(launch_unchecked)]
fn gemm_ternary_simdgroup_smem_estage(
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
    axis_swap: u32,
) {
    let words_per_row = blocks64 * 2u32;

    // Same dispatch-order knob as the base kernel (the probe pairs arms).
    let wg_p = if axis_swap != 0u32 {
        CUBE_POS_Y
    } else {
        CUBE_POS_X
    };
    let wg_m = if axis_swap != 0u32 {
        CUBE_POS_X
    } else {
        CUBE_POS_Y
    };
    let base_p = wg_p * SMEM_BP;
    let base_m = wg_m * SMEM_BM;

    let tid = UNIT_POS;
    let sg = tid / 32u32;
    let sg_m = sg / SMEM_SG_P; // 0..4
    let sg_p = sg % SMEM_SG_P; // 0..2

    // 16 accumulators — (mi, pj) 8×8 sub-tiles of this simdgroup's 32×32.
    let acc_0_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_0_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_0_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_0_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_1_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_2_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc_3_3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );

    // Shared memory: the W tile (128 rows × 32 k) and X tile (64 tokens ×
    // 32 k) staged ONCE per outer iteration, shared by all 8 simdgroups.
    // No separate epilogue region — the staged epilogue reuses tile_w (the
    // staging tiles are dead once the K loop drains), each simdgroup owning
    // the private 512-element slice `tile_w[sg*512 .. sg*512+512)`.
    let mut tile_w = Shared::<[f32]>::new_slice((SMEM_BM * SMEM_KT) as usize);
    let mut tile_x = Shared::<[f32]>::new_slice((SMEM_BP * SMEM_KT) as usize);

    // Per-thread element mapping inside each staged 8×8 tile (same as the
    // 32×32 kernel): lane ↦ (row = e/8, col = e%8).
    let lane = tid % 32u32;
    let e0 = lane * 2u32;
    let e1 = e0 + 1u32;
    let lr0 = e0 / 8u32;
    let lc0 = e0 % 8u32;
    let lr1 = e1 / 8u32;
    let lc1 = e1 % 8u32;

    let num_k_outer = n.div_ceil(SMEM_KT);
    let mut it = 0u32;
    while it < num_k_outer {
        let k_base = it * SMEM_KT;

        // ── Cooperative dequant: 128×32 weight tile, 16 elements/thread ──
        for j in 0..SMEM_W_J
        {
            let idx = tid + j * SMEM_THREADS;
            let r = idx / SMEM_KT;
            let kk = idx % SMEM_KT;
            let grow = base_m + r;
            let gcol = k_base + kk;
            if grow < m && gcol < n {
                let pos_word_idx = (grow * words_per_row + (gcol / 32u32)) as usize;
                let bit_pos = gcol % 32u32;
                let pos_bit = (pos_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
                let neg_bit = (neg_bits_u32[pos_word_idx] >> bit_pos) & 1u32;
                let one = f32::new(1.0f32);
                let zero = f32::new(0.0f32);
                let sign = select(pos_bit != 0u32, one, zero) - select(neg_bit != 0u32, one, zero);
                let scale = group_scale_f32[(grow * groups_per_row + (gcol / 128u32)) as usize];
                tile_w[idx as usize] = sign * scale;
            } else {
                tile_w[idx as usize] = f32::new(0.0f32);
            }
        }

        // ── Cooperative X load: 64×32 token tile, 8 elements/thread ──
        for j in 0..SMEM_X_J
        {
            let idx = tid + j * SMEM_THREADS;
            let t = idx / SMEM_KT;
            let kk = idx % SMEM_KT;
            let gtok = base_p + t;
            let gcol = k_base + kk;
            if gtok < p_tokens && gcol < n {
                tile_x[idx as usize] = input_batch[(gtok * n + gcol) as usize];
            } else {
                tile_x[idx as usize] = f32::new(0.0f32);
            }
        }

        sync_cube(); // both tiles ready for all 8 simdgroups

        // ── Executes: identical to the base kernel (bit-identity) ──
        let mut ki = 0u32;
        while ki < 4u32 {
            let k_lo = k_base + ki * 8u32;
            if k_lo < n {
                let a_base = sg_m * 32u32 * SMEM_KT + ki * 8u32;
                let b_base = sg_p * 32u32 * SMEM_KT + ki * 8u32;
                let a0 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let a1 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base + 8u32 * SMEM_KT) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let a2 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base + 2u32 * 8u32 * SMEM_KT) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let a3 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::RowMajor,
                    tile_w.slice((a_base + 3u32 * 8u32 * SMEM_KT) as usize, SMEM_W_LEN), SMEM_KT,
                );
                let b0 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base) as usize, SMEM_X_LEN), SMEM_KT,
                );
                let b1 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base + 8u32 * SMEM_KT) as usize, SMEM_X_LEN), SMEM_KT,
                );
                let b2 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base + 2u32 * 8u32 * SMEM_KT) as usize, SMEM_X_LEN), SMEM_KT,
                );
                let b3 = cmma::Matrix::<f32>::from_slice(
                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                    cmma::MatrixLayout::ColMajor,
                    tile_x.slice((b_base + 3u32 * 8u32 * SMEM_KT) as usize, SMEM_X_LEN), SMEM_KT,
                );
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b0, &acc_0_0, &acc_0_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b1, &acc_0_1, &acc_0_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b2, &acc_0_2, &acc_0_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a0, &b3, &acc_0_3, &acc_0_3);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b0, &acc_1_0, &acc_1_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b1, &acc_1_1, &acc_1_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b2, &acc_1_2, &acc_1_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a1, &b3, &acc_1_3, &acc_1_3);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b0, &acc_2_0, &acc_2_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b1, &acc_2_1, &acc_2_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b2, &acc_2_2, &acc_2_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a2, &b3, &acc_2_3, &acc_2_3);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b0, &acc_3_0, &acc_3_0);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b1, &acc_3_1, &acc_3_1);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b2, &acc_3_2, &acc_3_2);
                cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&a3, &b3, &acc_3_3, &acc_3_3);
            }
            ki += 1u32;
        }

        sync_cube(); // tiles consumed before the next staging pass overwrites
        it += 1u32;
    }

    // ── Staged epilogue: 2 halves × (8 stores → sync → scatter → sync) ──
    let m_row_base = base_m + sg_m * 32u32;
    let t_tok_base = base_p + sg_p * 32u32;
    let res_base = sg * 512u32;

    // Half 1: mi ∈ {0, 1} — slot t ↔ (mi, pj) = (t/4, t%4).
    cmma::store(tile_w.slice_mut(res_base as usize, (res_base + 64u32) as usize), &acc_0_0, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 64u32) as usize, (res_base + 128u32) as usize), &acc_0_1, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 128u32) as usize, (res_base + 192u32) as usize), &acc_0_2, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 192u32) as usize, (res_base + 256u32) as usize), &acc_0_3, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 256u32) as usize, (res_base + 320u32) as usize), &acc_1_0, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 320u32) as usize, (res_base + 384u32) as usize), &acc_1_1, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 384u32) as usize, (res_base + 448u32) as usize), &acc_1_2, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 448u32) as usize, (res_base + 512u32) as usize), &acc_1_3, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    for t in 0..8u32 {
        let ro = (t / 4u32) * 8u32;
        let co = (t % 4u32) * 8u32;
        let off = res_base + t * 64u32;
        if m_row_base + ro + lr0 < m && t_tok_base + co + lc0 < p_tokens {
            output_batch[((t_tok_base + co + lc0) * m + m_row_base + ro + lr0) as usize] =
                tile_w[(off + e0) as usize];
        }
        if m_row_base + ro + lr1 < m && t_tok_base + co + lc1 < p_tokens {
            output_batch[((t_tok_base + co + lc1) * m + m_row_base + ro + lr1) as usize] =
                tile_w[(off + e1) as usize];
        }
    }
    sync_cube();

    // Half 2: mi ∈ {2, 3} — same slots, overwritten only after half 1's
    // scatter drained (the barrier above).
    cmma::store(tile_w.slice_mut(res_base as usize, (res_base + 64u32) as usize), &acc_2_0, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 64u32) as usize, (res_base + 128u32) as usize), &acc_2_1, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 128u32) as usize, (res_base + 192u32) as usize), &acc_2_2, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 192u32) as usize, (res_base + 256u32) as usize), &acc_2_3, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 256u32) as usize, (res_base + 320u32) as usize), &acc_3_0, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 320u32) as usize, (res_base + 384u32) as usize), &acc_3_1, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 384u32) as usize, (res_base + 448u32) as usize), &acc_3_2, 8u32, cmma::MatrixLayout::RowMajor);
    cmma::store(tile_w.slice_mut((res_base + 448u32) as usize, (res_base + 512u32) as usize), &acc_3_3, 8u32, cmma::MatrixLayout::RowMajor);
    sync_cube();
    for t in 0..8u32 {
        let ro = (2u32 + t / 4u32) * 8u32;
        let co = (t % 4u32) * 8u32;
        let off = res_base + t * 64u32;
        if m_row_base + ro + lr0 < m && t_tok_base + co + lc0 < p_tokens {
            output_batch[((t_tok_base + co + lc0) * m + m_row_base + ro + lr0) as usize] =
                tile_w[(off + e0) as usize];
        }
        if m_row_base + ro + lr1 < m && t_tok_base + co + lc1 < p_tokens {
            output_batch[((t_tok_base + co + lc1) * m + m_row_base + ro + lr1) as usize] =
                tile_w[(off + e1) as usize];
        }
    }
    sync_cube();
}

/// Runtime toggle for the smem 128×64 tier (default ON — bit-identical by
/// construction + GOAT-gated; flip off via [`GemmTernarySimdgroupCubeCL::
/// set_prefill_use_smem_gemm`] for A/B runs).
#[cfg(feature = "ternary_gemm_simdgroup")]
static PREFILL_USE_SMEM_GEMM: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Issue 843 T3: the staged epilogue is the smem tier's DEFAULT (cooled-window
/// kernel A/B: +2.5–2.8% @16384 ×2 samples, +4.5% @8192 ×2, tie @2048,
/// bit-identical — `probe_843_t3_smem_longm`). Kill-switch for A/B runs:
/// [`GemmTernarySimdgroupCubeCL::set_prefill_smem_staged_epilogue`].
#[cfg(feature = "ternary_gemm_simdgroup")]
static PREFILL_SMEM_STAGED_EPILOGUE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "ternary_gemm_simdgroup")]
impl GemmTernarySimdgroupCubeCL {
    /// Launch the smem-staged 128(M)×64(P) multi-simdgroup ternary GEMM
    /// (Issue 771 T1).
    ///
    /// 8 simdgroups per workgroup share one staged W tile + X tile in
    /// threadgroup memory. Bit-identical to [`Self::launch_32x32`] by
    /// construction (same k-tile sequence, same operand bits). Best for
    /// `m >= 128` and `p_tokens >= 64`; smaller shapes stay on 32×32/8×32.
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`Self::launch_32x32`].
    pub unsafe fn launch_smem<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");
        unsafe {
            Self::launch_smem_variant::<R>(
                client,
                handle,
                input_handle,
                output_handle,
                p_tokens,
                false,
                Self::prefill_smem_staged_epilogue(),
            );
        }
    }

    /// Issue 843 T3 probe surface — the smem tier with two measured knobs:
    /// `axis_swapped` swaps the CubeCount axes (dispatch-order probe: the
    /// co-resident workgroup wave then shares W slabs at 0 vs X tiles at 1);
    /// `staged_epilogue` routes to the 4-barrier staged-epilogue twin. Both
    /// arms are bit-identity-safe by construction (the probe gate pins them
    /// against [`Self::launch_32x32`]); production keeps [`Self::launch_smem`].
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`Self::launch_32x32`] (buffer lengths
    /// must match the handle geometry; no concurrent mutation of the input
    /// handle while the kernel is in flight).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_smem_variant<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
        axis_swapped: bool,
        staged_epilogue: bool,
    ) {
        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_p = p.div_ceil(SMEM_BP).max(1);
        let num_wg_m = m.div_ceil(SMEM_BM).max(1);
        // Default: X = P-tiles (stride 64), Y = M-tiles (stride 128).
        let (gx, gy) = if axis_swapped {
            (num_wg_m, num_wg_p)
        } else {
            (num_wg_p, num_wg_m)
        };
        let swap_flag: u32 = if axis_swapped { 1 } else { 0 };

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            if staged_epilogue {
                gemm_ternary_simdgroup_smem_estage::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(gx, gy, 1),
                    CubeDim::new_1d(SMEM_THREADS),
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
                    swap_flag,
                );
            } else {
                gemm_ternary_simdgroup_smem::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(gx, gy, 1),
                    CubeDim::new_1d(SMEM_THREADS),
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
                    swap_flag,
                );
            }
        }
    }

    /// See [`PREFILL_USE_SMEM_GEMM`]. A/B hook for the Issue 771 gates.
    pub fn set_prefill_use_smem_gemm(on: bool) {
        PREFILL_USE_SMEM_GEMM.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// See [`PREFILL_USE_SMEM_GEMM`].
    pub fn prefill_smem_gemm_enabled() -> bool {
        PREFILL_USE_SMEM_GEMM.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// See [`PREFILL_SMEM_STAGED_EPILOGUE`]. A/B hook for the Issue 843 gate.
    pub fn set_prefill_smem_staged_epilogue(on: bool) {
        PREFILL_SMEM_STAGED_EPILOGUE.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// See [`PREFILL_SMEM_STAGED_EPILOGUE`].
    pub fn prefill_smem_staged_epilogue() -> bool {
        PREFILL_SMEM_STAGED_EPILOGUE.load(std::sync::atomic::Ordering::Relaxed)
    }
}
