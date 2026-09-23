//! Issue 734 Lever 2 — B-side DIRECT global→matrix loads via
//! `cmma_tensor_addressing` (NV_cooperative_matrix2
//! `OpCooperativeMatrixLoadTensorNV`): the staging-data-path rewrite of the
//! T7 int8 ternary GEMM.
//!
//! Bench 709 refuted the mma-rate hypothesis for the ~33 TFLOPS wall — the
//! wall is the shared-memory staging structure. This variant keeps the A-side
//! bit-plane staging (weights are static bit-planes; the packed-i8 A copy is
//! a separate 25.6 GB memory problem) but replaces the ENTIRE B-side staging
//! (global u32 loads → byte-extract ALU → smem stores → barrier → from_slice)
//! with per-k-step `TensorView::slice` + `Matrix::from_tensor` direct loads:
//!
//! - B view over the `[n, p]` token-major i8 quantization buffer, strides
//!   `[1, n]` (k contiguous within a token row — dim0=K/dim1=N-token, the
//!   contract the Bench-710 k-loop probe pinned: KN 256/256 exact, NK 0/256).
//! - `ClampToEdge` on the token dim replaces the staging kernel's manual
//!   `tok_c = min(tok, p-1)` clamping (probe-verified: OOB token columns
//!   duplicate the edge row; their outputs are never written).
//! - The quantize pre-pass emits i8 BYTES (`q[tok*n + k]`) instead of packed
//!   u32 words — identical byte count (u32-packing was already 1 B/elem), so
//!   the pre-pass cost and global B traffic are unchanged.
//!
//! Numerics contract: the B fragment VALUES and the per-sg mma execution
//! order are identical to the staging kernel (`launch_sg8`) — outputs are
//! expected BIT-IDENTICAL (the G1 gate asserts it).
//!
//! The B fragment loads are issued BEFORE the post-staging barrier so they
//! overlap the barrier wait instead of serializing behind it.

use cubecl::cmma;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::TernaryHandle;

/// Ternary group size = weight-scale group = 128 cols.
const GROUP_COLS: u32 = 128;
/// k per i8 mma instruction.
const K_STEP: u32 = 32;
/// Tokens per workgroup (2 sub-tiles of 16).
const TOK_TILE: u32 = 32;
/// Rows per workgroup (8 sub-tiles of 16).
const ROW_TILE: u32 = 128;

// ---------------------------------------------------------------------------
// Activation quantization pre-pass (byte-emitting variant)
// ---------------------------------------------------------------------------

/// Per-token hi/lo int8 quantization emitting raw BYTES `q[tok*n + k]` (the
/// staging twin emits packed u32 words; values are identical). One workgroup
/// per token row (128 threads): pass 1 computes the row max (shared-memory
/// reduction), pass 2 emits the bytes.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn quantize_rows_i8_hilo_bytes(
    input: &[f32],
    q_hi_b: &mut [i8],
    q_lo_b: &mut [i8],
    s_out: &mut [f32],
    n: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS;
    let nt = CUBE_DIM_X;
    let zero = f32::new(0.0f32);
    let base = row * n;

    // Pass 1: row max |x|.
    let mut mx = zero;
    let mut c = tid;
    while c < n {
        let v = input[(base + c) as usize];
        let a = if v < zero { zero - v } else { v };
        mx = if a > mx { a } else { mx };
        c += nt;
    }
    let mut red = Shared::<[f32]>::new_slice(128usize);
    red[tid as usize] = mx;
    sync_cube();
    let mut m = zero;
    let mut i = 0u32;
    while i < 128u32 {
        let v = red[i as usize];
        m = if v > m { v } else { m };
        i += 1u32;
    }
    let s = if m > zero {
        m / f32::new(127.0f32)
    } else {
        f32::new(1.0f32)
    };
    if tid == 0u32 {
        s_out[row as usize] = s;
    }

    // Pass 2: emit bytes (1 col per byte).
    let hi127 = f32::new(127.0f32);
    let lo127 = f32::new(-127.0f32);
    let hi64 = f32::new(64.0f32);
    let lo64 = f32::new(-64.0f32);
    let mut c2 = tid;
    while c2 < n {
        let x = input[(base + c2) as usize];
        let xf = x / s;
        let qh = xf.round();
        let qh = if qh > hi127 {
            hi127
        } else if qh < lo127 {
            lo127
        } else {
            qh
        };
        let rem = xf - qh;
        let ql = rem * f32::new(128.0f32);
        let ql = if ql > hi64 {
            hi64
        } else if ql < lo64 {
            lo64
        } else {
            ql
        };
        // Same-width i32→i8 casts are bit-preserving on the clamped range.
        let qh_i = i32::cast_from(qh);
        let ql_i = i32::cast_from(ql);
        q_hi_b[(base + c2) as usize] = i8::cast_from(qh_i);
        q_lo_b[(base + c2) as usize] = i8::cast_from(ql_i);
        c2 += nt;
    }
}

// ---------------------------------------------------------------------------
// GEMM kernel (B-direct)
// ---------------------------------------------------------------------------

/// Int8 cooperative-matrix ternary GEMM, B-side direct-load variant:
/// `output[p × m] = dequant(W) @ input^T` via i8×i8→i32 @ 16×16×32 tensor-core
/// mma. Workgroup = 256 threads (8 subgroups); tile = 128 rows × 32 tokens.
///
/// Same numerics as `gemm_ternary_cmma_i8_sg8` (the staging twin in
/// `gemm_ternary_cmma_i8_cubecl.rs`) — B fragment values and mma order are
/// identical; the only difference is the B load path (from_tensor direct vs
/// smem staging).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma_i8_direct(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    q_hi_b: &[i8],
    q_lo_b: &[i8],
    s_t: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;

    let base_tok = CUBE_POS_X * TOK_TILE;
    let base_row = CUBE_POS_Y * ROW_TILE;
    if base_row >= m || base_tok >= p_tokens {
        terminate!();
    }

    // ── B direct-load views over the [n, p] token-major i8 buffers:
    //    dim0 = k (stride 1, contiguous within a token row), dim1 = token
    //    (stride n). ClampToEdge on the token dim replaces the staging
    //    kernel's manual tok clamping (probe-verified). ──
    let b_hi_view = TensorView::<i8>::new(q_hi_b, seq![n, p_tokens])
        .with_strides(seq![1u32, n])
        .with_clamp_mode(TensorClampMode::ClampToEdge)
        .finish();
    let b_lo_view = TensorView::<i8>::new(q_lo_b, seq![n, p_tokens])
        .with_strides(seq![1u32, n])
        .with_clamp_mode(TensorClampMode::ClampToEdge)
        .finish();

    // ── Staging: 8 A sign tiles (the weights stay on the bit-plane staging —
    //    the packed-i8 A copy is a separate memory-budget problem). ──
    let mut a0 = Shared::<[i8]>::new_slice(512usize);
    let mut a1 = Shared::<[i8]>::new_slice(512usize);
    let mut a2 = Shared::<[i8]>::new_slice(512usize);
    let mut a3 = Shared::<[i8]>::new_slice(512usize);
    let mut a4 = Shared::<[i8]>::new_slice(512usize);
    let mut a5 = Shared::<[i8]>::new_slice(512usize);
    let mut a6 = Shared::<[i8]>::new_slice(512usize);
    let mut a7 = Shared::<[i8]>::new_slice(512usize);

    // ── Per-group partial stores: 4 named buffers per sg (32 total; the
    //    reduction reads only its own sg's set). ──
    let mut g0h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g0l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g0h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g0l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g1h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g1l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g1h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g1l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g2h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g2l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g2h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g2l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g3h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g3l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g3h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g3l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g4h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g4l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g4h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g4l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g5h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g5l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g5h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g5l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g6h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g6l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g6h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g6l1 = Shared::<[i32]>::new_slice(256usize);
    let mut g7h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g7l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g7h1 = Shared::<[i32]>::new_slice(256usize);
    let mut g7l1 = Shared::<[i32]>::new_slice(256usize);

    let tid = UNIT_POS; // 0..256
    let sg = tid / 32u32;
    let one_i = i32::new(1i64);
    let zero_i = i32::new(0i64);

    // ── Output ownership: thread tid owns row `r_own = tid/2`, token
    //    sub-tile `t_sub = tid%2`, 16 consecutive tokens within it. ──
    let r_own = tid / 2u32; // 0..128, row within the tile
    let t_sub = tid % 2u32; // token sub-tile
    let row_g = base_row + r_own;
    let row_c = if row_g < m { row_g } else { m - 1u32 };
    let r_in = r_own % 16u32; // row within the 16-row sub-tile
    let t_base = t_sub * 16u32;

    // 16 f32 output accumulators (registers).
    let mut o0 = f32::new(0.0f32);
    let mut o1 = f32::new(0.0f32);
    let mut o2 = f32::new(0.0f32);
    let mut o3 = f32::new(0.0f32);
    let mut o4 = f32::new(0.0f32);
    let mut o5 = f32::new(0.0f32);
    let mut o6 = f32::new(0.0f32);
    let mut o7 = f32::new(0.0f32);
    let mut o8 = f32::new(0.0f32);
    let mut o9 = f32::new(0.0f32);
    let mut o10 = f32::new(0.0f32);
    let mut o11 = f32::new(0.0f32);
    let mut o12 = f32::new(0.0f32);
    let mut o13 = f32::new(0.0f32);
    let mut o14 = f32::new(0.0f32);
    let mut o15 = f32::new(0.0f32);

    let one128 = f32::new(1.0f32 / 128.0f32);

    let mut g = 0u32;
    while g < groups_per_row {
        // Fresh i32 accumulators per group (Fill).
        #[allow(unused_mut)]
        let mut ahi0 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );
        #[allow(unused_mut)]
        let mut alo0 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );
        #[allow(unused_mut)]
        let mut ahi1 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );
        #[allow(unused_mut)]
        let mut alo1 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );

        let mut ks = 0u32;
        while ks < 4u32 {
            let k_base = g * GROUP_COLS + ks * K_STEP;

            // ── B DIRECT LOADS FIRST (the latency-hiding order): issued
            //    BEFORE the A staging so their global-load latency overlaps
            //    the (long) staging + barrier window instead of serializing
            //    behind it — from_tensor on runtime-offset slices of the
            //    [n, p] i8 buffers. Fragment (K=32, N=16 tokens): slice
            //    (k_base, tok_base) shape [32, 16]. ──
            let bh0_t = b_hi_view.slice(seq![k_base, base_tok], seq![32u32, 16u32]);
            let bh1_t = b_hi_view.slice(seq![k_base, base_tok + 16u32], seq![32u32, 16u32]);
            let bl0_t = b_lo_view.slice(seq![k_base, base_tok], seq![32u32, 16u32]);
            let bl1_t = b_lo_view.slice(seq![k_base, base_tok + 16u32], seq![32u32, 16u32]);
            let mbh0 = cmma::Matrix::<i8>::from_tensor(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                &bh0_t,
            );
            let mbh1 = cmma::Matrix::<i8>::from_tensor(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                &bh1_t,
            );
            let mbl0 = cmma::Matrix::<i8>::from_tensor(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                &bl0_t,
            );
            let mbl1 = cmma::Matrix::<i8>::from_tensor(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                &bl1_t,
            );

            // ── Stage A: 8 unrolled tile blocks; e = r_in*32 + k_in, 2
            //    elements per thread per tile. (Identical to the staging
            //    twin — the A side is NOT the rewrite target.) ──
            let e = tid;
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a0[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a0[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 16u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a1[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 16u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a1[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 32u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a2[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 32u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a2[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 48u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a3[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 48u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a3[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 64u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a4[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 64u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a4[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 80u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a5[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 80u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a5[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 96u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a6[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 96u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a6[e2 as usize] = i8::cast_from(sign);
            }
            {
                let r_in_l = e / 32u32;
                let k_in = e % 32u32;
                let row = base_row + 112u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a7[e as usize] = i8::cast_from(sign);
            }
            {
                let e2 = tid + 256u32;
                let r_in_l = e2 / 32u32;
                let k_in = e2 % 32u32;
                let row = base_row + 112u32 + r_in_l;
                let row_c2 = if row < m { row } else { m - 1u32 };
                let col = k_base + k_in;
                let bitp = col % 32u32;
                let word_off = col / 32u32;
                let posw = pos_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let negw = neg_bits_u32[(row_c2 * words_per_row + word_off) as usize];
                let sign = select((posw >> bitp) & 1u32 != 0u32, one_i, zero_i)
                    - select((negw >> bitp) & 1u32 != 0u32, one_i, zero_i);
                a7[e2 as usize] = i8::cast_from(sign);
            }

            sync_cube();

            // ── MMA: 4 direct B fragments + divergent A per sg (same
            //    execution order as the staging twin — bit-identity gate). ──
            if sg == 0u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a0,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 1u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a1,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 2u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a2,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 3u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a3,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 4u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a4,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 5u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a5,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 6u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a6,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }
            else if sg == 7u32 {
                let ma = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::RowMajor,
                    &a7,
                    32,
                );
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh0, &ahi0, &ahi0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl0, &alo0, &alo0);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh1, &ahi1, &ahi1);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl1, &alo1, &alo1);
            }

            sync_cube();

            ks += 1u32;
        }

        // ── Group boundary: store this group's partials (divergent per sg;
        //    barriers stay OUTSIDE the branches). ──
        if sg == 0u32 {
            cmma::store(&mut g0h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 1u32 {
            cmma::store(&mut g1h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 2u32 {
            cmma::store(&mut g2h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 3u32 {
            cmma::store(&mut g3h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 4u32 {
            cmma::store(&mut g4h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 5u32 {
            cmma::store(&mut g5h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 6u32 {
            cmma::store(&mut g6h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 7u32 {
            cmma::store(&mut g7h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7h1, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7l1, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }

        sync_cube();

        // ── Reduction: each thread updates its 16 outputs (reads only its
        //    OWN sg's buffer set — same ownership as the staging twin). ──
        {
            let sw = group_scale_f32[(row_c * groups_per_row + g) as usize];
            let bh = (r_in * 16u32) as usize;
            if sg == 0u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g0h0[bh]) + f32::cast_from(g0l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g0h0[bh + 1]) + f32::cast_from(g0l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g0h0[bh + 2]) + f32::cast_from(g0l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g0h0[bh + 3]) + f32::cast_from(g0l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g0h0[bh + 4]) + f32::cast_from(g0l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g0h0[bh + 5]) + f32::cast_from(g0l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g0h0[bh + 6]) + f32::cast_from(g0l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g0h0[bh + 7]) + f32::cast_from(g0l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g0h0[bh + 8]) + f32::cast_from(g0l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g0h0[bh + 9]) + f32::cast_from(g0l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g0h0[bh + 10]) + f32::cast_from(g0l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g0h0[bh + 11]) + f32::cast_from(g0l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g0h0[bh + 12]) + f32::cast_from(g0l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g0h0[bh + 13]) + f32::cast_from(g0l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g0h0[bh + 14]) + f32::cast_from(g0l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g0h0[bh + 15]) + f32::cast_from(g0l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g0h1[bh]) + f32::cast_from(g0l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g0h1[bh + 1]) + f32::cast_from(g0l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g0h1[bh + 2]) + f32::cast_from(g0l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g0h1[bh + 3]) + f32::cast_from(g0l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g0h1[bh + 4]) + f32::cast_from(g0l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g0h1[bh + 5]) + f32::cast_from(g0l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g0h1[bh + 6]) + f32::cast_from(g0l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g0h1[bh + 7]) + f32::cast_from(g0l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g0h1[bh + 8]) + f32::cast_from(g0l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g0h1[bh + 9]) + f32::cast_from(g0l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g0h1[bh + 10]) + f32::cast_from(g0l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g0h1[bh + 11]) + f32::cast_from(g0l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g0h1[bh + 12]) + f32::cast_from(g0l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g0h1[bh + 13]) + f32::cast_from(g0l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g0h1[bh + 14]) + f32::cast_from(g0l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g0h1[bh + 15]) + f32::cast_from(g0l1[bh + 15]) * one128);
                }
            }
            else if sg == 1u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g1h0[bh]) + f32::cast_from(g1l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g1h0[bh + 1]) + f32::cast_from(g1l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g1h0[bh + 2]) + f32::cast_from(g1l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g1h0[bh + 3]) + f32::cast_from(g1l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g1h0[bh + 4]) + f32::cast_from(g1l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g1h0[bh + 5]) + f32::cast_from(g1l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g1h0[bh + 6]) + f32::cast_from(g1l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g1h0[bh + 7]) + f32::cast_from(g1l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g1h0[bh + 8]) + f32::cast_from(g1l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g1h0[bh + 9]) + f32::cast_from(g1l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g1h0[bh + 10]) + f32::cast_from(g1l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g1h0[bh + 11]) + f32::cast_from(g1l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g1h0[bh + 12]) + f32::cast_from(g1l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g1h0[bh + 13]) + f32::cast_from(g1l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g1h0[bh + 14]) + f32::cast_from(g1l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g1h0[bh + 15]) + f32::cast_from(g1l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g1h1[bh]) + f32::cast_from(g1l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g1h1[bh + 1]) + f32::cast_from(g1l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g1h1[bh + 2]) + f32::cast_from(g1l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g1h1[bh + 3]) + f32::cast_from(g1l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g1h1[bh + 4]) + f32::cast_from(g1l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g1h1[bh + 5]) + f32::cast_from(g1l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g1h1[bh + 6]) + f32::cast_from(g1l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g1h1[bh + 7]) + f32::cast_from(g1l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g1h1[bh + 8]) + f32::cast_from(g1l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g1h1[bh + 9]) + f32::cast_from(g1l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g1h1[bh + 10]) + f32::cast_from(g1l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g1h1[bh + 11]) + f32::cast_from(g1l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g1h1[bh + 12]) + f32::cast_from(g1l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g1h1[bh + 13]) + f32::cast_from(g1l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g1h1[bh + 14]) + f32::cast_from(g1l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g1h1[bh + 15]) + f32::cast_from(g1l1[bh + 15]) * one128);
                }
            }
            else if sg == 2u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g2h0[bh]) + f32::cast_from(g2l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g2h0[bh + 1]) + f32::cast_from(g2l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g2h0[bh + 2]) + f32::cast_from(g2l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g2h0[bh + 3]) + f32::cast_from(g2l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g2h0[bh + 4]) + f32::cast_from(g2l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g2h0[bh + 5]) + f32::cast_from(g2l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g2h0[bh + 6]) + f32::cast_from(g2l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g2h0[bh + 7]) + f32::cast_from(g2l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g2h0[bh + 8]) + f32::cast_from(g2l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g2h0[bh + 9]) + f32::cast_from(g2l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g2h0[bh + 10]) + f32::cast_from(g2l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g2h0[bh + 11]) + f32::cast_from(g2l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g2h0[bh + 12]) + f32::cast_from(g2l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g2h0[bh + 13]) + f32::cast_from(g2l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g2h0[bh + 14]) + f32::cast_from(g2l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g2h0[bh + 15]) + f32::cast_from(g2l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g2h1[bh]) + f32::cast_from(g2l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g2h1[bh + 1]) + f32::cast_from(g2l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g2h1[bh + 2]) + f32::cast_from(g2l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g2h1[bh + 3]) + f32::cast_from(g2l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g2h1[bh + 4]) + f32::cast_from(g2l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g2h1[bh + 5]) + f32::cast_from(g2l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g2h1[bh + 6]) + f32::cast_from(g2l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g2h1[bh + 7]) + f32::cast_from(g2l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g2h1[bh + 8]) + f32::cast_from(g2l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g2h1[bh + 9]) + f32::cast_from(g2l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g2h1[bh + 10]) + f32::cast_from(g2l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g2h1[bh + 11]) + f32::cast_from(g2l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g2h1[bh + 12]) + f32::cast_from(g2l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g2h1[bh + 13]) + f32::cast_from(g2l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g2h1[bh + 14]) + f32::cast_from(g2l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g2h1[bh + 15]) + f32::cast_from(g2l1[bh + 15]) * one128);
                }
            }
            else if sg == 3u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g3h0[bh]) + f32::cast_from(g3l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g3h0[bh + 1]) + f32::cast_from(g3l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g3h0[bh + 2]) + f32::cast_from(g3l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g3h0[bh + 3]) + f32::cast_from(g3l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g3h0[bh + 4]) + f32::cast_from(g3l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g3h0[bh + 5]) + f32::cast_from(g3l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g3h0[bh + 6]) + f32::cast_from(g3l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g3h0[bh + 7]) + f32::cast_from(g3l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g3h0[bh + 8]) + f32::cast_from(g3l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g3h0[bh + 9]) + f32::cast_from(g3l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g3h0[bh + 10]) + f32::cast_from(g3l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g3h0[bh + 11]) + f32::cast_from(g3l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g3h0[bh + 12]) + f32::cast_from(g3l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g3h0[bh + 13]) + f32::cast_from(g3l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g3h0[bh + 14]) + f32::cast_from(g3l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g3h0[bh + 15]) + f32::cast_from(g3l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g3h1[bh]) + f32::cast_from(g3l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g3h1[bh + 1]) + f32::cast_from(g3l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g3h1[bh + 2]) + f32::cast_from(g3l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g3h1[bh + 3]) + f32::cast_from(g3l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g3h1[bh + 4]) + f32::cast_from(g3l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g3h1[bh + 5]) + f32::cast_from(g3l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g3h1[bh + 6]) + f32::cast_from(g3l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g3h1[bh + 7]) + f32::cast_from(g3l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g3h1[bh + 8]) + f32::cast_from(g3l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g3h1[bh + 9]) + f32::cast_from(g3l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g3h1[bh + 10]) + f32::cast_from(g3l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g3h1[bh + 11]) + f32::cast_from(g3l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g3h1[bh + 12]) + f32::cast_from(g3l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g3h1[bh + 13]) + f32::cast_from(g3l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g3h1[bh + 14]) + f32::cast_from(g3l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g3h1[bh + 15]) + f32::cast_from(g3l1[bh + 15]) * one128);
                }
            }
            else if sg == 4u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g4h0[bh]) + f32::cast_from(g4l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g4h0[bh + 1]) + f32::cast_from(g4l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g4h0[bh + 2]) + f32::cast_from(g4l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g4h0[bh + 3]) + f32::cast_from(g4l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g4h0[bh + 4]) + f32::cast_from(g4l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g4h0[bh + 5]) + f32::cast_from(g4l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g4h0[bh + 6]) + f32::cast_from(g4l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g4h0[bh + 7]) + f32::cast_from(g4l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g4h0[bh + 8]) + f32::cast_from(g4l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g4h0[bh + 9]) + f32::cast_from(g4l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g4h0[bh + 10]) + f32::cast_from(g4l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g4h0[bh + 11]) + f32::cast_from(g4l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g4h0[bh + 12]) + f32::cast_from(g4l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g4h0[bh + 13]) + f32::cast_from(g4l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g4h0[bh + 14]) + f32::cast_from(g4l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g4h0[bh + 15]) + f32::cast_from(g4l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g4h1[bh]) + f32::cast_from(g4l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g4h1[bh + 1]) + f32::cast_from(g4l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g4h1[bh + 2]) + f32::cast_from(g4l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g4h1[bh + 3]) + f32::cast_from(g4l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g4h1[bh + 4]) + f32::cast_from(g4l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g4h1[bh + 5]) + f32::cast_from(g4l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g4h1[bh + 6]) + f32::cast_from(g4l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g4h1[bh + 7]) + f32::cast_from(g4l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g4h1[bh + 8]) + f32::cast_from(g4l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g4h1[bh + 9]) + f32::cast_from(g4l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g4h1[bh + 10]) + f32::cast_from(g4l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g4h1[bh + 11]) + f32::cast_from(g4l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g4h1[bh + 12]) + f32::cast_from(g4l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g4h1[bh + 13]) + f32::cast_from(g4l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g4h1[bh + 14]) + f32::cast_from(g4l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g4h1[bh + 15]) + f32::cast_from(g4l1[bh + 15]) * one128);
                }
            }
            else if sg == 5u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g5h0[bh]) + f32::cast_from(g5l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g5h0[bh + 1]) + f32::cast_from(g5l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g5h0[bh + 2]) + f32::cast_from(g5l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g5h0[bh + 3]) + f32::cast_from(g5l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g5h0[bh + 4]) + f32::cast_from(g5l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g5h0[bh + 5]) + f32::cast_from(g5l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g5h0[bh + 6]) + f32::cast_from(g5l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g5h0[bh + 7]) + f32::cast_from(g5l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g5h0[bh + 8]) + f32::cast_from(g5l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g5h0[bh + 9]) + f32::cast_from(g5l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g5h0[bh + 10]) + f32::cast_from(g5l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g5h0[bh + 11]) + f32::cast_from(g5l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g5h0[bh + 12]) + f32::cast_from(g5l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g5h0[bh + 13]) + f32::cast_from(g5l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g5h0[bh + 14]) + f32::cast_from(g5l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g5h0[bh + 15]) + f32::cast_from(g5l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g5h1[bh]) + f32::cast_from(g5l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g5h1[bh + 1]) + f32::cast_from(g5l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g5h1[bh + 2]) + f32::cast_from(g5l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g5h1[bh + 3]) + f32::cast_from(g5l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g5h1[bh + 4]) + f32::cast_from(g5l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g5h1[bh + 5]) + f32::cast_from(g5l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g5h1[bh + 6]) + f32::cast_from(g5l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g5h1[bh + 7]) + f32::cast_from(g5l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g5h1[bh + 8]) + f32::cast_from(g5l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g5h1[bh + 9]) + f32::cast_from(g5l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g5h1[bh + 10]) + f32::cast_from(g5l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g5h1[bh + 11]) + f32::cast_from(g5l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g5h1[bh + 12]) + f32::cast_from(g5l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g5h1[bh + 13]) + f32::cast_from(g5l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g5h1[bh + 14]) + f32::cast_from(g5l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g5h1[bh + 15]) + f32::cast_from(g5l1[bh + 15]) * one128);
                }
            }
            else if sg == 6u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g6h0[bh]) + f32::cast_from(g6l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g6h0[bh + 1]) + f32::cast_from(g6l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g6h0[bh + 2]) + f32::cast_from(g6l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g6h0[bh + 3]) + f32::cast_from(g6l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g6h0[bh + 4]) + f32::cast_from(g6l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g6h0[bh + 5]) + f32::cast_from(g6l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g6h0[bh + 6]) + f32::cast_from(g6l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g6h0[bh + 7]) + f32::cast_from(g6l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g6h0[bh + 8]) + f32::cast_from(g6l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g6h0[bh + 9]) + f32::cast_from(g6l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g6h0[bh + 10]) + f32::cast_from(g6l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g6h0[bh + 11]) + f32::cast_from(g6l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g6h0[bh + 12]) + f32::cast_from(g6l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g6h0[bh + 13]) + f32::cast_from(g6l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g6h0[bh + 14]) + f32::cast_from(g6l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g6h0[bh + 15]) + f32::cast_from(g6l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g6h1[bh]) + f32::cast_from(g6l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g6h1[bh + 1]) + f32::cast_from(g6l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g6h1[bh + 2]) + f32::cast_from(g6l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g6h1[bh + 3]) + f32::cast_from(g6l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g6h1[bh + 4]) + f32::cast_from(g6l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g6h1[bh + 5]) + f32::cast_from(g6l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g6h1[bh + 6]) + f32::cast_from(g6l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g6h1[bh + 7]) + f32::cast_from(g6l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g6h1[bh + 8]) + f32::cast_from(g6l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g6h1[bh + 9]) + f32::cast_from(g6l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g6h1[bh + 10]) + f32::cast_from(g6l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g6h1[bh + 11]) + f32::cast_from(g6l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g6h1[bh + 12]) + f32::cast_from(g6l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g6h1[bh + 13]) + f32::cast_from(g6l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g6h1[bh + 14]) + f32::cast_from(g6l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g6h1[bh + 15]) + f32::cast_from(g6l1[bh + 15]) * one128);
                }
            }
            else if sg == 7u32 {
                if t_sub == 0u32 {
                    o0 += sw * (f32::cast_from(g7h0[bh]) + f32::cast_from(g7l0[bh]) * one128);
                    o1 += sw * (f32::cast_from(g7h0[bh + 1]) + f32::cast_from(g7l0[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g7h0[bh + 2]) + f32::cast_from(g7l0[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g7h0[bh + 3]) + f32::cast_from(g7l0[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g7h0[bh + 4]) + f32::cast_from(g7l0[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g7h0[bh + 5]) + f32::cast_from(g7l0[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g7h0[bh + 6]) + f32::cast_from(g7l0[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g7h0[bh + 7]) + f32::cast_from(g7l0[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g7h0[bh + 8]) + f32::cast_from(g7l0[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g7h0[bh + 9]) + f32::cast_from(g7l0[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g7h0[bh + 10]) + f32::cast_from(g7l0[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g7h0[bh + 11]) + f32::cast_from(g7l0[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g7h0[bh + 12]) + f32::cast_from(g7l0[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g7h0[bh + 13]) + f32::cast_from(g7l0[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g7h0[bh + 14]) + f32::cast_from(g7l0[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g7h0[bh + 15]) + f32::cast_from(g7l0[bh + 15]) * one128);
                }
                else {
                    o0 += sw * (f32::cast_from(g7h1[bh]) + f32::cast_from(g7l1[bh]) * one128);
                    o1 += sw * (f32::cast_from(g7h1[bh + 1]) + f32::cast_from(g7l1[bh + 1]) * one128);
                    o2 += sw * (f32::cast_from(g7h1[bh + 2]) + f32::cast_from(g7l1[bh + 2]) * one128);
                    o3 += sw * (f32::cast_from(g7h1[bh + 3]) + f32::cast_from(g7l1[bh + 3]) * one128);
                    o4 += sw * (f32::cast_from(g7h1[bh + 4]) + f32::cast_from(g7l1[bh + 4]) * one128);
                    o5 += sw * (f32::cast_from(g7h1[bh + 5]) + f32::cast_from(g7l1[bh + 5]) * one128);
                    o6 += sw * (f32::cast_from(g7h1[bh + 6]) + f32::cast_from(g7l1[bh + 6]) * one128);
                    o7 += sw * (f32::cast_from(g7h1[bh + 7]) + f32::cast_from(g7l1[bh + 7]) * one128);
                    o8 += sw * (f32::cast_from(g7h1[bh + 8]) + f32::cast_from(g7l1[bh + 8]) * one128);
                    o9 += sw * (f32::cast_from(g7h1[bh + 9]) + f32::cast_from(g7l1[bh + 9]) * one128);
                    o10 += sw * (f32::cast_from(g7h1[bh + 10]) + f32::cast_from(g7l1[bh + 10]) * one128);
                    o11 += sw * (f32::cast_from(g7h1[bh + 11]) + f32::cast_from(g7l1[bh + 11]) * one128);
                    o12 += sw * (f32::cast_from(g7h1[bh + 12]) + f32::cast_from(g7l1[bh + 12]) * one128);
                    o13 += sw * (f32::cast_from(g7h1[bh + 13]) + f32::cast_from(g7l1[bh + 13]) * one128);
                    o14 += sw * (f32::cast_from(g7h1[bh + 14]) + f32::cast_from(g7l1[bh + 14]) * one128);
                    o15 += sw * (f32::cast_from(g7h1[bh + 15]) + f32::cast_from(g7l1[bh + 15]) * one128);
                }
            }

            sync_cube();

            g += 1u32;
        }
    }

    // ── Epilogue: apply the per-token scale + guarded write (identical to
    //    the staging twin). ──
    if row_g < m {
        let tok0 = base_tok + t_base;
        if tok0 < p_tokens {
            output_batch[(tok0 * m + row_g) as usize] = o0 * s_t[tok0 as usize];
        }
        let tok1 = tok0 + 1u32;
        if tok1 < p_tokens {
            output_batch[(tok1 * m + row_g) as usize] = o1 * s_t[tok1 as usize];
        }
        let tok2 = tok0 + 2u32;
        if tok2 < p_tokens {
            output_batch[(tok2 * m + row_g) as usize] = o2 * s_t[tok2 as usize];
        }
        let tok3 = tok0 + 3u32;
        if tok3 < p_tokens {
            output_batch[(tok3 * m + row_g) as usize] = o3 * s_t[tok3 as usize];
        }
        let tok4 = tok0 + 4u32;
        if tok4 < p_tokens {
            output_batch[(tok4 * m + row_g) as usize] = o4 * s_t[tok4 as usize];
        }
        let tok5 = tok0 + 5u32;
        if tok5 < p_tokens {
            output_batch[(tok5 * m + row_g) as usize] = o5 * s_t[tok5 as usize];
        }
        let tok6 = tok0 + 6u32;
        if tok6 < p_tokens {
            output_batch[(tok6 * m + row_g) as usize] = o6 * s_t[tok6 as usize];
        }
        let tok7 = tok0 + 7u32;
        if tok7 < p_tokens {
            output_batch[(tok7 * m + row_g) as usize] = o7 * s_t[tok7 as usize];
        }
        let tok8 = tok0 + 8u32;
        if tok8 < p_tokens {
            output_batch[(tok8 * m + row_g) as usize] = o8 * s_t[tok8 as usize];
        }
        let tok9 = tok0 + 9u32;
        if tok9 < p_tokens {
            output_batch[(tok9 * m + row_g) as usize] = o9 * s_t[tok9 as usize];
        }
        let tok10 = tok0 + 10u32;
        if tok10 < p_tokens {
            output_batch[(tok10 * m + row_g) as usize] = o10 * s_t[tok10 as usize];
        }
        let tok11 = tok0 + 11u32;
        if tok11 < p_tokens {
            output_batch[(tok11 * m + row_g) as usize] = o11 * s_t[tok11 as usize];
        }
        let tok12 = tok0 + 12u32;
        if tok12 < p_tokens {
            output_batch[(tok12 * m + row_g) as usize] = o12 * s_t[tok12 as usize];
        }
        let tok13 = tok0 + 13u32;
        if tok13 < p_tokens {
            output_batch[(tok13 * m + row_g) as usize] = o13 * s_t[tok13 as usize];
        }
        let tok14 = tok0 + 14u32;
        if tok14 < p_tokens {
            output_batch[(tok14 * m + row_g) as usize] = o14 * s_t[tok14 as usize];
        }
        let tok15 = tok0 + 15u32;
        if tok15 < p_tokens {
            output_batch[(tok15 * m + row_g) as usize] = o15 * s_t[tok15 as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// Public API (methods on the existing GemmTernaryCmmaI8CubeCL — same
// capability family, same dispatch contract; declared in
// gemm_ternary_cmma_i8_cubecl.rs)
// ---------------------------------------------------------------------------

/// Launch the B-direct variant: byte-quantize pre-pass + direct-load GEMM.
///
/// # Safety
///
/// Same contract as `GemmTernaryCmmaI8CubeCL::launch_sg8`, PLUS the device
/// must report `features.matmul.cmma_tensor_addressing` — call
/// [`tensor_addressing_available`].
#[cfg(feature = "cubecl_runtime")]
pub(crate) unsafe fn launch_direct_gemm<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &TernaryHandle,
    input_handle: Handle,
    output_handle: Handle,
    p_tokens: usize,
) {
    let n = handle.n as u32;
    let m = handle.m as u32;
    debug_assert!(
        n.is_multiple_of(GROUP_COLS),
        "n must be a multiple of {GROUP_COLS} (the ternary group size)"
    );
    let blocks64 = handle.blocks64 as u32;
    let groups_per_row = handle.groups_per_row as u32;
    let p = p_tokens as u32;

    // Quantization scratch: q_hi/q_lo as raw i8 bytes ([p, n] token-major) +
    // per-token scales. Byte count identical to the staging path's packed
    // u32 words (u32-packing was already 1 B/elem).
    let q_bytes = p_tokens * handle.n;
    let q_hi = client.empty(q_bytes);
    let q_lo = client.empty(q_bytes);
    let s_t = client.empty(p_tokens * core::mem::size_of::<f32>());

    unsafe {
        quantize_rows_i8_hilo_bytes::launch_unchecked::<R>(
            client,
            CubeCount::Static(p, 1, 1),
            CubeDim::new_1d(128),
            BufferArg::from_raw_parts(input_handle.clone(), p_tokens * handle.n),
            BufferArg::from_raw_parts(q_hi.clone(), q_bytes),
            BufferArg::from_raw_parts(q_lo.clone(), q_bytes),
            BufferArg::from_raw_parts(s_t.clone(), p_tokens),
            n,
        );
    }

    let num_wg_x = p.div_ceil(TOK_TILE).max(1);
    let num_wg_y = m.div_ceil(ROW_TILE).max(1);

    let pos_len = handle.m * handle.blocks64 * 2;
    let neg_len = pos_len;
    let scale_len = handle.m * handle.groups_per_row;

    unsafe {
        gemm_ternary_cmma_i8_direct::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_wg_x, num_wg_y, 1),
            CubeDim::new_1d(256), // 8 subgroups
            BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
            BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
            BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
            BufferArg::from_raw_parts(q_hi, q_bytes),
            BufferArg::from_raw_parts(q_lo, q_bytes),
            BufferArg::from_raw_parts(s_t, p_tokens),
            BufferArg::from_raw_parts(output_handle, p_tokens * handle.m),
            blocks64,
            groups_per_row,
            n,
            m,
            p,
        );
    }
}

/// Whether the device reports `cmma_tensor_addressing` (the
/// NV_cooperative_matrix2 direct global→matrix load capability the B-direct
/// variant requires).
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn tensor_addressing_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.features().matmul.cmma_tensor_addressing
}
