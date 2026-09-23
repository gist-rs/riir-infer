//! Issue 734 Lever 2 (tile arm) — 128×64 output tiles for the int8 cmma
//! ternary GEMM: `TOK_TILE` 32→64, halving A-side global re-reads + staging
//! per output (the A half of the Bench-709 "staging/reduction data path"
//! wall) at the SAME ~40 KB workgroup smem (two-round group-partial buffer
//! reuse) and the same 2 wgs/SM occupancy class.
//!
//! Design deltas vs the shipping 128×32 staging kernel
//! (`gemm_ternary_cmma_i8_sg8`):
//! - 4 token sub-tiles (bh0..bh3 / bl0..bl3) instead of 2 → 8 mma
//!   accumulators per sg (hi/lo × 4 sub-tiles) and 32 f32 outputs per thread
//!   (thread owns row `tid/2`, token half `tid%2` → 32 consecutive tokens).
//! - The 8 per-group partial tiles round-trip through only 4 i32 shared
//!   buffers per sg (32 KB total, same as the 128×32 kernel) in TWO
//!   store→reduce rounds (sub-tiles 0,1 then 2,3) — partial buffers are
//!   sg-local (only the owning sg's threads read them), so reuse is safe
//!   behind uniform workgroup barriers.
//! - i16 group partials (|max| 128·127 = 16256 < 32767, exact) would halve
//!   the partial smem further, but cubecl 0.11's `cmma::store` is typed by
//!   the accumulator (i32) — an i16 store path is not expressible; recorded
//!   as the open lever if the API ever grows one.
//!
//! Numerics contract — outputs are expected BIT-IDENTICAL to `launch_sg8`:
//! same packed-u32 quantize pre-pass (shared launcher), same A/B fragment
//! VALUES, same per-sub-tile i32 accumulation (4 sequential 16×16×32 mma
//! adds per group — integer-exact), same reduction expression
//! `o += sw·(hi + lo/128)` in the same group order. The G1 gate asserts it.

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
/// Tokens per workgroup (4 sub-tiles of 16).
const TOK_TILE: u32 = 64;
/// Rows per workgroup (8 sub-tiles of 16).
const ROW_TILE: u32 = 128;

// ---------------------------------------------------------------------------
// GEMM kernel
// ---------------------------------------------------------------------------

/// Int8 cooperative-matrix ternary GEMM, 128×64 output-tile variant:
/// `output[p × m] = dequant(W) @ input^T` via i8×i8→i32 @ 16×16×32 tensor-core
/// mma. Workgroup = 256 threads (8 subgroups); each sg owns 16 rows (its A
/// tile) × all 64 tokens. Out-of-range rows/tokens stage CLAMPED data; their
/// outputs are never written back. See the module doc for the full contract.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma_i8_sg8_t64(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    q_hi_w: &[u32],
    q_lo_w: &[u32],
    s_t: &[f32],
    output_batch: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    p_tokens: u32,
) {
    let words_per_row = blocks64 * 2u32;
    let q_words_row = n / 4u32;

    let base_tok = CUBE_POS_X * TOK_TILE;
    let base_row = CUBE_POS_Y * ROW_TILE;
    if base_row >= m || base_tok >= p_tokens {
        terminate!();
    }

    // ── Staging: 8 A sign tiles + 8 B tiles (4 token sub-tiles × hi/lo). ──
    let mut a0 = Shared::<[i8]>::new_slice(512usize);
    let mut a1 = Shared::<[i8]>::new_slice(512usize);
    let mut a2 = Shared::<[i8]>::new_slice(512usize);
    let mut a3 = Shared::<[i8]>::new_slice(512usize);
    let mut a4 = Shared::<[i8]>::new_slice(512usize);
    let mut a5 = Shared::<[i8]>::new_slice(512usize);
    let mut a6 = Shared::<[i8]>::new_slice(512usize);
    let mut a7 = Shared::<[i8]>::new_slice(512usize);
    let mut bh0 = Shared::<[i8]>::new_slice(512usize);
    let mut bl0 = Shared::<[i8]>::new_slice(512usize);
    let mut bh1 = Shared::<[i8]>::new_slice(512usize);
    let mut bl1 = Shared::<[i8]>::new_slice(512usize);
    let mut bh2 = Shared::<[i8]>::new_slice(512usize);
    let mut bl2 = Shared::<[i8]>::new_slice(512usize);
    let mut bh3 = Shared::<[i8]>::new_slice(512usize);
    let mut bl3 = Shared::<[i8]>::new_slice(512usize);

    // ── Per-group partial buffers: 4 per sg (h0/l0/h1/l1), REUSED across the
    //    two store→reduce rounds (round 1 = token sub-tiles 0,1; round 2 =
    //    sub-tiles 2,3). 32 KB total — same as the 128×32 kernel. ──
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

    // ── Output ownership: thread tid owns row `r_own = tid/2`, token half
    //    `t_pair = tid%2` (32 consecutive tokens = 2 sub-tiles). All 32
    //    outputs of a thread live in ONE row (one s_w load per group) and
    //    its OWN sg's buffer set (r_own/16 == tid/32 == sg by
    //    construction). ──
    let r_own = tid / 2u32; // 0..128, row within the tile
    let t_pair = tid % 2u32; // token half: 0 → tokens 0..31, 1 → 32..63
    let row_g = base_row + r_own;
    let row_c = if row_g < m { row_g } else { m - 1u32 };
    let r_in = r_own % 16u32; // row within the 16-row sub-tile
    let t_base = t_pair * 32u32;

    // 32 f32 output accumulators (registers).
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
    let mut o16 = f32::new(0.0f32);
    let mut o17 = f32::new(0.0f32);
    let mut o18 = f32::new(0.0f32);
    let mut o19 = f32::new(0.0f32);
    let mut o20 = f32::new(0.0f32);
    let mut o21 = f32::new(0.0f32);
    let mut o22 = f32::new(0.0f32);
    let mut o23 = f32::new(0.0f32);
    let mut o24 = f32::new(0.0f32);
    let mut o25 = f32::new(0.0f32);
    let mut o26 = f32::new(0.0f32);
    let mut o27 = f32::new(0.0f32);
    let mut o28 = f32::new(0.0f32);
    let mut o29 = f32::new(0.0f32);
    let mut o30 = f32::new(0.0f32);
    let mut o31 = f32::new(0.0f32);

    let one128 = f32::new(1.0f32 / 128.0f32);

    let mut g = 0u32;
    while g < groups_per_row {
        // Fresh i32 accumulators per group (Fill): hi/lo × 4 token sub-tiles.
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
        #[allow(unused_mut)]
        let mut ahi2 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );
        #[allow(unused_mut)]
        let mut alo2 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );
        #[allow(unused_mut)]
        let mut ahi3 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );
        #[allow(unused_mut)]
        let mut alo3 = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize, 16usize, 32usize,
            cmma::MatrixLayout::Undefined,
            0i32,
        );

        let mut ks = 0u32;
        while ks < 4u32 {
            let k_base = g * GROUP_COLS + ks * K_STEP;

            // ── Stage A: 8 unrolled tile blocks; e = r_in*32 + k_in, 2
            //    elements per thread per tile. ──
            // A tile 0.
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
            // A tile 1.
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
            // A tile 2.
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
            // A tile 3.
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
            // A tile 4.
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
            // A tile 5.
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
            // A tile 6.
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
            // A tile 7.
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

            // ── Stage B: 8 unrolled tiles; TOKEN-MAJOR layout e = t_in*32 +
            //    k_in (k contiguous within a token row) — `from_slice(B,
            //    ColMajor, stride=K)` reads fragment(k, n) at buf[n*K + k].
            //    Sub-tile s covers tokens base_tok + s*16 + t_in. ──
            // bh0 (token sub-tile 0, hi).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh0[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh0[e2 as usize] = i8::cast_from(v_i);
            }
            // bh1 (token sub-tile 1, hi).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + 16u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh1[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + 16u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh1[e2 as usize] = i8::cast_from(v_i);
            }
            // bh2 (token sub-tile 2, hi).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + 32u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh2[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + 32u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh2[e2 as usize] = i8::cast_from(v_i);
            }
            // bh3 (token sub-tile 3, hi).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + 48u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh3[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + 48u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_hi_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bh3[e2 as usize] = i8::cast_from(v_i);
            }
            // bl0 (token sub-tile 0, lo).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl0[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl0[e2 as usize] = i8::cast_from(v_i);
            }
            // bl1 (token sub-tile 1, lo).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + 16u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl1[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + 16u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl1[e2 as usize] = i8::cast_from(v_i);
            }
            // bl2 (token sub-tile 2, lo).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + 32u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl2[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + 32u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl2[e2 as usize] = i8::cast_from(v_i);
            }
            // bl3 (token sub-tile 3, lo).
            {
                let t_in = e / 32u32;
                let k_in = e % 32u32;
                let tok = base_tok + 48u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl3[e as usize] = i8::cast_from(v_i);
            }
            {
                let e2 = tid + 256u32;
                let t_in = e2 / 32u32;
                let k_in = e2 % 32u32;
                let tok = base_tok + 48u32 + t_in;
                let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
                let col = k_base + k_in;
                let qw = q_lo_w[(tok_c * q_words_row + col / 4u32) as usize];
                let byte = ((qw >> ((col % 4u32) * 8u32)) & 0xFFu32) as i32;
                let v_i = (byte << 24) >> 24; // sign-extend
                bl3[e2 as usize] = i8::cast_from(v_i);
            }

            sync_cube();

            // ── MMA: 8 uniform B fragments + divergent A per sg (8 executes:
            //    4 token sub-tiles × hi/lo). ──
            let mbh0 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bh0,
                32,
            );
            let mbl0 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bl0,
                32,
            );
            let mbh1 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bh1,
                32,
            );
            let mbl1 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bl1,
                32,
            );
            let mbh2 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bh2,
                32,
            );
            let mbl2 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bl2,
                32,
            );
            let mbh3 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bh3,
                32,
            );
            let mbl3 = cmma::Matrix::<i8>::from_slice(
                cmma::MatrixIdent::B,
                16usize, 16usize, 32usize,
                cmma::MatrixLayout::ColMajor,
                &bl3,
                32,
            );
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
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
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh2, &ahi2, &ahi2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl2, &alo2, &alo2);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbh3, &ahi3, &ahi3);
                cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mbl3, &alo3, &alo3);
            }

            sync_cube();

            ks += 1u32;
        }

        // ── Group boundary, ROUND 1: store sub-tiles 0,1 partials (divergent
        //    per sg; barriers stay OUTSIDE the branches). ──
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

        // ── Reduce ROUND 1: t_pair==0 threads update their 32 outputs
        //    (sub-tile 0 → o0..o15 from h0/l0; sub-tile 1 → o16..o31 from
        //    h1/l1). Same arithmetic + order as the 128×32 kernel. ──
        {
            let sw = group_scale_f32[(row_c * groups_per_row + g) as usize];
            let bh = (r_in * 16u32) as usize; // base index into a 16×16 tile
            if t_pair == 0u32 {
                if sg == 0u32 {
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
                    o16 += sw * (f32::cast_from(g0h1[bh]) + f32::cast_from(g0l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g0h1[bh + 1]) + f32::cast_from(g0l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g0h1[bh + 2]) + f32::cast_from(g0l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g0h1[bh + 3]) + f32::cast_from(g0l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g0h1[bh + 4]) + f32::cast_from(g0l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g0h1[bh + 5]) + f32::cast_from(g0l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g0h1[bh + 6]) + f32::cast_from(g0l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g0h1[bh + 7]) + f32::cast_from(g0l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g0h1[bh + 8]) + f32::cast_from(g0l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g0h1[bh + 9]) + f32::cast_from(g0l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g0h1[bh + 10]) + f32::cast_from(g0l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g0h1[bh + 11]) + f32::cast_from(g0l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g0h1[bh + 12]) + f32::cast_from(g0l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g0h1[bh + 13]) + f32::cast_from(g0l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g0h1[bh + 14]) + f32::cast_from(g0l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g0h1[bh + 15]) + f32::cast_from(g0l1[bh + 15]) * one128);
                }
                else if sg == 1u32 {
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
                    o16 += sw * (f32::cast_from(g1h1[bh]) + f32::cast_from(g1l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g1h1[bh + 1]) + f32::cast_from(g1l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g1h1[bh + 2]) + f32::cast_from(g1l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g1h1[bh + 3]) + f32::cast_from(g1l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g1h1[bh + 4]) + f32::cast_from(g1l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g1h1[bh + 5]) + f32::cast_from(g1l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g1h1[bh + 6]) + f32::cast_from(g1l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g1h1[bh + 7]) + f32::cast_from(g1l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g1h1[bh + 8]) + f32::cast_from(g1l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g1h1[bh + 9]) + f32::cast_from(g1l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g1h1[bh + 10]) + f32::cast_from(g1l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g1h1[bh + 11]) + f32::cast_from(g1l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g1h1[bh + 12]) + f32::cast_from(g1l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g1h1[bh + 13]) + f32::cast_from(g1l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g1h1[bh + 14]) + f32::cast_from(g1l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g1h1[bh + 15]) + f32::cast_from(g1l1[bh + 15]) * one128);
                }
                else if sg == 2u32 {
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
                    o16 += sw * (f32::cast_from(g2h1[bh]) + f32::cast_from(g2l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g2h1[bh + 1]) + f32::cast_from(g2l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g2h1[bh + 2]) + f32::cast_from(g2l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g2h1[bh + 3]) + f32::cast_from(g2l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g2h1[bh + 4]) + f32::cast_from(g2l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g2h1[bh + 5]) + f32::cast_from(g2l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g2h1[bh + 6]) + f32::cast_from(g2l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g2h1[bh + 7]) + f32::cast_from(g2l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g2h1[bh + 8]) + f32::cast_from(g2l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g2h1[bh + 9]) + f32::cast_from(g2l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g2h1[bh + 10]) + f32::cast_from(g2l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g2h1[bh + 11]) + f32::cast_from(g2l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g2h1[bh + 12]) + f32::cast_from(g2l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g2h1[bh + 13]) + f32::cast_from(g2l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g2h1[bh + 14]) + f32::cast_from(g2l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g2h1[bh + 15]) + f32::cast_from(g2l1[bh + 15]) * one128);
                }
                else if sg == 3u32 {
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
                    o16 += sw * (f32::cast_from(g3h1[bh]) + f32::cast_from(g3l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g3h1[bh + 1]) + f32::cast_from(g3l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g3h1[bh + 2]) + f32::cast_from(g3l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g3h1[bh + 3]) + f32::cast_from(g3l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g3h1[bh + 4]) + f32::cast_from(g3l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g3h1[bh + 5]) + f32::cast_from(g3l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g3h1[bh + 6]) + f32::cast_from(g3l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g3h1[bh + 7]) + f32::cast_from(g3l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g3h1[bh + 8]) + f32::cast_from(g3l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g3h1[bh + 9]) + f32::cast_from(g3l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g3h1[bh + 10]) + f32::cast_from(g3l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g3h1[bh + 11]) + f32::cast_from(g3l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g3h1[bh + 12]) + f32::cast_from(g3l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g3h1[bh + 13]) + f32::cast_from(g3l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g3h1[bh + 14]) + f32::cast_from(g3l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g3h1[bh + 15]) + f32::cast_from(g3l1[bh + 15]) * one128);
                }
                else if sg == 4u32 {
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
                    o16 += sw * (f32::cast_from(g4h1[bh]) + f32::cast_from(g4l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g4h1[bh + 1]) + f32::cast_from(g4l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g4h1[bh + 2]) + f32::cast_from(g4l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g4h1[bh + 3]) + f32::cast_from(g4l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g4h1[bh + 4]) + f32::cast_from(g4l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g4h1[bh + 5]) + f32::cast_from(g4l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g4h1[bh + 6]) + f32::cast_from(g4l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g4h1[bh + 7]) + f32::cast_from(g4l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g4h1[bh + 8]) + f32::cast_from(g4l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g4h1[bh + 9]) + f32::cast_from(g4l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g4h1[bh + 10]) + f32::cast_from(g4l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g4h1[bh + 11]) + f32::cast_from(g4l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g4h1[bh + 12]) + f32::cast_from(g4l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g4h1[bh + 13]) + f32::cast_from(g4l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g4h1[bh + 14]) + f32::cast_from(g4l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g4h1[bh + 15]) + f32::cast_from(g4l1[bh + 15]) * one128);
                }
                else if sg == 5u32 {
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
                    o16 += sw * (f32::cast_from(g5h1[bh]) + f32::cast_from(g5l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g5h1[bh + 1]) + f32::cast_from(g5l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g5h1[bh + 2]) + f32::cast_from(g5l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g5h1[bh + 3]) + f32::cast_from(g5l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g5h1[bh + 4]) + f32::cast_from(g5l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g5h1[bh + 5]) + f32::cast_from(g5l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g5h1[bh + 6]) + f32::cast_from(g5l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g5h1[bh + 7]) + f32::cast_from(g5l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g5h1[bh + 8]) + f32::cast_from(g5l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g5h1[bh + 9]) + f32::cast_from(g5l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g5h1[bh + 10]) + f32::cast_from(g5l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g5h1[bh + 11]) + f32::cast_from(g5l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g5h1[bh + 12]) + f32::cast_from(g5l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g5h1[bh + 13]) + f32::cast_from(g5l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g5h1[bh + 14]) + f32::cast_from(g5l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g5h1[bh + 15]) + f32::cast_from(g5l1[bh + 15]) * one128);
                }
                else if sg == 6u32 {
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
                    o16 += sw * (f32::cast_from(g6h1[bh]) + f32::cast_from(g6l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g6h1[bh + 1]) + f32::cast_from(g6l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g6h1[bh + 2]) + f32::cast_from(g6l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g6h1[bh + 3]) + f32::cast_from(g6l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g6h1[bh + 4]) + f32::cast_from(g6l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g6h1[bh + 5]) + f32::cast_from(g6l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g6h1[bh + 6]) + f32::cast_from(g6l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g6h1[bh + 7]) + f32::cast_from(g6l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g6h1[bh + 8]) + f32::cast_from(g6l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g6h1[bh + 9]) + f32::cast_from(g6l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g6h1[bh + 10]) + f32::cast_from(g6l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g6h1[bh + 11]) + f32::cast_from(g6l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g6h1[bh + 12]) + f32::cast_from(g6l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g6h1[bh + 13]) + f32::cast_from(g6l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g6h1[bh + 14]) + f32::cast_from(g6l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g6h1[bh + 15]) + f32::cast_from(g6l1[bh + 15]) * one128);
                }
                else if sg == 7u32 {
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
                    o16 += sw * (f32::cast_from(g7h1[bh]) + f32::cast_from(g7l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g7h1[bh + 1]) + f32::cast_from(g7l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g7h1[bh + 2]) + f32::cast_from(g7l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g7h1[bh + 3]) + f32::cast_from(g7l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g7h1[bh + 4]) + f32::cast_from(g7l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g7h1[bh + 5]) + f32::cast_from(g7l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g7h1[bh + 6]) + f32::cast_from(g7l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g7h1[bh + 7]) + f32::cast_from(g7l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g7h1[bh + 8]) + f32::cast_from(g7l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g7h1[bh + 9]) + f32::cast_from(g7l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g7h1[bh + 10]) + f32::cast_from(g7l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g7h1[bh + 11]) + f32::cast_from(g7l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g7h1[bh + 12]) + f32::cast_from(g7l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g7h1[bh + 13]) + f32::cast_from(g7l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g7h1[bh + 14]) + f32::cast_from(g7l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g7h1[bh + 15]) + f32::cast_from(g7l1[bh + 15]) * one128);
                }
            }
        }

        sync_cube();

        // ── Group boundary, ROUND 2: store sub-tiles 2,3 partials into the
        //    SAME per-sg buffers. ──
        if sg == 0u32 {
            cmma::store(&mut g0h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 1u32 {
            cmma::store(&mut g1h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 2u32 {
            cmma::store(&mut g2h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 3u32 {
            cmma::store(&mut g3h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 4u32 {
            cmma::store(&mut g4h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 5u32 {
            cmma::store(&mut g5h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 6u32 {
            cmma::store(&mut g6h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 7u32 {
            cmma::store(&mut g7h0, &ahi2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7l0, &alo2, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7h1, &ahi3, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7l1, &alo3, 16, cmma::MatrixLayout::RowMajor);
        }

        sync_cube();

        // ── Reduce ROUND 2: t_pair==1 threads update their 32 outputs
        //    (sub-tile 2 → o0..o15 from h0/l0; sub-tile 3 → o16..o31 from
        //    h1/l1 — the round-2 stores overwrote the same buffers). ──
        {
            let sw = group_scale_f32[(row_c * groups_per_row + g) as usize];
            let bh = (r_in * 16u32) as usize; // base index into a 16×16 tile
            if t_pair == 1u32 {
                if sg == 0u32 {
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
                    o16 += sw * (f32::cast_from(g0h1[bh]) + f32::cast_from(g0l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g0h1[bh + 1]) + f32::cast_from(g0l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g0h1[bh + 2]) + f32::cast_from(g0l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g0h1[bh + 3]) + f32::cast_from(g0l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g0h1[bh + 4]) + f32::cast_from(g0l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g0h1[bh + 5]) + f32::cast_from(g0l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g0h1[bh + 6]) + f32::cast_from(g0l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g0h1[bh + 7]) + f32::cast_from(g0l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g0h1[bh + 8]) + f32::cast_from(g0l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g0h1[bh + 9]) + f32::cast_from(g0l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g0h1[bh + 10]) + f32::cast_from(g0l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g0h1[bh + 11]) + f32::cast_from(g0l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g0h1[bh + 12]) + f32::cast_from(g0l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g0h1[bh + 13]) + f32::cast_from(g0l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g0h1[bh + 14]) + f32::cast_from(g0l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g0h1[bh + 15]) + f32::cast_from(g0l1[bh + 15]) * one128);
                }
                else if sg == 1u32 {
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
                    o16 += sw * (f32::cast_from(g1h1[bh]) + f32::cast_from(g1l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g1h1[bh + 1]) + f32::cast_from(g1l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g1h1[bh + 2]) + f32::cast_from(g1l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g1h1[bh + 3]) + f32::cast_from(g1l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g1h1[bh + 4]) + f32::cast_from(g1l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g1h1[bh + 5]) + f32::cast_from(g1l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g1h1[bh + 6]) + f32::cast_from(g1l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g1h1[bh + 7]) + f32::cast_from(g1l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g1h1[bh + 8]) + f32::cast_from(g1l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g1h1[bh + 9]) + f32::cast_from(g1l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g1h1[bh + 10]) + f32::cast_from(g1l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g1h1[bh + 11]) + f32::cast_from(g1l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g1h1[bh + 12]) + f32::cast_from(g1l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g1h1[bh + 13]) + f32::cast_from(g1l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g1h1[bh + 14]) + f32::cast_from(g1l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g1h1[bh + 15]) + f32::cast_from(g1l1[bh + 15]) * one128);
                }
                else if sg == 2u32 {
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
                    o16 += sw * (f32::cast_from(g2h1[bh]) + f32::cast_from(g2l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g2h1[bh + 1]) + f32::cast_from(g2l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g2h1[bh + 2]) + f32::cast_from(g2l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g2h1[bh + 3]) + f32::cast_from(g2l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g2h1[bh + 4]) + f32::cast_from(g2l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g2h1[bh + 5]) + f32::cast_from(g2l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g2h1[bh + 6]) + f32::cast_from(g2l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g2h1[bh + 7]) + f32::cast_from(g2l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g2h1[bh + 8]) + f32::cast_from(g2l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g2h1[bh + 9]) + f32::cast_from(g2l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g2h1[bh + 10]) + f32::cast_from(g2l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g2h1[bh + 11]) + f32::cast_from(g2l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g2h1[bh + 12]) + f32::cast_from(g2l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g2h1[bh + 13]) + f32::cast_from(g2l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g2h1[bh + 14]) + f32::cast_from(g2l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g2h1[bh + 15]) + f32::cast_from(g2l1[bh + 15]) * one128);
                }
                else if sg == 3u32 {
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
                    o16 += sw * (f32::cast_from(g3h1[bh]) + f32::cast_from(g3l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g3h1[bh + 1]) + f32::cast_from(g3l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g3h1[bh + 2]) + f32::cast_from(g3l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g3h1[bh + 3]) + f32::cast_from(g3l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g3h1[bh + 4]) + f32::cast_from(g3l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g3h1[bh + 5]) + f32::cast_from(g3l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g3h1[bh + 6]) + f32::cast_from(g3l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g3h1[bh + 7]) + f32::cast_from(g3l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g3h1[bh + 8]) + f32::cast_from(g3l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g3h1[bh + 9]) + f32::cast_from(g3l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g3h1[bh + 10]) + f32::cast_from(g3l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g3h1[bh + 11]) + f32::cast_from(g3l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g3h1[bh + 12]) + f32::cast_from(g3l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g3h1[bh + 13]) + f32::cast_from(g3l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g3h1[bh + 14]) + f32::cast_from(g3l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g3h1[bh + 15]) + f32::cast_from(g3l1[bh + 15]) * one128);
                }
                else if sg == 4u32 {
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
                    o16 += sw * (f32::cast_from(g4h1[bh]) + f32::cast_from(g4l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g4h1[bh + 1]) + f32::cast_from(g4l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g4h1[bh + 2]) + f32::cast_from(g4l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g4h1[bh + 3]) + f32::cast_from(g4l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g4h1[bh + 4]) + f32::cast_from(g4l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g4h1[bh + 5]) + f32::cast_from(g4l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g4h1[bh + 6]) + f32::cast_from(g4l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g4h1[bh + 7]) + f32::cast_from(g4l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g4h1[bh + 8]) + f32::cast_from(g4l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g4h1[bh + 9]) + f32::cast_from(g4l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g4h1[bh + 10]) + f32::cast_from(g4l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g4h1[bh + 11]) + f32::cast_from(g4l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g4h1[bh + 12]) + f32::cast_from(g4l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g4h1[bh + 13]) + f32::cast_from(g4l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g4h1[bh + 14]) + f32::cast_from(g4l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g4h1[bh + 15]) + f32::cast_from(g4l1[bh + 15]) * one128);
                }
                else if sg == 5u32 {
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
                    o16 += sw * (f32::cast_from(g5h1[bh]) + f32::cast_from(g5l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g5h1[bh + 1]) + f32::cast_from(g5l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g5h1[bh + 2]) + f32::cast_from(g5l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g5h1[bh + 3]) + f32::cast_from(g5l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g5h1[bh + 4]) + f32::cast_from(g5l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g5h1[bh + 5]) + f32::cast_from(g5l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g5h1[bh + 6]) + f32::cast_from(g5l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g5h1[bh + 7]) + f32::cast_from(g5l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g5h1[bh + 8]) + f32::cast_from(g5l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g5h1[bh + 9]) + f32::cast_from(g5l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g5h1[bh + 10]) + f32::cast_from(g5l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g5h1[bh + 11]) + f32::cast_from(g5l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g5h1[bh + 12]) + f32::cast_from(g5l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g5h1[bh + 13]) + f32::cast_from(g5l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g5h1[bh + 14]) + f32::cast_from(g5l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g5h1[bh + 15]) + f32::cast_from(g5l1[bh + 15]) * one128);
                }
                else if sg == 6u32 {
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
                    o16 += sw * (f32::cast_from(g6h1[bh]) + f32::cast_from(g6l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g6h1[bh + 1]) + f32::cast_from(g6l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g6h1[bh + 2]) + f32::cast_from(g6l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g6h1[bh + 3]) + f32::cast_from(g6l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g6h1[bh + 4]) + f32::cast_from(g6l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g6h1[bh + 5]) + f32::cast_from(g6l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g6h1[bh + 6]) + f32::cast_from(g6l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g6h1[bh + 7]) + f32::cast_from(g6l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g6h1[bh + 8]) + f32::cast_from(g6l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g6h1[bh + 9]) + f32::cast_from(g6l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g6h1[bh + 10]) + f32::cast_from(g6l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g6h1[bh + 11]) + f32::cast_from(g6l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g6h1[bh + 12]) + f32::cast_from(g6l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g6h1[bh + 13]) + f32::cast_from(g6l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g6h1[bh + 14]) + f32::cast_from(g6l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g6h1[bh + 15]) + f32::cast_from(g6l1[bh + 15]) * one128);
                }
                else if sg == 7u32 {
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
                    o16 += sw * (f32::cast_from(g7h1[bh]) + f32::cast_from(g7l1[bh]) * one128);
                    o17 += sw * (f32::cast_from(g7h1[bh + 1]) + f32::cast_from(g7l1[bh + 1]) * one128);
                    o18 += sw * (f32::cast_from(g7h1[bh + 2]) + f32::cast_from(g7l1[bh + 2]) * one128);
                    o19 += sw * (f32::cast_from(g7h1[bh + 3]) + f32::cast_from(g7l1[bh + 3]) * one128);
                    o20 += sw * (f32::cast_from(g7h1[bh + 4]) + f32::cast_from(g7l1[bh + 4]) * one128);
                    o21 += sw * (f32::cast_from(g7h1[bh + 5]) + f32::cast_from(g7l1[bh + 5]) * one128);
                    o22 += sw * (f32::cast_from(g7h1[bh + 6]) + f32::cast_from(g7l1[bh + 6]) * one128);
                    o23 += sw * (f32::cast_from(g7h1[bh + 7]) + f32::cast_from(g7l1[bh + 7]) * one128);
                    o24 += sw * (f32::cast_from(g7h1[bh + 8]) + f32::cast_from(g7l1[bh + 8]) * one128);
                    o25 += sw * (f32::cast_from(g7h1[bh + 9]) + f32::cast_from(g7l1[bh + 9]) * one128);
                    o26 += sw * (f32::cast_from(g7h1[bh + 10]) + f32::cast_from(g7l1[bh + 10]) * one128);
                    o27 += sw * (f32::cast_from(g7h1[bh + 11]) + f32::cast_from(g7l1[bh + 11]) * one128);
                    o28 += sw * (f32::cast_from(g7h1[bh + 12]) + f32::cast_from(g7l1[bh + 12]) * one128);
                    o29 += sw * (f32::cast_from(g7h1[bh + 13]) + f32::cast_from(g7l1[bh + 13]) * one128);
                    o30 += sw * (f32::cast_from(g7h1[bh + 14]) + f32::cast_from(g7l1[bh + 14]) * one128);
                    o31 += sw * (f32::cast_from(g7h1[bh + 15]) + f32::cast_from(g7l1[bh + 15]) * one128);
                }
            }
        }

        sync_cube();

        g += 1u32;
    }

    // ── Epilogue: apply the per-token scale + guarded write. Each thread
    //    writes its 32 outputs (row base_row + r_own, tokens base_tok +
    //    t_base + j). ──
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
        let tok16 = tok0 + 16u32;
        if tok16 < p_tokens {
            output_batch[(tok16 * m + row_g) as usize] = o16 * s_t[tok16 as usize];
        }
        let tok17 = tok0 + 17u32;
        if tok17 < p_tokens {
            output_batch[(tok17 * m + row_g) as usize] = o17 * s_t[tok17 as usize];
        }
        let tok18 = tok0 + 18u32;
        if tok18 < p_tokens {
            output_batch[(tok18 * m + row_g) as usize] = o18 * s_t[tok18 as usize];
        }
        let tok19 = tok0 + 19u32;
        if tok19 < p_tokens {
            output_batch[(tok19 * m + row_g) as usize] = o19 * s_t[tok19 as usize];
        }
        let tok20 = tok0 + 20u32;
        if tok20 < p_tokens {
            output_batch[(tok20 * m + row_g) as usize] = o20 * s_t[tok20 as usize];
        }
        let tok21 = tok0 + 21u32;
        if tok21 < p_tokens {
            output_batch[(tok21 * m + row_g) as usize] = o21 * s_t[tok21 as usize];
        }
        let tok22 = tok0 + 22u32;
        if tok22 < p_tokens {
            output_batch[(tok22 * m + row_g) as usize] = o22 * s_t[tok22 as usize];
        }
        let tok23 = tok0 + 23u32;
        if tok23 < p_tokens {
            output_batch[(tok23 * m + row_g) as usize] = o23 * s_t[tok23 as usize];
        }
        let tok24 = tok0 + 24u32;
        if tok24 < p_tokens {
            output_batch[(tok24 * m + row_g) as usize] = o24 * s_t[tok24 as usize];
        }
        let tok25 = tok0 + 25u32;
        if tok25 < p_tokens {
            output_batch[(tok25 * m + row_g) as usize] = o25 * s_t[tok25 as usize];
        }
        let tok26 = tok0 + 26u32;
        if tok26 < p_tokens {
            output_batch[(tok26 * m + row_g) as usize] = o26 * s_t[tok26 as usize];
        }
        let tok27 = tok0 + 27u32;
        if tok27 < p_tokens {
            output_batch[(tok27 * m + row_g) as usize] = o27 * s_t[tok27 as usize];
        }
        let tok28 = tok0 + 28u32;
        if tok28 < p_tokens {
            output_batch[(tok28 * m + row_g) as usize] = o28 * s_t[tok28 as usize];
        }
        let tok29 = tok0 + 29u32;
        if tok29 < p_tokens {
            output_batch[(tok29 * m + row_g) as usize] = o29 * s_t[tok29 as usize];
        }
        let tok30 = tok0 + 30u32;
        if tok30 < p_tokens {
            output_batch[(tok30 * m + row_g) as usize] = o30 * s_t[tok30 as usize];
        }
        let tok31 = tok0 + 31u32;
        if tok31 < p_tokens {
            output_batch[(tok31 * m + row_g) as usize] = o31 * s_t[tok31 as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// Public API (methods on the existing GemmTernaryCmmaI8CubeCL — same
// capability family, same dispatch contract; declared in
// gemm_ternary_cmma_i8_cubecl.rs)
// ---------------------------------------------------------------------------

/// Launch the 128×64-tile variant: packed-u32 quantize pre-pass (the SAME
/// pre-pass as the staging kernel — identical q values are part of the
/// bit-identity contract) + the t64 GEMM.
///
/// # Safety
///
/// Same contract as `GemmTernaryCmmaI8CubeCL::launch_sg8`.
#[cfg(feature = "cubecl_runtime")]
pub(crate) unsafe fn launch_t64_gemm<R: Runtime>(
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

    // Quantization scratch: q_hi/q_lo packed u32 (n/4 words per row) +
    // per-token scales. Allocated + launched via the shared helper so the
    // pre-pass is byte-identical to the staging kernel's.
    let q_words = p_tokens * (handle.n / 4);
    let q_hi = client.empty(q_words * core::mem::size_of::<u32>());
    let q_lo = client.empty(q_words * core::mem::size_of::<u32>());
    let s_t = client.empty(p_tokens * core::mem::size_of::<f32>());
    crate::gemm_ternary_cmma_i8_cubecl::launch_quantize_hilo_packed::<R>(
        client,
        &input_handle,
        &q_hi,
        &q_lo,
        &s_t,
        p_tokens,
        n,
    );

    let num_wg_x = p.div_ceil(TOK_TILE).max(1);
    let num_wg_y = m.div_ceil(ROW_TILE).max(1);

    let pos_len = handle.m * handle.blocks64 * 2;
    let neg_len = pos_len;
    let scale_len = handle.m * handle.groups_per_row;

    unsafe {
        gemm_ternary_cmma_i8_sg8_t64::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_wg_x, num_wg_y, 1),
            CubeDim::new_1d(256), // 8 subgroups
            BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
            BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
            BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
            BufferArg::from_raw_parts(q_hi, q_words),
            BufferArg::from_raw_parts(q_lo, q_words),
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
