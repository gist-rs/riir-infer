//! Issue 734 Arm 5 (Bench 718) — occupancy via two-round partials split for
//! the int8 cmma ternary GEMM. IDENTICAL staging + pipelined k-loop to the
//! shipping `gemm_ternary_cmma_i8_sg8`; only the per-group partial
//! store→reduce is split into TWO rounds over 16 buffers instead of ONE round
//! over 32:
//!
//! - partials 32 KB → 16 KB, workgroup smem 44 → 28 KB → **3 wgs/SM on the
//!   4090** (was 2) — the occupancy attack on the staging phase's
//!   DRAM-latency wall (Bench 715/717: latency-bound at 2 wgs/SM; four shrink
//!   axes closed neutral — the phase needs MORE RESIDENT WARPS, not fewer
//!   instructions).
//! - ROUND 0: store {ahi0, alo0} → {g*h0, g*l0}; barrier; t_sub==0 threads
//!   reduce. ROUND 1: store {ahi1, alo1} → the SAME buffers; barrier;
//!   t_sub==1 threads reduce. Partial buffers are sg-local (only the owning
//!   sg's threads read them), so reuse behind uniform workgroup barriers is
//!   safe — the `gemm_ternary_cmma_i8_t64_cubecl` round-reuse precedent.
//! - +2 barriers per group (the post-store + trailing syncs already exist
//!   and are reused).
//!
//! Numerics contract — outputs are expected BIT-IDENTICAL to `launch_sg8`:
//! same packed-u32 quantize pre-pass (shared launcher), same A/B fragment
//! values, same pipelined k-loop, same per-thread reduce expression
//! `o += sw·(hi + lo/128)` in the same group order (each thread's
//! contribution is added exactly once per group, in its own round). The G1
//! gate asserts it via `RIIR_CMMA_I8_PSPLIT=1` + the Bench-710 FNV pins.

use cubecl::cmma;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::TernaryHandle;

/// Ternary group size = weight-scale group = 128 cols.
const GROUP_COLS: u32 = 128;
/// Tokens per workgroup (2 sub-tiles of 16).
const TOK_TILE: u32 = 32;
/// Rows per workgroup (8 sub-tiles of 16).
const ROW_TILE: u32 = 128;

// ---------------------------------------------------------------------------
// GEMM kernel
// ---------------------------------------------------------------------------

/// Int8 cooperative-matrix ternary GEMM, two-round-partials variant:
/// `output[p × m] = dequant(W) @ input^T` via i8×i8→i32 @ 16×16×32 tensor-core
/// mma. Workgroup = 256 threads (8 subgroups); tile = 128 rows × 32 tokens;
/// 28 KB workgroup smem → 3 wgs/SM on the 4090. See the module doc.
/// Int8 cooperative-matrix ternary GEMM, two-round-partials variant (Arm 5):
/// staging + pipelined k-loop identical to `gemm_ternary_cmma_i8_sg8`; the
/// per-group partials round-trip through two store→reduce rounds over 16
/// buffers (28 KB workgroup smem → 3 wgs/SM on the 4090). See the module doc
/// for the contract.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::collapsible_if, reason = "the measured round-reduce form")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma_i8_sg8_psplit(
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

    // ── Staging: 8 A sign tiles + 4 B tiles (2 token sub-tiles × hi/lo). ──
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
    // Arm-2 double-buffer SET1 (Bench 716): staging(ks+1) lands here while
    // mma(ks) reads SET0 (and vice versa) — no barrier between them.
    let mut a0b = Shared::<[i8]>::new_slice(512usize);
    let mut a1b = Shared::<[i8]>::new_slice(512usize);
    let mut a2b = Shared::<[i8]>::new_slice(512usize);
    let mut a3b = Shared::<[i8]>::new_slice(512usize);
    let mut a4b = Shared::<[i8]>::new_slice(512usize);
    let mut a5b = Shared::<[i8]>::new_slice(512usize);
    let mut a6b = Shared::<[i8]>::new_slice(512usize);
    let mut a7b = Shared::<[i8]>::new_slice(512usize);
    let mut bh0b = Shared::<[i8]>::new_slice(512usize);
    let mut bl0b = Shared::<[i8]>::new_slice(512usize);
    let mut bh1b = Shared::<[i8]>::new_slice(512usize);
    let mut bl1b = Shared::<[i8]>::new_slice(512usize);

    // ── Per-group partial stores: 2 named buffers per sg (16 total;
    //    TWO store→reduce rounds per group — round 1 reuses round 0's
    //    buffers, the t64 round-reuse precedent). 16 KB, not 32. ──
    let mut g0h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g0l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g1h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g1l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g2h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g2l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g3h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g3l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g4h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g4l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g5h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g5l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g6h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g6l0 = Shared::<[i32]>::new_slice(256usize);
    let mut g7h0 = Shared::<[i32]>::new_slice(256usize);
    let mut g7l0 = Shared::<[i32]>::new_slice(256usize);

    let tid = UNIT_POS; // 0..256
    let sg = tid / 32u32;

    // ── Output ownership: thread tid owns row `r_own = tid/2`, token
    //    sub-tile `t_sub = tid%2`, 16 consecutive tokens within it. All 16
    //    outputs of a thread live in ONE row (one s_w load per group) and
    //    its OWN sg's buffer set (r_own/16 == tid/32 == sg by
    //    construction). ──
    let r_own = tid / 2u32; // 0..128, row within the tile
    let t_sub = tid % 2u32; // token sub-tile
    let row_g = base_row + r_own;
    let row_c = if row_g < m { row_g } else { m - 1u32 };
    let r_in = r_own % 16u32; // row within the 16-row sub-tile
    let t_base = t_sub * 16u32;

    // ── Staging invariants (Bench 715 Arm-1 forms), hoisted OUT of the
    //    g/ks loops. A-REMAP: this thread owns row tid/2's k-half (bits
    //    hb..hb+16) of its OWN sg's tile (tile == tid/32 == sg) — 2 word
    //    loads per k-step instead of 32 warp-broadcast loads; same bytes
    //    land in the same smem positions (bit-identical by construction,
    //    pinned by the Bench-710 FNV gates). B-HOIST: the 4 tokens this
    //    thread touches (e/e2 × sub-tile), clamped and word-based ONCE
    //    (tok*q_words_row + lane/4); the byte shift is invariant. ──
    let lane = tid % 32u32;
    let a2_addr = row_c * words_per_row; // + word_off per k-step
    let a2_hb = t_sub * 16u32;           // this thread's base bit
    let a2_bb = (lane / 2u32) * 32u32 + a2_hb; // byte base within the tile
    let b2_sh = (lane % 4u32) * 8u32;
    let b2_lane4 = lane / 4u32;
    let b2_t0c = base_tok + sg;
    let b2_t1c = base_tok + sg + 8u32;
    let b2_t2c = base_tok + 16u32 + sg;
    let b2_t3c = base_tok + 16u32 + sg + 8u32;
    let b2_t0 = (if b2_t0c < p_tokens { b2_t0c } else { p_tokens - 1u32 }) * q_words_row
        + b2_lane4;
    let b2_t1 = (if b2_t1c < p_tokens { b2_t1c } else { p_tokens - 1u32 }) * q_words_row
        + b2_lane4;
    let b2_t2 = (if b2_t2c < p_tokens { b2_t2c } else { p_tokens - 1u32 }) * q_words_row
        + b2_lane4;
    let b2_t3 = (if b2_t3c < p_tokens { b2_t3c } else { p_tokens - 1u32 }) * q_words_row
        + b2_lane4;

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

        // ── Arm-2 PIPELINED k-loop (Bench 716): double-buffered software
        //    pipelining. The serial structure `stage → barrier → mma →
        //    barrier` idles the tensor cores during staging (Bench 714/715:
        //    staging is 58% and latency/serialization-bound — the phase cost
        //    must be HIDDEN, not shrunk). Here: a prologue stages k-step 0
        //    into SET0; each loop iteration then runs mma(ks) from
        //    SET[ks%2] while staging min(ks+1,3) into the OTHER set — no
        //    barrier between them, ONE barrier per k-step (was 2). The
        //    in-loop staging is guarded ks<3 (4 staging passes per group,
        //    same as serial). The staging loads stay INLINE with the staging
        //    (after the mma issue) — hoisting them above the mma keeps 10
        //    u32 live across the tensor phase and measured 0.93-0.97×
        //    (register pressure); the inline form measured a stable
        //    1.04-1.06× kernel-level over 9 interleaved ladder runs (the
        //    wall-probe M127 vs M63; the ceiling is 1.24× — staging is 3×
        //    the mma phase, so pipelining hides only the smaller phase).
        //    Overlapping two ALU/LSU-bound phases does NOT help (the Arm-3
        //    group-overlap rung measured 0.99× — reduce and staging compete
        //    for the same CUDA cores). Same bytes land in the set mma(ks)
        //    reads after a barrier → bit-identical by construction, pinned
        //    by the Bench-710 FNV gates. ──

        // Prologue: stage k-step 0 into SET0.
        {
            let posw = pos_bits_u32[(a2_addr + g * 4u32) as usize];
            let negw = neg_bits_u32[(a2_addr + g * 4u32) as usize];
            let ph = posw >> a2_hb;
            let nh = negw >> a2_hb;
            let kb4 = g * 32u32;
            let qh0_p = q_hi_w[(b2_t0 + kb4) as usize];
            let qh1_p = q_hi_w[(b2_t1 + kb4) as usize];
            let qh2_p = q_hi_w[(b2_t2 + kb4) as usize];
            let qh3_p = q_hi_w[(b2_t3 + kb4) as usize];
            let ql0_p = q_lo_w[(b2_t0 + kb4) as usize];
            let ql1_p = q_lo_w[(b2_t1 + kb4) as usize];
            let ql2_p = q_lo_w[(b2_t2 + kb4) as usize];
            let ql3_p = q_lo_w[(b2_t3 + kb4) as usize];
            let s0 = (ph & 1u32) as i32 - (nh & 1u32) as i32;
            let s1 = ((ph >> 1u32) & 1u32) as i32 - ((nh >> 1u32) & 1u32) as i32;
            let s2 = ((ph >> 2u32) & 1u32) as i32 - ((nh >> 2u32) & 1u32) as i32;
            let s3 = ((ph >> 3u32) & 1u32) as i32 - ((nh >> 3u32) & 1u32) as i32;
            let s4 = ((ph >> 4u32) & 1u32) as i32 - ((nh >> 4u32) & 1u32) as i32;
            let s5 = ((ph >> 5u32) & 1u32) as i32 - ((nh >> 5u32) & 1u32) as i32;
            let s6 = ((ph >> 6u32) & 1u32) as i32 - ((nh >> 6u32) & 1u32) as i32;
            let s7 = ((ph >> 7u32) & 1u32) as i32 - ((nh >> 7u32) & 1u32) as i32;
            let s8 = ((ph >> 8u32) & 1u32) as i32 - ((nh >> 8u32) & 1u32) as i32;
            let s9 = ((ph >> 9u32) & 1u32) as i32 - ((nh >> 9u32) & 1u32) as i32;
            let s10 = ((ph >> 10u32) & 1u32) as i32 - ((nh >> 10u32) & 1u32) as i32;
            let s11 = ((ph >> 11u32) & 1u32) as i32 - ((nh >> 11u32) & 1u32) as i32;
            let s12 = ((ph >> 12u32) & 1u32) as i32 - ((nh >> 12u32) & 1u32) as i32;
            let s13 = ((ph >> 13u32) & 1u32) as i32 - ((nh >> 13u32) & 1u32) as i32;
            let s14 = ((ph >> 14u32) & 1u32) as i32 - ((nh >> 14u32) & 1u32) as i32;
            let s15 = ((ph >> 15u32) & 1u32) as i32 - ((nh >> 15u32) & 1u32) as i32;
            let b0 = a2_bb as usize;
            if sg == 0u32 {
                a0[b0] = i8::cast_from(s0);
                a0[b0 + 1] = i8::cast_from(s1);
                a0[b0 + 2] = i8::cast_from(s2);
                a0[b0 + 3] = i8::cast_from(s3);
                a0[b0 + 4] = i8::cast_from(s4);
                a0[b0 + 5] = i8::cast_from(s5);
                a0[b0 + 6] = i8::cast_from(s6);
                a0[b0 + 7] = i8::cast_from(s7);
                a0[b0 + 8] = i8::cast_from(s8);
                a0[b0 + 9] = i8::cast_from(s9);
                a0[b0 + 10] = i8::cast_from(s10);
                a0[b0 + 11] = i8::cast_from(s11);
                a0[b0 + 12] = i8::cast_from(s12);
                a0[b0 + 13] = i8::cast_from(s13);
                a0[b0 + 14] = i8::cast_from(s14);
                a0[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 1u32 {
                a1[b0] = i8::cast_from(s0);
                a1[b0 + 1] = i8::cast_from(s1);
                a1[b0 + 2] = i8::cast_from(s2);
                a1[b0 + 3] = i8::cast_from(s3);
                a1[b0 + 4] = i8::cast_from(s4);
                a1[b0 + 5] = i8::cast_from(s5);
                a1[b0 + 6] = i8::cast_from(s6);
                a1[b0 + 7] = i8::cast_from(s7);
                a1[b0 + 8] = i8::cast_from(s8);
                a1[b0 + 9] = i8::cast_from(s9);
                a1[b0 + 10] = i8::cast_from(s10);
                a1[b0 + 11] = i8::cast_from(s11);
                a1[b0 + 12] = i8::cast_from(s12);
                a1[b0 + 13] = i8::cast_from(s13);
                a1[b0 + 14] = i8::cast_from(s14);
                a1[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 2u32 {
                a2[b0] = i8::cast_from(s0);
                a2[b0 + 1] = i8::cast_from(s1);
                a2[b0 + 2] = i8::cast_from(s2);
                a2[b0 + 3] = i8::cast_from(s3);
                a2[b0 + 4] = i8::cast_from(s4);
                a2[b0 + 5] = i8::cast_from(s5);
                a2[b0 + 6] = i8::cast_from(s6);
                a2[b0 + 7] = i8::cast_from(s7);
                a2[b0 + 8] = i8::cast_from(s8);
                a2[b0 + 9] = i8::cast_from(s9);
                a2[b0 + 10] = i8::cast_from(s10);
                a2[b0 + 11] = i8::cast_from(s11);
                a2[b0 + 12] = i8::cast_from(s12);
                a2[b0 + 13] = i8::cast_from(s13);
                a2[b0 + 14] = i8::cast_from(s14);
                a2[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 3u32 {
                a3[b0] = i8::cast_from(s0);
                a3[b0 + 1] = i8::cast_from(s1);
                a3[b0 + 2] = i8::cast_from(s2);
                a3[b0 + 3] = i8::cast_from(s3);
                a3[b0 + 4] = i8::cast_from(s4);
                a3[b0 + 5] = i8::cast_from(s5);
                a3[b0 + 6] = i8::cast_from(s6);
                a3[b0 + 7] = i8::cast_from(s7);
                a3[b0 + 8] = i8::cast_from(s8);
                a3[b0 + 9] = i8::cast_from(s9);
                a3[b0 + 10] = i8::cast_from(s10);
                a3[b0 + 11] = i8::cast_from(s11);
                a3[b0 + 12] = i8::cast_from(s12);
                a3[b0 + 13] = i8::cast_from(s13);
                a3[b0 + 14] = i8::cast_from(s14);
                a3[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 4u32 {
                a4[b0] = i8::cast_from(s0);
                a4[b0 + 1] = i8::cast_from(s1);
                a4[b0 + 2] = i8::cast_from(s2);
                a4[b0 + 3] = i8::cast_from(s3);
                a4[b0 + 4] = i8::cast_from(s4);
                a4[b0 + 5] = i8::cast_from(s5);
                a4[b0 + 6] = i8::cast_from(s6);
                a4[b0 + 7] = i8::cast_from(s7);
                a4[b0 + 8] = i8::cast_from(s8);
                a4[b0 + 9] = i8::cast_from(s9);
                a4[b0 + 10] = i8::cast_from(s10);
                a4[b0 + 11] = i8::cast_from(s11);
                a4[b0 + 12] = i8::cast_from(s12);
                a4[b0 + 13] = i8::cast_from(s13);
                a4[b0 + 14] = i8::cast_from(s14);
                a4[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 5u32 {
                a5[b0] = i8::cast_from(s0);
                a5[b0 + 1] = i8::cast_from(s1);
                a5[b0 + 2] = i8::cast_from(s2);
                a5[b0 + 3] = i8::cast_from(s3);
                a5[b0 + 4] = i8::cast_from(s4);
                a5[b0 + 5] = i8::cast_from(s5);
                a5[b0 + 6] = i8::cast_from(s6);
                a5[b0 + 7] = i8::cast_from(s7);
                a5[b0 + 8] = i8::cast_from(s8);
                a5[b0 + 9] = i8::cast_from(s9);
                a5[b0 + 10] = i8::cast_from(s10);
                a5[b0 + 11] = i8::cast_from(s11);
                a5[b0 + 12] = i8::cast_from(s12);
                a5[b0 + 13] = i8::cast_from(s13);
                a5[b0 + 14] = i8::cast_from(s14);
                a5[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 6u32 {
                a6[b0] = i8::cast_from(s0);
                a6[b0 + 1] = i8::cast_from(s1);
                a6[b0 + 2] = i8::cast_from(s2);
                a6[b0 + 3] = i8::cast_from(s3);
                a6[b0 + 4] = i8::cast_from(s4);
                a6[b0 + 5] = i8::cast_from(s5);
                a6[b0 + 6] = i8::cast_from(s6);
                a6[b0 + 7] = i8::cast_from(s7);
                a6[b0 + 8] = i8::cast_from(s8);
                a6[b0 + 9] = i8::cast_from(s9);
                a6[b0 + 10] = i8::cast_from(s10);
                a6[b0 + 11] = i8::cast_from(s11);
                a6[b0 + 12] = i8::cast_from(s12);
                a6[b0 + 13] = i8::cast_from(s13);
                a6[b0 + 14] = i8::cast_from(s14);
                a6[b0 + 15] = i8::cast_from(s15);
            }
            else if sg == 7u32 {
                a7[b0] = i8::cast_from(s0);
                a7[b0 + 1] = i8::cast_from(s1);
                a7[b0 + 2] = i8::cast_from(s2);
                a7[b0 + 3] = i8::cast_from(s3);
                a7[b0 + 4] = i8::cast_from(s4);
                a7[b0 + 5] = i8::cast_from(s5);
                a7[b0 + 6] = i8::cast_from(s6);
                a7[b0 + 7] = i8::cast_from(s7);
                a7[b0 + 8] = i8::cast_from(s8);
                a7[b0 + 9] = i8::cast_from(s9);
                a7[b0 + 10] = i8::cast_from(s10);
                a7[b0 + 11] = i8::cast_from(s11);
                a7[b0 + 12] = i8::cast_from(s12);
                a7[b0 + 13] = i8::cast_from(s13);
                a7[b0 + 14] = i8::cast_from(s14);
                a7[b0 + 15] = i8::cast_from(s15);
            }
            let be = tid as usize;
            let be2 = (tid + 256u32) as usize;
            {
                let byte = ((qh0_p >> b2_sh) & 0xFFu32) as i32;
                bh0[be] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((qh1_p >> b2_sh) & 0xFFu32) as i32;
                bh0[be2] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((qh2_p >> b2_sh) & 0xFFu32) as i32;
                bh1[be] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((qh3_p >> b2_sh) & 0xFFu32) as i32;
                bh1[be2] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((ql0_p >> b2_sh) & 0xFFu32) as i32;
                bl0[be] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((ql1_p >> b2_sh) & 0xFFu32) as i32;
                bl0[be2] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((ql2_p >> b2_sh) & 0xFFu32) as i32;
                bl1[be] = i8::cast_from((byte << 24) >> 24);
            }
            {
                let byte = ((ql3_p >> b2_sh) & 0xFFu32) as i32;
                bl1[be2] = i8::cast_from((byte << 24) >> 24);
            }
        }
        sync_cube();

        let mut ks = 0u32;
        while ks < 4u32 {
            let ks1 = if ks < 3u32 { ks + 1u32 } else { 3u32 };

            // ── MMA(ks) from SET[ks%2] (same fragment shapes as the serial
            //    loop; only the buffer set differs by parity). ──
            if (ks & 1u32) == 0u32 {
                let mb0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bh0,
                    32,
                );
                let mb1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bl0,
                    32,
                );
                let mb2 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bh1,
                    32,
                );
                let mb3 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bl1,
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
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 1u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a1,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 2u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a2,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 3u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a3,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 4u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a4,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 5u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a5,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 6u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a6,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 7u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a7,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
            }
            else {
                let mb0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bh0b,
                    32,
                );
                let mb1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bl0b,
                    32,
                );
                let mb2 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bh1b,
                    32,
                );
                let mb3 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize, 16usize, 32usize,
                    cmma::MatrixLayout::ColMajor,
                    &bl1b,
                    32,
                );
                if sg == 0u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a0b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 1u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a1b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 2u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a2b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 3u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a3b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 4u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a4b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 5u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a5b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 6u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a6b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
                else if sg == 7u32 {
                    let ma = cmma::Matrix::<i8>::from_slice(
                        cmma::MatrixIdent::A,
                        16usize, 16usize, 32usize,
                        cmma::MatrixLayout::RowMajor,
                        &a7b,
                        32,
                    );
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb0, &ahi0, &ahi0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb1, &alo0, &alo0);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb2, &ahi1, &ahi1);
                    cmma::execute::<i8, i8, i32, i32, cmma::Plane>(&ma, &mb3, &alo1, &alo1);
                }
            }

            // ── Staging(ks1) into the OTHER set — overlaps the mma above (no
            //    barrier between them; different buffers). Guarded ks<3: no
            //    dead re-stage on the last k-step. ──
            if ks < 3u32 {
                if (ks & 1u32) == 0u32 {
                    // → SET1
                    let wo_n = (a2_addr + g * 4u32 + ks1) as usize;
                    let kb4_n = g * 32u32 + ks1 * 8u32;
                    let posw_n = pos_bits_u32[wo_n];
                    let negw_n = neg_bits_u32[wo_n];
                    let qh0_n = q_hi_w[(b2_t0 + kb4_n) as usize];
                    let qh1_n = q_hi_w[(b2_t1 + kb4_n) as usize];
                    let qh2_n = q_hi_w[(b2_t2 + kb4_n) as usize];
                    let qh3_n = q_hi_w[(b2_t3 + kb4_n) as usize];
                    let ql0_n = q_lo_w[(b2_t0 + kb4_n) as usize];
                    let ql1_n = q_lo_w[(b2_t1 + kb4_n) as usize];
                    let ql2_n = q_lo_w[(b2_t2 + kb4_n) as usize];
                    let ql3_n = q_lo_w[(b2_t3 + kb4_n) as usize];
                    let ph = posw_n >> a2_hb;
                    let nh = negw_n >> a2_hb;
                    let s0 = (ph & 1u32) as i32 - (nh & 1u32) as i32;
                    let s1 = ((ph >> 1u32) & 1u32) as i32 - ((nh >> 1u32) & 1u32) as i32;
                    let s2 = ((ph >> 2u32) & 1u32) as i32 - ((nh >> 2u32) & 1u32) as i32;
                    let s3 = ((ph >> 3u32) & 1u32) as i32 - ((nh >> 3u32) & 1u32) as i32;
                    let s4 = ((ph >> 4u32) & 1u32) as i32 - ((nh >> 4u32) & 1u32) as i32;
                    let s5 = ((ph >> 5u32) & 1u32) as i32 - ((nh >> 5u32) & 1u32) as i32;
                    let s6 = ((ph >> 6u32) & 1u32) as i32 - ((nh >> 6u32) & 1u32) as i32;
                    let s7 = ((ph >> 7u32) & 1u32) as i32 - ((nh >> 7u32) & 1u32) as i32;
                    let s8 = ((ph >> 8u32) & 1u32) as i32 - ((nh >> 8u32) & 1u32) as i32;
                    let s9 = ((ph >> 9u32) & 1u32) as i32 - ((nh >> 9u32) & 1u32) as i32;
                    let s10 = ((ph >> 10u32) & 1u32) as i32 - ((nh >> 10u32) & 1u32) as i32;
                    let s11 = ((ph >> 11u32) & 1u32) as i32 - ((nh >> 11u32) & 1u32) as i32;
                    let s12 = ((ph >> 12u32) & 1u32) as i32 - ((nh >> 12u32) & 1u32) as i32;
                    let s13 = ((ph >> 13u32) & 1u32) as i32 - ((nh >> 13u32) & 1u32) as i32;
                    let s14 = ((ph >> 14u32) & 1u32) as i32 - ((nh >> 14u32) & 1u32) as i32;
                    let s15 = ((ph >> 15u32) & 1u32) as i32 - ((nh >> 15u32) & 1u32) as i32;
                    let b0 = a2_bb as usize;
                    if sg == 0u32 {
                        a0b[b0] = i8::cast_from(s0);
                        a0b[b0 + 1] = i8::cast_from(s1);
                        a0b[b0 + 2] = i8::cast_from(s2);
                        a0b[b0 + 3] = i8::cast_from(s3);
                        a0b[b0 + 4] = i8::cast_from(s4);
                        a0b[b0 + 5] = i8::cast_from(s5);
                        a0b[b0 + 6] = i8::cast_from(s6);
                        a0b[b0 + 7] = i8::cast_from(s7);
                        a0b[b0 + 8] = i8::cast_from(s8);
                        a0b[b0 + 9] = i8::cast_from(s9);
                        a0b[b0 + 10] = i8::cast_from(s10);
                        a0b[b0 + 11] = i8::cast_from(s11);
                        a0b[b0 + 12] = i8::cast_from(s12);
                        a0b[b0 + 13] = i8::cast_from(s13);
                        a0b[b0 + 14] = i8::cast_from(s14);
                        a0b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 1u32 {
                        a1b[b0] = i8::cast_from(s0);
                        a1b[b0 + 1] = i8::cast_from(s1);
                        a1b[b0 + 2] = i8::cast_from(s2);
                        a1b[b0 + 3] = i8::cast_from(s3);
                        a1b[b0 + 4] = i8::cast_from(s4);
                        a1b[b0 + 5] = i8::cast_from(s5);
                        a1b[b0 + 6] = i8::cast_from(s6);
                        a1b[b0 + 7] = i8::cast_from(s7);
                        a1b[b0 + 8] = i8::cast_from(s8);
                        a1b[b0 + 9] = i8::cast_from(s9);
                        a1b[b0 + 10] = i8::cast_from(s10);
                        a1b[b0 + 11] = i8::cast_from(s11);
                        a1b[b0 + 12] = i8::cast_from(s12);
                        a1b[b0 + 13] = i8::cast_from(s13);
                        a1b[b0 + 14] = i8::cast_from(s14);
                        a1b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 2u32 {
                        a2b[b0] = i8::cast_from(s0);
                        a2b[b0 + 1] = i8::cast_from(s1);
                        a2b[b0 + 2] = i8::cast_from(s2);
                        a2b[b0 + 3] = i8::cast_from(s3);
                        a2b[b0 + 4] = i8::cast_from(s4);
                        a2b[b0 + 5] = i8::cast_from(s5);
                        a2b[b0 + 6] = i8::cast_from(s6);
                        a2b[b0 + 7] = i8::cast_from(s7);
                        a2b[b0 + 8] = i8::cast_from(s8);
                        a2b[b0 + 9] = i8::cast_from(s9);
                        a2b[b0 + 10] = i8::cast_from(s10);
                        a2b[b0 + 11] = i8::cast_from(s11);
                        a2b[b0 + 12] = i8::cast_from(s12);
                        a2b[b0 + 13] = i8::cast_from(s13);
                        a2b[b0 + 14] = i8::cast_from(s14);
                        a2b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 3u32 {
                        a3b[b0] = i8::cast_from(s0);
                        a3b[b0 + 1] = i8::cast_from(s1);
                        a3b[b0 + 2] = i8::cast_from(s2);
                        a3b[b0 + 3] = i8::cast_from(s3);
                        a3b[b0 + 4] = i8::cast_from(s4);
                        a3b[b0 + 5] = i8::cast_from(s5);
                        a3b[b0 + 6] = i8::cast_from(s6);
                        a3b[b0 + 7] = i8::cast_from(s7);
                        a3b[b0 + 8] = i8::cast_from(s8);
                        a3b[b0 + 9] = i8::cast_from(s9);
                        a3b[b0 + 10] = i8::cast_from(s10);
                        a3b[b0 + 11] = i8::cast_from(s11);
                        a3b[b0 + 12] = i8::cast_from(s12);
                        a3b[b0 + 13] = i8::cast_from(s13);
                        a3b[b0 + 14] = i8::cast_from(s14);
                        a3b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 4u32 {
                        a4b[b0] = i8::cast_from(s0);
                        a4b[b0 + 1] = i8::cast_from(s1);
                        a4b[b0 + 2] = i8::cast_from(s2);
                        a4b[b0 + 3] = i8::cast_from(s3);
                        a4b[b0 + 4] = i8::cast_from(s4);
                        a4b[b0 + 5] = i8::cast_from(s5);
                        a4b[b0 + 6] = i8::cast_from(s6);
                        a4b[b0 + 7] = i8::cast_from(s7);
                        a4b[b0 + 8] = i8::cast_from(s8);
                        a4b[b0 + 9] = i8::cast_from(s9);
                        a4b[b0 + 10] = i8::cast_from(s10);
                        a4b[b0 + 11] = i8::cast_from(s11);
                        a4b[b0 + 12] = i8::cast_from(s12);
                        a4b[b0 + 13] = i8::cast_from(s13);
                        a4b[b0 + 14] = i8::cast_from(s14);
                        a4b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 5u32 {
                        a5b[b0] = i8::cast_from(s0);
                        a5b[b0 + 1] = i8::cast_from(s1);
                        a5b[b0 + 2] = i8::cast_from(s2);
                        a5b[b0 + 3] = i8::cast_from(s3);
                        a5b[b0 + 4] = i8::cast_from(s4);
                        a5b[b0 + 5] = i8::cast_from(s5);
                        a5b[b0 + 6] = i8::cast_from(s6);
                        a5b[b0 + 7] = i8::cast_from(s7);
                        a5b[b0 + 8] = i8::cast_from(s8);
                        a5b[b0 + 9] = i8::cast_from(s9);
                        a5b[b0 + 10] = i8::cast_from(s10);
                        a5b[b0 + 11] = i8::cast_from(s11);
                        a5b[b0 + 12] = i8::cast_from(s12);
                        a5b[b0 + 13] = i8::cast_from(s13);
                        a5b[b0 + 14] = i8::cast_from(s14);
                        a5b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 6u32 {
                        a6b[b0] = i8::cast_from(s0);
                        a6b[b0 + 1] = i8::cast_from(s1);
                        a6b[b0 + 2] = i8::cast_from(s2);
                        a6b[b0 + 3] = i8::cast_from(s3);
                        a6b[b0 + 4] = i8::cast_from(s4);
                        a6b[b0 + 5] = i8::cast_from(s5);
                        a6b[b0 + 6] = i8::cast_from(s6);
                        a6b[b0 + 7] = i8::cast_from(s7);
                        a6b[b0 + 8] = i8::cast_from(s8);
                        a6b[b0 + 9] = i8::cast_from(s9);
                        a6b[b0 + 10] = i8::cast_from(s10);
                        a6b[b0 + 11] = i8::cast_from(s11);
                        a6b[b0 + 12] = i8::cast_from(s12);
                        a6b[b0 + 13] = i8::cast_from(s13);
                        a6b[b0 + 14] = i8::cast_from(s14);
                        a6b[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 7u32 {
                        a7b[b0] = i8::cast_from(s0);
                        a7b[b0 + 1] = i8::cast_from(s1);
                        a7b[b0 + 2] = i8::cast_from(s2);
                        a7b[b0 + 3] = i8::cast_from(s3);
                        a7b[b0 + 4] = i8::cast_from(s4);
                        a7b[b0 + 5] = i8::cast_from(s5);
                        a7b[b0 + 6] = i8::cast_from(s6);
                        a7b[b0 + 7] = i8::cast_from(s7);
                        a7b[b0 + 8] = i8::cast_from(s8);
                        a7b[b0 + 9] = i8::cast_from(s9);
                        a7b[b0 + 10] = i8::cast_from(s10);
                        a7b[b0 + 11] = i8::cast_from(s11);
                        a7b[b0 + 12] = i8::cast_from(s12);
                        a7b[b0 + 13] = i8::cast_from(s13);
                        a7b[b0 + 14] = i8::cast_from(s14);
                        a7b[b0 + 15] = i8::cast_from(s15);
                    }
                    let be = tid as usize;
                    let be2 = (tid + 256u32) as usize;
                    {
                        let byte = ((qh0_n >> b2_sh) & 0xFFu32) as i32;
                        bh0b[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((qh1_n >> b2_sh) & 0xFFu32) as i32;
                        bh0b[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((qh2_n >> b2_sh) & 0xFFu32) as i32;
                        bh1b[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((qh3_n >> b2_sh) & 0xFFu32) as i32;
                        bh1b[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql0_n >> b2_sh) & 0xFFu32) as i32;
                        bl0b[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql1_n >> b2_sh) & 0xFFu32) as i32;
                        bl0b[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql2_n >> b2_sh) & 0xFFu32) as i32;
                        bl1b[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql3_n >> b2_sh) & 0xFFu32) as i32;
                        bl1b[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                }
                else {
                    // → SET0
                    let wo_n = (a2_addr + g * 4u32 + ks1) as usize;
                    let kb4_n = g * 32u32 + ks1 * 8u32;
                    let posw_n = pos_bits_u32[wo_n];
                    let negw_n = neg_bits_u32[wo_n];
                    let qh0_n = q_hi_w[(b2_t0 + kb4_n) as usize];
                    let qh1_n = q_hi_w[(b2_t1 + kb4_n) as usize];
                    let qh2_n = q_hi_w[(b2_t2 + kb4_n) as usize];
                    let qh3_n = q_hi_w[(b2_t3 + kb4_n) as usize];
                    let ql0_n = q_lo_w[(b2_t0 + kb4_n) as usize];
                    let ql1_n = q_lo_w[(b2_t1 + kb4_n) as usize];
                    let ql2_n = q_lo_w[(b2_t2 + kb4_n) as usize];
                    let ql3_n = q_lo_w[(b2_t3 + kb4_n) as usize];
                    let ph = posw_n >> a2_hb;
                    let nh = negw_n >> a2_hb;
                    let s0 = (ph & 1u32) as i32 - (nh & 1u32) as i32;
                    let s1 = ((ph >> 1u32) & 1u32) as i32 - ((nh >> 1u32) & 1u32) as i32;
                    let s2 = ((ph >> 2u32) & 1u32) as i32 - ((nh >> 2u32) & 1u32) as i32;
                    let s3 = ((ph >> 3u32) & 1u32) as i32 - ((nh >> 3u32) & 1u32) as i32;
                    let s4 = ((ph >> 4u32) & 1u32) as i32 - ((nh >> 4u32) & 1u32) as i32;
                    let s5 = ((ph >> 5u32) & 1u32) as i32 - ((nh >> 5u32) & 1u32) as i32;
                    let s6 = ((ph >> 6u32) & 1u32) as i32 - ((nh >> 6u32) & 1u32) as i32;
                    let s7 = ((ph >> 7u32) & 1u32) as i32 - ((nh >> 7u32) & 1u32) as i32;
                    let s8 = ((ph >> 8u32) & 1u32) as i32 - ((nh >> 8u32) & 1u32) as i32;
                    let s9 = ((ph >> 9u32) & 1u32) as i32 - ((nh >> 9u32) & 1u32) as i32;
                    let s10 = ((ph >> 10u32) & 1u32) as i32 - ((nh >> 10u32) & 1u32) as i32;
                    let s11 = ((ph >> 11u32) & 1u32) as i32 - ((nh >> 11u32) & 1u32) as i32;
                    let s12 = ((ph >> 12u32) & 1u32) as i32 - ((nh >> 12u32) & 1u32) as i32;
                    let s13 = ((ph >> 13u32) & 1u32) as i32 - ((nh >> 13u32) & 1u32) as i32;
                    let s14 = ((ph >> 14u32) & 1u32) as i32 - ((nh >> 14u32) & 1u32) as i32;
                    let s15 = ((ph >> 15u32) & 1u32) as i32 - ((nh >> 15u32) & 1u32) as i32;
                    let b0 = a2_bb as usize;
                    if sg == 0u32 {
                        a0[b0] = i8::cast_from(s0);
                        a0[b0 + 1] = i8::cast_from(s1);
                        a0[b0 + 2] = i8::cast_from(s2);
                        a0[b0 + 3] = i8::cast_from(s3);
                        a0[b0 + 4] = i8::cast_from(s4);
                        a0[b0 + 5] = i8::cast_from(s5);
                        a0[b0 + 6] = i8::cast_from(s6);
                        a0[b0 + 7] = i8::cast_from(s7);
                        a0[b0 + 8] = i8::cast_from(s8);
                        a0[b0 + 9] = i8::cast_from(s9);
                        a0[b0 + 10] = i8::cast_from(s10);
                        a0[b0 + 11] = i8::cast_from(s11);
                        a0[b0 + 12] = i8::cast_from(s12);
                        a0[b0 + 13] = i8::cast_from(s13);
                        a0[b0 + 14] = i8::cast_from(s14);
                        a0[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 1u32 {
                        a1[b0] = i8::cast_from(s0);
                        a1[b0 + 1] = i8::cast_from(s1);
                        a1[b0 + 2] = i8::cast_from(s2);
                        a1[b0 + 3] = i8::cast_from(s3);
                        a1[b0 + 4] = i8::cast_from(s4);
                        a1[b0 + 5] = i8::cast_from(s5);
                        a1[b0 + 6] = i8::cast_from(s6);
                        a1[b0 + 7] = i8::cast_from(s7);
                        a1[b0 + 8] = i8::cast_from(s8);
                        a1[b0 + 9] = i8::cast_from(s9);
                        a1[b0 + 10] = i8::cast_from(s10);
                        a1[b0 + 11] = i8::cast_from(s11);
                        a1[b0 + 12] = i8::cast_from(s12);
                        a1[b0 + 13] = i8::cast_from(s13);
                        a1[b0 + 14] = i8::cast_from(s14);
                        a1[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 2u32 {
                        a2[b0] = i8::cast_from(s0);
                        a2[b0 + 1] = i8::cast_from(s1);
                        a2[b0 + 2] = i8::cast_from(s2);
                        a2[b0 + 3] = i8::cast_from(s3);
                        a2[b0 + 4] = i8::cast_from(s4);
                        a2[b0 + 5] = i8::cast_from(s5);
                        a2[b0 + 6] = i8::cast_from(s6);
                        a2[b0 + 7] = i8::cast_from(s7);
                        a2[b0 + 8] = i8::cast_from(s8);
                        a2[b0 + 9] = i8::cast_from(s9);
                        a2[b0 + 10] = i8::cast_from(s10);
                        a2[b0 + 11] = i8::cast_from(s11);
                        a2[b0 + 12] = i8::cast_from(s12);
                        a2[b0 + 13] = i8::cast_from(s13);
                        a2[b0 + 14] = i8::cast_from(s14);
                        a2[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 3u32 {
                        a3[b0] = i8::cast_from(s0);
                        a3[b0 + 1] = i8::cast_from(s1);
                        a3[b0 + 2] = i8::cast_from(s2);
                        a3[b0 + 3] = i8::cast_from(s3);
                        a3[b0 + 4] = i8::cast_from(s4);
                        a3[b0 + 5] = i8::cast_from(s5);
                        a3[b0 + 6] = i8::cast_from(s6);
                        a3[b0 + 7] = i8::cast_from(s7);
                        a3[b0 + 8] = i8::cast_from(s8);
                        a3[b0 + 9] = i8::cast_from(s9);
                        a3[b0 + 10] = i8::cast_from(s10);
                        a3[b0 + 11] = i8::cast_from(s11);
                        a3[b0 + 12] = i8::cast_from(s12);
                        a3[b0 + 13] = i8::cast_from(s13);
                        a3[b0 + 14] = i8::cast_from(s14);
                        a3[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 4u32 {
                        a4[b0] = i8::cast_from(s0);
                        a4[b0 + 1] = i8::cast_from(s1);
                        a4[b0 + 2] = i8::cast_from(s2);
                        a4[b0 + 3] = i8::cast_from(s3);
                        a4[b0 + 4] = i8::cast_from(s4);
                        a4[b0 + 5] = i8::cast_from(s5);
                        a4[b0 + 6] = i8::cast_from(s6);
                        a4[b0 + 7] = i8::cast_from(s7);
                        a4[b0 + 8] = i8::cast_from(s8);
                        a4[b0 + 9] = i8::cast_from(s9);
                        a4[b0 + 10] = i8::cast_from(s10);
                        a4[b0 + 11] = i8::cast_from(s11);
                        a4[b0 + 12] = i8::cast_from(s12);
                        a4[b0 + 13] = i8::cast_from(s13);
                        a4[b0 + 14] = i8::cast_from(s14);
                        a4[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 5u32 {
                        a5[b0] = i8::cast_from(s0);
                        a5[b0 + 1] = i8::cast_from(s1);
                        a5[b0 + 2] = i8::cast_from(s2);
                        a5[b0 + 3] = i8::cast_from(s3);
                        a5[b0 + 4] = i8::cast_from(s4);
                        a5[b0 + 5] = i8::cast_from(s5);
                        a5[b0 + 6] = i8::cast_from(s6);
                        a5[b0 + 7] = i8::cast_from(s7);
                        a5[b0 + 8] = i8::cast_from(s8);
                        a5[b0 + 9] = i8::cast_from(s9);
                        a5[b0 + 10] = i8::cast_from(s10);
                        a5[b0 + 11] = i8::cast_from(s11);
                        a5[b0 + 12] = i8::cast_from(s12);
                        a5[b0 + 13] = i8::cast_from(s13);
                        a5[b0 + 14] = i8::cast_from(s14);
                        a5[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 6u32 {
                        a6[b0] = i8::cast_from(s0);
                        a6[b0 + 1] = i8::cast_from(s1);
                        a6[b0 + 2] = i8::cast_from(s2);
                        a6[b0 + 3] = i8::cast_from(s3);
                        a6[b0 + 4] = i8::cast_from(s4);
                        a6[b0 + 5] = i8::cast_from(s5);
                        a6[b0 + 6] = i8::cast_from(s6);
                        a6[b0 + 7] = i8::cast_from(s7);
                        a6[b0 + 8] = i8::cast_from(s8);
                        a6[b0 + 9] = i8::cast_from(s9);
                        a6[b0 + 10] = i8::cast_from(s10);
                        a6[b0 + 11] = i8::cast_from(s11);
                        a6[b0 + 12] = i8::cast_from(s12);
                        a6[b0 + 13] = i8::cast_from(s13);
                        a6[b0 + 14] = i8::cast_from(s14);
                        a6[b0 + 15] = i8::cast_from(s15);
                    }
                    else if sg == 7u32 {
                        a7[b0] = i8::cast_from(s0);
                        a7[b0 + 1] = i8::cast_from(s1);
                        a7[b0 + 2] = i8::cast_from(s2);
                        a7[b0 + 3] = i8::cast_from(s3);
                        a7[b0 + 4] = i8::cast_from(s4);
                        a7[b0 + 5] = i8::cast_from(s5);
                        a7[b0 + 6] = i8::cast_from(s6);
                        a7[b0 + 7] = i8::cast_from(s7);
                        a7[b0 + 8] = i8::cast_from(s8);
                        a7[b0 + 9] = i8::cast_from(s9);
                        a7[b0 + 10] = i8::cast_from(s10);
                        a7[b0 + 11] = i8::cast_from(s11);
                        a7[b0 + 12] = i8::cast_from(s12);
                        a7[b0 + 13] = i8::cast_from(s13);
                        a7[b0 + 14] = i8::cast_from(s14);
                        a7[b0 + 15] = i8::cast_from(s15);
                    }
                    let be = tid as usize;
                    let be2 = (tid + 256u32) as usize;
                    {
                        let byte = ((qh0_n >> b2_sh) & 0xFFu32) as i32;
                        bh0[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((qh1_n >> b2_sh) & 0xFFu32) as i32;
                        bh0[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((qh2_n >> b2_sh) & 0xFFu32) as i32;
                        bh1[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((qh3_n >> b2_sh) & 0xFFu32) as i32;
                        bh1[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql0_n >> b2_sh) & 0xFFu32) as i32;
                        bl0[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql1_n >> b2_sh) & 0xFFu32) as i32;
                        bl0[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql2_n >> b2_sh) & 0xFFu32) as i32;
                        bl1[be] = i8::cast_from((byte << 24) >> 24);
                    }
                    {
                        let byte = ((ql3_n >> b2_sh) & 0xFFu32) as i32;
                        bl1[be2] = i8::cast_from((byte << 24) >> 24);
                    }
                }
            }

            sync_cube();
            ks += 1u32;
        }

        // ── Group boundary, TWO-ROUND partials (Arm 5, Bench 718): the
        //    32-buffer single-round store/reduce is split into two
        //    store→reduce rounds over 16 buffers — partials 32 KB →
        //    16 KB, workgroup smem 44 → 28 KB → 3 wgs/SM on the 4090
        //    (was 2 — the occupancy attack on the latency-bound staging
        //    phase). ROUND 0 stores {ahi0, alo0} into {g*h0, g*l0} and
        //    reduces the t_sub==0 threads; ROUND 1 stores {ahi1, alo1}
        //    into the SAME buffers and reduces t_sub==1. Barriers stay
        //    OUTSIDE the divergent branches; the round-1 reads are
        //    separated from the next group's round-0 stores by the
        //    trailing sync below (+2 barriers/group vs the single-round
        //    form — the post-store + trailing syncs are reused). Same
        //    per-thread reduce expression + group order → bit-identical
        //    outputs (t64's round-reuse precedent; the Bench-710 FNV
        //    gates pin it). ──
        if sg == 0u32 {
            cmma::store(&mut g0h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 1u32 {
            cmma::store(&mut g1h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 2u32 {
            cmma::store(&mut g2h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 3u32 {
            cmma::store(&mut g3h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 4u32 {
            cmma::store(&mut g4h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 5u32 {
            cmma::store(&mut g5h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 6u32 {
            cmma::store(&mut g6h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 7u32 {
            cmma::store(&mut g7h0, &ahi0, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7l0, &alo0, 16, cmma::MatrixLayout::RowMajor);
        }
        sync_cube();

        // Reduce ROUND 0: t_sub==0 threads update their 16 outputs.
        {
            let sw = group_scale_f32[(row_c * groups_per_row + g) as usize];
            let bh = (r_in * 16u32) as usize; // base index into a 16×16 tile
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
            }
            else if (sg == 7u32) && (t_sub == 0u32) {
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
        }

        sync_cube();

        // ROUND 1: store {ahi1, alo1} into the SAME per-sg buffers.
        if sg == 0u32 {
            cmma::store(&mut g0h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g0l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 1u32 {
            cmma::store(&mut g1h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g1l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 2u32 {
            cmma::store(&mut g2h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g2l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 3u32 {
            cmma::store(&mut g3h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g3l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 4u32 {
            cmma::store(&mut g4h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g4l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 5u32 {
            cmma::store(&mut g5h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g5l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 6u32 {
            cmma::store(&mut g6h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g6l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }
        else if sg == 7u32 {
            cmma::store(&mut g7h0, &ahi1, 16, cmma::MatrixLayout::RowMajor);
            cmma::store(&mut g7l0, &alo1, 16, cmma::MatrixLayout::RowMajor);
        }

        sync_cube();

        // Reduce ROUND 1: t_sub==1 threads — the round-1 stores
        // overwrote the same buffers (values are ahi1/alo1).
        {
            let sw = group_scale_f32[(row_c * groups_per_row + g) as usize];
            let bh = (r_in * 16u32) as usize; // base index into a 16×16 tile
            if sg == 0u32 {
                if t_sub == 1u32 {
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
            }
            else if sg == 1u32 {
                if t_sub == 1u32 {
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
            }
            else if sg == 2u32 {
                if t_sub == 1u32 {
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
            }
            else if sg == 3u32 {
                if t_sub == 1u32 {
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
            }
            else if sg == 4u32 {
                if t_sub == 1u32 {
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
            }
            else if sg == 5u32 {
                if t_sub == 1u32 {
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
            }
            else if sg == 6u32 {
                if t_sub == 1u32 {
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
            }
            else if (sg == 7u32) && (t_sub == 1u32) {
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
        }

        sync_cube();

        g += 1u32;
    }

    // ── Epilogue: apply the per-token scale + guarded write. Each thread
    //    writes its 16 outputs (row base_row + r_own, tokens base_tok +
    //    t_base + i). ──
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
// Public API (method on the existing GemmTernaryCmmaI8CubeCL — declared in
// gemm_ternary_cmma_i8_cubecl.rs)
// ---------------------------------------------------------------------------

/// Launch the two-round-partials variant (Issue 734 Arm 5): packed-u32
/// quantize pre-pass (the SAME pre-pass as the staging kernel — identical q
/// values are part of the bit-identity contract) + the 28 KB psplit GEMM.
///
/// # Safety
///
/// Same contract as `GemmTernaryCmmaI8CubeCL::launch_sg8`.
#[cfg(feature = "cubecl_runtime")]
pub(crate) unsafe fn launch_psplit_gemm<R: Runtime>(
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
        gemm_ternary_cmma_i8_sg8_psplit::launch_unchecked::<R>(
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
