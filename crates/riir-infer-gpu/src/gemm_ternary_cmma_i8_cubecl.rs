//! NVIDIA int8 cooperative-matrix ternary GEMM for PREFILL — Issue 734 T7
//! (the post-Bench-708 frontier: close the 5.59× prefill gap vs same-box
//! llama.cpp, whose batched GEMM runs ~161 TFLOPS-effective vs our shipped
//! f16 cmma16 kernel's 33 TFLOPS).
//!
//! ## Why int8
//!
//! The runtime advertises `i8×i8→i32 @ 16×16×32` alongside the f16 shape the
//! T6 kernel uses (cmma_probe_734). The 4090's int8 tensor rate is 4× the
//! f16-with-f32-acc rate, and the i8 mma does 2× the K per instruction — the
//! T7 probe (`bench_734_t7_cmma_i8_probe`) measured **764.8 TOPS** for
//! load+mma (vs 327 TFLOPS-class f16) and **exact integer correctness**
//! (256/256 — integer products, exact i32 accumulation).
//!
//! ## Numerics — exact weights + hi/lo activations
//!
//! - **A (weights)**: ternary signs as int8 `{-1,0,+1}` — EXACT, no
//!   quantization error, and independent of the per-row group-scale
//!   distribution (a per-row int8 hi/lo split was analyzed and rejected:
//!   its ~2^-8-of-row-max absolute error gives ~3-6e-3 GEMM rel error when
//!   group scales vary within a row — over the 2e-3 gate).
//! - **B (activations)**: per-token scale + hi/lo int8 split.
//!   `x ≈ s_t·(q_hi + q_lo/128)`, error ≤ s_t/256 = max_row/32512 ≈ 3.1e-5
//!   of the row max — **better than the f16 path's measured 2.2e-4** (f16
//!   rounds each element at 2^-11 relative; the hi/lo split's uniform
//!   absolute error is 16× tighter at the row max and ~4× tighter at the
//!   rms, for post-rmsnorm-typical distributions).
//! - **Weight group scales** (`s_w[row][group]`, 128-col groups): the one
//!   factor that cannot ride the mma (it varies along BOTH the output-row
//!   and k axes). Handled by a **per-group reduction**: the i32 accumulators
//!   live for one 128-col group (4 k-steps of 32), are stored to shared at
//!   the group boundary, and every output accumulates
//!   `s_w[r][g]·(acc_hi + acc_lo/128)` into per-thread f32 registers.
//!   `s_t[tok]` is applied once at the epilogue.
//!
//! ## Structure (256 threads = 8 subgroups, 128 rows × 32 tokens tile)
//!
//! - Subgroup `sg = tid/32` owns row sub-tile `sg` (16 rows); the workgroup
//!   covers 2 token sub-tiles of 16.
//! - **Pipelined k-loop (Bench 716, Arm 2)**: a prologue stages k-step 0
//!   into SET0; each of the 4 loop iterations then runs the mma phase
//!   (4 uniform B `from_slice` + divergent-per-sg A `from_slice` + 4
//!   executes per sg: B_hi→acc_hi, B_lo→acc_lo per token sub-tile) from
//!   SET[ks%2] while staging k-step ks+1 into the OTHER set (double
//!   staging buffers, +6 KB smem = 44 KB total, still 2 wgs/SM) — ONE
//!   barrier per k-step (was 2: staging and mma no longer serialize;
//!   same bytes land in the set the mma reads after a barrier, so the
//!   output is bit-identical by construction — pinned by the Bench-710
//!   FNV gates). The in-loop staging is guarded `ks < 3` (4 staging
//!   passes per group, same as the serial form); the staging loads stay
//!   INLINE with the staging (hoisting them above the mma keeps 10 u32
//!   live across the tensor phase and LOSES 0.93-0.97× — register
//!   pressure; the inline form measured a stable +5% kernel-level).
//! - Per group boundary: divergent per-sg `cmma::store` of the 4 accs into
//!   that sg's named buffers (barriers stay OUTSIDE the divergent branches —
//!   the uniform-control-flow rule), one uniform barrier, then each thread
//!   reduces its OWN 16 outputs. Ownership by construction: thread `tid`
//!   owns row `tid/2` × token sub-tile `tid%2` — the threads owning
//!   sub-tile `s`'s rows are exactly subgroup `s`, so the reduction reads
//!   only its own sg's buffers (nested sg × t_sub branches, direct indexed
//!   loads, no value selects).
//!
//! The per-token activation quantization runs as a separate pre-pass kernel
//! (`quantize_rows_i8_hilo`, one workgroup per token row, packed 4 i8 per
//! u32 word — the q8kv layout; also sidesteps 8-bit-storage questions).

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
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
// Activation quantization pre-pass
// ---------------------------------------------------------------------------

/// Per-token hi/lo int8 quantization: `q_hi/q_lo` packed 4-per-u32, plus the
/// per-token scale `s_out[row]`.
///
/// One workgroup per token row (128 threads): pass 1 computes the row max
/// (shared-memory reduction), pass 2 emits the packed words.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn quantize_rows_i8_hilo(
    input: &[f32],
    q_hi_w: &mut [u32],
    q_lo_w: &mut [u32],
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

    // Pass 2: emit packed words (4 consecutive cols per word).
    let words = n / 4u32;
    let hi127 = f32::new(127.0f32);
    let lo127 = f32::new(-127.0f32);
    let hi64 = f32::new(64.0f32);
    let lo64 = f32::new(-64.0f32);
    let mut w = tid;
    while w < words {
        let mut wh = 0u32;
        let mut wl = 0u32;
        let mut j = 0u32;
        while j < 4u32 {
            let x = input[(base + w * 4u32 + j) as usize];
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
            // Same-width i32→u32 casts are bit-preserving; & 0xFF extracts
            // the two's-complement byte.
            let qh_i = i32::cast_from(qh);
            let ql_i = i32::cast_from(ql);
            wh |= (u32::cast_from(qh_i) & 0xFFu32) << (j * 8u32);
            wl |= (u32::cast_from(ql_i) & 0xFFu32) << (j * 8u32);
            j += 1u32;
        }
        q_hi_w[(row * words + w) as usize] = wh;
        q_lo_w[(row * words + w) as usize] = wl;
        w += nt;
    }
}

// ---------------------------------------------------------------------------
// GEMM kernel
// ---------------------------------------------------------------------------

/// Int8 cooperative-matrix ternary GEMM: `output[p × m] = dequant(W) @ input^T`
/// via i8×i8→i32 @ 16×16×32 tensor-core mma (exact weight signs + hi/lo
/// quantized activations; see the module doc for the numerics contract).
///
/// Workgroup = 256 threads (8 subgroups); tile = 128 rows × 32 tokens.
/// Out-of-range rows/tokens stage CLAMPED data; their outputs are never
/// written back.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_cmma_i8_sg8(
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

    // ── Per-group partial stores: 4 named buffers per sg (32 total; the
    //    reduction reads only its own sg's set — see ownership below). ──
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

        // ── Reduction: each thread updates its 16 outputs (row r_own, token
        //    sub-tile t_sub — reads only its OWN sg's buffer set, selected
        //    by the nested sg × t_sub branches; direct indexed loads). ──
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
// Shared pre-pass launcher (used by launch_sg8, the t64 tile variant, and
// available to sibling GEMM modules — identical q values are part of the
// cross-variant bit-identity contract)
// ---------------------------------------------------------------------------

/// Launch ONLY the packed-u32 hi/lo quantize pre-pass (`quantize_rows_i8_hilo`)
/// into caller-allocated scratch. Byte-identical to the pre-pass inside
/// [`GemmTernaryCmmaI8CubeCL::launch_sg8`] — every consumer of the i8 GEMM
/// family shares it so quantized values match across variants.
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn launch_quantize_hilo_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input_handle: &cubecl::server::Handle,
    q_hi: &cubecl::server::Handle,
    q_lo: &cubecl::server::Handle,
    s_t: &cubecl::server::Handle,
    p_tokens: usize,
    n: u32,
) {
    let q_words = p_tokens * (n as usize / 4);
    unsafe {
        quantize_rows_i8_hilo::launch_unchecked::<R>(
            client,
            CubeCount::Static(p_tokens as u32, 1, 1),
            CubeDim::new_1d(128),
            BufferArg::from_raw_parts(input_handle.clone(), p_tokens * n as usize),
            BufferArg::from_raw_parts(q_hi.clone(), q_words),
            BufferArg::from_raw_parts(q_lo.clone(), q_words),
            BufferArg::from_raw_parts(s_t.clone(), p_tokens),
            n,
        );
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Int8 cooperative-matrix ternary GEMM launcher (Issue 734 T7).
///
/// Computes `output_batch[p_tokens × m] = dequant_ternary(weight) @ input^T`
/// via the NVIDIA i8×i8→i32 @ 16×16×32 tensor-core path: exact weight signs
/// plus hi/lo per-token int8 quantized activations (quantized on-GPU by a
/// pre-pass kernel launched here).
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryCmmaI8CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmTernaryCmmaI8CubeCL {
    /// Debug/test helper: run ONLY the quantize pre-pass and read the packed
    /// results back. Verifies the hi/lo decomposition round-trips:
    /// `x ≈ s·(q_hi + q_lo/128)`.
    #[doc(hidden)]
    pub fn debug_quantize<R: Runtime>(
        client: &ComputeClient<R>,
        input: Vec<f32>,
        n: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let p = input.len() / n;
        let input_h = client.create_from_slice(f32::as_bytes(&input));
        let q_words = p * (n / 4);
        let q_hi = client.empty(q_words * core::mem::size_of::<u32>());
        let q_lo = client.empty(q_words * core::mem::size_of::<u32>());
        let s_t = client.empty(p * core::mem::size_of::<f32>());
        unsafe {
            quantize_rows_i8_hilo::launch_unchecked::<R>(
                client,
                CubeCount::Static(p as u32, 1, 1),
                CubeDim::new_1d(128),
                BufferArg::from_raw_parts(input_h, p * n),
                BufferArg::from_raw_parts(q_hi.clone(), q_words),
                BufferArg::from_raw_parts(q_lo.clone(), q_words),
                BufferArg::from_raw_parts(s_t.clone(), p),
                n as u32,
            );
        }
        let hi = u32::from_bytes(&client.read_one(q_hi).unwrap()).to_vec();
        let lo = u32::from_bytes(&client.read_one(q_lo).unwrap()).to_vec();
        let s = f32::from_bytes(&client.read_one(s_t).unwrap()).to_vec();
        (hi, lo, s)
    }
    /// Check whether the device supports cmma `(i8, i8, i32)` at 16×16×32
    /// (the NVIDIA int8 cooperative-matrix signature).
    pub fn i8_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
        use cubecl::ir::features::MmaConfig;
        use cubecl::ir::{ElemType, IntKind};

        client.features().matmul.cmma.contains(&MmaConfig {
            a_type: ElemType::Int(IntKind::I8).into(),
            b_type: ElemType::Int(IntKind::I8).into(),
            cd_type: ElemType::Int(IntKind::I32).into(),
            m: 16,
            n: 16,
            k: 32,
        })
    }

    /// Whether the device reports `cmma_tensor_addressing` (the
    /// NV_cooperative_matrix2 direct global→matrix load capability the
    /// B-direct variant requires). Issue 734 Lever 2.
    pub fn tensor_addressing_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
        crate::gemm_ternary_cmma_i8_direct_cubecl::tensor_addressing_available(client)
    }

    /// Launch the B-DIRECT variant (Issue 734 Lever 2): byte-quantize
    /// pre-pass + `from_tensor` direct global→matrix B loads per k-step
    /// (no smem staging round-trip). Same numerics contract as
    /// [`Self::launch_sg8`] — outputs expected bit-identical.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_sg8`], PLUS the device must report
    /// `cmma_tensor_addressing` — call [`Self::tensor_addressing_available`].
    pub unsafe fn launch_direct<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        unsafe {
            crate::gemm_ternary_cmma_i8_direct_cubecl::launch_direct_gemm(
                client,
                handle,
                input_handle,
                output_handle,
                p_tokens,
            );
        }
    }

    /// Launch the 128×64-tile variant (Issue 734 Lever 2 tile arm): same
    /// packed-u32 quantize pre-pass + a GEMM with `TOK_TILE` 64 (4 token
    /// sub-tiles), halving A-side re-reads/staging per output at the same
    /// ~40 KB smem (two-round group-partial buffer reuse).
    ///
    /// Outputs are expected BIT-IDENTICAL to [`Self::launch_sg8`] (same q
    /// values, same per-sub-tile i32 mma accumulation, same reduction
    /// arithmetic and group order) — the G1 gate asserts it.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_sg8`].
    pub unsafe fn launch_t64<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        unsafe {
            crate::gemm_ternary_cmma_i8_t64_cubecl::launch_t64_gemm(
                client,
                handle,
                input_handle,
                output_handle,
                p_tokens,
            );
        }
    }

    /// Launch the two-round-partials variant (Issue 734 Arm 5): same
    /// packed-u32 quantize pre-pass + the same staging/pipelined k-loop, but
    /// the per-group partials round-trip through two store→reduce rounds
    /// over 16 buffers (44→28 KB workgroup smem → 3 wgs/SM on the 4090).
    ///
    /// Outputs are expected BIT-IDENTICAL to [`Self::launch_sg8`] (same q
    /// values, same fragment values, same per-thread reduce expression and
    /// group order) — the G1 gate asserts it.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_sg8`].
    pub unsafe fn launch_psplit<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        unsafe {
            crate::gemm_ternary_cmma_i8_psplit_cubecl::launch_psplit_gemm(
                client,
                handle,
                input_handle,
                output_handle,
                p_tokens,
            );
        }
    }

    /// Launch the int8 sg8 kernel: quantize pre-pass + GEMM.
    ///
    /// Allocates the transient quantization scratch (`p×n` bytes ×2 + `p×4`)
    /// per call — the wgpu pool makes this a pool hit after warmup.
    ///
    /// # Safety
    ///
    /// - `input_handle` must hold `p_tokens × handle.n` f32 elements
    /// - `output_handle` must hold `p_tokens × handle.m` f32 elements
    /// - `p_tokens > 0`, `handle.n` a multiple of 128 (group size)
    /// - the device must support cmma `(i8, i8, i32)` at 16×16×32 — call
    ///   [`Self::i8_available`]
    pub unsafe fn launch_sg8<R: Runtime>(
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
        // per-token scales.
        let q_words = p_tokens * (handle.n / 4);
        let q_hi = client.empty(q_words * core::mem::size_of::<u32>());
        let q_lo = client.empty(q_words * core::mem::size_of::<u32>());
        let s_t = client.empty(p_tokens * core::mem::size_of::<f32>());

        unsafe {
            quantize_rows_i8_hilo::launch_unchecked::<R>(
                client,
                CubeCount::Static(p, 1, 1),
                CubeDim::new_1d(128),
                BufferArg::from_raw_parts(input_handle.clone(), p_tokens * handle.n),
                BufferArg::from_raw_parts(q_hi.clone(), q_words),
                BufferArg::from_raw_parts(q_lo.clone(), q_words),
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
            gemm_ternary_cmma_i8_sg8::launch_unchecked::<R>(
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
}
