//! Workgroup-tiled ternary bit-plane dequant+GEMM for PREFILL (Issue 734 T3).
//!
//! The 4090 prefill runs `GemmTernaryBatchedCubeCL` — the plane-cooperative
//! 4-row × 2-token kernel (Issue 637) — at ~2.9 TFLOPS effective (~3.5% of the
//! 4090's fp32 peak), because every fast GEMM path in `prefill_project` is
//! macOS-only (hand-written MSL) and cubecl-wgpu **panics** on CoopMma on every
//! platform (`Operation::CoopMma => panic!` — verified in the vendored
//! compiler source). Issue 734 T1a measured the consequence at P=2048: FFN
//! stages are 58.0% of the 46.9 s block, the GDN block 34.3% — both dominated
//! by these projections.
//!
//! This kernel is the classic GEMM answer: **one workgroup computes a
//! 64 × 64 output tile** (rows × tokens) with 256 threads, staging a
//! dequantized weight tile `[64 × 32]` + an activation tile `[32 × 64]` in
//! workgroup storage per K-chunk (TK=32 = one packed u32 word per row), then
//! running 16 f32 accumulators per thread over the staged tiles:
//!
//! ```text
//! per K-chunk per thread: ~8 dequant writes + 8 activation loads + 512 FMA
//! ```
//!
//! vs the Issue 637 kernel's 2-token amortization and per-(row,word) input
//! re-reads. Weight DRAM traffic per token-tile is amortized across all 64
//! tokens of the tile (the current kernel amortizes across 2).
//!
//! # Thread ownership
//!
//! Thread `t` (0..512) = `(gr = t/16, gq = t%16)` owns output rows
//! `{gr + ir*32}` (ir ∈ 0..4) × tokens `{gq*8 + iq}` (iq ∈ 0..8) — a 4-row ×
//! 8-token patch, 32 accumulators, register-blocked (per k: 4 wk + 8 xk
//! register loads, 32 FMAs) (cube locals must be compile-time
//! indexed, so no `acc[i]` arrays — the Issue 637 kernel's named-accumulator
//! pattern).
//!
//! # Numerics (G1 contract)
//!
//! Same math as the Issue 637 kernel — `select(pos) - select(neg)` per weight
//! element, scaled per group — with a DIFFERENT summation order (per-thread
//! sequential over K-chunks, no plane reduction), so like-for-like G1 is a
//! **relative-error gate** (≤1e-3 vs the Issue 637 kernel on random weights at
//! production shapes), never bit-identity.
//!
//! # Dispatch
//!
//! - CubeDim `new_1d(256)`
//! - CubeCount `Static(ceil(m/64), ceil(p/64), 1)` — both axes far below the
//!   65535 workgroup cap at production shapes (m≤34816 → ≤545; p≤4096 → 64).
//!
//! # Status
//!
//! Opt-in (dispatched behind `set_prefill_use_tiled_gemm(true)` in
//! `prefill_project`) until the Issue 734 GOAT lands: G1 relative-error +
//! G2 prefill tok/s vs the 43.69 @2048 baseline.

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::TernaryHandle;

/// Output rows per workgroup tile.
pub const TILE_M: u32 = 128;
/// Output tokens per workgroup tile.
pub const TILE_P: u32 = 128;
/// K-chunk (columns) per staged pass — one packed u32 word per row.
pub const TILE_K: u32 = 32;
/// Threads per workgroup.
pub const TILE_THREADS: u32 = 512;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Workgroup-tiled ternary GEMM: `output[p × m] = dequant(W[m × n]) @ input[p × n]^T`.
///
/// See the module docs for the tile/ownership scheme. Rows past `m` clamp to
/// row `m-1` and their results are dropped at write time (the Issue 637
/// kernel's pattern — tile-uniform guards, no divergence inside the K loop).
/// `n` is always a multiple of 32 (the ternary group size is 128), so the word
/// loop needs no column guard.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_tiled64(
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

    let tid = UNIT_POS; // 0..512
    // CUBE_POS_X/Y are the WORKGROUP indices (ABSOLUTE_POS_* are positions
    // within the cube — the Issue 637 kernel divides them by PLANE_DIM for
    // exactly this reason).
    let tile_row0 = CUBE_POS_X * TILE_M;
    let tile_tok0 = CUBE_POS_Y * TILE_P;

    if tile_row0 >= m || tile_tok0 >= p_tokens {
        terminate!();
    }

    // Workgroup-staged tiles.
    let mut w_tile = Shared::<[f32]>::new_slice((TILE_M * TILE_K) as usize);
    let mut x_tile = Shared::<[f32]>::new_slice((TILE_K * TILE_P) as usize);

    // Thread ownership: gr = tid/16 ∈ 0..32 covers rows {gr + ir*32} (ir ∈
    // 0..4 → 128 rows); gq = tid%16 ∈ 0..16 covers tokens {gq*8 + iq} (iq ∈
    // 0..8 → 128 tokens) — 32 outputs/thread, named accumulators (cube locals
    // are SSA; no runtime-indexed arrays — the Issue 637 kernel's pattern).
    let gr = tid / 16u32;
    let gq = tid % 16u32;

    let mut a_r0q0 = f32::new(0.0f32);
    let mut a_r0q1 = f32::new(0.0f32);
    let mut a_r0q2 = f32::new(0.0f32);
    let mut a_r0q3 = f32::new(0.0f32);
    let mut a_r0q4 = f32::new(0.0f32);
    let mut a_r0q5 = f32::new(0.0f32);
    let mut a_r0q6 = f32::new(0.0f32);
    let mut a_r0q7 = f32::new(0.0f32);
    let mut a_r1q0 = f32::new(0.0f32);
    let mut a_r1q1 = f32::new(0.0f32);
    let mut a_r1q2 = f32::new(0.0f32);
    let mut a_r1q3 = f32::new(0.0f32);
    let mut a_r1q4 = f32::new(0.0f32);
    let mut a_r1q5 = f32::new(0.0f32);
    let mut a_r1q6 = f32::new(0.0f32);
    let mut a_r1q7 = f32::new(0.0f32);
    let mut a_r2q0 = f32::new(0.0f32);
    let mut a_r2q1 = f32::new(0.0f32);
    let mut a_r2q2 = f32::new(0.0f32);
    let mut a_r2q3 = f32::new(0.0f32);
    let mut a_r2q4 = f32::new(0.0f32);
    let mut a_r2q5 = f32::new(0.0f32);
    let mut a_r2q6 = f32::new(0.0f32);
    let mut a_r2q7 = f32::new(0.0f32);
    let mut a_r3q0 = f32::new(0.0f32);
    let mut a_r3q1 = f32::new(0.0f32);
    let mut a_r3q2 = f32::new(0.0f32);
    let mut a_r3q3 = f32::new(0.0f32);
    let mut a_r3q4 = f32::new(0.0f32);
    let mut a_r3q5 = f32::new(0.0f32);
    let mut a_r3q6 = f32::new(0.0f32);
    let mut a_r3q7 = f32::new(0.0f32);

    // Local row indices (this thread's 4 rows).
    let lr0 = gr;
    let lr1 = gr + 32u32;
    let lr2 = gr + 64u32;
    let lr3 = gr + 96u32;

    // Local token indices (this thread's 8 tokens).
    let lq0 = gq * 8u32;
    let lq1 = gq * 8u32 + 1u32;
    let lq2 = gq * 8u32 + 2u32;
    let lq3 = gq * 8u32 + 3u32;
    let lq4 = gq * 8u32 + 4u32;
    let lq5 = gq * 8u32 + 5u32;
    let lq6 = gq * 8u32 + 6u32;
    let lq7 = gq * 8u32 + 7u32;

    let num_chunks = n.div_ceil(TILE_K);
    for chunk in 0u32..num_chunks {
        let w = chunk; // one word per row per chunk (TK=32 columns)
        let col_base = chunk * TILE_K;

        // ── Stage 1: dequantize the weight tile [TILE_M rows × TILE_K cols] ──
        // 4096 elements; 512 threads → 8 each. Element e: local row i=e/32,
        // bit c=e%32 of the (row, w) word, scaled by the group scale.
        let mut e = tid;
        for _s1 in 0u32..8u32 {
            let i = e / TILE_K;
            let c = e % TILE_K;
            let row = tile_row0 + i;
            let row_c = if row < m { row } else { m - 1u32 };
            let pw = pos_bits_u32[(row_c * words_per_row + w) as usize];
            let nw = neg_bits_u32[(row_c * words_per_row + w) as usize];
            let s = group_scale_f32[(row_c * groups_per_row + (w / 4u32)) as usize];
            let v = select((pw >> c) & 1u32 != 0u32, s, f32::new(0.0f32))
                - select((nw >> c) & 1u32 != 0u32, s, f32::new(0.0f32));
            w_tile[e as usize] = v;
            e += TILE_THREADS;
        }

        // ── Stage 2: load the activation tile [TILE_K cols × TILE_P tokens] ──
        // 4096 elements; 512 threads → 8 each. x_tile[k][q] =
        // input[token_clamped(tile_tok0+q) * n + col_base+k].
        let mut e2 = tid;
        for _s2 in 0u32..8u32 {
            let k = e2 / TILE_P;
            let q = e2 % TILE_P;
            let tok = tile_tok0 + q;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            let x = input_batch[(tok_c * n + col_base + k) as usize];
            x_tile[e2 as usize] = x;
            e2 += TILE_THREADS;
        }

        sync_cube();

        // ── Stage 3: FMA over the staged chunk — register-blocked 4×8 patch ──
        // Per k: 12 workgroup-storage loads into REGISTERS (4 w + 8 x), then
        // 32 FMAs from registers (2.67 FMA/load — the naive per-FMA LDS reads
        // cap the kernel at LDS bandwidth; measured 15→18 TFLOPS at 2:1,
        // this pushes further). Script-generated — regenerate, don't hand-edit.
        #[unroll]
        for k in 0u32..32u32 {
            let wk0 = w_tile[(lr0 * TILE_K + k) as usize];
            let wk1 = w_tile[(lr1 * TILE_K + k) as usize];
            let wk2 = w_tile[(lr2 * TILE_K + k) as usize];
            let wk3 = w_tile[(lr3 * TILE_K + k) as usize];
            let xk0 = x_tile[(k * TILE_P + lq0) as usize];
            let xk1 = x_tile[(k * TILE_P + lq1) as usize];
            let xk2 = x_tile[(k * TILE_P + lq2) as usize];
            let xk3 = x_tile[(k * TILE_P + lq3) as usize];
            let xk4 = x_tile[(k * TILE_P + lq4) as usize];
            let xk5 = x_tile[(k * TILE_P + lq5) as usize];
            let xk6 = x_tile[(k * TILE_P + lq6) as usize];
            let xk7 = x_tile[(k * TILE_P + lq7) as usize];
            a_r0q0 += wk0 * xk0;
            a_r0q1 += wk0 * xk1;
            a_r0q2 += wk0 * xk2;
            a_r0q3 += wk0 * xk3;
            a_r0q4 += wk0 * xk4;
            a_r0q5 += wk0 * xk5;
            a_r0q6 += wk0 * xk6;
            a_r0q7 += wk0 * xk7;
            a_r1q0 += wk1 * xk0;
            a_r1q1 += wk1 * xk1;
            a_r1q2 += wk1 * xk2;
            a_r1q3 += wk1 * xk3;
            a_r1q4 += wk1 * xk4;
            a_r1q5 += wk1 * xk5;
            a_r1q6 += wk1 * xk6;
            a_r1q7 += wk1 * xk7;
            a_r2q0 += wk2 * xk0;
            a_r2q1 += wk2 * xk1;
            a_r2q2 += wk2 * xk2;
            a_r2q3 += wk2 * xk3;
            a_r2q4 += wk2 * xk4;
            a_r2q5 += wk2 * xk5;
            a_r2q6 += wk2 * xk6;
            a_r2q7 += wk2 * xk7;
            a_r3q0 += wk3 * xk0;
            a_r3q1 += wk3 * xk1;
            a_r3q2 += wk3 * xk2;
            a_r3q3 += wk3 * xk3;
            a_r3q4 += wk3 * xk4;
            a_r3q5 += wk3 * xk5;
            a_r3q6 += wk3 * xk6;
            a_r3q7 += wk3 * xk7;
        }

        sync_cube();
    }

    // ── Writeback: guarded 32-element scatter (script-generated) ──
    let wr0 = tile_row0 + lr0;
    let wr1 = tile_row0 + lr1;
    let wr2 = tile_row0 + lr2;
    let wr3 = tile_row0 + lr3;
    let wq0 = tile_tok0 + lq0;
    let wq1 = tile_tok0 + lq1;
    let wq2 = tile_tok0 + lq2;
    let wq3 = tile_tok0 + lq3;
    let wq4 = tile_tok0 + lq4;
    let wq5 = tile_tok0 + lq5;
    let wq6 = tile_tok0 + lq6;
    let wq7 = tile_tok0 + lq7;

    if wr0 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr0) as usize] = a_r0q0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr0) as usize] = a_r0q1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr0) as usize] = a_r0q2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr0) as usize] = a_r0q3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr0) as usize] = a_r0q4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr0) as usize] = a_r0q5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr0) as usize] = a_r0q6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr0) as usize] = a_r0q7; }
    }
    if wr1 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr1) as usize] = a_r1q0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr1) as usize] = a_r1q1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr1) as usize] = a_r1q2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr1) as usize] = a_r1q3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr1) as usize] = a_r1q4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr1) as usize] = a_r1q5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr1) as usize] = a_r1q6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr1) as usize] = a_r1q7; }
    }
    if wr2 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr2) as usize] = a_r2q0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr2) as usize] = a_r2q1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr2) as usize] = a_r2q2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr2) as usize] = a_r2q3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr2) as usize] = a_r2q4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr2) as usize] = a_r2q5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr2) as usize] = a_r2q6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr2) as usize] = a_r2q7; }
    }
    if wr3 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr3) as usize] = a_r3q0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr3) as usize] = a_r3q1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr3) as usize] = a_r3q2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr3) as usize] = a_r3q3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr3) as usize] = a_r3q4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr3) as usize] = a_r3q5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr3) as usize] = a_r3q6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr3) as usize] = a_r3q7; }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Workgroup-tiled ternary dequant+GEMM launcher (Issue 734 T3).
///
/// Computes `output_batch[p_tokens × m] = dequant_ternary(weight[m × n]) @
/// input_batch[p_tokens × n]^T` — the same contract as
/// [`crate::gemm_ternary_batched_cubecl::GemmTernaryBatchedCubeCL`] with a
/// 64×64 workgroup tile instead of the 4×2 plane tile.
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryTiledCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmTernaryTiledCubeCL {
    /// Launch the tiled kernel.
    ///
    /// # Safety
    ///
    /// - `input_handle` must hold `p_tokens × handle.n` f32 elements
    /// - `output_handle` must hold `p_tokens × handle.m` f32 elements
    /// - `p_tokens > 0`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(TILE_K),
            "n must be a multiple of {TILE_K} (the ternary group size 128 guarantees it: n%128==0 => n%64==0)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = m.div_ceil(TILE_M).max(1);
        let num_wg_y = p.div_ceil(TILE_P).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_tiled64::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(TILE_THREADS),
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
}

// ---------------------------------------------------------------------------
// Issue 734 T6 — bank-conflict fixes + deeper register blocking
// ---------------------------------------------------------------------------
//
// The tuned kernel above runs at 19.88 TFLOPS (24% of the 4090's fp32 peak)
// with a 2.67:1 FMA:LDS-load ratio — and its workgroup-storage reads are
// BANK-CONFLICTED:
//
// - `w_tile[(lr * 32 + k)]`: every lane's address is a multiple of 32 words
//   plus k — the two 16-lane row groups read two addresses that land in ONE
//   bank -> 2-way conflict on every weight register load.
// - `x_tile[(k * 128 + gq*8 + iq)]`: the 8-word thread stride puts the 16
//   per-half-warp addresses on 4 banks -> 4-way conflict on every activation
//   load.
//
// Both variants below keep the EXACT per-output-element summation order of
// the shipping kernel (k ascending within one thread — an ownership remap
// changes WHO computes an output, never the accumulation order), so their
// outputs are bit-identical by construction. Script-generated bodies
// (`target/gen_t6_variants.py`) — regenerate, don't hand-edit.
//
// - `gemm_ternary_tiled_xfix` (variant B): token ownership remapped to
//   `lq = gq + iq*16` — consecutive lanes read consecutive banks (16
//   broadcast-clean addresses per warp). Weight loads keep their 2-way
//   conflict (isolates the x-fix gain). 512 threads, 32 acc/thread,
//   32 KB shared (unchanged).
// - `gemm_ternary_tiled_8x8` (variant C): 256 threads x 64 acc (8 rows x
//   8 tokens — 8:1 FMA:load) + w_tile row stride padded to 33 (the two
//   row-group banks split) + the x remap: every per-k load broadcast-clean.
//   33.3 KB shared.

/// Issue 734 T6 variant B — conflict-free activation reads (x ownership remap).
///
/// Identical geometry/arithmetic to [`gemm_ternary_tiled64`]; only the token
/// ownership mapping changes (`lq = gq + iq*16`), which changes which thread
/// computes which output — never the k-ascending accumulation order. Output is
/// bit-identical to [`gemm_ternary_tiled64`] by construction.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_tiled_xfix(
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

    let tid = UNIT_POS; // 0..512
    let tile_row0 = CUBE_POS_X * TILE_M;
    let tile_tok0 = CUBE_POS_Y * TILE_P;

    if tile_row0 >= m || tile_tok0 >= p_tokens {
        terminate!();
    }

    let mut w_tile = Shared::<[f32]>::new_slice((TILE_M * TILE_K) as usize);
    let mut x_tile = Shared::<[f32]>::new_slice((TILE_K * TILE_P) as usize);

    let gr = tid / 16u32;
    let gq = tid % 16u32;

    let mut a_0_0 = f32::new(0.0f32);
    let mut a_0_1 = f32::new(0.0f32);
    let mut a_0_2 = f32::new(0.0f32);
    let mut a_0_3 = f32::new(0.0f32);
    let mut a_0_4 = f32::new(0.0f32);
    let mut a_0_5 = f32::new(0.0f32);
    let mut a_0_6 = f32::new(0.0f32);
    let mut a_0_7 = f32::new(0.0f32);
    let mut a_1_0 = f32::new(0.0f32);
    let mut a_1_1 = f32::new(0.0f32);
    let mut a_1_2 = f32::new(0.0f32);
    let mut a_1_3 = f32::new(0.0f32);
    let mut a_1_4 = f32::new(0.0f32);
    let mut a_1_5 = f32::new(0.0f32);
    let mut a_1_6 = f32::new(0.0f32);
    let mut a_1_7 = f32::new(0.0f32);
    let mut a_2_0 = f32::new(0.0f32);
    let mut a_2_1 = f32::new(0.0f32);
    let mut a_2_2 = f32::new(0.0f32);
    let mut a_2_3 = f32::new(0.0f32);
    let mut a_2_4 = f32::new(0.0f32);
    let mut a_2_5 = f32::new(0.0f32);
    let mut a_2_6 = f32::new(0.0f32);
    let mut a_2_7 = f32::new(0.0f32);
    let mut a_3_0 = f32::new(0.0f32);
    let mut a_3_1 = f32::new(0.0f32);
    let mut a_3_2 = f32::new(0.0f32);
    let mut a_3_3 = f32::new(0.0f32);
    let mut a_3_4 = f32::new(0.0f32);
    let mut a_3_5 = f32::new(0.0f32);
    let mut a_3_6 = f32::new(0.0f32);
    let mut a_3_7 = f32::new(0.0f32);

    let lr0 = gr;
    let lr1 = gr + 32u32;
    let lr2 = gr + 64u32;
    let lr3 = gr + 96u32;
    let lq0 = gq;
    let lq1 = gq + 16u32;
    let lq2 = gq + 32u32;
    let lq3 = gq + 48u32;
    let lq4 = gq + 64u32;
    let lq5 = gq + 80u32;
    let lq6 = gq + 96u32;
    let lq7 = gq + 112u32;

    let num_chunks = n.div_ceil(TILE_K);
    for chunk in 0u32..num_chunks {
        let w = chunk;
        let col_base = chunk * TILE_K;

        // Stage 1: dequantize the weight tile — 4096 elements, 512 threads x 8.
        let mut e = tid;
        for _s1 in 0u32..8u32 {
            let i = e / TILE_K;
            let c = e % TILE_K;
            let row = tile_row0 + i;
            let row_c = if row < m { row } else { m - 1u32 };
            let pw = pos_bits_u32[(row_c * words_per_row + w) as usize];
            let nw = neg_bits_u32[(row_c * words_per_row + w) as usize];
            let s = group_scale_f32[(row_c * groups_per_row + (w / 4u32)) as usize];
            let v = select((pw >> c) & 1u32 != 0u32, s, f32::new(0.0f32))
                - select((nw >> c) & 1u32 != 0u32, s, f32::new(0.0f32));
            w_tile[e as usize] = v;
            e += 512u32;
        }

        // Stage 2: activation tile — 4096 elements, 512 threads x 8.
        let mut e2 = tid;
        for _s2 in 0u32..8u32 {
            let k = e2 / TILE_P;
            let q = e2 % TILE_P;
            let tok = tile_tok0 + q;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            let x = input_batch[(tok_c * n + col_base + k) as usize];
            x_tile[e2 as usize] = x;
            e2 += 512u32;
        }

        sync_cube();

        // Stage 3: FMA over the staged chunk — register-blocked 4x8 patch.
        #[unroll]
        for k in 0u32..32u32 {
            let wk0 = w_tile[(lr0 * 32u32 + k) as usize];
            let wk1 = w_tile[(lr1 * 32u32 + k) as usize];
            let wk2 = w_tile[(lr2 * 32u32 + k) as usize];
            let wk3 = w_tile[(lr3 * 32u32 + k) as usize];
            let xk0 = x_tile[(k * TILE_P + lq0) as usize];
            let xk1 = x_tile[(k * TILE_P + lq1) as usize];
            let xk2 = x_tile[(k * TILE_P + lq2) as usize];
            let xk3 = x_tile[(k * TILE_P + lq3) as usize];
            let xk4 = x_tile[(k * TILE_P + lq4) as usize];
            let xk5 = x_tile[(k * TILE_P + lq5) as usize];
            let xk6 = x_tile[(k * TILE_P + lq6) as usize];
            let xk7 = x_tile[(k * TILE_P + lq7) as usize];
            a_0_0 += wk0 * xk0;
            a_0_1 += wk0 * xk1;
            a_0_2 += wk0 * xk2;
            a_0_3 += wk0 * xk3;
            a_0_4 += wk0 * xk4;
            a_0_5 += wk0 * xk5;
            a_0_6 += wk0 * xk6;
            a_0_7 += wk0 * xk7;
            a_1_0 += wk1 * xk0;
            a_1_1 += wk1 * xk1;
            a_1_2 += wk1 * xk2;
            a_1_3 += wk1 * xk3;
            a_1_4 += wk1 * xk4;
            a_1_5 += wk1 * xk5;
            a_1_6 += wk1 * xk6;
            a_1_7 += wk1 * xk7;
            a_2_0 += wk2 * xk0;
            a_2_1 += wk2 * xk1;
            a_2_2 += wk2 * xk2;
            a_2_3 += wk2 * xk3;
            a_2_4 += wk2 * xk4;
            a_2_5 += wk2 * xk5;
            a_2_6 += wk2 * xk6;
            a_2_7 += wk2 * xk7;
            a_3_0 += wk3 * xk0;
            a_3_1 += wk3 * xk1;
            a_3_2 += wk3 * xk2;
            a_3_3 += wk3 * xk3;
            a_3_4 += wk3 * xk4;
            a_3_5 += wk3 * xk5;
            a_3_6 += wk3 * xk6;
            a_3_7 += wk3 * xk7;
        }

        sync_cube();
    }

    let wr0 = tile_row0 + lr0;
    let wr1 = tile_row0 + lr1;
    let wr2 = tile_row0 + lr2;
    let wr3 = tile_row0 + lr3;
    let wq0 = tile_tok0 + lq0;
    let wq1 = tile_tok0 + lq1;
    let wq2 = tile_tok0 + lq2;
    let wq3 = tile_tok0 + lq3;
    let wq4 = tile_tok0 + lq4;
    let wq5 = tile_tok0 + lq5;
    let wq6 = tile_tok0 + lq6;
    let wq7 = tile_tok0 + lq7;

    if wr0 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr0) as usize] = a_0_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr0) as usize] = a_0_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr0) as usize] = a_0_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr0) as usize] = a_0_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr0) as usize] = a_0_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr0) as usize] = a_0_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr0) as usize] = a_0_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr0) as usize] = a_0_7; }
    }
    if wr1 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr1) as usize] = a_1_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr1) as usize] = a_1_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr1) as usize] = a_1_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr1) as usize] = a_1_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr1) as usize] = a_1_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr1) as usize] = a_1_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr1) as usize] = a_1_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr1) as usize] = a_1_7; }
    }
    if wr2 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr2) as usize] = a_2_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr2) as usize] = a_2_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr2) as usize] = a_2_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr2) as usize] = a_2_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr2) as usize] = a_2_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr2) as usize] = a_2_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr2) as usize] = a_2_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr2) as usize] = a_2_7; }
    }
    if wr3 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr3) as usize] = a_3_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr3) as usize] = a_3_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr3) as usize] = a_3_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr3) as usize] = a_3_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr3) as usize] = a_3_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr3) as usize] = a_3_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr3) as usize] = a_3_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr3) as usize] = a_3_7; }
    }
}

/// Issue 734 T6 variant B launcher (x-remap, 512 threads).
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryTiledXfixCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmTernaryTiledXfixCubeCL {
    /// Launch. Same contract as [`GemmTernaryTiledCubeCL::launch`].
    ///
    /// # Safety
    ///
    /// - `input_handle` must hold `p_tokens x handle.n` f32 elements
    /// - `output_handle` must hold `p_tokens x handle.m` f32 elements
    /// - `p_tokens > 0`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(TILE_K),
            "n must be a multiple of {TILE_K} (the ternary group size 128 guarantees it)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = m.div_ceil(TILE_M).max(1);
        let num_wg_y = p.div_ceil(TILE_P).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_tiled_xfix::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(512),
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
}

/// Issue 734 T6 variant C — 8x8 register blocking, all loads conflict-free.
///
/// 256 threads x 64 accumulators (8 rows x 8 tokens — 8:1 FMA:load), w_tile
/// row stride padded to 33 (splits the two row-group banks), x ownership
/// remapped (`lq = gq + iq*16`). Per-output accumulation order unchanged —
/// bit-identical to [`gemm_ternary_tiled64`] by construction. 33.3 KB
/// workgroup storage.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_tiled_8x8(
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

    let tid = UNIT_POS; // 0..256
    let tile_row0 = CUBE_POS_X * TILE_M;
    let tile_tok0 = CUBE_POS_Y * TILE_P;

    if tile_row0 >= m || tile_tok0 >= p_tokens {
        terminate!();
    }

    // Row stride 33 (padded): the two 16-lane row groups' loads split banks.
    let mut w_tile = Shared::<[f32]>::new_slice((TILE_M * 33u32) as usize);
    let mut x_tile = Shared::<[f32]>::new_slice((TILE_K * TILE_P) as usize);

    let gr = tid / 16u32;
    let gq = tid % 16u32;

    let mut a_0_0 = f32::new(0.0f32);
    let mut a_0_1 = f32::new(0.0f32);
    let mut a_0_2 = f32::new(0.0f32);
    let mut a_0_3 = f32::new(0.0f32);
    let mut a_0_4 = f32::new(0.0f32);
    let mut a_0_5 = f32::new(0.0f32);
    let mut a_0_6 = f32::new(0.0f32);
    let mut a_0_7 = f32::new(0.0f32);
    let mut a_1_0 = f32::new(0.0f32);
    let mut a_1_1 = f32::new(0.0f32);
    let mut a_1_2 = f32::new(0.0f32);
    let mut a_1_3 = f32::new(0.0f32);
    let mut a_1_4 = f32::new(0.0f32);
    let mut a_1_5 = f32::new(0.0f32);
    let mut a_1_6 = f32::new(0.0f32);
    let mut a_1_7 = f32::new(0.0f32);
    let mut a_2_0 = f32::new(0.0f32);
    let mut a_2_1 = f32::new(0.0f32);
    let mut a_2_2 = f32::new(0.0f32);
    let mut a_2_3 = f32::new(0.0f32);
    let mut a_2_4 = f32::new(0.0f32);
    let mut a_2_5 = f32::new(0.0f32);
    let mut a_2_6 = f32::new(0.0f32);
    let mut a_2_7 = f32::new(0.0f32);
    let mut a_3_0 = f32::new(0.0f32);
    let mut a_3_1 = f32::new(0.0f32);
    let mut a_3_2 = f32::new(0.0f32);
    let mut a_3_3 = f32::new(0.0f32);
    let mut a_3_4 = f32::new(0.0f32);
    let mut a_3_5 = f32::new(0.0f32);
    let mut a_3_6 = f32::new(0.0f32);
    let mut a_3_7 = f32::new(0.0f32);
    let mut a_4_0 = f32::new(0.0f32);
    let mut a_4_1 = f32::new(0.0f32);
    let mut a_4_2 = f32::new(0.0f32);
    let mut a_4_3 = f32::new(0.0f32);
    let mut a_4_4 = f32::new(0.0f32);
    let mut a_4_5 = f32::new(0.0f32);
    let mut a_4_6 = f32::new(0.0f32);
    let mut a_4_7 = f32::new(0.0f32);
    let mut a_5_0 = f32::new(0.0f32);
    let mut a_5_1 = f32::new(0.0f32);
    let mut a_5_2 = f32::new(0.0f32);
    let mut a_5_3 = f32::new(0.0f32);
    let mut a_5_4 = f32::new(0.0f32);
    let mut a_5_5 = f32::new(0.0f32);
    let mut a_5_6 = f32::new(0.0f32);
    let mut a_5_7 = f32::new(0.0f32);
    let mut a_6_0 = f32::new(0.0f32);
    let mut a_6_1 = f32::new(0.0f32);
    let mut a_6_2 = f32::new(0.0f32);
    let mut a_6_3 = f32::new(0.0f32);
    let mut a_6_4 = f32::new(0.0f32);
    let mut a_6_5 = f32::new(0.0f32);
    let mut a_6_6 = f32::new(0.0f32);
    let mut a_6_7 = f32::new(0.0f32);
    let mut a_7_0 = f32::new(0.0f32);
    let mut a_7_1 = f32::new(0.0f32);
    let mut a_7_2 = f32::new(0.0f32);
    let mut a_7_3 = f32::new(0.0f32);
    let mut a_7_4 = f32::new(0.0f32);
    let mut a_7_5 = f32::new(0.0f32);
    let mut a_7_6 = f32::new(0.0f32);
    let mut a_7_7 = f32::new(0.0f32);

    let lr0 = gr;
    let lr1 = gr + 16u32;
    let lr2 = gr + 32u32;
    let lr3 = gr + 48u32;
    let lr4 = gr + 64u32;
    let lr5 = gr + 80u32;
    let lr6 = gr + 96u32;
    let lr7 = gr + 112u32;
    let lq0 = gq;
    let lq1 = gq + 16u32;
    let lq2 = gq + 32u32;
    let lq3 = gq + 48u32;
    let lq4 = gq + 64u32;
    let lq5 = gq + 80u32;
    let lq6 = gq + 96u32;
    let lq7 = gq + 112u32;

    let num_chunks = n.div_ceil(TILE_K);
    for chunk in 0u32..num_chunks {
        let w = chunk;
        let col_base = chunk * TILE_K;

        // Stage 1: dequantize the weight tile — 4096 elements, 256 threads x 16.
        let mut e = tid;
        for _s1 in 0u32..16u32 {
            let i = e / TILE_K;
            let c = e % TILE_K;
            let row = tile_row0 + i;
            let row_c = if row < m { row } else { m - 1u32 };
            let pw = pos_bits_u32[(row_c * words_per_row + w) as usize];
            let nw = neg_bits_u32[(row_c * words_per_row + w) as usize];
            let s = group_scale_f32[(row_c * groups_per_row + (w / 4u32)) as usize];
            let v = select((pw >> c) & 1u32 != 0u32, s, f32::new(0.0f32))
                - select((nw >> c) & 1u32 != 0u32, s, f32::new(0.0f32));
            w_tile[(i * 33u32 + c) as usize] = v;
            e += 256u32;
        }

        // Stage 2: activation tile — 4096 elements, 256 threads x 16.
        let mut e2 = tid;
        for _s2 in 0u32..16u32 {
            let k = e2 / TILE_P;
            let q = e2 % TILE_P;
            let tok = tile_tok0 + q;
            let tok_c = if tok < p_tokens { tok } else { p_tokens - 1u32 };
            let x = input_batch[(tok_c * n + col_base + k) as usize];
            x_tile[e2 as usize] = x;
            e2 += 256u32;
        }

        sync_cube();

        // Stage 3: FMA over the staged chunk — register-blocked 8x8 patch.
        #[unroll]
        for k in 0u32..32u32 {
            let wk0 = w_tile[(lr0 * 33u32 + k) as usize];
            let wk1 = w_tile[(lr1 * 33u32 + k) as usize];
            let wk2 = w_tile[(lr2 * 33u32 + k) as usize];
            let wk3 = w_tile[(lr3 * 33u32 + k) as usize];
            let wk4 = w_tile[(lr4 * 33u32 + k) as usize];
            let wk5 = w_tile[(lr5 * 33u32 + k) as usize];
            let wk6 = w_tile[(lr6 * 33u32 + k) as usize];
            let wk7 = w_tile[(lr7 * 33u32 + k) as usize];
            let xk0 = x_tile[(k * TILE_P + lq0) as usize];
            let xk1 = x_tile[(k * TILE_P + lq1) as usize];
            let xk2 = x_tile[(k * TILE_P + lq2) as usize];
            let xk3 = x_tile[(k * TILE_P + lq3) as usize];
            let xk4 = x_tile[(k * TILE_P + lq4) as usize];
            let xk5 = x_tile[(k * TILE_P + lq5) as usize];
            let xk6 = x_tile[(k * TILE_P + lq6) as usize];
            let xk7 = x_tile[(k * TILE_P + lq7) as usize];
            a_0_0 += wk0 * xk0;
            a_0_1 += wk0 * xk1;
            a_0_2 += wk0 * xk2;
            a_0_3 += wk0 * xk3;
            a_0_4 += wk0 * xk4;
            a_0_5 += wk0 * xk5;
            a_0_6 += wk0 * xk6;
            a_0_7 += wk0 * xk7;
            a_1_0 += wk1 * xk0;
            a_1_1 += wk1 * xk1;
            a_1_2 += wk1 * xk2;
            a_1_3 += wk1 * xk3;
            a_1_4 += wk1 * xk4;
            a_1_5 += wk1 * xk5;
            a_1_6 += wk1 * xk6;
            a_1_7 += wk1 * xk7;
            a_2_0 += wk2 * xk0;
            a_2_1 += wk2 * xk1;
            a_2_2 += wk2 * xk2;
            a_2_3 += wk2 * xk3;
            a_2_4 += wk2 * xk4;
            a_2_5 += wk2 * xk5;
            a_2_6 += wk2 * xk6;
            a_2_7 += wk2 * xk7;
            a_3_0 += wk3 * xk0;
            a_3_1 += wk3 * xk1;
            a_3_2 += wk3 * xk2;
            a_3_3 += wk3 * xk3;
            a_3_4 += wk3 * xk4;
            a_3_5 += wk3 * xk5;
            a_3_6 += wk3 * xk6;
            a_3_7 += wk3 * xk7;
            a_4_0 += wk4 * xk0;
            a_4_1 += wk4 * xk1;
            a_4_2 += wk4 * xk2;
            a_4_3 += wk4 * xk3;
            a_4_4 += wk4 * xk4;
            a_4_5 += wk4 * xk5;
            a_4_6 += wk4 * xk6;
            a_4_7 += wk4 * xk7;
            a_5_0 += wk5 * xk0;
            a_5_1 += wk5 * xk1;
            a_5_2 += wk5 * xk2;
            a_5_3 += wk5 * xk3;
            a_5_4 += wk5 * xk4;
            a_5_5 += wk5 * xk5;
            a_5_6 += wk5 * xk6;
            a_5_7 += wk5 * xk7;
            a_6_0 += wk6 * xk0;
            a_6_1 += wk6 * xk1;
            a_6_2 += wk6 * xk2;
            a_6_3 += wk6 * xk3;
            a_6_4 += wk6 * xk4;
            a_6_5 += wk6 * xk5;
            a_6_6 += wk6 * xk6;
            a_6_7 += wk6 * xk7;
            a_7_0 += wk7 * xk0;
            a_7_1 += wk7 * xk1;
            a_7_2 += wk7 * xk2;
            a_7_3 += wk7 * xk3;
            a_7_4 += wk7 * xk4;
            a_7_5 += wk7 * xk5;
            a_7_6 += wk7 * xk6;
            a_7_7 += wk7 * xk7;
        }

        sync_cube();
    }

    let wr0 = tile_row0 + lr0;
    let wr1 = tile_row0 + lr1;
    let wr2 = tile_row0 + lr2;
    let wr3 = tile_row0 + lr3;
    let wr4 = tile_row0 + lr4;
    let wr5 = tile_row0 + lr5;
    let wr6 = tile_row0 + lr6;
    let wr7 = tile_row0 + lr7;
    let wq0 = tile_tok0 + lq0;
    let wq1 = tile_tok0 + lq1;
    let wq2 = tile_tok0 + lq2;
    let wq3 = tile_tok0 + lq3;
    let wq4 = tile_tok0 + lq4;
    let wq5 = tile_tok0 + lq5;
    let wq6 = tile_tok0 + lq6;
    let wq7 = tile_tok0 + lq7;

    if wr0 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr0) as usize] = a_0_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr0) as usize] = a_0_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr0) as usize] = a_0_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr0) as usize] = a_0_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr0) as usize] = a_0_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr0) as usize] = a_0_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr0) as usize] = a_0_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr0) as usize] = a_0_7; }
    }
    if wr1 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr1) as usize] = a_1_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr1) as usize] = a_1_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr1) as usize] = a_1_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr1) as usize] = a_1_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr1) as usize] = a_1_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr1) as usize] = a_1_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr1) as usize] = a_1_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr1) as usize] = a_1_7; }
    }
    if wr2 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr2) as usize] = a_2_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr2) as usize] = a_2_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr2) as usize] = a_2_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr2) as usize] = a_2_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr2) as usize] = a_2_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr2) as usize] = a_2_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr2) as usize] = a_2_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr2) as usize] = a_2_7; }
    }
    if wr3 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr3) as usize] = a_3_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr3) as usize] = a_3_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr3) as usize] = a_3_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr3) as usize] = a_3_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr3) as usize] = a_3_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr3) as usize] = a_3_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr3) as usize] = a_3_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr3) as usize] = a_3_7; }
    }
    if wr4 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr4) as usize] = a_4_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr4) as usize] = a_4_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr4) as usize] = a_4_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr4) as usize] = a_4_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr4) as usize] = a_4_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr4) as usize] = a_4_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr4) as usize] = a_4_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr4) as usize] = a_4_7; }
    }
    if wr5 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr5) as usize] = a_5_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr5) as usize] = a_5_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr5) as usize] = a_5_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr5) as usize] = a_5_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr5) as usize] = a_5_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr5) as usize] = a_5_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr5) as usize] = a_5_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr5) as usize] = a_5_7; }
    }
    if wr6 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr6) as usize] = a_6_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr6) as usize] = a_6_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr6) as usize] = a_6_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr6) as usize] = a_6_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr6) as usize] = a_6_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr6) as usize] = a_6_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr6) as usize] = a_6_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr6) as usize] = a_6_7; }
    }
    if wr7 < m {
        if wq0 < p_tokens { output_batch[(wq0 * m + wr7) as usize] = a_7_0; }
        if wq1 < p_tokens { output_batch[(wq1 * m + wr7) as usize] = a_7_1; }
        if wq2 < p_tokens { output_batch[(wq2 * m + wr7) as usize] = a_7_2; }
        if wq3 < p_tokens { output_batch[(wq3 * m + wr7) as usize] = a_7_3; }
        if wq4 < p_tokens { output_batch[(wq4 * m + wr7) as usize] = a_7_4; }
        if wq5 < p_tokens { output_batch[(wq5 * m + wr7) as usize] = a_7_5; }
        if wq6 < p_tokens { output_batch[(wq6 * m + wr7) as usize] = a_7_6; }
        if wq7 < p_tokens { output_batch[(wq7 * m + wr7) as usize] = a_7_7; }
    }
}

/// Issue 734 T6 variant C launcher (8x8 blocking, 256 threads).
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryTiled8x8CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmTernaryTiled8x8CubeCL {
    /// Launch. Same contract as [`GemmTernaryTiledCubeCL::launch`].
    ///
    /// # Safety
    ///
    /// - `input_handle` must hold `p_tokens x handle.n` f32 elements
    /// - `output_handle` must hold `p_tokens x handle.m` f32 elements
    /// - `p_tokens > 0`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        let n = handle.n as u32;
        let m = handle.m as u32;
        debug_assert!(
            n.is_multiple_of(TILE_K),
            "n must be a multiple of {TILE_K} (the ternary group size 128 guarantees it)"
        );
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        let num_wg_x = m.div_ceil(TILE_M).max(1);
        let num_wg_y = p.div_ceil(TILE_P).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_tiled_8x8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(256),
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
}
