//! Batched ternary bit-plane dequant+GEM**M** kernel (Issue 637 T1).
//!
//! Extends [`crate::gemv_ternary_cubecl`] from single-vector GEMV to a batched
//! matrix-matrix multiply so a whole prompt can be pushed through a projection
//! in ONE dispatch:
//!
//! ```text
//! output_batch[P × m] = dequant_ternary(weight[m × n]) @ input_batch[P × n]^T
//! ```
//!
//! Both operands are row-major: `input_batch[tok * n + col]`,
//! `output_batch[tok * m + row]`. The weight is shared across the batch.
//!
//! # Why this exists (Issue 637)
//!
//! Every ternary kernel in `gemv_ternary_cubecl` takes a single `input_handle`
//! and a single `output_handle` — they are all GEM**V**, with no batch
//! dimension. The ternary forward therefore cannot batch a prompt even in
//! principle, and pushes prompt tokens through the one-token-at-a-time decode
//! path. riir-clippy Bench 010 measured the consequence on M3 Max Metal:
//! **prefill 17.89 tok/s vs decode 17.85 tok/s** — the same rate, against
//! llama.cpp's 149.95 prefill (8.38× ahead, earned by batching).
//!
//! # STATUS: correct but not a win — opt-in only, do NOT promote (Bench 641)
//!
//! G1 passes (worst relative error 2.0e-7 vs P sequential GEMVs). **G2 fails.**
//! The best measured tile reaches **1.08×** on the projection roll-up, **1.11×**
//! with per-shape best-of-both dispatch, against a ≥3× gate. Read the sweep
//! below before touching this file — the intended mechanism is NOT the one that
//! produces the measured wins, and the obvious "improvements" all make it worse.
//!
//! # The design rationale — and how the measurement refuted it
//!
//! This was written to be unlike [`crate::gemv_q4k_batched_cubecl`], which gives
//! each plane one `(position, row)` pair and so collapses *dispatch overhead*
//! while still re-reading weights once per position. Bench 010 shows the
//! resident ternary forward already runs at 121 GB/s against our own best
//! measured kernel of 132.6 GB/s (~91% of the kernel ceiling), so there is
//! almost no dispatch overhead left to reclaim — a dispatch-collapsing batch
//! "should" buy nothing. The win was therefore designed to come from two
//! amortizations:
//!
//! 1. **Weight DRAM traffic** — each `(row, word)` fetched once per token block
//!    rather than once per token, falling by [`TERNARY_GEMM_TOKENS_PER_PASS`]×.
//! 2. **Bit-extraction ALU** — the `(p >> b) & 1` sign tests depend only on
//!    `(row, bit)`, never on the token, so hoisting them above the token fan-out
//!    computes each once and reuses it across all `TB` tokens. Per row-token the
//!    inner-loop op count falls from `32 × (6 + 4)` to `32 × (6/TB + 4)`.
//!
//! Loads per row-token per word are `32/rows + 2/TB` — `rows` controls the
//! input-load count, `TB` the weight-load count. **Neither amortization is what
//! actually drives the measured result.** The tile sweep (P = 128, real Bonsai
//! shapes, same-session pairs) puts speedup in near-perfect *inverse* order to
//! the load math the design optimizes:
//!
//! | tile (rows × TB) | accumulators | loads/row-token | roll-up |
//! |---|---:|---:|---:|
//! | 2 × 2 | 4 | 17.0 | 0.43× |
//! | **4 × 2** | **8** | **9.0** | **1.08×** |
//! | 8 × 2 | 16 | 5.0 | 0.95× |
//! | 4 × 4 | 16 | 8.5 | 0.78× |
//! | 8 × 4 | 32 | 4.5 | 0.60× |
//!
//! The best load math (4.5) is the *worst* result, and the peak sits at an
//! interior point with mediocre load math. This is an occupancy trade, not a
//! bandwidth one: every accumulator held live across the word loop costs
//! occupancy, and past ~8 accumulators the lost parallelism outweighs the loads
//! saved — while below that, input-load pressure dominates instead. Both ends
//! of the sweep lose; only the middle is viable.
//!
//! # Where the wins actually come from
//!
//! Speedup is monotone in shape size, which is the tell — the same shader runs
//! everywhere, so only occupancy can vary with shape. At 4 × 2:
//!
//! | shape | rows | speedup |
//! |---|---:|---:|
//! | `ssm_alpha/beta` | 48 | 12.92× |
//! | `attn_k/v` | 1024 | 3.04× |
//! | `attn_gate` | 6144 | 1.21× |
//! | `ffn_gate/up` | 17408 | 1.00× |
//! | `ffn_down` | 5120 (cols 17408) | 0.90× |
//!
//! Small shapes leave the sequential arm launch-latency bound — 128 dispatches
//! of a kernel with one or two workgroups, which cannot fill the GPU — so
//! collapsing them into one dispatch is a large and genuine win. Large shapes
//! saturate the GPU in *both* arms, and there the batched kernel is at best at
//! parity per unit work. So the wins are launch-overhead amortization after
//! all — exactly the [`crate::gemv_q4k_batched_cubecl`] mechanism this kernel
//! was designed not to rely on. The designed weight/ALU amortization is real in
//! instruction counts but does not convert into throughput on M3 Metal.
//!
//! That caps the approach: `ffn_gate/up` and `ffn_down` alone are 63% of
//! projection time and neither can be won, so no tile choice reaches 3×.
//!
//! # Accumulator form
//!
//! A **single** partial per `(row, token)` —
//! `a += select(pos, x, 0) - select(neg, x, 0)` — rather than the GEMV kernel's
//! split `a_pos`/`a_neg` pair. Identical op count (2 selects + 1 sub + 1 add vs
//! 2 selects + 2 adds), half the registers, which the sweep shows is the axis
//! that matters. It also makes the result *not* bit-identical to the GEMV path
//! (the summation order differs), so G1 is a relative-error gate by
//! construction, not an equality gate.
//!
//! # Dispatch
//!
//! - CubeDim: `new_1d(256)` — 256 threads = 8 planes (Metal subgroup = 32)
//! - CubeCount: `Static(ceil(m / 32), ceil(P / 2), 1)`
//!   - X = row tiles (8 planes × 4 rows = 32 rows per workgroup)
//!   - Y = token tiles (2 tokens each)
//!
//! The 2D geometry mirrors [`crate::gemv_q4k_batched_cubecl`], which is the
//! shipped precedent that this dispatch shape works on wgpu/Metal and keeps
//! each dimension under the wgpu 65535 workgroup limit.

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Plane trait needed for plane_sum() resolution
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::TernaryHandle;

/// Output rows processed per plane.
///
/// Controls the **input**-load count: loads per row-token per word are
/// `32/rows + 2/TB`. 4 is the **measured** peak, not a derived one — widening
/// to 8 improves the load math (5.0 vs 9.0) yet *loses* (0.95× vs 1.08×), and
/// narrowing to 2 collapses to 0.43×. See the module docs' sweep table; the
/// binding constraint is accumulators held live across the word loop, not
/// loads.
///
/// Paired with the hand-unrolled accumulator count in
/// [`gemm_ternary_plane_rowtiled_batched`] — the two MUST agree or output rows
/// go unwritten. The G1 test sentinel-fills the output buffer precisely because
/// a silent mismatch is invisible to a relative-error check (CubeCL pools
/// buffers, so `client.empty()` can hand back a previous kernel's correct
/// results).
#[cfg(feature = "cubecl_runtime")]
pub(crate) const TERNARY_GEMM_ROWS_PER_PLANE: u32 = 4;

/// Prompt tokens processed per plane per pass.
///
/// Controls the **weight**-load count and the bit-extraction ALU share; both
/// fall by this factor per token in instruction counts. 2 keeps the tile at
/// `4 × 2 = 8` accumulators plus 8 per-word partials, which the sweep found to
/// be the occupancy sweet spot. Widening to 4 doubles the designed
/// amortization and measures **worse** on every shape (0.78× roll-up) — the
/// extra live state costs more occupancy than the amortization returns.
#[cfg(feature = "cubecl_runtime")]
pub(crate) const TERNARY_GEMM_TOKENS_PER_PASS: u32 = 2;

/// Rows covered by one workgroup: 8 planes × [`TERNARY_GEMM_ROWS_PER_PLANE`].
#[cfg(feature = "cubecl_runtime")]
const TERNARY_GEMM_ROWS_PER_WG: u32 = 32;

// ---------------------------------------------------------------------------
// Batched row-tiled ternary GEMM kernel
// ---------------------------------------------------------------------------

/// Row-tiled, token-tiled ternary bit-plane GEMM.
///
/// Each plane owns a `4 × 2` (row × token) output tile. Lanes stride over the
/// packed weight words cooperatively and `plane_sum()` reduces each of the 8
/// accumulators at the end.
///
/// Out-of-range rows and tokens are **clamped** rather than branched on, so
/// every load stays in bounds and the loop body is uniform across the plane;
/// the clamped results are discarded at write time. `row_base`/`tok_base`
/// depend only on the workgroup and plane index, never on `lane`, so the
/// guards are plane-uniform and cost no divergence.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_ternary_plane_rowtiled_batched(
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

    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let row_base = plane_id * TERNARY_GEMM_ROWS_PER_PLANE;
    let tok_base = ABSOLUTE_POS_Y * TERNARY_GEMM_TOKENS_PER_PASS;

    if row_base >= m || tok_base >= p_tokens {
        terminate!();
    }

    // Clamped row indices — rows past `m` alias row `m-1`.
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

    // Clamped token indices — tokens past `p_tokens` alias token `p_tokens-1`.
    // This is what makes ragged P (not a multiple of the tile) safe.
    let q0 = tok_base;
    let mut q1 = p_tokens - 1u32;
    if tok_base + 1u32 < p_tokens {
        q1 = tok_base + 1u32;
    }

    let mut acc00 = f32::new(0.0f32);
    let mut acc01 = f32::new(0.0f32);
    let mut acc10 = f32::new(0.0f32);
    let mut acc11 = f32::new(0.0f32);
    let mut acc20 = f32::new(0.0f32);
    let mut acc21 = f32::new(0.0f32);
    let mut acc30 = f32::new(0.0f32);
    let mut acc31 = f32::new(0.0f32);

    let mut w = lane;
    while w < words_per_row {
        let scale_idx = w / 4u32;
        let col_base = w * 32u32;

        // 8 weight loads — amortized across all 2 tokens. This is win #1.
        let p0 = pos_bits_u32[(r0 * words_per_row + w) as usize];
        let nw0 = neg_bits_u32[(r0 * words_per_row + w) as usize];
        let p1 = pos_bits_u32[(r1 * words_per_row + w) as usize];
        let nw1 = neg_bits_u32[(r1 * words_per_row + w) as usize];
        let p2 = pos_bits_u32[(r2 * words_per_row + w) as usize];
        let nw2 = neg_bits_u32[(r2 * words_per_row + w) as usize];
        let p3 = pos_bits_u32[(r3 * words_per_row + w) as usize];
        let nw3 = neg_bits_u32[(r3 * words_per_row + w) as usize];

        // Skip the word only when every row is zero across both planes.
        if (p0 | nw0 | p1 | nw1 | p2 | nw2 | p3 | nw3) != 0u32 {
            let s0 = group_scale_f32[(r0 * groups_per_row + scale_idx) as usize];
            let s1 = group_scale_f32[(r1 * groups_per_row + scale_idx) as usize];
            let s2 = group_scale_f32[(r2 * groups_per_row + scale_idx) as usize];
            let s3 = group_scale_f32[(r3 * groups_per_row + scale_idx) as usize];

            let mut a00 = f32::new(0.0f32);
            let mut a01 = f32::new(0.0f32);
            let mut a10 = f32::new(0.0f32);
            let mut a11 = f32::new(0.0f32);
            let mut a20 = f32::new(0.0f32);
            let mut a21 = f32::new(0.0f32);
            let mut a30 = f32::new(0.0f32);
            let mut a31 = f32::new(0.0f32);

            #[unroll]
            for b in 0u32..32u32 {
                let col = col_base + b;
                if col < n {
                    // Sign tests depend on (row, bit) only — computed ONCE and
                    // reused across all 2 tokens. This is win #2.
                    let pb0 = (p0 >> b) & 1u32 != 0u32;
                    let nb0 = (nw0 >> b) & 1u32 != 0u32;
                    let pb1 = (p1 >> b) & 1u32 != 0u32;
                    let nb1 = (nw1 >> b) & 1u32 != 0u32;
                    let pb2 = (p2 >> b) & 1u32 != 0u32;
                    let nb2 = (nw2 >> b) & 1u32 != 0u32;
                    let pb3 = (p3 >> b) & 1u32 != 0u32;
                    let nb3 = (nw3 >> b) & 1u32 != 0u32;

                    // One load per token, reused across all 4 rows.
                    let x0 = input_batch[(q0 * n + col) as usize];
                    let x1 = input_batch[(q1 * n + col) as usize];

                    a00 += select(pb0, x0, f32::new(0.0f32)) - select(nb0, x0, f32::new(0.0f32));
                    a01 += select(pb0, x1, f32::new(0.0f32)) - select(nb0, x1, f32::new(0.0f32));
                    a10 += select(pb1, x0, f32::new(0.0f32)) - select(nb1, x0, f32::new(0.0f32));
                    a11 += select(pb1, x1, f32::new(0.0f32)) - select(nb1, x1, f32::new(0.0f32));
                    a20 += select(pb2, x0, f32::new(0.0f32)) - select(nb2, x0, f32::new(0.0f32));
                    a21 += select(pb2, x1, f32::new(0.0f32)) - select(nb2, x1, f32::new(0.0f32));
                    a30 += select(pb3, x0, f32::new(0.0f32)) - select(nb3, x0, f32::new(0.0f32));
                    a31 += select(pb3, x1, f32::new(0.0f32)) - select(nb3, x1, f32::new(0.0f32));
                }
            }

            // Scale is per (row, group) — uniform across the token axis.
            acc00 += a00 * s0;
            acc01 += a01 * s0;
            acc10 += a10 * s1;
            acc11 += a11 * s1;
            acc20 += a20 * s2;
            acc21 += a21 * s2;
            acc30 += a30 * s3;
            acc31 += a31 * s3;
        }

        w += PLANE_DIM;
    }

    let o00 = plane_sum(acc00);
    let o01 = plane_sum(acc01);
    let o10 = plane_sum(acc10);
    let o11 = plane_sum(acc11);
    let o20 = plane_sum(acc20);
    let o21 = plane_sum(acc21);
    let o30 = plane_sum(acc30);
    let o31 = plane_sum(acc31);

    if lane == 0u32 {
        let r1_ok = row_base + 1u32 < m;
        let r2_ok = row_base + 2u32 < m;
        let r3_ok = row_base + 3u32 < m;

        output_batch[(q0 * m + r0) as usize] = o00;
        if r1_ok {
            output_batch[(q0 * m + r1) as usize] = o10;
        }
        if r2_ok {
            output_batch[(q0 * m + r2) as usize] = o20;
        }
        if r3_ok {
            output_batch[(q0 * m + r3) as usize] = o30;
        }

        if tok_base + 1u32 < p_tokens {
            output_batch[(q1 * m + r0) as usize] = o01;
            if r1_ok {
                output_batch[(q1 * m + r1) as usize] = o11;
            }
            if r2_ok {
                output_batch[(q1 * m + r2) as usize] = o21;
            }
            if r3_ok {
                output_batch[(q1 * m + r3) as usize] = o31;
            }
        }

    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Batched ternary bit-plane dequant+GEMM launcher (Issue 637).
///
/// Computes `output_batch[P × m] = dequant_ternary(weight) @ input_batch[P × n]^T`
/// in a single dispatch, amortizing weight loads and sign extraction across the
/// token axis. See the module docs for the tile shape and the two wins.
#[cfg(feature = "cubecl_runtime")]
pub struct GemmTernaryBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemmTernaryBatchedCubeCL {
    /// Launch the batched row-tiled ternary GEMM.
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
    /// - the device must support plane operations (Metal on Apple Silicon does);
    ///   call [`Self::has_plane`] to check
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
        p_tokens: usize,
    ) {
        debug_assert!(p_tokens > 0, "p_tokens must be non-zero");

        let wg_size = 256u32;

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let p = p_tokens as u32;

        // X = row tiles (8 planes × 4 rows), Y = token tiles (4 tokens each).
        let num_wg_x = m.div_ceil(TERNARY_GEMM_ROWS_PER_WG).max(1);
        let num_wg_y = p.div_ceil(TERNARY_GEMM_TOKENS_PER_PASS).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemm_ternary_plane_rowtiled_batched::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(wg_size),
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

    /// Check if the device supports plane (subgroup) operations.
    pub fn has_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
        client.features().plane.contains(Plane::Ops)
    }
}
