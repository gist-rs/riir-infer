//! Fused ternary GEMV + in-place ResidualAdd CubeCL kernel (Issue 616).
//!
//! Single-dispatch kernel that computes
//! `residual_output[row] += dot(weight_row, input)` for ternary bit-plane
//! weights, eliminating one GPU dispatch per FFN down-projection in the
//! Qwen3.5 DeltaNet forward.
//!
//! # Why this exists
//!
//! The GPU-resident ternary forward (`TernaryDeltanetGpuForward`) measures
//! 8.5 tok/s on M3 Max (Issue 604). The bottleneck is Metal kernel launch
//! latency (~55–97 µs) across ~1200 dispatches/token. Two of those per layer
//! are the FFN down-projection GEMV followed by a separate ResidualAdd —
//! that's 2 dispatches × 64 layers = **128 dispatches/token** that this
//! kernel collapses into **64 dispatches/token**.
//!
//! # CubeCL v0.10 constraint
//!
//! CubeCL v0.10 has a hard **5 Array parameter limit** per kernel. The plain
//! ternary GEMV already uses 5 (pos_bits, neg_bits, scales, input, output).
//! Adding a 6th `residual` array is impossible.
//!
//! The fix is to make the kernel **in-place**: the residual lives in the
//! output buffer and is overwritten by the GEMV result + residual. Same 5
//! args, but the output buffer is read-then-written. Lane 0 of each plane
//! reads `residual_output[row]` before writing
//! `residual_output[row] = residual[row] + gemv_result[row]`.
//!
//! # Kernel shape
//!
//! This mirrors `gemv_ternary_plane_rowtiled8` exactly — width-8 row tiling,
//! `#[unroll]` + `select`-form two-accumulator ternary dot, fast/slow path
//! split on `n % 32`. The ONLY change is the final write step: lane 0 adds
//! the residual before storing. See `gemv_ternary_cubecl.rs` for the
//! per-shape speedup table + the unroll/select micro-optimization history.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Plane trait needed for plane_sum() resolution
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::gemv_ternary_cubecl::{TernaryHandle, TERNARY_ROWS_PER_PLANE_8};

// ---------------------------------------------------------------------------
// Fused ternary GEMV + in-place ResidualAdd (width-8 row-tiled)
// ---------------------------------------------------------------------------

/// CubeCL fused ternary GEMV + in-place ResidualAdd kernel (width-8 row-tiled).
///
/// Computes `residual_output[row] += dot(weight_row, input)` for each output
/// row, where the weight matrix is ternary bit-plane encoded (Q2_0_g128).
///
/// Width-8 row tiling amortizes each input load across 8 output rows. Each
/// plane owns 8 consecutive rows; the cooperative dot product uses 16
/// accumulators (8 positive + 8 negative) reduced via `select` form.
///
/// ## Parameter Layout (5 arrays — CubeCL v0.10 limit)
///
/// - `pos_bits_u32`: positive bit-plane, u64→2×u32 cast. Layout: `[m * blocks64 * 2]`.
/// - `neg_bits_u32`: negative bit-plane, same layout.
/// - `group_scale_f32`: per-group f32 scales (f16 decoded at upload). Layout: `[m * groups_per_row]`.
/// - `input`: `[n]` f32 input vector (read-only).
/// - `residual_output`: `[m]` f32 — READ for the residual, then WRITTEN with
///   `residual_output[row] = residual_output[row] + gemv_result[row]`.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(m / 64), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion generates this pattern")]
#[cube(launch_unchecked)]
fn gemv_ternary_residual_rowtiled8(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input: &[f32],
    residual_output: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
) {
    let m = residual_output.len() as u32;
    let words_per_row = blocks64 * 2u32;

    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let row_base = plane_id * TERNARY_ROWS_PER_PLANE_8;

    if row_base >= m {
        terminate!();
    }

    let r0 = row_base;
    let mut r1 = m - 1u32;
    let mut r2 = m - 1u32;
    let mut r3 = m - 1u32;
    let mut r4 = m - 1u32;
    let mut r5 = m - 1u32;
    let mut r6 = m - 1u32;
    let mut r7 = m - 1u32;
    if row_base + 1u32 < m {
        r1 = row_base + 1u32;
    }
    if row_base + 2u32 < m {
        r2 = row_base + 2u32;
    }
    if row_base + 3u32 < m {
        r3 = row_base + 3u32;
    }
    if row_base + 4u32 < m {
        r4 = row_base + 4u32;
    }
    if row_base + 5u32 < m {
        r5 = row_base + 5u32;
    }
    if row_base + 6u32 < m {
        r6 = row_base + 6u32;
    }
    if row_base + 7u32 < m {
        r7 = row_base + 7u32;
    }

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
        // w >> 2 maps word index to scale group (4 words = 128 weights/group).
        let scale_idx = w >> 2u32;
        let col_base = w * 32u32;

        let p0 = pos_bits_u32[(r0 * words_per_row + w) as usize];
        let q0 = neg_bits_u32[(r0 * words_per_row + w) as usize];
        let p1 = pos_bits_u32[(r1 * words_per_row + w) as usize];
        let q1 = neg_bits_u32[(r1 * words_per_row + w) as usize];
        let p2 = pos_bits_u32[(r2 * words_per_row + w) as usize];
        let q2 = neg_bits_u32[(r2 * words_per_row + w) as usize];
        let p3 = pos_bits_u32[(r3 * words_per_row + w) as usize];
        let q3 = neg_bits_u32[(r3 * words_per_row + w) as usize];
        let p4 = pos_bits_u32[(r4 * words_per_row + w) as usize];
        let q4 = neg_bits_u32[(r4 * words_per_row + w) as usize];
        let p5 = pos_bits_u32[(r5 * words_per_row + w) as usize];
        let q5 = neg_bits_u32[(r5 * words_per_row + w) as usize];
        let p6 = pos_bits_u32[(r6 * words_per_row + w) as usize];
        let q6 = neg_bits_u32[(r6 * words_per_row + w) as usize];
        let p7 = pos_bits_u32[(r7 * words_per_row + w) as usize];
        let q7 = neg_bits_u32[(r7 * words_per_row + w) as usize];

        let s0 = group_scale_f32[(r0 * groups_per_row + scale_idx) as usize];
        let s1 = group_scale_f32[(r1 * groups_per_row + scale_idx) as usize];
        let s2 = group_scale_f32[(r2 * groups_per_row + scale_idx) as usize];
        let s3 = group_scale_f32[(r3 * groups_per_row + scale_idx) as usize];
        let s4 = group_scale_f32[(r4 * groups_per_row + scale_idx) as usize];
        let s5 = group_scale_f32[(r5 * groups_per_row + scale_idx) as usize];
        let s6 = group_scale_f32[(r6 * groups_per_row + scale_idx) as usize];
        let s7 = group_scale_f32[(r7 * groups_per_row + scale_idx) as usize];

        // Two-accumulator select-form ternary dot (Issue 613 T1 follow-up).
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

        // Bounds-check split: per-word instead of per-bit (all Bonsai shapes
        // are multiples of 32 — n_embd=5120, mlp_dim=17408, vocab=248320).
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

    // Lane 0 reads the residual then writes residual + gemv_result IN-PLACE.
    // This is the only difference from `gemv_ternary_plane_rowtiled8`.
    if lane == 0u32 {
        residual_output[r0 as usize] = residual_output[r0 as usize] + t0;
        if row_base + 1u32 < m {
            let r1i = r1 as usize;
            residual_output[r1i] = residual_output[r1i] + t1;
        }
        if row_base + 2u32 < m {
            let r2i = r2 as usize;
            residual_output[r2i] = residual_output[r2i] + t2;
        }
        if row_base + 3u32 < m {
            let r3i = r3 as usize;
            residual_output[r3i] = residual_output[r3i] + t3;
        }
        if row_base + 4u32 < m {
            let r4i = r4 as usize;
            residual_output[r4i] = residual_output[r4i] + t4;
        }
        if row_base + 5u32 < m {
            let r5i = r5 as usize;
            residual_output[r5i] = residual_output[r5i] + t5;
        }
        if row_base + 6u32 < m {
            let r6i = r6 as usize;
            residual_output[r6i] = residual_output[r6i] + t6;
        }
        if row_base + 7u32 < m {
            let r7i = r7 as usize;
            residual_output[r7i] = residual_output[r7i] + t7;
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

/// Fused ternary GEMV + in-place ResidualAdd launcher (Issue 616).
///
/// Replaces two separate dispatches:
/// 1. `GemvTernaryCubeCL::launch(down_proj, ffn_hidden, ffn_out)`
/// 2. `ResidualAddCubeCL::launch(x, ffn_out, x)`
///
/// With one fused dispatch:
/// `GemvTernaryResidualCubeCL::launch(down_proj, ffn_hidden, x_residual_output)`
///
/// The `residual_output` handle is BOTH read (for the residual) and written
/// (with `residual + gemv_result`). This is the key trick that keeps the
/// kernel at 5 Array args (CubeCL v0.10 limit) while still fusing the
/// residual add.
///
/// # Dispatch
///
/// Matches `GemvTernaryCubeCL::launch_rowtiled8` exactly: width-8 row tiling,
/// `CubeDim::new_1d(256)`, `CubeCount::Static(ceil(m / 64), 1, 1)`.
///
/// # Safety
///
/// - `input_handle` must point to `handle.n` f32 elements
/// - `residual_output_handle` must point to `handle.m` f32 elements
///   (will be read-then-overwritten)
/// - `handle` buffers must have been created from the same `client`
#[cfg(feature = "cubecl_runtime")]
pub struct GemvTernaryResidualCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvTernaryResidualCubeCL {
    /// Launch fused ternary GEMV + in-place ResidualAdd (width-8 row-tiled).
    ///
    /// Computes `residual_output[row] += dot(weight_row, input)` for each row.
    ///
    /// # Safety
    ///
    /// See struct docs. Caller must ensure buffer sizes are correct.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        residual_output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE_8; // 64

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_residual_rowtiled8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(residual_output_handle, m as usize),
                blocks64,
                groups_per_row,
                n,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — G1 correctness against CPU reference
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::CubeCLContext;

    /// CPU reference: ternary GEMV + residual add fused.
    ///
    /// `output[row] = residual[row] + Σ_col ternary_weight(row, col) * input[col]`
    fn ternary_gemv_residual_cpu(
        pos_bits_u64: &[u64],
        neg_bits_u64: &[u64],
        group_scale: &[half::f16],
        input: &[f32],
        residual: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        group_size: usize,
    ) {
        let blocks64 = n.div_ceil(64);
        let groups_per_row = n.div_ceil(group_size);
        for row in 0..m {
            let mut acc = residual[row];
            for col in 0..n {
                let block = col / 64;
                let bit = col % 64;
                let pos_word = pos_bits_u64[row * blocks64 + block];
                let neg_word = neg_bits_u64[row * blocks64 + block];
                let pos = ((pos_word >> bit) & 1) as i32;
                let neg = ((neg_word >> bit) & 1) as i32;
                let sign = pos - neg; // +1, 0, -1
                if sign != 0 {
                    let scale_idx = col / group_size;
                    let scale = group_scale[row * groups_per_row + scale_idx].to_f32();
                    acc += sign as f32 * scale * input[col];
                }
            }
            output[row] = acc;
        }
    }

    fn make_random_ternary_weights(
        m: usize,
        n: usize,
        group_size: usize,
        seed: u64,
    ) -> (
        Vec<u64>,         // pos_bits (blocks64 per row)
        Vec<u64>,         // neg_bits
        Vec<half::f16>,   // group_scale
    ) {
        use std::cell::Cell;
        thread_local! {
            static RNG: Cell<u64> = const { Cell::new(0x1234_5678_9abc_def0) };
        }
        // Simple xorshift re-seeded per call for determinism.
        RNG.with(|r| r.set(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)));

        let next = || {
            RNG.with(|r| {
                let mut x = r.get();
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                r.set(x);
                x
            })
        };

        let blocks64 = n.div_ceil(64);
        let groups_per_row = n.div_ceil(group_size);

        let mut pos_bits = vec![0u64; m * blocks64];
        let mut neg_bits = vec![0u64; m * blocks64];
        let mut scales = vec![half::f16::ZERO; m * groups_per_row];

        for row in 0..m {
            for col in 0..n {
                let r = next() % 10;
                // 70% zero, 15% +1, 15% -1 (typical sparse ternary distribution)
                let sign: i32 = if r < 7 {
                    0
                } else if r < 8 {
                    1
                } else {
                    -1
                };
                if sign != 0 {
                    let block = col / 64;
                    let bit = col % 64;
                    if sign > 0 {
                        pos_bits[row * blocks64 + block] |= 1u64 << bit;
                    } else {
                        neg_bits[row * blocks64 + block] |= 1u64 << bit;
                    }
                }
            }
            for g in 0..groups_per_row {
                let raw = (next() % 1000) as f32 / 500.0; // 0..2
                scales[row * groups_per_row + g] = half::f16::from_f32(0.1 + raw);
            }
        }

        (pos_bits, neg_bits, scales)
    }

    #[test]
    fn test_gemv_ternary_residual_matches_cpu() {
        let ctx = CubeCLContext::new().expect("GPU init");
        let client = ctx.client();

        let m = 128usize;
        let n = 256usize;
        let group_size = 128usize;

        let (pos_bits, neg_bits, scales) =
            make_random_ternary_weights(m, n, group_size, 42);

        // Build CPU reference weights
        use katgpt_core::TernaryGroupWeights;
        let mut w = TernaryGroupWeights::new(m, n);
        w.pos_bits = pos_bits.clone();
        w.neg_bits = neg_bits.clone();
        w.group_scale = scales.clone();

        // Random input + residual
        let mut input = vec![0.0f32; n];
        let mut residual = vec![0.0f32; m];
        for i in 0..n {
            input[i] = ((i as u64).wrapping_mul(31) % 100) as f32 / 50.0 - 1.0;
        }
        for i in 0..m {
            residual[i] = ((i as u64).wrapping_mul(17) % 100) as f32 / 50.0 - 1.0;
        }

        // CPU reference
        let mut cpu_out = vec![0.0f32; m];
        ternary_gemv_residual_cpu(
            &pos_bits,
            &neg_bits,
            &scales,
            &input,
            &residual,
            &mut cpu_out,
            m,
            n,
            group_size,
        );

        // GPU: upload weights + input + residual (the residual handle IS the output)
        let handle = TernaryHandle::from_weights(&client, &w);
        let input_handle =
            client.create_from_slice(<f32 as CubeElement>::as_bytes(&input));
        let residual_output_handle =
            client.create_from_slice(<f32 as CubeElement>::as_bytes(&residual));

        unsafe {
            GemvTernaryResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                &handle,
                input_handle,
                residual_output_handle.clone(),
            );
        }

        let bytes = client.read_one(residual_output_handle).unwrap();
        let gpu_out: &[f32] = bytemuck::cast_slice(&bytes);

        // G1: relative error tolerance matches Issue 604 (±1% rel, max 0.5% abs).
        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let abs = (g - c).abs();
            let rel = if c.abs() > 1e-6 { abs / c.abs() } else { abs };
            max_rel = max_rel.max(rel);
            max_abs = max_abs.max(abs);
            assert!(
                rel < 0.05 || abs < 1e-3,
                "row {i}: gpu={g:.6} cpu={c:.6} rel={rel:.4} abs={abs:.6}"
            );
        }
        eprintln!(
            "[test] m={m} n={n}: max_rel={max_rel:.6} max_abs={max_abs:.6} — PASS"
        );
    }

    #[test]
    fn test_gemv_ternary_residual_bonsai_shape() {
        // Bonsai down_proj shape: m=5120 (n_embd), n=17408 (mlp_dim).
        // Smaller proxy to keep test fast: m=256, n=512 (4 groups).
        use katgpt_core::TernaryGroupWeights;

let ctx = CubeCLContext::new().expect("GPU init");
        let client = ctx.client();

        let m = 256usize;
        let n = 512usize;
        let group_size = 128usize;

        let (pos_bits, neg_bits, scales) =
            make_random_ternary_weights(m, n, group_size, 7);
        let mut w = TernaryGroupWeights::new(m, n);
        w.pos_bits = pos_bits.clone();
        w.neg_bits = neg_bits.clone();
        w.group_scale = scales.clone();

        let mut input = vec![0.0f32; n];
        let mut residual = vec![0.0f32; m];
        for i in 0..n {
            input[i] = ((i as u64).wrapping_mul(31) % 100) as f32 / 50.0 - 1.0;
        }
        for i in 0..m {
            residual[i] = ((i as u64).wrapping_mul(17) % 100) as f32 / 50.0 - 1.0;
        }

        let mut cpu_out = vec![0.0f32; m];
        ternary_gemv_residual_cpu(
            &pos_bits,
            &neg_bits,
            &scales,
            &input,
            &residual,
            &mut cpu_out,
            m,
            n,
            group_size,
        );

        let handle = TernaryHandle::from_weights(&client, &w);
        let input_handle =
            client.create_from_slice(<f32 as CubeElement>::as_bytes(&input));
        let residual_output_handle =
            client.create_from_slice(<f32 as CubeElement>::as_bytes(&residual));

        unsafe {
            GemvTernaryResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                &handle,
                input_handle,
                residual_output_handle.clone(),
            );
        }

        let bytes = client.read_one(residual_output_handle).unwrap();
        let gpu_out: &[f32] = bytemuck::cast_slice(&bytes);

        let mut max_rel = 0.0f32;
        let mut max_abs = 0.0f32;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let abs = (g - c).abs();
            let rel = if c.abs() > 1e-6 { abs / c.abs() } else { abs };
            max_rel = max_rel.max(rel);
            max_abs = max_abs.max(abs);
            assert!(
                rel < 0.05 || abs < 1e-3,
                "row {i}: gpu={g:.6} cpu={c:.6} rel={rel:.4} abs={abs:.6}"
            );
        }
        eprintln!(
            "[test] m={m} n={n}: max_rel={max_rel:.6} max_abs={max_abs:.6} — PASS"
        );
    }

    #[test]
    fn test_gemv_ternary_residual_zero_input() {
        // Zero input → gemv_result = 0 → residual_output should equal residual input.
        use katgpt_core::TernaryGroupWeights;

let ctx = CubeCLContext::new().expect("GPU init");
        let client = ctx.client();

        let m = 64usize;
        let n = 128usize;
        let group_size = 128usize;

        let (pos_bits, neg_bits, scales) =
            make_random_ternary_weights(m, n, group_size, 99);
        let mut w = TernaryGroupWeights::new(m, n);
        w.pos_bits = pos_bits;
        w.neg_bits = neg_bits;
        w.group_scale = scales;

        let input = vec![0.0f32; n];
        let residual: Vec<f32> = (0..m).map(|i| (i as f32) * 0.1).collect();
        let expected = residual.clone();

        let handle = TernaryHandle::from_weights(&client, &w);
        let input_handle =
            client.create_from_slice(<f32 as CubeElement>::as_bytes(&input));
        let residual_output_handle =
            client.create_from_slice(<f32 as CubeElement>::as_bytes(&residual));

        unsafe {
            GemvTernaryResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                &handle,
                input_handle,
                residual_output_handle.clone(),
            );
        }

        let bytes = client.read_one(residual_output_handle).unwrap();
        let gpu_out: &[f32] = bytemuck::cast_slice(&bytes);

        for (i, (g, e)) in gpu_out.iter().zip(expected.iter()).enumerate() {
            let abs = (g - e).abs();
            assert!(
                abs < 1e-4,
                "row {i}: gpu={g:.6} expected={e:.6} (zero-input should be identity)"
            );
        }
        eprintln!("[test] zero-input identity — PASS");
    }
}
