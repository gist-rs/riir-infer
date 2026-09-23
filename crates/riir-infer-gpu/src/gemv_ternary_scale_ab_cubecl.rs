//! CubeCL ternary GEMV scale-format A/B variants (Issue 764 T2, M3 lane).
//!
//! Three kernel variants of `gemv_ternary_plane_rowtiled8` (the Metal
//! `launch()` default, Issue 613 T1-follow-up) isolating the group-scale axis
//! of the FFN GEMV cost — the #1 GPU consumer at 49–50% of decode time
//! (Bench 763), running at ~245–260 GB/s ≈ 61–65% of the ~400 GB/s peak.
//!
//! | Variant | Scale format | Scale bytes | ALU vs baseline | Purpose |
//! |---|---|---|---|---|
//! | `f16scale` | `&[f16]` (raw GGUF bits) | −50% | +1 `cast_from` per 4-word group (negligible) | **The candidate** — pure byte cut, lossless |
//! | `wordpair` | f32 (unchanged) | same | 2 independent word chains per lane (ILP probe) | Load-width probe |
//! | `noscale` | none (skipped) | −100% | −8 multiplies/iteration | **Diagnostic only** — bounds the whole scale axis |
//!
//! # Why f16-scale is not refuted by the Issue 628 ceiling table
//!
//! The 628 table refuted every path that changed ALU structure along with the
//! bytes: trit (−17.6% bytes, +div/mod per weight) 0.500×, threadgroup LUTs
//! 0.42–0.54×, interleaved (same bytes, one stream) 1.088×. `f16scale` is the
//! one untested combination: **bytes change, per-weight ALU does not** — one
//! `cast_from` per scale load, amortized over 128 weights. The scales are
//! *already* `Vec<f16>` in `TernaryGroupWeights`; the f32 baseline pays a CPU
//! decode plus 2× upload bytes for no arithmetic benefit. Arithmetic: FFN
//! 17.1B params ÷ 128/group ≈ 134M groups; f32 scales = ~534 MB of the ~4.8 GB
//! per-token FFN sweep (~11%); f16 halves that (~5.6% of the sweep).
//!
//! `f16 → f32` conversion is exact, so **G1 is bit-identity**, not tolerance.
//!
//! # These variants are NOT wired into production dispatch
//!
//! They exist for the A/B harness (`bench_767_issue764_ffn_gemv_scale_ab`).
//! A winning variant gets wired into `TernaryHandle`/`launch()` by its own
//! GOAT-gated unit (Issue 764 T3); a losing one stays here as the
//! reproducible artifact (the Issue 628 precedent).

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Plane trait needed for plane_sum() resolution
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use half::f16 as half_f16;

use crate::gemv_ternary_cubecl::{TernaryHandle, TERNARY_ROWS_PER_PLANE_8};

// ── Variant 1: f16 scale upload (the candidate) ─────────────────────

/// `gemv_ternary_plane_rowtiled8` with the group scales kept as **f16 on the
/// GPU** (the raw GGUF storage) instead of pre-decoded f32.
///
/// Body mirrors the baseline exactly except the 8 per-iteration scale loads:
/// `group_scale_f16: &[half_f16]` + `f32::cast_from(...)`. The conversion is
/// exact, so output is **bit-identical** to the baseline kernel.
///
/// Byte effect: the scale buffer halves (4 B → 2 B per group). For the Bonsai
/// FFN that is ~534 MB → ~267 MB per token, ~5.6% of the FFN weight sweep.
/// If the kernel is bandwidth-bound this converts near-fully into time; if
/// the dequant ALU is the wall (the Issue 628 conclusion) it buys ~nothing.
/// That discrimination is the point of the A/B.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_rowtiled8_f16scale(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f16: &[half_f16],
    input: &[f32],
    output: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
) {
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

        // f16 loads + exact in-kernel decode — the ONLY delta vs the baseline.
        let s0 = f32::cast_from(group_scale_f16[(r0 * groups_per_row + scale_idx) as usize]);
        let s1 = f32::cast_from(group_scale_f16[(r1 * groups_per_row + scale_idx) as usize]);
        let s2 = f32::cast_from(group_scale_f16[(r2 * groups_per_row + scale_idx) as usize]);
        let s3 = f32::cast_from(group_scale_f16[(r3 * groups_per_row + scale_idx) as usize]);
        let s4 = f32::cast_from(group_scale_f16[(r4 * groups_per_row + scale_idx) as usize]);
        let s5 = f32::cast_from(group_scale_f16[(r5 * groups_per_row + scale_idx) as usize]);
        let s6 = f32::cast_from(group_scale_f16[(r6 * groups_per_row + scale_idx) as usize]);
        let s7 = f32::cast_from(group_scale_f16[(r7 * groups_per_row + scale_idx) as usize]);

        // Two-accumulator select-form ternary dot (identical to baseline).
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

        if col_base + 32u32 <= n {
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

    if lane == 0u32 {
        output[r0 as usize] = t0;
        if row_base + 1u32 < m {
            output[r1 as usize] = t1;
        }
        if row_base + 2u32 < m {
            output[r2 as usize] = t2;
        }
        if row_base + 3u32 < m {
            output[r3 as usize] = t3;
        }
        if row_base + 4u32 < m {
            output[r4 as usize] = t4;
        }
        if row_base + 5u32 < m {
            output[r5 as usize] = t5;
        }
        if row_base + 6u32 < m {
            output[r6 as usize] = t6;
        }
        if row_base + 7u32 < m {
            output[r7 as usize] = t7;
        }
    }
}

// ── Variant 3: no-scale diagnostic ───────────────────────────────────

/// `gemv_ternary_plane_rowtiled8` with the scale loads AND multiplies removed
/// entirely — **diagnostic only, numerically wrong by design** (skips the
/// group rescale). Bounds the whole scale axis: its timing delta vs baseline
/// = scale bytes + scale multiplies. Read together with `f16scale` (bytes
/// only) it separates the bandwidth share from the ALU share of the scale
/// cost. Never wire into production dispatch.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_rowtiled8_noscale(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    input: &[f32],
    output: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
) {
    let words_per_row = blocks64 * 2u32;
    let _ = groups_per_row; // unused — kept for launch-shape parity

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

        if col_base + 32u32 <= n {
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

        // Diagnostic: NO scale multiply — the output is numerically wrong by
        // design; only its TIMING is meaningful.
        acc0 += a0p - a0n;
        acc1 += a1p - a1n;
        acc2 += a2p - a2n;
        acc3 += a3p - a3n;
        acc4 += a4p - a4n;
        acc5 += a5p - a5n;
        acc6 += a6p - a6n;
        acc7 += a7p - a7n;

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

    if lane == 0u32 {
        output[r0 as usize] = t0;
        if row_base + 1u32 < m {
            output[r1 as usize] = t1;
        }
        if row_base + 2u32 < m {
            output[r2 as usize] = t2;
        }
        if row_base + 3u32 < m {
            output[r3 as usize] = t3;
        }
        if row_base + 4u32 < m {
            output[r4 as usize] = t4;
        }
        if row_base + 5u32 < m {
            output[r5 as usize] = t5;
        }
        if row_base + 6u32 < m {
            output[r6 as usize] = t6;
        }
        if row_base + 7u32 < m {
            output[r7 as usize] = t7;
        }
    }
}

// ── Variant 2: word-pair load width / ILP probe ──────────────────────

/// `gemv_ternary_plane_rowtiled8` processing TWO words per loop iteration
/// (`w` and `w + PLANE_DIM`), keeping each lane's per-half accumulation order
/// identical to the baseline (`w` first, then `w + PLANE_DIM`), so per-lane
/// accumulators — and therefore the `plane_sum` outputs — are **bit-identical**.
///
/// Hypothesis under test: two independent weight-load dependency chains per
/// lane improve memory-level parallelism / hide latency (the "load width"
/// lever from the 4090 campaign's vectorized-load arm, expressed within
/// CubeCL's scalar-access model). The per-half `if wi < words_per_row` guard
/// fires only on the ragged tail (per-lane trip counts differ by at most one
/// pair); steady-state iterations are uniform.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_rowtiled8_wordpair(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    input: &[f32],
    output: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
) {
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
        #[unroll]
        for half in 0u32..2u32 {
            let wi = w + half * PLANE_DIM;
            if wi < words_per_row {
                let scale_idx = wi >> 2u32;
                let col_base = wi * 32u32;

                let p0 = pos_bits_u32[(r0 * words_per_row + wi) as usize];
                let q0 = neg_bits_u32[(r0 * words_per_row + wi) as usize];
                let p1 = pos_bits_u32[(r1 * words_per_row + wi) as usize];
                let q1 = neg_bits_u32[(r1 * words_per_row + wi) as usize];
                let p2 = pos_bits_u32[(r2 * words_per_row + wi) as usize];
                let q2 = neg_bits_u32[(r2 * words_per_row + wi) as usize];
                let p3 = pos_bits_u32[(r3 * words_per_row + wi) as usize];
                let q3 = neg_bits_u32[(r3 * words_per_row + wi) as usize];
                let p4 = pos_bits_u32[(r4 * words_per_row + wi) as usize];
                let q4 = neg_bits_u32[(r4 * words_per_row + wi) as usize];
                let p5 = pos_bits_u32[(r5 * words_per_row + wi) as usize];
                let q5 = neg_bits_u32[(r5 * words_per_row + wi) as usize];
                let p6 = pos_bits_u32[(r6 * words_per_row + wi) as usize];
                let q6 = neg_bits_u32[(r6 * words_per_row + wi) as usize];
                let p7 = pos_bits_u32[(r7 * words_per_row + wi) as usize];
                let q7 = neg_bits_u32[(r7 * words_per_row + wi) as usize];

                let s0 = group_scale_f32[(r0 * groups_per_row + scale_idx) as usize];
                let s1 = group_scale_f32[(r1 * groups_per_row + scale_idx) as usize];
                let s2 = group_scale_f32[(r2 * groups_per_row + scale_idx) as usize];
                let s3 = group_scale_f32[(r3 * groups_per_row + scale_idx) as usize];
                let s4 = group_scale_f32[(r4 * groups_per_row + scale_idx) as usize];
                let s5 = group_scale_f32[(r5 * groups_per_row + scale_idx) as usize];
                let s6 = group_scale_f32[(r6 * groups_per_row + scale_idx) as usize];
                let s7 = group_scale_f32[(r7 * groups_per_row + scale_idx) as usize];

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

                if col_base + 32u32 <= n {
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
            }
        }
        w += 2u32 * PLANE_DIM;
    }

    let t0 = plane_sum(acc0);
    let t1 = plane_sum(acc1);
    let t2 = plane_sum(acc2);
    let t3 = plane_sum(acc3);
    let t4 = plane_sum(acc4);
    let t5 = plane_sum(acc5);
    let t6 = plane_sum(acc6);
    let t7 = plane_sum(acc7);

    if lane == 0u32 {
        output[r0 as usize] = t0;
        if row_base + 1u32 < m {
            output[r1 as usize] = t1;
        }
        if row_base + 2u32 < m {
            output[r2 as usize] = t2;
        }
        if row_base + 3u32 < m {
            output[r3 as usize] = t3;
        }
        if row_base + 4u32 < m {
            output[r4 as usize] = t4;
        }
        if row_base + 5u32 < m {
            output[r5 as usize] = t5;
        }
        if row_base + 6u32 < m {
            output[r6 as usize] = t6;
        }
        if row_base + 7u32 < m {
            output[r7 as usize] = t7;
        }
    }
}

// ── Launchers ────────────────────────────────────────────────────────

/// Launchers for the Issue 764 T2 scale-A/B kernel variants.
///
/// Dispatch geometry is IDENTICAL to `GemvTernaryCubeCL::launch_rowtiled8`
/// (CubeDim 256, `ceil(m/64)` workgroups) — the variants differ only inside
/// the kernel body, so timing deltas attribute to the scale/loop axis alone.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvTernaryScaleAbCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvTernaryScaleAbCubeCL {
    /// Upload the group scales AS f16 (the raw GGUF storage — no CPU decode).
    ///
    /// `TernaryGroupWeights::group_scale` is already `Vec<half::f16>`; the
    /// baseline `TernaryHandle::from_weights` decodes it to f32 on the CPU
    /// and uploads 2× the bytes. This is the raw-bit upload for the f16scale
    /// arm.
    pub fn upload_group_scale_f16(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        handle: &TernaryHandle,
        group_scale: &[half_f16],
    ) -> Handle {
        assert_eq!(
            group_scale.len(),
            handle.m * handle.groups_per_row,
            "f16 scale upload length mismatch"
        );
        client.create_from_slice(bytemuck::cast_slice::<half_f16, u8>(group_scale))
    }

    /// f16-scale arm. `scale_f16` comes from [`Self::upload_group_scale_f16`].
    ///
    /// # Safety
    /// Same requirements as `GemvTernaryCubeCL::launch_rowtiled8` — buffers
    /// sized exactly as the kernel indexes them.
    pub unsafe fn launch_f16scale<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        scale_f16: &Handle,
        input_handle: Handle,
        output_handle: Handle,
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
            gemv_ternary_plane_rowtiled8_f16scale::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(scale_f16.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(output_handle, m as usize),
                blocks64,
                groups_per_row,
                n,
                m,
            );
        }
    }

    /// Word-pair (load-width / ILP probe) arm.
    ///
    /// # Safety
    /// Same requirements as `GemvTernaryCubeCL::launch_rowtiled8`.
    pub unsafe fn launch_wordpair<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
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
            gemv_ternary_plane_rowtiled8_wordpair::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(output_handle, m as usize),
                blocks64,
                groups_per_row,
                n,
                m,
            );
        }
    }

    /// No-scale DIAGNOSTIC arm — numerically wrong by design (skips the group
    /// rescale); only its timing is meaningful. Never wire into production.
    ///
    /// # Safety
    /// Same requirements as `GemvTernaryCubeCL::launch_rowtiled8`.
    pub unsafe fn launch_noscale<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
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

        unsafe {
            gemv_ternary_plane_rowtiled8_noscale::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(output_handle, m as usize),
                blocks64,
                groups_per_row,
                n,
                m,
            );
        }
    }
}
