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

use crate::gemv_ternary_cubecl::{TERNARY_ROWS_PER_PLANE_8, TernaryHandle};

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

// ── Issue 1130: the concat-epilogue split arm ───────────────────────

/// `gemv_ternary_plane_rowtiled8_f16scale` with the GDN qkv|z Split2
/// folded into the write epilogue (Issue 1130): rows `[0, split_row)`
/// land in `out_a` (the conv1d's `qkv`), rows `[split_row, m)` land in
/// `out_b` (the z-gate's `z_buf`) at `row - split_row`.
///
/// The dot/sum math is VERBATIM the f16scale kernel — same word loop,
/// same per-row fold order, same `plane_sum` — so per-row values are
/// bit-identical by construction; ONLY the destination address changes
/// (the movement-only discipline of the Issue-1129 dense-pair rung).
/// This eliminates the `input_proj_out` staging round-trip AND the
/// Split2 dispatch: −1 launch per GDN layer (−32/token on the 27B) and
/// −64 KB/layer of redundant read+write. Dispatched only on the f16
/// lane (the shipping decode dispatch since Bench 771) via
/// `GemvTernaryCubeCL::launch_split`; the branch is plane-uniform and
/// hoisted to the write epilogue — zero cost inside the dot loop.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_rowtiled8_f16scale_split(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f16: &[half_f16],
    input: &[f32],
    out_a: &mut [f32],
    out_b: &mut [f32],
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
    split_row: u32,
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
        if r0 < split_row {
            out_a[r0 as usize] = t0;
        } else {
            out_b[(r0 - split_row) as usize] = t0;
        }
        if row_base + 1u32 < m {
            if r1 < split_row {
                out_a[r1 as usize] = t1;
            } else {
                out_b[(r1 - split_row) as usize] = t1;
            }
        }
        if row_base + 2u32 < m {
            if r2 < split_row {
                out_a[r2 as usize] = t2;
            } else {
                out_b[(r2 - split_row) as usize] = t2;
            }
        }
        if row_base + 3u32 < m {
            if r3 < split_row {
                out_a[r3 as usize] = t3;
            } else {
                out_b[(r3 - split_row) as usize] = t3;
            }
        }
        if row_base + 4u32 < m {
            if r4 < split_row {
                out_a[r4 as usize] = t4;
            } else {
                out_b[(r4 - split_row) as usize] = t4;
            }
        }
        if row_base + 5u32 < m {
            if r5 < split_row {
                out_a[r5 as usize] = t5;
            } else {
                out_b[(r5 - split_row) as usize] = t5;
            }
        }
        if row_base + 6u32 < m {
            if r6 < split_row {
                out_a[r6 as usize] = t6;
            } else {
                out_b[(r6 - split_row) as usize] = t6;
            }
        }
        if row_base + 7u32 < m {
            if r7 < split_row {
                out_a[r7 as usize] = t7;
            } else {
                out_b[(r7 - split_row) as usize] = t7;
            }
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
        unsafe {
            Self::launch_f16scale_wg::<R>(
                client,
                handle,
                scale_f16,
                input_handle,
                output_handle,
                256,
            );
        }
    }

    /// [`Self::launch_f16scale`] with an explicit workgroup size — the
    /// Issue-1114 occupancy lever for the f16 lane (the shipping decode path
    /// dispatches THIS kernel, default-ON since Bench 771). The f16 kernel is
    /// plane-global and workgroup-size-agnostic exactly like its plain
    /// sibling (`plane_id = ABSOLUTE_POS_X / PLANE_DIM`, `row_base =
    /// plane_id · 8`, `plane_sum` reduction) — the wg remap is dispatch-only
    /// and byte-identical. Same `wg_size` contract as
    /// `GemvTernaryCubeCL::launch_rowtiled8_wg`.
    ///
    /// # Safety
    /// Same requirements as [`Self::launch_f16scale`].
    pub unsafe fn launch_f16scale_wg<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        scale_f16: &Handle,
        input_handle: Handle,
        output_handle: Handle,
        wg_size: u32,
    ) {
        assert!(
            wg_size >= 32 && wg_size.is_multiple_of(32),
            "wg_size must be a positive multiple of the 32-lane plane"
        );
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size;
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE_8;

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

    /// Issue 1130 — [`Self::launch_f16scale_wg`] with the qkv|z Split2 folded
    /// into the write epilogue: rows `[0, split_row)` land in `out_a_handle`
    /// (`qkv`), rows `[split_row, handle.m)` land in `out_b_handle` (`z_buf`)
    /// at `row - split_row`. Same geometry law as the wg remap: planes are
    /// addressed globally and the per-row math is verbatim, so this is
    /// BYTE-IDENTICAL per row at every wg size — only the destination
    /// address differs from the staging+Split2 chain.
    ///
    /// # Safety
    /// Same requirements as [`Self::launch_f16scale_wg`], with
    /// `out_a_handle` holding ≥ `split_row` and `out_b_handle` holding
    /// ≥ `handle.m - split_row` f32 elements; `split_row <= handle.m`
    /// (asserted). The caller must ensure no other queue operation aliases
    /// them while the kernel is in flight.
    pub unsafe fn launch_f16scale_split_wg<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        scale_f16: &Handle,
        input_handle: Handle,
        out_a_handle: Handle,
        out_b_handle: Handle,
        split_row: usize,
        wg_size: u32,
    ) {
        assert!(
            wg_size >= 32 && wg_size.is_multiple_of(32),
            "wg_size must be a positive multiple of the 32-lane plane"
        );
        assert!(
            split_row <= handle.m,
            "split_row {split_row} exceeds handle m {}",
            handle.m
        );
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size;
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE_8;

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;
        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_plane_rowtiled8_f16scale_split::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.pos_bits_u32.clone(), pos_len),
                BufferArg::from_raw_parts(handle.neg_bits_u32.clone(), neg_len),
                BufferArg::from_raw_parts(scale_f16.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(out_a_handle, split_row),
                BufferArg::from_raw_parts(out_b_handle, handle.m - split_row),
                blocks64,
                groups_per_row,
                n,
                m,
                split_row as u32,
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

#[cfg(test)]
mod concat_split_tests {
    use super::*;
    use crate::context::GpuContext;
    use crate::cubecl_runtime::ActiveRuntime;
    use crate::elementwise_cubecl::Split2CubeCL;
    use crate::gemv_ternary_cubecl::{
        GemvTernaryCubeCL, concat_split_launch_count, set_concat_split_fused,
    };

    /// Deterministic xorshift64 fills — no rand dep, reproducible across
    /// runs so G1 failures are diffable (the dense-pair test's helper shape).
    fn pseudo_bits(len: usize, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect()
    }

    fn pseudo_scales(len: usize, seed: u64) -> Vec<half_f16> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                half_f16::from_f32(0.5 + ((state % 1000) as f32) / 1000.0)
            })
            .collect()
    }

    fn pseudo_f32(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 1000.0
            })
            .collect()
    }

    fn read_f32(client: &ComputeClient<ActiveRuntime>, h: &Handle) -> Vec<f32> {
        let bytes = client.read_one(h.clone()).expect("read_one");
        f32::from_bytes(&bytes).to_vec()
    }

    /// G1 (Issue 1130): the split-fused launch is BIT-IDENTICAL to the
    /// incumbent chain (f16-scale GEMV → staging → Split2CubeCL) — same
    /// per-row values, same reduction order; only the destination address
    /// changes. Real folded shape (16384 = 10240 + 6144 rows × 5120), a
    /// boundary-spanning small shape (a plane straddles the split), and a
    /// doubly-ragged shape (n not a multiple of 32 → the slow path, m not a
    /// multiple of 8), at every shipping wg size {256, 128, 64} plus the
    /// production entry (launch_split, occupancy wg rule). Asserts f32
    /// to_bits equality, never tolerance; ×3 repeats to defeat buffer-pool
    /// recycling (the Bench-606 sentinel lesson); NaN-sentinel pre-fill
    /// asserts FULL coverage of both consumer buffers.
    #[test]
    fn test_concat_split_bit_identity() {
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        // (m, n, split_row): the real folded qkv|z geometry (conv_dim 10240 +
        // z_dim 6144 = m 16384, n 5120) + two ragged shapes.
        let shapes: &[(usize, usize, usize)] = &[
            (16384, 5120, 10240),
            (24, 128, 15),
            (13, 100, 5),
        ];

        // Arm the production entry for launch_split (the env is opt-in; the
        // setter is the harness override). Restored to OFF at the end — the
        // shipping default.
        set_concat_split_fused(true);

        for &(m, n, split_row) in shapes {
            let blocks64 = n.div_ceil(64);
            let groups_per_row = n.div_ceil(128);
            let pos = pseudo_bits(m * blocks64, 0x9E3779B97F4A7C15);
            let neg = pseudo_bits(m * blocks64, 0xBF58476D1CE4E5B9);
            let scales = pseudo_scales(m * groups_per_row, 0x94D049BB133111EB);
            let input = pseudo_f32(n, 0xA24BAED4963EE407);

            let handle = TernaryHandle::from_raw(&client, &pos, &neg, &scales, m, n);
            // The f16-scale lane is the shipping dispatch (DEFAULT ON since
            // Bench 771); the split arm rides it. Skip honestly if this
            // build ran with the f16 upload off.
            if handle.group_scale_f16.is_none() {
                eprintln!(
                    "Skipping ({m}×{n}): f16 scale buffers not uploaded (RIIR_GEMV_F16_SCALE=0 build)"
                );
                continue;
            }
            let x = client.create_from_slice(f32::as_bytes(&input));

            // Incumbent chain oracle: launch() → staging → Split2CubeCL —
            // the EXACT production chain (launch() picks the f16scale_wg
            // kernel at the occupancy wg for this m).
            let staging = client.empty(m * core::mem::size_of::<f32>());
            let ref_a = client.empty(split_row * core::mem::size_of::<f32>());
            let ref_b = client.empty((m - split_row) * core::mem::size_of::<f32>());
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &client,
                    &handle,
                    x.clone(),
                    staging.clone(),
                );
                Split2CubeCL::launch::<ActiveRuntime>(
                    &client,
                    staging.clone(),
                    ref_a.clone(),
                    ref_b.clone(),
                    split_row,
                    m - split_row,
                );
            }
            let ra = read_f32(&client, &ref_a);
            let rb = read_f32(&client, &ref_b);

            // wg sweep on the split kernel, then the production entry.
            for wg in [256u32, 128, 64] {
                for rep in 0..3 {
                    // NaN sentinel pre-fill — every row MUST be overwritten.
                    let out_a = client.create_from_slice(f32::as_bytes(
                        &vec![f32::NAN; split_row],
                    ));
                    let out_b = client.create_from_slice(f32::as_bytes(
                        &vec![f32::NAN; m - split_row],
                    ));
                    unsafe {
                        GemvTernaryScaleAbCubeCL::launch_f16scale_split_wg::<ActiveRuntime>(
                            &client,
                            &handle,
                            handle.group_scale_f16.as_ref().expect("checked above"),
                            x.clone(),
                            out_a.clone(),
                            out_b.clone(),
                            split_row,
                            wg,
                        );
                    }
                    let oa = read_f32(&client, &out_a);
                    let ob = read_f32(&client, &out_b);
                    assert_eq!(oa.len(), split_row, "({m}×{n}) out_a len");
                    assert_eq!(ob.len(), m - split_row, "({m}×{n}) out_b len");
                    for i in 0..split_row {
                        assert_eq!(
                            ra[i].to_bits(),
                            oa[i].to_bits(),
                            "({m}×{n}) wg {wg} rep {rep}: a-row {i} not bit-identical (ref {} vs split {})",
                            ra[i],
                            oa[i]
                        );
                    }
                    for i in 0..(m - split_row) {
                        assert_eq!(
                            rb[i].to_bits(),
                            ob[i].to_bits(),
                            "({m}×{n}) wg {wg} rep {rep}: b-row {i} not bit-identical (ref {} vs split {})",
                            rb[i],
                            ob[i]
                        );
                    }
                }
                println!("concat-split bit-identity ({m}×{n}, split {split_row}) wg {wg}: EXACT ×3 (both halves)");
            }

            // The production entry: launch_split at the occupancy wg rule +
            // the vacuous-guard counter.
            let counter_before = concat_split_launch_count();
            let out_a = client.create_from_slice(f32::as_bytes(&vec![f32::NAN; split_row]));
            let out_b = client.create_from_slice(f32::as_bytes(&vec![f32::NAN; m - split_row]));
            let fired = unsafe {
                GemvTernaryCubeCL::launch_split::<ActiveRuntime>(
                    &client,
                    &handle,
                    x.clone(),
                    out_a.clone(),
                    out_b.clone(),
                    split_row,
                )
            };
            assert!(fired, "({m}×{n}) launch_split must fire when armed + f16 present");
            assert_eq!(
                concat_split_launch_count(),
                counter_before + 1,
                "({m}×{n}) vacuous-guard counter must advance by exactly 1"
            );
            let oa = read_f32(&client, &out_a);
            let ob = read_f32(&client, &out_b);
            for i in 0..split_row {
                assert_eq!(ra[i].to_bits(), oa[i].to_bits(), "({m}×{n}) entry a-row {i}");
            }
            for i in 0..(m - split_row) {
                assert_eq!(rb[i].to_bits(), ob[i].to_bits(), "({m}×{n}) entry b-row {i}");
            }
            println!("concat-split launch_split entry ({m}×{n}, split {split_row}): EXACT + counter +1");
        }

        // The demote hatch: unarmed launch_split falls back (returns false)
        // WITHOUT touching the counter.
        set_concat_split_fused(false);
        let counter_before = concat_split_launch_count();
        let (m, n, split_row) = (24usize, 128usize, 15usize);
        let handle = TernaryHandle::from_raw(
            &client,
            &pseudo_bits(m * 2, 0x9E3779B97F4A7C15),
            &pseudo_bits(m * 2, 0xBF58476D1CE4E5B9),
            &pseudo_scales(m, 0x94D049BB133111EB),
            m,
            n,
        );
        let x = client.create_from_slice(f32::as_bytes(&pseudo_f32(n, 1)));
        let fired = unsafe {
            GemvTernaryCubeCL::launch_split::<ActiveRuntime>(
                &client,
                &handle,
                x.clone(),
                client.empty(split_row * core::mem::size_of::<f32>()),
                client.empty((m - split_row) * core::mem::size_of::<f32>()),
                split_row,
            )
        };
        assert!(!fired, "launch_split must NOT fire when unarmed");
        assert_eq!(
            concat_split_launch_count(),
            counter_before,
            "unarmed launch_split must not advance the counter"
        );
        println!("concat-split demote hatch: OFF → false, counter untouched");
    }
}
