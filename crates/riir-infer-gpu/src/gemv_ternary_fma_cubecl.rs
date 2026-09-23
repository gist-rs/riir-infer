//! Plan 563 / Bench 847 — B96 telescoped digit-FMA ternary GEMV (M3 lane).
//!
//! The Batch 96 corpus recipe `telescoped-digit-coefficients-byte-fma-dot`
//! (riir-clippy, Research 114 — PrismML/llama.cpp fork `mul_mv.metal`
//! L3089-3196, upstream ggml-org/llama.cpp PR 26980) cooked for OUR stack:
//! the shipping `gemv_ternary_plane_rowtiled8` pays 10 ALU ops per weight per
//! row (2 shift + 2 mask + 2 cmp + 2 select + 2 add); this kernel replaces the
//! per-bit extraction ladder with **four FMAs per packed byte** against
//! pre-combined input coefficients — zero shifts, zero masks, zero value
//! unpacking on the consume side.
//!
//! # Code format — TQ2_0 positional digit bytes (NOT the shipping planes)
//!
//! Our shipping format is sign-magnitude bit-planes (`pos_bits`/`neg_bits`).
//! The telescoped identity needs POSITIONAL digit codes, so
//! [`TernaryHandleFma::from_weights`] re-encodes at construction:
//! `digit = w + 1 ∈ {0,1,2}` (w = −1/0/+1), packed little-endian, 4 digits per
//! byte, 2 bytes per u32:
//!
//! ```text
//! u32 word = byte_a | (byte_b << 8)
//! byte_a   = d0 | d1<<2 | d2<<4 | d3<<6      (weights col+0..col+3)
//! byte_b   = d4 | d5<<2 | d6<<4 | d7<<6      (weights col+4..col+7)
//! ```
//!
//! Same bytes per weight as the planes (2 bits/weight, `words_per_row =
//! ceil(n/8)` u32 = the same total bytes as `blocks64·2` plane u32s), so the
//! A/B isolates ALU + issue slots, not bandwidth.
//!
//! # The identity (and its exactness + precision contracts)
//!
//! With `v` the byte as an exact integer-valued f32 (v ≤ 255 < 2²⁴) and
//! `f_k = floor(v · 4⁻ᵏ)` (== `v >> 2k`, BIT-EXACT: power-of-two scaling never
//! leaves the exact-integer f32 range), and digit weights `d_k` recovered via
//! `d_k = f_k − 4·f_{k+1}` (truncation, `f_4 = 0`), the dot telescopes:
//!
//! ```text
//! Σ_k d_k·y_k = v·y0 + f1·(y1 − 4·y0) + f2·(y2 − 4·y1) + f3·(y3 − 4·y2)
//! ```
//!
//! The input coefficients `c_k = y_k − 4·y_{k−1}` are computed ONCE per byte
//! and shared across all 8 rows of the plane; the codec's `−1` offset rides as
//! the accumulator seed: the per-group fold is `s_g · (T − Y)` where
//! `Y = Σ y` is the activation-slice mass — row-invariant, accumulated once per
//! lane per group (rule 2's input-side offset, composed with rule 1).
//!
//! **Precision contract** (honest, tolerance-gated — NEVER bit-identity): the
//! digit extraction is bit-exact, but the products reassociate AND the
//! telescoped terms carry intermediate magnitudes ~`v·|y|` (~255·|y|) that
//! cancel down to the ~`√128·|y|` true dot — a ~25× cancellation ratio. f32
//! keeps ~3 digits of headroom over it; measured G1 (Bench 847) pins max_rel
//! ≤ 1e-4 vs both the CPU reference and shipping rowtiled8. This is exactly
//! why the u16-width generalization (8 digits, v ≤ 65535, ~50× the
//! cancellation) was REFUTED at design time and the corpus rule's ≤-byte
//! eligibility respected.
//!
//! # Geometry — group-per-lane ownership
//!
//! Same as rowtiled8: 256-thread workgroup = 8 planes × 32 lanes, 8 rows per
//! plane, `ceil(m/64)` workgroups, `plane_sum` reduction, row clamping. The
//! ONE structural change: lanes own whole 128-weight GROUPS (`g = lane,
//! g += PLANE_DIM`) instead of strided words, so `Y` and the `c_k` stay
//! lane-local — zero cross-lane group reductions.
//!
//! `digit 3` (both plane bits set) is forbidden by the source format and never
//! emitted by [`pack_u32_digit_bytes`] (debug-asserted); if corrupt data
//! contained one, the oracles still decode it deterministically as `w = 2`.
//!
//! Toggle: `set_gemv_use_fma` / env `RIIR_GEMV_FMA` — **DEFAULT OFF**; the
//! production dispatch is NOT wired until the Bench 847 verdict passes the
//! ≥1.5% e2e promote-if gate (the Bench 768 bar).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Plane trait needed for plane_sum() resolution
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use katgpt_core::TernaryGroupWeights;

// ── Toggle (the f16-scale pattern, default OFF) ─────────────────────────────

static USE_FMA_GEMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FMA_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the byte-FMA path (the vacuous-guard counter —
/// accuracy gates compare OUTPUTS, so only this counter proves a toggle
/// actually reached the kernel).
static FMA_LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether the byte-FMA GEMV path is enabled. Env `RIIR_GEMV_FMA` is read
/// exactly once (first caller wins the OnceLock); the setter is authoritative
/// after. **DEFAULT OFF** until the Bench 847 promote-if gate.
pub fn fma_enabled() -> bool {
    if FMA_INITIALIZED
        .set(matches!(
            std::env::var("RIIR_GEMV_FMA")
                .unwrap_or_default()
                .to_lowercase()
                .as_str(),
            "1" | "on" | "true",
        ))
        .is_ok()
    {
        USE_FMA_GEMV.store(
            FMA_INITIALIZED.get().copied().unwrap_or(false),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_FMA_GEMV.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the byte-FMA GEMV path on/off (overrides the env var; the bench
/// harness's arm toggle).
pub fn set_gemv_use_fma(on: bool) {
    USE_FMA_GEMV.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Count of launches dispatched through the byte-FMA kernel (vacuous guard).
pub fn fma_launch_count() -> usize {
    FMA_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

// ── CPU repack: sign-magnitude planes → TQ2_0 positional digit bytes ───────

/// Re-encode [`TernaryGroupWeights`] bit-planes into the positional digit-byte
/// layout the telescoped kernel consumes.
///
/// `digit = w + 1 ∈ {0,1,2}`; weight `c` of a row lands at digit index
/// `c % 8` of u32 word `c / 8`, bit offset `8·(d/4) + 2·(d%4)` (byte A =
/// digits 0..3 at bits 0..7, byte B = digits 4..7 at bits 8..15; bits 16..31
/// zero — the kernel reads only the low two bytes per word).
///
/// Same total bytes as the plane layout: 8 weights per u32 = 2 bits/weight.
pub fn pack_u32_digit_bytes(w: &TernaryGroupWeights) -> Vec<u32> {
    let words_per_row = w.cols.div_ceil(8);
    let mut out = vec![0u32; w.rows * words_per_row];
    for r in 0..w.rows {
        let plane_base = r * w.blocks64;
        let row_base = r * words_per_row;
        for c in 0..w.cols {
            let blk = c / 64;
            let bit = c % 64;
            let pos = (w.pos_bits[plane_base + blk] >> bit) & 1;
            let neg = (w.neg_bits[plane_base + blk] >> bit) & 1;
            debug_assert!(
                !(pos == 1 && neg == 1),
                "both plane bits set is forbidden by the Q2_0_g128 format"
            );
            let digit = (1u64 + pos - neg) as u32; // {0,1,2}; never 3 on valid input
            let d = c % 8;
            let shift = 8 * (d / 4) + 2 * (d % 4);
            out[row_base + c / 8] |= digit << shift;
        }
    }
    out
}

// ── Handle ──────────────────────────────────────────────────────────────────

/// Paired GPU buffers for one telescoped digit-FMA ternary projection
/// (Plan 563). The GEMV twin of `riir_gpu::TernaryHandle` with the weight
/// payload re-encoded to positional digit bytes.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct TernaryHandleFma {
    /// Packed digit bytes as `Array<u32>`. Layout: `[rows * words_per_row]`
    /// u32 elements, 8 weights per word (2 bits/weight — same bytes as the
    /// plane layout).
    pub weights_u32: Handle,
    /// Pre-decoded group scales as `Array<f32>` (f16→f32 on CPU).
    /// Layout: `[rows * groups_per_row]` f32 elements.
    pub group_scale_f32: Handle,
    /// Output dimension (number of rows).
    pub m: usize,
    /// Input dimension (number of columns).
    pub n: usize,
    /// u32 words per row (= `n.div_ceil(8)`; 8 weights per word).
    pub words_per_row: usize,
    /// Groups per row (= `n.div_ceil(128)`).
    pub groups_per_row: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryHandleFma {
    /// Build from [`TernaryGroupWeights`] (the loader-side shape): repack the
    /// planes into digit bytes + upload two buffers.
    pub fn from_weights(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w: &TernaryGroupWeights,
    ) -> Self {
        let packed = pack_u32_digit_bytes(w);
        let scale_f32: Vec<f32> = w.group_scale.iter().map(|s| s.to_f32()).collect();
        let weights_u32 = client.create_from_slice(bytemuck::cast_slice(&packed));
        let group_scale_f32 = client.create_from_slice(f32::as_bytes(&scale_f32));
        Self {
            weights_u32,
            group_scale_f32,
            m: w.rows,
            n: w.cols,
            words_per_row: w.cols.div_ceil(8),
            groups_per_row: w.groups_per_row,
        }
    }
}

// ── Kernel ──────────────────────────────────────────────────────────────────

/// Output rows processed per plane — MUST match the hand-unrolled accumulator
/// count (the Bench-606 sentinel-coverage lesson: a mismatch here is invisible
/// to a relative-error check).
#[cfg(feature = "cubecl_runtime")]
pub(crate) const FMA_ROWS_PER_PLANE: u32 = 8;

/// Row-tiled telescoped digit-FMA ternary GEMV at tile width 8 (Plan 563).
///
/// Mirrors `riir_gpu`'s `gemv_ternary_plane_rowtiled8` geometry; the inner
/// loop is the B96 recipe — see the module docs for the identity, the
/// exactness/precision contracts, and the group-per-lane ownership rationale.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_fma_rowtiled8(
    weights_u32: &[u32],
    group_scale_f32: &[f32],
    input: &[f32],
    output: &mut [f32],
    words_per_row: u32,
    groups_per_row: u32,
    n: u32,
    m: u32,
) {
    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let row_base = plane_id * FMA_ROWS_PER_PLANE;

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

    // Digit-oracle + telescoping constants (f32::new — the sibling-kernel
    // convention; all exact powers of two).
    let c_four = f32::new(4.0f32);
    let c_quarter = f32::new(0.25f32);
    let c_1_16 = f32::new(0.0625f32);
    let c_1_64 = f32::new(0.015625f32);

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);
    let mut acc4 = f32::new(0.0f32);
    let mut acc5 = f32::new(0.0f32);
    let mut acc6 = f32::new(0.0f32);
    let mut acc7 = f32::new(0.0f32);

    // Group-per-lane ownership: the per-group Y mass and the c_k coefficients
    // are lane-local — no cross-lane group reduction anywhere.
    let mut g = lane;
    while g < groups_per_row {
        let col_base = g * 128u32;
        let fast = col_base + 128u32 <= n;

        let mut ysum = f32::new(0.0f32);
        let mut t0 = f32::new(0.0f32);
        let mut t1 = f32::new(0.0f32);
        let mut t2 = f32::new(0.0f32);
        let mut t3 = f32::new(0.0f32);
        let mut t4 = f32::new(0.0f32);
        let mut t5 = f32::new(0.0f32);
        let mut t6 = f32::new(0.0f32);
        let mut t7 = f32::new(0.0f32);

        if fast {
            // Fast path: the whole group is in bounds (all Bonsai shapes:
            // n ∈ {5120, 17408, 248320} are multiples of 128). 16 words per
            // group, each word = 2 telescoped bytes.
            #[unroll]
            for lw in 0u32..16u32 {
                let wi = g * 16u32 + lw;
                let col0 = col_base + lw * 8u32;

                let w0 = weights_u32[(r0 * words_per_row + wi) as usize];
                let w1 = weights_u32[(r1 * words_per_row + wi) as usize];
                let w2 = weights_u32[(r2 * words_per_row + wi) as usize];
                let w3 = weights_u32[(r3 * words_per_row + wi) as usize];
                let w4 = weights_u32[(r4 * words_per_row + wi) as usize];
                let w5 = weights_u32[(r5 * words_per_row + wi) as usize];
                let w6 = weights_u32[(r6 * words_per_row + wi) as usize];
                let w7 = weights_u32[(r7 * words_per_row + wi) as usize];

                // Byte A: digits 0..3 ↔ weights col0+0..col0+3.
                let y0 = input[(col0) as usize];
                let y1 = input[(col0 + 1u32) as usize];
                let y2 = input[(col0 + 2u32) as usize];
                let y3 = input[(col0 + 3u32) as usize];
                // Byte B: digits 4..7 ↔ weights col0+4..col0+7.
                let y4 = input[(col0 + 4u32) as usize];
                let y5 = input[(col0 + 5u32) as usize];
                let y6 = input[(col0 + 6u32) as usize];
                let y7 = input[(col0 + 7u32) as usize];

                // Row-invariant: the offset mass + telescoped coefficients,
                // shared across all 8 rows.
                ysum += (y0 + y1 + y2 + y3) + (y4 + y5 + y6 + y7);
                let c1 = y1 - c_four * y0;
                let c2 = y2 - c_four * y1;
                let c3 = y3 - c_four * y2;
                let c5 = y5 - c_four * y4;
                let c6 = y6 - c_four * y5;
                let c7 = y7 - c_four * y6;

                // Per row: v = the byte as an exact integer f32; the four
                // digit oracles are bit-exact; four FMAs per byte. This block
                // is the whole optimization — no shifts, no masks, no selects.
                let va0 = f32::cast_from(w0 & 0xFFu32);
                t0 += va0 * y0
                    + (va0 * c_quarter).floor() * c1
                    + (va0 * c_1_16).floor() * c2
                    + (va0 * c_1_64).floor() * c3;
                let vb0 = f32::cast_from((w0 >> 8u32) & 0xFFu32);
                t0 += vb0 * y4
                    + (vb0 * c_quarter).floor() * c5
                    + (vb0 * c_1_16).floor() * c6
                    + (vb0 * c_1_64).floor() * c7;

                let va1 = f32::cast_from(w1 & 0xFFu32);
                t1 += va1 * y0
                    + (va1 * c_quarter).floor() * c1
                    + (va1 * c_1_16).floor() * c2
                    + (va1 * c_1_64).floor() * c3;
                let vb1 = f32::cast_from((w1 >> 8u32) & 0xFFu32);
                t1 += vb1 * y4
                    + (vb1 * c_quarter).floor() * c5
                    + (vb1 * c_1_16).floor() * c6
                    + (vb1 * c_1_64).floor() * c7;

                let va2 = f32::cast_from(w2 & 0xFFu32);
                t2 += va2 * y0
                    + (va2 * c_quarter).floor() * c1
                    + (va2 * c_1_16).floor() * c2
                    + (va2 * c_1_64).floor() * c3;
                let vb2 = f32::cast_from((w2 >> 8u32) & 0xFFu32);
                t2 += vb2 * y4
                    + (vb2 * c_quarter).floor() * c5
                    + (vb2 * c_1_16).floor() * c6
                    + (vb2 * c_1_64).floor() * c7;

                let va3 = f32::cast_from(w3 & 0xFFu32);
                t3 += va3 * y0
                    + (va3 * c_quarter).floor() * c1
                    + (va3 * c_1_16).floor() * c2
                    + (va3 * c_1_64).floor() * c3;
                let vb3 = f32::cast_from((w3 >> 8u32) & 0xFFu32);
                t3 += vb3 * y4
                    + (vb3 * c_quarter).floor() * c5
                    + (vb3 * c_1_16).floor() * c6
                    + (vb3 * c_1_64).floor() * c7;

                let va4 = f32::cast_from(w4 & 0xFFu32);
                t4 += va4 * y0
                    + (va4 * c_quarter).floor() * c1
                    + (va4 * c_1_16).floor() * c2
                    + (va4 * c_1_64).floor() * c3;
                let vb4 = f32::cast_from((w4 >> 8u32) & 0xFFu32);
                t4 += vb4 * y4
                    + (vb4 * c_quarter).floor() * c5
                    + (vb4 * c_1_16).floor() * c6
                    + (vb4 * c_1_64).floor() * c7;

                let va5 = f32::cast_from(w5 & 0xFFu32);
                t5 += va5 * y0
                    + (va5 * c_quarter).floor() * c1
                    + (va5 * c_1_16).floor() * c2
                    + (va5 * c_1_64).floor() * c3;
                let vb5 = f32::cast_from((w5 >> 8u32) & 0xFFu32);
                t5 += vb5 * y4
                    + (vb5 * c_quarter).floor() * c5
                    + (vb5 * c_1_16).floor() * c6
                    + (vb5 * c_1_64).floor() * c7;

                let va6 = f32::cast_from(w6 & 0xFFu32);
                t6 += va6 * y0
                    + (va6 * c_quarter).floor() * c1
                    + (va6 * c_1_16).floor() * c2
                    + (va6 * c_1_64).floor() * c3;
                let vb6 = f32::cast_from((w6 >> 8u32) & 0xFFu32);
                t6 += vb6 * y4
                    + (vb6 * c_quarter).floor() * c5
                    + (vb6 * c_1_16).floor() * c6
                    + (vb6 * c_1_64).floor() * c7;

                let va7 = f32::cast_from(w7 & 0xFFu32);
                t7 += va7 * y0
                    + (va7 * c_quarter).floor() * c1
                    + (va7 * c_1_16).floor() * c2
                    + (va7 * c_1_64).floor() * c3;
                let vb7 = f32::cast_from((w7 >> 8u32) & 0xFFu32);
                t7 += vb7 * y4
                    + (vb7 * c_quarter).floor() * c5
                    + (vb7 * c_1_16).floor() * c6
                    + (vb7 * c_1_64).floor() * c7;
            }
        } else {
            // Ragged tail group (only the LAST group can be ragged): scalar
            // guarded loop, correctness over speed. Accumulates Σ digit·y —
            // the SAME pre-offset value as the fast path (the −Y fold below
            // applies to both).
            #[unroll]
            for c in 0u32..128u32 {
                let col = col_base + c;
                if col < n {
                    let y = input[col as usize];
                    ysum += y;
                    let wi = col >> 3u32;
                    let d = col & 7u32;
                    let shift = 8u32 * (d >> 2u32) + 2u32 * (d & 3u32);
                    let dg0 = (weights_u32[(r0 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg1 = (weights_u32[(r1 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg2 = (weights_u32[(r2 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg3 = (weights_u32[(r3 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg4 = (weights_u32[(r4 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg5 = (weights_u32[(r5 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg6 = (weights_u32[(r6 * words_per_row + wi) as usize] >> shift) & 3u32;
                    let dg7 = (weights_u32[(r7 * words_per_row + wi) as usize] >> shift) & 3u32;
                    t0 += f32::cast_from(dg0) * y;
                    t1 += f32::cast_from(dg1) * y;
                    t2 += f32::cast_from(dg2) * y;
                    t3 += f32::cast_from(dg3) * y;
                    t4 += f32::cast_from(dg4) * y;
                    t5 += f32::cast_from(dg5) * y;
                    t6 += f32::cast_from(dg6) * y;
                    t7 += f32::cast_from(dg7) * y;
                }
            }
        }

        // Per-group scale fold: the codec's −1 offset rides as the shared −Y.
        let s0 = group_scale_f32[(r0 * groups_per_row + g) as usize];
        let s1 = group_scale_f32[(r1 * groups_per_row + g) as usize];
        let s2 = group_scale_f32[(r2 * groups_per_row + g) as usize];
        let s3 = group_scale_f32[(r3 * groups_per_row + g) as usize];
        let s4 = group_scale_f32[(r4 * groups_per_row + g) as usize];
        let s5 = group_scale_f32[(r5 * groups_per_row + g) as usize];
        let s6 = group_scale_f32[(r6 * groups_per_row + g) as usize];
        let s7 = group_scale_f32[(r7 * groups_per_row + g) as usize];

        acc0 += s0 * (t0 - ysum);
        acc1 += s1 * (t1 - ysum);
        acc2 += s2 * (t2 - ysum);
        acc3 += s3 * (t3 - ysum);
        acc4 += s4 * (t4 - ysum);
        acc5 += s5 * (t5 - ysum);
        acc6 += s6 * (t6 - ysum);
        acc7 += s7 * (t7 - ysum);

        g += PLANE_DIM;
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

// ── Launcher ────────────────────────────────────────────────────────────────

/// Launcher for the telescoped digit-FMA GEMV (Plan 563).
#[cfg(feature = "cubecl_runtime")]
pub struct GemvTernaryFmaCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvTernaryFmaCubeCL {
    /// Launch the byte-FMA kernel. Same dispatch geometry as
    /// `GemvTernaryCubeCL::launch_rowtiled8` (256-thread workgroups, 8 planes,
    /// 8 rows/plane, `ceil(m/64)` workgroups).
    ///
    /// # Safety
    ///
    /// Same contract as `launch_rowtiled8`: `input_handle` holds `n` f32,
    /// `output_handle` holds `m` f32, and both handles must outlive the
    /// dispatch (CubeCL pools buffers).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandleFma,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * FMA_ROWS_PER_PLANE; // 64

        let m = handle.m as u32;
        let n = handle.n as u32;
        let words_per_row = handle.words_per_row as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let w_len = handle.m * handle.words_per_row;
        let scale_len = handle.m * handle.groups_per_row;

        FMA_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        unsafe {
            gemv_ternary_fma_rowtiled8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weights_u32.clone(), w_len),
                BufferArg::from_raw_parts(handle.group_scale_f32.clone(), scale_len),
                BufferArg::from_raw_parts(input_handle, n as usize),
                BufferArg::from_raw_parts(output_handle, m as usize),
                words_per_row,
                groups_per_row,
                n,
                m,
            );
        }
    }
}
