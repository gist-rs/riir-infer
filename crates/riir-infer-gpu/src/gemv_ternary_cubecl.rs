//! CubeCL ternary bit-plane GEMV kernel with per-group f16 scale (Issue 599).
//!
//! Performs `output[M] = dequant_ternary(weight) @ input[N]` with inline
//! bit-plane sign extraction during the dot product — no intermediate f32
//! weight materialization. Mirrors [`crate::gemv_q4k_cubecl`] structurally.
//!
//! # Weight format — `Q2_0_g128` (TernaryGroupWeights)
//!
//! Each weight is one of `{-1, 0, +1}`, encoded across two bit-planes:
//! - `pos_bits`: bit set → weight is `+1` (contributes `+scale·x`)
//! - `neg_bits`: bit set → weight is `-1` (contributes `-scale·x`)
//! - neither set → weight is `0` (contributes nothing)
//! - (both set is forbidden by the format)
//!
//! Per-128-element group, an f16 `group_scale` rescales the ±1 magnitude:
//! `effective_weight = group_scale[g] · sign(col)`.
//!
//! # Buffer layout (zero-copy from `TernaryGroupWeights`)
//!
//! Three GPU buffers per projection:
//!
//! | Buffer | Type | Layout per row | Source |
//! |--------|------|----------------|--------|
//! | `pos_bits_u32` | `Array<u32>` | `blocks64 · 2` u32 | `w.pos_bits` (u64→2×u32) |
//! | `neg_bits_u32` | `Array<u32>` | `blocks64 · 2` u32 | `w.neg_bits` (u64→2×u32) |
//! | `group_scale_f32` | `Array<f32>` | `groups_per_row` f32 | `w.group_scale` (f16→f32 on CPU) |
//!
//! `u64` is cast to 2×`u32` because CubeCL `Array<u64>` backend support is
//! inconsistent; `u32` is universal. Bit `b` of u64 block `B` lives at
//! `pos_bits_u32[row*blocks64*2 + B*2 + (b/32)]`, bit `(b%32)`.
//!
//! # Dispatch
//!
//! Uses plane (subgroup) cooperative dot product with `plane_sum()` reduction.
//! Each plane handles one output row. Lanes cooperatively compute the dot
//! product, reading contiguous weight elements for coalesced access.
//!
//! | Variant | CubeDim | CubeCount | Rows/cube |
//! |---------|---------|-----------|-----------|
//! | Plane | `new_1d(256)` | `ceil(m/8)` | 8 (Metal subgroup=32) |

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Plane trait needed for plane_sum() resolution
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use katgpt_core::TernaryGroupWeights;

// ── Issue 764 T2: f16-scale dispatch toggle (Bench 767/768) ──────────────

/// Runtime toggle for the f16 group-scale GEMV path (Issue 764 T2).
///
/// `true` ⇒ `GemvTernaryCubeCL::launch()` dispatches the f16-scale kernel
/// variant (scales uploaded as raw f16, decoded in-kernel — bit-identical
/// output, Bench 767 G1) when the handle carries the f16 buffer (uploaded at
/// construction only while the toggle is on). **DEFAULT ON since Bench
/// 771** (the Bench 768 promotion): env `RIIR_GEMV_F16_SCALE` is the
/// KILL-SWITCH — explicit "0"/"off"/"false" restores the legacy f32 path
/// (the f16 buffers then never upload).
///
/// Bench 768 e2e GOAT: +1.7/+1.8/+3.1/+4.2% end-to-end decode across four
/// executions (G1 bit-identical logits, verdict-stable) — PASS at the
/// ≥1.5% promote-if gate. The 4090-side concern is retired: production
/// decode there is the cudarc dp4a path — which already ships its own f16
/// wscale bits (Issue 734 T5, bit-identical + faster) — so this CubeCL
/// toggle is M3's lane.
static USE_F16_SCALE_GEMV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
static F16_SCALE_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the f16-scale path (the vacuous-guard
/// counter — outputs are bit-identical to the f32 path, so ONLY this counter
/// proves a toggle actually reached the kernel).
static F16_SCALE_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn f16_scale_enabled() -> bool {
    // Env is read exactly once (the first caller wins the OnceLock); the
    // AtomicBool is the live value and the setter is authoritative after.
    // DEFAULT ON (Bench 771, promoting the Bench 768 GOAT) — the env is the
    // kill-switch: explicit "0"/"off"/"false" restores the legacy f32 path.
    if F16_SCALE_INITIALIZED
        .set(!matches!(
            std::env::var("RIIR_GEMV_F16_SCALE")
                .unwrap_or_default()
                .to_lowercase()
                .as_str(),
            "0" | "off" | "false",
        ))
        .is_ok()
    {
        USE_F16_SCALE_GEMV.store(
            F16_SCALE_INITIALIZED.get().copied().unwrap_or(true),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_F16_SCALE_GEMV.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the f16-scale GEMV path on/off (overrides the env var; the bench
/// harness's arm toggle — set BEFORE forward construction so the f16 scale
/// buffers upload).
pub fn set_gemv_use_f16_scale(on: bool) {
    let _ = F16_SCALE_INITIALIZED.set(on);
    USE_F16_SCALE_GEMV.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the f16-scale path so far.
pub fn gemv_f16_scale_launch_count() -> usize {
    F16_SCALE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Group size for ternary weights (matches katgpt-types::GROUP_SIZE).
pub(crate) const TERNARY_GROUP_SIZE: u32 = 128;

/// Bits per u64 block.
pub(crate) const BITS_PER_BLOCK: u32 = 64;

// ── CPU helpers ────────────────────────────────────────────────────

/// Cast `Vec<u64>` bit-planes to `Vec<u32>` (2× size) for GPU upload.
///
/// Each u64 becomes two u32 words in little-endian order: `[low32, high32]`.
/// The kernel reconstructs bit `b` of the original u64 as
/// `word[b/32] >> (b%32) & 1`.
pub fn cast_u64_to_u32(src: &[u64]) -> Vec<u32> {
    bytemuck::cast_slice::<u64, u32>(src).to_vec()
}

/// Pre-decode f16 group scales to f32 (avoids GPU-side f16 conversion).
///
/// Returns a flat Vec matching the CPU `group_scale` layout, just as f32.
pub fn prepare_group_scale_f32(group_scale: &[half::f16]) -> Vec<f32> {
    group_scale.iter().map(|s| s.to_f32()).collect()
}

/// Prepare a block-contiguous GPU buffer from SoA `TernaryGroupWeights`
/// (Issue 650 Phase 2 prep).
///
/// Each 128-weight group is packed into 9 × u32 = 36 bytes:
///
/// ```text
/// u32[0]     = f32 scale bits (pre-decoded from f16)
/// u32[1..5]  = pos bit-plane (4 × u32 = 128 bits)
/// u32[5..9]  = neg bit-plane (4 × u32 = 128 bits)
/// ```
///
/// The GPU kernel reads one block (9 u32) per group in a single global-memory
/// access, eliminating the 3× overhead of separate pos/neg/scale arrays.
/// This is the GPU-friendly counterpart to `TernaryBlockAoS` (34 bytes, CPU).
/// The 2-byte difference (34→36) is the u32 alignment cost — negligible
/// (1.5% overhead) and worth it for clean GPU addressing.
///
/// Returns a flat `Vec<u32>` of length `rows * groups_per_row * 9`.
#[allow(dead_code)]
pub fn prepare_block_contiguous_u32(w: &TernaryGroupWeights) -> Vec<u32> {
    // Each group covers GROUP_SIZE (128) elements = 2 blocks of 64.
    // This is a fixed constant regardless of cols — the last group may have
    // fewer valid elements (zero-padded beyond cols), but the buffer layout
    // requires exactly 2 blocks per group for the GPU kernel's fixed 9-u32
    // indexing. (Bug fix 2026-08-13: was `blocks64 / groups_per_row` which
    // truncated to 1 when cols wasn't a multiple of 128, e.g. cols=192:
    // blocks64=3, groups_per_row=2, 3/2=1 ≠ 2.)
    let blocks_per_group = (TERNARY_GROUP_SIZE / 64) as usize;
    let u32_per_group = blocks_per_group * 2; // u32 words per bit-plane per group
    let total_u32 = w.rows * w.groups_per_row * (1 + 2 * u32_per_group);
    let mut buf = Vec::with_capacity(total_u32);
    for r in 0..w.rows {
        let row_block_base = r * w.blocks64;
        let group_base = r * w.groups_per_row;
        for g in 0..w.groups_per_row {
            // Scale (f16 → f32 bits).
            let scale_f32 = w.group_scale[group_base + g].to_f32();
            buf.push(scale_f32.to_bits());

            // Pos bits for this group: u64 words → u32 words.
            let b_start = g * blocks_per_group;
            for i in 0..blocks_per_group {
                let b = b_start + i;
                let val = if b < w.blocks64 {
                    w.pos_bits[row_block_base + b]
                } else {
                    0
                };
                buf.push((val & 0xFFFF_FFFF) as u32);
                buf.push((val >> 32) as u32);
            }

            // Neg bits for this group.
            for i in 0..blocks_per_group {
                let b = b_start + i;
                let val = if b < w.blocks64 {
                    w.neg_bits[row_block_base + b]
                } else {
                    0
                };
                buf.push((val & 0xFFFF_FFFF) as u32);
                buf.push((val >> 32) as u32);
            }
        }
    }
    debug_assert_eq!(buf.len(), total_u32);
    buf.shrink_to_fit();
    buf
}

/// Number of u32 elements per block-contiguous group (9: 1 scale + 4 pos + 4 neg).
/// Used by the GPU kernel in Phase 2 (Issue 650).
#[allow(dead_code)]
pub const U32_PER_BLOCK_GROUP: usize = 9;

/// CPU reference: block-contiguous GEMM `output = weights @ inputs` (Issue 650).
///
/// Takes the GPU-format block-contiguous buffer (from `prepare_block_contiguous_u32`)
/// and computes a batched matvec. This mirrors exactly what the GPU kernel will
/// do: read one 9-u32 block per group, extract scale + pos + neg, dequant, and
/// accumulate the dot product.
///
/// This is the CPU validation reference for the GPU kernel. If the GPU kernel's
/// output matches this function's output (within f32 tolerance), the kernel's
/// block-contiguous indexing is correct.
///
/// - `block_buf`: output of `prepare_block_contiguous_u32(w)`
/// - `inputs`: `[p_tokens, n]` row-major
/// - `output`: `[p_tokens, m]` row-major
/// - `m`, `n`, `groups_per_row`: weight matrix dimensions
#[allow(dead_code)]
pub fn block_contiguous_gemm_cpu_ref(
    block_buf: &[u32],
    inputs: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    groups_per_row: usize,
    p_tokens: usize,
) {
    assert_eq!(inputs.len(), p_tokens * n, "inputs shape mismatch");
    assert_eq!(output.len(), p_tokens * m, "output shape mismatch");
    assert_eq!(
        block_buf.len(),
        m * groups_per_row * U32_PER_BLOCK_GROUP,
        "block buffer size mismatch"
    );

    for tok in 0..p_tokens {
        let x_base = tok * n;
        let out_base = tok * m;
        for row in 0..m {
            let mut row_sum = 0.0f32;
            for g in 0..groups_per_row {
                let blk_base = (row * groups_per_row + g) * U32_PER_BLOCK_GROUP;
                let scale = f32::from_bits(block_buf[blk_base]);
                let g_start = g * 128;
                let g_end = (g_start + 128).min(n);
                let mut group_acc = 0.0f32;
                for col in g_start..g_end {
                    let local = col - g_start;
                    let word_idx = blk_base + 1 + (local >> 5); // +1 skips scale
                    let bit_pos = local & 31;
                    let pos = (block_buf[word_idx] >> bit_pos) & 1;
                    let neg_word_idx = blk_base + 1 + 4 + (local >> 5); // +1+4 skips scale+pos
                    let neg = (block_buf[neg_word_idx] >> bit_pos) & 1;
                    let sign = pos as i32 - neg as i32;
                    group_acc += sign as f32 * inputs[x_base + col];
                }
                row_sum += scale * group_acc;
            }
            output[out_base + row] = row_sum;
        }
    }
}

// ── CubeCL ternary GEMV kernel (plane-cooperative) ─────────────────

/// CubeCL ternary bit-plane dequant+GEMV kernel using plane (subgroup)
/// cooperative dot product.
///
/// Each plane handles one output row. Lanes cooperatively compute the dot
/// product by iterating over groups (128 elements = 2 u64 blocks = 4 u32
/// words), extracting signs on-the-fly:
///
/// ```text
/// for each group g of 128 elements:
///   load scale = group_scale_f32[row, g]
///   for each element this lane handles (stride = PLANE_DIM, 4 per group):
///     block = col / 64
///     word  = pos_bits_u32[row*blocks64*2 + block*2 + (col%64)/32]
///     pos   = (word >> (col%32)) & 1
///     ... same for neg_bits ...
///     sign  = pos - neg   // +1, 0, -1
///     partial += (sign as f32) * scale * input[col]
/// reduce: plane_sum(partial) → output[row]
/// ```
///
/// # Buffer sizes
///
/// - `pos_bits_u32`: `m × blocks64 × 2` u32 elements
/// - `neg_bits_u32`: `m × blocks64 × 2` u32 elements
/// - `group_scale_f32`: `m × groups_per_row` f32 elements
/// - `input`: `n` f32 elements
/// - `output`: `m` f32 elements
///
/// # Params
///
/// - `blocks64`: u32 words per row per bit-plane is `blocks64 * 2` (u64→u32 cast)
/// - `groups_per_row`: f32 scales per row
/// - `n`: input dimension (must be a multiple of `TERNARY_GROUP_SIZE` = 128
///   for the clean group loop; the kernel also handles a ragged final group)
/// - `m`: output row count, taken from the weight handle. Passed **explicitly**
///   rather than derived from `output.len()` — `BufferArg::from_raw_parts` does
///   not constrain the kernel-visible length, so an oversized output handle
///   would silently widen the row loop (Issue 639).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane(
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
    let words_per_row = blocks64 * 2u32; // u64 → 2× u32

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let row_word_base = row * words_per_row;
    let row_scale_base = row * groups_per_row;

    let mut partial = f32::new(0.0f32);

    let mut g = 0u32;
    while g < groups_per_row {
        let scale = group_scale_f32[(row_scale_base + g) as usize];

        // Group g covers cols [g*128, g*128+128). Each lane handles 4 cols
        // (stride PLANE_DIM=32): lane 0 → {0,32,64,96}, lane 1 → {1,33,...}, etc.
        let group_col_base = g * TERNARY_GROUP_SIZE;

        let mut k = lane;
        while k < TERNARY_GROUP_SIZE {
            let col = group_col_base + k;

            if col < n {
                // Which u64 block does this col live in?
                let block_idx = col / BITS_PER_BLOCK; // 0 or 1 within this group
                // Which u32 word within the block? (col%64)/32 → 0 (low) or 1 (high)
                let word_in_block = (col % BITS_PER_BLOCK) / 32u32;
                // Bit position within the u32 word
                let bit_pos = col % 32u32;

                let pos_word_idx = (row_word_base + block_idx * 2u32 + word_in_block) as usize;
                let neg_word_idx = pos_word_idx; // same layout for neg plane

                let pos_word = pos_bits_u32[pos_word_idx];
                let neg_word = neg_bits_u32[neg_word_idx];

                let pos = (pos_word >> bit_pos) & 1u32;
                let neg = (neg_word >> bit_pos) & 1u32;

                // sign = pos - neg → +1, 0, or -1 (as f32)
                let sign_f = (pos as f32) - (neg as f32);

                partial += sign_f * scale * input[col as usize];
            }

            k += PLANE_DIM;
        }

        g += 1u32;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane
    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row
    if lane == 0u32 {
        output[row as usize] = result;
    }
}

/// Word-packed ternary GEMV: one u32 bit-plane word (= 32 weights) per lane load.
///
/// # Why this exists (Issue 606)
///
/// [`gemv_ternary_plane`] was structurally copied from the Q4_K kernel, which
/// assigns each lane a *stride-`PLANE_DIM`* slice of the block. That is correct
/// for Q4_K (4 bits/weight → stride-32 lanes touch distinct bytes), but it is
/// pathological for a 1-bit plane: cols `[0,32)` all live in the **same u32
/// word**, so all 32 lanes issue a separate load of that one word and each
/// extracts a single bit. The kernel therefore issues `2·n` loads per row to
/// fetch `2·n/32` words — **32× redundant memory instructions**, leaving it
/// instruction-issue bound at ~2% of memory bandwidth.
///
/// This variant inverts the mapping: lane `L` loads word `L, L+32, L+64, …` and
/// unrolls all 32 bits of it locally. Loads per row drop from `2·n` to
/// `2·n/32`, and adjacent lanes read *adjacent* words → fully coalesced.
///
/// # Layout invariant
///
/// The u64→2×u32 cast is little-endian, so flat u32 word `W` covers columns
/// `[W·32, W·32+32)` contiguously: `W = B·2 + b/32` and `col = B·64 + b`, hence
/// `col = W·32 + (b % 32)`. Group `g` (128 cols) therefore spans words
/// `[g·4, g·4+4)`, so the scale for word `W` is `group_scale[W / 4]`.
///
/// Because the scale is constant across a whole word, it factors out of the
/// 32-bit inner loop — one multiply per 32 weights instead of one per weight.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_packed(
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
    let words_per_row = blocks64 * 2u32; // u64 → 2× u32

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let row_word_base = row * words_per_row;
    let row_scale_base = row * groups_per_row;

    let mut partial = f32::new(0.0f32);

    // Lane L walks words L, L+PLANE_DIM, L+2·PLANE_DIM, … → coalesced.
    let mut w = lane;
    while w < words_per_row {
        let word_idx = (row_word_base + w) as usize;
        let pos_word = pos_bits_u32[word_idx];
        let neg_word = neg_bits_u32[word_idx];

        // One scale per 128 cols = per 4 words; constant across this word.
        let scale = group_scale_f32[(row_scale_base + w / 4u32) as usize];

        let col_base = w * 32u32;
        let mut acc = f32::new(0.0f32);

        // Skip whole words that are entirely zero in both planes — ternary
        // weights are ~35% zeros, and an all-zero word costs no input loads.
        if pos_word != 0u32 || neg_word != 0u32 {
            let mut b = 0u32;
            while b < 32u32 {
                let col = col_base + b;
                if col < n {
                    let pos = (pos_word >> b) & 1u32;
                    let neg = (neg_word >> b) & 1u32;
                    let sign_f = (pos as f32) - (neg as f32);
                    acc += sign_f * input[col as usize];
                }
                b += 1u32;
            }
        }

        partial += acc * scale;

        w += PLANE_DIM;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane
    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row
    if lane == 0u32 {
        output[row as usize] = result;
    }
}

/// Output rows processed per plane by [`gemv_ternary_plane_rowtiled`].
///
/// 4 keeps all accumulators in registers while cutting the dominant input-load
/// count 4×. Measured best on M3 Metal: width 2 gives 1.19×, width 4 gives
/// 2.01×, width 8 gives 1.90× (it starves occupancy on the small shapes). See
/// Bench 606 §"Row-tile width sweep".
///
/// Paired with [`gemv_ternary_plane_rowtiled`]'s hand-unrolled accumulator
/// count — the two MUST agree or output rows go unwritten, which Bench 606's
/// G1 catches via a sentinel pre-fill.
#[cfg(feature = "cubecl_runtime")]
pub(crate) const TERNARY_ROWS_PER_PLANE: u32 = 4;

/// Row-tiled word-packed ternary GEMV — one input load feeds 4 output rows.
///
/// # Why this exists (Issue 606 T3)
///
/// [`gemv_ternary_plane_packed`] fixed the *weight* access pattern, but the
/// resulting instruction mix is lopsided. Per 32-weight word a lane issues:
///
/// - **2** global loads of weight (`pos_word`, `neg_word`), and
/// - **32** global loads of input (one per bit).
///
/// So input loads outnumber weight loads **16:1**, and the kernel is bound by
/// input-load issue rather than by weight bandwidth — which is exactly why it
/// measured ~16% of the *weight*-bandwidth roofline. Streaming the weights
/// faster cannot help while the input side costs 16× more instructions.
///
/// This variant is the standard GEMV answer: give each plane
/// `TERNARY_ROWS_PER_PLANE` consecutive output rows and hold one accumulator
/// per row in registers. The input element for bit `b` is loaded **once** and
/// reused across all 4 rows, so per 4 rows × 1 word the load count falls from
/// `4 × (2 + 32) = 136` to `8 + 32 = 40` — a **3.4× reduction**, with the
/// arithmetic unchanged.
///
/// # Divergence
///
/// `row` depends only on the plane index, never on `lane`, so the per-row
/// bounds guard is plane-uniform and costs no divergence. `plane_sum` is
/// called unconditionally by every lane for every row; only the final write is
/// guarded.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_rowtiled(
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
    let row_base = plane_id * TERNARY_ROWS_PER_PLANE;

    if row_base >= m {
        terminate!();
    }

    // Clamped row indices: rows past `m` alias row `m-1` so every load stays in
    // bounds; their results are discarded at write time. Written as guarded
    // assignments because CubeCL's prelude has no `min` for `u32`.
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

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);

    let mut w = lane;
    while w < words_per_row {
        let scale_idx = w / 4u32;
        let col_base = w * 32u32;

        let p0 = pos_bits_u32[(r0 * words_per_row + w) as usize];
        let n0 = neg_bits_u32[(r0 * words_per_row + w) as usize];
        let p1 = pos_bits_u32[(r1 * words_per_row + w) as usize];
        let n1 = neg_bits_u32[(r1 * words_per_row + w) as usize];
        let p2 = pos_bits_u32[(r2 * words_per_row + w) as usize];
        let n2 = neg_bits_u32[(r2 * words_per_row + w) as usize];
        let p3 = pos_bits_u32[(r3 * words_per_row + w) as usize];
        let n3 = neg_bits_u32[(r3 * words_per_row + w) as usize];

        // Skip the whole word only when every row is zero there. Ternary is
        // ~30% zeros, so this fires rarely at tile width 4 — it is kept because
        // it costs one OR-chain and pays for 32 input loads when it hits.
        if (p0 | n0 | p1 | n1 | p2 | n2 | p3 | n3) != 0u32 {
            let s0 = group_scale_f32[(r0 * groups_per_row + scale_idx) as usize];
            let s1 = group_scale_f32[(r1 * groups_per_row + scale_idx) as usize];
            let s2 = group_scale_f32[(r2 * groups_per_row + scale_idx) as usize];
            let s3 = group_scale_f32[(r3 * groups_per_row + scale_idx) as usize];

            // Two-accumulator select-form ternary dot (Issue 613 T1).
            //
            // Replaces the 32-iteration runtime while-loop + float arithmetic
            // (((p>>b)&1) as f32 - ((n>>b)&1) as f32) * x with compile-time
            // unrolled select-based conditional adds:
            //   a_pos += select(0.0, x, (p>>b)&1 != 0)   // sums x where weight is +1
            //   a_neg += select(0.0, x, (n>>b)&1 != 0)   // sums x where weight is -1
            //   result = scale * (a_pos - a_neg)
            //
            // Why: llama.cpp's Metal Q2_0 kernel reaches 23% roofline with this
            // exact select + unroll shape; our original while-loop measured at
            // the same 23% bandwidth share but lower tok/s due to loop overhead
            // and float-cast instruction pressure. The `select` intrinsic maps
            // to a single branchless conditional-move on Metal (Issue 613 T1).
            let mut a0p = f32::new(0.0f32);
            let mut a0n = f32::new(0.0f32);
            let mut a1p = f32::new(0.0f32);
            let mut a1n = f32::new(0.0f32);
            let mut a2p = f32::new(0.0f32);
            let mut a2n = f32::new(0.0f32);
            let mut a3p = f32::new(0.0f32);
            let mut a3n = f32::new(0.0f32);

            #[unroll]
            for b in 0u32..32u32 {
                let col = col_base + b;
                if col < n {
                    // The one load this whole tile exists to amortize.
                    let x = input[col as usize];
                    let pb0 = (p0 >> b) & 1u32 != 0u32;
                    let nb0 = (n0 >> b) & 1u32 != 0u32;
                    let pb1 = (p1 >> b) & 1u32 != 0u32;
                    let nb1 = (n1 >> b) & 1u32 != 0u32;
                    let pb2 = (p2 >> b) & 1u32 != 0u32;
                    let nb2 = (n2 >> b) & 1u32 != 0u32;
                    let pb3 = (p3 >> b) & 1u32 != 0u32;
                    let nb3 = (n3 >> b) & 1u32 != 0u32;
                    a0p += select(pb0, x, f32::new(0.0f32));
                    a0n += select(nb0, x, f32::new(0.0f32));
                    a1p += select(pb1, x, f32::new(0.0f32));
                    a1n += select(nb1, x, f32::new(0.0f32));
                    a2p += select(pb2, x, f32::new(0.0f32));
                    a2n += select(nb2, x, f32::new(0.0f32));
                    a3p += select(pb3, x, f32::new(0.0f32));
                    a3n += select(nb3, x, f32::new(0.0f32));
                }
            }

            let a0 = a0p - a0n;
            let a1 = a1p - a1n;
            let a2 = a2p - a2n;
            let a3 = a3p - a3n;

            acc0 += a0 * s0;
            acc1 += a1 * s1;
            acc2 += a2 * s2;
            acc3 += a3 * s3;
        }

        w += PLANE_DIM;
    }

    let t0 = plane_sum(acc0);
    let t1 = plane_sum(acc1);
    let t2 = plane_sum(acc2);
    let t3 = plane_sum(acc3);

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
    }
}

/// Output rows processed per plane by [`gemv_ternary_plane_rowtiled8`].
///
/// Paired with that kernel's hand-unrolled accumulator count — the two MUST
/// agree or rows go unwritten. Bench 606's G1 asserts full output coverage via
/// a sentinel pre-fill precisely because a silent mismatch here is invisible to
/// a relative-error check (CubeCL pools buffers, so `client.empty()` can return
/// the previous kernel's correct results).
#[cfg(feature = "cubecl_runtime")]
pub(crate) const TERNARY_ROWS_PER_PLANE_8: u32 = 8;

/// Row-tiled word-packed ternary GEMV at tile width 8.
///
/// Same idea as [`gemv_ternary_plane_rowtiled`] but amortizing each input load
/// across 8 output rows instead of 4, halving the per-row input-load count
/// again. Costs 8 accumulators plus 16 weight words live per iteration, so it
/// trades register pressure for load traffic; whether that wins is a measured
/// question, not an obvious one — see Bench 606 §"Row-tile width sweep".
///
/// **Originally a LOSER on M3 Metal** (Bench 606 T3: 1.90–1.95× vs stride,
/// against tile-4's 2.01–2.02×) — tied tile-4 on large shapes but collapsed on
/// small ones (`ssm_alpha/beta` 48 rows, invoked 96×/token, cost 90.5 µs vs
/// tile-4's 52.4 µs, because a 64-row workgroup leaves almost no workgroups to
/// fill the GPU).
///
/// **Promoted to the Metal `launch()` default by Issue 613 T1-follow-up**
/// (2026-08-11). Applying the same `#[unroll]` + `select`-based accumulation
/// as tile-4 flipped the verdict: width 8 now beats width 4 by a median +15.5%
/// on M3 Metal (19.38 vs 16.78 tok/s projection-only, 3 runs). The original
/// occupancy diagnosis held only for the unoptimized `while b < 32` +
/// float-arithmetic inner loop — once unrolled, the instruction pressure drops
/// enough that the 8 accumulators fit without spilling, and the extra
/// input-load amortization (8 rows per load vs 4) becomes a net win on the
/// large dominant shapes (`ffn_gate/up`, `ffn_down`, `lm_head`).
///
/// **NOT measured on CUDA** — the Metal/CUDA win divergence documented in
/// Issue 607 (packed won on Metal, lost 2.1× on CUDA) means width-8-on-CUDA
/// must be swept before any CUDA promotion (Issue 613 T2).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_rowtiled8(
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
        // w >> 2 maps word index to scale group (4 words = 128 weights/group).
        // Explicit shift — GPU shader compilers don't always lower u32 / to >>.
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

        // The original zero-word skip `if (p0|q0|...|p7|q7) != 0` was removed:
        // at tile width 8, P(all 16 weight words simultaneously zero) ≈
        // (0.3^32)^8 ≈ 10^-150 — the check never fires and its 15-OR chain
        // was pure instruction overhead. The scale loads below are now
        // unconditional; when weights are zero the accumulator just adds
        // zero, producing the identical result.
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

        // Bounds-check split: the per-bit `if col < n` guard is dead for any
        // shape where `n` is a multiple of 32 (all Bonsai shapes: 5120,
        // 17408, 248320). Check once per word, not per bit. The slow path
        // only fires for the ragged tail of non-multiple-of-32 shapes.
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

// ── Interleaved 2-bit-code ternary GEMV (Issue 628 T1 layout A) ─────────

/// Row-tiled interleaved ternary GEMV at tile width 8 — Issue 628 T1.
///
/// Mirrors [`gemv_ternary_plane_rowtiled8`] exactly, except:
/// - ONE code buffer instead of TWO (pos + neg)
/// - Each u32 holds 16 2-bit codes instead of 32 1-bit flags
/// - `words_per_row = blocks64 * 4` (not `blocks64 * 2`)
/// - `scale_idx = w >> 3` (not `w >> 2`)
/// - Inner loop: `code = (word >> (b * 2)) & 3; is_pos = code == 2; is_neg = code == 0`
///
/// Same total bytes streamed (2 bits/weight), same ALU structure (dual
/// accumulator + select). The ONLY difference is one load stream instead of
/// two — the load-stream hypothesis.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_interleaved_rowtiled8(
    codes_u32: &[u32],
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
    if row_base + 1u32 < m { r1 = row_base + 1u32; }
    if row_base + 2u32 < m { r2 = row_base + 2u32; }
    if row_base + 3u32 < m { r3 = row_base + 3u32; }
    if row_base + 4u32 < m { r4 = row_base + 4u32; }
    if row_base + 5u32 < m { r5 = row_base + 5u32; }
    if row_base + 6u32 < m { r6 = row_base + 6u32; }
    if row_base + 7u32 < m { r7 = row_base + 7u32; }

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
        // Each u32 word covers 16 weight positions (2 bits each).
        // scale_idx: 128 weights per group → 8 words per group → w >> 3.
        let scale_idx = w >> 3u32;
        let col_base = w * 16u32;

        // ONE load per row (vs two in the bit-plane kernel).
        let c0 = codes_u32[(r0 * words_per_row + w) as usize];
        let c1 = codes_u32[(r1 * words_per_row + w) as usize];
        let c2 = codes_u32[(r2 * words_per_row + w) as usize];
        let c3 = codes_u32[(r3 * words_per_row + w) as usize];
        let c4 = codes_u32[(r4 * words_per_row + w) as usize];
        let c5 = codes_u32[(r5 * words_per_row + w) as usize];
        let c6 = codes_u32[(r6 * words_per_row + w) as usize];
        let c7 = codes_u32[(r7 * words_per_row + w) as usize];

        let s0 = group_scale_f32[(r0 * groups_per_row + scale_idx) as usize];
        let s1 = group_scale_f32[(r1 * groups_per_row + scale_idx) as usize];
        let s2 = group_scale_f32[(r2 * groups_per_row + scale_idx) as usize];
        let s3 = group_scale_f32[(r3 * groups_per_row + scale_idx) as usize];
        let s4 = group_scale_f32[(r4 * groups_per_row + scale_idx) as usize];
        let s5 = group_scale_f32[(r5 * groups_per_row + scale_idx) as usize];
        let s6 = group_scale_f32[(r6 * groups_per_row + scale_idx) as usize];
        let s7 = group_scale_f32[(r7 * groups_per_row + scale_idx) as usize];

        // Dual-accumulator select-form ternary dot from 2-bit codes.
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

        if col_base + 16u32 <= n {
            // Fast path: no per-bit bounds check.
            #[unroll]
            for b in 0u32..16u32 {
                let x = input[(col_base + b) as usize];
                let code0 = (c0 >> (b * 2u32)) & 3u32;
                let code1 = (c1 >> (b * 2u32)) & 3u32;
                let code2 = (c2 >> (b * 2u32)) & 3u32;
                let code3 = (c3 >> (b * 2u32)) & 3u32;
                let code4 = (c4 >> (b * 2u32)) & 3u32;
                let code5 = (c5 >> (b * 2u32)) & 3u32;
                let code6 = (c6 >> (b * 2u32)) & 3u32;
                let code7 = (c7 >> (b * 2u32)) & 3u32;
                // code 0 → -1 (neg), code 1 → 0 (skip), code 2 → +1 (pos)
                a0p += select(code0 == 2u32, x, f32::new(0.0f32));
                a0n += select(code0 == 0u32, x, f32::new(0.0f32));
                a1p += select(code1 == 2u32, x, f32::new(0.0f32));
                a1n += select(code1 == 0u32, x, f32::new(0.0f32));
                a2p += select(code2 == 2u32, x, f32::new(0.0f32));
                a2n += select(code2 == 0u32, x, f32::new(0.0f32));
                a3p += select(code3 == 2u32, x, f32::new(0.0f32));
                a3n += select(code3 == 0u32, x, f32::new(0.0f32));
                a4p += select(code4 == 2u32, x, f32::new(0.0f32));
                a4n += select(code4 == 0u32, x, f32::new(0.0f32));
                a5p += select(code5 == 2u32, x, f32::new(0.0f32));
                a5n += select(code5 == 0u32, x, f32::new(0.0f32));
                a6p += select(code6 == 2u32, x, f32::new(0.0f32));
                a6n += select(code6 == 0u32, x, f32::new(0.0f32));
                a7p += select(code7 == 2u32, x, f32::new(0.0f32));
                a7n += select(code7 == 0u32, x, f32::new(0.0f32));
            }
        } else {
            // Slow path: ragged tail.
            #[unroll]
            for b in 0u32..16u32 {
                let col = col_base + b;
                if col < n {
                    let x = input[col as usize];
                    let code0 = (c0 >> (b * 2u32)) & 3u32;
                    let code1 = (c1 >> (b * 2u32)) & 3u32;
                    let code2 = (c2 >> (b * 2u32)) & 3u32;
                    let code3 = (c3 >> (b * 2u32)) & 3u32;
                    let code4 = (c4 >> (b * 2u32)) & 3u32;
                    let code5 = (c5 >> (b * 2u32)) & 3u32;
                    let code6 = (c6 >> (b * 2u32)) & 3u32;
                    let code7 = (c7 >> (b * 2u32)) & 3u32;
                    a0p += select(code0 == 2u32, x, f32::new(0.0f32));
                    a0n += select(code0 == 0u32, x, f32::new(0.0f32));
                    a1p += select(code1 == 2u32, x, f32::new(0.0f32));
                    a1n += select(code1 == 0u32, x, f32::new(0.0f32));
                    a2p += select(code2 == 2u32, x, f32::new(0.0f32));
                    a2n += select(code2 == 0u32, x, f32::new(0.0f32));
                    a3p += select(code3 == 2u32, x, f32::new(0.0f32));
                    a3n += select(code3 == 0u32, x, f32::new(0.0f32));
                    a4p += select(code4 == 2u32, x, f32::new(0.0f32));
                    a4n += select(code4 == 0u32, x, f32::new(0.0f32));
                    a5p += select(code5 == 2u32, x, f32::new(0.0f32));
                    a5n += select(code5 == 0u32, x, f32::new(0.0f32));
                    a6p += select(code6 == 2u32, x, f32::new(0.0f32));
                    a6n += select(code6 == 0u32, x, f32::new(0.0f32));
                    a7p += select(code7 == 2u32, x, f32::new(0.0f32));
                    a7n += select(code7 == 0u32, x, f32::new(0.0f32));
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
        if row_base + 1u32 < m { output[r1 as usize] = t1; }
        if row_base + 2u32 < m { output[r2 as usize] = t2; }
        if row_base + 3u32 < m { output[r3 as usize] = t3; }
        if row_base + 4u32 < m { output[r4 as usize] = t4; }
        if row_base + 5u32 < m { output[r5 as usize] = t5; }
        if row_base + 6u32 < m { output[r6 as usize] = t6; }
        if row_base + 7u32 < m { output[r7 as usize] = t7; }
    }
}

// ── Base-3 trit-packed ternary GEMV (Issue 628 T2 layout B) ──────────

/// Trits per byte (base-3 encoding: 3^5 = 243 ≤ 256).
pub(crate) const TRITS_PER_BYTE: u32 = 5;

/// Bytes per group, padded to the next u32 boundary for clean word indexing.
/// `ceil(128 / 5) = 26` live bytes → padded to `28` (7 u32 words).
/// Effective bits/weight: 28 × 8 / 128 = 1.75 (vs 2.125 for bit-plane → 17.6% cut).
pub(crate) const TRIT_BYTES_PER_GROUP: u32 = 28;

/// u32 words per trit group (= `TRIT_BYTES_PER_GROUP / 4`).
pub(crate) const TRIT_WORDS_PER_GROUP: u32 = TRIT_BYTES_PER_GROUP / 4; // 7

/// Row-tiled base-3 trit-packed ternary GEMV at tile width 8 — Issue 628 T2.
///
/// Mirrors [`gemv_ternary_interleaved_rowtiled8`] structurally, except:
/// - Weights arrive **5-per-byte in base-3** (1.75 bits/weight vs 2.125)
/// - Each u32 word holds 4 bytes = 20 trit positions
/// - Trit decode via `rem % 3; rem /= 3` (sequential, constant divisor →
///   compiler optimises to multiply-by-inverse)
/// - `scale_idx = w / TRIT_WORDS_PER_GROUP` (7 words per group)
///
/// Same row-tiled structure (8 rows per plane, 32 lanes per plane, plane_sum
/// reduction). Same dual-accumulator select-form ternary dot. The ONLY
/// difference is the weight format: fewer bytes at the cost of a div/mod decode.
///
/// # Buffer layout (group-aligned trit format)
///
/// Each group of 128 weights occupies exactly `TRIT_BYTES_PER_GROUP = 28`
/// bytes (26 live + 2 pad). Bytes are packed 4-per-u32:
/// - `trits_u32[row * words_per_row + word]` holds 4 consecutive bytes
/// - Byte `b` within word `w` = `(trits_u32[w] >> (b*8)) & 0xFF`
/// - Trit `k` (0..4) within byte `b`: `rem = byte; code_k = rem % 3; rem /= 3`
/// - Code 0 → -1, code 1 → 0, code 2 → +1
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_ternary_trit_rowtiled8(
    trits_u32: &[u32],
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
    if row_base + 1u32 < m { r1 = row_base + 1u32; }
    if row_base + 2u32 < m { r2 = row_base + 2u32; }
    if row_base + 3u32 < m { r3 = row_base + 3u32; }
    if row_base + 4u32 < m { r4 = row_base + 4u32; }
    if row_base + 5u32 < m { r5 = row_base + 5u32; }
    if row_base + 6u32 < m { r6 = row_base + 6u32; }
    if row_base + 7u32 < m { r7 = row_base + 7u32; }

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
        // Group index: 7 words per group (28 bytes / 4).
        let scale_idx = w / TRIT_WORDS_PER_GROUP;
        // Word index within the group (avoids a second integer division).
        let word_in_group = w - scale_idx * TRIT_WORDS_PER_GROUP;
        // Actual column base: group g starts at col g*128; within the group,
        // each word covers 20 trit positions. The group-aligned format pads
        // each 128-weight group to 28 bytes (140 trit positions), so the
        // flat word index w*20 does NOT equal the actual column — we must
        // compute from the group structure.
        let col_base = scale_idx * TERNARY_GROUP_SIZE + word_in_group * 4u32 * TRITS_PER_BYTE;

        // ONE load per row (same as interleaved — single buffer).
        let c0 = trits_u32[(r0 * words_per_row + w) as usize];
        let c1 = trits_u32[(r1 * words_per_row + w) as usize];
        let c2 = trits_u32[(r2 * words_per_row + w) as usize];
        let c3 = trits_u32[(r3 * words_per_row + w) as usize];
        let c4 = trits_u32[(r4 * words_per_row + w) as usize];
        let c5 = trits_u32[(r5 * words_per_row + w) as usize];
        let c6 = trits_u32[(r6 * words_per_row + w) as usize];
        let c7 = trits_u32[(r7 * words_per_row + w) as usize];

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

        if col_base + 4u32 * TRITS_PER_BYTE <= n {
            // Fast path: all 20 positions are in-bounds, no per-trit check.
            #[unroll]
            for byte_idx in 0u32..4u32 {
                let shift = byte_idx * 8u32;
                let bv0 = (c0 >> shift) & 0xFFu32;
                let bv1 = (c1 >> shift) & 0xFFu32;
                let bv2 = (c2 >> shift) & 0xFFu32;
                let bv3 = (c3 >> shift) & 0xFFu32;
                let bv4 = (c4 >> shift) & 0xFFu32;
                let bv5 = (c5 >> shift) & 0xFFu32;
                let bv6 = (c6 >> shift) & 0xFFu32;
                let bv7 = (c7 >> shift) & 0xFFu32;

                let bcol = col_base + byte_idx * TRITS_PER_BYTE;

                let mut rem0 = bv0;
                let mut rem1 = bv1;
                let mut rem2 = bv2;
                let mut rem3 = bv3;
                let mut rem4 = bv4;
                let mut rem5 = bv5;
                let mut rem6 = bv6;
                let mut rem7 = bv7;

                #[unroll]
                for k in 0u32..TRITS_PER_BYTE {
                    let col = bcol + k;
                    let x = input[col as usize];
                    let code0 = rem0 % 3u32; rem0 = rem0 / 3u32;
                    let code1 = rem1 % 3u32; rem1 = rem1 / 3u32;
                    let code2 = rem2 % 3u32; rem2 = rem2 / 3u32;
                    let code3 = rem3 % 3u32; rem3 = rem3 / 3u32;
                    let code4 = rem4 % 3u32; rem4 = rem4 / 3u32;
                    let code5 = rem5 % 3u32; rem5 = rem5 / 3u32;
                    let code6 = rem6 % 3u32; rem6 = rem6 / 3u32;
                    let code7 = rem7 % 3u32; rem7 = rem7 / 3u32;
                    a0p += select(code0 == 2u32, x, f32::new(0.0f32));
                    a0n += select(code0 == 0u32, x, f32::new(0.0f32));
                    a1p += select(code1 == 2u32, x, f32::new(0.0f32));
                    a1n += select(code1 == 0u32, x, f32::new(0.0f32));
                    a2p += select(code2 == 2u32, x, f32::new(0.0f32));
                    a2n += select(code2 == 0u32, x, f32::new(0.0f32));
                    a3p += select(code3 == 2u32, x, f32::new(0.0f32));
                    a3n += select(code3 == 0u32, x, f32::new(0.0f32));
                    a4p += select(code4 == 2u32, x, f32::new(0.0f32));
                    a4n += select(code4 == 0u32, x, f32::new(0.0f32));
                    a5p += select(code5 == 2u32, x, f32::new(0.0f32));
                    a5n += select(code5 == 0u32, x, f32::new(0.0f32));
                    a6p += select(code6 == 2u32, x, f32::new(0.0f32));
                    a6n += select(code6 == 0u32, x, f32::new(0.0f32));
                    a7p += select(code7 == 2u32, x, f32::new(0.0f32));
                    a7n += select(code7 == 0u32, x, f32::new(0.0f32));
                }
            }
        } else {
            // Slow path: ragged tail, per-trit bounds check.
            #[unroll]
            for byte_idx in 0u32..4u32 {
                let shift = byte_idx * 8u32;
                let bv0 = (c0 >> shift) & 0xFFu32;
                let bv1 = (c1 >> shift) & 0xFFu32;
                let bv2 = (c2 >> shift) & 0xFFu32;
                let bv3 = (c3 >> shift) & 0xFFu32;
                let bv4 = (c4 >> shift) & 0xFFu32;
                let bv5 = (c5 >> shift) & 0xFFu32;
                let bv6 = (c6 >> shift) & 0xFFu32;
                let bv7 = (c7 >> shift) & 0xFFu32;

                let bcol = col_base + byte_idx * TRITS_PER_BYTE;

                let mut rem0 = bv0;
                let mut rem1 = bv1;
                let mut rem2 = bv2;
                let mut rem3 = bv3;
                let mut rem4 = bv4;
                let mut rem5 = bv5;
                let mut rem6 = bv6;
                let mut rem7 = bv7;

                #[unroll]
                for k in 0u32..TRITS_PER_BYTE {
                    let col = bcol + k;
                    if col < n {
                        let x = input[col as usize];
                        let code0 = rem0 % 3u32; rem0 = rem0 / 3u32;
                        let code1 = rem1 % 3u32; rem1 = rem1 / 3u32;
                        let code2 = rem2 % 3u32; rem2 = rem2 / 3u32;
                        let code3 = rem3 % 3u32; rem3 = rem3 / 3u32;
                        let code4 = rem4 % 3u32; rem4 = rem4 / 3u32;
                        let code5 = rem5 % 3u32; rem5 = rem5 / 3u32;
                        let code6 = rem6 % 3u32; rem6 = rem6 / 3u32;
                        let code7 = rem7 % 3u32; rem7 = rem7 / 3u32;
                        a0p += select(code0 == 2u32, x, f32::new(0.0f32));
                        a0n += select(code0 == 0u32, x, f32::new(0.0f32));
                        a1p += select(code1 == 2u32, x, f32::new(0.0f32));
                        a1n += select(code1 == 0u32, x, f32::new(0.0f32));
                        a2p += select(code2 == 2u32, x, f32::new(0.0f32));
                        a2n += select(code2 == 0u32, x, f32::new(0.0f32));
                        a3p += select(code3 == 2u32, x, f32::new(0.0f32));
                        a3n += select(code3 == 0u32, x, f32::new(0.0f32));
                        a4p += select(code4 == 2u32, x, f32::new(0.0f32));
                        a4n += select(code4 == 0u32, x, f32::new(0.0f32));
                        a5p += select(code5 == 2u32, x, f32::new(0.0f32));
                        a5n += select(code5 == 0u32, x, f32::new(0.0f32));
                        a6p += select(code6 == 2u32, x, f32::new(0.0f32));
                        a6n += select(code6 == 0u32, x, f32::new(0.0f32));
                        a7p += select(code7 == 2u32, x, f32::new(0.0f32));
                        a7n += select(code7 == 0u32, x, f32::new(0.0f32));
                    } else {
                        // Still need to advance the div/mod state for all rows.
                        rem0 = rem0 / 3u32;
                        rem1 = rem1 / 3u32;
                        rem2 = rem2 / 3u32;
                        rem3 = rem3 / 3u32;
                        rem4 = rem4 / 3u32;
                        rem5 = rem5 / 3u32;
                        rem6 = rem6 / 3u32;
                        rem7 = rem7 / 3u32;
                    }
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
        if row_base + 1u32 < m { output[r1 as usize] = t1; }
        if row_base + 2u32 < m { output[r2 as usize] = t2; }
        if row_base + 3u32 < m { output[r3 as usize] = t3; }
        if row_base + 4u32 < m { output[r4 as usize] = t4; }
        if row_base + 5u32 < m { output[r5 as usize] = t5; }
        if row_base + 6u32 < m { output[r6 as usize] = t6; }
        if row_base + 7u32 < m { output[r7 as usize] = t7; }
    }
}

/// Repack `TernaryGroupWeights` (bit-plane) → group-aligned base-3 trit bytes.
///
/// Each group of 128 weights occupies exactly `TRIT_BYTES_PER_GROUP = 28` bytes
/// (26 live + 2 pad), padded to the next u32 boundary for clean word indexing.
/// The first trit of each group is at position 0 of its first byte — no
/// cross-group byte sharing (unlike the CPU `TernaryTritWeights` format).
///
/// Returns `(trits_u32, words_per_row)` where `trits_u32` is the flat byte
/// buffer reinterpreted as u32 (4 bytes per word).
pub fn repack_to_group_aligned_trits(w: &TernaryGroupWeights) -> (Vec<u32>, usize) {
    let groups_per_row = w.groups_per_row;
    let bytes_per_group = TRIT_BYTES_PER_GROUP as usize; // 28
    let words_per_row = groups_per_row * (TRIT_WORDS_PER_GROUP as usize); // groups × 7
    let total_bytes = w.rows * groups_per_row * bytes_per_group;
    let mut bytes = vec![0u8; total_bytes];

    let pow3 = [1u8, 3, 9, 27, 81];

    for r in 0..w.rows {
        for g in 0..groups_per_row {
            let group_byte_base = (r * groups_per_row + g) * bytes_per_group;
            let w_start = g * TERNARY_GROUP_SIZE as usize;
            let w_end = (w_start + TERNARY_GROUP_SIZE as usize).min(w.cols);

            for b in 0..bytes_per_group {
                let mut byte_val: u8 = 0;
                for k in 0..5usize {
                    let col = w_start + b * 5 + k;
                    let trit: u8 = if col < w_end {
                        // TernaryGroupWeights::get returns i8 in {-1, 0, +1}.
                        // Map to base-3 digit: -1→0, 0→1, +1→2.
                        (w.get(r, col) + 1) as u8
                    } else {
                        1 // pad = zero weight (code 1)
                    };
                    byte_val += trit * pow3[k];
                }
                bytes[group_byte_base + b] = byte_val;
            }
        }
    }

    let words: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
    (words, words_per_row)
}

/// Group-aligned base-3 trit handle (Issue 628 T2 layout B).
///
/// Sibling to [`InterleavedTernaryHandle`] — holds ONE trit buffer (1.75
/// bits/weight, base-3 packed 5-per-byte, group-aligned to 28 bytes) + the
/// group scales. The kernel decodes trits via div/mod (constant divisor 3 →
/// compiler optimises to multiply-by-inverse), testing whether the 17.6% byte
/// reduction improves throughput on bandwidth-bound shapes.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct TernaryTritHandle {
    /// Group-aligned trit bytes as `Array<u32>` (4 bytes per word).
    pub trits_u32: Handle,
    /// Pre-decoded group scales as `Array<f32>` (same as [`TernaryHandle`]).
    pub group_scale_f32: Handle,
    /// Output dimension (number of rows).
    pub m: usize,
    /// Input dimension (number of columns).
    pub n: usize,
    /// Groups per row (= `n.div_ceil(128)`).
    pub groups_per_row: usize,
    /// u32 words per row in `trits_u32` (= `groups_per_row * 7`).
    pub words_per_row: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryTritHandle {
    /// Build from a [`TernaryGroupWeights`] by repacking the bit-planes into
    /// group-aligned base-3 trit bytes, then uploading to GPU.
    pub fn from_weights(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w: &TernaryGroupWeights,
    ) -> Self {
        let scale_f32 = prepare_group_scale_f32(&w.group_scale);
        let (trits, words_per_row) = repack_to_group_aligned_trits(w);

        let trits_u32 = client.create_from_slice(bytemuck::cast_slice(&trits));
        let group_scale_f32 = client.create_from_slice(f32::as_bytes(&scale_f32));

        Self {
            trits_u32,
            group_scale_f32,
            m: w.rows,
            n: w.cols,
            groups_per_row: w.groups_per_row,
            words_per_row,
        }
    }
}

// ── LUT / subset-sum ternary GEMV (Issue 606 T3c path 1) ───────────────

/// Activation columns per LUT tile.
///
/// 1024 is forced by the geometry, not chosen: at `PLANE_DIM = 32` a tile of
/// 1024 columns is exactly **32 u32 words** (one per lane) and exactly **256
/// groups of 4** (one per thread at `CubeDim = 256`). Both the build and the
/// consume mapping come out one-item-per-thread with no remainder loop.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_lut_gemv"))]
const LUT_TILE_COLS: u32 = 1024;

/// Subset-sum table entries per tile: 256 groups × 16 masks = 4096 f32 = 16 KB
/// of threadgroup memory (Metal's budget is 32 KB, so occupancy is preserved).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_lut_gemv"))]
const LUT_ENTRIES_PER_TILE: u32 = 4096;

/// Subset-sum ("LUT") ternary GEMV — 2 lookups + 1 subtract per FOUR weights.
///
/// # The idea (Issue 606 T3c path 1)
///
/// After T3's row tiling the kernel is ALU-bound: every weight costs 2 shifts,
/// 2 masks, 2 int→float converts, a subtract and an FMA. But a ternary dot over
/// four weights only ever takes one of a *small* set of values. Writing `p` for
/// the 4-bit pattern of `+1` weights and `q` for the `-1` pattern,
///
/// ```text
/// Σ_{i<4} sign_i · x_i  =  ( Σ_{i∈p} x_i ) − ( Σ_{i∈q} x_i )  =  S[p] − S[q]
/// ```
///
/// where `S[m] = Σ_{i∈m} x_i` over the 16 subsets of a 4-activation group. So a
/// 4-weight dot collapses to **two table reads and one subtract** — replacing
/// roughly 28 ops with about 8.
///
/// # Why this is not the Q4_K LUT, and why the Q4_K negative does not transfer
///
/// `katgpt_core::simd_lut_dequant` (Plan 431/452, feature `simd_lut_dequant`)
/// builds `lut[code] = (signed(code) − z) · s`: it is keyed by the **weight
/// value** and baked from that group's scale and zero-point. Its Q4_K
/// integration (Plan 486, Bench 487) recorded a *negative* — per-block LUT
/// rebuild ate the fusion win — because those inputs change per row **and** per
/// sub-block, so a 5120-wide row rebuilds ~160 tables and each one amortizes
/// over only 32 elements.
///
/// This table is keyed by the **activation subset** and contains no weight, no
/// scale and no zero-point. It therefore depends on nothing that varies down the
/// rows: one build serves **every output row in the matrix**. That is a
/// structurally different amortization regime, not a tuning difference — which
/// is the reason to expect a different verdict here.
///
/// Group scales still exist and are still per-row, but they multiply *outside*
/// the table: one scale multiply per 32 weights, unchanged from
/// [`gemv_ternary_plane_rowtiled`].
///
/// # Geometry
///
/// | | value | why |
/// |---|---|---|
/// | tile | 1024 cols | = 32 words = `PLANE_DIM`; = 256 groups = `CubeDim` |
/// | table | 4096 f32 (16 KB) | 256 groups × 16 masks |
/// | build | 1 group per thread | 4 loads + 12 adds, no loop |
/// | consume | 1 word per lane | 8 nibble-pairs → 16 reads |
///
/// Build cost is amortized over `8 planes × TERNARY_ROWS_PER_PLANE` = 32 rows
/// per workgroup, and over every tile iteration.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_lut_gemv"))]
#[cube(launch_unchecked)]
fn gemv_ternary_plane_lut(
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
    let words_per_tile = LUT_TILE_COLS / 32u32; // 32

    let plane_id = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let t = UNIT_POS;
    let row_base = plane_id * TERNARY_ROWS_PER_PLANE;

    // Rows past `m` alias `m-1` so loads stay in bounds; discarded at write.
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

    let mut lut = Shared::<[f32]>::new_slice(LUT_ENTRIES_PER_TILE as usize);

    let mut acc0 = f32::new(0.0f32);
    let mut acc1 = f32::new(0.0f32);
    let mut acc2 = f32::new(0.0f32);
    let mut acc3 = f32::new(0.0f32);

    let n_tiles = n.div_ceil(LUT_TILE_COLS);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_col_base = tile * LUT_TILE_COLS;

        // ── Build: thread `t` owns group `t` of this tile ──
        // Columns past `n` contribute 0, so the tail needs no special case.
        let c0 = tile_col_base + t * 4u32;
        let mut x0 = f32::new(0.0f32);
        let mut x1 = f32::new(0.0f32);
        let mut x2 = f32::new(0.0f32);
        let mut x3 = f32::new(0.0f32);
        if c0 < n {
            x0 = input[c0 as usize];
        }
        if c0 + 1u32 < n {
            x1 = input[(c0 + 1u32) as usize];
        }
        if c0 + 2u32 < n {
            x2 = input[(c0 + 2u32) as usize];
        }
        if c0 + 3u32 < n {
            x3 = input[(c0 + 3u32) as usize];
        }

        // 12 adds for 16 subset sums, sharing the two pair-sums.
        let x01 = x0 + x1;
        let x23 = x2 + x3;
        let base = t * 16u32;
        lut[base as usize] = f32::new(0.0f32);
        lut[(base + 1u32) as usize] = x0;
        lut[(base + 2u32) as usize] = x1;
        lut[(base + 3u32) as usize] = x01;
        lut[(base + 4u32) as usize] = x2;
        lut[(base + 5u32) as usize] = x0 + x2;
        lut[(base + 6u32) as usize] = x1 + x2;
        lut[(base + 7u32) as usize] = x01 + x2;
        lut[(base + 8u32) as usize] = x3;
        lut[(base + 9u32) as usize] = x0 + x3;
        lut[(base + 10u32) as usize] = x1 + x3;
        lut[(base + 11u32) as usize] = x01 + x3;
        lut[(base + 12u32) as usize] = x23;
        lut[(base + 13u32) as usize] = x0 + x23;
        lut[(base + 14u32) as usize] = x1 + x23;
        lut[(base + 15u32) as usize] = x01 + x23;

        sync_cube();

        // ── Consume: lane `L` owns word `tile*32 + L`, i.e. local cols
        // [L*32, L*32+32) = local groups [L*8, L*8+8). ──
        if row_base < m {
            let w = tile * words_per_tile + lane;
            if w < words_per_row {
                let scale_idx = w / 4u32;
                let group_base = lane * 8u32;

                let p0 = pos_bits_u32[(r0 * words_per_row + w) as usize];
                let q0 = neg_bits_u32[(r0 * words_per_row + w) as usize];
                let p1 = pos_bits_u32[(r1 * words_per_row + w) as usize];
                let q1 = neg_bits_u32[(r1 * words_per_row + w) as usize];
                let p2 = pos_bits_u32[(r2 * words_per_row + w) as usize];
                let q2 = neg_bits_u32[(r2 * words_per_row + w) as usize];
                let p3 = pos_bits_u32[(r3 * words_per_row + w) as usize];
                let q3 = neg_bits_u32[(r3 * words_per_row + w) as usize];

                let mut a0 = f32::new(0.0f32);
                let mut a1 = f32::new(0.0f32);
                let mut a2 = f32::new(0.0f32);
                let mut a3 = f32::new(0.0f32);

                let mut j = 0u32;
                while j < 8u32 {
                    let shift = j * 4u32;
                    let gb = (group_base + j) * 16u32;
                    a0 += lut[(gb + ((p0 >> shift) & 15u32)) as usize]
                        - lut[(gb + ((q0 >> shift) & 15u32)) as usize];
                    a1 += lut[(gb + ((p1 >> shift) & 15u32)) as usize]
                        - lut[(gb + ((q1 >> shift) & 15u32)) as usize];
                    a2 += lut[(gb + ((p2 >> shift) & 15u32)) as usize]
                        - lut[(gb + ((q2 >> shift) & 15u32)) as usize];
                    a3 += lut[(gb + ((p3 >> shift) & 15u32)) as usize]
                        - lut[(gb + ((q3 >> shift) & 15u32)) as usize];
                    j += 1u32;
                }

                acc0 += a0 * group_scale_f32[(r0 * groups_per_row + scale_idx) as usize];
                acc1 += a1 * group_scale_f32[(r1 * groups_per_row + scale_idx) as usize];
                acc2 += a2 * group_scale_f32[(r2 * groups_per_row + scale_idx) as usize];
                acc3 += a3 * group_scale_f32[(r3 * groups_per_row + scale_idx) as usize];
            }
        }

        // Re-arm before the next tile overwrites the table.
        sync_cube();

        tile += 1u32;
    }

    let t0 = plane_sum(acc0);
    let t1 = plane_sum(acc1);
    let t2 = plane_sum(acc2);
    let t3 = plane_sum(acc3);

    if row_base < m && lane == 0u32 {
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
    }
}

// ---------------------------------------------------------------------------
// Handle — paired GPU buffers for one ternary projection
// ---------------------------------------------------------------------------

/// Paired GPU handles for one ternary (Q2_0_g128) projection.
///
/// Stores the two bit-planes (as u32 arrays) and pre-decoded group scales
/// (as f32) as separate CubeCL handles.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct TernaryHandle {
    /// Positive bit-plane as `Array<u32>` (u64→2×u32 cast).
    /// Layout: `[rows * blocks64 * 2]` u32 elements.
    pub pos_bits_u32: Handle,
    /// Negative bit-plane as `Array<u32>` (u64→2×u32 cast).
    /// Layout: `[rows * blocks64 * 2]` u32 elements.
    pub neg_bits_u32: Handle,
    /// Pre-decoded group scales as `Array<f32>` (f16→f32 on CPU).
    /// Layout: `[rows * groups_per_row]` f32 elements.
    pub group_scale_f32: Handle,
    /// RAW f16 group scales (Issue 764 T2, Benches 767/768; DEFAULT ON since
    /// Bench 771) — the same values in their native 2-byte GGUF storage, for
    /// the f16-scale kernel arm (bit-identical output, ~half the scale
    /// bytes). Uploaded at construction ONLY while the toggle is on
    /// (`RIIR_GEMV_F16_SCALE=0` skips — then this is `None` and every launch
    /// takes the f32 path); dispatched only when
    /// [`crate::set_gemv_use_f16_scale`] is on.
    pub group_scale_f16: Option<Handle>,
    /// Output dimension (number of rows).
    pub m: usize,
    /// Input dimension (number of columns).
    pub n: usize,
    /// u64 blocks per row (= `n.div_ceil(64)`).
    pub blocks64: usize,
    /// Groups per row (= `n.div_ceil(128)`).
    pub groups_per_row: usize,
    /// Metal-side weight cache (Plan 534 T1). Populated by `upload_to_metal`
    /// at model load time when `metal_tensor_gemm` is compiled + macOS.
    /// `None` otherwise — the CubeCL path runs unchanged.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    pub metal: Option<crate::gemm_ternary_metal_tensor::MetalWeightCache>,
    /// Wgpu-side weight cache (Issue 657; Issue 727 H3 makes it LAZY).
    /// Populated at FIRST DISPATCH through the wgpu MSL passthrough path
    /// (`get_or_init` in `prefill_project`) — a GPU→GPU copy from the CubeCL
    /// handles, so it can happen any time after construction. Populated
    /// lazily iff `PREFILL_USE_METAL_TENSOR_WGPU` actually fires; builds that
    /// never enable the flag never pay the copy. `None` inside = a permanent
    /// cache failure (device extraction errors are persistent).
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    pub wgpu_cache: std::sync::OnceLock<Option<crate::gemm_ternary_metal_wgpu::WgpuWeightCache>>,
    /// Zero-copy RAW-Metal weight cache (Issue 663 T5). Populated by
    /// `MetalTensorZeroCopyGemm::cache_weights` at model load time when
    /// `metal_tensor_gemm` is compiled + macOS + the wgpu-hal fork is
    /// vendored. Holds raw `id<MTLBuffer>` pointers borrowed from the
    /// CubeCL pool — zero copy, no staging buffer. `None` otherwise.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    pub zerocopy_cache: Option<crate::gemm_ternary_metal_zero_copy::ZeroCopyWeightCache>,
    /// Issue 734 Arm 6 (Bench 720): CUDA-side weight mirror for the raw-CUDA
    /// mma prefill arm — populated LAZILY at first dispatch (`get_or_init` in
    /// `prefill_project`, the Issue 727 H3 pattern; GPU→host→CUDA, one time
    /// per weight). `None` inside = permanent mirror failure (that weight
    /// falls back to CubeCL).
    #[cfg(all(
        feature = "ternary_gemv_cuda_raw",
        feature = "ternary_gemm_batched",
        not(target_os = "macos")
    ))]
    pub cuda_mma_cache:
        std::sync::OnceLock<Option<std::sync::Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>>>,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryHandle {
    /// Upload a ternary projection matrix to CubeCL GPU buffers.
    ///
    /// Creates three GPU buffers:
    /// 1. `pos_bits_u32`: pos bit-plane, u64→2×u32 cast
    /// 2. `neg_bits_u32`: neg bit-plane, u64→2×u32 cast
    /// 3. `group_scale_f32`: f16 scales decoded to f32
    pub fn from_weights(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w: &TernaryGroupWeights,
    ) -> Self {
        let pos_u32 = cast_u64_to_u32(&w.pos_bits);
        let neg_u32 = cast_u64_to_u32(&w.neg_bits);
        let scale_f32 = prepare_group_scale_f32(&w.group_scale);

        let pos_bits_u32 = client.create_from_slice(bytemuck::cast_slice(&pos_u32));
        let neg_bits_u32 = client.create_from_slice(bytemuck::cast_slice(&neg_u32));
        let group_scale_f32 = client.create_from_slice(f32::as_bytes(&scale_f32));
        // Issue 764 T2: the raw f16 scales ride along ONLY when the toggle is
        // on at construction (the env is read once, so production
        // construction/dispatch agree; a mid-process setter flip cannot
        // conjure a buffer that was never uploaded — the launch() branch
        // falls back to f32 and the counter stays 0, which the bench_768
        // vacuous guard catches). DEFAULT ON since Bench 771 — the f32 set
        // stays resident as the kill-switch fallback (`=0` builds skip this
        // upload entirely).
        let group_scale_f16 = if f16_scale_enabled() {
            Some(client.create_from_slice(bytemuck::cast_slice(&w.group_scale)))
        } else {
            None
        };

        Self {
            pos_bits_u32,
            neg_bits_u32,
            group_scale_f32,
            group_scale_f16,
            m: w.rows,
            n: w.cols,
            blocks64: w.blocks64,
            groups_per_row: w.groups_per_row,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            metal: None,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            wgpu_cache: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            zerocopy_cache: None,
            #[cfg(all(
                feature = "ternary_gemv_cuda_raw",
                feature = "ternary_gemm_batched",
                not(target_os = "macos")
            ))]
            cuda_mma_cache: std::sync::OnceLock::new(),
        }
    }

    /// Upload a ternary projection from raw bit-plane slices (no `TernaryGroupWeights` dep).
    ///
    /// Useful for tests that construct bit-planes directly.
    pub fn from_raw(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        pos_bits: &[u64],
        neg_bits: &[u64],
        group_scale: &[half::f16],
        rows: usize,
        cols: usize,
    ) -> Self {
        let blocks64 = cols.div_ceil(64);
        let groups_per_row = cols.div_ceil(TERNARY_GROUP_SIZE as usize);
        debug_assert_eq!(
            pos_bits.len(),
            rows * blocks64,
            "pos_bits length mismatch"
        );
        debug_assert_eq!(
            neg_bits.len(),
            rows * blocks64,
            "neg_bits length mismatch"
        );
        debug_assert_eq!(
            group_scale.len(),
            rows * groups_per_row,
            "group_scale length mismatch"
        );

        let pos_u32 = cast_u64_to_u32(pos_bits);
        let neg_u32 = cast_u64_to_u32(neg_bits);
        let scale_f32 = prepare_group_scale_f32(group_scale);

        let pos_bits_u32 = client.create_from_slice(bytemuck::cast_slice(&pos_u32));
        let neg_bits_u32 = client.create_from_slice(bytemuck::cast_slice(&neg_u32));
        let group_scale_f32 = client.create_from_slice(f32::as_bytes(&scale_f32));
        let group_scale_f16 = f16_scale_enabled()
            .then(|| client.create_from_slice(bytemuck::cast_slice(group_scale)));

        Self {
            pos_bits_u32,
            neg_bits_u32,
            group_scale_f32,
            group_scale_f16,
            m: rows,
            n: cols,
            blocks64,
            groups_per_row,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            metal: None,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            wgpu_cache: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            zerocopy_cache: None,
            #[cfg(all(
                feature = "ternary_gemv_cuda_raw",
                feature = "ternary_gemm_batched",
                not(target_os = "macos")
            ))]
            cuda_mma_cache: std::sync::OnceLock::new(),
        }
    }

    /// Concatenate two ternary weight matrices vertically (row-stack).
    ///
    /// Used by Issue 642 F2 to fuse gate_proj + up_proj into a single GEMV.
    /// Both matrices MUST share the same `cols` (they consume the same input).
    /// The resulting handle has `rows = w1.rows + w2.rows`.
    pub fn from_two_weights(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w1: &TernaryGroupWeights,
        w2: &TernaryGroupWeights,
    ) -> Self {
        Self::from_weights_concat(client, &[w1, w2])
    }

    /// Concatenate N ternary weight matrices vertically (row-stack).
    ///
    /// Used by Issue 642 F3 to fuse DeltaNet in_proj_qkv + z + a + b into a
    /// single GEMV. All matrices MUST share the same `cols`.
    /// The resulting handle has `rows = Σ weights[i].rows`.
    pub fn from_weights_concat(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        weights: &[&TernaryGroupWeights],
    ) -> Self {
        assert!(!weights.is_empty(), "from_weights_concat requires ≥1 weight");
        let cols = weights[0].cols;
        let blocks64 = weights[0].blocks64;
        let groups_per_row = weights[0].groups_per_row;
        let total_rows: usize = weights.iter().map(|w| w.rows).sum();

        // Validate all weights share the same layout parameters.
        for w in weights {
            debug_assert_eq!(w.cols, cols, "all weights must share cols (input dim)");
            debug_assert_eq!(w.blocks64, blocks64, "blocks64 mismatch");
            debug_assert_eq!(w.groups_per_row, groups_per_row, "groups_per_row mismatch");
        }

        // Concatenate bit-planes and scales row-wise.
        let mut pos_u32 = Vec::new();
        let mut neg_u32 = Vec::new();
        let mut scale_f32 = Vec::new();
        let mut scale_f16: Vec<half::f16> = Vec::new();
        for w in weights {
            pos_u32.extend_from_slice(&cast_u64_to_u32(&w.pos_bits));
            neg_u32.extend_from_slice(&cast_u64_to_u32(&w.neg_bits));
            scale_f32.extend_from_slice(&prepare_group_scale_f32(&w.group_scale));
            scale_f16.extend_from_slice(&w.group_scale);
        }

        let pos_bits_u32 = client.create_from_slice(bytemuck::cast_slice(&pos_u32));
        let neg_bits_u32 = client.create_from_slice(bytemuck::cast_slice(&neg_u32));
        let group_scale_f32 = client.create_from_slice(f32::as_bytes(&scale_f32));
        let group_scale_f16 = f16_scale_enabled()
            .then(|| client.create_from_slice(bytemuck::cast_slice(&scale_f16)));

        Self {
            pos_bits_u32,
            neg_bits_u32,
            group_scale_f32,
            group_scale_f16,
            m: total_rows,
            n: cols,
            blocks64,
            groups_per_row,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            metal: None,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            wgpu_cache: std::sync::OnceLock::new(),
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            zerocopy_cache: None,
            #[cfg(all(
                feature = "ternary_gemv_cuda_raw",
                feature = "ternary_gemm_batched",
                not(target_os = "macos")
            ))]
            cuda_mma_cache: std::sync::OnceLock::new(),
        }
    }

    /// Upload this weight matrix's raw bit-plane data to Metal buffers (Plan 534 T1).
    ///
    /// Populates the `metal` field with a pre-uploaded `MetalWeightCache`.
    /// Called once per prefill-path weight matrix at model load time when
    /// `metal_tensor_gemm` is compiled + macOS. The CubeCL handles are
    /// unchanged; this is a parallel Metal-side cache for the matmul2d path.
    ///
    /// Requires the raw u32 bit-plane slices + f32 scales. These must match
    /// the layout uploaded to CubeCL: pos/neg as `cast_u64_to_u32` output,
    /// scales as `prepare_group_scale_f32` output.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    pub fn upload_to_metal(
        &mut self,
        gemm: &crate::gemm_ternary_metal_tensor::MetalTensorGemm,
        pos_bits_u32: &[u32],
        neg_bits_u32: &[u32],
        group_scale_f32: &[f32],
    ) {
        self.metal = Some(crate::gemm_ternary_metal_tensor::MetalWeightCache::upload(
            gemm,
            pos_bits_u32,
            neg_bits_u32,
            group_scale_f32,
        ));
    }
}

/// Repack two bit-plane slices (pos + neg u64 per block) into a single
/// interleaved 2-bit-code u32 array (Issue 628 T1 layout A).
///
/// This is the CPU-side repacker that runs at weight-upload time. The output
/// is uploaded to a single GPU buffer, replacing the two-plane `pos_bits_u32`
/// + `neg_bits_u32` pair.
///
/// Returns `(codes, words_per_row)` where `codes.len() = rows *
/// words_per_row` and `words_per_row = blocks64 * 4`.
///
/// # Panics
///
/// If `pos_bits` and `neg_bits` have different lengths, or if any weight has
/// BOTH pos and neg bits set (the representation invariant).
#[cfg(feature = "cubecl_runtime")]
pub fn repack_bitplanes_to_interleaved(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    rows: usize,
    blocks64: usize,
) -> (Vec<u32>, usize) {
    let words_per_row_plane = blocks64 * 2; // u32 per row per plane
    let words_per_row_code = blocks64 * 4; // u32 per row (16 codes each)
    let total_words = rows * words_per_row_code;
    let mut codes = vec![0u32; total_words];

    for row in 0..rows {
        let plane_base = row * words_per_row_plane;
        let code_base = row * words_per_row_code;

        for w in 0..words_per_row_plane {
            let p_word = pos_bits_u32[plane_base + w];
            let n_word = neg_bits_u32[plane_base + w];

            // Each u32 word covers 32 weight positions (1 bit each).
            // We pack them into 2 u32s of 16 2-bit codes each.
            // First half (positions 0..15) → code_base + w*2
            // Second half (positions 16..31) → code_base + w*2 + 1
            let mut code_lo: u32 = 0; // positions 0..15
            let mut code_hi: u32 = 0; // positions 16..31

            for b in 0..16u32 {
                // Lower half: positions 0..15
                let pb_lo = (p_word >> b) & 1;
                let nb_lo = (n_word >> b) & 1;
                let code_lo_val: u32 = if pb_lo == 1 {
                    2 // +1
                } else if nb_lo == 1 {
                    0 // -1
                } else {
                    1 // 0
                };
                code_lo |= code_lo_val << (b * 2);

                // Upper half: positions 16..31
                let pb_hi = (p_word >> (b + 16)) & 1;
                let nb_hi = (n_word >> (b + 16)) & 1;
                let code_hi_val: u32 = if pb_hi == 1 {
                    2
                } else if nb_hi == 1 {
                    0
                } else {
                    1
                };
                code_hi |= code_hi_val << (b * 2);
            }

            codes[code_base + w * 2] = code_lo;
            codes[code_base + w * 2 + 1] = code_hi;
        }
    }

    (codes, words_per_row_code)
}

/// Single-buffer interleaved 2-bit-code handle (Issue 628 T1 layout A).
///
/// Sibling to [`TernaryHandle`] — holds ONE code buffer (2 bits/weight, 16
/// codes per u32) + the group scales. The kernel reads from one load stream
/// instead of two, testing whether Metal's memory system coalesces a single
/// stream more efficiently than two interleaved ones.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct InterleavedTernaryHandle {
    /// Interleaved 2-bit codes as `Array<u32>`.
    /// Layout: `[rows * blocks64 * 4]` u32 elements (16 codes per u32).
    pub codes_u32: Handle,
    /// Pre-decoded group scales as `Array<f32>` (same as [`TernaryHandle`]).
    pub group_scale_f32: Handle,
    /// Output dimension (number of rows).
    pub m: usize,
    /// Input dimension (number of columns).
    pub n: usize,
    /// u64 blocks per row (= `n.div_ceil(64)`).
    pub blocks64: usize,
    /// Groups per row (= `n.div_ceil(128)`).
    pub groups_per_row: usize,
    /// u32 words per row in `codes_u32` (= `blocks64 * 4`).
    pub words_per_row: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl InterleavedTernaryHandle {
    /// Build from a [`TernaryGroupWeights`] by repacking the two bit-planes
    /// into interleaved 2-bit codes, then uploading to GPU.
    pub fn from_weights(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        w: &TernaryGroupWeights,
    ) -> Self {
        let pos_u32 = cast_u64_to_u32(&w.pos_bits);
        let neg_u32 = cast_u64_to_u32(&w.neg_bits);
        let scale_f32 = prepare_group_scale_f32(&w.group_scale);

        let (codes, words_per_row) =
            repack_bitplanes_to_interleaved(&pos_u32, &neg_u32, w.rows, w.blocks64);

        let codes_u32 = client.create_from_slice(bytemuck::cast_slice(&codes));
        let group_scale_f32 = client.create_from_slice(f32::as_bytes(&scale_f32));

        Self {
            codes_u32,
            group_scale_f32,
            m: w.rows,
            n: w.cols,
            blocks64: w.blocks64,
            groups_per_row: w.groups_per_row,
            words_per_row,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: launcher
// ---------------------------------------------------------------------------

/// CubeCL ternary bit-plane dequant+GEMV launcher.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = GpuContext::new()?;
/// let client = ctx.cubecl_client();
///
/// let handle = TernaryHandle::from_weights(&client, &ternary_weights);
/// let input_handle = client.create_from_slice(f32::as_bytes(&input));
/// let output_handle = client.empty(rows * 4);
///
/// unsafe {
///     GemvTernaryCubeCL::launch::<ActiveRuntime>(
///         &client, &handle, input_handle, output_handle,
///     );
/// }
///
/// let result = f32::from_bytes(&client.read_one(output_handle).unwrap());
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct GemvTernaryCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvTernaryCubeCL {
    /// Launch ternary bit-plane dequant+GEMV kernel.
    ///
    /// Each plane handles one output row with cooperative dot product + `plane_sum()`.
    /// Workgroup: 256 threads → 8 planes (with plane_dim=32 on Metal).
    /// Dispatch: `ceil(m / 8)` workgroups.
    ///
    /// The packed-vs-stride choice is **backend-conditional** — see below.
    ///
    /// # Safety
    ///
    /// - `input_handle` must point to `handle.n` f32 elements
    /// - `output_handle` must point to `handle.m` f32 elements
    /// - Buffers in `handle` must have been created from the same `client`
    ///
    /// # Why this branches on backend (Issue 607 G5, measured 2026-08-11)
    ///
    /// Issue 606 measured `launch_packed` at **1.48×** over `launch_plane` on M3
    /// Metal and promoted it to the unconditional default. Re-measuring on an
    /// RTX 4090 (native CUDA, `ROOFLINE_GBS=1008`) **reversed** that result — the
    /// packed kernel is **0.48×**, i.e. 2.1× *slower*, and the loss is
    /// concentrated in the two shapes that dominate the roll-up:
    ///
    /// | shape | rows | am stride | am packed | speedup |
    /// |---|---|---|---|---|
    /// | `ffn_gate/up` | 17408 | 129.6 µs | 325.2 µs | **0.40×** |
    /// | `attn_q` | 12288 | 95.3 | 221.9 | **0.43×** |
    /// | `attn_k/v` | 1024 | 40.9 | 28.8 | 1.42× |
    /// | `ssm_alpha/beta` | 48 | 42.6 | 31.0 | 1.37× |
    ///
    /// Mechanism: Issue 606's premise is that the stride kernel wastes `32×`
    /// redundant loads because all 32 lanes read the *same* u32 word. On CUDA
    /// that access is warp-uniform, so it coalesces into a single transaction
    /// broadcast to the whole warp — the redundancy is nearly free. The packed
    /// kernel trades it for a per-lane 32-iteration scalar bit unroll, which is
    /// pure added ALU work. At large `m` the GPU is already saturated
    /// (`m/8` workgroups) so that extra work is exposed; at small `m` the kernel
    /// is launch-latency-bound and packed's fewer loads still win. Metal's
    /// simdgroup load path does not broadcast as cheaply, which is why the same
    /// kernel wins there.
    ///
    /// The crossover therefore depends on `m` as well as backend, but two points
    /// either side of it is not enough to site a threshold honestly. Selecting
    /// per backend is exactly what the measurement supports; a shape-adaptive
    /// rule is Issue 606 T3 work once the crossover is actually swept.
    ///
    /// # Issue 613 T1-follow-up — Metal now dispatches `launch_rowtiled8`
    ///
    /// History:
    /// - Issue 606 T3 promoted `launch_rowtiled` (tile width 4) to the Metal
    ///   default at 2.01× over stride (10.84 tok/s projection).
    /// - Issue 613 T1 added `#[unroll]` + `select` to the width-4 inner loop,
    ///   lifting it to 15.16–15.82 tok/s (+40–43%).
    /// - Issue 613 T1-follow-up applied the SAME `#[unroll]` + `select`
    ///   treatment to the width-8 kernel (`gemv_ternary_plane_rowtiled8`).
    ///   Previously width 8 LOST to width 4 (9.48 vs 15.50 tok/s) because its
    ///   inner loop was still the unoptimized `while b < 32` + float-arithmetic
    ///   form. Once unrolled, width 8's extra input-load amortization wins:
    ///   median 19.38 tok/s vs width-4's 16.78 tok/s (+15.5%, 3 runs).
    ///
    /// Per-shape, width 8 wins on the dominant large shapes (`ffn_gate/up`,
    /// `ffn_down`, `lm_head`) and ties on small shapes where occupancy is the
    /// bottleneck either way. The previous "width 8 starves occupancy"
    /// diagnosis held only for the unoptimized inner loop — once the loop is
    /// unrolled the instruction pressure drops enough that the extra
    /// accumulators fit in registers without spilling.
    ///
    /// # Issue 613 T2 — CUDA now dispatches `launch_rowtiled8` too
    ///
    /// Issue 613 T2 (measured 2026-08-12 on RTX 4090 cubecl-cuda,
    /// `bench_606_ternary_gemv_packed::bench606_g2_packed_vs_stride`, 4 runs,
    /// `ROOFLINE_GBS=1008`) swept all four variants on CUDA. Result:
    ///
    /// | variant | tok/s (projection-only) | vs stride |
    /// |---|---|---|
    /// | stride (`launch_plane`) | 9.6–16.7 (median ~10) | 1.00× |
    /// | packed (`launch_packed`) | 4.3–8.0 (median ~4.4) | 0.45–0.48× (Issue 607 G5a holds) |
    /// | rowtiled (width 4) | 12.4–21.4 (median ~12.5) | 1.27–1.34× |
    /// | **rowtile8 (width 8)** | **19.5–33.4 (median ~19.7)** | **2.00–2.05×** |
    ///
    /// Unlike packed, row-tiled **wins on CUDA**. The mechanism: row-tiling
    /// amortizes each input load across N output rows (4 for `rowtiled`, 8
    /// for `rowtile8`). On CUDA the input load was already cheap
    /// (warp-broadcast), but the input *bandwidth* at scale was not — at the
    /// large FFN shapes (17408×5120) the input vector is 20 KB and is read
    /// once per output row by `launch_plane`. Row-tiling cuts those reads by
    /// 4–8×, and unlike packed it does NOT add a 32-iteration scalar bit
    /// unroll (the ALU cost that made packed lose 0.48× on CUDA). G1 PASS on
    /// all four variants on CUDA (max_rel_err ≤ 5.962e-5 vs CPU SWAR).
    ///
    /// Both branches now dispatch `launch_rowtiled8`. The 2.03× median CUDA
    /// speedup over `launch_plane` directly lifts the projection-only upper
    /// bound from ~10 tok/s to ~20 tok/s; combined with Issue 608 T3's dp4a
    /// integration, this is the path that closes the 1.95× gap to llama.cpp.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        // Issue 764 T2 (Bench 767/768): the f16-scale arm — same kernel body,
        // scales decoded in-kernel from raw f16 (bit-identical output, G1
        // PASS). Dispatched only when the runtime toggle is on AND the handle
        // carries the f16 buffer; the counter is the vacuous guard for the
        // e2e A/B (outputs are bit-identical, so timing alone cannot prove
        // the toggle reached the kernel).
        if let Some(scale_f16) = handle
            .group_scale_f16
            .as_ref()
            .filter(|_| f16_scale_enabled())
        {
            F16_SCALE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return unsafe {
                crate::gemv_ternary_scale_ab_cubecl::GemvTernaryScaleAbCubeCL::launch_f16scale::<R>(
                    client, handle, scale_f16, input_handle, output_handle,
                )
            };
        }
        // Both backends now dispatch width-8 row-tiled:
        //   - Metal: Issue 613 T1-follow-up, 1.155× over width-4 (19.38 vs
        //     16.78 tok/s projection-only, M3 Max).
        //   - CUDA: Issue 613 T2, 2.03× over stride (median ~19.7 vs ~9.7
        //     tok/s projection-only, RTX 4090 cubecl-cuda).
        // Packed stays out of `launch()` on both backends: 0.48× on CUDA
        // (Issue 607 G5a) and superseded by rowtiled on Metal (Issue 613
        // T1-follow-up). It remains the correctness reference + opt-in
        // escape hatch via `launch_packed`.
        unsafe {
            Self::launch_rowtiled8::<R>(client, handle, input_handle, output_handle);
        }
    }

    /// Launch the subset-sum ("LUT") ternary GEMV (Issue 606 T3c path 1).
    ///
    /// Dispatch geometry matches [`Self::launch_rowtiled`] — 32 rows per
    /// workgroup — because the LUT tiling is over the *activation* axis and does
    /// not change the row decomposition. `CubeDim` must stay 256 and `PLANE_DIM`
    /// 32: the kernel's build step assumes exactly one 4-activation group per
    /// thread, and its consume step exactly one u32 word per lane.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    #[cfg(feature = "ternary_lut_gemv")]
    pub unsafe fn launch_lut<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE; // 32

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_plane_lut::launch_unchecked::<R>(
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

    /// Launch the row-tiled word-packed ternary GEMV (Issue 606 T3).
    ///
    /// Each plane owns `TERNARY_ROWS_PER_PLANE` consecutive rows, so a
    /// workgroup covers `(wg_size / plane_size) * ROWS_PER_PLANE` = 32 rows.
    /// See [`gemv_ternary_plane_rowtiled`].
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_rowtiled<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE; // 32

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_plane_rowtiled::launch_unchecked::<R>(
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

    /// Launch the row-tiled ternary GEMV at tile width 8 (Issue 606 T3 sweep).
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_rowtiled8<R: Runtime>(
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
            gemv_ternary_plane_rowtiled8::launch_unchecked::<R>(
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

    /// Launch the interleaved 2-bit-code ternary GEMV (Issue 628 T1 layout A).
    ///
    /// Same dispatch geometry as [`Self::launch_rowtiled8`]; the difference is
    /// entirely inside the kernel — ONE code buffer (2 bits/weight) instead of
    /// TWO bit-plane buffers (1 bit/weight each). Tests the load-stream
    /// hypothesis: same total bytes, one coalesced stream instead of two.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_interleaved<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &InterleavedTernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE_8; // 64

        let m = handle.m as u32;
        let n = handle.n as u32;
        let words_per_row = handle.words_per_row as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let codes_len = handle.m * handle.words_per_row;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_interleaved_rowtiled8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.codes_u32.clone(), codes_len),
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

    /// Launch the base-3 trit-packed ternary GEMV (Issue 628 T2 layout B).
    ///
    /// Same dispatch geometry as [`Self::launch_interleaved`]; the difference is
    /// the weight format — base-3 trits (1.75 bits/weight, 5 per byte) instead
    /// of 2-bit codes (2.125 bits/weight). Tests the byte-reduction hypothesis:
    /// 17.6% fewer bytes → faster kernel on bandwidth-bound shapes.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_trit<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryTritHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32;
        let planes_per_wg = wg_size / plane_size; // 8
        let rows_per_wg = planes_per_wg * TERNARY_ROWS_PER_PLANE_8; // 64

        let m = handle.m as u32;
        let n = handle.n as u32;
        let words_per_row = handle.words_per_row as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let trits_len = handle.m * handle.words_per_row;
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_trit_rowtiled8::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.trits_u32.clone(), trits_len),
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
    ///
    /// Same dispatch geometry as [`Self::launch_plane`]; the difference is
    /// entirely inside the kernel (one u32 word per lane load instead of 32
    /// redundant loads of the same word). See [`gemv_ternary_plane_packed`].
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_packed<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Conservative: Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len; // same layout
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_plane_packed::launch_unchecked::<R>(
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

    /// Launch plane (subgroup) ternary dequant+GEMV kernel.
    ///
    /// Each plane handles one output row with cooperative dot product + `plane_sum()`.
    /// Workgroup: 256 threads → 8 planes (with plane_dim=32 on Metal).
    /// Dispatch: `ceil(m / 8)` workgroups.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch()`.
    pub unsafe fn launch_plane<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &TernaryHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Conservative: Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8

        let m = handle.m as u32;
        let n = handle.n as u32;
        let blocks64 = handle.blocks64 as u32;
        let groups_per_row = handle.groups_per_row as u32;

        let num_wg = m.div_ceil(rows_per_wg).max(1);

        let pos_len = handle.m * handle.blocks64 * 2;
        let neg_len = pos_len; // same layout
        let scale_len = handle.m * handle.groups_per_row;

        unsafe {
            gemv_ternary_plane::launch_unchecked::<R>(
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
}

// ---------------------------------------------------------------------------
// TernaryMatvecHook impl — the Issue 599 GPU unblock path
// ---------------------------------------------------------------------------

/// GPU-accelerated ternary matvec hook for the DeltaNet ternary forward.
///
/// Implements [`katgpt_core::TernaryMatvecHook`] by dispatching each matvec
/// to the CubeCL `gemv_ternary_plane` kernel on Metal/GPU.
///
/// # Handle caching
///
/// Each `TernaryGroupWeights` projection is uploaded to GPU once (on first
/// `matvec` call for that weight pointer) and cached by
/// `(pos_bits.as_ptr(), neg_bits.as_ptr())`. Subsequent calls hit the cache.
/// The weights themselves are immutable after model load, so pointer identity
/// is stable for the lifetime of the hook.
///
/// # Per-call overhead
///
/// Each `matvec` call: uploads the input vector (~20 KB for n_embd=5120),
/// launches the CubeCL kernel, downloads the output vector (~55 KB for
/// mlp_hidden=13824). Measured at ~3.2 ms p50 per dispatch for a 13824x5120
/// projection on M3 Max Metal (vs ~14.6 ms CPU SIMD) — a 4.5x speedup.
#[cfg(feature = "cubecl_runtime")]
pub struct GpuTernaryMatvec {
    client: ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    handles: std::sync::Mutex<std::collections::HashMap<(usize, usize), TernaryHandle>>,
}

#[cfg(feature = "cubecl_runtime")]
impl GpuTernaryMatvec {
    /// Create a GPU ternary matvec hook with the given CubeCL client.
    pub fn new(client: ComputeClient<crate::cubecl_runtime::ActiveRuntime>) -> Self {
        Self {
            client,
            handles: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Pre-upload a projection's weights to GPU (avoids lazy-upload on first call).
    pub fn preupload(&self, w: &TernaryGroupWeights) {
        let key = (w.pos_bits.as_ptr() as usize, w.neg_bits.as_ptr() as usize);
        let mut map = self.handles.lock().unwrap();
        map.entry(key)
            .or_insert_with(|| TernaryHandle::from_weights(&self.client, w));
    }

    /// Pre-upload ALL projections from a loaded model.
    /// Call once after loading the model, before the first forward pass.
    ///
    /// Generic over any type that exposes its projections as
    /// `TernaryGroupWeights` references. This avoids riir-gpu needing a dep
    /// on the ternary weight types (which would break the dense GPU forward
    /// dispatch via the `Proj::Ternary` variant).
    pub fn preupload_all<'a, I>(&self, projections: I)
    where
        I: IntoIterator<Item = &'a TernaryGroupWeights>,
    {
        for w in projections {
            self.preupload(w);
        }
    }

    fn get_or_upload(&self, w: &TernaryGroupWeights) -> TernaryHandle {
        let key = (w.pos_bits.as_ptr() as usize, w.neg_bits.as_ptr() as usize);
        let mut map = self.handles.lock().unwrap();
        map.entry(key)
            .or_insert_with(|| TernaryHandle::from_weights(&self.client, w))
            .clone()
    }
}

#[cfg(feature = "cubecl_runtime")]
impl katgpt_core::TernaryMatvecHook for GpuTernaryMatvec {
    fn matvec(&self, w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
        if w.rows == 0 {
            return;
        }
        debug_assert_eq!(x.len(), w.cols, "GpuTernaryMatvec input dim mismatch");
        debug_assert_eq!(y.len(), w.rows, "GpuTernaryMatvec output dim mismatch");

        let handle = self.get_or_upload(w);

        // Upload input + dispatch + download output
        let input_handle = self.client.create_from_slice(f32::as_bytes(x));
        let output_handle = self.client.empty(w.rows * core::mem::size_of::<f32>());

        unsafe {
            GemvTernaryCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &self.client,
                &handle,
                input_handle,
                output_handle.clone(),
            );
        }

        let bytes = self
            .client
            .read_one(output_handle)
            .expect("GpuTernaryMatvec: read output");
        let gpu_y = f32::from_bytes(&bytes);
        y.copy_from_slice(gpu_y);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

/// CPU-only tests for the block-contiguous preparation + GEMM reference (Issue 650).
/// These do NOT require GPU — they validate the buffer layout + indexing
/// that the GPU kernel will consume in Phase 2.
#[cfg(all(test, feature = "ternary_gemv"))]
mod block_contiguous_tests {
    use super::*;
    use katgpt_core::TernaryGroupWeights;

    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn filled(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut s = seed;
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = pseudo(&mut s);
                let q = match v {
                    v if v > 0.33 => 1i8,
                    v if v < -0.33 => -1i8,
                    _ => 0i8,
                };
                w.set(r, c, q);
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 0.5 + 0.25 * (g % 4) as f32);
            }
        }
        w
    }

    /// G1: block-contiguous GEMM CPU reference must match the SoA matvec
    /// for each token independently. This validates the buffer layout
    /// (`prepare_block_contiguous_u32`) and the indexing logic
    /// (`block_contiguous_gemm_cpu_ref`) that the GPU kernel will mirror.
    #[test]
    fn block_contiguous_gemm_matches_soa_per_token() {
        let shapes: &[(usize, usize)] = &[
            (48, 128),    // ssm shape
            (64, 256),    // small multi-group
            (128, 512),   // attn shape
            (256, 1024),  // medium projection
        ];
        let p_tokens = 4; // batch size

        for &(m, n) in shapes {
            let w = filled(m, n, 42 + m as u64);
            let block_buf = prepare_block_contiguous_u32(&w);

            // Random input batch [p_tokens, n].
            let mut seed = 100 + m as u64;
            let inputs: Vec<f32> = (0..p_tokens * n).map(|_| pseudo(&mut seed)).collect();

            // Block-contiguous GEMM.
            let mut output_bc = vec![0.0f32; p_tokens * m];
            block_contiguous_gemm_cpu_ref(
                &block_buf,
                &inputs,
                &mut output_bc,
                m,
                n,
                w.groups_per_row,
                p_tokens,
            );

            // SoA reference: one matvec per token.
            let mut output_soa = vec![0.0f32; p_tokens * m];
            for tok in 0..p_tokens {
                let x = &inputs[tok * n..(tok + 1) * n];
                let y = &mut output_soa[tok * m..(tok + 1) * m];
                katgpt_core::simd::simd_ternary_group_matvec(&w, x, y);
            }

            // Compare — allow small f32 rounding from different summation order.
            for i in 0..p_tokens * m {
                let rel_err =
                    (output_bc[i] - output_soa[i]).abs() / output_soa[i].abs().max(1e-6);
                assert!(
                    rel_err < 1e-5,
                    "block-contiguous vs SoA mismatch at shape ({m}×{n}), \
                     token batch index {i}: {bc} vs {soa} (rel_err {rel_err})",
                    bc = output_bc[i],
                    soa = output_soa[i],
                );
            }
        }
    }

    /// Block buffer size must match `m * groups_per_row * U32_PER_BLOCK_GROUP`.
    #[test]
    fn block_buffer_size_correct() {
        let w = filled(64, 512, 7);
        let buf = prepare_block_contiguous_u32(&w);
        assert_eq!(
            buf.len(),
            64 * w.groups_per_row * U32_PER_BLOCK_GROUP,
            "block buffer size must match expected layout"
        );
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::context::GpuContext;
    use crate::cubecl_runtime::ActiveRuntime;
    use katgpt_core::simd::simd_ternary_group_matvec;

    /// Run CubeCL ternary GEMV and return the result.
    fn run_ternary_gemv(w: &TernaryGroupWeights, input: &[f32]) -> Vec<f32> {
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();

        let handle = TernaryHandle::from_weights(&client, w);
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(w.rows * core::mem::size_of::<f32>());

        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &client,
                &handle,
                input_handle,
                output_handle.clone(),
            );
        }

        let bytes = client
            .read_one(output_handle)
            .expect("should read output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Build a small TernaryGroupWeights with known pattern.
    fn build_test_weights(rows: usize, cols: usize) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(rows, cols);
        // Deterministic fill: cycle +1, 0, -1, +1, 0, -1, ...
        let pattern = [1i8, 0, -1];
        for r in 0..rows {
            for c in 0..cols {
                let v = pattern[(r * cols + c) % 3];
                w.set(r, c, v);
            }
        }
        // Set a non-trivial group scale so the scale path is exercised.
        use half::f16;
        for r in 0..rows {
            for g in 0..w.groups_per_row {
                w.group_scale[r * w.groups_per_row + g] =
                    f16::from_f32(1.0 + 0.1 * (g as f32));
            }
        }
        w
    }

    #[test]
    fn test_ternary_gemv_matches_cpu_small() {
        // Small case: 8 rows × 128 cols (1 group per row, 2 blocks per row).
        let rows = 8;
        let cols = 128;
        let w = build_test_weights(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();

        let cpu_result = {
            let mut y = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &input, &mut y);
            y
        };
        let gpu_result = run_ternary_gemv(&w, &input);

        assert_eq!(gpu_result.len(), rows, "output length mismatch");
        for r in 0..rows {
            // Denominator clamped to 1.0 (matches katgpt-types SIMD-vs-scalar test pattern):
            // raw relative error explodes on near-zero rows where both paths agree to
            // ~5 significant figures but the denominator is tiny.
            let denom = cpu_result[r].abs().max(1.0);
            let rel_err = ((gpu_result[r] - cpu_result[r]).abs()) / denom;
            assert!(
                rel_err < 1e-4,
                "row {r}: cpu={} gpu={} rel_err={}",
                cpu_result[r],
                gpu_result[r],
                rel_err
            );
        }
    }

    #[test]
    fn test_ternary_gemv_matches_cpu_large() {
        // Larger case mirroring a real projection shape (trimmed for test speed):
        // 256 rows × 512 cols (4 groups per row, 8 blocks per row).
        let rows = 256;
        let cols = 512;
        let w = build_test_weights(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.01).sin()).collect();

        let cpu_result = {
            let mut y = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &input, &mut y);
            y
        };
        let gpu_result = run_ternary_gemv(&w, &input);

        assert_eq!(gpu_result.len(), rows, "output length mismatch");
        let mut max_rel = 0.0f32;
        for r in 0..rows {
            let denom = cpu_result[r].abs().max(1.0);
            let rel_err = ((gpu_result[r] - cpu_result[r]).abs()) / denom;
            max_rel = max_rel.max(rel_err);
        }
        assert!(
            max_rel < 1e-4,
            "max relative error {max_rel} exceeds 1e-4 tolerance"
        );
    }

    #[test]
    fn test_ternary_gemv_zero_weights() {
        // All-zero weights → all-zero output (both planes clear).
        let rows = 4;
        let cols = 128;
        let mut w = TernaryGroupWeights::new(rows, cols);
        // Default new() has all bits clear → all zeros. Set scale to nonzero
        // to confirm the zero-sign path produces 0 regardless.
        use half::f16;
        for r in 0..rows {
            for g in 0..w.groups_per_row {
                w.group_scale[r * w.groups_per_row + g] = f16::from_f32(2.5);
            }
        }
        let input = vec![1.0f32; cols];

        let gpu_result = run_ternary_gemv(&w, &input);

        for r in 0..rows {
            assert!(
                gpu_result[r].abs() < 1e-6,
                "row {r}: expected 0, got {}",
                gpu_result[r]
            );
        }
    }

    #[test]
    fn test_ternary_gemv_ragged_cols() {
        // Cols not a multiple of 128 → ragged final group. 200 cols = 1 full
        // group (128) + 1 ragged group (72 elements).
        let rows = 4;
        let cols = 200;
        let w = build_test_weights(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.05).collect();

        let cpu_result = {
            let mut y = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &input, &mut y);
            y
        };
        let gpu_result = run_ternary_gemv(&w, &input);

        for r in 0..rows {
            let denom = cpu_result[r].abs().max(1.0);
            let rel_err = ((gpu_result[r] - cpu_result[r]).abs()) / denom;
            assert!(
                rel_err < 1e-4,
                "row {r}: cpu={} gpu={} rel_err={}",
                cpu_result[r],
                gpu_result[r],
                rel_err
            );
        }
    }

    // ── Issue 642 F2: concatenated gate+up GEMV tests ──

    /// Verify TernaryHandle::from_two_weights concatenated GEMV matches
    /// running the two projections separately.
    #[test]
    fn test_gemv_concat_two_weights_matches_separate() {
        let ctx = GpuContext::new().expect("GpuContext should init");
        let client = ctx.cubecl_client();

        let rows = 16usize; // mlp per projection
        let cols = 256usize; // n_embd
        let w1 = build_test_weights(rows, cols);
        let w2 = build_test_weights(rows, cols);
        let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.01).sin()).collect();

        // Separate path: 2 GEMVs
        let gate_gpu = run_ternary_gemv(&w1, &input);
        let up_gpu = run_ternary_gemv(&w2, &input);

        // Concatenated path: 1 GEMV from from_two_weights
        let combined = TernaryHandle::from_two_weights(&client, &w1, &w2);
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty((2 * rows) * core::mem::size_of::<f32>());
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &client,
                &combined,
                input_handle,
                output_handle.clone(),
            );
        }
        let bytes = client.read_one(output_handle).expect("should read output");
        let combined_gpu = f32::from_bytes(&bytes);

        assert_eq!(combined_gpu.len(), 2 * rows);

        // First `rows` elements = gate; next `rows` = up
        let mut max_err = 0.0f32;
        for r in 0..rows {
            let gate_err = (combined_gpu[r] - gate_gpu[r]).abs();
            let up_err = (combined_gpu[rows + r] - up_gpu[r]).abs();
            max_err = max_err.max(gate_err).max(up_err);
            assert!(
                gate_err < 1e-5,
                "gate row {r}: separate={:.6} concat={:.6} err={:.2e}",
                gate_gpu[r],
                combined_gpu[r],
                gate_err
            );
            assert!(
                up_err < 1e-5,
                "up row {r}: separate={:.6} concat={:.6} err={:.2e}",
                up_gpu[r],
                combined_gpu[rows + r],
                up_err
            );
        }
        println!("concat two_weights GEMV ({rows}+{rows} rows × {cols}): max_err = {max_err}");
    }
}
