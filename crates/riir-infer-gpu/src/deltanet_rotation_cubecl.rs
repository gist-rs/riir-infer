//! Plan 602 Phase B (B1) — CubeCL/Metal Hadamard rotation kernels for the
//! Bonsai-2 folded-ternary runtime (the M3 mirror of Issue 980 T4's cudarc
//! lane, `deltanet_rotation_cudarc.rs`).
//!
//! The CPU contract lives in `riir-infer-core::deltanet::rotation`
//! (`rotate_forward_inplace` / `rotate_inverse_inplace` /
//! `permute_gdn_v_grouped_inplace`); every kernel here is a numeric twin of
//! those functions, dispatched through the CubeCL (Metal/wgpu) client so the
//! folded matmuls consume rotated activations GPU-resident. The rotation runs
//! between the RMSNorm and the int8 activation quantize — it commutes with
//! neither.
//!
//! Kernel set (only dispatched when the loaded model declares
//! `prism.hadamard`; the split, non-fused set — the cudarc K1–K5 fusion
//! escalation is a named follow-up rung in Plan 602 B5, not the baseline):
//!
//! | kernel | CPU twin | used at |
//! |---|---|---|
//! | `fwht_rotate_forward_f32` | `rotate_forward_inplace` | every folded matmul input (sign→FWHT) |
//! | `fwht_rotate_inverse_f32` | `rotate_inverse_inplace` | token-embedding lookup (FWHT→sign) |
//! | `fwht_forward_copy_f32` | memcpy + forward | whole-prefill staging (the T4-ALT twin) |
//! | `fwht_rotate_forward_f32_r4` / `fwht_forward_copy_f32_r4` | (radix-4 schedule of the two above) | Issue 1144: hblock 1024, 4 barriers instead of 11 — DEFAULT-ON, `RIIR_FWHT_R4=0` kill-switch |
//! | `gdn_v_permute_f32` | `permute_gdn_v_grouped_inplace` | `ssm_out` input (tiled→grouped heads) |
//! | `gdn_permute_rotate_f32` | (permute + `rotate_forward_inplace` fused) | Issue 1113 lever 1: GDN fold tail in ONE kernel |
//! | `swiglu_rotate_fused_f32` | (the cudarc K5 twin, f32 tail) | Issue 1113 lever 1: SwiGLU + sign + FWHT in ONE kernel |
//!
//! The two fused kernels are the CubeCL port of the cudarc lane's K-fusion
//! escalation (Plan 602 B5's named rung) with the quantize tail omitted — the
//! CubeCL GEMV consumes f32 activations directly. Both are bit-identical to
//! their split chains BY CONSTRUCTION (the gather/gating feeds the exact
//! incumbent butterfly; the only arithmetic beyond movement is the sign
//! multiply), so the only proof the toggle reached the kernel is the launch
//! counter (the Bench 768 vacuous guard). Default OFF
//! (`RIIR_GDN_PERMUTE_ROTATE_FUSED` / `RIIR_SWIGLU_ROTATE_FUSED`). ⚠ The
//! fuller variant that also folded the per-head RMSNorm+z-gate (the cudarc
//! K3 shape) was built and measured 1-ULP-divergent — the incumbent zgate's
//! compiled fma contraction does not reproduce across kernel compilations —
//! so the zgate stays its own bit-untouched launch.
//!
//! The single-vector and batched ([p × width]) forms are ONE kernel each:
//! signs index by `base % width`, so a decode-width vector (total == width)
//! and a prefill batch tile the same dispatch.
//!
//! Numeric parity: the FWHT applies the per-butterfly `1/√2` (the CPU/fork
//! unitary map), not a single end-stage `1/√n` — matching the CPU lane's
//! rounding shape and the cudarc twins' exact arithmetic (each element is
//! `(a ± b) * 1/√2`; no add-mul pair exists for the compiler to contract,
//! so the split is bit-stable on Metal).
//!
//! The butterfly body is deliberately INLINE in each FWHT kernel (three
//! mechanical copies): a `#[cube]` helper cannot take a `Shared` argument
//! (`Shared` is not `LaunchArg`), and a `macro_rules!` inside a `#[cube]`
//! body would expand after the cube pass. The three copies carry this
//! comment as the dedup marker.
//!
//! The dense escape-set GEMV (Bonsai-2's dense `ssm_alpha/beta`, 48×5120) is
//! deliberately NOT re-implemented here: `GemvCubeCL::launch_plane`
//! (`gemv_cubecl.rs`) is the same fp32 row-GEMV over dense weights, and the
//! Phase-B2 wiring consumes it directly (substrate-first — no second GEMV).

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
use cubecl::prelude::*;

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
use cubecl::server::Handle;

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
use riir_infer_core::deltanet::rotation::TernaryRotationConfig;

/// The FWHT kernels' shared-memory bound (one Hadamard block per cube,
/// 1024 f32 slots). `prism.hadamard` block sizes above this refuse at
/// [`RotationTablesCubeCL::build`] (Bonsai-2 is 1024).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub(crate) const MAX_FWHT_BLOCK: usize = 1024;

/// Threads (units) per cube — the repo-wide CubeCL dispatch width.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
const FWHT_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// In-place forward rotation on a folded matmul's input:
/// `x ← (1/√n)·H_n·(S⊙x)` per block — sign FIRST, Hadamard SECOND
/// (`rotate_forward_inplace`'s order). Signs index by `base % width`, so one
/// dispatch serves both the decode vector (total == width) and a prefill
/// `[p × width]` batch.
///
/// Dispatch: one cube per Hadamard block, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
// clippy: the butterfly constant must be a literal for the cube macro; it is
// exactly `FRAC_1_SQRT_2` at f32 precision (the cudarc twin's value).
#[allow(clippy::approx_constant)]
#[cube(launch_unchecked)]
fn fwht_rotate_forward_f32(x: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = x.len() as u32;
    let width = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let threads = 256u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * hblock;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let mut smem = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        smem[i as usize] = x[(base + i) as usize] * signs[(row_off + i) as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off = bi % half;
            let a = smem[(blk + off) as usize];
            let b = smem[(blk + off + half) as usize];
            smem[(blk + off) as usize] = (a + b) * inv_sqrt2;
            smem[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut j = tid;
    while j < hblock {
        x[(base + j) as usize] = smem[j as usize];
        j += threads;
    }
}

/// In-place inverse rotation for the token-embedding lookup result:
/// `z ← S⊙((1/√n)·H_n·z)` per block — Hadamard FIRST, sign SECOND
/// (`rotate_inverse_inplace`'s order; composing with the forward gives
/// `S·H·H·S = I` per block).
///
/// Dispatch: one cube per Hadamard block, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn fwht_rotate_inverse_f32(x: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = x.len() as u32;
    let width = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let threads = 256u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * hblock;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let mut smem = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        smem[i as usize] = x[(base + i) as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off = bi % half;
            let a = smem[(blk + off) as usize];
            let b = smem[(blk + off + half) as usize];
            smem[(blk + off) as usize] = (a + b) * inv_sqrt2;
            smem[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut j = tid;
    while j < hblock {
        x[(base + j) as usize] = smem[j as usize] * signs[(row_off + j) as usize];
        j += threads;
    }
}

/// Copy-rotate staging twin (the cudarc T4-ALT `fwht_forward_copy_batched`):
/// reads the PRIMAL `src`, writes the rotated `dst` — one memory pass where
/// copy + in-place rotate costs two, and no `&mut` alias on the destination.
/// Mathematically identical to copy + `fwht_rotate_forward_f32` (sign first,
/// Hadamard second).
///
/// Dispatch: one cube per Hadamard block, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn fwht_forward_copy_f32(src: &[f32], dst: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = dst.len() as u32;
    let width = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let threads = 256u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * hblock;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let mut smem = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        smem[i as usize] = src[(base + i) as usize] * signs[(row_off + i) as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off = bi % half;
            let a = smem[(blk + off) as usize];
            let b = smem[(blk + off + half) as usize];
            smem[(blk + off) as usize] = (a + b) * inv_sqrt2;
            smem[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut j = tid;
    while j < hblock {
        dst[(base + j) as usize] = smem[j as usize];
        j += threads;
    }
}

/// Issue 1144 — radix-4 schedule of [`fwht_rotate_forward_f32`] at hblock == 1024.
///
/// The incumbent runs a fill pass then 10 butterfly stages, each behind a
/// `sync_cube` (11 barriers). Here each of the 256 threads owns one quad per
/// pass and runs TWO stages on it in registers — the lowest pair straight
/// from global (no fill barrier), the highest pair straight to global (no
/// write-back barrier), the middle three ping-ponging two smem halves — so
/// 1024 = 4^5 needs 5 passes and 4 barriers. Every element sees the
/// same `(a ± b) * inv_sqrt2` at the same half, in the same order, from the
/// same operands: bit-identical by construction
/// (`fwht_radix4_matches_incumbent_bitexact` + the CPU-oracle tests).
/// INLINE butterfly (dedup marker: fwht_butterfly_copy — radix-4 form).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn fwht_rotate_forward_f32_r4(x: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = x.len() as u32;
    let width = params[0usize] as u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index. The
    // launcher only selects this kernel at hblock == 1024 == 4·threads.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * 1024u32;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut smem = Shared::<[f32]>::new_slice(2 * MAX_FWHT_BLOCK);
    // Contraction fence: Metal compiles with fast math, so a product that
    // feeds the next stage's add IN A REGISTER may fuse into an fma (the
    // incumbent never sees that — every stage round-trips smem). `select`
    // on a runtime-true predicate (width > 0) the compiler cannot fold
    // keeps each product a rounded f32 value.
    let live = width > 0u32;
    let zero = f32::new(0.0f32);

    // Pass 1 (halves 1, 2): the thread's contiguous quad, straight from
    // global — sign first, then the two lowest butterfly stages in registers.
    let q = tid * 4u32;
    let mut v0 = select(
        live,
        x[(base + q) as usize] * signs[(row_off + q) as usize],
        zero,
    );
    let mut v1 = select(
        live,
        x[(base + q + 1u32) as usize] * signs[(row_off + q + 1u32) as usize],
        zero,
    );
    let mut v2 = select(
        live,
        x[(base + q + 2u32) as usize] * signs[(row_off + q + 2u32) as usize],
        zero,
    );
    let mut v3 = select(
        live,
        x[(base + q + 3u32) as usize] * signs[(row_off + q + 3u32) as usize],
        zero,
    );
    let mut t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
    let mut t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
    let mut t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
    let mut t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
    smem[q as usize] = (t0 + t2) * inv_sqrt2;
    smem[(q + 2u32) as usize] = (t0 - t2) * inv_sqrt2;
    smem[(q + 1u32) as usize] = (t1 + t3) * inv_sqrt2;
    smem[(q + 3u32) as usize] = (t1 - t3) * inv_sqrt2;
    sync_cube();

    // Passes 2-4 (halves 4·8, 16·32, 64·128): quad {e, e+h, e+2h, e+3h}.
    // Ping-pong halves of `smem` (read `rd`, write `wr`), so each pass needs
    // ONE barrier — no thread overwrites a slot another thread still reads.
    let mut rd = 0u32;
    let mut h = 4u32;
    while h < 256u32 {
        let wr = 1024u32 - rd;
        let e = (tid / h) * (4u32 * h) + tid % h;
        v0 = smem[(rd + e) as usize];
        v1 = smem[(rd + e + h) as usize];
        v2 = smem[(rd + e + 2u32 * h) as usize];
        v3 = smem[(rd + e + 3u32 * h) as usize];
        t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
        t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
        t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
        t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
        smem[(wr + e) as usize] = (t0 + t2) * inv_sqrt2;
        smem[(wr + e + 2u32 * h) as usize] = (t0 - t2) * inv_sqrt2;
        smem[(wr + e + h) as usize] = (t1 + t3) * inv_sqrt2;
        smem[(wr + e + 3u32 * h) as usize] = (t1 - t3) * inv_sqrt2;
        sync_cube();
        rd = wr;
        h = h * 4u32;
    }

    // Pass 5 (halves 256, 512): quad {tid + k·256} — straight to global.
    v0 = smem[(rd + tid) as usize];
    v1 = smem[(rd + tid + 256u32) as usize];
    v2 = smem[(rd + tid + 512u32) as usize];
    v3 = smem[(rd + tid + 768u32) as usize];
    t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
    t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
    t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
    t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
    x[(base + tid) as usize] = (t0 + t2) * inv_sqrt2;
    x[(base + tid + 512u32) as usize] = (t0 - t2) * inv_sqrt2;
    x[(base + tid + 256u32) as usize] = (t1 + t3) * inv_sqrt2;
    x[(base + tid + 768u32) as usize] = (t1 - t3) * inv_sqrt2;
}

/// Issue 1144 — radix-4 schedule of [`fwht_forward_copy_f32`] at hblock == 1024.
///
/// The incumbent runs a fill pass then 10 butterfly stages, each behind a
/// `sync_cube` (11 barriers). Here each of the 256 threads owns one quad per
/// pass and runs TWO stages on it in registers — the lowest pair straight
/// from global (no fill barrier), the highest pair straight to global (no
/// write-back barrier), the middle three ping-ponging two smem halves — so
/// 1024 = 4^5 needs 5 passes and 4 barriers. Every element sees the
/// same `(a ± b) * inv_sqrt2` at the same half, in the same order, from the
/// same operands: bit-identical by construction
/// (`fwht_radix4_matches_incumbent_bitexact` + the CPU-oracle tests).
/// INLINE butterfly (dedup marker: fwht_butterfly_copy — radix-4 form).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn fwht_forward_copy_f32_r4(src: &[f32], dst: &mut [f32], signs: &[f32], params: &[f32]) {
    let total = dst.len() as u32;
    let width = params[0usize] as u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index. The
    // launcher only selects this kernel at hblock == 1024 == 4·threads.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * 1024u32;
    if base >= total {
        terminate!();
    }
    let row_off = base % width;
    let tid = UNIT_POS;
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut smem = Shared::<[f32]>::new_slice(2 * MAX_FWHT_BLOCK);
    // Contraction fence: Metal compiles with fast math, so a product that
    // feeds the next stage's add IN A REGISTER may fuse into an fma (the
    // incumbent never sees that — every stage round-trips smem). `select`
    // on a runtime-true predicate (width > 0) the compiler cannot fold
    // keeps each product a rounded f32 value.
    let live = width > 0u32;
    let zero = f32::new(0.0f32);

    // Pass 1 (halves 1, 2): the thread's contiguous quad, straight from
    // global — sign first, then the two lowest butterfly stages in registers.
    let q = tid * 4u32;
    let mut v0 = select(
        live,
        src[(base + q) as usize] * signs[(row_off + q) as usize],
        zero,
    );
    let mut v1 = select(
        live,
        src[(base + q + 1u32) as usize] * signs[(row_off + q + 1u32) as usize],
        zero,
    );
    let mut v2 = select(
        live,
        src[(base + q + 2u32) as usize] * signs[(row_off + q + 2u32) as usize],
        zero,
    );
    let mut v3 = select(
        live,
        src[(base + q + 3u32) as usize] * signs[(row_off + q + 3u32) as usize],
        zero,
    );
    let mut t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
    let mut t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
    let mut t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
    let mut t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
    smem[q as usize] = (t0 + t2) * inv_sqrt2;
    smem[(q + 2u32) as usize] = (t0 - t2) * inv_sqrt2;
    smem[(q + 1u32) as usize] = (t1 + t3) * inv_sqrt2;
    smem[(q + 3u32) as usize] = (t1 - t3) * inv_sqrt2;
    sync_cube();

    // Passes 2-4 (halves 4·8, 16·32, 64·128): quad {e, e+h, e+2h, e+3h}.
    // Ping-pong halves of `smem` (read `rd`, write `wr`), so each pass needs
    // ONE barrier — no thread overwrites a slot another thread still reads.
    let mut rd = 0u32;
    let mut h = 4u32;
    while h < 256u32 {
        let wr = 1024u32 - rd;
        let e = (tid / h) * (4u32 * h) + tid % h;
        v0 = smem[(rd + e) as usize];
        v1 = smem[(rd + e + h) as usize];
        v2 = smem[(rd + e + 2u32 * h) as usize];
        v3 = smem[(rd + e + 3u32 * h) as usize];
        t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
        t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
        t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
        t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
        smem[(wr + e) as usize] = (t0 + t2) * inv_sqrt2;
        smem[(wr + e + 2u32 * h) as usize] = (t0 - t2) * inv_sqrt2;
        smem[(wr + e + h) as usize] = (t1 + t3) * inv_sqrt2;
        smem[(wr + e + 3u32 * h) as usize] = (t1 - t3) * inv_sqrt2;
        sync_cube();
        rd = wr;
        h = h * 4u32;
    }

    // Pass 5 (halves 256, 512): quad {tid + k·256} — straight to global.
    v0 = smem[(rd + tid) as usize];
    v1 = smem[(rd + tid + 256u32) as usize];
    v2 = smem[(rd + tid + 512u32) as usize];
    v3 = smem[(rd + tid + 768u32) as usize];
    t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
    t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
    t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
    t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
    dst[(base + tid) as usize] = (t0 + t2) * inv_sqrt2;
    dst[(base + tid + 512u32) as usize] = (t0 - t2) * inv_sqrt2;
    dst[(base + tid + 256u32) as usize] = (t1 + t3) * inv_sqrt2;
    dst[(base + tid + 768u32) as usize] = (t1 - t3) * inv_sqrt2;
}

/// `gdn_v_grouped` head permute, tiled `[hd, nk, rep]` → grouped
/// `[hd, rep, nk]` (`permute_gdn_v_grouped_inplace`'s whole-head-block
/// gather, expressed per element). `tmp` is the source copy (the caller
/// stages `x` into it first — no in-place race); `x` is the destination.
/// Batched by construction: the row index is `i / v_dim`.
///
/// Dispatch: `ceil(total/256)` cubes, `CubeDim::new_1d(256)`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[cube(launch_unchecked)]
fn gdn_v_permute_f32(tmp: &[f32], x: &mut [f32], params: &[f32]) {
    let total = x.len();
    let v_dim = params[0usize] as usize;
    let hd = params[1usize] as usize;
    let n_k = params[2usize] as usize;
    let rep = params[3usize] as usize;
    // 2-D grid (see `split_grid`): rebuild the linear unit index.
    let i = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM + UNIT_POS) as usize;
    if i >= total {
        terminate!();
    }
    let row = i / v_dim;
    let j = i % v_dim;
    let head = j / hd; // grouped head index: nk * rep + r
    let off = j % hd;
    let nk = head / rep;
    let r = head % rep;
    let src_head = r * n_k + nk; // tiled head index
    x[i] = tmp[row * v_dim + src_head * hd + off];
}

/// Issue 1113 lever 1 — the GDN-output fold tail in ONE kernel: tiled→grouped
/// head permute (a gather — bit-neutral movement) + sign + FWHT, reading the
/// post-zgate `recurrent_out` at the SOURCE head and writing the rotated
/// grouped vector to `out` (the wiring passes `permute_tmp`, which the split
/// chain needed only for its own staging copy). Replaces Copy +
/// gdn_v_permute + fwht_rotate_forward (3 launches → 1; −2/GDN layer).
///
/// Bit-identity is BY CONSTRUCTION, not by modeled arithmetic: the split
/// chain's copy/permute are movement, the sign is a single multiply, and the
/// butterfly is this file's `fwht_rotate_forward_f32` body verbatim (the K5
/// fused test pins that shape against the incumbent kernel). The full-K3
/// variant that ALSO folded the per-head RMSNorm + z-gate was built first and
/// measured 1-ULP-divergent against the split chain — the incumbent zgate's
/// compiled arithmetic carries a compiler contraction (fma-class) that a
/// second kernel's compilation does not reproduce (the exact-statement-order
/// CPU mirror agrees with the fused kernel, not the incumbent) — so the
/// zgate stays its own bit-untouched launch and only the movement+rotation
/// tail fuses.
///
/// One cube per Hadamard window over the DESTINATION (grouped) indices; the
/// head geometry never constrains the window (heads may straddle it — the
/// gather resolves any position).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn gdn_permute_rotate_f32(
    src: &[f32],     // [v_dim] post-zgate recurrent_out (tiled/source head order)
    signs: &[f32],   // [v_dim]
    out: &mut [f32], // [v_dim] rotated grouped output (must not alias src)
    params: &[f32],  // [head_dim, n_k, rep, hblock]
) {
    let head_dim = params[0usize] as u32;
    let n_k = params[1usize] as u32;
    let rep = params[2usize] as u32;
    let hblock = params[3usize] as u32;
    let total = out.len() as u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * hblock;
    if base >= total {
        terminate!();
    }
    let tid = UNIT_POS;
    let threads = 256u32;
    let mut buf = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    // Gather the window's source values (tiled head S = r·n_k + nk per
    // grouped head H) and apply the sign at the DESTINATION index — the
    // split chain's permute + fwht sign step, fused.
    let mut i = tid;
    while i < hblock {
        let gj = base + i;
        let off = gj % head_dim;
        let h = gj / head_dim;
        let nk = h / rep;
        let r = h % rep;
        let s = ((r * n_k + nk) * head_dim + off) as usize;
        buf[i as usize] = src[s] * signs[gj as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off2 = bi % half;
            let a = buf[(blk + off2) as usize];
            let b = buf[(blk + off2 + half) as usize];
            buf[(blk + off2) as usize] = (a + b) * inv_sqrt2;
            buf[(blk + off2 + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut k = tid;
    while k < hblock {
        out[(base + k) as usize] = buf[k as usize];
        k += threads;
    }
}

/// K5 — SwiGLU + sign + FWHT in ONE kernel (Issue 1113 lever 1 rung 2; the
/// CubeCL twin of the cudarc `swiglu_sign_fwht_quantize_f32`, f32 tail): reads
/// the concatenated `[gate | up]` buffer, computes `g·σ(g)·u` at the
/// destination window element, applies the sign, and rotates — writing
/// `ffn_hidden` directly (the split chain's DeltanetGatingConcat wrote it from
/// `ffn_gate_up` too, so the buffers differ and there is no staging race).
/// Replaces DeltanetGatingConcat + fwht_rotate_forward (2 launches → 1).
/// Bit identity: `(g·σ(g)·u)` is the gating kernel's expression verbatim, then
/// the rotation's `x·sign`, then its exact butterfly.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
#[allow(clippy::approx_constant)] // FRAC_1_SQRT_2 at f32 precision, literal for the cube macro
#[cube(launch_unchecked)]
fn swiglu_rotate_fused_f32(
    gate_up: &[f32], // [2*n] concatenated gate | up
    out: &mut [f32], // [n] rotated SwiGLU output
    signs: &[f32],   // [n]
    params: &[f32],  // [n, hblock]
) {
    let n = params[0usize] as u32;
    let hblock = params[1usize] as u32;
    let total = out.len() as u32;
    // 2-D grid (see `split_grid`): rebuild the linear cube index.
    let base = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * hblock;
    if base >= total {
        terminate!();
    }
    let tid = UNIT_POS;
    let threads = 256u32;
    let mut buf = Shared::<[f32]>::new_slice(MAX_FWHT_BLOCK);
    let mut i = tid;
    while i < hblock {
        let j = base + i;
        let g = gate_up[j as usize];
        let neg_g = f32::new(0.0f32) - g;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
        let u = gate_up[(n + j) as usize];
        buf[i as usize] = g * sig * u * signs[j as usize];
        i += threads;
    }
    sync_cube();
    // INLINE butterfly (dedup marker: fwht_butterfly_copy)
    let inv_sqrt2 = f32::new(0.70710678f32);
    let mut step = 2u32;
    while step <= hblock {
        let half = step / 2u32;
        let mut bi = tid;
        while bi < hblock / 2u32 {
            let blk = (bi / half) * step;
            let off2 = bi % half;
            let a = buf[(blk + off2) as usize];
            let b = buf[(blk + off2 + half) as usize];
            buf[(blk + off2) as usize] = (a + b) * inv_sqrt2;
            buf[(blk + off2 + half) as usize] = (a - b) * inv_sqrt2;
            bi += threads;
        }
        sync_cube();
        step = step * 2u32;
    }
    let mut k = tid;
    while k < hblock {
        out[(base + k) as usize] = buf[k as usize];
        k += threads;
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

/// CubeCL (Metal) Hadamard-rotation launchers — the split-kernel set only
/// (the fused K1–K5 escalation is Plan 602 B5's named follow-up rung).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub struct RotationCubeCL;

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
impl RotationCubeCL {
    /// In-place forward rotation over `total` elements (a width vector or a
    /// `[p × width]` batch).
    ///
    /// # Safety
    ///
    /// - `x_handle`: `total` f32 elements (mutated in place)
    /// - `signs_handle`: `width` f32 elements (+1.0/-1.0)
    /// - `total` must be a multiple of `hblock`; `hblock` a power of two
    ///   `<= 1024`; `width` a multiple of `hblock`
    pub unsafe fn launch_forward<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        signs_handle: Handle,
        total: usize,
        width: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[width as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (total / hblock.max(1)) as u32;
        // Issue 1144: the radix-4 schedule at hblock 1024 (bit-identical).
        let kernel = match hblock == MAX_FWHT_BLOCK && fwht_r4_enabled() {
            true => fwht_rotate_forward_f32_r4::launch_unchecked::<R>,
            false => fwht_rotate_forward_f32::launch_unchecked::<R>,
        };
        unsafe {
            kernel(
                client,
                split_grid(n_cubes),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(x_handle, total),
                BufferArg::from_raw_parts(signs_handle, width),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// In-place inverse rotation (the embedding-lookup twin).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_forward`].
    pub unsafe fn launch_inverse<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        signs_handle: Handle,
        total: usize,
        width: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[width as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (total / hblock.max(1)) as u32;
        unsafe {
            fwht_rotate_inverse_f32::launch_unchecked::<R>(
                client,
                split_grid(n_cubes),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(x_handle, total),
                BufferArg::from_raw_parts(signs_handle, width),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// Copy-rotate staging: reads `src_handle` (primal), writes `dst_handle`
    /// (rotated) — `memcpy + launch_forward` in one memory pass.
    ///
    /// # Safety
    ///
    /// - `src_handle` / `dst_handle`: `total` f32 elements each
    /// - rest as [`Self::launch_forward`]
    pub unsafe fn launch_forward_copy<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        dst_handle: Handle,
        signs_handle: Handle,
        total: usize,
        width: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[width as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (total / hblock.max(1)) as u32;
        // Issue 1144: the radix-4 schedule at hblock 1024 (bit-identical).
        let kernel = match hblock == MAX_FWHT_BLOCK && fwht_r4_enabled() {
            true => fwht_forward_copy_f32_r4::launch_unchecked::<R>,
            false => fwht_forward_copy_f32::launch_unchecked::<R>,
        };
        unsafe {
            kernel(
                client,
                split_grid(n_cubes),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(src_handle, total),
                BufferArg::from_raw_parts(dst_handle, total),
                BufferArg::from_raw_parts(signs_handle, width),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// Issue 1113 lever 1 — fused GDN-output fold tail (gather + sign +
    /// FWHT; Copy + gdn_v_permute + fwht_forward in one launch). Guarded by
    /// [`gdn_permute_rotate_ok`].
    ///
    /// # Safety
    ///
    /// - `src_handle` / `out_handle`: `v_dim` f32 elements each (`out` must
    ///   NOT alias `src` — the wiring passes `permute_tmp`)
    /// - `signs_handle`: `v_dim` f32
    /// - [`gdn_permute_rotate_ok`]`(v_dim, hblock)` must hold
    pub unsafe fn launch_gdn_permute_rotate<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        signs_handle: Handle,
        out_handle: Handle,
        v_dim: usize,
        head_dim: usize,
        n_k: usize,
        rep: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[head_dim as f32, n_k as f32, rep as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (v_dim / hblock.max(1)) as u32;
        GDN_PERMUTE_ROTATE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            gdn_permute_rotate_f32::launch_unchecked::<R>(
                client,
                split_grid(n_cubes),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(src_handle, v_dim),
                BufferArg::from_raw_parts(signs_handle, v_dim),
                BufferArg::from_raw_parts(out_handle, v_dim),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }

    /// K5 fused SwiGLU + sign + FWHT (Issue 1113 lever 1 rung 2). Guarded by
    /// [`swiglu_rotate_fused_ok`].
    ///
    /// # Safety
    ///
    /// - `gate_up_handle`: `2 * n` f32; `out_handle`: `n` f32 (must NOT alias
    ///   `gate_up`)
    /// - `signs_handle`: `n` f32
    /// - [`swiglu_rotate_fused_ok`]`(n, hblock)` must hold
    pub unsafe fn launch_swiglu_rotate_fused<R: Runtime>(
        client: &ComputeClient<R>,
        gate_up_handle: Handle,
        signs_handle: Handle,
        out_handle: Handle,
        n: usize,
        hblock: usize,
    ) {
        let params: &[f32] = &[n as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_cubes = (n / hblock.max(1)) as u32;
        SWIGLU_ROTATE_FUSED_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            swiglu_rotate_fused_f32::launch_unchecked::<R>(
                client,
                split_grid(n_cubes),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(gate_up_handle, 2 * n),
                BufferArg::from_raw_parts(out_handle, n),
                BufferArg::from_raw_parts(signs_handle, n),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// `gdn_v_grouped` tiled→grouped head permute over a `[p × v_dim]`
    /// batch (`p == 1` for decode). `tmp_handle` is the staged source copy
    /// of `x`; `x_handle` is the in-place destination.
    ///
    /// # Safety
    ///
    /// - `tmp_handle` / `x_handle`: `total = p * v_dim` f32 elements each
    /// - `v_dim == n_v_heads * head_dim`; `rep == n_v_heads / n_k_groups`
    pub unsafe fn launch_gdn_v_permute<R: Runtime>(
        client: &ComputeClient<R>,
        tmp_handle: Handle,
        x_handle: Handle,
        total: usize,
        v_dim: usize,
        hd: usize,
        n_k: usize,
        rep: usize,
    ) {
        let params: &[f32] = &[v_dim as f32, hd as f32, n_k as f32, rep as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = total.div_ceil(FWHT_THREADS as usize) as u32;
        unsafe {
            gdn_v_permute_f32::launch_unchecked::<R>(
                client,
                split_grid(n_wg),
                CubeDim::new_1d(FWHT_THREADS),
                BufferArg::from_raw_parts(tmp_handle, total),
                BufferArg::from_raw_parts(x_handle, total),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

/// Per-dimension workgroup-count ceiling (wgpu/Metal `maxComputeWorkgroupsPerDimension`).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
const MAX_GRID_DIM: u32 = 65_535;

/// Lay `n` cubes out as `(x, y)` with `x ≤ MAX_GRID_DIM`; the kernels rebuild
/// the linear index as `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` and the
/// overshoot cubes (`x * y > n`) fall to their existing bounds guard.
///
/// A 1-D `n` was fine for decode (`p == 1`) and broke batch prefill: at
/// P = 4096 over `v_dim` 6144 the permute alone is 98 304 workgroups
/// (Issue 1004 R1's Bonsai-2 e2e run, wgpu validation error).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
fn split_grid(n: u32) -> CubeCount {
    let n = n.max(1);
    let x = n.min(MAX_GRID_DIM);
    CubeCount::Static(x, n.div_ceil(x), 1)
}

// ---------------------------------------------------------------------------
// K-fusion runtime toggles (the Bench 768 pattern: env default OFF, a force
// setter for the bench harness, and a launch counter as the vacuous guard —
// the fused outputs are bit-identical to the split chains, so ONLY the
// counter proves the toggle reached the kernel)
// ---------------------------------------------------------------------------

/// Issue 1144: the forward FWHT launchers take the radix-4 kernels at
/// hblock 1024 (bit-identical) unless `RIIR_FWHT_R4` is set to anything but
/// 1/on/true — the `dense_pair_env_semantics` contract (unset = ON).
/// [`set_fwht_r4`] is the harness toggle.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static USE_FWHT_R4: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static FWHT_R4_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
fn fwht_r4_enabled() -> bool {
    if FWHT_R4_INITIALIZED
        .set(crate::gemv_cubecl::dense_pair_env_semantics(std::env::var(
            "RIIR_FWHT_R4",
        )))
        .is_ok()
    {
        USE_FWHT_R4.store(
            FWHT_R4_INITIALIZED.get().copied().unwrap_or(true),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_FWHT_R4.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the radix-4 FWHT schedule on/off (overrides `RIIR_FWHT_R4`).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn set_fwht_r4(on: bool) {
    let _ = FWHT_R4_INITIALIZED.set(on);
    USE_FWHT_R4.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// K3 (GDN fold tail: gather+sign+FWHT) armed: env `RIIR_GDN_PERMUTE_ROTATE_FUSED`
/// ("1"/"on"/"true"), else OFF; [`set_gdn_permute_rotate_fused`] overrides.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static USE_GDN_PERMUTE_ROTATE_FUSED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static GDN_PERMUTE_ROTATE_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static GDN_PERMUTE_ROTATE_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// K5 (SwiGLU + rotate) armed: env `RIIR_SWIGLU_ROTATE_FUSED`,
/// else OFF; [`set_swiglu_rotate_fused`] overrides.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static USE_SWIGLU_ROTATE_FUSED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static SWIGLU_ROTATE_FUSED_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
static SWIGLU_ROTATE_FUSED_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "on" | "true"
    )
}

/// Force the fused GDN fold-tail path on/off (overrides the env; the bench
/// harness's arm toggle).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn set_gdn_permute_rotate_fused(on: bool) {
    let _ = GDN_PERMUTE_ROTATE_INITIALIZED.set(on);
    USE_GDN_PERMUTE_ROTATE_FUSED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Force the K5 SwiGLU+rotate fused path on/off (overrides the env).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn set_swiglu_rotate_fused(on: bool) {
    let _ = SWIGLU_ROTATE_FUSED_INITIALIZED.set(on);
    USE_SWIGLU_ROTATE_FUSED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the fused GDN fold-tail path is currently armed (dispatch check).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn gdn_permute_rotate_fused_enabled() -> bool {
    if GDN_PERMUTE_ROTATE_INITIALIZED
        .set(env_flag("RIIR_GDN_PERMUTE_ROTATE_FUSED"))
        .is_ok()
    {
        USE_GDN_PERMUTE_ROTATE_FUSED.store(
            GDN_PERMUTE_ROTATE_INITIALIZED
                .get()
                .copied()
                .unwrap_or(false),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_GDN_PERMUTE_ROTATE_FUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the K5 fused SwiGLU+rotate path is currently armed (dispatch check).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn swiglu_rotate_fused_enabled() -> bool {
    if SWIGLU_ROTATE_FUSED_INITIALIZED
        .set(env_flag("RIIR_SWIGLU_ROTATE_FUSED"))
        .is_ok()
    {
        USE_SWIGLU_ROTATE_FUSED.store(
            SWIGLU_ROTATE_FUSED_INITIALIZED
                .get()
                .copied()
                .unwrap_or(false),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_SWIGLU_ROTATE_FUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Launches dispatched through the fused GDN fold-tail path (the vacuous guard).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn gdn_permute_rotate_fused_launch_count() -> usize {
    GDN_PERMUTE_ROTATE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Launches dispatched through the K5 fused path (the vacuous guard).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn swiglu_rotate_fused_launch_count() -> usize {
    SWIGLU_ROTATE_FUSED_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fused GDN fold-tail geometry guard: whole Hadamard windows over the
/// grouped vector (the head mapping never constrains the window — the gather
/// resolves any position).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn gdn_permute_rotate_ok(v_dim: usize, hblock: usize) -> bool {
    v_dim.is_multiple_of(hblock) && (2..=MAX_FWHT_BLOCK).contains(&hblock)
}

/// K5 fused-kernel geometry guard: whole Hadamard windows over the SwiGLU
/// output width.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub fn swiglu_rotate_fused_ok(n: usize, hblock: usize) -> bool {
    n.is_multiple_of(hblock) && (2..=MAX_FWHT_BLOCK).contains(&hblock)
}

// ---------------------------------------------------------------------------
// Sign tables
// ---------------------------------------------------------------------------

/// GPU-resident rotation tables for one folded model: the width-keyed sign
/// vectors uploaded as f32 (+1.0/-1.0) plus the parsed geometry — the
/// CubeCL mirror of the cudarc lane's `RotationTables`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub struct RotationTablesCubeCL {
    pub block_size: usize,
    pub signs: Vec<(usize, Handle)>,
    pub gdn_v_grouped: bool,
    pub gdn_v_heads: usize,
    pub gdn_k_groups: usize,
    pub inverse_embedding: bool,
}

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
impl RotationTablesCubeCL {
    /// Build from the parsed CPU config: upload the sign vectors as f32.
    /// Errors (not panics) on a block size this kernel set cannot serve —
    /// the loader already refused non-power-of-2 sizes, this bounds the
    /// shared-memory window.
    pub fn build<R: Runtime>(
        client: &ComputeClient<R>,
        cfg: &TernaryRotationConfig,
    ) -> Result<Self, String> {
        if cfg.block_size > MAX_FWHT_BLOCK || !cfg.block_size.is_power_of_two() {
            return Err(format!(
                "prism.hadamard block_size {} exceeds the CubeCL FWHT kernel's power-of-2 \
                 shared-memory bound ({MAX_FWHT_BLOCK})",
                cfg.block_size
            ));
        }
        let mut signs = Vec::with_capacity(cfg.signs.len());
        for (width, v) in &cfg.signs {
            let row: Vec<f32> = v.iter().map(|&s| s as f32).collect();
            debug_assert_eq!(row.len(), *width);
            signs.push((*width, client.create_from_slice(f32::as_bytes(&row))));
        }
        Ok(Self {
            block_size: cfg.block_size,
            signs,
            gdn_v_grouped: cfg.gdn_v_grouped,
            gdn_v_heads: cfg.gdn_v_heads,
            gdn_k_groups: cfg.gdn_k_groups,
            inverse_embedding: cfg.inverse_embedding,
        })
    }

    /// The sign vector for a folded input width (the loader validated the
    /// key set — a miss here is a wiring bug, not a model state).
    pub fn signs_for_width(&self, width: usize) -> &Handle {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, s)| s)
            .unwrap_or_else(|| panic!("rotation sign table missing width {width} (loader validated the set — this is a wiring bug)"))
    }
}

// ---------------------------------------------------------------------------
// Tests — every kernel against its CPU twin (Plan 602 B1's G1 shape)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime", feature = "ternary_gemv"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use riir_infer_core::deltanet::rotation::{
        permute_gdn_v_grouped_inplace, rotate_forward_inplace,
    };

    /// Deterministic LCG — mixed-magnitude signed values, no external deps
    /// (the cudarc twin's generator, so fixtures match across lanes).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((self.0 >> 33) as f32) / (u32::MAX >> 1) as f32;
            lo + bits * (hi - lo)
        }
    }

    fn signs_vec(n: usize, seed: u64) -> Vec<i8> {
        let mut lcg = Lcg(seed);
        (0..n)
            .map(|_| {
                if lcg.next_f32(-1.0, 1.0) >= 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    }

    /// Forward rotation: GPU must be BIT-IDENTICAL to the CPU twin — the
    /// per-butterfly `(a±b)·1/√2` shape leaves no contraction room, and B4's
    /// Metal-vs-CPU parity gate inherits this assertion's shape.
    #[test]
    fn fwht_forward_matches_cpu_bitexact() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        for &(width, hblock) in &[(5120usize, 1024usize), (6144, 1024), (5120, 512)] {
            let mut lcg = Lcg(0x602 + width as u64);
            let x: Vec<f32> = (0..width).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
            let signs = signs_vec(width, 0x5117 + width as u64);

            let mut cpu = x.clone();
            rotate_forward_inplace(&mut cpu, Some(&signs), hblock);

            let x_handle = client.create_from_slice(f32::as_bytes(&x));
            let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
            let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
            unsafe {
                RotationCubeCL::launch_forward::<ActiveRuntime>(
                    &client,
                    x_handle.clone(),
                    signs_handle,
                    width,
                    width,
                    hblock,
                );
            }
            let bytes = client.read_one(x_handle).expect("read rotated");
            let gpu = f32::from_bytes(&bytes);

            assert_eq!(gpu.len(), width, "width {width}");
            for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert!(
                    c.to_bits() == g.to_bits(),
                    "width {width} hblock {hblock} elem {i}: cpu {c} (bits {:x}) vs gpu {g} (bits {:x})",
                    c.to_bits(),
                    g.to_bits()
                );
            }
        }
    }

    /// Forward then inverse composes to the identity per block (S·H·H·S = I).
    #[test]
    fn fwht_forward_then_inverse_is_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let width = 5120usize;
        let hblock = 1024usize;
        let mut lcg = Lcg(0x1d2c);
        let x: Vec<f32> = (0..width).map(|_| lcg.next_f32(-3.0, 3.0)).collect();
        let signs = signs_vec(width, 0x7ee5);

        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle.clone(),
                width,
                width,
                hblock,
            );
            RotationCubeCL::launch_inverse::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle,
                width,
                width,
                hblock,
            );
        }
        let bytes = client.read_one(x_handle).expect("read roundtrip");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&orig, &got)) in x.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (orig - got).abs() <= 1e-4,
                "roundtrip elem {i}: {orig} vs {got}"
            );
        }
    }

    /// The batched form is the same kernel: a 3-row `[p × width]` batch must
    /// equal the CPU twin applied row-wise (row signs repeat via `% width`).
    #[test]
    fn fwht_forward_batched_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let width = 5120usize;
        let hblock = 1024usize;
        let rows = 3usize;
        let total = rows * width;
        let mut lcg = Lcg(0xba7c);
        let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(-1.5, 1.5)).collect();
        let signs = signs_vec(width, 0x5eed);

        let mut cpu = x.clone();
        for r in 0..rows {
            rotate_forward_inplace(&mut cpu[r * width..(r + 1) * width], Some(&signs), hblock);
        }

        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle,
                total,
                width,
                hblock,
            );
        }
        let bytes = client.read_one(x_handle).expect("read batched");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "batched elem {i}: cpu {c} vs gpu {g}"
            );
        }
    }

    /// Issue 1144 G1 — the radix-4 kernels are to_bits-equal to the
    /// incumbents (both forward forms) at the Bonsai decode widths (5120,
    /// 6144, 17408), a 3-row batch, and adversarial magnitudes.
    #[test]
    fn fwht_radix4_matches_incumbent_bitexact() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let hblock = MAX_FWHT_BLOCK;
        for &(width, rows, lo, hi) in &[
            (5120usize, 1usize, -3.0f32, 3.0f32),
            (6144, 1, -2.0, 2.0),
            (17408, 1, -1.0e4, 1.0e4),
            (5120, 3, -1.0e-3, 1.0e-3),
        ] {
            let total = rows * width;
            let mut lcg = Lcg(0x1144 + width as u64 + rows as u64);
            let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(lo, hi)).collect();
            let signs: Vec<f32> = signs_vec(width, 0x5eed).iter().map(|&v| v as f32).collect();
            let sh = client.create_from_slice(f32::as_bytes(&signs));
            let params = [width as f32, hblock as f32];
            let ph = client.create_from_slice(f32::as_bytes(&params));
            let n_cubes = (total / hblock) as u32;
            let run = |r4: bool, copy: bool| -> Vec<f32> {
                let xh = client.create_from_slice(f32::as_bytes(&x));
                let dh = client.empty(total * core::mem::size_of::<f32>());
                // SAFETY: handles sized exactly above (total / width / 2 f32).
                unsafe {
                    let (grid, dim) = (split_grid(n_cubes), CubeDim::new_1d(FWHT_THREADS));
                    let sa = BufferArg::from_raw_parts(sh.clone(), width);
                    let pa = BufferArg::from_raw_parts(ph.clone(), 2);
                    let xa = BufferArg::from_raw_parts(xh.clone(), total);
                    match (r4, copy) {
                        (true, true) => {
                            fwht_forward_copy_f32_r4::launch_unchecked::<ActiveRuntime>(
                                &client,
                                grid,
                                dim,
                                xa,
                                BufferArg::from_raw_parts(dh.clone(), total),
                                sa,
                                pa,
                            )
                        }
                        (false, true) => fwht_forward_copy_f32::launch_unchecked::<ActiveRuntime>(
                            &client,
                            grid,
                            dim,
                            xa,
                            BufferArg::from_raw_parts(dh.clone(), total),
                            sa,
                            pa,
                        ),
                        (true, false) => fwht_rotate_forward_f32_r4::launch_unchecked::<
                            ActiveRuntime,
                        >(&client, grid, dim, xa, sa, pa),
                        (false, false) => {
                            fwht_rotate_forward_f32::launch_unchecked::<ActiveRuntime>(
                                &client, grid, dim, xa, sa, pa,
                            )
                        }
                    }
                }
                let out = if copy { dh } else { xh };
                f32::from_bytes(&client.read_one(out).expect("read")).to_vec()
            };
            for copy in [false, true] {
                let a = run(false, copy);
                let b = run(true, copy);
                // Loud zero: the rotation must have run and changed the vector.
                assert!(
                    a.iter().zip(&x).any(|(p, q)| p != q),
                    "incumbent did not run"
                );
                for i in 0..total {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "elem {i} width={width} rows={rows} copy={copy}"
                    );
                }
            }
        }
    }

    /// The copy-rotate staging twin equals copy + in-place forward.
    #[test]
    fn fwht_forward_copy_matches_inplace() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let width = 6144usize;
        let hblock = 1024usize;
        let mut lcg = Lcg(0xc0ffee);
        let src: Vec<f32> = (0..width).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let signs = signs_vec(width, 0xd57);

        let mut cpu = src.clone();
        rotate_forward_inplace(&mut cpu, Some(&signs), hblock);

        let src_handle = client.create_from_slice(f32::as_bytes(&src));
        let dst_handle = client.empty(width * core::mem::size_of::<f32>());
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward_copy::<ActiveRuntime>(
                &client,
                src_handle,
                dst_handle.clone(),
                signs_handle,
                width,
                width,
                hblock,
            );
        }
        let bytes = client.read_one(dst_handle).expect("read staged");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "copy-rotate elem {i}: cpu {c} vs gpu {g}"
            );
        }
    }

    /// The gdn_v permute: GPU equals the CPU twin on a small grouped-V
    /// geometry (batched row + the whole-head gather).
    #[test]
    fn gdn_v_permute_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // n_v = 8 heads, n_k = 2 groups → rep = 4; hd = 16 → v_dim = 128.
        let n_v = 8usize;
        let n_k = 2usize;
        let hd = 16usize;
        let v_dim = n_v * hd;
        let rows = 2usize; // batched (decode is rows == 1)
        let total = rows * v_dim;
        let rep = n_v / n_k;

        let mut lcg = Lcg(0x9e7);
        let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(-2.0, 2.0)).collect();

        let mut cpu = x.clone();
        let mut tmp = vec![0.0f32; total];
        for r in 0..rows {
            let (xr, tr) = (
                &mut cpu[r * v_dim..(r + 1) * v_dim],
                &mut tmp[r * v_dim..(r + 1) * v_dim],
            );
            permute_gdn_v_grouped_inplace(xr, tr, n_v, n_k);
        }

        // GPU: tmp = staged source copy, x = destination (in-place contract).
        let tmp_handle = client.create_from_slice(f32::as_bytes(&x));
        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        unsafe {
            RotationCubeCL::launch_gdn_v_permute::<ActiveRuntime>(
                &client,
                tmp_handle,
                x_handle.clone(),
                total,
                v_dim,
                hd,
                n_k,
                rep,
            );
        }
        let bytes = client.read_one(x_handle).expect("read permuted");
        let gpu = f32::from_bytes(&bytes);
        for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "permute elem {i}: cpu {c} vs gpu {g}"
            );
        }
    }

    /// Batch prefill sizes exceed the 65 535 per-dimension workgroup limit:
    /// at P = 4096 the Bonsai-2 GDN permute alone is 98 304 workgroups (the
    /// Issue 1004 R1 e2e run died on that wgpu validation error). Both launch
    /// shapes go past the limit here and are checked bit-exact against the
    /// CPU reference over EVERY row, so a cube that the 2-D split drops or
    /// double-counts reds.
    #[test]
    fn launches_past_the_65535_grid_dim() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // FWHT: width 128, hblock 32 → 4 cubes/row; 17 000 rows → 68 000 cubes.
        let (width, hblock, rows) = (128usize, 32usize, 17_000usize);
        let total = width * rows;
        assert!(total / hblock > MAX_GRID_DIM as usize);
        let mut lcg = Lcg(0x1004);
        let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let signs = signs_vec(width, 0x65535);
        let mut cpu = x.clone();
        for row in cpu.chunks_exact_mut(width) {
            rotate_forward_inplace(row, Some(&signs), hblock);
        }
        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        let signs_f32: Vec<f32> = signs.iter().map(|&s| s as f32).collect();
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
        unsafe {
            RotationCubeCL::launch_forward::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                signs_handle,
                total,
                width,
                hblock,
            );
        }
        let gpu = f32::from_bytes(&client.read_one(x_handle).expect("read rotated")).to_vec();
        let bad = cpu
            .iter()
            .zip(&gpu)
            .filter(|(c, g)| c.to_bits() != g.to_bits())
            .count();
        assert_eq!(
            bad, 0,
            "FWHT past the grid limit: {bad} of {total} elements differ"
        );

        // Permute: the Bonsai-2 GDN shape (48 v-heads, 16 k-groups, hd 128) at
        // 3 000 rows → 72 000 workgroups of 256.
        let (n_v, n_k, hd, rows) = (48usize, 16usize, 128usize, 3_000usize);
        let v_dim = n_v * hd;
        let total = rows * v_dim;
        assert!(total.div_ceil(256) > MAX_GRID_DIM as usize);
        let x: Vec<f32> = (0..total).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let mut cpu = x.clone();
        let mut tmp = vec![0.0f32; v_dim];
        for row in cpu.chunks_exact_mut(v_dim) {
            permute_gdn_v_grouped_inplace(row, &mut tmp, n_v, n_k);
        }
        let tmp_handle = client.create_from_slice(f32::as_bytes(&x));
        let x_handle = client.create_from_slice(f32::as_bytes(&x));
        unsafe {
            RotationCubeCL::launch_gdn_v_permute::<ActiveRuntime>(
                &client,
                tmp_handle,
                x_handle.clone(),
                total,
                v_dim,
                hd,
                n_k,
                n_v / n_k,
            );
        }
        let gpu = f32::from_bytes(&client.read_one(x_handle).expect("read permuted")).to_vec();
        let bad = cpu
            .iter()
            .zip(&gpu)
            .filter(|(c, g)| c.to_bits() != g.to_bits())
            .count();
        assert_eq!(
            bad, 0,
            "permute past the grid limit: {bad} of {total} elements differ"
        );
    }

    /// Tables build: validates the block-size bound + uploads width-keyed
    /// signs; `signs_for_width` resolves each key and misses loudly.
    #[test]
    fn tables_build_and_signs_for_width() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let signs_5120 = signs_vec(5120, 0xa11);
        let signs_6144 = signs_vec(6144, 0xb22);
        let signs_17408 = signs_vec(17408, 0xc33);
        let cfg = TernaryRotationConfig {
            block_size: 1024,
            signs: vec![
                (5120usize, signs_5120.clone()),
                (6144, signs_6144.clone()),
                (17408, signs_17408.clone()),
            ],
            gdn_v_grouped: true,
            gdn_v_heads: 48,
            gdn_k_groups: 16,
            inverse_embedding: true,
        };
        let tables = RotationTablesCubeCL::build::<ActiveRuntime>(&client, &cfg)
            .expect("tables build within bound");
        assert_eq!(tables.block_size, 1024);
        assert_eq!(tables.signs.len(), 3);
        for w in [5120usize, 6144, 17408] {
            let h = tables.signs_for_width(w);
            let bytes = client.read_one(h.clone()).expect("read signs");
            let gpu = f32::from_bytes(&bytes);
            let want: Vec<f32> = match w {
                5120 => signs_5120.clone(),
                6144 => signs_6144.clone(),
                _ => signs_17408.clone(),
            }
            .iter()
            .map(|&s| s as f32)
            .collect();
            assert_eq!(gpu.len(), w);
            assert_eq!(gpu, want, "width {w} sign upload");
        }

        // The bound refuses (not panics) on an oversized block.
        let bad = TernaryRotationConfig {
            block_size: 2048,
            ..cfg
        };
        assert!(RotationTablesCubeCL::build::<ActiveRuntime>(&client, &bad).is_err());
    }

    /// Fused GDN fold tail (Issue 1113 lever 1): the fused gather+sign+FWHT
    /// kernel must be BIT-IDENTICAL to the split 3-launch chain (Copy →
    /// gdn_v_permute → fwht_forward) — the G3 shape at the kernel level, the
    /// same oracle the wiring falls back to. Bit-exactness is by construction
    /// (movement + single multiply + the incumbent butterfly), asserted here.
    #[test]
    fn gdn_permute_rotate_matches_split_chain_bitexact() {
        use crate::elementwise_cubecl::CopyCubeCL;

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-2-class geometry at two windows: head_dim 128 inside a 1024
        // window (8 heads/cube), n_k 4 · rep 4 (grouped), plus an ungrouped
        // arm (rep = 1 — the permute degenerates to identity).
        for &(head_dim, n_k, rep) in &[(128usize, 4usize, 4usize), (128usize, 16usize, 1usize)] {
            let hblock = 1024usize;
            let heads = n_k * rep;
            let v_dim = heads * head_dim;

            let mut lcg = Lcg(0x1113 + rep as u64);
            let x: Vec<f32> = (0..v_dim).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
            let signs_f32: Vec<f32> = signs_vec(v_dim, 0x5113 + rep as u64)
                .iter()
                .map(|&s| s as f32)
                .collect();

            // ── split chain: copy → permute → in-place FWHT ──
            let x_handle = client.create_from_slice(f32::as_bytes(&x));
            let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));
            let tmp_handle = client.create_from_slice(f32::as_bytes(&x)); // staging copy
            unsafe {
                CopyCubeCL::launch::<ActiveRuntime>(
                    &client,
                    x_handle.clone(),
                    tmp_handle.clone(),
                    v_dim,
                );
                RotationCubeCL::launch_gdn_v_permute::<ActiveRuntime>(
                    &client,
                    tmp_handle.clone(),
                    x_handle.clone(),
                    v_dim,
                    v_dim,
                    head_dim,
                    n_k,
                    rep,
                );
                RotationCubeCL::launch_forward::<ActiveRuntime>(
                    &client,
                    x_handle.clone(),
                    signs_handle.clone(),
                    v_dim,
                    v_dim,
                    hblock,
                );
            }

            // ── fused kernel (reads the PRIMAL fixture — the split chain
            //    mutated its own copy in place) ──
            let out_handle = client.create_from_slice(f32::as_bytes(&vec![0.0f32; v_dim]));
            unsafe {
                RotationCubeCL::launch_gdn_permute_rotate::<ActiveRuntime>(
                    &client,
                    client.create_from_slice(f32::as_bytes(&x)),
                    signs_handle.clone(),
                    out_handle.clone(),
                    v_dim,
                    head_dim,
                    n_k,
                    rep,
                    hblock,
                );
            }

            let split_bytes = client.read_one(x_handle).expect("read split");
            let split = f32::from_bytes(&split_bytes);
            let fused_bytes = client.read_one(out_handle).expect("read fused");
            let fused = f32::from_bytes(&fused_bytes);
            for (i, (a, b)) in split.iter().zip(fused.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "hd {head_dim} n_k {n_k} rep {rep} elem {i}: split {a} vs fused {b}"
                );
            }
        }
    }

    /// K5 (Issue 1113 lever 1 rung 2): fused SwiGLU+sign+FWHT must be
    /// BIT-IDENTICAL to DeltanetGatingConcat → fwht_forward.
    #[test]
    fn swiglu_rotate_fused_matches_split_chain_bitexact() {
        use crate::deltanet_cubecl::DeltanetGatingConcatCubeCL;

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let (n, hblock) = (2048usize, 1024usize);
        let mut lcg = Lcg(0x1155);
        let gate_up: Vec<f32> = (0..2 * n).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let signs_f32: Vec<f32> = signs_vec(n, 0x5115).iter().map(|&s| s as f32).collect();

        let gate_up_handle = client.create_from_slice(f32::as_bytes(&gate_up));
        let signs_handle = client.create_from_slice(f32::as_bytes(&signs_f32));

        // split chain: gating writes ffn_hidden, then the in-place rotation
        let split_handle = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
        unsafe {
            DeltanetGatingConcatCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_up_handle.clone(),
                split_handle.clone(),
                n,
            );
            RotationCubeCL::launch_forward::<ActiveRuntime>(
                &client,
                split_handle.clone(),
                signs_handle.clone(),
                n,
                n,
                hblock,
            );
        }

        // fused kernel
        let fused_handle = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
        unsafe {
            RotationCubeCL::launch_swiglu_rotate_fused::<ActiveRuntime>(
                &client,
                gate_up_handle.clone(),
                signs_handle.clone(),
                fused_handle.clone(),
                n,
                hblock,
            );
        }

        let split_bytes = client.read_one(split_handle).expect("read split");
        let split = f32::from_bytes(&split_bytes);
        let fused_bytes = client.read_one(fused_handle).expect("read fused");
        let fused = f32::from_bytes(&fused_bytes);
        for (i, (a, b)) in split.iter().zip(fused.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "elem {i}: split {a} vs fused {b}");
        }
    }
}
