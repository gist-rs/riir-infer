//! The batched fused norm+gating family for the dense-qwen38 prefill lane
//! (issue 1173 — bench 943's named lever).
//!
//! Bench 943 (the GDN-bucket GPU-timestamp census) measured the prefill
//! lane's non-GEMM share as a **per-section launch floor**: every small
//! kernel (norms, z-gating, residuals, casts) reads a size-independent
//! ~0.5–0.7 ms section wall at pp5120 while the SAME kernels read
//! 0.01–0.11 ms on an idle queue — the cost is the boundary, not the
//! work. The lever is fewer boundary groups: fuse the adjacent
//! norm-shaped pairs into ONE kernel each.
//!
//! Three fusions, each an independent promote-only env arm (the env
//! law: arms only on the exact literal `1`; unset/anything else is the
//! incumbent launch sequence byte-identical):
//!
//! | arm | env | fuses | sites |
//! |---|---|---|---|
//! | F1 | `RIIR_QWEN38_FUSE_SSM_ZGATE=1` | `RmsNormBatched(rec_b → rec_n_b)` + `DeltanetZGating` + the half-B f16 cast | the ssm output site, one per GDN layer |
//! | F2 | `RIIR_QWEN38_FUSE_NORM_CAST=1` | `RmsNormBatched` + the half-B f16 cast | input_norm + mlp_norm, every layer |
//! | F3 | `RIIR_QWEN38_FUSE_RESID_NORM=1` | `ResidualAdd` + the following `RmsNormBatched` (+ f16 cast) | post-attention residual → mlp norm (same layer) and mlp residual → next layer's input_norm (cross-layer) |
//!
//! # Numerics law — bit-identical, never lossy
//!
//! Each kernel copies its host kernel's phases VERBATIM (the same strided
//! x² accumulation, the same 256-slot pairwise shared-memory reduction
//! tree, the same `inv_rms = 1/sqrt(mean(x²)+eps)`, the same left-to-right
//! multiply order, and `half::f16::cast_from` for the twin — the exact
//! rounding the incumbent `CastF32ToF16CubeCL` uses). The G1
//! (`tests/qwen38_fused_norm_gating_g1.rs`) asserts GPU-vs-GPU BITWISE
//! equality against the unfused launch sequence on both the f32 and the
//! f16 outputs. No re-association, no approximation: this is a launch
//! mechanics change, not a numerics change.
//!
//! # Buffer law
//!
//! The f16 twin argument is `Option` at the launcher: when the half-B
//! posture (`RIIR_QWEN38_DENSE_MM2D_F16ACT=1`) carries no twin the
//! launcher binds a ONE-ELEMENT view of the γ buffer as the twin slot
//! (valid, never touched) and drives the twin write with an explicit
//! `params[3]` flag. ⚠ NEVER gate the twin write on the slot's `len()`:
//! a ZERO-LENGTH binding is clamped to the WHOLE buffer by the wgpu/Metal
//! binding layer, so a `len()==0` twin slot reads as full-length and a
//! length-gated kernel would trample the f32 output through the aliased
//! slot (measured — the G1's twin-less arms pin the failure mode).
//!
//! # Boundary notes
//!
//! - The fused kernels write the SAME buffers the unfused sequence writes
//!   (`rec_n_b` keeps its post-z-gate values, `x_b` keeps its
//!   post-residual values), so every existing reader sees identical bytes.
//! - F3's cross-layer arm moves the NEXT layer's input_norm launch into
//!   THIS layer's mlp_residual step (the fused kernel writes the next
//!   layer's `norm_x_b` + twin with the next layer's γ). The census
//!   instrument is unaffected: it pins gdn layer 0, whose input_norm is
//!   never prefetched, and the fused wall lands inside the instrumented
//!   layer's own `mlp_residual` section (issue 1173 wires this; bench 943
//!   section list unchanged).

use std::sync::OnceLock;

use cubecl::prelude::*;
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Env arms (promote-only law: the exact literal `1` arms; anything else —
// including unset — is the incumbent launch sequence byte-identical)
// ---------------------------------------------------------------------------

static FUSE_SSM_ZGATE: OnceLock<bool> = OnceLock::new();
static FUSE_NORM_CAST: OnceLock<bool> = OnceLock::new();
static FUSE_RESID_NORM: OnceLock<bool> = OnceLock::new();

/// `RIIR_QWEN38_FUSE_SSM_ZGATE=1` arms F1 (norm_ssm + z-gate + f16 twin).
#[must_use]
pub fn fuse_ssm_zgate() -> bool {
    *FUSE_SSM_ZGATE
        .get_or_init(|| std::env::var("RIIR_QWEN38_FUSE_SSM_ZGATE").as_deref() == Ok("1"))
}

/// `RIIR_QWEN38_FUSE_NORM_CAST=1` arms F2 (batched norm + f16 twin).
#[must_use]
pub fn fuse_norm_cast() -> bool {
    *FUSE_NORM_CAST
        .get_or_init(|| std::env::var("RIIR_QWEN38_FUSE_NORM_CAST").as_deref() == Ok("1"))
}

/// `RIIR_QWEN38_FUSE_RESID_NORM=1` arms F3 (residual add + next norm).
#[must_use]
pub fn fuse_resid_norm() -> bool {
    *FUSE_RESID_NORM
        .get_or_init(|| std::env::var("RIIR_QWEN38_FUSE_RESID_NORM").as_deref() == Ok("1"))
}

/// The optional-twin slot binding, shared by every fused launcher (issue
/// 1175). The USED arm binds the twin at its chunk byte offset; the UNUSED
/// arm binds a ONE-ELEMENT view of `fallback` (a γ buffer — in-bounds by
/// construction) at OFFSET 0.
///
/// ⚠ Two laws baked in here:
///
/// 1. **Never a zero-length bind** — the wgpu/Metal binding layer clamps a
///    0-length view to the WHOLE buffer, so a length-gated kernel would
///    read the slot as full-length and trample the f32 output through the
///    alias (the bench-944 instrument law).
/// 2. **Offset 0 on the fallback** — the audit (issue 1175) found the
///    pre-helper sites rode the TWIN layout offset (`r0·dim·2` bytes) on
///    the fallback too, which runs past a γ buffer's end for any chunked
///    twin-less launch (in-bounds only because the G1 shapes never
///    chunked). The slot is never dereferenced, so offset 0 is
///    behavior-identical and always valid.
///
/// The kernel-side twin write is gated by the explicit `params[3]` flag —
/// NEVER the slot's `len()`.
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn optional_twin_binding(
    twin: Option<&Handle>,
    fallback: &Handle,
    total: usize,
    chunk_byte_offset: u64,
) -> (Handle, u64, usize) {
    match twin {
        Some(h) => (h.clone(), chunk_byte_offset, total),
        None => (fallback.clone(), 0, 1),
    }
}

// ---------------------------------------------------------------------------
// F1 kernel: batched RMSNorm + z-gating + optional f16 twin
// ---------------------------------------------------------------------------

/// Fused batched RMSNorm + z-gating (+ optional f16 staging twin) — ONE
/// launch replacing `RmsNormBatched` → `DeltanetZGating` (→ `CastF32ToF16`).
///
/// `output[idx] = rmsnorm(input)[idx] * silu(z[idx])` with the unfused
/// pair's exact op order (`((x·inv_rms)·g)·z·σ(z)` — z-gating reads the
/// norm's write in place). Phases 1–3 are the `rmsnorm_batched_f32` tree
/// VERBATIM (same accumulation order → bit-identical `inv_rms`).
///
/// `output_h` is `Option` at the launcher: when the posture carries no
/// twin the launcher binds a ONE-ELEMENT view of the γ buffer as the twin
/// slot (valid binding, never touched) and drives the twin write with the
/// explicit `params[3]` flag. ⚠ The flag — NOT the slice length — is the
/// law: a ZERO-LENGTH binding is clamped to the WHOLE buffer by the
/// wgpu/Metal binding layer, so a `len()==0` twin slot reads as full-length
/// and a length-gated kernel would trample the f32 output through the
/// aliased slot (the failure mode the G1's twin-less arm pins).
///
/// ## Dispatch
///
/// `CubeCount::Static(rows, 1, 1)` workgroups of 256 threads (one per row,
/// chunked at the launcher under Metal's 65535 x-cap).
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::len_zero,
    reason = "#[cube] kernel context: the slice method surface proven here is .len() (the cast kernel precedent); is_empty is not a proven kernel-side call"
)]
#[cube(launch_unchecked)]
fn rmsnorm_zgate_batched_f32(
    input: &[f32],
    gamma: &[f32],
    z: &[f32],
    params: &[f32], // [inv_dim, eps, dim, has_twin]
    output: &mut [f32],
    output_h: &mut [half::f16],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let with_h = params[3usize] > f32::new(0.5f32);
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let row = CUBE_POS_X;
    let row_offset = row * dim;

    // ── Phase 1: Strided accumulation of x² for this row ──
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let x = input[(row_offset + i) as usize];
        partial_sq += x * x;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction (the batched-norm tree,
    //    verbatim — the summation order IS the bit-identity contract) ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);
    smem[tid as usize] = partial_sq;
    sync_cube();

    if tid < 128u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        smem[0usize] = smem[0usize] + smem[1usize];
    }
    sync_cube();

    // ── Phase 3: Broadcast inv_rms via shared memory ──
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        smem[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
    sync_cube();

    let inv_rms = smem[0usize];

    // ── Phase 4: normalize, gamma, z-gate, write (f32 + optional twin) ──
    let mut j = tid;
    while j < dim {
        let idx = (row_offset + j) as usize;
        let x = input[idx];
        let g = gamma[j as usize];
        let z_val = z[idx];
        // silu(z) = z * sigmoid(z) — the DeltanetZGating expression verbatim
        let neg_z = f32::new(0.0f32) - z_val;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_z.exp());
        let v = x * inv_rms * g * z_val * sig;
        output[idx] = v;
        if with_h {
            output_h[idx] = half::f16::cast_from(v);
        }
        j += cube_size;
    }
}

/// Launcher for the fused norm + z-gate (+ twin) kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct RmsNormZgateBatchedF16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl RmsNormZgateBatchedF16CubeCL {
    /// Launch fused batched norm + z-gating.
    ///
    /// # Safety
    ///
    /// - `input_handle` / `z_handle` / `output_handle`: `rows * dim` f32
    ///   elements (`output` may alias nothing — separate from `input`)
    /// - `gamma_handle`: `dim` f32 elements
    /// - `output_h_handle`: `Some` = `rows * dim` f16 elements written by
    ///   the kernel; `None` = no twin (a one-element γ view is bound as the
    ///   twin slot and never touched — the `params[3]` flag gates the twin
    ///   write, never the slot length: a ZERO-LENGTH binding is clamped to
    ///   the whole buffer by the wgpu/Metal binding layer)
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: buffer handles + shape are the fused-kernel interface"
    )]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        z_handle: Handle,
        output_handle: Handle,
        output_h_handle: Option<Handle>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        // Metal grid guard (Issue 730): one workgroup per row, chunked on
        // row boundaries — rows are independent and the chunk math is
        // row-relative (CUBE_POS_X), so chunking is bit-identical (the
        // RmsNormBatchedCubeCL precedent).
        const MAX_WG_X: u32 = 65535;
        let inv_dim = 1.0f32 / dim as f32;
        let dim_f = dim as f32;
        let mut r0 = 0usize;
        while r0 < rows {
            let rc = (MAX_WG_X as usize).min(rows - r0);
            let has_twin = output_h_handle.is_some();
            let params: &[f32] = &[inv_dim, eps, dim_f, if has_twin { 1.0 } else { 0.0 }];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
            let total = rc * dim;
            let (oh, oh_off, oh_len) = optional_twin_binding(
                output_h_handle.as_ref(),
                &gamma_handle,
                total,
                (r0 * dim * 2) as u64,
            );
            // SAFETY: Caller guarantees buffer sizes; the slices stay inside.
            unsafe {
                rmsnorm_zgate_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(rc as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        input_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(gamma_handle.clone(), dim),
                    BufferArg::from_raw_parts(
                        z_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(params_handle, 4),
                    BufferArg::from_raw_parts(
                        output_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(oh.offset_start(oh_off), oh_len),
                );
            }
            r0 += rc;
        }
    }
}

// ---------------------------------------------------------------------------
// F2 kernel: batched RMSNorm + optional f16 twin
// ---------------------------------------------------------------------------

/// Fused batched RMSNorm (+ optional f16 staging twin) — ONE launch
/// replacing `RmsNormBatched` → `CastF32ToF16` at the input_norm and
/// mlp_norm sites. Phase-identical to `rmsnorm_batched_f32` with the twin
/// write folded into phase 4.
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::len_zero,
    reason = "#[cube] kernel context: the slice method surface proven here is .len() (the cast kernel precedent); is_empty is not a proven kernel-side call"
)]
#[cube(launch_unchecked)]
fn rmsnorm_twin_batched_f32(
    input: &[f32],
    gamma: &[f32],
    params: &[f32], // [inv_dim, eps, dim, has_twin]
    output: &mut [f32],
    output_h: &mut [half::f16],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let with_h = params[3usize] > f32::new(0.5f32);
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let row = CUBE_POS_X;
    let row_offset = row * dim;

    // ── Phase 1: Strided accumulation of x² for this row ──
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let x = input[(row_offset + i) as usize];
        partial_sq += x * x;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction (verbatim) ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);
    smem[tid as usize] = partial_sq;
    sync_cube();

    if tid < 128u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        smem[0usize] = smem[0usize] + smem[1usize];
    }
    sync_cube();

    // ── Phase 3: Broadcast inv_rms via shared memory ──
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        smem[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
    sync_cube();

    let inv_rms = smem[0usize];

    // ── Phase 4: normalize, gamma, write (f32 + optional twin) ──
    let mut j = tid;
    while j < dim {
        let idx = (row_offset + j) as usize;
        let v = input[idx] * inv_rms * gamma[j as usize];
        output[idx] = v;
        if with_h {
            output_h[idx] = half::f16::cast_from(v);
        }
        j += cube_size;
    }
}

/// Launcher for the fused norm + twin kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct RmsNormTwinBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl RmsNormTwinBatchedCubeCL {
    /// Launch fused batched norm (+ twin). Same contract as
    /// [`RmsNormZgateBatchedF16CubeCL::launch`] minus the z buffer.
    ///
    /// # Safety
    ///
    /// - `input_handle` / `output_handle`: `rows * dim` f32 elements
    /// - `gamma_handle`: `dim` f32 elements
    /// - `output_h_handle`: `Some` = `rows * dim` f16 elements; `None` = none
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        output_handle: Handle,
        output_h_handle: Option<Handle>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        const MAX_WG_X: u32 = 65535;
        let inv_dim = 1.0f32 / dim as f32;
        let dim_f = dim as f32;
        let mut r0 = 0usize;
        while r0 < rows {
            let rc = (MAX_WG_X as usize).min(rows - r0);
            let params: &[f32] = &[
                inv_dim,
                eps,
                dim_f,
                if output_h_handle.is_some() { 1.0 } else { 0.0 },
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
            let total = rc * dim;
            let (oh, oh_off, oh_len) = optional_twin_binding(
                output_h_handle.as_ref(),
                &gamma_handle,
                total,
                (r0 * dim * 2) as u64,
            );
            // SAFETY: Caller guarantees buffer sizes; the slices stay inside.
            unsafe {
                rmsnorm_twin_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(rc as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        input_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(gamma_handle.clone(), dim),
                    BufferArg::from_raw_parts(params_handle, 4),
                    BufferArg::from_raw_parts(
                        output_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(oh.offset_start(oh_off), oh_len),
                );
            }
            r0 += rc;
        }
    }
}

// ---------------------------------------------------------------------------
// F3 kernel: residual add + batched RMSNorm + optional f16 twin
// ---------------------------------------------------------------------------

/// Fused residual-add + batched RMSNorm (+ optional f16 twin) — ONE launch
/// replacing `ResidualAdd` → `RmsNormBatched` (→ `CastF32ToF16`).
///
/// `x[idx] += residual[idx]` happens in phase 1 (each thread updates and
/// squares its OWN row's elements — no cross-thread hazard; the workgroup
/// owns its row exclusively), then phases 2–3 reduce the POST-add values
/// exactly as the unfused norm kernel reads them back. Phase 4 re-reads
/// the updated `x` (the unfused norm's input).
#[cfg(feature = "cubecl_runtime")]
#[allow(
    clippy::len_zero,
    reason = "#[cube] kernel context: the slice method surface proven here is .len() (the cast kernel precedent); is_empty is not a proven kernel-side call"
)]
#[cube(launch_unchecked)]
fn residual_rmsnorm_twin_batched_f32(
    x: &mut [f32],
    residual: &[f32],
    gamma: &[f32],
    params: &[f32], // [inv_dim, eps, dim, has_twin]
    norm_out: &mut [f32],
    norm_out_h: &mut [half::f16],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let with_h = params[3usize] > f32::new(0.5f32);
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let row = CUBE_POS_X;
    let row_offset = row * dim;

    // ── Phase 1: residual add (in-place) + strided x² of the POST-add row ──
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let idx = (row_offset + i) as usize;
        let v = x[idx] + residual[idx];
        x[idx] = v;
        partial_sq += v * v;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction (verbatim) ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);
    smem[tid as usize] = partial_sq;
    sync_cube();

    if tid < 128u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        smem[0usize] = smem[0usize] + smem[1usize];
    }
    sync_cube();

    // ── Phase 3: Broadcast inv_rms via shared memory ──
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        smem[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
    sync_cube();

    let inv_rms = smem[0usize];

    // ── Phase 4: normalize the updated x, write norm_out (+ optional twin) ──
    let mut j = tid;
    while j < dim {
        let idx = (row_offset + j) as usize;
        let v = x[idx] * inv_rms * gamma[j as usize];
        norm_out[idx] = v;
        if with_h {
            norm_out_h[idx] = half::f16::cast_from(v);
        }
        j += cube_size;
    }
}

/// Launcher for the fused residual + norm (+ twin) kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct ResidualRmsNormTwinBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ResidualRmsNormTwinBatchedCubeCL {
    /// Launch fused residual add + norm (+ twin).
    ///
    /// # Safety
    ///
    /// - `x_handle`: `rows * dim` f32 elements, updated IN PLACE with the
    ///   residual (the incumbent `ResidualAdd(x, res → x)` semantics)
    /// - `residual_handle`: `rows * dim` f32 elements (read-only; must not
    ///   alias `x_handle`)
    /// - `gamma_handle`: `dim` f32 elements
    /// - `norm_out_handle`: `rows * dim` f32 elements (must not alias
    ///   `x_handle`)
    /// - `norm_out_h_handle`: `Some` = `rows * dim` f16 elements; `None` = none
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: buffer handles + shape are the fused-kernel interface"
    )]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        residual_handle: Handle,
        gamma_handle: Handle,
        norm_out_handle: Handle,
        norm_out_h_handle: Option<Handle>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        const MAX_WG_X: u32 = 65535;
        let inv_dim = 1.0f32 / dim as f32;
        let dim_f = dim as f32;
        let mut r0 = 0usize;
        while r0 < rows {
            let rc = (MAX_WG_X as usize).min(rows - r0);
            let params: &[f32] = &[
                inv_dim,
                eps,
                dim_f,
                if norm_out_h_handle.is_some() {
                    1.0
                } else {
                    0.0
                },
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
            let total = rc * dim;
            let (oh, oh_off, oh_len) = optional_twin_binding(
                norm_out_h_handle.as_ref(),
                &gamma_handle,
                total,
                (r0 * dim * 2) as u64,
            );
            // SAFETY: Caller guarantees buffer sizes; the slices stay inside.
            unsafe {
                residual_rmsnorm_twin_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(rc as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        x_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(
                        residual_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(gamma_handle.clone(), dim),
                    BufferArg::from_raw_parts(params_handle, 4),
                    BufferArg::from_raw_parts(
                        norm_out_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(oh.offset_start(oh_off), oh_len),
                );
            }
            r0 += rc;
        }
    }
}
