//! CubeCL RMSNorm and residual add kernels (Plan 106 T2.12).
//!
//! GPU-accelerated normalization and residual addition for Gemma 2 decode.
//! Replaces CPU fallback ops in `gemma2_cubecl.rs` to eliminate sync points
//! in the hybrid CPU/CubeCL forward pass.
//!
//! # Kernels
//!
//! | Kernel | Algorithm | Dispatch | SharedMemory |
//! |--------|-----------|----------|--------------|
//! | `rmsnorm_f32` | Strided reduce + broadcast + normalize | 1 WG × 256 threads | 1 KB |
//! | `residual_add_f32` | Element-wise `a[i] + b[i]` | `ceil(n/256)` WG × 256 threads | None |
//!
//! # CubeCL v0.10 Constraints
//!
//! - Exactly 4 Array parameters for `rmsnorm_f32` (input, gamma, params, output).
//! - 3 Array parameters for `residual_add_f32` (a, b, output).
//! - No conditional expressions as values — use `if { }` statements.
//! - Unrolled parallel reductions (8 hardcoded steps), `sync_cube()` between steps.
//! - `UNIT_POS` is `u32`, `ABSOLUTE_POS` is `usize` — cast appropriately.
//! - `f32::new(literal)` for constants in `#[cube]` context.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.
//!
//! # Parameters Array (RMSNorm)
//!
//! `rmsnorm_f32` takes a `params: &[f32]` with 3 elements to avoid
//! u32→f32 casting issues in CubeCL v0.10:
//! - `params[0]` = `inv_dim` (1.0 / dim as f32, precomputed on CPU)
//! - `params[1]` = `eps` (rms_norm_eps, typically 1e-6 for Gemma 2)
//! - `params[2]` = `dim` (Issue 639 — must NOT be derived from `input.len()`,
//!   which reports the backing allocation, not the bound length)

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::debug_assert_binding_at_least;

// ---------------------------------------------------------------------------
// RMSNorm kernel
// ---------------------------------------------------------------------------

/// CubeCL RMSNorm kernel with learnable gamma (single-workgroup, shared memory reduction).
///
/// Computes `output[i] = input[i] * inv_rms * gamma[i]` where
/// `inv_rms = 1 / sqrt(mean(x²) + eps)`.
///
/// Uses a single workgroup of 256 threads with strided accumulation to handle
/// any dimension (including Gemma 2's n_embd=2048). Shared memory is used for
/// an unrolled parallel sum reduction of `x²` values.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; dim]` — input vector.
/// - `gamma`: `[f32; dim]` — learnable scale (with +1 offset pre-applied).
/// - `params`: `[f32; 3]` — `[inv_dim, eps, dim]` precomputed on CPU.
/// - `output`: `[f32; dim]` — normalized output vector.
///
/// `dim` is passed **explicitly** rather than read from `input.len()`. See
/// Issue 639: `BufferArg::from_raw_parts(handle, len)` does not constrain the
/// kernel-visible length, so `input.len()` reports the whole backing allocation
/// when the handle is oversized (e.g. a row view of a `[P, dim]` buffer). That
/// made this kernel reduce x² over the following rows while `inv_dim` still said
/// `1/dim` — a silent, shape-dependent **scale** error. Taking `dim` from
/// `params` makes the bound length non-load-bearing, matching
/// `rmsnorm_batched_f32`.
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rmsnorm_f32(
    input: &[f32],
    gamma: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // ── Phase 1: Strided accumulation of x² ──
    // Each thread accumulates elements at tid, tid+256, tid+512, ...
    // This handles dim > 256 (e.g., Gemma 2 n_embd=2048).
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let x = input[i as usize];
        partial_sq += x * x;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    smem[tid as usize] = partial_sq;
    sync_cube();

    // Unrolled parallel sum reduction (128→64→32→16→8→4→2→1)
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

    // ── Phase 4: Normalize and apply gamma ──
    let mut j = tid;
    while j < dim {
        let x = input[j as usize];
        let g = gamma[j as usize];
        output[j as usize] = x * inv_rms * g;
        j += cube_size;
    }
}

// ---------------------------------------------------------------------------
// Residual add kernel
// ---------------------------------------------------------------------------

/// CubeCL element-wise residual add: `output[i] = a[i] + b[i]`.
///
/// Simple element-wise kernel with bounds checking.
/// No shared memory needed — each thread writes one element.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn residual_add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    let n = a.len();
    let tid = ABSOLUTE_POS;

    if tid < n {
        output[tid] = a[tid] + b[tid];
    }
}

// ---------------------------------------------------------------------------
// Batched RMSNorm kernel (Plan 482 T4)
// ---------------------------------------------------------------------------

/// CubeCL batched RMSNorm kernel — processes `[seq_len × dim]` in one dispatch.
///
/// One workgroup per row (position). `CUBE_POS_X` selects the row; `UNIT_POS`
/// is the thread within the row's workgroup (0..255, strided for dim > 256).
///
/// ## Parameter Layout
///
/// - `input`: `[f32; seq_len * dim]` — batched input (row-major).
/// - `gamma`: `[f32; dim]` — learnable scale (shared across rows).
/// - `params`: `[f32; 3]` — `[inv_dim, eps, dim]` precomputed on CPU.
/// - `output`: `[f32; seq_len * dim]` — normalized output.
///
/// ## Dispatch
///
/// `CubeCount::Static(seq_len, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rmsnorm_batched_f32(
    input: &[f32],
    gamma: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
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

    // ── Phase 2: Shared memory parallel reduction ──
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

    // ── Phase 4: Normalize and apply gamma for this row ──
    let mut j = tid;
    while j < dim {
        let x = input[(row_offset + j) as usize];
        let g = gamma[j as usize];
        output[(row_offset + j) as usize] = x * inv_rms * g;
        j += cube_size;
    }
}

// ---------------------------------------------------------------------------
// Launcher structs
// ---------------------------------------------------------------------------

/// CubeCL RMSNorm launcher.
///
/// Wraps the `rmsnorm_f32` kernel with precomputed parameters.
/// Single workgroup handles the full dimension via strided access.
#[cfg(feature = "cubecl_runtime")]
pub struct RmsNormCubeCL;

/// CubeCL batched RMSNorm launcher (Plan 482 T4).
///
/// Processes `[seq_len × dim]` in a single dispatch — one workgroup per row.
/// Replaces `seq_len` separate `RmsNormCubeCL::launch` calls.
#[cfg(feature = "cubecl_runtime")]
pub struct RmsNormBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in T2.12+ forward pass wiring
impl RmsNormCubeCL {
    /// Launch RMSNorm kernel: `output[i] = input[i] * inv_rms * gamma[i]`.
    ///
    /// Computes `inv_rms = 1 / sqrt(mean(x²) + eps)` on GPU.
    /// Precomputes `inv_dim` and `eps` on CPU, passes via params buffer
    /// to avoid u32→f32 casting issues in CubeCL v0.10.
    ///
    /// Dispatch: `(1, 1, 1)` workgroups of 256 threads (strided access).
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `input_handle`: `dim` f32 elements
    /// - `gamma_handle`: `dim` f32 elements
    /// - `output_handle`: `dim` f32 elements
    ///
    /// `dim` must be > 0. `eps` must be > 0.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        output_handle: Handle,
        dim: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        // `dim` travels in params, not via `input.len()` — see the kernel doc
        // and Issue 639. An oversized `input_handle` is now harmless.
        let params: &[f32] = &[inv_dim, eps, dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));

        // Issue 639 T5: an oversized binding is no longer a correctness bug,
        // but an UNDERsized one is still out-of-bounds UB. Fail loudly in debug.
        debug_assert_binding_at_least(&input_handle, dim, "RmsNormCubeCL::input");
        debug_assert_binding_at_least(&output_handle, dim, "RmsNormCubeCL::output");

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rmsnorm_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, dim),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(output_handle, dim),
            );
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in Plan 482 batched forward
impl RmsNormBatchedCubeCL {
    /// Launch batched RMSNorm: normalizes each row of `[seq_len × dim]` independently.
    ///
    /// Dispatch: `(seq_len, 1, 1)` workgroups of 256 threads (one per row).
    ///
    /// # Safety
    ///
    /// - `input_handle`: `seq_len * dim` f32 elements
    /// - `gamma_handle`: `dim` f32 elements
    /// - `output_handle`: `seq_len * dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        output_handle: Handle,
        seq_len: usize,
        dim: usize,
        eps: f32,
    ) {
        // Metal grid guard (Issue 730, 2026-08-19): one workgroup per row with
        // the row count taken from the CALLER (the GDN per-head norm widens it
        // to p * n_v_heads = 98304 at p=2048 on Bonsai-27B) — exceeds Metal's
        // 65535 x-cap. Rows are independent (one wg per row, row-relative
        // math from CUBE_POS_X), so row-boundary chunking with sliced
        // input/output is bit-identical. Covers every caller.
        const MAX_WG_X: u32 = 65535;
        let inv_dim = 1.0f32 / dim as f32;
        let dim_f = dim as f32;
        let mut r0 = 0usize;
        while r0 < seq_len {
            let rc = (MAX_WG_X as usize).min(seq_len - r0);
            let params: &[f32] = &[inv_dim, eps, dim_f];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
            let total = rc * dim;
            // SAFETY: Caller guarantees buffer sizes; the slices stay inside.
            unsafe {
                rmsnorm_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(rc as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        input_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(gamma_handle.clone(), dim),
                    BufferArg::from_raw_parts(params_handle, 3),
                    BufferArg::from_raw_parts(
                        output_handle.clone().offset_start((r0 * dim * 4) as u64),
                        total,
                    ),
                );
            }
            r0 += rc;
        }
    }
}

/// CubeCL residual add launcher.
///
/// Wraps the `residual_add_f32` kernel for element-wise `a + b`.
#[cfg(feature = "cubecl_runtime")]
pub struct ResidualAddCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in T2.12+ forward pass wiring
impl ResidualAddCubeCL {
    /// Launch residual add kernel: `output[i] = a[i] + b[i]`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `a_handle`: `n` f32 elements
    /// - `b_handle`: `n` f32 elements
    /// - `output_handle`: `n` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_handle: Handle,
        b_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        // Metal grid guard (Issue 730): n/256 workgroups exceeds the 65535
        // x-cap from n >= 16.8M (p * n_embd at p=3277; 81920 wg at p=4096).
        // The kernel bounds itself by the slice length (`a.len()`), so chunked
        // launches with sliced handles + exact lengths are bit-identical.
        const MAX_WG_X: u32 = 65535;
        let wg: u32 = 256;
        let elems_per_chunk = (MAX_WG_X as usize) * wg as usize;
        let mut e0 = 0usize;
        while e0 < n {
            let ec = elems_per_chunk.min(n - e0);
            let num_wg = (ec as u32).div_ceil(wg).max(1);
            // SAFETY: Caller guarantees buffer sizes; slices stay inside.
            unsafe {
                residual_add_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg),
                    BufferArg::from_raw_parts(a_handle.clone().offset_start((e0 * 4) as u64), ec),
                    BufferArg::from_raw_parts(b_handle.clone().offset_start((e0 * 4) as u64), ec),
                    BufferArg::from_raw_parts(
                        output_handle.clone().offset_start((e0 * 4) as u64),
                        ec,
                    ),
                );
            }
            e0 += ec;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused dual-buffer RMSNorm (Q+K normalization in 1 dispatch, Issue 642 F4)
// ---------------------------------------------------------------------------

/// Fused Q+K RMSNorm: normalizes Q heads and K heads in a **single dispatch**.
///
/// Replaces 2 separate `RmsNormBatchedCubeCL::launch` calls (one for Q, one
/// for K) with 1 dispatch. Saves 1 dispatch per Attention layer (Issue 642 F4).
///
/// One workgroup per head. Workgroups `0..n_q` normalize Q; workgroups
/// `n_q..n_q+n_k` normalize K. All heads share the same `head_dim` + `eps`.
/// Both inputs are normalized **in-place** (input == output handle).
///
/// ## Parameter Layout
///
/// - `q`: `[f32; n_q * head_dim]` — Q heads, normalized in-place
/// - `k`: `[f32; n_k * head_dim]` — K heads, normalized in-place
/// - `q_gamma`: `[f32; head_dim]` — Q norm gamma
/// - `k_gamma`: `[f32; head_dim]` — K norm gamma
/// - `params`: `[f32; 4]` — `[inv_dim, eps, dim, n_q]`
///
/// ## Dispatch
///
/// `CubeCount::Static(n_q + n_k, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rmsnorm_qk_fused_f32(
    q: &mut [f32],
    k: &mut [f32],
    q_gamma: &[f32],
    k_gamma: &[f32],
    params: &[f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let n_q = params[3usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let wg = CUBE_POS_X;

    // Select which buffer/gamma this workgroup operates on.
    // Workgroups 0..n_q → Q; workgroups n_q..n_q+n_k → K.
    let is_q = wg < n_q;
    let head = if is_q { wg } else { wg - n_q };
    let row_offset = head * dim;

    // ── Phase 1: Strided accumulation of x² for this head ──
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let idx = (row_offset + i) as usize;
        let x = if is_q { q[idx] } else { k[idx] };
        partial_sq += x * x;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction (same as rmsnorm_batched) ──
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

    // ── Phase 4: Normalize and apply gamma for this head (in-place) ──
    let mut j = tid;
    while j < dim {
        let idx = (row_offset + j) as usize;
        let g = if is_q { q_gamma[j as usize] } else { k_gamma[j as usize] };
        let x = if is_q { q[idx] } else { k[idx] };
        let normalized = x * inv_rms * g;
        if is_q {
            q[idx] = normalized;
        } else {
            k[idx] = normalized;
        }
        j += cube_size;
    }
}

/// Fused Q+K RMSNorm launcher (Issue 642 F4).
///
/// Normalizes Q and K heads in a single dispatch. Both inputs are modified
/// in-place. Saves 1 dispatch per Attention layer vs two separate
/// `RmsNormBatchedCubeCL::launch` calls.
#[cfg(feature = "cubecl_runtime")]
pub struct RmsNormQkFusedCubeCL;

/// CubeCL kernel: fused per-head RMSNorm + z-gating (Issue 642 F6).
///
/// Replaces the two-dispatch DeltaNet output path
/// (RmsNormBatched → DeltanetZGating) with a single dispatch. Each workgroup
/// normalizes one head of `output` `[n_heads × head_dim]`, applies `gamma`, then
/// multiplies by `silu(z[idx])`.
///
/// Dispatch: `CubeCount::Static(n_heads, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rmsnorm_zgate_fused_f32(
    output: &mut [f32],
    gamma: &[f32],
    z: &[f32],
    params: &[f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let row = CUBE_POS_X;
    let row_offset = row * dim;

    // ── Phase 1: Strided accumulation of x² for this row ──
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let x = output[(row_offset + i) as usize];
        partial_sq += x * x;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction ──
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

    // ── Phase 4: Normalize, apply gamma, then z-gate (silu(z)) in-place ──
    let mut j = tid;
    while j < dim {
        let idx = (row_offset + j) as usize;
        let x = output[idx];
        let g = gamma[j as usize];
        let z_val = z[idx];
        // silu(z) = z * sigmoid(z)
        let neg_z = f32::new(0.0f32) - z_val;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_z.exp());
        output[idx] = x * inv_rms * g * z_val * sig;
        j += cube_size;
    }
}

#[cfg(feature = "cubecl_runtime")]
impl RmsNormQkFusedCubeCL {
    /// Launch fused Q+K RMSNorm: normalizes Q and K heads in one dispatch.
    ///
    /// Dispatch: `(n_q + n_kv, 1, 1)` workgroups of 256 threads.
    /// Workgroups 0..n_q normalize Q; n_q..n_q+n_kv normalize K.
    ///
    /// # Safety
    ///
    /// - `q_handle`: `n_q * head_dim` f32 elements (modified in-place)
    /// - `k_handle`: `n_kv * head_dim` f32 elements (modified in-place)
    /// - `q_gamma_handle`: `head_dim` f32 elements
    /// - `k_gamma_handle`: `head_dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        q_gamma_handle: Handle,
        k_gamma_handle: Handle,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
    ) {
        // Metal grid guard (Issue 726, 2026-08-19): one workgroup per row with
        // n_q + n_kv rows — at prefill scale (p * (n_head + n_kv_head)) the
        // x-dimension exceeds Metal's 65535 cap from p >= 2341 at Bonsai
        // dims. The launch contract here takes TOTAL row counts; chunking on
        // token boundaries is done by the CALLER when it knows p — this
        // wrapper additionally self-chunks row batches so any call is safe:
        // rows are independent (one workgroup per row, in-place normalize),
        // and the kernel selects its buffer/gamma purely from `wg < n_q`.
        // Slicing on ROW boundaries keeps q rows in the q slice and k rows in
        // the k slice — a chunk never mixes buffers.
        const MAX_WG_X: u32 = 65535;
        let inv_dim = 1.0f32 / head_dim as f32;
        let dim_f = head_dim as f32;
        let max_rows = MAX_WG_X as usize;
        // Chunk the q block and the k block independently on row multiples.
        let mut q0 = 0usize;
        while q0 < n_q {
            let rc = max_rows.min(n_q - q0);
            let params: &[f32] = &[inv_dim, eps, dim_f, rc as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
            // SAFETY: caller guarantees buffer sizes; the slice stays inside.
            unsafe {
                rmsnorm_qk_fused_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(rc as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(
                        q_handle.clone().offset_start((q0 * head_dim * 4) as u64),
                        rc * head_dim,
                    ),
                    // Zero-length k slice: no k rows in this launch. The
                    // kernel never reads k when every wg < n_q (= rc).
                    BufferArg::from_raw_parts(k_handle.clone(), 0),
                    BufferArg::from_raw_parts(q_gamma_handle.clone(), head_dim),
                    BufferArg::from_raw_parts(k_gamma_handle.clone(), head_dim),
                    BufferArg::from_raw_parts(params_handle, 4),
                );
            }
            q0 += rc;
        }
        let mut k0 = 0usize;
        while k0 < n_kv {
            let rc = max_rows.min(n_kv - k0);
            let params: &[f32] = &[inv_dim, eps, dim_f, 0f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
            // SAFETY: same as the q loop; n_q = 0 routes every wg to k.
            unsafe {
                rmsnorm_qk_fused_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(rc as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(q_handle.clone(), 0),
                    BufferArg::from_raw_parts(
                        k_handle.clone().offset_start((k0 * head_dim * 4) as u64),
                        rc * head_dim,
                    ),
                    BufferArg::from_raw_parts(q_gamma_handle.clone(), head_dim),
                    BufferArg::from_raw_parts(k_gamma_handle.clone(), head_dim),
                    BufferArg::from_raw_parts(params_handle, 4),
                );
            }
            k0 += rc;
        }
    }
}

/// Fused RMSNorm + z-gating launcher (Issue 642 F6).
///
/// Normalizes each head of `output` `[n_heads × head_dim]` via RMSNorm with
/// `gamma`, then multiplies by `silu(z[idx])` — all in a single dispatch.
/// Replaces the two-dispatch DeltaNet output path
/// (RmsNormBatched → DeltanetZGating). The `output` buffer is modified in-place.
///
/// Dispatch: `(n_heads, 1, 1)` workgroups of 256 threads (one per head).
#[cfg(feature = "cubecl_runtime")]
pub struct RmsNormZgateFusedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl RmsNormZgateFusedCubeCL {
    /// Launch fused per-head RMSNorm + z-gating.
    ///
    /// # Safety
    ///
    /// - `output_handle`: `n_heads * head_dim` f32 elements (modified in-place)
    /// - `gamma_handle`: `head_dim` f32 elements
    /// - `z_handle`: `n_heads * head_dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        output_handle: Handle,
        gamma_handle: Handle,
        z_handle: Handle,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / head_dim as f32;
        let dim_f = head_dim as f32;
        let params: &[f32] = &[inv_dim, eps, dim_f];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let total = n_heads * head_dim;

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rmsnorm_zgate_fused_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_heads as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(output_handle, total),
                BufferArg::from_raw_parts(gamma_handle, head_dim),
                BufferArg::from_raw_parts(z_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fused ResidualAdd + RMSNorm kernel (Issue 645)
// ---------------------------------------------------------------------------

/// Fused ResidualAdd + RMSNorm kernel — saves 1 dispatch per fusion site.
///
/// Computes `x[i] += residual[i]` (in-place residual add) then
/// `norm_out[i] = x[i] * inv_rms * gamma[i]` where
/// `inv_rms = 1 / sqrt(mean(x²) + eps)` — all in a single dispatch.
///
/// Replaces the two-dispatch sequence (ResidualAdd → RmsNorm) at sub-block
/// boundaries in the decode path (Issue 645). Each site saves 1 dispatch:
/// - Mid-layer (after attention/deltanet, before FFN): 64 dispatches/token
/// - Cross-layer (after FFN, before next layer's input norm): 63 dispatches/token
///
/// The residual add is folded into Phase 1 (before the squaring), so the
/// accumulated `partial_sq` reflects the post-residual values. Phase 4 reads
/// the already-updated `x[j]` (the post-residual value written in Phase 1).
///
/// ## Parameter Layout
///
/// - `x`: `[f32; dim]` — hidden state (read-write: updated with residual)
/// - `residual`: `[f32; dim]` — the layer output to add (read-only)
/// - `gamma`: `[f32; dim]` — learnable scale (read-only)
/// - `params`: `[f32; 3]` — `[inv_dim, eps, dim]` precomputed on CPU
/// - `norm_out`: `[f32; dim]` — normalized output (write-only)
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn residual_add_rmsnorm_f32(
    x: &mut [f32],
    residual: &[f32],
    gamma: &[f32],
    params: &[f32],
    norm_out: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // ── Phase 1: Residual add (in-place) + strided accumulation of x² ──
    // The residual add happens BEFORE the squaring, so partial_sq reflects
    // the post-residual values. This is the ONLY change from rmsnorm_f32.
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let val = x[i as usize] + residual[i as usize];
        x[i as usize] = val;
        partial_sq += val * val;
        i += cube_size;
    }

    // ── Phase 2: Shared memory parallel reduction (same as rmsnorm_f32) ──
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

    // ── Phase 4: Normalize and apply gamma ──
    // Reads x[j] which is already the post-residual value written in Phase 1.
    let mut j = tid;
    while j < dim {
        let val = x[j as usize];
        let g = gamma[j as usize];
        norm_out[j as usize] = val * inv_rms * g;
        j += cube_size;
    }
}

/// Fused ResidualAdd + RMSNorm launcher (Issue 645).
///
/// Computes `x[i] += residual[i]` then `norm_out = rmsnorm(x, gamma, eps)`
/// in a single dispatch. Saves 1 dispatch per fusion site vs separate
/// `ResidualAddCubeCL::launch` + `RmsNormCubeCL::launch`.
///
/// Dispatch: `(1, 1, 1)` workgroups of 256 threads (strided access).
#[cfg(feature = "cubecl_runtime")]
pub struct ResidualAddRmsNormCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ResidualAddRmsNormCubeCL {
    /// Launch fused ResidualAdd + RMSNorm.
    ///
    /// Updates `x_handle` in-place with `x[i] += residual[i]`, then writes
    /// `norm_out_handle[i] = x[i] * inv_rms * gamma[i]`.
    ///
    /// # Safety
    ///
    /// - `x_handle`: `dim` f32 elements (modified in-place with residual add)
    /// - `residual_handle`: `dim` f32 elements (read-only)
    /// - `gamma_handle`: `dim` f32 elements (read-only)
    /// - `norm_out_handle`: `dim` f32 elements (write-only)
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        residual_handle: Handle,
        gamma_handle: Handle,
        norm_out_handle: Handle,
        dim: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        let params: &[f32] = &[inv_dim, eps, dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));

        debug_assert_binding_at_least(&x_handle, dim, "ResidualAddRmsNormCubeCL::x");
        debug_assert_binding_at_least(&residual_handle, dim, "ResidualAddRmsNormCubeCL::residual");
        debug_assert_binding_at_least(&norm_out_handle, dim, "ResidualAddRmsNormCubeCL::norm_out");

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            residual_add_rmsnorm_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, dim),
                BufferArg::from_raw_parts(residual_handle, dim),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(norm_out_handle, dim),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Mean-centered LayerNorm kernels (plan 611 S1a — the T7 GAP: the norms
// family was RMS-shaped only; the encoder lane's `layer_norm_nobias_into`
// semantics — mean-centered, bias-free — had no CubeCL form)
// ---------------------------------------------------------------------------

/// CubeCL batched mean-centered LayerNorm — `[rows × dim]`, one workgroup
/// per row. Computes `y = (x − μ) / sqrt(var + eps) · gamma` where `μ =
/// mean(x)` and `var = mean((x − μ)²)` — the CPU lane's TWO-PASS form
/// (`ops::layer_norm_nobias_into`: candle materializes `centered`, squares,
/// then means). Plan 611 S1a shipped the one-pass `E[x²] − μ²` form and the
/// issue-016 probe priced the divergence: the residual stream reaches
/// ±4000 by mid-stack, and where a row carries a DC component the
/// cancellation factor E[x²]/var amplifies the f32 rounding into exactly
/// the long-sequence G5 outliers (probe: relative drift growing 2.6e-6 →
/// 7.7e-5 with depth, fixtures at seq ≥ 400 blowing the 1e-3 prob gate
/// while short fixtures sat at 2e-6). The second walk costs one extra
/// strided read per row; the reduction ORDER still differs from the CPU
/// lane's sequential `vec_sum` (strided-256 tree here) — that residual is
/// the same class as the matmuls' and prices at ~1e-6 prob drift.
/// Bias-free by contract — the encoder lane pins "NO bias tensors
/// anywhere in the encoder"; a biased form is a different kernel.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; rows * dim]` — row-major batched input.
/// - `gamma`: `[f32; dim]` — learnable scale (shared across rows).
/// - `params`: `[f32; 3]` — `[inv_dim, eps, dim]` precomputed on CPU
///   (the Issue-639 discipline: `dim` travels in params, never read from
///   `input.len()`, so an oversized backing allocation is harmless).
///   `inv_dim` is the CPU lane's exact candle scale `(1f64 / d) as f32` —
///   NOT `1f32 / d` (one ulp apart at d = 1152).
/// - `output`: `[f32; rows * dim]` — normalized output.
///
/// ## Dispatch
///
/// `CubeCount::Static(rows, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn layernorm_mean_batched_f32(
    input: &[f32],
    gamma: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let row = CUBE_POS_X;
    let row_offset = row * dim;

    // ── Pass 1: strided accumulation of Σx for this row, then the tree
    // reduce; slot 0 becomes μ ──
    let mut partial_sum = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        partial_sum += input[(row_offset + i) as usize];
        i += cube_size;
    }
    let mut smem_sum = Shared::<[f32]>::new_slice(256usize);
    smem_sum[tid as usize] = partial_sum;
    sync_cube();
    if tid < 128u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        smem_sum[tid as usize] = smem_sum[tid as usize] + smem_sum[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        smem_sum[0usize] = (smem_sum[0usize] + smem_sum[1usize]) * inv_dim;
    }
    sync_cube();
    let mean = smem_sum[0usize];

    // ── Pass 2: strided accumulation of Σ(x − μ)² for this row, then the
    // tree reduce; slot 0 becomes inv_std (the CPU lane's var =
    // mean(centered²), the SAME centered values, a different summation
    // order — the documented residual) ──
    let mut partial_sq = f32::new(0.0f32);
    let mut j = tid;
    while j < dim {
        let c = input[(row_offset + j) as usize] - mean;
        partial_sq += c * c;
        j += cube_size;
    }
    let mut smem_sq = Shared::<[f32]>::new_slice(256usize);
    smem_sq[tid as usize] = partial_sq;
    sync_cube();
    if tid < 128u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        smem_sq[tid as usize] = smem_sq[tid as usize] + smem_sq[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        let var = (smem_sq[0usize] + smem_sq[1usize]) * inv_dim;
        smem_sq[0usize] = f32::new(1.0f32) / (var + eps).sqrt();
    }
    sync_cube();
    let inv_std = smem_sq[0usize];

    // ── Phase 4: normalize and apply gamma for this row ──
    let mut k = tid;
    while k < dim {
        let x = input[(row_offset + k) as usize];
        let g = gamma[k as usize];
        output[(row_offset + k) as usize] = (x - mean) * inv_std * g;
        k += cube_size;
    }
}

/// CubeCL batched mean-centered LayerNorm launcher (plan 611 S1a).
///
/// Processes `[rows × dim]` in one dispatch — one workgroup per row, the
/// `RmsNormBatchedCubeCL` pattern. Bias-free by contract (the encoder
/// lane's `layer_norm_nobias_into` semantics).
#[cfg(feature = "cubecl_runtime")]
pub struct LayerNormMeanBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl LayerNormMeanBatchedCubeCL {
    /// Launch batched mean-centered LayerNorm over `[rows × dim]`.
    ///
    /// Dispatch: `(rows, 1, 1)` workgroups of 256 threads (one per row).
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `input_handle`: `rows * dim` f32 elements
    /// - `gamma_handle`: `dim` f32 elements
    /// - `output_handle`: `rows * dim` f32 elements
    ///
    /// `rows` and `dim` must be > 0. `eps` must be > 0.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        output_handle: Handle,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        // The CPU lane's exact candle mean scale (`ops::layer_norm_nobias_into`):
        // `(1f64 / d) as f32`, not `1f32 / d` — one ulp apart at d = 1152.
        let inv_dim = (1.0f64 / dim as f64) as f32;
        let params: &[f32] = &[inv_dim, eps, dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));

        let row_elems = rows * dim;
        debug_assert_binding_at_least(&input_handle, row_elems, "LayerNormMeanBatched::input");
        debug_assert_binding_at_least(&output_handle, row_elems, "LayerNormMeanBatched::output");

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            layernorm_mean_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, row_elems),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(output_handle, row_elems),
            );
        }
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use crate::cubecl_runtime::ActiveRuntime;

    use crate::cubecl_runtime::CubeCLContext;

    use super::*;

    /// CPU reference RMSNorm with gamma.
    ///
    /// Computes `output[i] = x[i] * inv_rms * gamma[i]` where
    /// `inv_rms = 1 / sqrt(mean(x²) + eps)`.
    fn rmsnorm_gamma_cpu(data: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let dim = data.len();
        let sum_sq: f32 = data.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        data.iter()
            .zip(gamma.iter())
            .map(|(&x, &g)| x * inv_rms * g)
            .collect()
    }

    // ── RMSNorm tests ──

    /// Verify RMSNorm with gamma=1 produces correctly normalized output.
    #[test]
    fn test_rmsnorm_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let dim = 256usize;
        let eps = 1e-6f32;

        // Input: constant 2.0 vector
        let input: Vec<f32> = vec![2.0f32; dim];
        // Gamma: all ones (identity scale)
        let gamma: Vec<f32> = vec![1.0f32; dim];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        let expected = rmsnorm_gamma_cpu(&input, &gamma, eps);

        assert_eq!(output.len(), dim);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }

    /// Verify RMSNorm with hardcoded known values.
    #[test]
    fn test_rmsnorm_known_values() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let dim = 8usize;
        let eps = 1e-6f32;

        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma: &[f32] = &[1.0; 8];

        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(gamma));
        let output_handle = client.empty(dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        let expected = rmsnorm_gamma_cpu(input, gamma, eps);

        assert_eq!(output.len(), dim);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }

    /// Verify RMSNorm with non-trivial gamma scales the output.
    #[test]
    fn test_rmsnorm_gamma() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let dim = 16usize;
        let eps = 1e-6f32;

        let input: Vec<f32> = vec![3.0f32; dim];
        // Gamma: 0.5 for all elements → output should be half of identity-rmsnorm
        let gamma: Vec<f32> = vec![0.5f32; dim];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        let expected = rmsnorm_gamma_cpu(&input, &gamma, eps);

        assert_eq!(output.len(), dim);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }

    /// Verify RMSNorm with dim > 256 (strided access, Gemma 2 n_embd=2048).
    #[test]
    fn test_rmsnorm_large_dim() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let dim = 2048usize; // Gemma 2 n_embd
        let eps = 1e-6f32;

        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let gamma: Vec<f32> = vec![1.0f32; dim];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        let expected = rmsnorm_gamma_cpu(&input, &gamma, eps);

        assert_eq!(output.len(), dim);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-4,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }

    /// CPU reference batched RMSNorm.
    ///
    /// Mirrors `katgpt_speculative::weaver::rmsnorm_into` applied to each row
    /// of a `[seq_len × dim]` buffer. For each row r:
    /// `output[r*dim + j] = input[r*dim + j] * inv_rms_r * gamma[j]`
    fn rmsnorm_batched_cpu(input: &[f32], gamma: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        let batch = input.len() / dim;
        let mut output = vec![0.0f32; input.len()];
        for b in 0..batch {
            let offset = b * dim;
            let row = &input[offset..offset + dim];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
            for j in 0..dim {
                output[offset + j] = row[j] * inv_rms * gamma[j];
            }
        }
        output
    }

    // ── Batched RMSNorm tests (Plan 436 T2.2) ──

    /// Plan 436 T2.2: Batched RMSNorm parity test.
    ///
    /// Validates the batched kernel against the CPU reference at Weaver-shaped
    /// dimensions (seq_len=5 = max_depth+1, dim=128). Uses identity gamma to
    /// isolate the normalization path. This exercises the `CUBE_POS_X` row
    /// indexing and per-row shared-memory reduction that the single-row tests
    /// don't cover.
    #[test]
    fn test_rmsnorm_batched_weaver_shape() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let seq_len = 5usize; // Weaver max_depth + 1
        let dim = 128usize; // small hidden for fast test (real: 2048)
        let eps = 1e-6f32; // Weaver rms_eps

        // Deterministic LCG input (matches T2.1 batched GEMV test convention).
        let mut seed = 42u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let input: Vec<f32> = (0..seq_len * dim).map(|_| lcg()).collect();
        let gamma: Vec<f32> = vec![1.0f32; dim]; // identity scale

        // CPU reference.
        let expected = rmsnorm_batched_cpu(&input, &gamma, dim, eps);

        // Upload to GPU.
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(seq_len * dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                seq_len,
                dim,
                eps,
            );
        }

        let bytes = client
            .read_one(output_handle)
            .expect("should read batched RMSNorm output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), seq_len * dim);
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-4,
                "batched RMSNorm element {i}: expected {exp}, got {got} (err {err})"
            );
        }
        println!("batched RMSNorm ({seq_len}×{dim}): max_err = {max_err}");
    }

    /// Plan 436 T2.2: Batched RMSNorm with non-trivial gamma scaling.
    ///
    /// Verifies the gamma multiply path works in batched mode. Gamma = 0.5
    /// for all elements → each output should be exactly half of the
    /// identity-gamma output. Catches bugs where gamma indexing is wrong in
    /// the per-row dispatch.
    #[test]
    fn test_rmsnorm_batched_gamma() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let seq_len = 5usize;
        let dim = 128usize;
        let eps = 1e-6f32;

        let mut seed = 12345u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let input: Vec<f32> = (0..seq_len * dim).map(|_| lcg()).collect();
        let gamma: Vec<f32> = vec![0.5f32; dim]; // non-trivial scale

        let expected = rmsnorm_batched_cpu(&input, &gamma, dim, eps);

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(seq_len * dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                seq_len,
                dim,
                eps,
            );
        }

        let bytes = client
            .read_one(output_handle)
            .expect("should read batched RMSNorm output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), seq_len * dim);
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-4,
                "batched RMSNorm (gamma) element {i}: expected {exp}, got {got} (err {err})"
            );
        }
        println!("batched RMSNorm gamma ({seq_len}×{dim}): max_err = {max_err}");
    }

    // ── Fused Q+K RMSNorm tests (Issue 642 F4) ──

    /// CPU reference for batched RMSNorm with different gamma per buffer.
    fn rmsnorm_qk_fused_cpu(
        q: &[f32],
        k: &[f32],
        q_gamma: &[f32],
        k_gamma: &[f32],
        n_q: usize,
        n_kv: usize,
        dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let q_out = rmsnorm_batched_cpu(q, q_gamma, dim, eps);
        let k_out = rmsnorm_batched_cpu(k, k_gamma, dim, eps);
        debug_assert_eq!(q_out.len(), n_q * dim);
        debug_assert_eq!(k_out.len(), n_kv * dim);
        (q_out, k_out)
    }

    /// Verify fused Q+K RMSNorm matches two separate RMSNormBatched calls.
    #[test]
    fn test_rmsnorm_qk_fused_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_q = 8usize;
        let n_kv = 4usize; // GQA: fewer KV heads
        let dim = 128usize;
        let eps = 1e-6f32;

        let mut seed = 999u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let q_input: Vec<f32> = (0..n_q * dim).map(|_| lcg()).collect();
        let k_input: Vec<f32> = (0..n_kv * dim).map(|_| lcg()).collect();
        let q_gamma: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let k_gamma: Vec<f32> = (0..dim).map(|_| lcg()).collect();

        // CPU reference
        let (q_expected, k_expected) =
            rmsnorm_qk_fused_cpu(&q_input, &k_input, &q_gamma, &k_gamma, n_q, n_kv, dim, eps);

        // GPU fused path (in-place)
        let q_handle = client.create_from_slice(f32::as_bytes(&q_input));
        let k_handle = client.create_from_slice(f32::as_bytes(&k_input));
        let q_gamma_handle = client.create_from_slice(f32::as_bytes(&q_gamma));
        let k_gamma_handle = client.create_from_slice(f32::as_bytes(&k_gamma));

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormQkFusedCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle.clone(),
                k_handle.clone(),
                q_gamma_handle,
                k_gamma_handle,
                n_q,
                n_kv,
                dim,
                eps,
            );
        }

        let q_bytes = client.read_one(q_handle).expect("should read Q output");
        let q_output = f32::from_bytes(&q_bytes);
        let k_bytes = client.read_one(k_handle).expect("should read K output");
        let k_output = f32::from_bytes(&k_bytes);

        assert_eq!(q_output.len(), n_q * dim);
        assert_eq!(k_output.len(), n_kv * dim);

        let mut max_q_err = 0.0f32;
        for (i, (&exp, &got)) in q_expected.iter().zip(q_output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_q_err {
                max_q_err = err;
            }
            assert!(
                err < 1e-4,
                "fused Q+K RMSNorm Q element {i}: expected {exp}, got {got} (err {err})"
            );
        }

        let mut max_k_err = 0.0f32;
        for (i, (&exp, &got)) in k_expected.iter().zip(k_output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_k_err {
                max_k_err = err;
            }
            assert!(
                err < 1e-4,
                "fused Q+K RMSNorm K element {i}: expected {exp}, got {got} (err {err})"
            );
        }

        println!(
            "fused Q+K RMSNorm ({n_q}+{n_kv} heads × {dim}): max_q_err = {max_q_err}, max_k_err = {max_k_err}"
        );
    }

    // ── Fused RMSNorm + z-gating tests (Issue 642 F6) ──

    /// CPU reference for fused per-head RMSNorm + z-gating.
    ///
    /// Mirrors `rmsnorm_batched_cpu` followed by element-wise `*= silu(z)`.
    fn rmsnorm_zgate_fused_cpu(
        input: &[f32],
        gamma: &[f32],
        z: &[f32],
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        // Step 1: RMSNorm each head
        let mut out = rmsnorm_batched_cpu(input, gamma, head_dim, eps);
        debug_assert_eq!(out.len(), n_heads * head_dim);
        // Step 2: z-gating: out[i] *= silu(z[i])
        for i in 0..out.len() {
            let z_val = z[i];
            let sig = 1.0 / (1.0 + (-z_val).exp());
            out[i] *= z_val * sig;
        }
        out
    }

    /// Verify fused RMSNorm + z-gating matches separate RMSNormBatched + ZGating.
    #[test]
    fn test_rmsnorm_zgate_fused_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-27B DeltaNet shape: n_v_heads=4, head_dim=128
        let n_heads = 4usize;
        let head_dim = 128usize;
        let eps = 1e-6f32;
        let total = n_heads * head_dim;

        let mut seed = 12345u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let input: Vec<f32> = (0..total).map(|_| lcg()).collect();
        let gamma: Vec<f32> = (0..head_dim).map(|_| lcg()).collect();
        let z: Vec<f32> = (0..total).map(|_| lcg()).collect();

        // CPU reference: rmsnorm then z-gate
        let expected = rmsnorm_zgate_fused_cpu(&input, &gamma, &z, n_heads, head_dim, eps);

        // GPU fused path (in-place on output)
        let output_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let z_handle = client.create_from_slice(f32::as_bytes(&z));

        // SAFETY: Buffers are correctly sized.
        unsafe {
            RmsNormZgateFusedCubeCL::launch::<ActiveRuntime>(
                &client,
                output_handle.clone(),
                gamma_handle,
                z_handle,
                n_heads,
                head_dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), total);

        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-4,
                "fused RMSNorm+z-gate element {i}: expected {exp}, got {got} (err {err})"
            );
        }

        println!("fused RMSNorm+z-gate ({n_heads} heads × {head_dim}): max_err = {max_err}");
    }

    // ── Residual add tests ──

    /// Verify simple a + b residual addition.
    #[test]
    fn test_residual_add_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let output_handle = client.empty(n * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            ResidualAddCubeCL::launch::<ActiveRuntime>(
                &client,
                a_handle,
                b_handle,
                output_handle.clone(),
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), n);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }

    /// Verify adding zeros doesn't change the input.
    #[test]
    fn test_residual_add_zero() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n = 128usize;
        let a: Vec<f32> = vec![42.0f32; n];
        let b: Vec<f32> = vec![0.0f32; n];

        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let output_handle = client.empty(n * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            ResidualAddCubeCL::launch::<ActiveRuntime>(
                &client,
                a_handle,
                b_handle,
                output_handle.clone(),
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), n);
        for (i, &got) in output.iter().enumerate() {
            assert!(
                (got - 42.0f32).abs() < 1e-5,
                "element {i}: expected 42.0, got {got}"
            );
        }
    }

    // ── Fused ResidualAdd + RMSNorm tests (Issue 645) ──

    /// CPU reference for fused ResidualAdd + RMSNorm.
    ///
    /// Mirrors `ResidualAddCubeCL` (a + b) followed by `RmsNormCubeCL`.
    /// Returns (x_post_residual, norm_out).
    fn residual_add_rmsnorm_cpu(
        x: &[f32],
        residual: &[f32],
        gamma: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let _dim = x.len();
        let x_post: Vec<f32> = x.iter().zip(residual.iter()).map(|(&a, &b)| a + b).collect();
        let norm_out = rmsnorm_gamma_cpu(&x_post, gamma, eps);
        (x_post, norm_out)
    }

    /// Verify fused ResidualAdd + RMSNorm matches separate dispatches.
    ///
    /// G1 gate for Issue 645: the fused kernel must produce bit-identical
    /// results (within float tolerance) to the two-dispatch sequence
    /// (ResidualAdd → RmsNorm).
    #[test]
    fn test_residual_add_rmsnorm_fused_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-27B hidden dim: 5120 (tests strided access, dim > 256)
        let dim = 5120usize;
        let eps = 1e-6f32;

        let mut seed = 777u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let x_input: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let residual: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let gamma: Vec<f32> = (0..dim).map(|_| lcg()).collect();

        // CPU reference
        let (x_expected, norm_expected) =
            residual_add_rmsnorm_cpu(&x_input, &residual, &gamma, eps);

        // GPU fused path
        let x_handle = client.create_from_slice(f32::as_bytes(&x_input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let norm_out_handle = client.empty(dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            ResidualAddRmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                residual_handle,
                gamma_handle,
                norm_out_handle.clone(),
                dim,
                eps,
            );
        }

        let x_bytes = client.read_one(x_handle).expect("should read x output");
        let x_output = f32::from_bytes(&x_bytes);
        let norm_bytes = client.read_one(norm_out_handle).expect("should read norm output");
        let norm_output = f32::from_bytes(&norm_bytes);

        assert_eq!(x_output.len(), dim);
        assert_eq!(norm_output.len(), dim);

        // Check x was updated with residual add
        let mut max_x_err = 0.0f32;
        for (i, (&exp, &got)) in x_expected.iter().zip(x_output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_x_err {
                max_x_err = err;
            }
            assert!(
                err < 1e-5,
                "fused ResAdd+RMSNorm x element {i}: expected {exp}, got {got} (err {err})"
            );
        }

        // Check norm_out matches separate RMSNorm
        let mut max_norm_err = 0.0f32;
        for (i, (&exp, &got)) in norm_expected.iter().zip(norm_output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_norm_err {
                max_norm_err = err;
            }
            assert!(
                err < 1e-4,
                "fused ResAdd+RMSNorm norm element {i}: expected {exp}, got {got} (err {err})"
            );
        }

        println!(
            "fused ResAdd+RMSNorm (dim={dim}): max_x_err = {max_x_err}, max_norm_err = {max_norm_err}"
        );
    }

    /// Verify fused ResidualAdd + RMSNorm with dim=256 (single stride, no wrapping).
    #[test]
    fn test_residual_add_rmsnorm_fused_dim256() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let dim = 256usize;
        let eps = 1e-6f32;

        let x_input: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let residual: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01 - 0.5).sin()).collect();
        let gamma: Vec<f32> = vec![1.0f32; dim]; // identity scale

        let (x_expected, norm_expected) =
            residual_add_rmsnorm_cpu(&x_input, &residual, &gamma, eps);

        let x_handle = client.create_from_slice(f32::as_bytes(&x_input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let norm_out_handle = client.empty(dim * core::mem::size_of::<f32>());

        unsafe {
            ResidualAddRmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                x_handle.clone(),
                residual_handle,
                gamma_handle,
                norm_out_handle.clone(),
                dim,
                eps,
            );
        }

        let x_bytes = client.read_one(x_handle).expect("should read x");
        let x_output = f32::from_bytes(&x_bytes);
        let norm_bytes = client.read_one(norm_out_handle).expect("should read norm");
        let norm_output = f32::from_bytes(&norm_bytes);

        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in norm_expected.iter().zip(norm_output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(err < 1e-5, "dim256 norm element {i}: expected {exp}, got {got}");
        }
        // Also verify x was updated
        for (i, (&exp, &got)) in x_expected.iter().zip(x_output.iter()).enumerate() {
            assert!((exp - got).abs() < 1e-5, "dim256 x element {i}: expected {exp}, got {got}");
        }

        println!("fused ResAdd+RMSNorm (dim={dim}): max_norm_err = {max_err}");
    }

    // ── Mean-centered LayerNorm tests (plan 611 S1a) ──

    /// CPU reference, bias-free mean-centered LayerNorm — the
    /// `layer_norm_nobias_into` semantics: `y = (x − μ) / sqrt(var + eps)
    /// · gamma`, two-pass (μ first, then Σ(x−μ)² — the reference order;
    /// the kernel's one-pass E[x²]−μ² form differs only in reduction
    /// order, inside the 1e-5 tolerance).
    fn layernorm_mean_cpu(data: &[f32], gamma: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        let rows = data.len() / dim;
        let mut out = vec![0.0f32; data.len()];
        for r in 0..rows {
            let off = r * dim;
            let mean: f32 = data[off..off + dim].iter().sum::<f32>() / dim as f32;
            let var: f32 = data[off..off + dim]
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f32>()
                / dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for (d, v) in data[off..off + dim].iter().enumerate() {
                out[off + d] = (v - mean) * inv_std * gamma[d];
            }
        }
        out
    }

    /// Verify mean-centered LN normalizes each row to zero mean, unit
    /// variance (gamma = 1), across a multi-row batch with dim > 256 so
    /// the strided walk is exercised.
    #[test]
    fn test_layernorm_mean_identity_batched() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let (rows, dim) = (7usize, 1024usize);
        let eps = 1e-5f32;
        // Deterministic pseudo-random rows (xorshift), mean ≈ 3, varied
        // scale per row — a zero-mean input would make the mean subtraction
        // untestable.
        let mut s = 0x2545F4914F6CDD1Du64;
        let input: Vec<f32> = (0..rows * dim)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                3.0 + 5.0 * ((i % (7 + i / dim)) as f32) * ((s % 2000) as f32 / 1000.0 - 1.0)
            })
            .collect();
        let gamma: Vec<f32> = vec![1.0f32; dim];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(rows * dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            LayerNormMeanBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                rows,
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);
        let expected = layernorm_mean_cpu(&input, &gamma, dim, eps);

        assert_eq!(output.len(), rows * dim);
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-4, "element {i}: expected {exp}, got {got}");
        }

        // The NORMALIZATION claim, on row 0: output mean ≈ 0, var ≈ 1.
        let out_mean: f32 = output[0..dim].iter().sum::<f32>() / dim as f32;
        let out_var: f32 = output[0..dim].iter().map(|v| v * v).sum::<f32>() / dim as f32;
        assert!(out_mean.abs() < 1e-4, "row 0 mean {out_mean} ≠ 0");
        assert!((out_var - 1.0).abs() < 1e-2, "row 0 var {out_var} ≠ 1");
        println!("mean-LN batched ({rows}×{dim}): max_err = {max_err:.2e}");
    }

    /// Verify mean-centered LN with a non-trivial gamma, small geometry.
    #[test]
    fn test_layernorm_mean_gamma() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let (rows, dim) = (3usize, 64usize);
        let eps = 1e-5f32;
        let input: Vec<f32> = (0..rows * dim)
            .map(|i| ((i * 37 + 11) % 23) as f32 - 8.0)
            .collect();
        let gamma: Vec<f32> = (0..dim).map(|d| 0.5 + (d % 5) as f32 * 0.25).collect();

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(rows * dim * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            LayerNormMeanBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                output_handle.clone(),
                rows,
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);
        let expected = layernorm_mean_cpu(&input, &gamma, dim, eps);

        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
    }
}
