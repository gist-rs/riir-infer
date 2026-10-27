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
fn rmsnorm_f32(input: &[f32], gamma: &[f32], params: &[f32], output: &mut [f32]) {
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
fn rmsnorm_batched_f32(input: &[f32], gamma: &[f32], params: &[f32], output: &mut [f32]) {
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
        let g = if is_q {
            q_gamma[j as usize]
        } else {
            k_gamma[j as usize]
        };
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
fn rmsnorm_zgate_fused_f32(output: &mut [f32], gamma: &[f32], z: &[f32], params: &[f32]) {
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

/// Issue 1143 — [`residual_add_rmsnorm_f32`] with 4 loads in flight per lane.
///
/// The decode norm is ONE 256-thread workgroup over `dim` (5120 on Bonsai):
/// each lane walks `dim / 256` = 20 strided element pairs, one DRAM round
/// trip at a time — the Issue-1142 load-latency shape on a tiny grid. This
/// variant issues four iterations' loads before folding them into
/// `partial_sq` in the SAME i order with the SAME `+=` (Phase 1), and hoists
/// four `x`/`gamma` pairs ahead of the stores (Phase 4). Phases 2-3 are
/// line-for-line the incumbent, so every lane's accumulation sequence and
/// every output expression is unchanged — bit-identical by construction
/// (`test_residual_add_rmsnorm_mlp4_bit_identity`). Depth measured on idle
/// bmo (Bonsai-2): 4 → +7.3% e2e (3/3); 8 → 0.4% SLOWER than 4 (3/3, not
/// taken). Each hoisted index is
/// distinct from every stored index in its group, so hoisting the `x` loads
/// over the `x` stores reads the same values.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn residual_add_rmsnorm_f32_mlp4(
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

    // ── Phase 1: residual add + x², four iterations' loads in flight ──
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i + 3u32 * cube_size < dim {
        let i1 = i + cube_size;
        let i2 = i + 2u32 * cube_size;
        let i3 = i + 3u32 * cube_size;
        let x0 = x[i as usize];
        let x1 = x[i1 as usize];
        let x2 = x[i2 as usize];
        let x3 = x[i3 as usize];
        let r0 = residual[i as usize];
        let r1 = residual[i1 as usize];
        let r2 = residual[i2 as usize];
        let r3 = residual[i3 as usize];
        let v0 = x0 + r0;
        let v1 = x1 + r1;
        let v2 = x2 + r2;
        let v3 = x3 + r3;
        x[i as usize] = v0;
        x[i1 as usize] = v1;
        x[i2 as usize] = v2;
        x[i3 as usize] = v3;
        partial_sq += v0 * v0;
        partial_sq += v1 * v1;
        partial_sq += v2 * v2;
        partial_sq += v3 * v3;
        i += 4u32 * cube_size;
    }
    while i < dim {
        let val = x[i as usize] + residual[i as usize];
        x[i as usize] = val;
        partial_sq += val * val;
        i += cube_size;
    }

    // ── Phase 2: shared-memory tree reduction (identical) ──
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

    // ── Phase 3: broadcast inv_rms (identical) ──
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        smem[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
    sync_cube();

    let inv_rms = smem[0usize];

    // ── Phase 4: normalize + gamma, four x/gamma pairs in flight ──
    let mut j = tid;
    while j + 3u32 * cube_size < dim {
        let j1 = j + cube_size;
        let j2 = j + 2u32 * cube_size;
        let j3 = j + 3u32 * cube_size;
        let a0 = x[j as usize];
        let a1 = x[j1 as usize];
        let a2 = x[j2 as usize];
        let a3 = x[j3 as usize];
        let g0 = gamma[j as usize];
        let g1 = gamma[j1 as usize];
        let g2 = gamma[j2 as usize];
        let g3 = gamma[j3 as usize];
        norm_out[j as usize] = a0 * inv_rms * g0;
        norm_out[j1 as usize] = a1 * inv_rms * g1;
        norm_out[j2 as usize] = a2 * inv_rms * g2;
        norm_out[j3 as usize] = a3 * inv_rms * g3;
        j += 4u32 * cube_size;
    }
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

        // SAFETY: Caller guarantees correct buffer sizes. Issue 1143: the
        // 4-deep variant is bit-identical; `RIIR_NORM_MLP` selects it.
        unsafe {
            let x_arg = BufferArg::from_raw_parts(x_handle, dim);
            let residual_arg = BufferArg::from_raw_parts(residual_handle, dim);
            let gamma_arg = BufferArg::from_raw_parts(gamma_handle, dim);
            let params_arg = BufferArg::from_raw_parts(params_handle, 3);
            let norm_out_arg = BufferArg::from_raw_parts(norm_out_handle, dim);
            match norm_mlp_enabled() {
                true => residual_add_rmsnorm_f32_mlp4::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(256),
                    x_arg,
                    residual_arg,
                    gamma_arg,
                    params_arg,
                    norm_out_arg,
                ),
                false => residual_add_rmsnorm_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(256),
                    x_arg,
                    residual_arg,
                    gamma_arg,
                    params_arg,
                    norm_out_arg,
                ),
            }
        }
    }
}

/// Issue 1143: the decode `ResidualAddRmsNormCubeCL` dispatches
/// [`residual_add_rmsnorm_f32_mlp4`] (4 loads in flight per lane,
/// bit-identical) unless `RIIR_NORM_MLP` is set to anything but 1/on/true —
/// the `dense_pair_env_semantics` contract (unset = ON).
/// [`set_norm_mlp`] is the harness arm toggle.
#[cfg(feature = "cubecl_runtime")]
static USE_NORM_MLP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[cfg(feature = "cubecl_runtime")]
static NORM_MLP_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

#[cfg(feature = "cubecl_runtime")]
fn norm_mlp_enabled() -> bool {
    if NORM_MLP_INITIALIZED
        .set(crate::gemv_cubecl::dense_pair_env_semantics(std::env::var(
            "RIIR_NORM_MLP",
        )))
        .is_ok()
    {
        USE_NORM_MLP.store(
            NORM_MLP_INITIALIZED.get().copied().unwrap_or(true),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_NORM_MLP.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the 4-deep residual-add norm on/off (overrides `RIIR_NORM_MLP`).
#[cfg(feature = "cubecl_runtime")]
pub fn set_norm_mlp(on: bool) {
    let _ = NORM_MLP_INITIALIZED.set(on);
    USE_NORM_MLP.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Fused ResidualAdd + RMSNorm + forward-rotate staging (Issue 1134).
///
/// Phases 1-4 are line-for-line [`residual_add_rmsnorm_f32`] (the G1
/// contract: the norm outputs are the same bits the incumbent pair
/// produces). Phase 5 then stages the Hadamard rotation INTO the same
/// launch: for each `hblock` chunk, `smem[i] = norm_out[base+i] *
/// signs[base+i]` (sign FIRST — `rotate_forward_inplace`'s order, the
/// decode single-row case has `row_off == base`), then the SAME inline
/// butterfly as `fwht_forward_copy_f32` (same `inv_sqrt2` literal, same
/// step loop, same sync placement), then a strided write to `rot_out`.
/// One launch replaces `ResidualAddRmsNormCubeCL::launch` +
/// `RotationCubeCL::launch_forward_copy`; the single 256-thread workgroup
/// walks the `dim / hblock` chunks serially (compute-light: the whole
/// phase-5 is ~n·log2(hblock) fused multiply-adds against a ~17 us saved
/// dispatch floor).
///
/// ## Parameter Layout
///
/// - `params`: `[f32; 5]` — `[inv_dim, eps, dim, width, hblock]` (decode
///   single-row: `width == dim`; the guard refuses otherwise).
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)` — the same shape
/// as the norm twin, so the serial chunk walk is the only new cost.
///
/// ## Geometry guard (caller-side)
///
/// `dim % hblock == 0`, `hblock <= 1024` (MAX_FWHT_BLOCK), `width ==
/// dim`. A miss falls back to the incumbent two-launch chain.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
#[allow(clippy::approx_constant)]
fn residual_add_rmsnorm_rotate_f32(
    x: &mut [f32],
    residual: &[f32],
    gamma: &[f32],
    signs: &[f32],
    params: &[f32],
    norm_out: &mut [f32],
    rot_out: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let hblock = params[4usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // ── Phase 1: Residual add (in-place) + strided x² accumulation ──
    // (identical to residual_add_rmsnorm_f32)
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let val = x[i as usize] + residual[i as usize];
        x[i as usize] = val;
        partial_sq += val * val;
        i += cube_size;
    }

    // ── Phase 2: shared-memory tree reduction (identical) ──
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

    // ── Phase 3: broadcast inv_rms (identical) ──
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        smem[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
    sync_cube();

    let inv_rms = smem[0usize];

    // ── Phase 4: normalize + gamma (identical) ──
    let mut j = tid;
    while j < dim {
        let val = x[j as usize] * inv_rms * gamma[j as usize];
        norm_out[j as usize] = val;
        j += cube_size;
    }

    // ── Phase 5: the rotate epilogue — sign FIRST, then the inline
    // butterfly, chunk by chunk (the fwht_forward_copy_f32 twin; the decode
    // single-row case has row_off == base). MAX_FWHT_BLOCK-sized smem (the
    // macro slice; 1024 f32 = 4 KB) — the geometry guard keeps hblock
    // within it. ──
    let mut smem_rot = Shared::<[f32]>::new_slice(1024usize);
    let inv_sqrt2 = f32::new(0.70710678f32);
    let n_chunks = dim / hblock;
    let mut c = 0u32;
    while c < n_chunks {
        let base = c * hblock;
        let mut i = tid;
        while i < hblock {
            smem_rot[i as usize] = norm_out[(base + i) as usize] * signs[(base + i) as usize];
            i += cube_size;
        }
        sync_cube();
        let mut step = 2u32;
        while step <= hblock {
            let half = step / 2u32;
            let mut bi = tid;
            while bi < hblock / 2u32 {
                let blk = (bi / half) * step;
                let off = bi % half;
                let a = smem_rot[(blk + off) as usize];
                let b = smem_rot[(blk + off + half) as usize];
                smem_rot[(blk + off) as usize] = (a + b) * inv_sqrt2;
                smem_rot[(blk + off + half) as usize] = (a - b) * inv_sqrt2;
                bi += cube_size;
            }
            sync_cube();
            step = step * 2u32;
        }
        let mut j2 = tid;
        while j2 < hblock {
            rot_out[(base + j2) as usize] = smem_rot[j2 as usize];
            j2 += cube_size;
        }
        sync_cube();
        c += 1u32;
    }
}

/// Fused ResidualAdd + RMSNorm + forward-rotate staging launcher (Issue 1134).
///
/// Replaces `ResidualAddRmsNormCubeCL::launch` +
/// `RotationCubeCL::launch_forward_copy` with one dispatch in the folded
/// (`bonsai2_hadamard`) decode posture. Phases 1-4 are the incumbent norm
/// bit-for-bit; phase 5 replicates `fwht_forward_copy_f32` exactly (sign
/// first, same butterfly order) — the fused outputs are to_bits-equal to
/// the two-launch chain, so only the REACH counter proves the toggle fired
/// (the Bench-768 vacuous-guard pattern).
///
/// # Safety
///
/// - `x_handle`: `dim` f32 (modified in-place with the residual add)
/// - `residual_handle` / `gamma_handle` / `signs_handle`: `dim` f32 each
///   (read-only; `signs` is the width-`dim` sign vector)
/// - `norm_out_handle` / `rot_out_handle`: `dim` f32 each (write-only)
/// - the caller-side geometry guard: `dim % hblock == 0`, `hblock <=
///   1024`, `width == dim`
#[cfg(feature = "cubecl_runtime")]
pub struct ResidualAddRmsNormRotateCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ResidualAddRmsNormRotateCubeCL {
    /// Launch fused ResidualAdd + RMSNorm + forward-rotate staging.
    ///
    /// # Safety
    ///
    /// - `x_handle`: `dim` f32 (modified in-place with the residual add)
    /// - `residual_handle` / `gamma_handle` / `signs_handle`: `dim` f32
    ///   each, read-only (`signs` is the width-`dim` sign vector)
    /// - `norm_out_handle` / `rot_out_handle`: `dim` f32 each, write-only
    /// - the caller MUST hold the geometry guard: `dim % hblock == 0`,
    ///   `hblock <= 1024`, `hblock` a power of two (`norm_rotate_geometry_ok`)
    #[allow(clippy::too_many_arguments, reason = "fused kernel wiring")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        residual_handle: Handle,
        gamma_handle: Handle,
        signs_handle: Handle,
        norm_out_handle: Handle,
        rot_out_handle: Handle,
        dim: usize,
        hblock: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        let params: &[f32] = &[inv_dim, eps, dim as f32, dim as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));

        debug_assert_binding_at_least(&x_handle, dim, "ResidualAddRmsNormRotateCubeCL::x");
        debug_assert_binding_at_least(
            &residual_handle,
            dim,
            "ResidualAddRmsNormRotateCubeCL::residual",
        );
        debug_assert_binding_at_least(&gamma_handle, dim, "ResidualAddRmsNormRotateCubeCL::gamma");
        debug_assert_binding_at_least(&signs_handle, dim, "ResidualAddRmsNormRotateCubeCL::signs");
        debug_assert_binding_at_least(
            &norm_out_handle,
            dim,
            "ResidualAddRmsNormRotateCubeCL::norm_out",
        );
        debug_assert_binding_at_least(
            &rot_out_handle,
            dim,
            "ResidualAddRmsNormRotateCubeCL::rot_out",
        );

        // SAFETY: Caller guarantees correct buffer sizes + the geometry
        // guard (dim % hblock == 0, hblock <= 1024).
        unsafe {
            residual_add_rmsnorm_rotate_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, dim),
                BufferArg::from_raw_parts(residual_handle, dim),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(signs_handle, dim),
                BufferArg::from_raw_parts(params_handle, 5),
                BufferArg::from_raw_parts(norm_out_handle, dim),
                BufferArg::from_raw_parts(rot_out_handle, dim),
            );
        }
    }
}

/// Issue 1145 — the MULTI-CUBE fused ResidualAdd + RMSNorm + forward-rotate
/// staging (the multi-cube shape that could win where Issue 1134's single
/// workgroup measured −20% on its serial chunk walk). **PROMOTED DEFAULT-ON
/// 2026-10-27** — the T3 G2 paired ABBA on idle bmo read +2.83% median
/// (3/3 positive) at the G1 digest pin, over the 1.5% bar;
/// `RIIR_NORM_ROT_MULTI=0` is the demote hatch.
///
/// One launch of `dim / hblock` cubes (5 at Bonsai-2's 5120/1024) replaces
/// `ResidualAddRmsNormCubeCL::launch` + `RotationCubeCL::launch_forward_copy`
/// — two dispatches per norm site, the 128 staging launches/token the
/// bench_763 census priced at +0.71 ms/token. Each cube:
///
/// 1. Redundantly runs the incumbent's Phase 1 + tree reduction over the
///    WHOLE vector — the same per-lane strided order, the same Issue-1143
///    4-deep hoist, the same `+=` sequence, the same tree — so every cube
///    computes the SAME bits for `inv_rms` (the reads are 2×20 KB at
///    Bonsai-2, cache-resident after the first cube).
/// 2. Normalizes ITS chunk into `norm_out[chunk]` (the incumbent's
///    `a * inv_rms * g` order, values read from the thread's own Phase-1
///    stores — no barrier needed), then stages the rotated chunk with the
///    Issue-1144 radix-4 schedule into `rot_out[chunk]` (the r4 body
///    verbatim, fast-math fence included).
///
/// ## The double-buffer contract (the point of `x_src` / `x_dst`)
///
/// The incumbent adds the residual IN PLACE (`x += residual`). A cube
/// writing its own chunk of `x` would race the other cubes still reading
/// `x` for their redundant reductions (the Issue-1136 no-owner class). This
/// kernel therefore READS `x_src` (the pre-add stream — whole vector, every
/// cube) and WRITES `x_dst` (its own chunk only); the caller swaps the two
/// handles after the launch so `x` stays canonical for every later reader
/// (the prefill ping-pong precedent, `prefill_tokens_chunk`'s `x_alt`).
///
/// ## Geometry guard (caller-side — the launcher's only posture)
///
/// `hblock == 1024 == 4·256` — the radix-4 schedule AND the Phase-1
/// group-aligned in-chunk test both require it (the same launcher-only
/// trust the r4 staging kernel carries) — and `dim % hblock == 0`. A miss
/// must fall back to the incumbent chain
/// (see [`norm_rotate_multi_geometry_ok`]).
///
/// ## Parameter Layout
///
/// - `params`: `[f32; 4]` — `[inv_dim, eps, dim, hblock]` (decode
///   single-row: the sign row offset is `base % dim == base`).
///
/// ## Dispatch
///
/// `dim / hblock` cubes (the r4 launcher's 2-D layout),
/// `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
#[allow(clippy::approx_constant)]
fn residual_add_rmsnorm_rotate_multi_f32(
    x_src: &[f32],
    x_dst: &mut [f32],
    residual: &[f32],
    gamma: &[f32],
    signs: &[f32],
    params: &[f32],
    norm_out: &mut [f32],
    rot_out: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = params[2usize] as u32;
    let hblock = params[3usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;
    // 2-D grid (`split_grid`'s layout): rebuild the linear cube index — one
    // cube per hblock chunk, the r4 staging launcher's own mapping.
    let cube = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let base = cube * hblock;
    let chunk_end = base + hblock;

    // ── Phase 1: residual add (redundant over the whole vector — the mlp4
    // 4-deep hoist, same strided order, same `+=` sequence) + x² ──
    // In-chunk stores are GROUP-aligned: one hoisted group spans
    // [i, i + 768] ⊂ one 1024-block, so `i >= base` decides the whole
    // group (holds only at hblock == 4·256 — the launcher's guard).
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i + 3u32 * cube_size < dim {
        let i1 = i + cube_size;
        let i2 = i + 2u32 * cube_size;
        let i3 = i + 3u32 * cube_size;
        let x0 = x_src[i as usize];
        let x1 = x_src[i1 as usize];
        let x2 = x_src[i2 as usize];
        let x3 = x_src[i3 as usize];
        let r0 = residual[i as usize];
        let r1 = residual[i1 as usize];
        let r2 = residual[i2 as usize];
        let r3 = residual[i3 as usize];
        let v0 = x0 + r0;
        let v1 = x1 + r1;
        let v2 = x2 + r2;
        let v3 = x3 + r3;
        if i >= base && i < chunk_end {
            x_dst[i as usize] = v0;
            x_dst[i1 as usize] = v1;
            x_dst[i2 as usize] = v2;
            x_dst[i3 as usize] = v3;
        }
        partial_sq += v0 * v0;
        partial_sq += v1 * v1;
        partial_sq += v2 * v2;
        partial_sq += v3 * v3;
        i += 4u32 * cube_size;
    }
    while i < dim {
        let val = x_src[i as usize] + residual[i as usize];
        if i >= base && i < chunk_end {
            x_dst[i as usize] = val;
        }
        partial_sq += val * val;
        i += cube_size;
    }

    // ── Phase 2: shared-memory tree reduction (identical) ──
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

    // ── Phase 3: broadcast inv_rms (identical) ──
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        smem[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
    sync_cube();

    let inv_rms = smem[0usize];

    // ── Phase 4: normalize + gamma, IN-CHUNK (the mlp4 4-deep hoist over
    // x_dst + gamma — the thread's own Phase-1 stores, so no barrier) ──
    let mut j = tid;
    while j + 3u32 * cube_size < hblock {
        let j1 = j + cube_size;
        let j2 = j + 2u32 * cube_size;
        let j3 = j + 3u32 * cube_size;
        let a0 = x_dst[(base + j) as usize];
        let a1 = x_dst[(base + j1) as usize];
        let a2 = x_dst[(base + j2) as usize];
        let a3 = x_dst[(base + j3) as usize];
        let g0 = gamma[(base + j) as usize];
        let g1 = gamma[(base + j1) as usize];
        let g2 = gamma[(base + j2) as usize];
        let g3 = gamma[(base + j3) as usize];
        norm_out[(base + j) as usize] = a0 * inv_rms * g0;
        norm_out[(base + j1) as usize] = a1 * inv_rms * g1;
        norm_out[(base + j2) as usize] = a2 * inv_rms * g2;
        norm_out[(base + j3) as usize] = a3 * inv_rms * g3;
        j += 4u32 * cube_size;
    }
    while j < hblock {
        let a = x_dst[(base + j) as usize];
        let g = gamma[(base + j) as usize];
        norm_out[(base + j) as usize] = a * inv_rms * g;
        j += cube_size;
    }
    // Phase 5 reads norm_out CROSS-thread (the radix-4 pass-1 quads), so the
    // chunk's Phase-4 stores need one barrier before the butterfly.
    sync_cube();

    // ── Phase 5: the rotate epilogue — the Issue-1144 radix-4 schedule
    // (fwht_forward_copy_f32_r4's body verbatim, reading THIS cube's chunk
    // of norm_out as src and writing rot_out as dst; sign first, the
    // fast-math fence included). ──
    let mut smem_rot = Shared::<[f32]>::new_slice(2048usize);
    let inv_sqrt2 = f32::new(0.70710678f32);
    // Contraction fence: Metal compiles with fast math, so a product that
    // feeds the next stage's add IN A REGISTER may fuse into an fma (the
    // incumbent never sees that — every stage round-trips smem). `select`
    // on a runtime-true predicate (dim > 0) the compiler cannot fold keeps
    // each product a rounded f32 value.
    let live = dim > 0u32;
    let zero = f32::new(0.0f32);

    // Pass 1 (halves 1, 2): the thread's contiguous quad, straight from
    // norm_out — sign first, then the two lowest butterfly stages in
    // registers.
    let q = tid * 4u32;
    let mut v0 = select(
        live,
        norm_out[(base + q) as usize] * signs[(base + q) as usize],
        zero,
    );
    let mut v1 = select(
        live,
        norm_out[(base + q + 1u32) as usize] * signs[(base + q + 1u32) as usize],
        zero,
    );
    let mut v2 = select(
        live,
        norm_out[(base + q + 2u32) as usize] * signs[(base + q + 2u32) as usize],
        zero,
    );
    let mut v3 = select(
        live,
        norm_out[(base + q + 3u32) as usize] * signs[(base + q + 3u32) as usize],
        zero,
    );
    let mut t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
    let mut t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
    let mut t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
    let mut t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
    smem_rot[q as usize] = (t0 + t2) * inv_sqrt2;
    smem_rot[(q + 2u32) as usize] = (t0 - t2) * inv_sqrt2;
    smem_rot[(q + 1u32) as usize] = (t1 + t3) * inv_sqrt2;
    smem_rot[(q + 3u32) as usize] = (t1 - t3) * inv_sqrt2;
    sync_cube();

    // Passes 2-4 (halves 4·8, 16·32, 64·128): quad {e, e+h, e+2h, e+3h}.
    // Ping-pong halves of `smem_rot` (read `rd`, write `wr`), so each pass
    // needs ONE barrier — no thread overwrites a slot another thread still
    // reads.
    let mut rd = 0u32;
    let mut h = 4u32;
    while h < 256u32 {
        let wr = 1024u32 - rd;
        let e = (tid / h) * (4u32 * h) + tid % h;
        v0 = smem_rot[(rd + e) as usize];
        v1 = smem_rot[(rd + e + h) as usize];
        v2 = smem_rot[(rd + e + 2u32 * h) as usize];
        v3 = smem_rot[(rd + e + 3u32 * h) as usize];
        t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
        t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
        t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
        t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
        smem_rot[(wr + e) as usize] = (t0 + t2) * inv_sqrt2;
        smem_rot[(wr + e + 2u32 * h) as usize] = (t0 - t2) * inv_sqrt2;
        smem_rot[(wr + e + h) as usize] = (t1 + t3) * inv_sqrt2;
        smem_rot[(wr + e + 3u32 * h) as usize] = (t1 - t3) * inv_sqrt2;
        sync_cube();
        rd = wr;
        h = h * 4u32;
    }

    // Pass 5 (halves 256, 512): quad {tid + k·256} — straight to rot_out.
    v0 = smem_rot[(rd + tid) as usize];
    v1 = smem_rot[(rd + tid + 256u32) as usize];
    v2 = smem_rot[(rd + tid + 512u32) as usize];
    v3 = smem_rot[(rd + tid + 768u32) as usize];
    t0 = select(live, (v0 + v1) * inv_sqrt2, zero);
    t1 = select(live, (v0 - v1) * inv_sqrt2, zero);
    t2 = select(live, (v2 + v3) * inv_sqrt2, zero);
    t3 = select(live, (v2 - v3) * inv_sqrt2, zero);
    rot_out[(base + tid) as usize] = (t0 + t2) * inv_sqrt2;
    rot_out[(base + tid + 512u32) as usize] = (t0 - t2) * inv_sqrt2;
    rot_out[(base + tid + 256u32) as usize] = (t1 + t3) * inv_sqrt2;
    rot_out[(base + tid + 768u32) as usize] = (t1 - t3) * inv_sqrt2;
}

/// Fused multi-cube ResidualAdd + RMSNorm + forward-rotate staging launcher
/// (Issue 1145).
///
/// Replaces `ResidualAddRmsNormCubeCL::launch` +
/// `RotationCubeCL::launch_forward_copy` with ONE dispatch of `dim / hblock`
/// cubes in the folded (`bonsai2_hadamard`) decode posture — the multi-cube
/// shape that could win where Issue 1134's single-workgroup serial walk
/// measured −20%: the chunks' butterflies run in parallel, and the
/// redundant whole-vector reduction per cube is cache-resident. All three
/// outputs are to_bits-equal to the two-launch chain by construction (mlp4
/// Phases 1-4 + the r4 body, verbatim), so only the REACH counter proves
/// the arm fired (the Bench-768 pattern). The caller swaps `x`/`x_alt`
/// after the launch — the kernel reads `x_src` whole-vector while writing
/// `x_dst` in-chunk, and the double-buffer is what makes the multi-cube
/// reduction legal (the Issue-1136 no-owner race, avoided by construction).
///
/// All three outputs are to_bits-equal to the two-launch chain by
/// construction (mlp4 Phases 1-4 + the r4 body, verbatim), so only the
/// REACH counter proves the toggle fired (the Bench-768 pattern).
/// Default OFF (`RIIR_NORM_ROT_MULTI`); the caller swaps `x`/`x_alt` after
/// the launch — the kernel reads `x_src` whole-vector while writing `x_dst`
/// in-chunk, and the double-buffer is what makes the multi-cube reduction
/// legal (the Issue-1136 no-owner race, avoided by construction).
#[cfg(feature = "cubecl_runtime")]
pub struct ResidualAddRmsNormRotateMultiCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ResidualAddRmsNormRotateMultiCubeCL {
    /// Launch the multi-cube fused ResidualAdd + RMSNorm + rotate staging.
    ///
    /// # Safety
    ///
    /// - `x_src_handle`: `dim` f32 (read-only — the pre-add stream)
    /// - `x_dst_handle`: `dim` f32 (write-only — the post-add stream; MUST
    ///   NOT alias `x_src_handle` — the caller passes the double-buffer)
    /// - `residual_handle` / `gamma_handle` / `signs_handle`: `dim` f32
    ///   each, read-only (`signs` is the width-`dim` sign vector)
    /// - `norm_out_handle` / `rot_out_handle`: `dim` f32 each (write-only)
    /// - the caller MUST hold [`norm_rotate_multi_geometry_ok`]`(dim,
    ///   hblock)`: `hblock == 1024` (the radix-4 schedule + the
    ///   group-aligned in-chunk test) and `dim % hblock == 0`
    #[allow(clippy::too_many_arguments, reason = "fused kernel wiring")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_src_handle: Handle,
        x_dst_handle: Handle,
        residual_handle: Handle,
        gamma_handle: Handle,
        signs_handle: Handle,
        norm_out_handle: Handle,
        rot_out_handle: Handle,
        dim: usize,
        hblock: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        let params: &[f32] = &[inv_dim, eps, dim as f32, hblock as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));

        debug_assert_binding_at_least(
            &x_src_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::x_src",
        );
        debug_assert_binding_at_least(
            &x_dst_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::x_dst",
        );
        debug_assert_binding_at_least(
            &residual_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::residual",
        );
        debug_assert_binding_at_least(
            &gamma_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::gamma",
        );
        debug_assert_binding_at_least(
            &signs_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::signs",
        );
        debug_assert_binding_at_least(
            &norm_out_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::norm_out",
        );
        debug_assert_binding_at_least(
            &rot_out_handle,
            dim,
            "ResidualAddRmsNormRotateMultiCubeCL::rot_out",
        );

        // The r4 launcher's 2-D grid layout (`split_grid`): decode n_cubes =
        // dim/1024 is tiny, the split is overflow hygiene only.
        let n_cubes = (dim / hblock.max(1)) as u32;
        let gx = n_cubes.clamp(1, 65_535);
        // SAFETY: Caller guarantees correct buffer sizes + the geometry
        // guard (hblock == 1024, dim % hblock == 0).
        unsafe {
            residual_add_rmsnorm_rotate_multi_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(gx, n_cubes.div_ceil(gx).max(1), 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_src_handle, dim),
                BufferArg::from_raw_parts(x_dst_handle, dim),
                BufferArg::from_raw_parts(residual_handle, dim),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(signs_handle, dim),
                BufferArg::from_raw_parts(params_handle, 4),
                BufferArg::from_raw_parts(norm_out_handle, dim),
                BufferArg::from_raw_parts(rot_out_handle, dim),
            );
        }
    }
}

/// The caller-side geometry guard for [`ResidualAddRmsNormRotateMultiCubeCL`]:
/// `hblock == 1024` — the Issue-1144 radix-4 schedule AND the Phase-1
/// group-aligned in-chunk test both require exactly 4·256 — and
/// `dim % hblock == 0`. The rest of [`norm_rotate_geometry_ok`] is implied
/// (1024 is a power of two within its bound; the decode launch is
/// single-row so `width == dim`).
#[cfg(feature = "cubecl_runtime")]
pub fn norm_rotate_multi_geometry_ok(dim: usize, hblock: usize) -> bool {
    hblock == 1024 && dim.is_multiple_of(hblock)
}

/// The caller-side geometry guard for [`ResidualAddRmsNormRotateCubeCL`]:
/// `width == dim`, `dim % hblock == 0`, `hblock <= 1024` (the smem slice),
/// `hblock` a power of two (the butterfly's step walk assumes it).
#[cfg(feature = "cubecl_runtime")]
pub fn norm_rotate_geometry_ok(dim: usize, width: usize, hblock: usize) -> bool {
    dim == width
        && dim.is_multiple_of(hblock)
        && hblock <= 1024
        && hblock.is_power_of_two()
        && hblock >= 2
}

// Issue 1134 runtime toggle (the Bench 768 pattern: env default OFF, a force
// setter for the bench harness, a launch counter + one-shot REACH line as
// the vacuous guards — the fused outputs are bit-identical to the
// norm-then-staging chain, so ONLY the counter proves the toggle reached
// the kernel).

/// Armed: env `RIIR_NORM_ROT_FUSED` ("1"/"on"/"true"), else OFF;
/// [`set_norm_rotate_fused`] overrides.
#[cfg(feature = "cubecl_runtime")]
static USE_NORM_ROTATE_FUSED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(feature = "cubecl_runtime")]
static NORM_ROTATE_FUSED_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[cfg(feature = "cubecl_runtime")]
static NORM_ROTATE_FUSED_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "cubecl_runtime")]
static NORM_ROTATE_FUSED_REACHED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Force the fused norm+rotate path on/off (overrides the env; the bench
/// harness's arm toggle).
#[cfg(feature = "cubecl_runtime")]
pub fn set_norm_rotate_fused(on: bool) {
    let _ = NORM_ROTATE_FUSED_INITIALIZED.set(on);
    USE_NORM_ROTATE_FUSED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the fused norm+rotate path is currently armed (dispatch check).
#[cfg(feature = "cubecl_runtime")]
pub fn norm_rotate_fused_enabled() -> bool {
    if NORM_ROTATE_FUSED_INITIALIZED
        .set(env_flag("RIIR_NORM_ROT_FUSED"))
        .is_ok()
    {
        USE_NORM_ROTATE_FUSED.store(
            NORM_ROTATE_FUSED_INITIALIZED
                .get()
                .copied()
                .unwrap_or(false),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_NORM_ROTATE_FUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Launches dispatched through the fused norm+rotate path (the vacuous guard).
#[cfg(feature = "cubecl_runtime")]
pub fn norm_rotate_fused_launch_count() -> usize {
    NORM_ROTATE_FUSED_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// One-shot stderr REACH line (the Bench-768 guard): confirms the fused
/// path actually dispatched at least once in this process.
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn norm_rotate_fused_mark_reached() {
    if NORM_ROTATE_FUSED_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        let _ = NORM_ROTATE_FUSED_REACHED.set(());
        eprintln!("[norm_rot_fused] REACH: the fused norm+rotate staging dispatched");
    }
}

// Issue 1145 runtime toggle (the Bench 768 pattern). **PROMOTED DEFAULT-ON
// 2026-10-27** (the T3 G2 on idle bmo: paired ABBA ×3 — 17.629/17.458/17.518
// → 17.035/17.032/17.097 ms/token = +3.49/+2.50/+2.46%, median +2.83% ≥ the
// 1.5% bar, 3/3 positive; G1 digest `d06bffced92427cb` + last token 279
// EXACT in all six arms; G4 zero per-token alloc — the x_alt double buffer
// is one construction-time allocation). The `dense_pair_env_semantics`
// contract (unset = ON); `RIIR_NORM_ROT_MULTI=0` demotes byte-identically
// to the incumbent norm + r4-staging pair; [`set_norm_rotate_multi`] is the
// harness override. A launch counter + one-shot REACH line stay as the
// vacuous guards — the multi-cube outputs are bit-identical to the
// norm-then-staging chain, so ONLY the counter proves the toggle reached
// the kernel).

/// Armed by default; `RIIR_NORM_ROT_MULTI` set to anything but 1/on/true
/// demotes (the `dense_pair_env_semantics` contract, unset = ON);
/// [`set_norm_rotate_multi`] overrides.
#[cfg(feature = "cubecl_runtime")]
static USE_NORM_ROTATE_MULTI: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
#[cfg(feature = "cubecl_runtime")]
static NORM_ROTATE_MULTI_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[cfg(feature = "cubecl_runtime")]
static NORM_ROTATE_MULTI_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "cubecl_runtime")]
static NORM_ROTATE_MULTI_REACHED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Force the multi-cube fused norm+rotate path on/off (overrides the env;
/// the bench harness's arm toggle).
#[cfg(feature = "cubecl_runtime")]
pub fn set_norm_rotate_multi(on: bool) {
    let _ = NORM_ROTATE_MULTI_INITIALIZED.set(on);
    USE_NORM_ROTATE_MULTI.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the multi-cube fused norm+rotate path is currently armed
/// (dispatch check). Default ON (the Issue-1145 T3 promotion);
/// `RIIR_NORM_ROT_MULTI=0` is DEMOTE-ONLY.
#[cfg(feature = "cubecl_runtime")]
pub fn norm_rotate_multi_enabled() -> bool {
    if NORM_ROTATE_MULTI_INITIALIZED
        .set(crate::gemv_cubecl::dense_pair_env_semantics(std::env::var(
            "RIIR_NORM_ROT_MULTI",
        )))
        .is_ok()
    {
        USE_NORM_ROTATE_MULTI.store(
            NORM_ROTATE_MULTI_INITIALIZED.get().copied().unwrap_or(true),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_NORM_ROTATE_MULTI.load(std::sync::atomic::Ordering::Relaxed)
}

/// Launches dispatched through the multi-cube fused norm+rotate path (the
/// vacuous guard).
#[cfg(feature = "cubecl_runtime")]
pub fn norm_rotate_multi_launch_count() -> usize {
    NORM_ROTATE_MULTI_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// One-shot stderr REACH line (the Bench-768 guard): confirms the multi-cube
/// path actually dispatched at least once in this process.
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn norm_rotate_multi_mark_reached() {
    if NORM_ROTATE_MULTI_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        let _ = NORM_ROTATE_MULTI_REACHED.set(());
        eprintln!("[norm_rot_multi] REACH: the multi-cube fused norm+rotate staging dispatched");
    }
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "on" | "true"
    )
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
fn layernorm_mean_batched_f32(input: &[f32], gamma: &[f32], params: &[f32], output: &mut [f32]) {
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

    /// Issue 1134 G1 — the fused ResidualAddRmsNorm+rotate is to_bits-equal
    /// to the incumbent two-launch chain (ResidualAddRmsNorm then
    /// RotationCubeCL::launch_forward_copy) at the production geometry
    /// (n=5120, hblock=1024), an off-shape (n=2048, hblock=512), and the
    /// edge rows (all-zero — inv_rms pins to 1/sqrt(eps); ±large; the
    /// alternating sign pattern the folded basis stresses).
    ///
    /// Correctness is load-independent: safe to run under GPU co-tenants
    /// (only G2 timing needs the 649-exclusive window).
    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    fn norm_rot_g1_case(dim: usize, hblock: usize, x: Vec<f32>, eps: f32) {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // The sign vector: the loader's ±1 i8 table rendered as f32 (the
        // kernels read an f32 sign buffer). Deterministic alternation over
        // 7-period blocks — not aligned to hblock, so the chunk walks see
        // mixed signs.
        let signs: Vec<f32> = (0..dim)
            .map(|i| if (i / 7) % 2 == 0 { 1.0f32 } else { -1.0f32 })
            .collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| 1.0f32 + (i % 11) as f32 * 0.031f32)
            .collect();
        let residual: Vec<f32> = (0..dim)
            .map(|i| ((i % 23) as f32 - 11.0f32) * 0.37f32)
            .collect();

        // ── The incumbent chain ──
        let x_a = client.create_from_slice(f32::as_bytes(&x));
        let res_a = client.create_from_slice(f32::as_bytes(&residual));
        let gamma_a = client.create_from_slice(f32::as_bytes(&gamma));
        let norm_a = client.empty(dim * core::mem::size_of::<f32>());
        unsafe {
            ResidualAddRmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                x_a.clone(),
                res_a.clone(),
                gamma_a.clone(),
                norm_a.clone(),
                dim,
                eps,
            );
        }
        let signs_a = client.create_from_slice(f32::as_bytes(&signs));
        let rot_a = client.empty(dim * core::mem::size_of::<f32>());
        unsafe {
            crate::deltanet_rotation_cubecl::RotationCubeCL::launch_forward_copy::<ActiveRuntime>(
                &client,
                norm_a.clone(),
                rot_a.clone(),
                signs_a.clone(),
                dim,
                dim,
                hblock,
            );
        }

        // ── The fused launch ──
        let x_b = client.create_from_slice(f32::as_bytes(&x));
        let res_b = client.create_from_slice(f32::as_bytes(&residual));
        let gamma_b = client.create_from_slice(f32::as_bytes(&gamma));
        let signs_b = client.create_from_slice(f32::as_bytes(&signs));
        let norm_b = client.empty(dim * core::mem::size_of::<f32>());
        let rot_b = client.empty(dim * core::mem::size_of::<f32>());
        unsafe {
            ResidualAddRmsNormRotateCubeCL::launch::<ActiveRuntime>(
                &client,
                x_b.clone(),
                res_b.clone(),
                gamma_b.clone(),
                signs_b.clone(),
                norm_b.clone(),
                rot_b.clone(),
                dim,
                hblock,
                eps,
            );
        }

        // ── to_bits equality on BOTH outputs (norm + the staged rotation) ──
        let norm_a_bytes = client.read_one(norm_a).expect("read");
        let rot_a_bytes = client.read_one(rot_a).expect("read");
        let norm_b_bytes = client.read_one(norm_b).expect("read");
        let rot_b_bytes = client.read_one(rot_b).expect("read");
        let x_a_bytes = client.read_one(x_a).expect("read");
        let x_b_bytes = client.read_one(x_b).expect("read");
        let norm_a_v = f32::from_bytes(&norm_a_bytes);
        let rot_a_v = f32::from_bytes(&rot_a_bytes);
        let norm_b_v = f32::from_bytes(&norm_b_bytes);
        let rot_b_v = f32::from_bytes(&rot_b_bytes);
        let x_a_v = f32::from_bytes(&x_a_bytes);
        let x_b_v = f32::from_bytes(&x_b_bytes);

        for i in 0..dim {
            assert_eq!(
                norm_a_v[i].to_bits(),
                norm_b_v[i].to_bits(),
                "norm_out[{i}] bit-mismatch at dim={dim} hblock={hblock}"
            );
            assert_eq!(
                rot_a_v[i].to_bits(),
                rot_b_v[i].to_bits(),
                "rot_out[{i}] bit-mismatch at dim={dim} hblock={hblock}"
            );
            assert_eq!(
                x_a_v[i].to_bits(),
                x_b_v[i].to_bits(),
                "x-in-place[{i}] bit-mismatch at dim={dim} hblock={hblock}"
            );
        }
    }

    /// Issue 1143 G1 — the 4-deep residual-add norm is to_bits-equal to the
    /// incumbent on all three outputs' worth of state (x in place, norm_out)
    /// at the Bonsai decode width, a ragged tail (dim not a multiple of
    /// 4·256 nor of 256), and a sub-group width (dim < 1024 — the hoisted
    /// loop never runs).
    fn norm_mlp4_g1_case(dim: usize) {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..dim)
            .map(|i| ((i % 37) as f32 - 18.0f32) * 0.113f32 + ((i % 5) as f32) * 121.7f32)
            .collect();
        let residual: Vec<f32> = (0..dim)
            .map(|i| ((i % 23) as f32 - 11.0f32) * 0.37f32)
            .collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| 1.0f32 + (i % 11) as f32 * 0.031f32)
            .collect();
        let params: &[f32] = &[1.0f32 / dim as f32, eps, dim as f32];
        let run = |mlp: bool| -> (Vec<f32>, Vec<f32>) {
            let xh = client.create_from_slice(f32::as_bytes(&x));
            let rh = client.create_from_slice(f32::as_bytes(&residual));
            let gh = client.create_from_slice(f32::as_bytes(&gamma));
            let ph = client.create_from_slice(f32::as_bytes(params));
            let oh = client.empty(dim * core::mem::size_of::<f32>());
            // SAFETY: every handle was created above with exactly `dim` (params: 3) f32.
            unsafe {
                let args = (
                    BufferArg::from_raw_parts(xh.clone(), dim),
                    BufferArg::from_raw_parts(rh, dim),
                    BufferArg::from_raw_parts(gh, dim),
                    BufferArg::from_raw_parts(ph, 3),
                    BufferArg::from_raw_parts(oh.clone(), dim),
                );
                match mlp {
                    true => residual_add_rmsnorm_f32_mlp4::launch_unchecked::<ActiveRuntime>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(256),
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                    ),
                    false => residual_add_rmsnorm_f32::launch_unchecked::<ActiveRuntime>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(256),
                        args.0,
                        args.1,
                        args.2,
                        args.3,
                        args.4,
                    ),
                }
            }
            let xb = client.read_one(xh).expect("read");
            let ob = client.read_one(oh).expect("read");
            (f32::from_bytes(&xb).to_vec(), f32::from_bytes(&ob).to_vec())
        };
        let (x_a, o_a) = run(false);
        let (x_b, o_b) = run(true);
        let mut nonzero = 0usize;
        for i in 0..dim {
            assert_eq!(x_a[i].to_bits(), x_b[i].to_bits(), "x[{i}] dim={dim}");
            assert_eq!(
                o_a[i].to_bits(),
                o_b[i].to_bits(),
                "norm_out[{i}] dim={dim}"
            );
            nonzero += usize::from(o_b[i] != 0.0);
        }
        // Loud-zero guard: an unlaunched kernel leaves both outputs equal zeros.
        assert!(
            nonzero > dim / 2,
            "norm_out mostly zero — kernel did not run (dim={dim})"
        );
    }

    #[test]
    fn test_residual_add_rmsnorm_mlp4_bit_identity() {
        norm_mlp4_g1_case(5120); // Bonsai decode width: 20 trips, 5 hoisted groups
        norm_mlp4_g1_case(5120 + 2 * 256 + 37); // ragged: 2 single-step tail trips + partial lanes
        norm_mlp4_g1_case(768); // < 4·256: the hoisted loop never runs
    }

    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    #[test]
    fn test_norm_rotate_fused_bit_identity_production_geometry() {
        // The Bonsai-2 production shape: n_embd=5120, Hadamard block 1024.
        let dim = 5120usize;
        let hblock = 1024usize;
        let eps = 1e-6f32;
        let mixed: Vec<f32> = (0..dim)
            .map(|i| ((i % 37) as f32 - 18.0f32) * 0.113f32 + ((i % 5) as f32) * 121.7f32)
            .collect();
        norm_rot_g1_case(dim, hblock, mixed, eps);
    }

    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    #[test]
    fn test_norm_rotate_fused_bit_identity_off_shape() {
        // Off-shape: n=2048, hblock=512 (2 chunks, half the smem window).
        let dim = 2048usize;
        let hblock = 512usize;
        let eps = 1e-6f32;
        let mixed: Vec<f32> = (0..dim)
            .map(|i| ((i % 13) as f32 - 6.0f32) * 3.71f32)
            .collect();
        norm_rot_g1_case(dim, hblock, mixed, eps);
    }

    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    #[test]
    fn test_norm_rotate_fused_bit_identity_edge_rows() {
        let dim = 1024usize;
        let hblock = 1024usize;
        let eps = 1e-6f32;
        // All-zero row: mean_sq = 0, inv_rms = 1/sqrt(eps) — the pins branch.
        norm_rot_g1_case(dim, hblock, vec![0.0f32; dim], eps);
        // ±large (the residual-stream ±4000 regime from the plan-611 note).
        norm_rot_g1_case(
            dim,
            hblock,
            (0..dim)
                .map(|i| if i % 2 == 0 { 4000.0f32 } else { -4000.0f32 })
                .collect(),
            eps,
        );
        // Alternating ±1 (the maximal-sign-stress row).
        norm_rot_g1_case(
            dim,
            hblock,
            (0..dim)
                .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
                .collect(),
            eps,
        );
    }

    /// Issue 1145 G1 — the MULTI-CUBE fused ResidualAddRmsNorm+rotate is
    /// to_bits-equal to the incumbent two-launch chain
    /// (ResidualAddRmsNorm then RotationCubeCL::launch_forward_copy) on ALL
    /// THREE outputs — the post-add stream (read back from the kernel's
    /// `x_dst`, not in-place `x`), `norm_out`, and the staged rotation — at
    /// the production geometry (n=5120, hblock=1024, 5 cubes), an off-shape
    /// (n=2048, hblock=1024, 2 cubes), and the edge rows (all-zero —
    /// inv_rms pins to 1/sqrt(eps); ±large; the alternating sign pattern the
    /// folded basis stresses). The multi lane's launcher-only posture is
    /// hblock == 1024 (the radix-4 schedule + the group-aligned in-chunk
    /// test), so every case pins that shape.
    ///
    /// Correctness is load-independent: safe to run under GPU co-tenants
    /// (only G2 timing needs the 649-exclusive window).
    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    fn norm_rot_multi_g1_case(dim: usize, hblock: usize, x: Vec<f32>, eps: f32) {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // The sign vector: the loader's ±1 i8 table rendered as f32 (the
        // kernels read an f32 sign buffer). Deterministic alternation over
        // 7-period blocks — not aligned to hblock, so the chunk walks see
        // mixed signs.
        let signs: Vec<f32> = (0..dim)
            .map(|i| if (i / 7) % 2 == 0 { 1.0f32 } else { -1.0f32 })
            .collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| 1.0f32 + (i % 11) as f32 * 0.031f32)
            .collect();
        let residual: Vec<f32> = (0..dim)
            .map(|i| ((i % 23) as f32 - 11.0f32) * 0.37f32)
            .collect();

        // ── The incumbent chain ──
        let x_a = client.create_from_slice(f32::as_bytes(&x));
        let res_a = client.create_from_slice(f32::as_bytes(&residual));
        let gamma_a = client.create_from_slice(f32::as_bytes(&gamma));
        let norm_a = client.empty(dim * core::mem::size_of::<f32>());
        unsafe {
            ResidualAddRmsNormCubeCL::launch::<ActiveRuntime>(
                &client,
                x_a.clone(),
                res_a.clone(),
                gamma_a.clone(),
                norm_a.clone(),
                dim,
                eps,
            );
        }
        let signs_a = client.create_from_slice(f32::as_bytes(&signs));
        let rot_a = client.empty(dim * core::mem::size_of::<f32>());
        unsafe {
            crate::deltanet_rotation_cubecl::RotationCubeCL::launch_forward_copy::<ActiveRuntime>(
                &client,
                norm_a.clone(),
                rot_a.clone(),
                signs_a.clone(),
                dim,
                dim,
                hblock,
            );
        }

        // ── The multi-cube launch (the double-buffer split) ──
        let x_src = client.create_from_slice(f32::as_bytes(&x));
        let x_dst = client.create_from_slice(f32::as_bytes(&x));
        let res_b = client.create_from_slice(f32::as_bytes(&residual));
        let gamma_b = client.create_from_slice(f32::as_bytes(&gamma));
        let signs_b = client.create_from_slice(f32::as_bytes(&signs));
        let norm_b = client.empty(dim * core::mem::size_of::<f32>());
        let rot_b = client.empty(dim * core::mem::size_of::<f32>());
        unsafe {
            ResidualAddRmsNormRotateMultiCubeCL::launch::<ActiveRuntime>(
                &client,
                x_src.clone(),
                x_dst.clone(),
                res_b.clone(),
                gamma_b.clone(),
                signs_b.clone(),
                norm_b.clone(),
                rot_b.clone(),
                dim,
                hblock,
                eps,
            );
        }

        // ── to_bits equality on ALL THREE outputs (post-add x via x_dst,
        // norm, the staged rotation) ──
        let norm_a_bytes = client.read_one(norm_a).expect("read");
        let rot_a_bytes = client.read_one(rot_a).expect("read");
        let norm_b_bytes = client.read_one(norm_b).expect("read");
        let rot_b_bytes = client.read_one(rot_b).expect("read");
        let x_a_bytes = client.read_one(x_a).expect("read");
        let x_b_bytes = client.read_one(x_dst).expect("read");
        let norm_a_v = f32::from_bytes(&norm_a_bytes);
        let rot_a_v = f32::from_bytes(&rot_a_bytes);
        let norm_b_v = f32::from_bytes(&norm_b_bytes);
        let rot_b_v = f32::from_bytes(&rot_b_bytes);
        let x_a_v = f32::from_bytes(&x_a_bytes);
        let x_b_v = f32::from_bytes(&x_b_bytes);

        for i in 0..dim {
            assert_eq!(
                norm_a_v[i].to_bits(),
                norm_b_v[i].to_bits(),
                "norm_out[{i}] bit-mismatch at dim={dim} hblock={hblock}"
            );
            assert_eq!(
                rot_a_v[i].to_bits(),
                rot_b_v[i].to_bits(),
                "rot_out[{i}] bit-mismatch at dim={dim} hblock={hblock}"
            );
            assert_eq!(
                x_a_v[i].to_bits(),
                x_b_v[i].to_bits(),
                "x-post-add[{i}] bit-mismatch at dim={dim} hblock={hblock}"
            );
        }
    }

    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    #[test]
    fn test_norm_rotate_multi_bit_identity_production_geometry() {
        // The Bonsai-2 production shape: n_embd=5120, Hadamard block 1024
        // (5 cubes — the census's actual posture).
        let dim = 5120usize;
        let hblock = 1024usize;
        let eps = 1e-6f32;
        let mixed: Vec<f32> = (0..dim)
            .map(|i| ((i % 37) as f32 - 18.0f32) * 0.113f32 + ((i % 5) as f32) * 121.7f32)
            .collect();
        norm_rot_multi_g1_case(dim, hblock, mixed, eps);
    }

    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    #[test]
    fn test_norm_rotate_multi_bit_identity_off_shape() {
        // Off-shape: n=2048, hblock=1024 (2 cubes — fewer chunks than the
        // production 5, same launcher-only hblock posture).
        let dim = 2048usize;
        let hblock = 1024usize;
        let eps = 1e-6f32;
        let mixed: Vec<f32> = (0..dim)
            .map(|i| ((i % 13) as f32 - 6.0f32) * 3.71f32)
            .collect();
        norm_rot_multi_g1_case(dim, hblock, mixed, eps);
    }

    #[cfg(feature = "ternary_gemv")] // RotationCubeCL (the incumbent arm) is ternary_gemv-gated
    #[test]
    fn test_norm_rotate_multi_bit_identity_edge_rows() {
        let dim = 1024usize;
        let hblock = 1024usize;
        let eps = 1e-6f32;
        // All-zero row: mean_sq = 0, inv_rms = 1/sqrt(eps) — the pins branch.
        norm_rot_multi_g1_case(dim, hblock, vec![0.0f32; dim], eps);
        // ±large (the residual-stream ±4000 regime from the plan-611 note).
        norm_rot_multi_g1_case(
            dim,
            hblock,
            (0..dim)
                .map(|i| if i % 2 == 0 { 4000.0f32 } else { -4000.0f32 })
                .collect(),
            eps,
        );
        // Alternating ±1 (the maximal-sign-stress row).
        norm_rot_multi_g1_case(
            dim,
            hblock,
            (0..dim)
                .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
                .collect(),
            eps,
        );
    }

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
        let x_post: Vec<f32> = x
            .iter()
            .zip(residual.iter())
            .map(|(&a, &b)| a + b)
            .collect();
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
        let norm_bytes = client
            .read_one(norm_out_handle)
            .expect("should read norm output");
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
            assert!(
                err < 1e-5,
                "dim256 norm element {i}: expected {exp}, got {got}"
            );
        }
        // Also verify x was updated
        for (i, (&exp, &got)) in x_expected.iter().zip(x_output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "dim256 x element {i}: expected {exp}, got {got}"
            );
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
