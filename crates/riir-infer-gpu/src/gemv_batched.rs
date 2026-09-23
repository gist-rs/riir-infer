//! Shared GPU-resident GEMV launch + batched read helpers (Issue 393 / 396).
//!
//! Extracted from `kimi_k3_gpu_backward` so both the forward batched path
//! (Issue 396) and the backward batched path (Issue 394) can consume them
//! without a cross-feature dependency. The backward module re-exports these
//! for backwards compatibility.
//!
//! # Pattern
//!
//! The per-GEMV `read_one()` round-trip dominates wall-clock at scale
//! (~1.3 ms per sync × thousands of GEMVs). The fix:
//!
//! 1. `launch_gemv(...)` for each independent GEMV → returns `Handle` (no sync)
//! 2. `read_batched(client, handles)` once to drain all pending outputs
//!
//! Sync points are at natural data-dependency boundaries (e.g. before a CPU
//! element-wise op), not per-GEMV.

use cubecl::prelude::*;
use cubecl::server::Handle;
use crate::cubecl_runtime::ActiveRuntime;

use crate::gemv_autotune::GemvAutotune;

/// Launch a GEMV keeping the output GPU-resident (no host sync).
///
/// Computes `output[m_out] = weight[m_out × n_in] @ input[n_in]` entirely on
/// the GPU. The output `Handle` is returned for chaining into subsequent GPU
/// kernels or batched read-back at a sync boundary.
///
/// Unlike the legacy per-GEMV `dispatch` closure that did `launch + read_one`
/// (one round-trip per GEMV), this defers the read to the caller, allowing N
/// independent GEMVs to share one [`read_batched`].
///
/// # Safety
///
/// `weight` must have `m_out × n_in` f32 elements; `input` must have `n_in`
/// f32 elements. The kernel is `unsafe` because CubeCL's autotune launch is
/// `unsafe` (raw GPU dispatch).
#[allow(clippy::too_many_arguments)]
pub fn launch_gemv(
    client: &ComputeClient<ActiveRuntime>,
    gemv_autotune: &GemvAutotune,
    weight: &Handle,
    input: &[f32],
    m_out: usize,
    n_in: usize,
) -> Handle {
    let input_handle = client.create_from_slice(f32::as_bytes(input));
    launch_gemv_handle(client, gemv_autotune, weight, input_handle, m_out, n_in)
}

/// Launch CubeCL GEMV from a **GPU-resident input handle** (Issue 401 Phase 2.2).
///
/// Variant of [`launch_gemv`] for the handle-resident forward path: skips the
/// `create_from_slice` upload (the input is already on the GPU). Used by the
/// fused Dense FFN path where the SiTU activation output stays on GPU + feeds
/// directly into the down-projection GEMV without a CPU round-trip.
///
/// # Safety
///
/// Same contract as [`launch_gemv`] — the caller guarantees `weight` has
/// `m_out × n_in` f32 elements + `input_handle` has `n_in` f32 elements.
#[allow(clippy::too_many_arguments)]
pub fn launch_gemv_handle(
    client: &ComputeClient<ActiveRuntime>,
    gemv_autotune: &GemvAutotune,
    weight: &Handle,
    input_handle: Handle,
    m_out: usize,
    n_in: usize,
) -> Handle {
    let output_handle = client.empty(m_out * core::mem::size_of::<f32>());
    // SAFETY: weight has m_out × n_in elements, input_handle has n_in elements.
    unsafe {
        gemv_autotune.launch::<ActiveRuntime>(
            client,
            weight.clone(),
            input_handle,
            output_handle.clone(),
            m_out,
            n_in,
        );
    }
    output_handle
}

/// Read multiple GPU output handles in ONE synchronization round-trip.
///
/// `client.read(vec)` issues a single `read_async` covering all handles, vs
/// `read_one` per handle which issues N separate round-trips. Each handle
/// becomes one `Vec<f32>`.
///
/// Use after a batch of independent `launch_gemv` calls to drain all outputs
/// at a natural sync boundary (e.g. before a CPU element-wise op).
pub fn read_batched(client: &ComputeClient<ActiveRuntime>, handles: Vec<Handle>) -> Vec<Vec<f32>> {
    if handles.is_empty() {
        return Vec::new();
    }
    let bytes_vec = client.read(handles);
    bytes_vec
        .into_iter()
        .map(|b| f32::from_bytes(&b).to_vec())
        .collect()
}

/// Read a single GPU handle as a flat `Vec<f32>`.
///
/// Convenience wrapper around [`read_batched`] for the single-handle case.
pub fn read_handle(client: &ComputeClient<ActiveRuntime>, handle: Handle) -> Vec<f32> {
    read_batched(client, vec![handle]).remove(0)
}
