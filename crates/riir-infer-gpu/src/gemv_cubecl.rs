//! CubeCL GEMV kernel for decode-time matrix-vector multiplication (Plan 106 T2.3).
//!
//! Implements `output[M] = weight[M,N] @ input[N]` using CubeCL's `#[cube]` DSL.
//!
//! Two variants:
//! - **Plane (subgroup)**: Cooperative dot product with `plane_sum()` reduction.
//!   Each plane/subgroup handles one output row. Best performance on Metal.
//! - **Shared memory tiling**: Fallback when planes unavailable.
//!   Input vector tiled into shared memory, one thread per output row.
//!
//! # Dispatch
//!
//! | Variant | CubeDim          | CubeCount              | Rows/cube |
//! |---------|------------------|------------------------|-----------|
//! | Plane   | `new_1d(256)`    | `ceil(m / 8), 1, 1`   | 8         |
//! | Tiled   | `new_1d(256)`    | `ceil(m / 256), 1, 1` | 256       |
//!
//! # CODA Hook (Track 3)
//!
//! The GEMV output stays in registers before global write. Track 3 epilogue
//! visitors can consume the accumulator before writeback, enabling fused
//! GEMV + activation + residual in a single dispatch.

#[cfg(feature = "cubecl_runtime")]
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Plane (subgroup) GEMV — primary kernel
// ---------------------------------------------------------------------------

/// CubeCL GEMV kernel using plane (subgroup) cooperative dot product.
///
/// Each plane handles one output row. Lanes cooperatively compute the dot product:
/// lane `j` accumulates elements at indices `j, j + PLANE_DIM, j + 2*PLANE_DIM, ...`.
/// `plane_sum()` reduces partial sums — single `simd_sum()` on Metal.
///
/// Matches `gemv_subgroup.wgsl` algorithm via CubeCL's portable `Plane` abstraction.
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_plane_f32(weight: &[f32], input: &[f32], output: &mut [f32]) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    // Each plane handles one output row.
    // row = global_thread_index / plane_size
    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let row_offset = row * n;

    // Cooperative dot product: lane j handles elements j, j + PLANE_DIM, ...
    // Adjacent lanes read contiguous weight elements → fully coalesced access.
    let mut partial = f32::new(0.0f32);

    let mut k = lane;
    while k < n {
        partial += weight[(row_offset + k) as usize] * input[k as usize];
        k += PLANE_DIM;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane.
    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row.
    if lane == 0 {
        output[row as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Batched plane GEMV — multiple input vectors, single weight matrix (Plan 436 T2.1)
// ---------------------------------------------------------------------------

/// CubeCL batched plane GEMV: `output[batch, out_dim] = input[batch, in_dim] @ weight[out_dim, in_dim]^T`.
///
/// Each plane handles one output row and loops over all batch positions. The
/// weight row is read once per batch iteration — on Metal the row stays in L1
/// cache across the `batch` iterations, so the effective weight bandwidth is
/// one stream-through regardless of batch size.
///
/// This mirrors the CPU `matmul_vec_batched` optimization: one weight read,
/// `batch` outputs. For Weaver (batch=seq_len=5), this reduces the 40 single
/// GEMV dispatches (8 weights × 5 positions) to 7 batched dispatches (one per
/// weight matrix) plus 5 single dispatches for w_down (per-position after
/// SwiGLU activation).
///
/// # Weight layout
///
/// `weight` is `[out_dim, in_dim]` row-major — the same layout as
/// [`gemv_plane_f32`]. The CPU `WeaverWeights` stores `[in_dim, out_dim]`, so
/// a transpose is needed at upload time (see `weaver_gpu::transpose_weight`).
///
/// # Dispatch
///
/// `CubeCount::Static(ceil(out_dim / 8), 1, 1)`, `CubeDim::new_1d(256)`.
/// Each workgroup has 8 planes (256 threads / plane_size 32) → 8 output rows.
///
/// # Params
///
/// `params[0] = batch as f32`, `params[1] = in_dim as f32`,
/// `params[2] = out_dim as f32` — `out_dim` is passed explicitly (NOT derived
/// from `output.len()`) because the output buffer may be oversized relative to
/// `batch * out_dim`; `output.len()` returns the full allocation size and
/// deriving from it mis-strides every batch row >= 1 (Issue 697/698 G3).
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_batched_plane_f32(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let batch = params[0usize] as u32;
    let in_dim = params[1usize] as u32;
    // `out_dim` MUST come from params, never `output.len() / batch` — the
    // output buffer may be OVERSIZED relative to `batch * out_dim` (the weaver
    // per-depth path launches batch=seq_len=2 into a (max_depth+1)-row
    // scratch), and `output.len()` returns the full allocation size. Deriving
    // `out_dim` from it scrambles the write offset `b * out_dim + row` for
    // every batch row >= 1 — the Issue 697/698 G3 CPU↔GPU divergence root
    // cause (row 1 landed at [160..224) in a 320-float buffer while the
    // consumer read [64..128)).
    let out_dim = params[2usize] as u32;

    // Each plane handles one output row.
    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= out_dim {
        terminate!();
    }

    let row_offset = row * in_dim;

    // Loop over batch positions. The weight row (`row_offset..row_offset+in_dim`)
    // stays in L1 cache across iterations; each iteration reads a different
    // input row from the `input[batch, in_dim]` buffer.
    let mut b = 0u32;
    while b < batch {
        let mut partial = f32::new(0.0f32);
        let input_offset = b * in_dim;

        let mut k = lane;
        while k < in_dim {
            partial += weight[(row_offset + k) as usize] * input[(input_offset + k) as usize];
            k += PLANE_DIM;
        }

        let result = plane_sum(partial);

        if lane == 0 {
            output[(b * out_dim + row) as usize] = result;
        }

        b += 1u32;
    }
}

// ---------------------------------------------------------------------------
// Shared memory GEMV — fallback kernel (no subgroups required)
// ---------------------------------------------------------------------------

/// CubeCL GEMV kernel using shared memory tiling (no subgroup required).
///
/// Each thread computes one output row (full dot product). The input vector
/// is tiled into shared memory so all threads share the same input tile,
/// improving memory access patterns.
///
/// **Plan 481 fix:** The early `terminate!()` for threads with `row >= m` was
/// moved to AFTER the cooperative shared-memory loading. Previously, when
/// `m < wg_size` (e.g. LoRA shapes 2×32, 16×2), only `m` threads survived to
/// load the input tile, leaving `n - m` input elements uninitialized in shared
/// memory → garbage dot product. All threads in the workgroup must participate
/// in the cooperative load + both `sync_cube()` barriers; only the computation
/// and output write are gated by `row < m`.
///
/// Matches `gemv.wgsl` algorithm.
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_tile_f32(weight: &[f32], input: &[f32], output: &mut [f32]) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    // Each thread handles one output row.
    let row = ABSOLUTE_POS as u32;

    let row_offset = row * n;
    let mut sum = f32::new(0.0f32);

    // Shared memory for input vector tile (256 elements = 1 KB).
    let mut tile_input = Shared::<[f32]>::new_slice(256usize);

    let tile_size = 256u32;
    let num_tiles = n.div_ceil(tile_size);

    let mut tile_idx = 0u32;
    while tile_idx < num_tiles {
        let tile_start = tile_idx * tile_size;

        // Cooperatively load tile of input into shared memory.
        // Thread t loads element tile_start + t (or 0 if out of bounds).
        // ALL threads in the workgroup must participate — do NOT terminate
        // before this point, or shared memory will be partially uninitialized.
        let t = UNIT_POS;
        let input_idx = tile_start + t;
        if input_idx < n {
            tile_input[t as usize] = input[input_idx as usize];
        } else {
            tile_input[t as usize] = f32::new(0.0f32);
        }

        sync_cube();

        // Accumulate partial dot product from this tile.
        // Only threads with a valid output row compute; the rest idle through
        // the sync barriers to keep the workgroup synchronized.
        if row < m {
            let mut k = 0u32;
            while k < tile_size {
                let weight_idx = tile_start + k;
                if weight_idx < n {
                    sum += weight[(row_offset + weight_idx) as usize] * tile_input[k as usize];
                }
                k += 1u32;
            }
        }

        sync_cube();
        tile_idx += 1u32;
    }

    if row < m {
        output[row as usize] = sum;
    }
}

// ---------------------------------------------------------------------------
// Element-wise add kernel (for all-GPU LoRA: output = base + delta)
// ---------------------------------------------------------------------------

/// CubeCL element-wise add: `output[i] = a[i] + b[i]`.
///
/// Used by the all-GPU LoRA path (`gemv_lora_gpu`) to combine the base GEMV
/// output with the LoRA delta: `output = W @ input + B_scaled @ (A @ input)`.
///
/// Each thread handles one element. No shared memory, no cooperative loading —
/// the early `terminate!()` is safe here (unlike `gemv_tile_f32`).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    let n = a.len();
    let tid = ABSOLUTE_POS;

    if tid >= n {
        terminate!();
    }

    output[tid] = a[tid] + b[tid];
}

// ---------------------------------------------------------------------------
// Public API: auto-selecting launcher
// ---------------------------------------------------------------------------

/// CubeCL GEMV launcher with automatic plane/shared-memory selection.
///
/// Selects the plane (subgroup) kernel when the device supports it,
/// otherwise falls back to the shared-memory tiled kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvCubeCL {
    /// Launch GEMV: `output[M] = weight[M,N] @ input[N]`.
    ///
    /// Auto-selects plane or tiled kernel based on device capabilities.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - weight: M×N f32 elements
    /// - input: N f32 elements
    /// - output: M f32 elements
    ///
    /// Launch GEMV: `output[M] = weight[M,N] @ input[N]`.
    ///
    /// Auto-selects plane or tiled kernel based on device capabilities.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - weight: M×N f32 elements
    /// - input: N f32 elements
    /// - output: M f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let has_plane = client.features().plane.contains(Plane::Ops);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            if has_plane {
                Self::launch_plane::<R>(client, weight_handle, input_handle, output_handle, m, n);
            } else {
                Self::launch_tiled::<R>(client, weight_handle, input_handle, output_handle, m, n);
            }
        }
    }

    /// Launch plane (subgroup) GEMV kernel.
    ///
    /// Each plane handles one output row with cooperative dot product + `plane_sum()`.
    /// Workgroup: 256 threads → 8 planes (with plane_dim=32 on Metal).
    /// Dispatch: `ceil(m / 8)` workgroups.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: M×N f32 elements
    /// - `input_handle`: N f32 elements
    /// - `output_handle`: M f32 elements
    pub unsafe fn launch_plane<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Conservative: Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg = (m as u32).div_ceil(rows_per_wg).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            gemv_plane_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(weight_handle, m * n),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }

    /// Launch shared-memory tiled GEMV kernel (fallback, no subgroups).
    ///
    /// Each thread computes one output row. Input tiled into shared memory.
    /// Workgroup: 256 threads → 256 rows per workgroup.
    /// Dispatch: `ceil(m / 256)` workgroups.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: M×N f32 elements
    /// - `input_handle`: N f32 elements
    /// - `output_handle`: M f32 elements
    pub unsafe fn launch_tiled<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let wg_size = 256u32;
        let num_wg = (m as u32).div_ceil(wg_size).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            gemv_tile_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(weight_handle, m * n),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Batched GEMV launcher (Plan 436 T2.1)
// ---------------------------------------------------------------------------

/// CubeCL batched GEMV launcher.
///
/// Computes `output[batch, out_dim] = input[batch, in_dim] @ weight[out_dim, in_dim]^T`
/// in a single dispatch — one plane per output row, looping over all batch
/// positions. This is the GPU equivalent of the CPU `matmul_vec_batched`:
/// each weight row is read once and reused across all batch outputs.
///
/// For Weaver, `batch = seq_len = 5` and the 7 batched weight matrices
/// (w_c, w_q, w_k, w_v, w_o, w_gate, w_up) each dispatch once, reducing 40
/// single-GEMV dispatches to 7.
///
/// # Weight layout
///
/// The weight handle must be `[out_dim, in_dim]` row-major — i.e., each
/// contiguous row is one output's contribution from all input dimensions.
/// This is the transpose of the CPU `WeaverWeights` `[in_dim, out_dim]` layout.
/// Use [`crate::weaver_gpu::transpose_weight`] at upload time.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Wired in Plan 436 Phase 2 forward steps
impl GemvBatchedCubeCL {
    /// Launch batched plane GEMV.
    ///
    /// `output[b, j] = Σ_i input[b, i] * weight[j, i]` for all `b`, `j`.
    ///
    /// Dispatch: `ceil(out_dim / 8)` workgroups of 256 threads. Each workgroup
    /// processes 8 output rows (8 planes × plane_size 32 = 256 threads).
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: `out_dim * in_dim` f32 elements, `[out_dim, in_dim]` layout
    /// - `input_handle`: `batch * in_dim` f32 elements, `[batch, in_dim]` layout
    /// - `output_handle`: `batch * out_dim` f32 elements, `[batch, out_dim]` layout
    ///
    /// `batch`, `in_dim`, `out_dim` must all be > 0. The device must support
    /// plane (subgroup) operations.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Conservative: Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg = (out_dim as u32).div_ceil(rows_per_wg).max(1);

        // out_dim rides in params[2] — see the kernel doc for why it must NOT
        // be derived from `output.len()` (oversized-scratch mis-stride, Issue
        // 697/698 G3).
        let params: &[f32] = &[batch as f32, in_dim as f32, out_dim as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            gemv_batched_plane_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(weight_handle, out_dim * in_dim),
                BufferArg::from_raw_parts(input_handle, batch * in_dim),
                BufferArg::from_raw_parts(output_handle, batch * out_dim),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// CubeCL element-wise add launcher.
///
/// Wraps the `add_f32` kernel. Computes `output[i] = a[i] + b[i]`.
/// Used by the all-GPU LoRA path to combine base GEMV output with LoRA delta.
#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // GPU scaffolding — wired by the all-GPU LoRA path (Issue 429 clippy).
pub struct AddCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // GPU scaffolding — wired by the all-GPU LoRA path (Issue 429 clippy).
impl AddCubeCL {
    /// Launch element-wise add: `output[n] = a[n] + b[n]`.
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
        let num_wg = (n as u32).div_ceil(256u32).max(1);

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            add_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(a_handle, n),
                BufferArg::from_raw_parts(b_handle, n),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use cubecl::features::Plane;
    use crate::cubecl_runtime::ActiveRuntime;

    use crate::cubecl_runtime::CubeCLContext;

    use super::*;

    /// Verify plane GEMV with 4×4 identity matrix.
    #[test]
    fn test_gemv_plane_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 4usize;
        let n = 4usize;

        // Identity matrix (row-major)
        let weight: &[f32] = &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let expected: &[f32] = &[1.0, 2.0, 3.0, 4.0];

        let weight_handle = client.create_from_slice(f32::as_bytes(weight));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvCubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("plane GEMV identity: {output:?}");
    }

    /// Verify plane GEMV with a general 3×4 matrix.
    #[test]
    fn test_gemv_plane_general() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 3usize;
        let n = 4usize;
        let weight: &[f32] = &[
            1.0, 2.0, 3.0, 4.0, // row 0: 1+4+9+16 = 30
            5.0, 6.0, 7.0, 8.0, // row 1: 5+12+21+32 = 70
            9.0, 10.0, 11.0, 12.0, // row 2: 9+20+33+48 = 110
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let expected: &[f32] = &[30.0, 70.0, 110.0];

        let weight_handle = client.create_from_slice(f32::as_bytes(weight));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvCubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-3,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("plane GEMV general: {output:?}");
    }

    /// Verify tiled GEMV (shared memory fallback).
    #[test]
    fn test_gemv_tiled() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 4usize;
        let n = 8usize;

        // 4×8 partial identity: row i picks input[i]
        let weight: &[f32] = &[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected: &[f32] = &[1.0, 2.0, 3.0, 4.0];

        let weight_handle = client.create_from_slice(f32::as_bytes(weight));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvCubeCL::launch_tiled::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("tiled GEMV: {output:?}");
    }

    /// Verify auto-select picks plane or tiled based on device capability.
    #[test]
    fn test_gemv_auto_select() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 4usize;
        let n = 4usize;

        // Scaled identity: 2 * I
        let weight: &[f32] = &[
            2.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, 2.0,
        ];
        let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let expected: &[f32] = &[2.0, 4.0, 6.0, 8.0];

        let weight_handle = client.create_from_slice(f32::as_bytes(weight));
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvCubeCL::launch::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-5,
                "element {i}: expected {exp}, got {got}"
            );
        }

        let mode = if client.features().plane.contains(Plane::Ops) {
            "plane"
        } else {
            "tiled"
        };
        println!("auto-select GEMV ({mode}): {output:?}");
    }

    /// Verify plane GEMV with 256×256 diagonal matrix (typical LLM hidden dim).
    #[test]
    fn test_gemv_plane_large() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let m = 256usize;
        let n = 256usize;

        // Diagonal matrix with values 1..=256
        let mut weight = vec![0.0f32; m * n];
        for i in 0..m.min(n) {
            weight[i * n + i] = (i + 1) as f32;
        }
        // All-ones input → output[i] = (i+1)
        let input = vec![1.0f32; n];
        let expected: Vec<f32> = (0..m)
            .map(|i| if i < n { (i + 1) as f32 } else { 0.0 })
            .collect();

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvCubeCL::launch_plane::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(err < 1e-3, "element {i}: expected {exp}, got {got}");
        }
        println!("large GEMV ({m}×{n}): max_error = {max_err}");
    }

    /// Verify tiled GEMV with non-power-of-2 dimensions (tests partial tile handling).
    #[test]
    fn test_gemv_tiled_non_power_of_2() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let m = 3usize;
        let n = 300usize; // > 256, requires 2 tiles (256 + 44)

        // Row i: all zeros except position i = 1.0
        let mut weight = vec![0.0f32; m * n];
        for i in 0..m {
            weight[i * n + i] = 1.0;
        }
        // Input: 1.0, 2.0, 3.0, 4.0, ... (n values)
        let input: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        // output[i] = input[i] = (i+1)
        let expected: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvCubeCL::launch_tiled::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let bytes = client.read_one(output_handle).expect("should read output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), m);
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            assert!(
                (exp - got).abs() < 1e-3,
                "element {i}: expected {exp}, got {got}"
            );
        }
        println!("tiled GEMV non-power-of-2 ({m}×{n}): {output:?}");
    }

    /// Plan 481 T1: Test tiled GEMV with LoRA-sized small dimensions.
    ///
    /// These shapes correspond to the LoRA A/B adapter GEMVs that were
    /// reported as producing incorrect results (Issue 004 / Plan 481 Bug 2):
    ///   - LoRA A (Q):   2×32  (rank=2, in_dim=32)
    ///   - LoRA B (Q):  32×2   (out_dim=32, rank=2)
    ///   - LoRA B (K/V): 16×2  (out_dim=16, rank=2)
    ///   - 8×32, 32×8 for variety
    ///
    /// If this test PASSES, the tiled kernel is correct for small dims in
    /// isolation, and the historical bug was in autotune selection or buffer
    /// aliasing in the training forward — not the kernel itself.
    #[test]
    fn test_gemv_tiled_small_dim_lora() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let shapes: &[(usize, usize)] = &[
            (2, 32), // LoRA A (Q/KV): rank × in_dim
            (32, 2), // LoRA B (Q):   out_dim × rank
            (16, 2), // LoRA B (K/V): out_dim × rank
            (8, 32), // larger rank
            (32, 8), // transposed
            (2, 8),  // very small both
            (4, 2),  // n < 4
            (2, 4),  // m < 4
        ];

        for &(m, n) in shapes {
            // Random-ish weight and input (deterministic seed)
            let mut weight = vec![0.0f32; m * n];
            for (i, w) in weight.iter_mut().enumerate() {
                *w = ((i as f32 * 0.1).sin() * 10.0).round() / 10.0;
            }
            let input: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + 1.0) * 0.5).round() / 10.0)
                .collect();

            // CPU reference: output[j] = sum_k weight[j*n+k] * input[k]
            let expected: Vec<f32> = (0..m)
                .map(|j| {
                    let mut s = 0.0f32;
                    for k in 0..n {
                        s += weight[j * n + k] * input[k];
                    }
                    s
                })
                .collect();

            let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
            let input_handle = client.create_from_slice(f32::as_bytes(&input));
            let output_handle = client.empty(m * core::mem::size_of::<f32>());

            unsafe {
                GemvCubeCL::launch_tiled::<ActiveRuntime>(
                    &client,
                    weight_handle,
                    input_handle,
                    output_handle.clone(),
                    m,
                    n,
                );
            }

            let bytes = client.read_one(output_handle).expect("should read output");
            let output = f32::from_bytes(&bytes);

            assert_eq!(output.len(), m, "tiled ({m}×{n}): output length mismatch");
            let mut max_err = 0.0f32;
            for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
                let err = (exp - got).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 1e-4,
                    "tiled ({m}×{n}) element {i}: expected {exp}, got {got} (err {err})"
                );
            }
            println!("tiled GEMV small-dim ({m}×{n}): max_err = {max_err}");
        }
    }

    /// Plan 481 T1: Test plane GEMV with LoRA-sized small dimensions.
    ///
    /// The plane variant was reported as producing incorrect results for
    /// non-square matrices where m < n (e.g., K/V projections at 16×32).
    /// This test checks whether the plane kernel is correct in isolation.
    #[test]
    fn test_gemv_plane_small_dim_lora() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let shapes: &[(usize, usize)] = &[
            (2, 32),  // LoRA A (Q/KV): m < n
            (32, 2),  // LoRA B (Q):   m > n
            (16, 2),  // LoRA B (K/V): m > n
            (16, 32), // K/V projection: m < n (the reported bug case)
            (8, 32),  // larger rank
            (32, 8),  // transposed
        ];

        for &(m, n) in shapes {
            let mut weight = vec![0.0f32; m * n];
            for (i, w) in weight.iter_mut().enumerate() {
                *w = ((i as f32 * 0.1).sin() * 10.0).round() / 10.0;
            }
            let input: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + 1.0) * 0.5).round() / 10.0)
                .collect();

            // CPU reference
            let expected: Vec<f32> = (0..m)
                .map(|j| {
                    let mut s = 0.0f32;
                    for k in 0..n {
                        s += weight[j * n + k] * input[k];
                    }
                    s
                })
                .collect();

            let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
            let input_handle = client.create_from_slice(f32::as_bytes(&input));
            let output_handle = client.empty(m * core::mem::size_of::<f32>());

            unsafe {
                GemvCubeCL::launch_plane::<ActiveRuntime>(
                    &client,
                    weight_handle,
                    input_handle,
                    output_handle.clone(),
                    m,
                    n,
                );
            }

            let bytes = client.read_one(output_handle).expect("should read output");
            let output = f32::from_bytes(&bytes);

            assert_eq!(output.len(), m, "plane ({m}×{n}): output length mismatch");
            let mut max_err = 0.0f32;
            for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
                let err = (exp - got).abs();
                if err > max_err {
                    max_err = err;
                }
                assert!(
                    err < 1e-3,
                    "plane ({m}×{n}) element {i}: expected {exp}, got {got} (err {err})"
                );
            }
            println!("plane GEMV small-dim ({m}×{n}): max_err = {max_err}");
        }
    }

    // -----------------------------------------------------------------------
    // Batched GEMV tests (Plan 436 T2.1)
    // -----------------------------------------------------------------------

    /// CPU reference for batched matmul: `output[batch, out] = input[batch, in] @ weight[in, out]`.
    ///
    /// Mirrors `katgpt_speculative::weaver::matmul_vec_batched` — weight is
    /// `[in_dim, out_dim]` row-major (the CPU Weaver layout).
    fn cpu_matmul_vec_batched(
        input: &[f32],
        weight: &[f32],
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        output: &mut [f32],
    ) {
        output[..batch * out_dim].fill(0.0);
        for i in 0..in_dim {
            let row = &weight[i * out_dim..(i + 1) * out_dim];
            for b in 0..batch {
                let xi = input[b * in_dim + i];
                let out_row = &mut output[b * out_dim..(b + 1) * out_dim];
                for j in 0..out_dim {
                    out_row[j] += xi * row[j];
                }
            }
        }
    }

    /// Transpose `[in_dim, out_dim]` → `[out_dim, in_dim]`.
    ///
    /// The GPU plane GEMV kernels expect `[out_dim, in_dim]` layout (each
    /// contiguous row is an output's contribution from all inputs). The CPU
    /// `WeaverWeights` stores `[in_dim, out_dim]`. This helper bridges the gap
    /// at upload time.
    fn transpose_weight(weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; in_dim * out_dim];
        for i in 0..in_dim {
            for j in 0..out_dim {
                out[j * in_dim + i] = weight[i * out_dim + j];
            }
        }
        out
    }

    /// Plan 436 T2.1: Batched GEMV parity test.
    ///
    /// Generates a random `[in_dim, out_dim]` weight (CPU layout), transposes
    /// it to GPU layout, runs the batched plane GEMV kernel, and compares
    /// against the CPU `matmul_vec_batched` reference.
    ///
    /// Uses Weaver-shaped dimensions (hidden=128, seq_len=5) at a small scale
    /// to keep the test fast while exercising the plane loop + batch iteration.
    #[test]
    fn test_gemv_batched_plane_weaver_shape() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let batch = 5usize; // seq_len = max_depth + 1
        let in_dim = 128usize; // small hidden for fast test (real: 2304)
        let out_dim = 128usize; // square for conditioning/QKV/O

        // Random-ish input and weight (deterministic LCG for reproducibility).
        let mut seed = 42u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0 // range [-1, 1)
        };
        let input: Vec<f32> = (0..batch * in_dim).map(|_| lcg()).collect();
        let weight_cpu: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg()).collect();

        // CPU reference.
        let mut expected = vec![0.0f32; batch * out_dim];
        cpu_matmul_vec_batched(&input, &weight_cpu, in_dim, out_dim, batch, &mut expected);

        // Transpose weight to GPU layout [out_dim, in_dim].
        let weight_gpu = transpose_weight(&weight_cpu, in_dim, out_dim);

        // Upload to GPU.
        let weight_handle = client.create_from_slice(f32::as_bytes(&weight_gpu));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(batch * out_dim * core::mem::size_of::<f32>());

        unsafe {
            GemvBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                batch,
                in_dim,
                out_dim,
            );
        }

        let bytes = client
            .read_one(output_handle)
            .expect("should read batched GEMV output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), batch * out_dim, "batched GEMV: output length");
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-3,
                "batched GEMV element {i}: expected {exp}, got {got} (err {err})"
            );
        }
        println!("batched plane GEMV ({batch}×{in_dim}×{out_dim}): max_err = {max_err}");
    }

    /// Plan 436 T2.1: Batched GEMV with non-square weight (gate/up projection shape).
    ///
    /// Tests `[hidden, d_ff]` weight — in_dim=128, out_dim=256. This is the
    /// w_gate / w_up shape (h → d_ff). Ensures the kernel handles rectangular
    /// matrices where out_dim > in_dim.
    #[test]
    fn test_gemv_batched_plane_rectangular() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        let batch = 5usize;
        let in_dim = 128usize; // hidden
        let out_dim = 256usize; // d_ff (larger than hidden)

        let mut seed = 12345u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let input: Vec<f32> = (0..batch * in_dim).map(|_| lcg()).collect();
        let weight_cpu: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg()).collect();

        let mut expected = vec![0.0f32; batch * out_dim];
        cpu_matmul_vec_batched(&input, &weight_cpu, in_dim, out_dim, batch, &mut expected);

        let weight_gpu = transpose_weight(&weight_cpu, in_dim, out_dim);

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight_gpu));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(batch * out_dim * core::mem::size_of::<f32>());

        unsafe {
            GemvBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                batch,
                in_dim,
                out_dim,
            );
        }

        let bytes = client
            .read_one(output_handle)
            .expect("should read batched GEMV output");
        let output = f32::from_bytes(&bytes);

        assert_eq!(output.len(), batch * out_dim, "batched GEMV: output length");
        let mut max_err = 0.0f32;
        for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
            let err = (exp - got).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < 1e-3,
                "batched GEMV (rect) element {i}: expected {exp}, got {got} (err {err})"
            );
        }
        println!("batched plane GEMV rect ({batch}×{in_dim}×{out_dim}): max_err = {max_err}");
    }

    /// Issue 697/698 G3 regression: batched GEMV into an OVERSIZED output
    /// buffer must write batch rows at the correct `b * out_dim + row`
    /// offsets.
    ///
    /// The weaver per-depth path launches `batch = seq_len = 2` into a
    /// `(max_depth + 1)`-row scratch (5 rows at default config). The kernel
    /// previously derived `out_dim = output.len() / batch` — but
    /// `output.len()` is the FULL allocation size (5·h), so out_dim computed
    /// as 160 (for h=64) instead of 64, and every batch row ≥ 1 landed at a
    /// stray offset (row 1 written to [160..224) while the consumer read
    /// [64..128)). This test pins the correct behavior with an oversized
    /// buffer — the exact trap that let every existing exact-size test pass
    /// while the weaver per-depth G3 CPU↔GPU parity was broken.
    #[test]
    fn test_gemv_batched_plane_oversized_output_weaver_scratch() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        if !client.features().plane.contains(Plane::Ops) {
            eprintln!("Skipping: device does not support plane ops");
            return;
        }

        // Weaver per-depth shape: batch=2 (seq_len), h=64; scratch sized for
        // max_depth+1 = 5 rows → 320-float output buffer (2.5× oversized).
        let batch = 2usize;
        let in_dim = 64usize;
        let out_dim = 64usize;
        let scratch_rows = 5usize;

        let mut seed = 99u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let input: Vec<f32> = (0..batch * in_dim).map(|_| lcg()).collect();
        let weight_cpu: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg()).collect();

        let mut expected = vec![0.0f32; batch * out_dim];
        cpu_matmul_vec_batched(&input, &weight_cpu, in_dim, out_dim, batch, &mut expected);

        let weight_gpu = transpose_weight(&weight_cpu, in_dim, out_dim);
        let weight_handle = client.create_from_slice(f32::as_bytes(&weight_gpu));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        // OVERSIZED: scratch_rows * out_dim instead of batch * out_dim.
        let output_handle =
            client.empty(scratch_rows * out_dim * core::mem::size_of::<f32>());

        unsafe {
            GemvBatchedCubeCL::launch::<ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                batch,
                in_dim,
                out_dim,
            );
        }

        let bytes = client
            .read_one(output_handle)
            .expect("should read batched GEMV output");
        let output = f32::from_bytes(&bytes);
        assert_eq!(output.len(), scratch_rows * out_dim);

        // Both logical batch rows must land at [0..64) and [64..128).
        for b in 0..batch {
            for j in 0..out_dim {
                let exp = expected[b * out_dim + j];
                let got = output[b * out_dim + j];
                assert!(
                    (exp - got).abs() < 1e-3,
                    "oversized-buffer row {b} col {j}: expected {exp}, got {got} \
                     (kernel wrote rows at wrong offsets — Issue 697/698 G3 regression)"
                );
            }
        }
        // And NOTHING may be written into the oversized tail [batch*out_dim..
        // scratch_rows*out_dim) — the pre-fix kernel wrote row 1's results at
        // [out_dim_wrong..) = [160..224) in the 320-float buffer.
        for (k, &v) in output
            .iter()
            .enumerate()
            .skip(batch * out_dim)
            .take(scratch_rows * out_dim - batch * out_dim)
        {
            assert_eq!(
                v, 0.0,
                "oversized tail [{k}] = {v} — kernel wrote beyond batch*out_dim",
            );
        }
        println!(
            "batched plane GEMV oversized-output ({batch}×{in_dim}×{out_dim} into {scratch_rows} rows): OK, tail clean"
        );
    }
}
