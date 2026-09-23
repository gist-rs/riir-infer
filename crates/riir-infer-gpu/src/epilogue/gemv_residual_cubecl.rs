//! Fused GEMV + ResidualAdd CubeCL kernels (Plan 106 Track 3 T2.10/T2.12).
//!
//! Single-dispatch kernels that compute `output[row] = dot(weight_row, input) + residual[row]`,
//! eliminating one GPU dispatch per use in the Gemma 2 decode path.
//!
//! # Usage Sites
//!
//! In Gemma 2's post-norm architecture, after each sub-layer projection the output
//! is normed then added to the residual. However, some fusion patterns combine
//! the projection + residual add directly:
//!
//! ```text
//! Post-attention (CODA variant):
//!   hidden = Wo @ attn_out + residual    ← fused (GEMV + residual)
//!   normed = rmsnorm(hidden)             ← separate or fused with next
//!
//! Post-MLP (CODA variant):
//!   hidden = down @ mlp_hidden + residual2  ← fused (GEMV + residual)
//!   normed = rmsnorm(hidden)                 ← separate or fused with next
//! ```
//!
//! # Dispatch Savings
//!
//! | Fusion | Before | After | Savings |
//! |--------|--------|-------|---------|
//! | GEMV(Wo) + ResidualAdd | 2 dispatches | 1 | 1 |
//! | GEMV(down) + ResidualAdd | 2 dispatches | 1 | 1 |
//! | **Per layer** | | | **2** |
//! | **26 layers** | | | **52** |
//!
//! Combined with fused RMSNorm + ResidualAdd (52 savings), total = **104 dispatches saved**.
//!
//! # CubeCL v0.10 Constraints
//!
//! - Exactly 4 Array parameters (weight, input, residual, output).
//! - Plane (subgroup) cooperative dot product with `plane_sum()` reduction.
//! - `ABSOLUTE_POS_X / PLANE_DIM` for row index, `UNIT_POS_PLANE` for lane.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Fused GEMV + ResidualAdd: F32 weights
// ---------------------------------------------------------------------------

/// CubeCL fused GEMV + ResidualAdd kernel (F32 weights, plane cooperative).
///
/// Computes `output[row] = dot(weight_row, input) + residual[row]`.
///
/// Each plane (subgroup) handles one output row via cooperative dot product.
/// Lane 0 performs the residual add after `plane_sum()` reduction.
///
/// ## Parameter Layout
///
/// - `weight`: `[f32; m * n]` — weight matrix (row-major, m output rows × n input cols).
/// - `input`: `[f32; n]` — input vector.
/// - `residual`: `[f32; m]` — residual connection vector.
/// - `output`: `[f32; m]` — GEMV output + residual.
///
/// ## Dispatch
///
/// `CubeCount::Static(m, 1, 1)`, `CubeDim::new_1d(plane_size)`.
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_plane_residual_f32(
    weight: &[f32],
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let row_offset = row * n;

    // Cooperative dot product: lane j handles elements j, j + PLANE_DIM, ...
    let mut partial = f32::new(0.0f32);

    let mut k = lane;
    while k < n {
        partial += weight[(row_offset + k) as usize] * input[k as usize];
        k += PLANE_DIM;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane.
    let result = plane_sum(partial);

    // Lane 0 writes the final result + residual.
    if lane == 0u32 {
        output[row as usize] = result + residual[row as usize];
    }
}

// ---------------------------------------------------------------------------
// Fused GEMV + ResidualAdd: F16 weights
// ---------------------------------------------------------------------------

/// CubeCL fused GEMV + ResidualAdd kernel (F16 weights, plane cooperative).
///
/// Computes `output[row] = dot(f16_weight_row, input) + residual[row]`.
/// f16 weights are cast to f32 during accumulation for f32 precision.
///
/// ## Parameter Layout
///
/// - `weight`: `[f16; m * n]` — weight matrix in f16 format.
/// - `input`: `[f32; n]` — input vector (f32).
/// - `residual`: `[f32; m]` — residual connection vector.
/// - `output`: `[f32; m]` — GEMV output + residual.
///
/// ## Dispatch
///
/// `CubeCount::Static(m, 1, 1)`, `CubeDim::new_1d(plane_size)`.
#[cfg(feature = "cubecl_runtime")]
#[cfg_attr(
    feature = "gemv_fma_contract",
    cube(launch_unchecked, fast_math = FastMath::AllowContraction.into())
)]
#[cfg_attr(
    not(feature = "gemv_fma_contract"),
    cube(launch_unchecked)
)]
fn gemv_plane_residual_f16(
    weight: &[half::f16],
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    let row_offset = row * n;

    // Cooperative dot product: lane j handles elements j, j + PLANE_DIM, ...
    // f16 weight cast to f32 before accumulation.
    let mut partial = f32::new(0.0f32);

    let mut k = lane;
    while k < n {
        let w = f32::cast_from(weight[(row_offset + k) as usize]);
        partial += w * input[k as usize];
        k += PLANE_DIM;
    }

    // Hardware SIMD reduction: sum all lane partials in the plane.
    let result = plane_sum(partial);

    // Lane 0 writes the final result + residual.
    if lane == 0u32 {
        output[row as usize] = result + residual[row as usize];
    }
}

// ---------------------------------------------------------------------------
// Launcher structs
// ---------------------------------------------------------------------------

/// CubeCL fused GEMV + ResidualAdd launcher (F32 weights).
///
/// Wraps `gemv_plane_residual_f32` with correct dispatch parameters.
/// Each plane (subgroup) handles one output row — fully coalesced memory access.
///
/// Replaces two separate dispatches:
/// 1. `GemvCubeCL::launch(weight, input, output, m, n)`
/// 2. `ResidualAddCubeCL::launch(output, residual, output, m)`
///
/// With one fused dispatch:
/// `GemvResidualCubeCL::launch(weight, input, residual, output, m, n)`
#[cfg(feature = "cubecl_runtime")]
pub struct GemvResidualCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvResidualCubeCL {
    /// Launch fused GEMV + ResidualAdd kernel (F32 weights).
    ///
    /// Computes `output[row] = dot(weight_row, input) + residual[row]`
    /// for each output row in a single GPU dispatch.
    ///
    /// Dispatch: `m` workgroups of `plane_size` threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: `m * n` f32 elements
    /// - `input_handle`: `n` f32 elements
    /// - `residual_handle`: `m` f32 elements
    /// - `output_handle`: `m` f32 elements
    ///
    /// The client must support `Plane::Ops` (subgroup operations).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        residual_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let plane_size = 32u32; // Metal simdgroup size

        // SAFETY: Caller guarantees correct buffer sizes and plane support.
        unsafe {
            gemv_plane_residual_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(m as u32, 1, 1),
                CubeDim::new_1d(plane_size),
                BufferArg::from_raw_parts(weight_handle, m * n),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(residual_handle, m),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }
}

/// CubeCL fused GEMV + ResidualAdd launcher (F16 weights).
///
/// Wraps `gemv_plane_residual_f16` with correct dispatch parameters.
/// f16 weights provide 2× bandwidth savings over f32 with f32 accumulation precision.
///
/// Replaces two separate dispatches with one fused dispatch, same as F32 variant.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvResidualF16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvResidualF16CubeCL {
    /// Launch fused GEMV + ResidualAdd kernel (F16 weights).
    ///
    /// Computes `output[row] = dot(f16_weight_row, input) + residual[row]`
    /// for each output row in a single GPU dispatch.
    ///
    /// Dispatch: `m` workgroups of `plane_size` threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_handle`: `m * n` f16 elements
    /// - `input_handle`: `n` f32 elements
    /// - `residual_handle`: `m` f32 elements
    /// - `output_handle`: `m` f32 elements
    ///
    /// The client must support `Plane::Ops` (subgroup operations).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_handle: Handle,
        input_handle: Handle,
        residual_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let plane_size = 32u32; // Metal simdgroup size

        // SAFETY: Caller guarantees correct buffer sizes and plane support.
        unsafe {
            gemv_plane_residual_f16::launch_unchecked::<R>(
                client,
                CubeCount::Static(m as u32, 1, 1),
                CubeDim::new_1d(plane_size),
                BufferArg::from_raw_parts(weight_handle, m * n),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(residual_handle, m),
                BufferArg::from_raw_parts(output_handle, m),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::CubeCLContext;

    /// CPU reference: GEMV + ResidualAdd fused.
    fn gemv_residual_cpu(
        weight: &[f32],
        input: &[f32],
        residual: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m];
        for row in 0..m {
            let mut dot = 0.0f32;
            for col in 0..n {
                dot += weight[row * n + col] * input[col];
            }
            output[row] = dot + residual[row];
        }
        output
    }

    /// CPU reference: standalone GEMV.
    fn gemv_cpu(weight: &[f32], input: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; m];
        for row in 0..m {
            let mut dot = 0.0f32;
            for col in 0..n {
                dot += weight[row * n + col] * input[col];
            }
            output[row] = dot;
        }
        output
    }

    fn max_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_gemv_residual_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let m = 4;
        let n = 4;

        // Identity matrix, ones input, zero residual → output = input
        let weight: Vec<f32> = (0..m * n)
            .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
            .collect();
        let input = vec![1.0f32; n];
        let residual = vec![0.0f32; m];

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(m * 4);

        unsafe {
            GemvResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                residual_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = vec![1.0f32; m];
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_gemv_residual_known_values() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let m = 3;
        let n = 4;

        // weight = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        let weight: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let input = vec![1.0, 2.0, 3.0, 4.0f32];
        let residual = vec![0.1, 0.2, 0.3f32];

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(m * 4);

        unsafe {
            GemvResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                residual_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = gemv_residual_cpu(&weight, &input, &residual, m, n);
        let err = max_error(output, &expected);
        assert!(err < 1e-4, "max error {err} exceeds tolerance");

        // Verify manually:
        // row 0: 1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16 = 30, + 0.1 = 30.1
        // row 1: 5*1 + 6*2 + 7*3 + 8*4 = 5+12+21+32 = 70, + 0.2 = 70.2
        // row 2: 9*1 + 10*2 + 11*3 + 12*4 = 9+20+33+48 = 110, + 0.3 = 110.3
        assert!((output[0] - 30.1).abs() < 1e-4);
        assert!((output[1] - 70.2).abs() < 1e-4);
        assert!((output[2] - 110.3).abs() < 1e-4);
    }

    #[test]
    fn test_gemv_residual_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let m = 64;
        let n = 128;

        let weight: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 3.7) % 2.0) - 1.0).collect();
        let residual: Vec<f32> = (0..m).map(|i| (i as f32 * 0.1).cos()).collect();

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let fused_output_handle = client.empty(m * 4);

        // Fused kernel
        unsafe {
            GemvResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle.clone(),
                input_handle.clone(),
                residual_handle.clone(),
                fused_output_handle.clone(),
                m,
                n,
            );
        }

        let fused_bytes = client.read_one(fused_output_handle).unwrap();
        let fused_output = f32::from_bytes(&fused_bytes);

        // Separate kernels: GEMV then ResidualAdd
        let gemv_output_handle = client.empty(m * 4);
        unsafe {
            crate::gemv_cubecl::GemvCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                gemv_output_handle.clone(),
                m,
                n,
            );
        }

        let separate_output_handle = client.empty(m * 4);
        unsafe {
            crate::norms_cubecl::ResidualAddCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gemv_output_handle,
                residual_handle,
                separate_output_handle.clone(),
                m,
            );
        }

        let separate_bytes = client.read_one(separate_output_handle).unwrap();
        let separate_output = f32::from_bytes(&separate_bytes);

        // Fused should match separate to within floating-point precision
        let err = max_error(fused_output, separate_output);
        assert!(
            err < 1e-5,
            "fused vs separate max error {err} exceeds tolerance"
        );
    }

    #[test]
    fn test_gemv_residual_gemma2_dims() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        // Wo: [n_embd=2048, n_heads*head_dim=2048] — square for Gemma 2 2B
        let m = 2048;
        let n = 2048;

        let weight: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let residual: Vec<f32> = (0..m).map(|i| (i as f32 * 0.005).sin()).collect();

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(m * 4);

        unsafe {
            GemvResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                residual_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = gemv_residual_cpu(&weight, &input, &residual, m, n);
        let err = max_error(output, &expected);
        assert!(
            err < 1e-3,
            "max error {err} exceeds tolerance for gemma2 dims {m}×{n}"
        );
    }

    #[test]
    fn test_gemv_residual_zero_residual() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let m = 8;
        let n = 16;

        let weight: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let residual = vec![0.0f32; m];

        let weight_handle = client.create_from_slice(f32::as_bytes(&weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(m * 4);

        unsafe {
            GemvResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle.clone(),
                input_handle.clone(),
                residual_handle.clone(),
                output_handle.clone(),
                m,
                n,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        // With zero residual, result should match standalone GEMV
        let gemv_expected = gemv_cpu(&weight, &input, m, n);
        let err = max_error(output, &gemv_expected);
        assert!(
            err < 1e-5,
            "max error vs standalone GEMV {err} exceeds tolerance"
        );
    }

    // ── F16 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_gemv_residual_f16_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let m = 4;
        let n = 4;

        let weight_f32: Vec<f32> = (0..m * n)
            .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
            .collect();
        let weight_f16: Vec<half::f16> =
            weight_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let input = vec![1.0f32; n];
        let residual = vec![0.0f32; m];

        let weight_handle = client.create_from_slice(bytemuck::cast_slice(&weight_f16));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(m * 4);

        unsafe {
            GemvResidualF16CubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                residual_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = vec![1.0f32; m];
        let err = max_error(output, &expected);
        assert!(err < 1e-2, "max error {err} exceeds f16 tolerance");
    }

    #[test]
    fn test_gemv_residual_f16_matches_f32() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let m = 32;
        let n = 64;

        let weight_f32: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let weight_f16: Vec<half::f16> =
            weight_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 2.3) % 2.0) - 1.0).collect();
        let residual: Vec<f32> = (0..m).map(|i| (i as f32 * 0.1).cos()).collect();

        // F16 fused
        let weight_f16_handle = client.create_from_slice(bytemuck::cast_slice(&weight_f16));
        let input_handle_f16 = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle_f16 = client.create_from_slice(f32::as_bytes(&residual));
        let output_f16_handle = client.empty(m * 4);

        unsafe {
            GemvResidualF16CubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_f16_handle,
                input_handle_f16,
                residual_handle_f16,
                output_f16_handle.clone(),
                m,
                n,
            );
        }

        let output_f16_bytes = client.read_one(output_f16_handle).unwrap();
        let output_f16 = f32::from_bytes(&output_f16_bytes);

        // F32 fused
        let weight_f32_handle = client.create_from_slice(f32::as_bytes(&weight_f32));
        let input_handle_f32 = client.create_from_slice(f32::as_bytes(&input));
        let residual_handle_f32 = client.create_from_slice(f32::as_bytes(&residual));
        let output_f32_handle = client.empty(m * 4);

        unsafe {
            GemvResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_f32_handle,
                input_handle_f32,
                residual_handle_f32,
                output_f32_handle.clone(),
                m,
                n,
            );
        }

        let output_f32_bytes = client.read_one(output_f32_handle).unwrap();
        let output_f32 = f32::from_bytes(&output_f32_bytes);

        // F16 should be close to F32 but with f16 precision loss
        let err = max_error(output_f16, output_f32);
        assert!(
            err < 0.1,
            "f16 vs f32 max error {err} exceeds f16 tolerance"
        );
    }
}
