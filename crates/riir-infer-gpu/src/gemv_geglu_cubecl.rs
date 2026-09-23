//! Fused dual GEMV + GeGLU CubeCL kernel with F32 weights (Plan 106 Track 3 T2.13 alternative).
//!
//! Single-dispatch kernel that computes:
//! 1. `gate[row] = dot(weight_gate[row], input)` — GEMV with f32 weights
//! 2. `up[row]   = dot(weight_up[row], input)`   — GEMV with f32 weights
//! 3. `output[row] = GELU(gate[row]) * up[row]`  — GeGLU activation
//!
//! This replaces 3 separate GPU dispatches (gate GEMV + up GEMV + GeGLU) with 1,
//! saving 2 dispatches per layer × 26 layers = **52 dispatches** per decode token
//! for the F32 weight path.
//!
//! # Dispatch Savings
//!
//! | Location | Before | After | Savings |
//! |----------|--------|-------|---------|
//! | MLP gate+up+GeGLU (F32) | 3 dispatches | 1 | 2 |
//! | **Per layer** | | | **2** |
//! | **26 layers** | | | **52** |
//!
//! # Algorithm
//!
//! Plane (subgroup) cooperative:
//! - Two passes per workgroup: first computes gate dot product, then up dot product.
//! - Each plane handles one output row for each pass.
//! - After both dot products, lane 0 applies GeGLU and writes the result.
//!
//! # CubeCL v0.10 Constraints
//!
//! - Exactly 4 Array parameters (weight_gate, weight_up, input, output).
//! - Plane (subgroup) cooperative dot product with `plane_sum()` reduction.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use cubecl::features::Plane;

// ---------------------------------------------------------------------------
// Fused dual GEMV + GeGLU: F32 weights, plane cooperative
// ---------------------------------------------------------------------------

/// CubeCL fused dual GEMV + GeGLU kernel (F32 weights, plane cooperative).
///
/// Computes for each output row:
/// 1. `gate = dot(weight_gate_row, input)` — cooperative dot product
/// 2. `up   = dot(weight_up_row, input)`   — cooperative dot product
/// 3. `output = GELU(gate) * up`           — GeGLU activation
///
/// All in a single GPU dispatch, eliminating 2 dispatch overheads vs separate ops.
///
/// ## Parameter Layout
///
/// - `weight_gate`: `[f32; m * n]` — gate projection weights (row-major).
/// - `weight_up`: `[f32; m * n]` — up projection weights (row-major).
/// - `input`: `[f32; n]` — input vector.
/// - `output`: `[f32; m]` — GeGLU result.
///
/// ## Dispatch
///
/// `CubeCount::Static(m, 1, 1)`, `CubeDim::new_1d(plane_size)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_geglu_plane_f32(
    weight_gate: &[f32],
    weight_up: &[f32],
    input: &[f32],
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

    // ── Gate dot product ──
    let mut gate_partial = f32::new(0.0f32);
    let mut k = lane;
    while k < n {
        gate_partial += weight_gate[(row_offset + k) as usize] * input[k as usize];
        k += PLANE_DIM;
    }
    let gate = plane_sum(gate_partial);

    // ── Up dot product ──
    let mut up_partial = f32::new(0.0f32);
    let mut k2 = lane;
    while k2 < n {
        up_partial += weight_up[(row_offset + k2) as usize] * input[k2 as usize];
        k2 += PLANE_DIM;
    }
    let up = plane_sum(up_partial);

    // ── GeGLU: output = GELU(gate) * up ──
    // Only lane 0 has the final reduced values.
    if lane == 0u32 {
        // GELU tanh approximation:
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_over_pi = f32::new(0.797_884_6f32);
        let coeff = f32::new(0.044715f32);
        let half_val = f32::new(0.5f32);
        let one = f32::new(1.0f32);

        let x_cubed = gate * gate * gate;
        let inner = sqrt_2_over_pi * (gate + coeff * x_cubed);
        let tanh_val = f32::tanh(inner);
        let gelu = half_val * gate * (one + tanh_val);

        output[row as usize] = gelu * up;
    }
}

// ---------------------------------------------------------------------------
// Tiled fallback (no subgroups)
// ---------------------------------------------------------------------------

/// Tiled dual GEMV + GeGLU fallback kernel (F32 weights, no subgroups).
///
/// Uses `UNIT_POS` to identify the thread (row). Each thread computes one output row
/// by iterating over all input columns for both gate and up dot products,
/// then applies GeGLU.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_geglu_tiled_f32(
    weight_gate: &[f32],
    weight_up: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let n = input.len() as u32;
    let m = output.len() as u32;

    let row = UNIT_POS;

    if row >= m {
        terminate!();
    }

    let row_offset = row * n;

    // ── Gate dot product ──
    let mut gate = f32::new(0.0f32);
    let mut k = 0u32;
    while k < n {
        gate += weight_gate[(row_offset + k) as usize] * input[k as usize];
        k += 1u32;
    }

    // ── Up dot product ──
    let mut up = f32::new(0.0f32);
    let mut k2 = 0u32;
    while k2 < n {
        up += weight_up[(row_offset + k2) as usize] * input[k2 as usize];
        k2 += 1u32;
    }

    // ── GeGLU: output = GELU(gate) * up ──
    let sqrt_2_over_pi = f32::new(0.797_884_6f32);
    let coeff = f32::new(0.044715f32);
    let half_val = f32::new(0.5f32);
    let one = f32::new(1.0f32);

    let x_cubed = gate * gate * gate;
    let inner = sqrt_2_over_pi * (gate + coeff * x_cubed);
    let tanh_val = f32::tanh(inner);
    let gelu = half_val * gate * (one + tanh_val);

    output[row as usize] = gelu * up;
}

// ---------------------------------------------------------------------------
// Launcher struct
// ---------------------------------------------------------------------------

/// CubeCL fused dual GEMV + GeGLU launcher (F32 weights).
///
/// Replaces three separate dispatches:
/// 1. `GemvCubeCL::launch(weight_gate, input, gate_output, m, n)`
/// 2. `GemvCubeCL::launch(weight_up, input, up_output, m, n)`
/// 3. `GegluCubeCL::launch(gate_output, up_output, output, m)`
///
/// With one fused dispatch:
/// `GemvGegluCubeCL::launch(weight_gate, weight_up, input, output, m, n)`
#[cfg(feature = "cubecl_runtime")]
pub struct GemvGegluCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvGegluCubeCL {
    /// Launch fused dual GEMV + GeGLU kernel (F32 weights).
    ///
    /// Auto-selects plane (subgroup) or tiled fallback based on client features.
    ///
    /// Dispatch: `m` workgroups of `plane_size` or `256` threads.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_gate_handle`: `m * n` f32 elements
    /// - `weight_up_handle`: `m * n` f32 elements
    /// - `input_handle`: `n` f32 elements
    /// - `output_handle`: `m` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_gate_handle: Handle,
        weight_up_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        m: usize,
        n: usize,
    ) {
        let has_planes = client.features().plane.contains(Plane::Ops);

        if has_planes {
            let plane_size = 32u32; // Metal simdgroup size
            // SAFETY: Caller guarantees correct buffer sizes.
            unsafe {
                gemv_geglu_plane_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(m as u32, 1, 1),
                    CubeDim::new_1d(plane_size),
                    BufferArg::from_raw_parts(weight_gate_handle, m * n),
                    BufferArg::from_raw_parts(weight_up_handle, m * n),
                    BufferArg::from_raw_parts(input_handle, n),
                    BufferArg::from_raw_parts(output_handle, m),
                );
            }
        } else {
            // Tiled fallback: each thread computes one row
            let cube_dim = 256u32;
            // SAFETY: Caller guarantees correct buffer sizes.
            unsafe {
                gemv_geglu_tiled_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(m as u32, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(weight_gate_handle, m * n),
                    BufferArg::from_raw_parts(weight_up_handle, m * n),
                    BufferArg::from_raw_parts(input_handle, n),
                    BufferArg::from_raw_parts(output_handle, m),
                );
            }
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

    /// CPU reference: dual GEMV + GeGLU.
    fn gemv_geglu_cpu(
        weight_gate: &[f32],
        weight_up: &[f32],
        input: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m];
        for row in 0..m {
            let mut gate = 0.0f32;
            let mut up = 0.0f32;
            for col in 0..n {
                gate += weight_gate[row * n + col] * input[col];
                up += weight_up[row * n + col] * input[col];
            }
            // GELU tanh
            let x_cubed = gate * gate * gate;
            let inner = 0.797_884_6 * (gate + 0.044715 * x_cubed);
            let tanh_val = inner.tanh();
            let gelu = 0.5 * gate * (1.0 + tanh_val);
            output[row] = gelu * up;
        }
        output
    }

    #[test]
    fn test_gemv_geglu_f32_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL init");
        let client = ctx.client();

        // Identity weight (2×2), gate=[1,0; 0,1], up=[1,0; 0,1], input=[1,1]
        let gate_weight = vec![1.0f32, 0.0, 0.0, 1.0];
        let up_weight = vec![1.0f32, 0.0, 0.0, 1.0];
        let input = vec![1.0f32, 1.0];

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate_weight));
        let up_handle = client.create_from_slice(f32::as_bytes(&up_weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(2 * core::mem::size_of::<f32>());

        unsafe {
            GemvGegluCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                input_handle,
                output_handle.clone(),
                2,
                2,
            );
        }

        let result_bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&result_bytes);
        let expected = gemv_geglu_cpu(&gate_weight, &up_weight, &input, 2, 2);

        for i in 0..2 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-6,
                "Element {i}: expected {}, got {}",
                expected[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_gemv_geglu_f32_general() {
        let ctx = CubeCLContext::new().expect("CubeCL init");
        let client = ctx.client();

        // 3×4 matrices
        let gate_weight = vec![
            1.0f32, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5, -1.0, 2.0, -3.0, 4.0,
        ];
        let up_weight = vec![
            0.1f32, 0.2, 0.3, 0.4, 1.0, 1.0, 1.0, 1.0, -0.5, 0.5, -0.5, 0.5,
        ];
        let input = vec![1.0f32, -1.0, 2.0, -2.0];

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate_weight));
        let up_handle = client.create_from_slice(f32::as_bytes(&up_weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(3 * core::mem::size_of::<f32>());

        unsafe {
            GemvGegluCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                input_handle,
                output_handle.clone(),
                3,
                4,
            );
        }

        let result_bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&result_bytes);
        let expected = gemv_geglu_cpu(&gate_weight, &up_weight, &input, 3, 4);

        for i in 0..3 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Element {i}: expected {}, got {} (diff {})",
                expected[i],
                result[i],
                (result[i] - expected[i]).abs()
            );
        }
    }

    #[test]
    fn test_gemv_geglu_f32_gemma2_dims() {
        let ctx = CubeCLContext::new().expect("CubeCL init");
        let client = ctx.client();

        // Gemma 2 MLP dimensions scaled down: m=64, n=128
        let m = 64usize;
        let n = 128usize;

        let gate_weight: Vec<f32> = (0..m * n)
            .map(|i| ((i % 97) as f32 - 48.0) / 100.0)
            .collect();
        let up_weight: Vec<f32> = (0..m * n)
            .map(|i| ((i % 53) as f32 - 26.0) / 100.0)
            .collect();
        let input: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate_weight));
        let up_handle = client.create_from_slice(f32::as_bytes(&up_weight));
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvGegluCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                input_handle,
                output_handle.clone(),
                m,
                n,
            );
        }

        let result_bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&result_bytes);
        let expected = gemv_geglu_cpu(&gate_weight, &up_weight, &input, m, n);

        let mut max_err = 0.0f32;
        for i in 0..m {
            let err = (result[i] - expected[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // f32 precision: should be near-exact (no quantization noise)
        assert!(
            max_err < 1e-4,
            "Max error {max_err} exceeds threshold for gemma2-like dims ({m}×{n})"
        );
    }
}
