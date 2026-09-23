//! Fused triple QKV GEMV CubeCL kernel with F16 weights.
//!
//! Single-dispatch kernel that computes all three Q/K/V projections:
//! 1. `Q[row] = dot(weight_q[row], input)` — GEMV with f16 weights
//! 2. `K[row] = dot(weight_k[row], input)` — GEMV with f16 weights
//! 3. `V[row] = dot(weight_v[row], input)` — GEMV with f16 weights
//!
//! This replaces 3 separate GEMV dispatches with 1, saving 2 dispatches per
//! attention layer × 26 layers = **52 dispatches** per decode token.
//!
//! # Dispatch Savings
//!
//! | Location    | Before       | After | Savings |
//! |-------------|--------------|-------|---------|
//! | Q/K/V GEMVs | 3 dispatches | 1     | 2       |
//! | **Per layer** |             |       | **2**   |
//! | **26 layers** |             |       | **52**  |
//!
//! # Weight Layout
//!
//! The combined weight buffer concatenates Q, K, V weight matrices contiguously:
//! ```text
//! weight_qkv = [Wq; q_dim × n | Wk; kv_dim × n | Wv; kv_dim × n]
//! ```
//!
//! The output buffer follows the same layout:
//! ```text
//! output = [Q; q_dim | K; kv_dim | V; kv_dim]
//! ```
//!
//! # Algorithm
//!
//! Each plane (subgroup) handles one output row:
//! - Determine which section (Q, K, or V) this row belongs to based on output index.
//! - Look up the correct weight offset into the combined weight buffer.
//! - Cooperative dot product with `plane_sum()` reduction.
//! - Lane 0 writes the result to the combined output buffer.
//!
//! # CubeCL v0.10 Constraints
//!
//! - Exactly 4 Array parameters: `weight_qkv`, `input`, `output`, `params`.
//! - Scalar parameters (q_dim, kv_dim, n) packed into a small `params` f32 array.
//! - Plane (subgroup) cooperative dot product with `plane_sum()` reduction.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use half::f16 as half_f16;

// ---------------------------------------------------------------------------
// Fused triple QKV GEMV: F16 weights, plane cooperative
// ---------------------------------------------------------------------------

/// CubeCL fused triple QKV GEMV kernel (F16 weights, plane cooperative).
///
/// Computes for each output row:
/// 1. Determine which section (Q, K, or V) the row belongs to.
/// 2. Look up the correct weight row offset in the combined weight buffer.
/// 3. `output[row] = dot(f16_weight_row, input)` — cooperative dot product.
///
/// All in a single GPU dispatch, eliminating 2 dispatch overheads vs separate Q/K/V GEMVs.
///
/// ## Parameter Layout
///
/// - `weight_qkv`: `[f16; (q_dim + 2*kv_dim) * n]` — combined Q|K|V weights (row-major).
/// - `input`: `[f32; n]` — input vector.
/// - `output`: `[f32; q_dim + 2*kv_dim]` — combined Q|K|V outputs.
/// - `params`: `[f32; 3]` — `[q_dim as f32, kv_dim as f32, n as f32]`.
///
/// ## Dispatch
///
/// `CubeCount::Static(total_rows, 1, 1)`, `CubeDim::new_1d(plane_size)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_qkv_plane_f16(
    weight_qkv: &[half_f16],
    input: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let q_dim = params[0usize] as u32;
    let kv_dim = params[1usize] as u32;
    let n = params[2usize] as u32;

    let total_rows = q_dim + 2u32 * kv_dim;
    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= total_rows {
        terminate!();
    }

    // Determine which section (Q/K/V) and the weight offset.
    // Weight buffer layout: [Wq; q_dim*n | Wk; kv_dim*n | Wv; kv_dim*n]
    let weight_offset = if row < q_dim {
        // Q section: rows [0, q_dim) → weight offset = row * n
        row * n
    } else if row < q_dim + kv_dim {
        // K section: rows [q_dim, q_dim+kv_dim) → weight offset = q_dim*n + (row-q_dim)*n
        q_dim * n + (row - q_dim) * n
    } else {
        // V section: rows [q_dim+kv_dim, q_dim+2*kv_dim) → weight offset = (q_dim+kv_dim)*n + (row-q_dim-kv_dim)*n
        (q_dim + kv_dim) * n + (row - q_dim - kv_dim) * n
    };

    // Cooperative dot product: lane j handles elements j, j + PLANE_DIM, ...
    let mut partial = f32::new(0.0f32);
    let mut k = lane;
    while k < n {
        let w = f32::cast_from(weight_qkv[(weight_offset + k) as usize]);
        partial += w * input[k as usize];
        k += PLANE_DIM;
    }

    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row.
    if lane == 0u32 {
        output[row as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Launcher struct
// ---------------------------------------------------------------------------

/// CubeCL fused triple QKV GEMV launcher (F16 weights).
///
/// Replaces three separate GEMV dispatches:
/// 1. `GemvF16CubeCL::launch(weight_q, input, q_output, q_dim, n)`
/// 2. `GemvF16CubeCL::launch(weight_k, input, k_output, kv_dim, n)`
/// 3. `GemvF16CubeCL::launch(weight_v, input, v_output, kv_dim, n)`
///
/// With one fused dispatch:
/// `GemvQkvF16CubeCL::launch(weight_qkv, input, output, params, q_dim, kv_dim, n)`
///
/// The caller must concatenate Q, K, V weight matrices into a single buffer:
/// ```text
/// weight_qkv = [Wq; q_dim*n | Wk; kv_dim*n | Wv; kv_dim*n]
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct GemvQkvF16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvQkvF16CubeCL {
    /// Launch fused triple QKV GEMV kernel (F16 weights).
    ///
    /// Computes `output[row] = dot(weight_qkv_row, input)` for each row across
    /// all three Q/K/V sections in a single GPU dispatch.
    ///
    /// Dispatch: `total_rows` workgroups of `planeSize` threads,
    /// where `total_rows = q_dim + 2 * kv_dim`.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `weight_qkv_handle`: `(q_dim + 2 * kv_dim) * n` f16 elements
    /// - `input_handle`: `n` f32 elements
    /// - `output_handle`: `q_dim + 2 * kv_dim` f32 elements
    /// - `params_handle`: 3 f32 elements `[q_dim as f32, kv_dim as f32, n as f32]`
    ///
    /// The client must support `Plane::Ops` (subgroup operations).
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        weight_qkv_handle: Handle,
        input_handle: Handle,
        output_handle: Handle,
        params_handle: Handle,
        q_dim: usize,
        kv_dim: usize,
        n: usize,
    ) {
        let plane_size = 32u32; // Metal simdgroup size
        let total_rows = q_dim + 2 * kv_dim;
        let total_weight = total_rows * n;

        // SAFETY: Caller guarantees correct buffer sizes and plane support.
        // The params_handle must contain 3 f32 elements: [q_dim, kv_dim, n].
        unsafe {
            gemv_qkv_plane_f16::launch_unchecked::<R>(
                client,
                CubeCount::Static(total_rows as u32, 1, 1),
                CubeDim::new_1d(plane_size),
                BufferArg::from_raw_parts(weight_qkv_handle, total_weight),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, total_rows),
                BufferArg::from_raw_parts(params_handle, 3),
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

    /// CPU reference: triple QKV GEMV.
    fn gemv_qkv_cpu(
        weight_q: &[f32],
        weight_k: &[f32],
        weight_v: &[f32],
        input: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n: usize,
    ) -> Vec<f32> {
        let total_rows = q_dim + 2 * kv_dim;
        let mut output = vec![0.0f32; total_rows];
        for row in 0..q_dim {
            for col in 0..n {
                output[row] += weight_q[row * n + col] * input[col];
            }
        }
        for row in 0..kv_dim {
            for col in 0..n {
                output[q_dim + row] += weight_k[row * n + col] * input[col];
            }
        }
        for row in 0..kv_dim {
            for col in 0..n {
                output[q_dim + kv_dim + row] += weight_v[row * n + col] * input[col];
            }
        }
        output
    }

    fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
        use rayon::prelude::*;
        let f16: Vec<half_f16> = data.par_iter().map(|&x| half_f16::from_f32(x)).collect();
        bytemuck::cast_slice::<half_f16, u8>(&f16).to_vec()
    }

    /// Concatenate Q, K, V weight matrices into a single buffer.
    fn concat_qkv_weights(weight_q: &[f32], weight_k: &[f32], weight_v: &[f32]) -> Vec<f32> {
        let mut combined = Vec::with_capacity(weight_q.len() + weight_k.len() + weight_v.len());
        combined.extend_from_slice(weight_q);
        combined.extend_from_slice(weight_k);
        combined.extend_from_slice(weight_v);
        combined
    }

    fn launch_and_verify(
        weight_q: &[f32],
        weight_k: &[f32],
        weight_v: &[f32],
        input: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n: usize,
        tolerance: f32,
        test_name: &str,
    ) {
        let ctx = CubeCLContext::new().expect("CubeCL init");
        let client = ctx.client();

        let total_rows = q_dim + 2 * kv_dim;
        let combined_weights = concat_qkv_weights(weight_q, weight_k, weight_v);
        let weight_bytes = f32_to_f16_bytes(&combined_weights);
        let params: Vec<f32> = vec![q_dim as f32, kv_dim as f32, n as f32];

        let weight_handle = client.create_from_slice(&weight_bytes);
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(total_rows * core::mem::size_of::<f32>());
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        unsafe {
            GemvQkvF16CubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                weight_handle,
                input_handle,
                output_handle.clone(),
                params_handle,
                q_dim,
                kv_dim,
                n,
            );
        }

        let result_bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&result_bytes);
        let expected = gemv_qkv_cpu(weight_q, weight_k, weight_v, input, q_dim, kv_dim, n);

        let mut max_err = 0.0f32;
        for i in 0..total_rows {
            let err = (result[i] - expected[i]).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < tolerance,
                "{test_name}: Element {i}: expected {}, got {} (diff {})",
                expected[i],
                result[i],
                err
            );
        }
        // Also report max error for debugging
        eprintln!("{test_name}: max_err = {max_err:.6} (tolerance = {tolerance})");
    }

    #[test]
    fn test_gemv_qkv_f16_identity() {
        // Identity weight matrices for Q (2×2), K (2×2), V (2×2).
        // Input = [1, 1] → each dot product = 1+1 = 2.
        let weight_q = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let weight_k = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let weight_v = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let input = vec![1.0f32, 1.0];

        // Expected: [2, 2, 2, 2, 2, 2]
        launch_and_verify(
            &weight_q, &weight_k, &weight_v, &input, 2, 2, 2, 0.01, "identity",
        );
    }

    #[test]
    fn test_gemv_qkv_f16_general() {
        // 2×3 matrices for Q, K, V with distinct values.
        let weight_q = vec![
            1.0f32, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let weight_k = vec![
            0.1f32, 0.2, 0.3, // row 0
            0.4, 0.5, 0.6, // row 1
        ];
        let weight_v = vec![
            -1.0f32, 0.0, 1.0, // row 0
            1.0, -1.0, 0.0, // row 1
        ];
        let input = vec![1.0f32, -1.0, 2.0];

        // Q: row0 = 1*1 + 2*(-1) + 3*2 = 5, row1 = 4*1 + 5*(-1) + 6*2 = 11
        // K: row0 = 0.1*1 + 0.2*(-1) + 0.3*2 = 0.5, row1 = 0.4 + (-0.5) + 1.2 = 1.1
        // V: row0 = -1*1 + 0*(-1) + 1*2 = 1, row1 = 1*1 + (-1)*(-1) + 0*2 = 2
        // Expected: [5, 11, 0.5, 1.1, 1, 2]
        launch_and_verify(
            &weight_q, &weight_k, &weight_v, &input, 2, 2, 3, 0.01, "general",
        );
    }

    #[test]
    fn test_gemv_qkv_f16_gemma2_dims() {
        // Realistic Gemma 2 dimensions scaled down: q_dim=8, kv_dim=4, n=8.
        // In real Gemma 2: q_dim=2048, kv_dim=512, n=2048 (with GQA).
        let q_dim = 8usize;
        let kv_dim = 4usize;
        let n = 8usize;

        let weight_q: Vec<f32> = (0..q_dim * n)
            .map(|i| ((i % 97) as f32 - 48.0) / 100.0)
            .collect();
        let weight_k: Vec<f32> = (0..kv_dim * n)
            .map(|i| ((i % 53) as f32 - 26.0) / 100.0)
            .collect();
        let weight_v: Vec<f32> = (0..kv_dim * n)
            .map(|i| ((i % 71) as f32 - 35.0) / 100.0)
            .collect();
        let input: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();

        launch_and_verify(
            &weight_q,
            &weight_k,
            &weight_v,
            &input,
            q_dim,
            kv_dim,
            n,
            0.1,
            "gemma2_dims",
        );
    }
}
