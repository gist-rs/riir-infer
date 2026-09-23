//! Fused RMSNorm + ResidualAdd CubeCL kernel (Plan 106 Track 3 T2.10/T2.11).
//!
//! Single-dispatch kernel that computes `output[i] = rmsnorm(input)[i] + residual[i]`,
//! eliminating one GPU dispatch per use in the Gemma 2 decode path.
//!
//! # Gemma 2 Post-Norm Pattern
//!
//! Gemma 2 applies RMSNorm after each sub-layer output, then adds the residual:
//!
//! ```text
//! Post-attention:  normed = rmsnorm(wo_out, post_attn_norm)
//!                  hidden = normed + residual       ← separate dispatch
//!
//! Post-MLP:        normed = rmsnorm(down_out, post_mlp_norm)
//!                  hidden = normed + residual2      ← separate dispatch
//! ```
//!
//! This kernel fuses both into a single dispatch:
//!
//! ```text
//! hidden[i] = input[i] * inv_rms * gamma[i] + residual[i]
//! ```
//!
//! # Dispatch Savings
//!
//! | Location | Before | After | Savings |
//! |----------|--------|-------|---------|
//! | Post-attention | rmsnorm + residual_add (2) | fused (1) | 1 |
//! | Post-MLP | rmsnorm + residual_add (2) | fused (1) | 1 |
//! | **Per layer** | | | **2** |
//! | **26 layers** | | | **52** |
//!
//! # CubeCL v0.10 Constraints
//!
//! - Exactly 5 Array parameters (input, gamma, params, residual, output).
//! - No conditional expressions as values — use `if { }` statements.
//! - Unrolled parallel reductions (8 hardcoded steps), `sync_cube()` between steps.
//! - `UNIT_POS` is `u32`, `ABSOLUTE_POS` is `usize` — cast appropriately.
//! - `f32::new(literal)` for constants in `#[cube]` context.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Fused RMSNorm + ResidualAdd kernel
// ---------------------------------------------------------------------------

/// CubeCL fused RMSNorm + ResidualAdd kernel.
///
/// Computes `output[i] = input[i] * inv_rms * gamma[i] + residual[i]` where
/// `inv_rms = 1 / sqrt(mean(x²) + eps)`.
///
/// Uses a single workgroup of 256 threads with strided accumulation to handle
/// any dimension (including Gemma 2's n_embd=2048). Shared memory is used for
/// an unrolled parallel sum reduction of `x²` values.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; dim]` — input vector (e.g., Wo output or down output).
/// - `gamma`: `[f32; dim]` — learnable scale (with +1 offset pre-applied).
/// - `params`: `[f32; 2]` — `[inv_dim, eps]` precomputed on CPU.
/// - `residual`: `[f32; dim]` — residual connection (e.g., pre-attention hidden).
/// - `output`: `[f32; dim]` — normalized + residual output.
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rmsnorm_residual_f32(
    input: &[f32],
    gamma: &[f32],
    params: &[f32],
    residual: &[f32],
    output: &mut [f32],
) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = input.len() as u32;
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

    // ── Phase 4: Normalize, apply gamma, add residual ──
    // Fused: output[i] = input[i] * inv_rms * gamma[i] + residual[i]
    let mut j = tid;
    while j < dim {
        let x = input[j as usize];
        let g = gamma[j as usize];
        let r = residual[j as usize];
        output[j as usize] = x * inv_rms * g + r;
        j += cube_size;
    }
}

// ---------------------------------------------------------------------------
// Batched variant (Plan 482 T4) — one workgroup per row for [seq_len × dim]
// ---------------------------------------------------------------------------

/// CubeCL batched fused RMSNorm + ResidualAdd — processes `[seq_len × dim]`.
///
/// One workgroup per row (position). `CUBE_POS_X` selects the row.
///
/// ## Dispatch
///
/// `CubeCount::Static(seq_len, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn rmsnorm_residual_batched_f32(
    input: &[f32],
    gamma: &[f32],
    params: &[f32],
    residual: &[f32],
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

    // ── Phase 4: Normalize, apply gamma, add residual for this row ──
    let mut j = tid;
    while j < dim {
        let x = input[(row_offset + j) as usize];
        let g = gamma[j as usize];
        let r = residual[(row_offset + j) as usize];
        output[(row_offset + j) as usize] = x * inv_rms * g + r;
        j += cube_size;
    }
}

// ---------------------------------------------------------------------------
// Launcher struct
// ---------------------------------------------------------------------------

/// CubeCL fused RMSNorm + ResidualAdd launcher.
///
/// Wraps the `rmsnorm_residual_f32` kernel with precomputed parameters.
/// Single workgroup handles the full dimension via strided access.
///
/// Replaces two separate dispatches:
/// 1. `RmsNormCubeCL::launch(input, gamma, output, dim, eps)`
/// 2. `ResidualAddCubeCL::launch(output, residual, output, dim)`
///
/// With one fused dispatch:
/// `NormResidualCubeCL::launch(input, gamma, residual, output, dim, eps)`
#[cfg(feature = "cubecl_runtime")]
pub struct NormResidualCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl NormResidualCubeCL {
    /// Launch fused RMSNorm + ResidualAdd kernel.
    ///
    /// Computes `output[i] = input[i] * inv_rms * gamma[i] + residual[i]`
    /// where `inv_rms = 1 / sqrt(mean(input²) + eps)`.
    ///
    /// Dispatch: `(1, 1, 1)` workgroups of 256 threads (strided access).
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `input_handle`: `dim` f32 elements
    /// - `gamma_handle`: `dim` f32 elements
    /// - `residual_handle`: `dim` f32 elements
    /// - `output_handle`: `dim` f32 elements
    ///
    /// `dim` must be > 0. `eps` must be > 0.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        residual_handle: Handle,
        output_handle: Handle,
        dim: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        let params: &[f32] = &[inv_dim, eps];
        let params_handle = client.create_from_slice(f32::as_bytes(params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rmsnorm_residual_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, dim),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(params_handle, 2),
                BufferArg::from_raw_parts(residual_handle, dim),
                BufferArg::from_raw_parts(output_handle, dim),
            );
        }
    }
}

/// CubeCL batched fused RMSNorm + ResidualAdd launcher (Plan 482 T4).
///
/// Processes `[seq_len × dim]` in one dispatch — one workgroup per row.
#[cfg(feature = "cubecl_runtime")]
pub struct NormResidualBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in Plan 482 batched forward
impl NormResidualBatchedCubeCL {
    /// Launch batched fused RMSNorm + ResidualAdd.
    ///
    /// For each row `r`: `output[r, j] = input[r, j] * inv_rms_r * gamma[j] + residual[r, j]`
    /// where `inv_rms_r = 1 / sqrt(mean(input[r,:]²) + eps)`.
    ///
    /// Dispatch: `(seq_len, 1, 1)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `seq_len * dim` f32 elements
    /// - `gamma_handle`: `dim` f32 elements
    /// - `residual_handle`: `seq_len * dim` f32 elements
    /// - `output_handle`: `seq_len * dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        gamma_handle: Handle,
        residual_handle: Handle,
        output_handle: Handle,
        seq_len: usize,
        dim: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        let dim_f = dim as f32;
        let params: &[f32] = &[inv_dim, eps, dim_f];
        let params_handle = client.create_from_slice(f32::as_bytes(params));
        let total = seq_len * dim;

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            rmsnorm_residual_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(seq_len as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, total),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(residual_handle, total),
                BufferArg::from_raw_parts(output_handle, total),
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

    /// CPU reference: RMSNorm + ResidualAdd fused.
    fn rmsnorm_residual_cpu(
        input: &[f32],
        gamma: &[f32],
        residual: &[f32],
        dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let sum_sq: f32 = input[..dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        input[..dim]
            .iter()
            .zip(gamma[..dim].iter())
            .zip(residual[..dim].iter())
            .map(|((x, g), r)| x * inv_rms * g + r)
            .collect()
    }

    /// CPU reference: standalone RMSNorm.
    fn rmsnorm_cpu(input: &[f32], gamma: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        let sum_sq: f32 = input[..dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        input[..dim]
            .iter()
            .zip(gamma[..dim].iter())
            .map(|(x, g)| x * inv_rms * g)
            .collect()
    }

    fn max_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_norm_residual_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 256;
        let eps = 1e-6;

        // Input = all ones, gamma = all ones, residual = all zeros
        // Expected: rmsnorm(1s) = 1s (since mean(1²) = 1, inv_rms = 1), + 0 = 1s
        let input = vec![1.0f32; dim];
        let gamma = vec![1.0f32; dim];
        let residual = vec![0.0f32; dim];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                residual_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = vec![1.0f32; dim];
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_norm_residual_known_values() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 8;
        let eps = 1e-6;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let gamma = vec![1.0; dim]; // identity gamma
        let residual = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8f32];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                residual_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = rmsnorm_residual_cpu(&input, &gamma, &residual, dim, eps);
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");

        // Verify it matches separate rmsnorm + residual
        let normed = rmsnorm_cpu(&input, &gamma, dim, eps);
        let separate: Vec<f32> = normed
            .iter()
            .zip(residual.iter())
            .map(|(n, r)| n + r)
            .collect();
        let err2 = max_error(output, &separate);
        assert!(
            err2 < 1e-5,
            "max error vs separate {err2} exceeds tolerance"
        );
    }

    #[test]
    fn test_norm_residual_gamma_scaling() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 4;
        let eps = 1e-6;

        let input = vec![2.0, 4.0, 6.0, 8.0f32];
        let gamma = vec![2.0, 0.5, 3.0, 1.0f32];
        let residual = vec![1.0, 1.0, 1.0, 1.0f32];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                residual_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = rmsnorm_residual_cpu(&input, &gamma, &residual, dim, eps);
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_norm_residual_large_dim() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 2048; // Gemma 2 n_embd
        let eps = 1e-6;

        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| 1.0 + (i as f32 * 0.001).cos() * 0.1)
            .collect();
        let residual: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.005).cos() * 0.5).collect();

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                residual_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        let expected = rmsnorm_residual_cpu(&input, &gamma, &residual, dim, eps);
        let err = max_error(output, &expected);
        assert!(
            err < 1e-4,
            "max error {err} exceeds tolerance for dim={dim}"
        );
    }

    #[test]
    fn test_norm_residual_zero_input() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 16;
        let eps = 1e-6;

        // Zero input → rmsnorm gives 0 (since inv_rms = 1/sqrt(eps) is finite),
        // so output = 0 * gamma + residual = residual
        let input = vec![0.0f32; dim];
        let gamma = vec![2.0f32; dim];
        let residual: Vec<f32> = (0..dim).map(|i| i as f32).collect();

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                residual_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let output_bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&output_bytes);

        // With zero input: normed = 0 * inv_rms * gamma = 0, output = 0 + residual
        let expected = &residual;
        let err = max_error(output, expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_norm_residual_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 512;
        let eps = 1e-6;

        // Random-ish input
        let input: Vec<f32> = (0..dim).map(|i| ((i as f32 * 7.13) % 3.0) - 1.0).collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| 1.0 + (i as f32 * 0.01).sin() * 0.2)
            .collect();
        let residual: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let residual_handle = client.create_from_slice(f32::as_bytes(&residual));
        let fused_output_handle = client.empty(dim * 4);

        // Fused kernel
        unsafe {
            NormResidualCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle.clone(),
                gamma_handle.clone(),
                residual_handle.clone(),
                fused_output_handle.clone(),
                dim,
                eps,
            );
        }

        let fused_bytes = client.read_one(fused_output_handle).unwrap();
        let fused_output = f32::from_bytes(&fused_bytes);

        // Separate kernels: rmsnorm then residual_add
        let normed_handle = client.empty(dim * 4);
        unsafe {
            crate::norms_cubecl::RmsNormCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                normed_handle.clone(),
                dim,
                eps,
            );
        }

        let separate_output_handle = client.empty(dim * 4);
        unsafe {
            crate::norms_cubecl::ResidualAddCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                normed_handle,
                residual_handle,
                separate_output_handle.clone(),
                dim,
            );
        }

        let separate_bytes = client.read_one(separate_output_handle).unwrap();
        let separate_output = f32::from_bytes(&separate_bytes);

        // Fused should match separate to within floating-point precision
        let err = max_error(fused_output, separate_output);
        assert!(
            err < 1e-6,
            "fused vs separate max error {err} exceeds tolerance"
        );
    }
}
