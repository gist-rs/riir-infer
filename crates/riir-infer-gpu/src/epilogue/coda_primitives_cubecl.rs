//! CODA epilogue primitives — building blocks for delayed RMSNorm reparameterization (Plan 106 T2.11).
//!
//! These kernels implement the 4 remaining CODA epilogue visitors needed for
//! full GEMM-epilogue fusion in the Gemma 2 decode path. Together with the
//! already-implemented `NormResidualCubeCL` and `GemvResidualCubeCL`, they
//! enable the CODA target of 14 dispatches per layer (26.3% reduction vs baseline 19).
//!
//! # CODA Reparameterization
//!
//! The core CODA identity (§3.2.1):
//! ```text
//! RMSNorm(x@W + z) * γ @ W' = r * ((x@W + z) * γ) @ W'
//! ```
//!
//! This allows delaying the row-wise RMSNorm scale `r` past the next GEMM,
//! enabling multi-operation fusion in a single kernel dispatch.
//!
//! # Kernels
//!
//! 1. **PartialRMS** — computes `r[m] = 1/sqrt(mean(hidden[m,:]²) + eps)` for each row
//!    (block-wise parallel reduction, stores per-row scale factors)
//! 2. **NormWeightScale** — `output[n] = data[n] * gamma[n]` (element-wise broadcast)
//! 3. **RowScale** — `output[row] = data[row] * scale[row]` (delayed RMSNorm)
//! 4. **SwiGLU** — `output[i] = gate[i] * sigmoid(gate[i]) * up[i]` (activation)
//!
//! # Integration Note
//!
//! These kernels are building blocks for a future wiring pass (T2.12-T2.14).
//! They don't change the forward pass until explicitly wired in. The design
//! decision about norm ordering (pre-norm vs post-norm) is deferred to
//! the wiring phase.
// WIP: CODA epilogue primitives (Plan 106 T2.11) await epilogue-visitor wiring.
#![allow(dead_code)]
#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// 1. Partial RMS: compute per-row RMS scale factors
// ---------------------------------------------------------------------------

/// CubeCL kernel: compute RMS scale factors for a 1D vector.
///
/// Computes `output[0] = 1 / sqrt(mean(input²) + eps)` — a single scalar
/// representing the inverse RMS of the input vector. Used as the first
/// phase of CODA delayed RMSNorm: extract the scale, then apply it later
/// fused with a GEMV.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; dim]` — input vector.
/// - `params`: `[f32; 2]` — `[inv_dim, eps]`.
/// - `output`: `[f32; 1]` — single scalar: `1/sqrt(mean(x²) + eps)`.
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn partial_rms_f32(input: &[f32], params: &[f32], output: &mut [f32]) {
    let inv_dim = params[0usize];
    let eps = params[1usize];
    let dim = input.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // Phase 1: Strided accumulation of x²
    let mut partial_sq = f32::new(0.0f32);
    let mut i = tid;
    while i < dim {
        let x = input[i as usize];
        partial_sq += x * x;
        i += cube_size;
    }

    // Phase 2: Shared memory parallel reduction (128→64→...→1)
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

    // Phase 3: Thread 0 writes the scale factor
    if tid < 1u32 {
        let mean_sq = smem[0usize] * inv_dim;
        output[0usize] = f32::new(1.0f32) / (mean_sq + eps).sqrt();
    }
}

/// Launcher for partial RMS scale factor computation.
///
/// Returns a single-element handle containing `1/sqrt(mean(input²) + eps)`.
#[cfg(feature = "cubecl_runtime")]
pub struct PartialRmsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl PartialRmsCubeCL {
    /// Compute `output[0] = 1/sqrt(mean(input²) + eps)`.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `dim` f32 elements
    /// - `output_handle`: at least 1 f32 element
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
        dim: usize,
        eps: f32,
    ) {
        let inv_dim = 1.0f32 / dim as f32;
        let params: &[f32] = &[inv_dim, eps];
        let params_handle = client.create_from_slice(f32::as_bytes(params));

        unsafe {
            partial_rms_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, dim),
                BufferArg::from_raw_parts(params_handle, 2),
                BufferArg::from_raw_parts(output_handle, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. NormWeightScale: element-wise gamma broadcast multiply
// ---------------------------------------------------------------------------

/// CubeCL kernel: apply gamma scaling `output[i] = data[i] * gamma[i]`.
///
/// In CODA, this replaces the full RMSNorm when `rms_scale` is delayed.
/// The gamma vector has +1 offset pre-applied (Gemma 2 convention).
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn norm_weight_scale_f32(data: &[f32], gamma: &[f32], output: &mut [f32]) {
    let dim = data.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    let mut i = tid;
    while i < dim {
        output[i as usize] = data[i as usize] * gamma[i as usize];
        i += cube_size;
    }
}

/// Launcher for gamma broadcast multiply.
#[cfg(feature = "cubecl_runtime")]
pub struct NormWeightScaleCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl NormWeightScaleCubeCL {
    /// Compute `output[i] = data[i] * gamma[i]`.
    ///
    /// # Safety
    ///
    /// All handles must have `dim` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        data_handle: Handle,
        gamma_handle: Handle,
        output_handle: Handle,
        dim: usize,
    ) {
        unsafe {
            norm_weight_scale_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(data_handle, dim),
                BufferArg::from_raw_parts(gamma_handle, dim),
                BufferArg::from_raw_parts(output_handle, dim),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. RowScale: delayed RMSNorm broadcast (per-element scale by a scalar)
// ---------------------------------------------------------------------------

/// CubeCL kernel: apply delayed RMS scale `output[i] = data[i] * scale[0]`.
///
/// In CODA, the RMS scale factor `r` computed by `PartialRmsCubeCL` is
/// applied here, fused with or after a GEMV. For decode (batch_size=1),
/// this is a simple scalar broadcast multiply.
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn row_scale_f32(data: &[f32], scale: &[f32], output: &mut [f32]) {
    let dim = data.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let r = scale[0usize];

    let mut i = tid;
    while i < dim {
        output[i as usize] = data[i as usize] * r;
        i += cube_size;
    }
}

/// Launcher for delayed RMS scale application.
#[cfg(feature = "cubecl_runtime")]
pub struct RowScaleCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl RowScaleCubeCL {
    /// Compute `output[i] = data[i] * scale[0]`.
    ///
    /// # Safety
    ///
    /// - `data_handle`: `dim` f32 elements
    /// - `scale_handle`: at least 1 f32 element
    /// - `output_handle`: `dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        data_handle: Handle,
        scale_handle: Handle,
        output_handle: Handle,
        dim: usize,
    ) {
        unsafe {
            row_scale_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(data_handle, dim),
                BufferArg::from_raw_parts(scale_handle, 1),
                BufferArg::from_raw_parts(output_handle, dim),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. SwiGLU: gate * sigmoid(gate) * up activation
// ---------------------------------------------------------------------------

/// CubeCL kernel: SwiGLU activation `output[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
///
/// SwiGLU is the activation used in Gemma 2's MLP block. In the non-fused
/// path, gate and up projections are separate GEMVs followed by a separate
/// GeGLU dispatch. This kernel is the epilogue version that can be fused
/// with the gate+up GEMV.
///
/// Note: We use `x * sigmoid(x)` (SiLU/Swish) rather than `x * tanh_approx(x)`
/// to match the mathematical definition. The existing `dispatch_geglu_gpu`
/// uses `gelu_tanh` approximation; this kernel provides the sigmoid variant
/// for CODA fusion where precision semantics may differ.
///
/// # Wiring-pass requirement (Issue 674)
///
/// SwiGLU (SiLU) and GeGLU (tanh-approx GELU) are DIFFERENT activations, not
/// approximations of each other. The DispatchBudget's Gemma-2 MLP section
/// lists a GeGLU (tanh-approx) step — if this SiLU kernel is ever wired as
/// that step's replacement, model outputs diverge from the reference. The
/// wiring pass must select the activation variant per model config, not
/// reuse this kernel unconditionally.
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let dim = gate.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    let mut i = tid;
    while i < dim {
        let g = gate[i as usize];
        // sigmoid(x) = 1 / (1 + exp(-x))
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + (-g).exp());
        output[i as usize] = g * sig * up[i as usize];
        i += cube_size;
    }
}

/// Launcher for SwiGLU activation.
#[cfg(feature = "cubecl_runtime")]
pub struct SwigluCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SwigluCubeCL {
    /// Compute `output[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
    ///
    /// # Safety
    ///
    /// - `gate_handle`: `dim` f32 elements
    /// - `up_handle`: `dim` f32 elements
    /// - `output_handle`: `dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        gate_handle: Handle,
        up_handle: Handle,
        output_handle: Handle,
        dim: usize,
    ) {
        unsafe {
            swiglu_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(gate_handle, dim),
                BufferArg::from_raw_parts(up_handle, dim),
                BufferArg::from_raw_parts(output_handle, dim),
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

    fn max_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // ── PartialRMS tests ──────────────────────────────────────────────

    #[test]
    fn test_partial_rms_ones() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 256;
        let eps = 1e-6;

        // All ones: mean(1²) = 1.0, inv_rms = 1/sqrt(1 + eps) ≈ 1.0
        let input = vec![1.0f32; dim];
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(4);

        unsafe {
            PartialRmsCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&bytes);
        let expected = 1.0 / (1.0 + eps).sqrt();
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "inv_rms = {}, expected {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_partial_rms_known_values() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 8;
        let eps = 1e-6;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let sum_sq: f32 = input.iter().map(|v| v * v).sum();
        let expected = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(4);

        unsafe {
            PartialRmsCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&bytes);
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "inv_rms = {}, expected {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_partial_rms_large_dim() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 2048; // Gemma 2 n_embd
        let eps = 1e-6;

        let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let sum_sq: f32 = input.iter().map(|v| v * v).sum();
        let expected = 1.0 / (sum_sq / dim as f32 + eps).sqrt();

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = client.empty(4);

        unsafe {
            PartialRmsCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                dim,
                eps,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&bytes);
        assert!(
            (result[0] - expected).abs() < 1e-4,
            "inv_rms = {}, expected {}",
            result[0],
            expected
        );
    }

    // ── NormWeightScale tests ─────────────────────────────────────────

    #[test]
    fn test_norm_weight_scale_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 64;

        let data = vec![2.0f32; dim];
        let gamma = vec![1.0f32; dim];
        let data_handle = client.create_from_slice(f32::as_bytes(&data));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormWeightScaleCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                data_handle,
                gamma_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &data);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_norm_weight_scale_known() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 4;

        let data = vec![1.0, 2.0, 3.0, 4.0f32];
        let gamma = vec![2.0, 0.5, 3.0, 1.0f32];
        let expected: Vec<f32> = data.iter().zip(gamma.iter()).map(|(d, g)| d * g).collect();

        let data_handle = client.create_from_slice(f32::as_bytes(&data));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));
        let output_handle = client.empty(dim * 4);

        unsafe {
            NormWeightScaleCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                data_handle,
                gamma_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    // ── RowScale tests ────────────────────────────────────────────────

    #[test]
    fn test_row_scale_identity() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 64;

        let data = vec![3.0f32; dim];
        let scale = vec![1.0f32];
        let data_handle = client.create_from_slice(f32::as_bytes(&data));
        let scale_handle = client.create_from_slice(f32::as_bytes(&scale));
        let output_handle = client.empty(dim * 4);

        unsafe {
            RowScaleCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                data_handle,
                scale_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &data);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_row_scale_half() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 8;

        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0f32];
        let scale = vec![0.5f32];
        let expected: Vec<f32> = data.iter().map(|d| d * 0.5).collect();

        let data_handle = client.create_from_slice(f32::as_bytes(&data));
        let scale_handle = client.create_from_slice(f32::as_bytes(&scale));
        let output_handle = client.empty(dim * 4);

        unsafe {
            RowScaleCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                data_handle,
                scale_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    // ── SwiGLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_swiglu_zeros() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 16;

        // sigmoid(0) = 0.5, so output = 0 * 0.5 * up = 0
        let gate = vec![0.0f32; dim];
        let up = vec![1.0f32; dim];
        let expected = vec![0.0f32; dim];

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let up_handle = client.create_from_slice(f32::as_bytes(&up));
        let output_handle = client.empty(dim * 4);

        unsafe {
            SwigluCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_swiglu_known_values() {
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 4;

        let gate = vec![1.0, 2.0, -1.0, 0.5f32];
        let up = vec![1.0, 2.0, 3.0, 4.0f32];

        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * sigmoid(*g) * u)
            .collect();

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let up_handle = client.create_from_slice(f32::as_bytes(&up));
        let output_handle = client.empty(dim * 4);

        unsafe {
            SwigluCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &expected);
        assert!(err < 1e-5, "max error {err} exceeds tolerance");
    }

    #[test]
    fn test_swiglu_large() {
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 2048;

        let gate: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let up: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * sigmoid(*g) * u)
            .collect();

        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let up_handle = client.create_from_slice(f32::as_bytes(&up));
        let output_handle = client.empty(dim * 4);

        unsafe {
            SwigluCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &expected);
        assert!(
            err < 1e-4,
            "max error {err} exceeds tolerance for dim={dim}"
        );
    }

    // ── End-to-end CODA decomposition test ────────────────────────────

    #[test]
    fn test_coda_decomposition_matches_rmsnorm() {
        // Verify: rmsnorm(x, gamma) == row_scale(norm_weight_scale(x, gamma), partial_rms(x))
        // i.e., x * gamma * inv_rms == (x * gamma) * inv_rms
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let dim = 512;
        let eps = 1e-6;

        let input: Vec<f32> = (0..dim).map(|i| ((i as f32 * 7.13) % 3.0) - 1.0).collect();
        let gamma: Vec<f32> = (0..dim)
            .map(|i| 1.0 + (i as f32 * 0.01).sin() * 0.2)
            .collect();

        // CPU reference: rmsnorm(input, gamma)
        let sum_sq: f32 = input.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        let expected: Vec<f32> = input
            .iter()
            .zip(gamma.iter())
            .map(|(x, g)| x * inv_rms * g)
            .collect();

        // GPU: CODA decomposition
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let gamma_handle = client.create_from_slice(f32::as_bytes(&gamma));

        // Step 1: partial_rms → scale factor
        let scale_handle = client.empty(4);
        unsafe {
            PartialRmsCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle.clone(),
                scale_handle.clone(),
                dim,
                eps,
            );
        }

        // Step 2: norm_weight_scale → x * gamma
        let scaled_handle = client.empty(dim * 4);
        unsafe {
            NormWeightScaleCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                gamma_handle,
                scaled_handle.clone(),
                dim,
            );
        }

        // Step 3: row_scale → (x * gamma) * inv_rms
        let output_handle = client.empty(dim * 4);
        unsafe {
            RowScaleCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                scaled_handle,
                scale_handle,
                output_handle.clone(),
                dim,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        let output = f32::from_bytes(&bytes);
        let err = max_error(output, &expected);
        assert!(
            err < 1e-4,
            "CODA decomposition max error {err} exceeds tolerance"
        );
    }
}
