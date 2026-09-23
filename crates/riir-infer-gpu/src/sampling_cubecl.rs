//! CubeCL GPU sampling kernels for decode fusion (Plan 171 T3).
//!
//! GPU-resident argmax and temperature-scaled argmax to avoid downloading
//! full logits (~1 MB) per decode token. Only 4 bytes (token ID) are
//! transferred back to CPU.
//!
//! # Kernels
//!
//! | Kernel | Algorithm | Download |
//! |--------|-----------|----------|
//! | `argmax_f32` | Parallel reduction argmax → u32 index | 4 bytes |
//! | `temperature_argmax_f32` | Scale by 1/T, then argmax | 4 bytes |
//!
//! # Key Insight
//!
//! Gemma 2 logit softcapping (`tanh(x/cap) * cap`) is a monotonically
//! increasing function, so it does NOT change which logit is maximum.
//! Therefore argmax is valid without softcap — no GPU softcap kernel needed.

// WIP: GPU sampling kernels (Plan 171 T3) await decode-fusion wiring.
#![allow(dead_code)]
#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Argmax kernel
// ---------------------------------------------------------------------------

/// GPU argmax kernel: finds the index of the maximum element in a 1D f32 array.
///
/// Returns a single `u32` value — the index of the maximum element.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; vocab_size]` — logits from lm_head.
/// - `output`: `[u32; 1]` — index of maximum logit.
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn argmax_f32(input: &[f32], output: &mut [u32]) {
    let dim = input.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // Phase 1: Strided accumulation — each thread tracks its local max + index
    // Use -1e30, NOT NEG_INFINITY: WGSL has no infinity literal, so naga fails
    // to emit MSL for this shader and `create_shader_module_passthrough` aborts
    // with "Failed to generate the backend-specific code". The kernel then never
    // runs and `output` silently keeps whatever was already in the buffer —
    // which reads as 0 from a fresh allocation, i.e. a plausible-looking answer.
    // Same convention as `elementwise_cubecl.rs`.
    let mut local_max = f32::new(-1e30f32);
    let mut local_idx = u32::new(0i64);
    let mut i = tid;
    while i < dim {
        let val = input[i as usize];
        if val > local_max {
            local_max = val;
            local_idx = i;
        }
        i += cube_size;
    }

    // Phase 2: Shared memory parallel reduction for max + index
    let mut smem_val = Shared::<[f32]>::new_slice(256usize);
    let mut smem_idx = Shared::<[u32]>::new_slice(256usize);

    smem_val[tid as usize] = local_max;
    smem_idx[tid as usize] = local_idx;
    sync_cube();

    // Unrolled parallel max reduction (128→64→32→16→8→4→2→1)
    if tid < 128u32 {
        let other = smem_val[(tid + 128u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 128u32) as usize];
        }
    }
    sync_cube();
    if tid < 64u32 {
        let other = smem_val[(tid + 64u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 64u32) as usize];
        }
    }
    sync_cube();
    if tid < 32u32 {
        let other = smem_val[(tid + 32u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 32u32) as usize];
        }
    }
    sync_cube();
    if tid < 16u32 {
        let other = smem_val[(tid + 16u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 16u32) as usize];
        }
    }
    sync_cube();
    if tid < 8u32 {
        let other = smem_val[(tid + 8u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 8u32) as usize];
        }
    }
    sync_cube();
    if tid < 4u32 {
        let other = smem_val[(tid + 4u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 4u32) as usize];
        }
    }
    sync_cube();
    if tid < 2u32 {
        let other = smem_val[(tid + 2u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 2u32) as usize];
        }
    }
    sync_cube();
    if tid < 1u32 {
        let other = smem_val[1usize];
        if other > smem_val[0usize] {
            smem_val[0usize] = other;
            smem_idx[0usize] = smem_idx[1usize];
        }
        output[0usize] = smem_idx[0usize];
    }
}

/// Launcher for the GPU argmax kernel.
///
/// Finds the index of the maximum value in `input` and writes it to `output`.
///
/// # Safety
///
/// - `input` must contain at least `vocab_size` f32 elements.
/// - `output` must be pre-allocated with at least 1 u32 element (4 bytes).
///
/// Visibility note: `pub` (not `pub(crate)`) because a REMAINING
/// engine-gpu module (`gemma2_cubecl`, the S5 hinge) consumes this type
/// through the module re-export — cross-crate visibility is the one
/// forced divergence from the source file (riir-gpu's copy keeps
/// `pub(crate)` until that module moves).
#[cfg(feature = "cubecl_runtime")]
pub struct ArgmaxCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ArgmaxCubeCL {
    /// Launch argmax kernel on GPU.
    ///
    /// # Safety
    ///
    /// Caller must ensure `input` contains `vocab_size` f32 elements and
    /// `output` is pre-allocated with 4 bytes (1 u32).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input: Handle,
        output: Handle,
        vocab_size: usize,
    ) {
        unsafe {
            argmax_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input, vocab_size),
                BufferArg::from_raw_parts(output, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Temperature Argmax kernel
// ---------------------------------------------------------------------------

/// GPU temperature-scaled argmax kernel.
///
/// Divides logits by temperature before finding argmax. Equivalent to
/// temperature sampling with T→0 (greedy). At temperature=1.0, behaves
/// identically to plain argmax.
///
/// Note: For proper stochastic sampling with temperature, a full softmax
/// + random sampling is needed. This kernel is for greedy (T=0) decoding
/// where temperature only matters for softcap-equivalent scaling.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; vocab_size]` — logits from lm_head.
/// - `output`: `[u32; 1]` — index of maximum (scaled) logit.
/// - `params`: `[f32; 1]` — `[inv_temperature]` (1.0 / temperature).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn temperature_argmax_f32(input: &[f32], output: &mut [u32], params: &[f32]) {
    let inv_temp = params[0usize];
    let dim = input.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // Phase 1: Strided accumulation with temperature scaling
    // -1e30 not NEG_INFINITY — see the note on `argmax_f32` above.
    let mut local_max = f32::new(-1e30f32);
    let mut local_idx = u32::new(0i64);
    let mut i = tid;
    while i < dim {
        let val = input[i as usize] * inv_temp;
        if val > local_max {
            local_max = val;
            local_idx = i;
        }
        i += cube_size;
    }

    // Phase 2: Shared memory parallel reduction for max + index
    let mut smem_val = Shared::<[f32]>::new_slice(256usize);
    let mut smem_idx = Shared::<[u32]>::new_slice(256usize);

    smem_val[tid as usize] = local_max;
    smem_idx[tid as usize] = local_idx;
    sync_cube();

    // Unrolled parallel max reduction
    if tid < 128u32 {
        let other = smem_val[(tid + 128u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 128u32) as usize];
        }
    }
    sync_cube();
    if tid < 64u32 {
        let other = smem_val[(tid + 64u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 64u32) as usize];
        }
    }
    sync_cube();
    if tid < 32u32 {
        let other = smem_val[(tid + 32u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 32u32) as usize];
        }
    }
    sync_cube();
    if tid < 16u32 {
        let other = smem_val[(tid + 16u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 16u32) as usize];
        }
    }
    sync_cube();
    if tid < 8u32 {
        let other = smem_val[(tid + 8u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 8u32) as usize];
        }
    }
    sync_cube();
    if tid < 4u32 {
        let other = smem_val[(tid + 4u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 4u32) as usize];
        }
    }
    sync_cube();
    if tid < 2u32 {
        let other = smem_val[(tid + 2u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 2u32) as usize];
        }
    }
    sync_cube();
    if tid < 1u32 {
        let other = smem_val[1usize];
        if other > smem_val[0usize] {
            smem_val[0usize] = other;
            smem_idx[0usize] = smem_idx[1usize];
        }
        output[0usize] = smem_idx[0usize];
    }
}

/// Launcher for the GPU temperature-scaled argmax kernel.
///
/// Divides logits by temperature before argmax. For greedy decoding (T=0),
/// use `ArgmaxCubeCL` instead (no scaling overhead).
#[cfg(feature = "cubecl_runtime")]
pub(crate) struct TemperatureArgmaxCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TemperatureArgmaxCubeCL {
    /// Launch temperature-scaled argmax kernel on GPU.
    ///
    /// # Safety
    ///
    /// Caller must ensure `input` contains `vocab_size` f32 elements,
    /// `output` is pre-allocated with 4 bytes (1 u32), and
    /// `params` contains 1 f32 element (inv_temperature).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input: Handle,
        output: Handle,
        params: Handle,
        vocab_size: usize,
    ) {
        unsafe {
            temperature_argmax_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input, vocab_size),
                BufferArg::from_raw_parts(output, 1),
                BufferArg::from_raw_parts(params, 1),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl_runtime::CubeCLContext;

    /// Poison value pre-written into every argmax output buffer.
    ///
    /// The kernels used to fail shader compilation on Metal, so `output` was
    /// never written and simply kept whatever the buffer already held. With
    /// `client.empty()` that is usually 0 — which is the *correct* answer for
    /// `test_argmax_single_element` and `test_argmax_all_equal`, so those two
    /// passed on a kernel that never ran. Seeding a poison value makes
    /// "did not run" impossible to confuse with "returned 0".
    const ARGMAX_SENTINEL: u32 = 0xDEAD_BEEF;

    /// Output handle pre-filled with [`ARGMAX_SENTINEL`].
    fn sentinel_out(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
    ) -> Handle {
        client.create_from_slice(u32::as_bytes(&[ARGMAX_SENTINEL]))
    }

    /// Decode an argmax result, failing loudly if the kernel never wrote.
    fn read_argmax(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        output: Handle,
    ) -> u32 {
        let bytes = client.read_one(output).expect("should read output");
        let v = u32::from_bytes(&bytes)[0];
        assert_ne!(
            v, ARGMAX_SENTINEL,
            "argmax kernel never wrote its output — shader failed to compile? \
             (check for a wgpu 'Failed to generate the backend-specific code' \
             validation error above)"
        );
        v
    }

    #[test]
    fn test_argmax_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = sentinel_out(&client);

        unsafe {
            ArgmaxCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                input.len(),
            );
        }

        let result = [read_argmax(&client, output_handle)];
        assert_eq!(result[0], 1u32, "argmax should find index 1 (value 5.0)");
    }

    #[test]
    fn test_argmax_single_element() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![42.0];
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = sentinel_out(&client);

        unsafe {
            ArgmaxCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                input.len(),
            );
        }

        let result = [read_argmax(&client, output_handle)];
        assert_eq!(
            result[0], 0u32,
            "argmax of single element should be index 0"
        );
    }

    #[test]
    fn test_argmax_all_equal() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![3.0, 3.0, 3.0, 3.0];
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = sentinel_out(&client);

        unsafe {
            ArgmaxCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                input.len(),
            );
        }

        let result = [read_argmax(&client, output_handle)];
        // All equal — first index wins (no subsequent value is strictly greater)
        assert_eq!(result[0], 0u32);
    }

    #[test]
    fn test_argmax_large_vocab() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Simulate vocab_size=256128 (Gemma 2 2B)
        let mut input = vec![0.1f32; 256128];
        input[100000] = 99.0; // max at index 100000
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let output_handle = sentinel_out(&client);

        unsafe {
            ArgmaxCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                output_handle.clone(),
                input.len(),
            );
        }

        let result = [read_argmax(&client, output_handle)];
        assert_eq!(result[0], 100000u32, "argmax should find index 100000");
    }
}
