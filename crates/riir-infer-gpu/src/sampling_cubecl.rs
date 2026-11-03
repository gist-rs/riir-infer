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
// Last-index argmax kernel (Plan 626 T3a — the host-rule twin)
// ---------------------------------------------------------------------------

/// GPU argmax kernel with the league harness's tie rule: LAST index on ties.
///
/// [`argmax_f32`] pins FIRST-index (the determinism law, riir-ai Issue 940 —
/// strict-greater merges keep the lowest index on an exact-f32 tie). The greedy
/// decode league harness (`bench_831_decode_long` / `bench_809`) instead picks
/// its token with `max_by(total_cmp)`, which returns the LAST maximal element
/// — so an exact-f32 tie diverges the two walks and breaks the resident lane's
/// G1 EXACT bar (plan 626). This twin keeps the same reduction shape and the
/// same 4-byte download, with the merge rule flipped to the lexicographic max
/// over `(value, index)`: prefer the higher value, on an equal value prefer
/// the HIGHER index.
///
/// ⛔ The merge compares BOTH fields, and that is load-bearing (measured
/// 2026-11-03): the smem butterfly pairs `(tid, tid+128) … (tid, tid+2),
/// (tid, tid+1)` — slot order at the final step is NOT index order, because a
/// right-side champion can migrate into a left slot (on `[1.0, 5.0, 5.0,
/// 2.0]` the step-2 pair (0,2) moves `(5.0, idx2)` into slot 0, so the final
/// pair is `(idx2 @ slot0, idx1 @ slot1)`). A value-only `>=` merge would
/// take the slot-1 challenger there and return idx1 — NOT the last index.
/// With the full (value, then higher-index) comparison the result is a
/// genuine total order over distinct indices: ANY reduction tree — this
/// butterfly included — reproduces `max_by(total_cmp)` exactly.
///
/// Known sub-case (documented, out of scope): `total_cmp` orders `-0.0 <
/// +0.0` while the GPU `==` merge treats them equal — a `-0.0/+0.0` tie at
/// the maximum picks the higher index here vs the `+0.0` index on the host.
/// The tie fixture uses finite non-zero values; NaN logits are
/// upstream-screened (`finite.rs`) and never reach this kernel.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; vocab_size]` — logits from lm_head.
/// - `output`: `[u32; 1]` — index of the LAST maximal logit.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn argmax_f32_last(input: &[f32], output: &mut [u32]) {
    let dim = input.len() as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // Phase 1: Strided accumulation — last-index twin of `argmax_f32`'s loop.
    // `>=` is the FULL lexicographic take here: the scan index ascends, so an
    // equal value always carries the higher index. -1e30 not NEG_INFINITY —
    // see the note on `argmax_f32` above (WGSL has no infinity literal; naga
    // would fail the MSL emit and the kernel would never run, leaving the
    // poison sentinel in the output buffer).
    let mut local_max = f32::new(-1e30f32);
    let mut local_idx = u32::new(0i64);
    let mut i = tid;
    while i < dim {
        let val = input[i as usize];
        if val >= local_max {
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

    // Unrolled parallel reduction (128→64→32→16→8→4→2→1) with the FULL
    // lexicographic merge: take the right-hand challenger iff its value is
    // strictly greater, OR the values are equal AND its index is higher.
    // Plain nested ifs — the butterfly's slot order is not index order (the
    // doc comment's measured trap), so "challenger = higher index" is FALSE
    // in general and a value-only merge would misorder ties.
    if tid < 128u32 {
        let other = smem_val[(tid + 128u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 128u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 128u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 64u32 {
        let other = smem_val[(tid + 64u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 64u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 64u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 32u32 {
        let other = smem_val[(tid + 32u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 32u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 32u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 16u32 {
        let other = smem_val[(tid + 16u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 16u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 16u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 8u32 {
        let other = smem_val[(tid + 8u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 8u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 8u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 4u32 {
        let other = smem_val[(tid + 4u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 4u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 4u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 2u32 {
        let other = smem_val[(tid + 2u32) as usize];
        if other > smem_val[tid as usize] {
            smem_val[tid as usize] = other;
            smem_idx[tid as usize] = smem_idx[(tid + 2u32) as usize];
        } else {
            if other == smem_val[tid as usize] {
                let other_idx = smem_idx[(tid + 2u32) as usize];
                if other_idx > smem_idx[tid as usize] {
                    smem_idx[tid as usize] = other_idx;
                }
            }
        }
    }
    sync_cube();
    if tid < 1u32 {
        let other = smem_val[1usize];
        if other > smem_val[0usize] {
            smem_val[0usize] = other;
            smem_idx[0usize] = smem_idx[1usize];
        } else {
            if other == smem_val[0usize] {
                let other_idx = smem_idx[1usize];
                if other_idx > smem_idx[0usize] {
                    smem_idx[0usize] = other_idx;
                }
            }
        }
        output[0usize] = smem_idx[0usize];
    }
}

/// Launcher for the last-index argmax kernel (Plan 626 T3a).
///
/// Same dispatch shape and 4-byte download as [`ArgmaxCubeCL`]; the tie rule
/// is the league harness's `max_by(total_cmp)` (last index on ties). The
/// resident greedy lane dispatches THIS kernel — the first-index kernel is
/// untouched and keeps its pinned semantics (its 4 sentinel tests stand).
#[cfg(feature = "cubecl_runtime")]
pub struct ArgmaxLastCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ArgmaxLastCubeCL {
    /// Launch the last-index argmax kernel on GPU.
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
            argmax_f32_last::launch_unchecked::<R>(
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
    fn sentinel_out(client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>) -> Handle {
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

    // ── Plan 626 T3a — the last-index twin (`argmax_f32_last`) ────────────
    //
    // The league harness argmax is `max_by(total_cmp)` = LAST-index-on-ties;
    // the first-index kernel above pins the determinism law and is UNTOUCHED.
    // These tests pin the twin AND the divergence class: on an exact-f32 tie
    // the two kernels disagree EXACTLY at the tie, which is why the resident
    // greedy lane must dispatch the twin (plan 626 T3a).

    /// Deterministic pseudo-random f32 fill (LCG; no rand dep) in (−1, 1).
    fn pseudo_random_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    /// The host rule the twin must reproduce: `max_by(total_cmp)` = the LAST
    /// maximal element's index.
    fn host_argmax_last(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("non-empty")
            .0
    }

    fn run_last_kernel(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        input: &[f32],
    ) -> u32 {
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = sentinel_out(client);
        unsafe {
            ArgmaxLastCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                client,
                input_handle,
                output_handle.clone(),
                input.len(),
            );
        }
        read_argmax(client, output_handle)
    }

    #[test]
    fn test_argmax_last_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(run_last_kernel(&client, &input), 1u32);
    }

    #[test]
    fn test_argmax_last_single_element() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        assert_eq!(run_last_kernel(&client, &[42.0]), 0u32);
    }

    /// The twin's defining case: all-equal takes the LAST index (the
    /// first-index kernel takes 0 — its `test_argmax_all_equal` pins that).
    #[test]
    fn test_argmax_last_all_equal() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![3.0, 3.0, 3.0, 3.0];
        assert_eq!(
            run_last_kernel(&client, &input),
            3u32,
            "last-index twin must take the final maximal index (host max_by rule)"
        );
    }

    /// A finite non-zero tie pair mid-vector: the twin takes the higher
    /// index, exactly what `max_by(total_cmp)` returns.
    #[test]
    fn test_argmax_last_tie_pair() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![1.0, 5.0, 5.0, 2.0];
        assert_eq!(
            run_last_kernel(&client, &input),
            host_argmax_last(&input) as u32,
        );
    }

    /// The T3a divergence class, tested never latent: on a tie pair at slots
    /// 0/1 the two kernels diverge EXACTLY at the tie — the strict-greater
    /// tree keeps slot 0 (first-index), the twin takes the higher index
    /// (the host `max_by(total_cmp)` rule).
    ///
    /// The vector is [5.0, 5.0, …] NOT [1.0, 5.0, 5.0, 2.0]: the incumbent's
    /// butterfly pairs (0,2),(1,3) at the 2-step, so a mid-vector tie's
    /// right-side champion can migrate into slot 0 and win there (measured:
    /// [1, 5, 5, 2] → the incumbent returns 2 — tree-order luck, neither
    /// first- nor last-index). The adjacent-slots-0/1 tie isolates the RULE
    /// difference from the butterfly accident.
    #[test]
    fn test_argmax_tie_rules_diverge_exactly_at_the_tie() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![5.0, 5.0, 0.0, 0.0];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let first_out = sentinel_out(&client);
        unsafe {
            ArgmaxCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                first_out.clone(),
                input.len(),
            );
        }
        let first = read_argmax(&client, first_out);

        let last = run_last_kernel(&client, &input);

        assert_eq!(
            first, 0u32,
            "first-index kernel keeps the slot-0 champion on a 0/1 tie"
        );
        assert_eq!(
            last, 1u32,
            "last-index twin takes the tie's higher index (host rule)"
        );
        assert_ne!(first, last, "the tie must diverge the two rules");
    }

    /// The twin is tree-order-independent, the incumbent is NOT (the doc
    /// comment's measured trap): on the mid-vector tie the twin still equals
    /// the host `max_by(total_cmp)` pick even though the butterfly migrated
    /// the right champion into slot 0 (the incumbent there returns 2 by
    /// tree-order luck — asserted, so the class stays visible).
    #[test]
    fn test_argmax_last_is_tree_order_independent_on_mid_vector_tie() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let input: Vec<f32> = vec![1.0, 5.0, 5.0, 2.0];

        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let first_out = sentinel_out(&client);
        unsafe {
            ArgmaxCubeCL::launch::<crate::cubecl_runtime::ActiveRuntime>(
                &client,
                input_handle,
                first_out.clone(),
                input.len(),
            );
        }
        let first = read_argmax(&client, first_out);

        let last = run_last_kernel(&client, &input);

        assert_eq!(
            last,
            host_argmax_last(&input) as u32,
            "the twin must equal max_by(total_cmp) regardless of tree order"
        );
        assert_eq!(
            first, 2u32,
            "the incumbent's butterfly returns 2 here (tree-order luck, neither rule) — \
             this is why the resident lane dispatches the twin"
        );
    }

    /// The twin reproduces `max_by(total_cmp)` on deterministic pseudo-random
    /// logits at league scale (4096 ≫ the 256-wide reduction tree, maxima
    /// seeded at several positions including the last element).
    #[test]
    fn test_argmax_last_matches_host_max_by_total_cmp() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n = 4096usize;
        for seed in [7u64, 0x9E37u64, 0xDEADu64] {
            let mut input = pseudo_random_f32(n, seed);
            // Guarantee a sharp max so the case is decisive; alternate its
            // position across seeds (front / middle / last).
            let pos = match seed {
                7 => 0,
                0x9E37 => n / 2,
                _ => n - 1,
            };
            input[pos] = 0.999;
            assert_eq!(
                run_last_kernel(&client, &input),
                host_argmax_last(&input) as u32,
                "seed {seed}: twin must equal the host max_by(total_cmp) pick"
            );
        }
    }
}
