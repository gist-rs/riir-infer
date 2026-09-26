//! CubeCL element-wise activation + reduction kernels (Issue 401 Phase 1.4-1.6).
//!
//! GPU-resident sigmoid, SiLU, SiTU, softmax, and top-k to eliminate CPU sync
//! points in the hybrid KimiK3 forward path. These are the "missing kernels"
//! listed in Issue 401's substrate audit — the existing CubeCL kernels
//! (`RmsNormCubeCL`, `RopeCubeCL`, `GegluCubeCL`, `ArgmaxCubeCL`) cover Gemma2's
//! hot path, but KimiK3's MLA/KDA/MoE layers use different activations.
//!
//! # Kernels
//!
//! | Kernel | Algorithm | Dispatch | SharedMemory |
//! |--------|-----------|----------|--------------|
//! | `sigmoid_f32` | Element-wise `1/(1+exp(-x))` | `ceil(n/256)` WG × 256 threads | None |
//! | `silu_f32` | Element-wise `x * sigmoid(x)` | `ceil(n/256)` WG × 256 threads | None |
//! | `situ_f32` | Element-wise SiLU with 2 betas | `ceil(n/256)` WG × 256 threads | None |
//! | `softmax_f32` | Strided max reduce + exp + sum + normalize | 1 WG × 256 threads | 1 KB |
//! | `topk_f32` | Strided argmax select + accumulate | 1 WG × 256 threads | val+idx |
//!
//! # Numerically stable sigmoid
//!
//! `sigmoid_f32` mirrors `katgpt_core::sigmoid`: positive input uses
//! `1/(1+exp(-x))`, negative input uses `exp(x)/(1+exp(x))` — avoids the
//! `exp(large)` overflow path that produces NaN.
//!
//! # CubeCL v0.10 Constraints (same as `norms_cubecl.rs`)
//!
//! - No conditional expressions as values — use `if { }` statements.
//! - `UNIT_POS` is `u32`, `ABSOLUTE_POS` is `usize` — cast appropriately.
//! - `f32::new(literal)` for constants in `#[cube]` context.
//! - `terminate!()` instead of `return` in `#[cube]` kernels.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::debug_assert_binding_at_least;

// ---------------------------------------------------------------------------
// Sigmoid kernel
// ---------------------------------------------------------------------------

/// CubeCL element-wise sigmoid: `output[i] = 1 / (1 + exp(-input[i]))`.
///
/// Numerically stable: positive input uses `1/(1+exp(-x))`, negative input
/// uses `exp(x)/(1+exp(x))`. Matches `katgpt_core::sigmoid` bit-for-bit on
/// the stable branch (f32 fp non-associativity aside).
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn sigmoid_f32(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    let tid = ABSOLUTE_POS;

    if tid < n {
        let x = input[tid];
        // Branch-free stable sigmoid: pick the branch that avoids overflow.
        // CubeCL v0.10 doesn't allow conditional expressions as values, so
        // use explicit if/else with a mutable accumulator.
        let result = if x >= f32::new(0.0f32) {
            f32::new(1.0f32) / (f32::new(1.0f32) + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (f32::new(1.0f32) + ex)
        };
        output[tid] = result;
    }
}

// ---------------------------------------------------------------------------
// SiLU (swish) kernel
// ---------------------------------------------------------------------------

/// CubeCL element-wise SiLU (swish): `output[i] = input[i] * sigmoid(input[i])`.
///
/// Used by KDA ShortConv output (`silu_inplace` in `kda_cubecl.rs`) and the
/// Dense FFN activation path. Consumes the same numerically-stable sigmoid
/// branch as `sigmoid_f32`.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn silu_f32(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    let tid = ABSOLUTE_POS;

    if tid < n {
        let x = input[tid];
        let s = if x >= f32::new(0.0f32) {
            f32::new(1.0f32) / (f32::new(1.0f32) + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (f32::new(1.0f32) + ex)
        };
        output[tid] = x * s;
    }
}

// ---------------------------------------------------------------------------
// SiTU (Kimi-K3 Dense FFN / MoE expert activation) kernel
// ---------------------------------------------------------------------------

/// CubeCL element-wise SiTU: Kimi-K3's gated Dense FFN / expert activation.
///
/// Mirrors `katgpt_types::math::situ` + `moe_cubecl::situ_inplace` bit-for-bit
/// (f32 fp non-associativity aside). The formula is a gated activation with
/// TWO inputs (gate projection `g` + up projection `u`) and two betas:
///
/// ```text
/// gate_sigmoid = sigmoid(g)              // numerically stable
/// gate_tanh    = tanh(g / beta)          // beta bounds the gate's swing
/// up_t         = lb * tanh(u / lb)       // linear_beta soft-clamps up (if present)
/// out[i]       = beta * gate_tanh * gate_sigmoid * (up_t if has_lb else u[i])
/// ```
///
/// Used by both the Dense FFN (`dense_situ_ffn_gpu_batched`) and every MoE
/// expert (`situ_expert_forward_gpu`). Pre-fix (commit `1b79ca843`) this
/// kernel computed `x * sigmoid(x*β_lin + β)` — a swiGLU-like gate that is NOT
/// the real SiTU; the parity test passed only because it tested against the
/// same wrong reference. Fixed 2026-08-04 to match the production formula.
///
/// ## Parameter Layout
///
/// - `gate`: `[f32; n]` — gate projection output (W_gate · h).
/// - `up`:   `[f32; n]` — up projection output (W_up · h).
/// - `params`: `[f32; 3]` — `[beta, linear_beta, has_linear_beta]` where
///   `has_linear_beta != 0.0` selects the tanh-clamped up path (matches
///   `Option<f32>` = Some vs None).
/// - `output`: `[f32; n]` — activated values.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn situ_f32(
    gate: &[f32],
    up: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    let beta = params[0usize];
    let linear_beta = params[1usize];
    let has_lb = params[2usize];
    let inv_beta = f32::new(1.0f32) / beta;
    let n = gate.len();
    let tid = ABSOLUTE_POS;

    if tid < n {
        let g = gate[tid];
        let u = up[tid];
        // Numerically stable sigmoid(g) — same branch as `sigmoid_f32`.
        let gate_sigmoid = if g >= f32::new(0.0f32) {
            f32::new(1.0f32) / (f32::new(1.0f32) + (-g).exp())
        } else {
            let eg = g.exp();
            eg / (f32::new(1.0f32) + eg)
        };
        let gate_tanh = f32::tanh(g * inv_beta);
        // up_t: tanh-clamped when linear_beta is set, identity otherwise.
        let up_t = if has_lb != f32::new(0.0f32) {
            let inv_lb = f32::new(1.0f32) / linear_beta;
            linear_beta * f32::tanh(u * inv_lb)
        } else {
            u
        };
        output[tid] = beta * gate_tanh * gate_sigmoid * up_t;
    }
}

// ---------------------------------------------------------------------------
// Softmax kernel (single workgroup, shared-memory reduction)
// ---------------------------------------------------------------------------

/// CubeCL softmax kernel: numerically stable `exp(x - max) / sum(exp(x - max))`.
///
/// Single workgroup of 256 threads handles the full vector via strided access.
/// Shared memory is used for two unrolled parallel reductions: max + sum.
///
/// Mirrors the `rmsnorm_f32` reduction structure — same dispatch shape, same
/// shared-memory pattern, just a different reduction operator (max+exp+sum
/// instead of sum-of-squares).
///
/// ## Parameter Layout
///
/// - `input`: `[f32; dim]` — attention scores (or any logits).
/// - `params`: `[f32; 1]` — `[dim as f32]` (passed to avoid `input.len()` calls
///   that may trigger CubeCL v0.10 WGSL gen issues).
/// - `output`: `[f32; dim]` — softmax probabilities (sum to 1).
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
///
/// ## Limitation
///
/// Currently single-workgroup: handles `dim` up to ~16384 (256 threads ×
/// 64-element stride). For MLA attention where `dim = seq_len` (typically
/// ≤ 4096 for training), this is fine. For larger dims (vocab softmax), a
/// multi-workgroup variant would be needed.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn softmax_f32(input: &[f32], params: &[f32], output: &mut [f32]) {
    let dim = params[0usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // ── Phase 1: Strided max reduction ──
    // Each thread scans elements at tid, tid+256, tid+512, ... tracking the max.
    // Use -1e30 instead of NEG_INFINITY to avoid potential WGSL gen issues.
    let mut local_max = f32::new(-1e30f32);
    let mut i = tid;
    while i < dim {
        let x = input[i as usize];
        if x > local_max {
            local_max = x;
        }
        i += cube_size;
    }

    // ── Phase 2: Shared memory max reduction ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);
    smem[tid as usize] = local_max;
    sync_cube();

    // Unrolled parallel max reduction (128→64→32→16→8→4→2→1)
    if tid < 128u32 {
        let other = smem[(tid + 128u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 64u32 {
        let other = smem[(tid + 64u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 32u32 {
        let other = smem[(tid + 32u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 16u32 {
        let other = smem[(tid + 16u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 8u32 {
        let other = smem[(tid + 8u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 4u32 {
        let other = smem[(tid + 4u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 2u32 {
        let other = smem[(tid + 2u32) as usize];
        if other > smem[tid as usize] {
            smem[tid as usize] = other;
        }
    }
    sync_cube();
    if tid < 1u32 {
        let other = smem[1usize];
        if other > smem[0usize] {
            smem[0usize] = other;
        }
    }
    sync_cube();

    // Broadcast the global max via shared memory.
    let global_max = smem[0usize];

    // ── Phase 3: Compute sum of exp(x - max) (without storing — recompute later) ──
    let mut local_sum = f32::new(0.0f32);
    let mut j = tid;
    while j < dim {
        let x = input[j as usize];
        local_sum += (x - global_max).exp();
        j += cube_size;
    }

    // ── Phase 4: Shared memory sum reduction ──
    smem[tid as usize] = local_sum;
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

    let inv_sum = f32::new(1.0f32) / smem[0usize];

    // ── Phase 5: Normalize — recompute exp(x-max)/sum and write to output ──
    let mut k = tid;
    while k < dim {
        let x = input[k as usize];
        output[k as usize] = (x - global_max).exp() * inv_sum;
        k += cube_size;
    }
}

// ---------------------------------------------------------------------------
// Top-K kernel (single workgroup, iterative selection)
// ---------------------------------------------------------------------------

/// CubeCL top-k selection kernel: finds the indices of the `k` largest elements.
///
/// Uses a single workgroup of 256 threads. For each of the `k` selection
/// rounds, the workgroup does a strided max-reduction (same pattern as
/// `argmax_f32` in `sampling_cubecl.rs`), marks the selected element as
/// `-1e30` (effectively -infinity for comparison purposes; NEG_INFINITY is
/// avoided because CubeCL v0.10's WGSL backend can't compile it) in the
/// mutable `scratch` buffer so it won't be picked again, then proceeds to
/// the next round.
///
/// Output is `[u32; k]` — the indices of the top-k elements, in descending
/// order of value.
///
/// ## Parameter Layout
///
/// - `input`: `[f32; n]` — scores (e.g. biased MoE router scores). Read-only.
/// - `scratch`: `[f32; n]` — mutable working copy. Caller pre-fills with `input`.
///   Selected elements are marked `-inf` here between rounds.
/// - `params`: `[f32; 2]` — `[k as f32, n as f32]`.
/// - `output_indices`: `[u32; k]` — selected indices (descending value order).
/// - `output_values`: `[f32; k]` — selected values (descending order).
///
/// ## Dispatch
///
/// `CubeCount::Static(1, 1, 1)`, `CubeDim::new_1d(256)`.
///
/// ## Limitation
///
/// Single-workgroup: handles `n` up to ~16384. `k` must be ≤ 256.
/// For MoE with `n_r` routed experts (typically 8-64) and `k_r` selected
/// (typically 4-8), this is well within bounds.
///
/// ## Why scratch is a separate buffer (not shared memory)
///
/// CubeCL v0.10 requires `Shared::new_slice(size)` to take a compile-time
/// constant `usize` — a runtime `n` derived from params fails to compile
/// (`usize: From<NativeExpand<usize>>` not satisfied). The scratch buffer is
/// a regular GPU buffer passed as a kernel argument, sized at launch time.
///
/// ## Cost
///
/// `O(k * n)` — each of the `k` rounds scans the full scratch. For `k=8, n=64`
/// that's 512 element reads per round, 4096 total. At 256 threads/round, that's
/// 16 rounds × ~2µs/round ≈ 32µs total — negligible vs the GEMV dispatches.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn topk_f32(
    input: &[f32],
    scratch: &mut [f32],
    params: &[f32],
    output_indices: &mut [u32],
    output_values: &mut [f32],
) {
    let k = params[0usize] as u32;
    let n = params[1usize] as u32;
    let cube_size = 256u32;
    let tid = UNIT_POS;

    // Copy input → scratch on the first round (if not already copied by caller).
    // We do it here so the kernel is self-contained — the caller passes an
    // uninitialized scratch buffer of the right size.
    let mut i_init = tid;
    while i_init < n {
        scratch[i_init as usize] = input[i_init as usize];
        i_init += cube_size;
    }
    sync_cube();

    // Reduction scratch (256 threads — fixed compile-time size).
    let mut smem_val = Shared::<[f32]>::new_slice(256usize);
    let mut smem_idx = Shared::<[u32]>::new_slice(256usize);

    let mut round = u32::new(0i64);
    while round < k {
        // ── Strided max scan over scratch ──
        // Use -1e30 instead of NEG_INFINITY to avoid WGSL gen issues.
        let mut local_max = f32::new(-1e30f32);
        let mut local_idx = u32::new(0i64);
        let mut i_scan = tid;
        while i_scan < n {
            let val = scratch[i_scan as usize];
            if val > local_max {
                local_max = val;
                local_idx = i_scan;
            }
            i_scan += cube_size;
        }

        smem_val[tid as usize] = local_max;
        smem_idx[tid as usize] = local_idx;
        sync_cube();

        // ── Unrolled parallel max reduction (128→64→32→16→8→4→2→1) ──
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
        }
        sync_cube();

        // Thread 0 writes the result + marks the selected element as -inf in scratch.
        if tid < 1u32 {
            let best_idx = smem_idx[0usize];
            let best_val = smem_val[0usize];
            output_indices[round as usize] = best_idx;
            output_values[round as usize] = best_val;
            scratch[best_idx as usize] = f32::new(-1e30f32);
        }
        sync_cube();

        round += u32::new(1i64);
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fill-zeros kernel (zero-fill an existing buffer in-place)
// ---------------------------------------------------------------------------

/// CubeCL fill-zeros: `output[i] = 0.0`. Zero-fills an existing GPU buffer
/// in-place — eliminates the per-`reset_state()` GPU allocation that
/// fragmented the CubeCL memory pool (Plan 528 T0.2 blocker).
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn fill_zeros_f32(output: &mut [f32]) {
    let n = output.len();
    let tid = ABSOLUTE_POS;

    if tid < n {
        output[tid] = f32::new(0.0f32);
    }
}

/// CubeCL copy: `output[i] = input[i]`. Copies an existing GPU buffer to
/// another existing GPU buffer — no allocation, no CPU sync. Used by
/// `checkpoint_speculative_gpu` / `rollback_speculative_gpu` (Issue 665
/// Phase 2) to snapshot DeltaNet recurrent state + conv state without
/// draining the GPU pipeline.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(n/256), 1, 1)`, `CubeDim::new_1d(256)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn copy_f32(input: &[f32], output: &mut [f32]) {
    let n = output.len();
    let tid = ABSOLUTE_POS;

    if tid < n {
        output[tid] = input[tid];
    }
}

// ---------------------------------------------------------------------------
// Launchers (public API mirroring RmsNormCubeCL / ArgmaxCubeCL)
// ---------------------------------------------------------------------------

/// CubeCL sigmoid launcher.
///
/// Wraps the `sigmoid_f32` kernel. Element-wise — no shared memory needed.
#[cfg(feature = "cubecl_runtime")]
pub struct SigmoidCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SigmoidCubeCL {
    /// Launch sigmoid kernel: `output[i] = sigmoid(input[i])`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `n` f32 elements
    /// - `output_handle`: `n` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        let n_wg = n.div_ceil(256).max(1) as u32;
        unsafe {
            sigmoid_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

/// CubeCL SiLU (swish) launcher.
///
/// Wraps the `silu_f32` kernel. Element-wise — no shared memory needed.
#[cfg(feature = "cubecl_runtime")]
pub struct SiluCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SiluCubeCL {
    /// Launch SiLU kernel: `output[i] = input[i] * sigmoid(input[i])`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `n` f32 elements
    /// - `output_handle`: `n` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        let n_wg = n.div_ceil(256).max(1) as u32;
        unsafe {
            silu_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

/// CubeCL SiTU launcher.
///
/// Wraps the `situ_f32` kernel. Element-wise gated activation with two inputs
/// (gate + up) and two betas. Mirrors `katgpt_types::math::situ`.
#[cfg(feature = "cubecl_runtime")]
pub struct SituCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SituCubeCL {
    /// Launch SiTU kernel (Kimi-K3 Dense FFN / MoE expert activation).
    ///
    /// Computes `out[i] = beta * tanh(g[i]/beta) * sigmoid(g[i]) * up_t[i]`
    /// where `up_t[i] = linear_beta * tanh(u[i]/linear_beta)` when
    /// `has_linear_beta != 0.0`, else `up_t[i] = u[i]`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `gate_handle`: `n` f32 elements
    /// - `up_handle`:   `n` f32 elements
    /// - `params_handle`: 3 f32 elements `[beta, linear_beta, has_linear_beta]`
    ///   (`has_linear_beta != 0.0` ⇒ use the tanh-clamped up path)
    /// - `output_handle`: `n` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        gate_handle: Handle,
        up_handle: Handle,
        params_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        let n_wg = n.div_ceil(256).max(1) as u32;
        unsafe {
            situ_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(gate_handle, n),
                BufferArg::from_raw_parts(up_handle, n),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

/// CubeCL softmax launcher.
///
/// Wraps the `softmax_f32` kernel. Single-workgroup handles the full vector
/// via strided access. Numerically stable (subtracts max before exp).
#[cfg(feature = "cubecl_runtime")]
pub struct SoftmaxCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl SoftmaxCubeCL {
    /// Launch softmax kernel: `output = softmax(input)` (numerically stable).
    ///
    /// Dispatch: `(1, 1, 1)` workgroup of 256 threads (strided access).
    ///
    /// # Safety
    ///
    /// - `input_handle`: `dim` f32 elements
    /// - `output_handle`: `dim` f32 elements
    ///
    /// `dim` must be > 0 and ≤ ~16384 (256 threads × 64-element stride).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
        dim: usize,
    ) {
        let params: &[f32] = &[dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        unsafe {
            softmax_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, dim),
                BufferArg::from_raw_parts(params_handle, 1),
                BufferArg::from_raw_parts(output_handle, dim),
            );
        }
    }
}

/// CubeCL top-k launcher.
///
/// Wraps the `topk_f32` kernel. Single-workgroup selects the `k` largest
/// elements via iterative argmax with mark-invalid in a scratch buffer.
#[cfg(feature = "cubecl_runtime")]
pub struct TopKCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TopKCubeCL {
    /// Launch top-k kernel: finds indices + values of the `k` largest elements.
    ///
    /// Dispatch: `(1, 1, 1)` workgroup of 256 threads (strided + iterative).
    ///
    /// # Safety
    ///
    /// - `input_handle`: `n` f32 elements (read-only)
    /// - `scratch_handle`: `n` f32 elements (mutable — kernel copies input here
    ///   + marks selected as -inf between rounds)
    /// - `output_indices_handle`: `k` u32 elements
    /// - `output_values_handle`: `k` f32 elements
    ///
    /// `k` must be > 0 and ≤ 256. `n` must be > 0 and ≤ ~16384.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        scratch_handle: Handle,
        output_indices_handle: Handle,
        output_values_handle: Handle,
        n: usize,
        k: usize,
    ) {
        let params: &[f32] = &[k as f32, n as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        unsafe {
            topk_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(scratch_handle, n),
                BufferArg::from_raw_parts(params_handle, 2),
                BufferArg::from_raw_parts(output_indices_handle, k),
                BufferArg::from_raw_parts(output_values_handle, k),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// FillZeros launcher
// ---------------------------------------------------------------------------

/// CubeCL fill-zeros launcher.
///
/// Wraps the `fill_zeros_f32` kernel. Zero-fills an existing GPU buffer
/// in-place — no new allocation. Used by `TernaryDeltanetGpuForward::reset_state`
/// to avoid per-move GPU buffer churn that fragmented the memory pool.
#[cfg(feature = "cubecl_runtime")]
pub struct FillZerosCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl FillZerosCubeCL {
    /// Launch fill-zeros kernel: `output[i] = 0.0` for all `i` in `[0, n)`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `output_handle`: `n` f32 elements (modified in-place)
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        output_handle: Handle,
        n: usize,
    ) {
        let n_wg = n.div_ceil(256).max(1) as u32;
        unsafe {
            fill_zeros_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

/// CubeCL copy launcher.
///
/// Wraps the `copy_f32` kernel. Copies an existing GPU buffer to another
/// existing GPU buffer — no allocation, no CPU sync. Used by
/// `checkpoint_speculative_gpu` / `rollback_speculative_gpu` (Issue 665
/// Phase 2) to snapshot DeltaNet recurrent state + conv state entirely on
/// the GPU side.
#[cfg(feature = "cubecl_runtime")]
pub struct CopyCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl CopyCubeCL {
    /// Launch copy kernel: `output[i] = input[i]` for all `i` in `[0, n)`.
    ///
    /// Dispatch: `ceil(n/256)` workgroups of 256 threads.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `n` f32 elements (read-only)
    /// - `output_handle`: `n` f32 elements (modified in-place)
    /// - `input_handle` and `output_handle` must NOT alias
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        let n_wg = n.div_ceil(256).max(1) as u32;
        unsafe {
            copy_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, n),
                BufferArg::from_raw_parts(output_handle, n),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4-way split kernel (Issue 642 F3)
// ---------------------------------------------------------------------------

/// CubeCL kernel: split a concatenated buffer into 4 separate output buffers.
///
/// Copies elements from a flat `input` buffer into 4 outputs based on offset
/// parameters. Used by Issue 642 F3 to split the concatenated DeltaNet input
/// projection GEMV output (`qkv | z | a | b`) into the 4 separate downstream
/// buffers (`qkv`, `z_buf`, `a_raw`, `b_raw`).
///
/// Dispatch: `ceil(total/256)` workgroups of 256 threads, where
/// `total = len1 + len2 + len3 + len4`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn split4_f32(
    input: &[f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    out4: &mut [f32],
    params: &[f32],
) {
    let len1 = params[0usize] as usize;
    let len2 = params[1usize] as usize;
    let len3 = params[2usize] as usize;
    let len4 = params[3usize] as usize;
    let total = len1 + len2 + len3 + len4;
    let idx = ABSOLUTE_POS;

    if idx >= total {
        terminate!();
    }

    let val = input[idx];
    if idx < len1 {
        out1[idx] = val;
    } else if idx < len1 + len2 {
        out2[idx - len1] = val;
    } else if idx < len1 + len2 + len3 {
        out3[idx - len1 - len2] = val;
    } else {
        out4[idx - len1 - len2 - len3] = val;
    }
}

/// CubeCL kernel: split a concatenated buffer into 2 separate output buffers
/// (Plan 602 B2 — the folded-model qkv|z fan-out; the dense a/b escape set
/// writes its own buffers via the f32 dense GEMV, so no third/fourth region
/// exists on that path).
///
/// Dispatch: `ceil(total/256)` workgroups of 256 threads, where
/// `total = len1 + len2`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn split2_f32(input: &[f32], out1: &mut [f32], out2: &mut [f32], params: &[f32]) {
    let len1 = params[0usize] as usize;
    let len2 = params[1usize] as usize;
    let total = len1 + len2;
    let idx = ABSOLUTE_POS;

    if idx >= total {
        terminate!();
    }

    let val = input[idx];
    if idx < len1 {
        out1[idx] = val;
    } else {
        out2[idx - len1] = val;
    }
}

/// Launcher for the 4-way split kernel (Issue 642 F3).
#[cfg(feature = "cubecl_runtime")]
pub struct Split4CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl Split4CubeCL {
    /// Launch 4-way split: copies `input[0..len1]` → `out1`,
    /// `input[len1..len1+len2]` → `out2`, etc.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `len1 + len2 + len3 + len4` f32 elements
    /// - `out1_handle`: `len1` f32 elements
    /// - `out2_handle`: `len2` f32 elements
    /// - `out3_handle`: `len3` f32 elements
    /// - `out4_handle`: `len4` f32 elements
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        out1_handle: Handle,
        out2_handle: Handle,
        out3_handle: Handle,
        out4_handle: Handle,
        len1: usize,
        len2: usize,
        len3: usize,
        len4: usize,
    ) {
        let total = len1 + len2 + len3 + len4;
        let params: [f32; 4] = [len1 as f32, len2 as f32, len3 as f32, len4 as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let n_wg = total.div_ceil(256).max(1) as u32;

        unsafe {
            split4_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, total),
                BufferArg::from_raw_parts(out1_handle, len1),
                BufferArg::from_raw_parts(out2_handle, len2),
                BufferArg::from_raw_parts(out3_handle, len3),
                BufferArg::from_raw_parts(out4_handle, len4),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

/// Launcher for the 2-way split kernel (Plan 602 B2 — the folded GDN input
/// split: the concat carries qkv|z only, the dense escape-set a/b dispatch
/// as separate f32 GEMVs, so the fan-out needs two regions not four).
#[cfg(feature = "cubecl_runtime")]
pub struct Split2CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl Split2CubeCL {
    /// Launch 2-way split: copies `input[0..len1]` → `out1`,
    /// `input[len1..len1+len2]` → `out2`.
    ///
    /// # Safety
    ///
    /// - `input_handle`: `len1 + len2` f32 elements
    /// - `out1_handle`: `len1` f32 elements
    /// - `out2_handle`: `len2` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        out1_handle: Handle,
        out2_handle: Handle,
        len1: usize,
        len2: usize,
    ) {
        let total = len1 + len2;
        let params: [f32; 2] = [len1 as f32, len2 as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let n_wg = total.div_ceil(256).max(1) as u32;

        unsafe {
            split2_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input_handle, total),
                BufferArg::from_raw_parts(out1_handle, len1),
                BufferArg::from_raw_parts(out2_handle, len2),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The encoder-lane elementwise op family (plan 611 S1b — T7 op-layer
// unification). The trivial half of the laya `Backend` trait's ops over
// CubeCL: add / add_bias_row / scale / relu / gelu_erf / glu_gelu_gate /
// copy_at.
//
// Offset-bearing ops (`add`, `copy_at`) bind WHOLE parent handles and take
// their element offsets through the params array: wgpu enforces
// `min_storage_buffer_offset_alignment` (32 B) on storage-buffer bindings,
// and the forward's offsets are element-arbitrary (the attention slab adds
// sit at `head · seq²`), so byte-offset handle views only ever cover the
// d-multiple cases. Params triples are blake3-cached when the
// `params_handle_cache` feature is on.
//
// gelu is the erf form — `x · ½ · (1 + erf(x/√2))` — through CubeCL's
// `f32::erf`, the same kernel shape the crate's test-only `gelu_scalar_verify`
// pinned; the CPU reference in the tests below is `libm::erff` (the laya
// lane's own gelu), and the two implementations differ by a few ulp — the
// tests use an absolute tolerance, never exact equality, on erf-bearing ops.
//
// Handle trivia the S1b work measured (worth keeping for S2's bind
// decisions): CubeCL's layout policy PADS allocations to a bucket (a
// 64-byte create lands in a 256-byte buffer) and returns handles whose
// `offset_end` ALREADY carries the slack — a fresh handle's live range is
// `[offset_start, size − offset_end)`, i.e. always read sizes through
// `size_in_used()`, never `size()`. And byte-offset views are only bindable
// at 32-byte-aligned offsets (wgpu's storage-buffer alignment limit), which
// is exactly why the offset ops above take params instead.
// ---------------------------------------------------------------------------

/// Scalar erf-gelu: `x · ½ · (1 + erf(x/√2))`.
#[cfg(feature = "cubecl_runtime")]
#[allow(unstable_name_collisions)] // CubeCL's f32::erf() may collide with a future std method
#[cube]
fn gelu_erf_scalar(x: f32) -> f32 {
    let sqrt2 = f32::new(comptime!(2.0f32.sqrt()));
    x * (f32::erf(x / sqrt2) + f32::new(1.0f32)) / f32::new(2.0f32)
}

/// `x[x_off + i] += y[y_off + i]` for `i in [0, n)` — in place on the whole
/// parent `x`, with the offsets as kernel params.
///
/// Offsets ride the params array rather than byte-offset handle views:
/// wgpu enforces `min_storage_buffer_offset_alignment` (32 B) on storage
/// bindings, and the forward's offsets are element-arbitrary (the attention
/// slab adds sit at `head · seq²`), so views would only cover the
/// d-multiple cases. Params are the portable answer; the per-op cost is
/// one params upload on a new (n, x_off, y_off) triple — blake3-cached
/// when the params_handle_cache feature is on.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn add_f32(x: &mut [f32], y: &[f32], params: &[f32]) {
    let n = params[0usize] as u32;
    let x_off = params[1usize] as u32;
    let y_off = params[2usize] as u32;
    let tid = ABSOLUTE_POS as u32;
    if tid < n {
        x[(x_off + tid) as usize] += y[(y_off + tid) as usize];
    }
}

/// `x[i] += bias[i % d]` — the row-broadcast bias add over `[rows × d]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn add_bias_row_f32(x: &mut [f32], bias: &[f32], params: &[f32]) {
    let n = x.len();
    let tid = ABSOLUTE_POS;
    if tid < n {
        let d = params[0usize] as u32;
        let col = (tid as u32) % d;
        x[tid] += bias[col as usize];
    }
}

/// `x[i] *= s` — the scalar (params[0]) multiply, in place.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn scale_f32(x: &mut [f32], params: &[f32]) {
    let n = x.len();
    let tid = ABSOLUTE_POS;
    if tid < n {
        x[tid] *= params[0usize];
    }
}

/// `x[i] = max(0, x[i])`, in place.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn relu_f32(x: &mut [f32]) {
    let n = x.len();
    let tid = ABSOLUTE_POS;
    if tid < n {
        let v = x[tid];
        if v < f32::new(0.0f32) {
            x[tid] = f32::new(0.0f32);
        }
    }
}

/// erf-gelu, in place.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gelu_erf_f32(x: &mut [f32]) {
    let n = x.len();
    let tid = ABSOLUTE_POS;
    if tid < n {
        x[tid] = gelu_erf_scalar(x[tid]);
    }
}

/// `out[r·I + j] = gelu(fused[r·2I + j]) · fused[r·2I + I + j]` — the MLP-up
/// projection's GLU gate: the fused `[rows × 2I]` buffer's first half is the
/// value, second half the gate (the laya lane's exact semantics).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn glu_gelu_gate_f32(fused: &[f32], out: &mut [f32], params: &[f32]) {
    let n = out.len();
    let tid = ABSOLUTE_POS;
    if tid < n {
        let i_sz = params[0usize] as u32;
        let t = tid as u32;
        let row = t / i_sz;
        let j = t % i_sz;
        let base = row * i_sz * 2u32;
        let v = fused[(base + j) as usize];
        let gate = fused[(base + i_sz + j) as usize];
        out[tid] = gelu_erf_scalar(v) * gate;
    }
}

/// `dst[dst_off + i] = src[src_off + i]` for `i in [0, n)` — the offset
/// copy over WHOLE parent binds (see [`add_f32`] for why the offsets ride
/// the params array, not handle views).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn copy_at_f32(src: &[f32], dst: &mut [f32], params: &[f32]) {
    let n = params[0usize] as u32;
    let src_off = params[1usize] as u32;
    let dst_off = params[2usize] as u32;
    let tid = ABSOLUTE_POS as u32;
    if tid < n {
        dst[(dst_off + tid) as usize] = src[(src_off + tid) as usize];
    }
}

// -- launchers --------------------------------------------------------------

fn elementwise_wg_count(n: usize) -> u32 {
    n.div_ceil(256).max(1) as u32
}

/// CubeCL launcher for [`add_f32`] — the laya `Backend::add` shape: in-place
/// `x += y` at element offsets over WHOLE parent binds (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct AddCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl AddCubeCL {
    /// Launch in-place add of `len` elements at the given parent offsets.
    ///
    /// # Safety
    ///
    /// `x_handle` must back `x_len` f32 and `y_handle` `y_len` f32, with
    /// `x_off + len <= x_len` and `y_off + len <= y_len`.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        x_len: usize,
        y_handle: Handle,
        y_len: usize,
        x_off: usize,
        y_off: usize,
        len: usize,
    ) {
        debug_assert_binding_at_least(&x_handle, x_len, "Add::x");
        debug_assert_binding_at_least(&y_handle, y_len, "Add::y");
        assert!(x_off + len <= x_len, "add: x extent");
        assert!(y_off + len <= y_len, "add: y extent");
        let params: &[f32] = &[len as f32, x_off as f32, y_off as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = elementwise_wg_count(len);
        // SAFETY: extents asserted above; the kernel bounds-checks tid < n.
        unsafe {
            add_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, x_len),
                BufferArg::from_raw_parts(y_handle, y_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// CubeCL launcher for [`copy_at_f32`] — the laya `Backend::copy_at` shape
/// (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct CopyAtCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl CopyAtCubeCL {
    /// Launch the offset copy over whole parent binds.
    ///
    /// # Safety
    ///
    /// `src_handle` must back `src_len` f32 and `dst_handle` `dst_len` f32,
    /// with `src_off + len <= src_len` and `dst_off + len <= dst_len`.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        src_len: usize,
        dst_handle: Handle,
        dst_len: usize,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) {
        debug_assert_binding_at_least(&src_handle, src_len, "CopyAt::src");
        debug_assert_binding_at_least(&dst_handle, dst_len, "CopyAt::dst");
        assert!(src_off + len <= src_len, "copy_at: src extent");
        assert!(dst_off + len <= dst_len, "copy_at: dst extent");
        let params: &[f32] = &[len as f32, src_off as f32, dst_off as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = elementwise_wg_count(len);
        // SAFETY: extents asserted above; the kernel bounds-checks tid < n.
        unsafe {
            copy_at_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(src_handle, src_len),
                BufferArg::from_raw_parts(dst_handle, dst_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// CubeCL launcher for [`add_bias_row_f32`] (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct AddBiasRowCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl AddBiasRowCubeCL {
    /// Launch the row-broadcast bias add over the whole `x` (`[rows × d]`).
    ///
    /// # Safety
    ///
    /// `x_handle` must back `x_len` f32; `bias_handle` must back `d` f32;
    /// `x_len` must be a multiple of `d`.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        x_len: usize,
        bias_handle: Handle,
        d: usize,
    ) {
        debug_assert_binding_at_least(&x_handle, x_len, "AddBiasRow::x");
        debug_assert_binding_at_least(&bias_handle, d, "AddBiasRow::bias");
        assert!(d > 0, "add_bias_row: d must be > 0");
        assert_eq!(x_len % d, 0, "add_bias_row: row extent");
        let params: &[f32] = &[d as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = elementwise_wg_count(x_len);
        // SAFETY: caller guarantees the binding sizes above.
        unsafe {
            add_bias_row_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, x_len),
                BufferArg::from_raw_parts(bias_handle, d),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

/// CubeCL launcher for [`scale_f32`] (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct ScaleCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ScaleCubeCL {
    /// Launch the in-place scalar multiply over `n` elements.
    ///
    /// # Safety
    ///
    /// `x_handle` must back ≥ `n` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_handle: Handle,
        n: usize,
        s: f32,
    ) {
        debug_assert_binding_at_least(&x_handle, n, "Scale::x");
        let params: &[f32] = &[s];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = elementwise_wg_count(n);
        // SAFETY: caller guarantees the binding size above.
        unsafe {
            scale_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, n),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

/// CubeCL launcher for [`relu_f32`] (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct ReluCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl ReluCubeCL {
    /// Launch the in-place ReLU over `n` elements.
    ///
    /// # Safety
    ///
    /// `x_handle` must back ≥ `n` f32 elements.
    pub unsafe fn launch<R: Runtime>(client: &ComputeClient<R>, x_handle: Handle, n: usize) {
        debug_assert_binding_at_least(&x_handle, n, "Relu::x");
        let n_wg = elementwise_wg_count(n);
        // SAFETY: caller guarantees the binding size above.
        unsafe {
            relu_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, n),
            );
        }
    }
}

/// CubeCL launcher for [`gelu_erf_f32`] (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct GeluErfCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GeluErfCubeCL {
    /// Launch the in-place erf-gelu over `n` elements.
    ///
    /// # Safety
    ///
    /// `x_handle` must back ≥ `n` f32 elements.
    pub unsafe fn launch<R: Runtime>(client: &ComputeClient<R>, x_handle: Handle, n: usize) {
        debug_assert_binding_at_least(&x_handle, n, "GeluErf::x");
        let n_wg = elementwise_wg_count(n);
        // SAFETY: caller guarantees the binding size above.
        unsafe {
            gelu_erf_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_handle, n),
            );
        }
    }
}

/// CubeCL launcher for [`glu_gelu_gate_f32`] (plan 611 S1b).
#[cfg(feature = "cubecl_runtime")]
pub struct GluGeluGateCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GluGeluGateCubeCL {
    /// Launch the GLU gate: `out[r·I + j] = gelu(fused[r·2I + j]) ·
    /// fused[r·2I + I + j]` over `rows × i_sz` outputs.
    ///
    /// # Safety
    ///
    /// `fused_handle` must back `rows * 2 * i_sz` f32; `out_handle` must back
    /// `rows * i_sz` f32; `rows`/`i_sz` must be > 0.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        fused_handle: Handle,
        out_handle: Handle,
        rows: usize,
        i_sz: usize,
    ) {
        let n = rows * i_sz;
        debug_assert_binding_at_least(&fused_handle, rows * 2 * i_sz, "GluGeluGate::fused");
        debug_assert_binding_at_least(&out_handle, n, "GluGeluGate::out");
        assert!(rows > 0 && i_sz > 0, "glu_gelu_gate: rows/i_sz must be > 0");
        let params: &[f32] = &[i_sz as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(params));
        let n_wg = elementwise_wg_count(n);
        // SAFETY: caller guarantees the binding sizes above.
        unsafe {
            glu_gelu_gate_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(fused_handle, rows * 2 * i_sz),
                BufferArg::from_raw_parts(out_handle, n),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    /// Verify Split2CubeCL correctly splits a qkv|z concat into 2 outputs
    /// (Plan 602 B2 — the folded-model GDN input fan-out).
    #[test]
    fn test_split2_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-2 folded concat shape (trimmed): qkv | z
        let len1 = 16; // qkv
        let len2 = 8; // z
        let total = len1 + len2;

        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.5).collect();

        let input_h = client.create_from_slice(f32::as_bytes(&input));
        let out1_h = client.empty(len1 * core::mem::size_of::<f32>());
        let out2_h = client.empty(len2 * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            Split2CubeCL::launch::<ActiveRuntime>(
                &client,
                input_h,
                out1_h.clone(),
                out2_h.clone(),
                len1,
                len2,
            );
        }

        let o1 = f32::from_bytes(&client.read_one(out1_h).unwrap()).to_vec();
        let o2 = f32::from_bytes(&client.read_one(out2_h).unwrap()).to_vec();

        assert_eq!(o1.len(), len1);
        assert_eq!(o2.len(), len2);
        for i in 0..len1 {
            assert_eq!(o1[i], input[i], "out1[{i}] mismatch");
        }
        for i in 0..len2 {
            assert_eq!(o2[i], input[len1 + i], "out2[{i}] mismatch");
        }
    }

    /// Verify Split4CubeCL correctly splits a concatenated buffer into 4 outputs.
    #[test]
    fn test_split4_basic() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-27B DeltaNet shape: qkv_dim=1536, z_dim=512, n_v_heads=4 ×2
        let len1 = 16; // qkv (trimmed for test speed)
        let len2 = 8; // z
        let len3 = 4; // a_raw
        let len4 = 4; // b_raw
        let total = len1 + len2 + len3 + len4;

        // Fill input with distinct values per region
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.5).collect();

        let input_h = client.create_from_slice(f32::as_bytes(&input));
        let out1_h = client.empty(len1 * core::mem::size_of::<f32>());
        let out2_h = client.empty(len2 * core::mem::size_of::<f32>());
        let out3_h = client.empty(len3 * core::mem::size_of::<f32>());
        let out4_h = client.empty(len4 * core::mem::size_of::<f32>());

        // SAFETY: Buffers are correctly sized.
        unsafe {
            Split4CubeCL::launch::<ActiveRuntime>(
                &client,
                input_h,
                out1_h.clone(),
                out2_h.clone(),
                out3_h.clone(),
                out4_h.clone(),
                len1,
                len2,
                len3,
                len4,
            );
        }

        let o1 = f32::from_bytes(&client.read_one(out1_h).unwrap()).to_vec();
        let o2 = f32::from_bytes(&client.read_one(out2_h).unwrap()).to_vec();
        let o3 = f32::from_bytes(&client.read_one(out3_h).unwrap()).to_vec();
        let o4 = f32::from_bytes(&client.read_one(out4_h).unwrap()).to_vec();

        assert_eq!(o1.len(), len1);
        assert_eq!(o2.len(), len2);
        assert_eq!(o3.len(), len3);
        assert_eq!(o4.len(), len4);

        // Verify each output matches the corresponding input region
        for i in 0..len1 {
            assert_eq!(o1[i], input[i], "out1[{i}] mismatch");
        }
        for i in 0..len2 {
            assert_eq!(o2[i], input[len1 + i], "out2[{i}] mismatch");
        }
        for i in 0..len3 {
            assert_eq!(o3[i], input[len1 + len2 + i], "out3[{i}] mismatch");
        }
        for i in 0..len4 {
            assert_eq!(o4[i], input[len1 + len2 + len3 + i], "out4[{i}] mismatch");
        }

        println!("split4 ({len1}+{len2}+{len3}+{len4} = {total}): all regions correct");
    }

    // ── plan 611 S1b: the encoder-lane elementwise family ──────────────

    fn lcg_vec(n: usize) -> Vec<f32> {
        let mut s = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32) / 8_388_608.0 - 1.0
            })
            .collect()
    }

    fn gelu_ref(x: f32) -> f32 {
        x * 0.5 * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
    }

    #[test]
    fn test_add_offsets_match_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let x_parent = lcg_vec(16);
        let y_parent = lcg_vec(12);
        let (x_off, y_off, len) = (4usize, 2usize, 8usize);

        // CPU reference on copies.
        let mut cpu_x = x_parent.clone();
        for i in 0..len {
            cpu_x[x_off + i] += y_parent[y_off + i];
        }

        let x_h = client.create_from_slice(f32::as_bytes(&x_parent));
        let y_h = client.create_from_slice(f32::as_bytes(&y_parent));
        // SAFETY: parents sized 16 / 12; offsets + len fit both.
        unsafe {
            AddCubeCL::launch::<ActiveRuntime>(
                &client,
                x_h.clone(),
                x_parent.len(),
                y_h.clone(),
                y_parent.len(),
                x_off,
                y_off,
                len,
            );
        }

        let got = f32::from_bytes(&client.read_one(x_h).unwrap()).to_vec();
        assert_eq!(got.len(), x_parent.len());
        for i in 0..x_parent.len() {
            assert_eq!(got[i], cpu_x[i], "add offset mismatch at {i}");
        }
        // The y parent must be untouched.
        let y_back = f32::from_bytes(&client.read_one(y_h).unwrap()).to_vec();
        assert_eq!(y_back, y_parent, "y parent modified");
    }

    #[test]
    fn test_add_bias_row_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let (rows, d) = (6usize, 16usize);
        let x: Vec<f32> = lcg_vec(rows * d);
        let bias: Vec<f32> = lcg_vec(d).into_iter().map(|v| v * 0.5 + 0.25).collect();

        let mut cpu_x = x.clone();
        for r in 0..rows {
            for j in 0..d {
                cpu_x[r * d + j] += bias[j];
            }
        }

        let x_h = client.create_from_slice(f32::as_bytes(&x));
        let b_h = client.create_from_slice(f32::as_bytes(&bias));
        // SAFETY: bindings sized rows*d / d.
        unsafe {
            AddBiasRowCubeCL::launch::<ActiveRuntime>(&client, x_h.clone(), rows * d, b_h, d);
        }

        let got = f32::from_bytes(&client.read_one(x_h).unwrap()).to_vec();
        for i in 0..cpu_x.len() {
            assert_eq!(got[i], cpu_x[i], "add_bias_row mismatch at {i}");
        }
    }

    #[test]
    fn test_scale_relu_match_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let x = lcg_vec(64);
        let s = 0.7071f32;

        let mut cpu_x = x.clone();
        for v in cpu_x.iter_mut() {
            *v *= s;
        }
        let x_h = client.create_from_slice(f32::as_bytes(&x));
        // SAFETY: binding sized 64.
        unsafe {
            ScaleCubeCL::launch::<ActiveRuntime>(&client, x_h.clone(), x.len(), s);
        }
        let got = f32::from_bytes(&client.read_one(x_h.clone()).unwrap()).to_vec();
        for i in 0..x.len() {
            assert_eq!(got[i], cpu_x[i], "scale mismatch at {i}");
        }

        for v in cpu_x.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        // SAFETY: binding sized 64.
        unsafe {
            ReluCubeCL::launch::<ActiveRuntime>(&client, x_h.clone(), x.len());
        }
        let got = f32::from_bytes(&client.read_one(x_h).unwrap()).to_vec();
        for i in 0..x.len() {
            assert_eq!(got[i], cpu_x[i], "relu mismatch at {i}");
        }
    }

    #[test]
    fn test_gelu_erf_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Spread beyond ±1 so the erf tail regions are exercised.
        let x: Vec<f32> = lcg_vec(128).into_iter().map(|v| v * 3.0).collect();
        let cpu_x: Vec<f32> = x.iter().map(|v| gelu_ref(*v)).collect();

        let x_h = client.create_from_slice(f32::as_bytes(&x));
        // SAFETY: binding sized 128.
        unsafe {
            GeluErfCubeCL::launch::<ActiveRuntime>(&client, x_h.clone(), x.len());
        }
        let got = f32::from_bytes(&client.read_one(x_h).unwrap()).to_vec();
        let mut max = 0.0f32;
        for i in 0..x.len() {
            max = max.max((got[i] - cpu_x[i]).abs());
        }
        // cubecl f32::erf (Metal/WGSL intrinsic) vs libm::erff — a few ulp
        // on values up to |gelu| ~ 3; 2e-5 is generous and still catches a
        // wrong-form gelu (tanh approx differs by up to ~1e-2, sigmoid approx
        // by up to ~2e-2).
        assert!(max <= 2e-5, "gelu_erf drift {max:.4e}");
        println!("gelu_erf vs libm::erff: max abs drift {max:.3e}");
    }

    #[test]
    fn test_glu_gelu_gate_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let (rows, i_sz) = (8usize, 24usize);
        let fused: Vec<f32> = lcg_vec(rows * 2 * i_sz).into_iter().map(|v| v * 2.0).collect();
        let mut cpu_out = vec![0f32; rows * i_sz];
        for r in 0..rows {
            for j in 0..i_sz {
                let v = fused[r * 2 * i_sz + j];
                let gate = fused[r * 2 * i_sz + i_sz + j];
                cpu_out[r * i_sz + j] = gelu_ref(v) * gate;
            }
        }

        let f_h = client.create_from_slice(f32::as_bytes(&fused));
        let o_h = client.empty(rows * i_sz * core::mem::size_of::<f32>());
        // SAFETY: bindings sized rows*2*i_sz / rows*i_sz.
        unsafe {
            GluGeluGateCubeCL::launch::<ActiveRuntime>(&client, f_h, o_h.clone(), rows, i_sz);
        }
        let got = f32::from_bytes(&client.read_one(o_h).unwrap()).to_vec();
        assert_eq!(got.len(), rows * i_sz);
        let mut max = 0.0f32;
        for i in 0..got.len() {
            max = max.max((got[i] - cpu_out[i]).abs());
        }
        assert!(max <= 2e-5, "glu_gelu_gate drift {max:.4e}");
        println!("glu_gelu_gate vs cpu: max abs drift {max:.3e}");
    }
}
