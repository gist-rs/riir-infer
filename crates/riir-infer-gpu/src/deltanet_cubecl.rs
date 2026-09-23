//! CubeCL DeltaNet linear recurrence kernel (Plan 182, Phase 2).
//!
//! Implements the DeltaNet recurrent state update and read for decode-time inference:
//! ```text
//! S_t = diag(β) * S_{t-1} + v ⊗ k^T   (state update: rank-1 outer product)
//! y = S_t * q                            (state read: matrix-vector product)
//! ```
//!
//! # Architecture (Qwen 3.5-0.8B)
//!
//! - 16 linear attention heads, head_dim = 128
//! - Each head maintains a [128 × 128] recurrent state matrix (64 KB per head)
//! - State lives in GPU global memory (persistent across decode steps)
//!
//! # Dispatch
//!
//! | Kernel | CubeDim | CubeCount | Responsibility |
//! |--------|---------|-----------|----------------|
//! | `deltanet_recurrence_f32` | `new_1d(128)` | `(n_head, 1, 1)` | 1 workgroup/head |
//! | `deltanet_conv1d_f32` | `new_1d(conv_dim)` | `(1, 1, 1)` | elementwise |
//! | `deltanet_gating_f32` | `new_1d(n)` | `(ceil(n/128), 1, 1)` | elementwise |

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(all(
    feature = "cubecl_runtime",
    any(feature = "deltanet_recurrence_rowpar", feature = "ternary_deltanet_chunked_prefill")
))]
#[allow(unused_imports, reason = "Plane trait needed for plane_sum() resolution")]
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::ActiveRuntime;

// ---------------------------------------------------------------------------
// DeltaNet recurrence decode kernel (T5)
// ---------------------------------------------------------------------------

/// CubeCL DeltaNet gated recurrence kernel (decode mode).
///
/// Implements the full Gated DeltaNet step per head:
/// 1. Decay: `S *= g`
/// 2. Retrieve: `kv_mem = S * k`
/// 3. Delta: `delta = β * (v - kv_mem)`
/// 4. Update: `S += delta ⊗ k^T`
/// 5. Read: `output = (S * q) / sqrt(d)`
///
/// ## Dispatch
///
/// - `CubeDim::new_1d(head_dim)` — one thread per column of state matrix
/// - `CubeCount::Static(n_head, 1, 1)` — one workgroup per head
///
/// ## Parameter Layout
///
/// - `qkv`: combined query + key + value `[3 * n_head * head_dim]`
/// - `params`: `[head_dim_f32, n_head_f32]` (Issue 604 T1 — n_head moved from
///   hardcoded constant to runtime param so the kernel serves any model)
/// - `state`: persistent recurrent state `[n_head * head_dim * head_dim]` (read-write)
/// - `beta`: per-head beta values `[n_head]` (GPU-resident)
/// - `decay`: per-head decay values `[n_head]` (GPU-resident)
/// - `output`: read result `[n_head * head_dim]` (write-only)
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_recurrence_f32(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    params: &[f32],
    state: &mut [f32],
    output: &mut [f32],
) {
    // ── Constants ──
    // n_head was hardcoded to 16 (Qwen3.5-0.8B) — parameterized in Issue 604 T1
    // so the same kernel serves Bonsai-27B (n_v_heads=48 after Q/K expansion).
    // Q/K are expanded to n_v_heads BEFORE reaching this kernel (see
    // `expand_heads_into` in the CPU reference forward); the qkv layout is
    // [q (n_head*hd), k (n_head*hd), v (n_head*hd)] with a single n_head.
    let n_head = params[1usize] as u32;
    let cube_size = 128u32; // must match CubeDim::new_1d(128)
    let head_dim_f32 = params[0usize];
    let head_dim = head_dim_f32 as u32;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let col = UNIT_POS; // thread index within workgroup (0..head_dim-1)

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    let head = head_idx;
    let head_off_q = (head * head_dim) as usize;
    let head_off_k = (n_head * head_dim + head * head_dim) as usize;
    let head_off_v = (2 * n_head * head_dim + head * head_dim) as usize;
    let state_off = (head * head_dim * head_dim) as usize;
    let col_us = col as usize;
    let head_dim_us = head_dim as usize;

    let beta_val = beta[head as usize];
    let decay_val = decay[head as usize];

    // Load k[col] and q[col] for this thread
    let k_col = qkv[head_off_k + col_us];
    let q_col = qkv[head_off_q + col_us];

    // Shared memory for workgroup-wide dot-product reductions (Issue 610).
    // plane_sum only reduces within a 32-thread subgroup (plane), not across
    // the full 128-thread workgroup. On Metal (plane_dim=32), that captures
    // only ~1/4 of the dot product. This shared-memory tree reduction fixes it.
    // Size matches cube_size (128) — the kernel is dispatched with CubeDim::new_1d(128).
    let mut smem = Shared::<[f32]>::new_slice(128usize);

    // Step 1: Decay old state
    for row in 0..head_dim_us {
        let idx = state_off + row * head_dim_us + col_us;
        state[idx] = state[idx] * decay_val;
    }

    // Step 2-4: Retrieve + compute delta + update state
    // For each row, each thread contributes s[row, col] * k[col]
    // Full-workgroup reduction via shared memory + sync_cube (Issue 610 fix).
    for row in 0..head_dim_us {
        let idx = state_off + row * head_dim_us + col_us;
        let s_val = state[idx];
        let partial = s_val * k_col;

        // Workgroup-wide reduction: write partial, sync, tree-reduce
        smem[col_us] = partial;
        sync_cube();

        // Tree reduction: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
        if col < 64u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 64];
        }
        sync_cube();
        if col < 32u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 32];
        }
        sync_cube();
        if col < 16u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 16];
        }
        sync_cube();
        if col < 8u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 8];
        }
        sync_cube();
        if col < 4u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 4];
        }
        sync_cube();
        if col < 2u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 2];
        }
        sync_cube();
        if col < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();

        let kv_mem_row = smem[0usize];

        // Compute delta in thread 0, broadcast via smem[0].
        // Issue 616: the original port (Issue 610) did the ENTIRE state update
        // (S[row, :] += k[:] * delta_row) serially in thread 0 — head_dim serial
        // global memory writes per row, with all other threads idle. This was
        // the single biggest bottleneck in both the cudarc and CubeCL forwards.
        //
        // Fix: thread 0 computes delta, broadcasts via smem[0], then ALL threads
        // update their own column in parallel — 1 global write per thread
        // instead of head_dim serial writes.
        if col == 0u32 {
            let v_row = qkv[head_off_v + row];
            smem[0usize] = beta_val * (v_row - kv_mem_row);
        }
        sync_cube();

        // Parallel state update: S[row, col] += k[col] * delta_row
        let delta_row = smem[0usize];
        state[state_off + row * head_dim_us + col_us] =
            state[state_off + row * head_dim_us + col_us] + k_col * delta_row;

        // Sync before next row (all threads wrote state[row, :])
        sync_cube();
    }

    // Step 5 (read): output[h, row] = Σ_c S[h, row, c] * q[h, c] / sqrt(d)
    let scale = f32::new(1.0f32) / head_dim_f32.sqrt();

    for row in 0..head_dim_us {
        let idx = state_off + row * head_dim_us + col_us;
        let s_val = state[idx];
        let partial = s_val * q_col;

        // Workgroup-wide reduction (same pattern as step 2-4)
        smem[col_us] = partial;
        sync_cube();

        if col < 64u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 64];
        }
        sync_cube();
        if col < 32u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 32];
        }
        sync_cube();
        if col < 16u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 16];
        }
        sync_cube();
        if col < 8u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 8];
        }
        sync_cube();
        if col < 4u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 4];
        }
        sync_cube();
        if col < 2u32 {
            smem[col_us] = smem[col_us] + smem[col_us + 2];
        }
        sync_cube();
        if col < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();

        let dot = smem[0usize];

        if col == 0u32 {
            output[(head * head_dim + row as u32) as usize] = dot * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaNet recurrence — row-parallel, register-blocked (Issue 619)
// ---------------------------------------------------------------------------

/// Row-parallel, register-blocked DeltaNet recurrence (Issue 619).
///
/// Metal-native distillation of the CUDA win in Issue 617. The rows of the
/// per-head state matrix are **independent**: step 2 for row `r` reads only
/// `S[r, :]`, step 4 writes only `S[r, :]`, step 5 reads only `S[r, :]`. The
/// legacy [`deltanet_recurrence_f32`] walks them in a serial loop inside one
/// workgroup per head; this kernel gives each row its own cube.
///
/// ## Why this is a Metal win (and why it is NOT just the CUDA fix)
///
/// On Apple silicon a `threadgroup_barrier` costs ~2 cycles, and swapping SIMD
/// shuffles for barrier-mediated threadgroup memory costs only ~2.2%. So barrier
/// *count* — the dominant term in the CUDA fix — is a weak lever here. The two
/// levers that do matter on Apple are:
///
/// 1. **Threadgroup memory traffic.** Scattered threadgroup access costs up to
///    3.2× effective bandwidth. The legacy kernel runs a 7-level strided smem
///    tree reduction per row — 1792 scattered smem round-trips per head. This
///    kernel uses **zero threadgroup memory**: one cube is exactly one plane, so
///    `plane_sum` reduces the whole dot product natively in registers. That is
///    the "Tier 1" (register + shuffle) regime Apple's hardware rewards.
///
/// 2. **Occupancy.** Apple hides memory latency by cycling resident SIMD-groups.
///    The legacy dispatch is `n_head` cubes (48 for Bonsai-27B → 192 SIMD-groups).
///    This dispatch is `n_head × head_dim` cubes (6144 → 6144 SIMD-groups), a 32×
///    increase in latency-hiding headroom.
///
/// ## The dominant win: register blocking
///
/// Because a cube now *owns* its row exclusively, the row is loaded into
/// registers once and every step operates on registers:
///
/// | | legacy | this kernel |
/// |---|---|---|
/// | global passes over `S` | 4 read + 2 write | **1 read + 1 write** |
/// | threadgroup bytes | 512 B/head | **0** |
/// | barriers per head | ~2304 | **0** |
///
/// This kernel is memory-bound, so cutting 6 passes to 2 is a ~3× reduction in
/// the quantity that actually gates it.
///
/// ## Dispatch contract
///
/// - `CubeCount::Static(n_head, head_dim, 1)` — one cube per (head, row)
/// - `CubeDim::new_1d(32)` — one cube is exactly one plane
/// - Requires `head_dim == 4 * plane_dim` (i.e. 128 on Metal and CUDA, where
///   `plane_dim == 32`). The launcher enforces this and callers must fall back to
///   [`deltanet_recurrence_f32`] otherwise. Bonsai-27B and Qwen3.5-0.8B both use
///   `head_dim = 128`.
///
/// Numerics: `plane_sum` reduces in a different order than the smem tree, so
/// results are FP-equivalent, not bit-identical (the CUDA counterpart measured
/// ~9e-5 max diff, well inside the existing 1e-4 CPU tolerance).
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
#[cube(launch_unchecked)]
fn deltanet_recurrence_f32_rowpar(
    qkv: &[f32],
    beta: &[f32],
    decay: &[f32],
    params: &[f32],
    state: &mut [f32],
    output: &mut [f32],
) {
    let head_dim_f32 = params[0usize];
    let head_dim = head_dim_f32 as u32;
    let n_head = params[1usize] as u32;

    // One cube per (head, row). Rows are independent — see the doc comment.
    let head = CUBE_POS_X;
    let row = CUBE_POS_Y;

    if head >= n_head {
        terminate!();
    }
    if row >= head_dim {
        terminate!();
    }

    let lane = UNIT_POS_PLANE;
    let stride = PLANE_DIM;

    let head_off_q = (head * head_dim) as usize;
    let head_off_k = (n_head * head_dim + head * head_dim) as usize;
    let head_off_v = (2u32 * n_head * head_dim + head * head_dim) as usize;
    // Base of THIS row within the [head_dim × head_dim] state block for this head.
    let row_off = (head * head_dim * head_dim + row * head_dim) as usize;

    let beta_val = beta[head as usize];
    let decay_val = decay[head as usize];

    // Column indices owned by this lane: lane, lane+32, lane+64, lane+96.
    let c0 = lane as usize;
    let c1 = (lane + stride) as usize;
    let c2 = (lane + 2u32 * stride) as usize;
    let c3 = (lane + 3u32 * stride) as usize;

    // ── Single global read pass: row of S, plus this lane's k and q slices ──
    let mut s0 = state[row_off + c0];
    let mut s1 = state[row_off + c1];
    let mut s2 = state[row_off + c2];
    let mut s3 = state[row_off + c3];

    let k0 = qkv[head_off_k + c0];
    let k1 = qkv[head_off_k + c1];
    let k2 = qkv[head_off_k + c2];
    let k3 = qkv[head_off_k + c3];

    // ── Step 1: decay (registers) ──
    s0 *= decay_val;
    s1 *= decay_val;
    s2 *= decay_val;
    s3 *= decay_val;

    // ── Step 2: retrieve kv_mem = Σ_c S[row, c] · k[c] ──
    // The cube is exactly one plane, so plane_sum covers the FULL dot product.
    // (The Issue 610 smem tree existed only because the cube was 128 threads,
    // i.e. 4 planes, and plane_sum caught just 1/4 of it.)
    let acc_k = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
    let kv_mem_row = plane_sum(acc_k);

    // ── Step 3: delta. plane_sum broadcasts, so every lane has it — no barrier,
    // no thread-0 bottleneck, no smem hand-off. ──
    let v_row = qkv[head_off_v + row as usize];
    let delta_row = beta_val * (v_row - kv_mem_row);

    // ── Step 4: rank-1 update S[row, :] += k[:] · delta (registers) ──
    s0 += k0 * delta_row;
    s1 += k1 * delta_row;
    s2 += k2 * delta_row;
    s3 += k3 * delta_row;

    // ── Step 5: read out[row] = (Σ_c S[row, c] · q[c]) / sqrt(d) ──
    let q0 = qkv[head_off_q + c0];
    let q1 = qkv[head_off_q + c1];
    let q2 = qkv[head_off_q + c2];
    let q3 = qkv[head_off_q + c3];

    let acc_q = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    let dot = plane_sum(acc_q);

    // ── Single global write pass: the updated row ──
    state[row_off + c0] = s0;
    state[row_off + c1] = s1;
    state[row_off + c2] = s2;
    state[row_off + c3] = s3;

    if lane == 0u32 {
        let scale = f32::new(1.0f32) / head_dim_f32.sqrt();
        output[(head * head_dim + row) as usize] = dot * scale;
    }
}

// ---------------------------------------------------------------------------
// DeltaNet conv1d kernel (T6)
// ---------------------------------------------------------------------------

/// CubeCL depthwise conv1d update kernel for DeltaNet q/k/v preprocessing.
///
/// Implements causal conv1d with SiLU activation for a single new token:
/// 1. Shift conv state left by 1, append new input
/// 2. Apply depthwise convolution: `out[ch] = Σ_k state[ch,k] * weight[ch,k]`
/// 3. SiLU activation: `out[ch] = out[ch] * sigmoid(out[ch])`
///
/// ## Parameter Layout
///
/// - `input`: current token input `[conv_dim]` (modified in-place with output)
/// - `conv_weight`: depthwise weights `[conv_dim, kernel_size]`
/// - `conv_state`: sliding window `[conv_dim, kernel_size]` (modified in-place)
/// - `params`: `[conv_dim_f32, kernel_size_f32]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_conv1d_f32(
    input: &mut [f32],
    conv_weight: &[f32],
    conv_state: &mut [f32],
    params: &[f32],
) {
    let conv_dim = params[0] as u32;
    let kernel_size = params[1] as u32;
    let ch = ABSOLUTE_POS;

    if ch >= conv_dim as usize {
        terminate!();
    }

    let _conv_dim_us = conv_dim as usize;
    let kernel_size_us = kernel_size as usize;

    // Step 1: Shift conv_state left by 1, append input
    let state_off = ch * kernel_size_us;
    for k in 0..kernel_size_us - 1 {
        conv_state[state_off + k] = conv_state[state_off + k + 1];
    }
    conv_state[state_off + kernel_size_us - 1] = input[ch];

    // Step 2: Depthwise convolution
    let weight_off = ch * kernel_size_us;
    let mut sum = f32::new(0.0f32);
    for k in 0..kernel_size_us {
        sum += conv_state[state_off + k] * conv_weight[weight_off + k];
    }

    // Step 3: SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    let neg_sum = f32::new(0.0f32) - sum;
    let exp_neg = neg_sum.exp();
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + exp_neg);
    input[ch] = sum * sig;
}

// ---------------------------------------------------------------------------
// DeltaNet gating kernel — SwiGLU-style (T7)
// ---------------------------------------------------------------------------

/// CubeCL elementwise SwiGLU gating for DeltaNet recurrent output.
///
/// Computes `output[i] = gate[i] * sigmoid(gate[i]) * up[i]`
/// which is `silu(gate) * up`.
///
/// ## Parameter Layout
///
/// - `gate`: gate values `[n]`
/// - `up`: up values `[n]`
/// - `output`: result `[n]`
/// - `params`: `[n_elements_f32]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_gating_f32(
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let n = params[0] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= n {
        terminate!();
    }

    let g = gate[idx];
    // SiLU: g * sigmoid(g)
    let neg_g = f32::new(0.0f32) - g;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
    output[idx] = g * sig * up[idx];
}

// ---------------------------------------------------------------------------
// Safe launcher wrappers
// ---------------------------------------------------------------------------

/// Launcher for the DeltaNet recurrence kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetRecurrenceCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetRecurrenceCubeCL {
    /// The kernel's head→workgroup mapping (`cube_size = 128`) and its
    /// shared-memory reduction tree (128 slots, 7 fixed levels) are HARDCODED
    /// to head_dim == 128. A different `head_dim` silently misassigns heads
    /// (head_dim=64 double-assigns head 0 and skips half the heads) and reads
    /// uninitialized smem slots (Issue 673 Bug B).
    ///
    /// NOTE: [`DeltanetRecurrenceRowParCubeCL::supports`] has the SAME
    /// restriction, so a head_dim≠128 model currently has NO working recurrence
    /// kernel — the rowpar doc's "fall back to DeltanetRecurrenceCubeCL"
    /// contract only holds for head_dim == 128. Extending to other head_dims
    /// requires a runtime-depth reduction tree here (file an issue first).
    #[inline]
    #[must_use]
    pub fn supports(head_dim: usize) -> bool {
        head_dim == 128
    }

    /// Launch the DeltaNet recurrence kernel.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `qkv` has `3 * n_head * head_dim` f32 elements
    /// - `state` has `n_head * head_dim * head_dim` f32 elements
    /// - `output` has `n_head * head_dim` f32 bytes allocated
    /// - `betas` and `decays` each have `n_head` elements
#[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_handle: Handle,
        state_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        head_dim: usize,
        betas: &[f32],
        decays: &[f32],
    ) {
        // Upload beta/decay as separate GPU buffers
        let beta_handle = client.create_from_slice(f32::as_bytes(betas));
        let decay_handle = client.create_from_slice(f32::as_bytes(decays));

        // SAFETY: caller guarantees buffer sizes.
        unsafe {
            Self::launch_with_gpu_handles(
                client,
                qkv_handle,
                beta_handle,
                decay_handle,
                state_handle,
                output_handle,
                n_head,
                head_dim,
            );
        }
    }

    /// Launch the DeltaNet recurrence kernel with GPU-resident beta/decay handles.
    ///
    /// This variant avoids any CPU sync — beta/decay are already on GPU.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `qkv` has `3 * n_head * head_dim` f32 elements
    /// - `beta_handle`, `decay_handle` each have `n_head` f32 elements
    /// - `state` has `n_head * head_dim * head_dim` f32 elements
    /// - `output` has `n_head * head_dim` f32 bytes allocated
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch_with_gpu_handles<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_handle: Handle,
        beta_handle: Handle,
        decay_handle: Handle,
        state_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        head_dim: usize,
    ) {
        // Issue 673 Bug B: the kernel's cube_size + smem tree are hardcoded to
        // 128 (see `supports`). Guard loudly instead of computing silently
        // wrong heads — debug_assert mirrors the rowpar launcher pattern.
        debug_assert!(
            Self::supports(head_dim),
            "legacy recurrence requires head_dim == 128 (hardcoded cube_size + smem tree), got {head_dim}"
        );

        // params: [head_dim, n_head]. n_head is now passed at launch time
        // (Issue 604 T1) — previously hardcoded to 16 in the kernel.
        let params: [f32; 2] = [head_dim as f32, n_head as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let qkv_len = 3 * n_head * head_dim;
        let state_len = n_head * head_dim * head_dim;
        let output_len = n_head * head_dim;

        unsafe {
            deltanet_recurrence_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(qkv_handle, qkv_len),
                BufferArg::from_raw_parts(beta_handle, n_head),
                BufferArg::from_raw_parts(decay_handle, n_head),
                BufferArg::from_raw_parts(params_handle, 2),
                BufferArg::from_raw_parts(state_handle, state_len),
                BufferArg::from_raw_parts(output_handle, output_len),
            );
        }
    }
}

/// Launcher for the row-parallel, register-blocked recurrence kernel (Issue 619).
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
pub struct DeltanetRecurrenceRowParCubeCL;

#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
impl DeltanetRecurrenceRowParCubeCL {
    /// Plane width assumed by the kernel. 32 on Metal and CUDA.
    pub const PLANE: usize = 32;
    /// Columns each lane holds in registers (`head_dim / PLANE`).
    pub const COLS_PER_LANE: usize = 4;

    /// Whether the row-parallel kernel can serve this `head_dim`.
    ///
    /// The kernel is register-blocked with a fixed unroll of
    /// [`Self::COLS_PER_LANE`], so it requires
    /// `head_dim == PLANE * COLS_PER_LANE` (128). Callers MUST consult this and
    /// fall back to [`DeltanetRecurrenceCubeCL`] when it returns `false` — note
    /// the legacy kernel carries the SAME 128 restriction (Issue 673 Bug B), so
    /// head_dim≠128 currently has no working recurrence kernel.
    #[inline]
    #[must_use]
    pub fn supports(head_dim: usize) -> bool {
        head_dim == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Launch the row-parallel recurrence kernel with GPU-resident beta/decay.
    ///
    /// Drop-in replacement for
    /// [`DeltanetRecurrenceCubeCL::launch_with_gpu_handles`] — identical buffer
    /// contract, identical semantics, FP-equivalent output.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `head_dim` satisfies [`Self::supports`]
    /// - `qkv` has `3 * n_head * head_dim` f32 elements
    /// - `beta_handle`, `decay_handle` each have `n_head` f32 elements
    /// - `state` has `n_head * head_dim * head_dim` f32 elements
    /// - `output` has `n_head * head_dim` f32 elements
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch_with_gpu_handles<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_handle: Handle,
        beta_handle: Handle,
        decay_handle: Handle,
        state_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        head_dim: usize,
    ) {
        debug_assert!(
            Self::supports(head_dim),
            "row-parallel recurrence requires head_dim == {}, got {head_dim}; \
             caller must fall back to DeltanetRecurrenceCubeCL",
            Self::PLANE * Self::COLS_PER_LANE
        );

        let params: [f32; 2] = [head_dim as f32, n_head as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let qkv_len = 3 * n_head * head_dim;
        let state_len = n_head * head_dim * head_dim;
        let output_len = n_head * head_dim;

        unsafe {
            deltanet_recurrence_f32_rowpar::launch_unchecked::<R>(
                client,
                // One cube per (head, row) — 32× the cubes of the legacy dispatch.
                CubeCount::Static(n_head as u32, head_dim as u32, 1),
                // One cube == one plane, so plane_sum reduces the full dot product.
                CubeDim::new_1d(Self::PLANE as u32),
                BufferArg::from_raw_parts(qkv_handle, qkv_len),
                BufferArg::from_raw_parts(beta_handle, n_head),
                BufferArg::from_raw_parts(decay_handle, n_head),
                BufferArg::from_raw_parts(params_handle, 2),
                BufferArg::from_raw_parts(state_handle, state_len),
                BufferArg::from_raw_parts(output_handle, output_len),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaNet recurrence — multi-token row-parallel (Issue 734 T4)
// ---------------------------------------------------------------------------

/// Multi-token row-parallel, register-blocked DeltaNet recurrence (Issue 734 T4).
///
/// The prefill bottleneck on the 4090 is NOT kernel work but the GPU-side
/// inter-kernel gap of the P dependent per-token dispatches (Issue 734 T1a:
/// ~34 µs × 2048 dispatches/layer × 48 layers ≈ 3.3 s of a 8.7 s block). This
/// kernel keeps the EXACT per-token arithmetic of
/// [`deltanet_recurrence_f32_rowpar`] (the shipping sequential path — same
/// lane-owned columns, same `plane_sum` reductions, same op order →
/// bit-identical output) but loops over ALL `p` tokens INSIDE one dispatch:
///
/// ```text
/// one cube per (head, row) · 32 lanes · lane owns 4 columns in registers
/// for t in 0..p:  load k/q/v + β/α → decay → plane_sum(mem) → δ → update
///                 → plane_sum(readout) → out[t]
/// ```
///
/// The state row lives in 4 registers per lane for the WHOLE layer — read
/// once at kernel start, written once at end. Per-token traffic is the k/q/v
/// vector (L2-broadcast across the 128 row-cubes of the head) + one v element.
///
/// Why not the chunkwise-parallel solve (Bench 662's algorithm, shipped in
/// [`crate::deltanet_delta_rule_chunked`])? It is algebraically exact and
/// kernel-tested to ~1e-7 on synthetic data, but on REAL model data its
/// different summation order diverges from the sequential path at the same
/// ~4e-3/layer class the Issue 721 tree-verify kernels measured — through 48
/// layers that compounds to O(1) logit differences (measured max_rel 1.6–2.8,
/// argmax stable, e2e G1 FAIL at the 1e-2 gate). The multi-token kernel has
/// ZERO structural divergence — it is the sequential path with the dispatch
/// gaps removed.
///
/// Same constraint as rowpar: `head_dim == 128` (PLANE=32, COLS_PER_LANE=4).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
#[cube(launch_unchecked)]
fn deltanet_recurrence_multi_token_f32(
    qkvx: &[f32],
    beta: &[f32],
    decay: &[f32],
    params: &[f32],
    state: &mut [f32],
    output: &mut [f32],
) {
    let head_dim_f32 = params[0usize];
    let head_dim = head_dim_f32 as u32;
    let n_head = params[1usize] as u32;
    let p = params[2usize] as u32;
    let v_dim = params[3usize] as u32; // n_head * head_dim

    // One cube per (head, row) — identical to the rowpar dispatch.
    let head = CUBE_POS_X;
    let row = CUBE_POS_Y;

    if head >= n_head {
        terminate!();
    }
    if row >= head_dim {
        terminate!();
    }

    let lane = UNIT_POS_PLANE;
    let stride = PLANE_DIM;

    let qkvx_stride = 3u32 * v_dim; // per-token qkvx row
    let head_off = head * head_dim; // h*d within a q/k/v block
    let k_base = v_dim + head_off; // K block base within a token row
    let v_base = 2u32 * v_dim + head_off;
    // Base of THIS row within the [head_dim × head_dim] state block for this head.
    let row_off = (head * head_dim * head_dim + row * head_dim) as usize;

    // Column indices owned by this lane: lane, lane+32, lane+64, lane+96.
    let c0 = lane as usize;
    let c1 = (lane + stride) as usize;
    let c2 = (lane + 2u32 * stride) as usize;
    let c3 = (lane + 3u32 * stride) as usize;

    // ── Load the state row into registers ONCE — held for all p tokens ──
    let mut s0 = state[row_off + c0];
    let mut s1 = state[row_off + c1];
    let mut s2 = state[row_off + c2];
    let mut s3 = state[row_off + c3];

    let scale = f32::new(1.0f32) / head_dim_f32.sqrt();

    let mut t = 0u32;
    while t < p {
        let t_row = (t * qkvx_stride) as usize;
        let scalar_idx = (t * n_head + head) as usize;
        let beta_val = beta[scalar_idx];
        let decay_val = decay[scalar_idx];

        let k0 = qkvx[t_row + k_base as usize + c0];
        let k1 = qkvx[t_row + k_base as usize + c1];
        let k2 = qkvx[t_row + k_base as usize + c2];
        let k3 = qkvx[t_row + k_base as usize + c3];

        // Step 1: decay (registers).
        s0 *= decay_val;
        s1 *= decay_val;
        s2 *= decay_val;
        s3 *= decay_val;

        // Step 2: retrieve kv_mem = Σ_c S[row, c] · k[c] (plane covers the row).
        let acc_k = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        let kv_mem_row = plane_sum(acc_k);

        // Step 3: delta.
        let v_row = qkvx[t_row + v_base as usize + row as usize];
        let delta_row = beta_val * (v_row - kv_mem_row);

        // Step 4: rank-1 update (registers).
        s0 += k0 * delta_row;
        s1 += k1 * delta_row;
        s2 += k2 * delta_row;
        s3 += k3 * delta_row;

        // Step 5: read out[row] = (Σ_c S[row, c] · q[c]) / sqrt(d).
        let q0 = qkvx[t_row + head_off as usize + c0];
        let q1 = qkvx[t_row + head_off as usize + c1];
        let q2 = qkvx[t_row + head_off as usize + c2];
        let q3 = qkvx[t_row + head_off as usize + c3];
        let acc_q = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        let dot = plane_sum(acc_q);

        if lane == 0u32 {
            output[(t * v_dim + head * head_dim + row) as usize] = dot * scale;
        }

        t += 1;
    }

    // ── Single global write pass: the updated row ──
    state[row_off + c0] = s0;
    state[row_off + c1] = s1;
    state[row_off + c2] = s2;
    state[row_off + c3] = s3;
}

/// Launcher for the multi-token row-parallel recurrence kernel (Issue 734 T4).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub struct DeltanetRecurrenceMultiTokenCubeCL;

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
impl DeltanetRecurrenceMultiTokenCubeCL {
    /// Plane width assumed by the kernel. 32 on Metal and CUDA.
    pub const PLANE: usize = 32;
    /// Columns each lane holds in registers (`head_dim / PLANE`).
    pub const COLS_PER_LANE: usize = 4;

    /// Whether the multi-token kernel can serve this `head_dim` (128 only —
    /// same register-blocked unroll as [`DeltanetRecurrenceRowParCubeCL]).
    #[inline]
    #[must_use]
    pub fn supports(head_dim: usize) -> bool {
        head_dim == Self::PLANE * Self::COLS_PER_LANE
    }

    /// Process ALL `p` tokens of a layer in ONE dispatch — bit-identical to `p`
    /// sequential [`DeltanetRecurrenceRowParCubeCL`] launches (same per-token
    /// arithmetic in the same order; only the token addressing differs).
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `head_dim` satisfies [`Self::supports`]
    /// - `qkvx` has `p * 3 * n_head * head_dim` f32 elements (token-major
    ///   `[q (v_dim) | k (v_dim) | v (v_dim)]` per token — the prefill `qkvx_b`)
    /// - `beta`, `decay` each have `p * n_head` f32 elements (token-major)
    /// - `state` has `n_head * head_dim * head_dim` f32 elements (updated in place)
    /// - `output` has `p * n_head * head_dim` f32 elements (token-major)
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch_with_gpu_handles<R: Runtime>(
        client: &ComputeClient<R>,
        qkvx_handle: Handle,
        beta_handle: Handle,
        decay_handle: Handle,
        state_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        head_dim: usize,
        p: usize,
    ) {
        debug_assert!(
            Self::supports(head_dim),
            "multi-token recurrence requires head_dim == {}, got {head_dim}",
            Self::PLANE * Self::COLS_PER_LANE
        );

        let v_dim = n_head * head_dim;
        let params: [f32; 4] = [head_dim as f32, n_head as f32, p as f32, v_dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let qkvx_len = p * 3 * v_dim;
        let state_len = n_head * head_dim * head_dim;
        let output_len = p * v_dim;

        unsafe {
            deltanet_recurrence_multi_token_f32::launch_unchecked::<R>(
                client,
                // One cube per (head, row) — identical to the rowpar dispatch.
                CubeCount::Static(n_head as u32, head_dim as u32, 1),
                // One cube == one plane, so plane_sum reduces the full dot product.
                CubeDim::new_1d(Self::PLANE as u32),
                BufferArg::from_raw_parts(qkvx_handle, qkvx_len),
                BufferArg::from_raw_parts(beta_handle, p * n_head),
                BufferArg::from_raw_parts(decay_handle, p * n_head),
                BufferArg::from_raw_parts(params_handle, 4),
                BufferArg::from_raw_parts(state_handle, state_len),
                BufferArg::from_raw_parts(output_handle, output_len),
            );
        }
    }
}

/// Launcher for the DeltaNet conv1d kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetConv1dCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetConv1dCubeCL {
    /// Launch the depthwise conv1d + SiLU kernel.
    ///
    /// # Safety
    ///
    /// Caller must ensure all handles have correct sizes.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_handle: Handle,
        conv_weight_handle: Handle,
        conv_state_handle: Handle,
        conv_dim: usize,
        kernel_size: usize,
    ) {
        let params: [f32; 2] = [conv_dim as f32, kernel_size as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let total_weight = conv_dim * kernel_size;
        let total_state = conv_dim * kernel_size;

        // Use fixed workgroup size (128) with multiple workgroups to stay
        // within Metal's 1024 thread/workgroup limit. The kernel uses
        // ABSOLUTE_POS so multi-workgroup dispatch is correct.
        let wg_size = 128u32;
        let num_wg = (conv_dim as u32).div_ceil(wg_size).max(1);

        unsafe {
            deltanet_conv1d_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(input_handle, conv_dim),
                BufferArg::from_raw_parts(conv_weight_handle, total_weight),
                BufferArg::from_raw_parts(conv_state_handle, total_state),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Launcher for the DeltaNet SwiGLU gating kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetGatingCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetGatingCubeCL {
    /// Launch the SwiGLU gating kernel.
    ///
    /// # Safety
    ///
    /// Caller must ensure all handles have `n` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        gate_handle: Handle,
        up_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        // Metal grid guard (Issue 730): n/128 workgroups exceeds the 65535
        // x-cap from n >= 8.4M (p * mlp at p=482 with Bonsai's mlp 17408;
        // 278528 wg at p=2048). Pure elementwise — chunk + slice + adjust
        // the bound. Bit-identical.
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let elems_per_chunk = (MAX_WG_X as usize) * wg_size as usize;
        let mut e0 = 0usize;
        while e0 < n {
            let ec = elems_per_chunk.min(n - e0);
            let params: [f32; 1] = [ec as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (ec as u32).div_ceil(wg_size).max(1);
            unsafe {
                deltanet_gating_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(gate_handle.clone().offset_start((e0 * 4) as u64), ec),
                    BufferArg::from_raw_parts(up_handle.clone().offset_start((e0 * 4) as u64), ec),
                    BufferArg::from_raw_parts(output_handle.clone().offset_start((e0 * 4) as u64), ec),
                    BufferArg::from_raw_parts(params_handle, 1),
                );
            }
            e0 += ec;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused SwiGLU from concatenated gate_up buffer (Issue 642 F2)
// ---------------------------------------------------------------------------

/// CubeCL kernel: SwiGLU gating reading gate and up from a single concatenated buffer.
///
/// Computes `output[i] = gate_up[i] * sigmoid(gate_up[i]) * gate_up[mlp + i]`
/// which is `silu(gate) * up`, where `gate` = `gate_up[0..mlp]` and
/// `up` = `gate_up[mlp..2*mlp]`.
///
/// This replaces the two-dispatch FFN input path
/// (gate_proj GEMV + up_proj GEMV + DeltanetGating) with
/// (gate_up GEMV + DeltanetGatingConcat) — saving 1 dispatch per layer.
///
/// ## Parameter Layout
///
/// - `gate_up`: concatenated values `[2 * n]` (gate[0..n] then up[0..n])
/// - `output`: result `[n]`
/// - `params`: `[n_elements_f32]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_gating_concat_f32(
    gate_up: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let n = params[0] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= n {
        terminate!();
    }

    let g = gate_up[idx];
    // SiLU: g * sigmoid(g)
    let neg_g = f32::new(0.0f32) - g;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
    // up value lives at offset n in the concatenated buffer
    let u = gate_up[n + idx];
    output[idx] = g * sig * u;
}

/// Launcher for the fused SwiGLU gating kernel (concatenated gate_up buffer).
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetGatingConcatCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetGatingConcatCubeCL {
    /// Launch the fused SwiGLU gating kernel reading from a concatenated
    /// `[gate | up]` buffer of size `2 * n`.
    ///
    /// # Safety
    ///
    /// `gate_up_handle` must have `2 * n` f32 elements.
    /// `output_handle` must have `n` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        gate_up_handle: Handle,
        output_handle: Handle,
        n: usize,
    ) {
        let params: [f32; 1] = [n as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let wg_size = 128u32;
        let num_wg = (n as u32).div_ceil(wg_size);

        unsafe {
            deltanet_gating_concat_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(gate_up_handle, 2 * n),
                BufferArg::from_raw_parts(output_handle, n),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Beta/decay computation kernel (Issue 599 GPU-resident forward)
// ---------------------------------------------------------------------------

/// CubeCL kernel: compute per-head beta and decay from raw projections.
///
/// Mirrors the CPU code in `forward_deltanet_layer`:
/// ```text
/// beta[h]  = sigmoid(b_raw[h])
/// g        = a_log[h] * softplus(a_raw[h] + dt_bias[h])
/// decay[h] = exp(g)
/// ```
///
/// One thread per head. Dispatch: `CubeCount::Static(1,1,1)`, `CubeDim::new_1d(n_head)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_beta_decay_f32(
    a_raw: &[f32],
    b_raw: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    beta_out: &mut [f32],
    decay_out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let h = ABSOLUTE_POS;

    if h >= n_head {
        terminate!();
    }

    // beta = sigmoid(b_raw[h])
    let b_val = b_raw[h];
    let neg_b = f32::new(0.0f32) - b_val;
    let beta = f32::new(1.0f32) / (f32::new(1.0f32) + neg_b.exp());
    beta_out[h] = beta;

    // g = a_log[h] * softplus(a_raw[h] + dt_bias[h])
    let a_val = a_raw[h] + dt_bias[h];
    // softplus with clamping (matches CPU: x > 20 → x, x < -20 → 0)
    let sp = if a_val > f32::new(20.0f32) {
        a_val
    } else if a_val < f32::new(-20.0f32) {
        f32::new(0.0f32)
    } else {
        (f32::new(1.0f32) + a_val.exp()).ln()
    };
    let g = a_log[h] * sp;
    decay_out[h] = g.exp();
}

/// Launch beta/decay computation kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetBetaDecayCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetBetaDecayCubeCL {
    /// Compute per-head beta and decay from raw a/b projections.
    ///
    /// # Safety
    /// All handles must have at least `n_head` f32 elements.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_raw_handle: Handle,
        b_raw_handle: Handle,
        a_log_handle: Handle,
        dt_bias_handle: Handle,
        beta_out_handle: Handle,
        decay_out_handle: Handle,
        n_head: usize,
    ) {
        let params: [f32; 1] = [n_head as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        unsafe {
            deltanet_beta_decay_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n_head.max(1) as u32),
                BufferArg::from_raw_parts(a_raw_handle, n_head),
                BufferArg::from_raw_parts(b_raw_handle, n_head),
                BufferArg::from_raw_parts(a_log_handle, n_head),
                BufferArg::from_raw_parts(dt_bias_handle, n_head),
                BufferArg::from_raw_parts(beta_out_handle, n_head),
                BufferArg::from_raw_parts(decay_out_handle, n_head),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Batched beta/decay kernel (Issue 637 T5)
// ---------------------------------------------------------------------------

/// CubeCL kernel: `deltanet_beta_decay_f32` over `p` tokens in one dispatch.
///
/// The single-token kernel is pure per-head elementwise arithmetic with no
/// cross-head or cross-token coupling, so batching is just a wider index space.
/// `prefill` calls it once per token per DeltaNet layer — 128 x 48 = 6144
/// dispatches at P=128 — and each dispatch costs ~25 CPU allocations plus launch
/// latency (Issue 638). Collapsing them to 48 is the point.
///
/// ## Layout
///
/// - `a_raw`, `b_raw`, `beta_out`, `decay_out`: `[p, n_head]` row-major.
/// - `a_log`, `dt_bias`: `[n_head]` — per-head weights shared across tokens, so
///   they are indexed `i % n_head`.
/// - `params`: `[n_head, total]` where `total = p * n_head`.
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(total / 256), 1, 1)`, `CubeDim::new_1d(256)` — unlike
/// the single-token kernel, which uses one workgroup of `n_head` threads and
/// would exceed the max workgroup size at `p * n_head`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_beta_decay_batched_f32(
    a_raw: &[f32],
    b_raw: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    beta_out: &mut [f32],
    decay_out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let total = params[1usize] as usize;
    let i = ABSOLUTE_POS;

    if i >= total {
        terminate!();
    }

    // Per-head weights repeat every n_head entries; per-token buffers are flat.
    let h = i % n_head;

    // beta = sigmoid(b_raw[i])
    let b_val = b_raw[i];
    let neg_b = f32::new(0.0f32) - b_val;
    let beta = f32::new(1.0f32) / (f32::new(1.0f32) + neg_b.exp());
    beta_out[i] = beta;

    // g = a_log[h] * softplus(a_raw[i] + dt_bias[h])
    let a_val = a_raw[i] + dt_bias[h];
    let sp = if a_val > f32::new(20.0f32) {
        a_val
    } else if a_val < f32::new(-20.0f32) {
        f32::new(0.0f32)
    } else {
        (f32::new(1.0f32) + a_val.exp()).ln()
    };
    let g = a_log[h] * sp;
    decay_out[i] = g.exp();
}

/// Launch batched beta/decay over `p` tokens (Issue 637 T5).
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetBetaDecayBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetBetaDecayBatchedCubeCL {
    /// Compute per-head beta and decay for `p` tokens in one dispatch.
    ///
    /// Bit-identical to `p` sequential [`DeltanetBetaDecayCubeCL::launch`] calls:
    /// every output element is an independent function of its own inputs, so
    /// there is no accumulation order to differ.
    ///
    /// # Safety
    /// - `a_raw`, `b_raw`, `beta_out`, `decay_out`: >= `p * n_head` f32 elements.
    /// - `a_log`, `dt_bias`: >= `n_head` f32 elements.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_raw_handle: Handle,
        b_raw_handle: Handle,
        a_log_handle: Handle,
        dt_bias_handle: Handle,
        beta_out_handle: Handle,
        decay_out_handle: Handle,
        n_head: usize,
        p: usize,
    ) {
        let total = p * n_head;
        let params: [f32; 2] = [n_head as f32, total as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            deltanet_beta_decay_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(a_raw_handle, total),
                BufferArg::from_raw_parts(b_raw_handle, total),
                BufferArg::from_raw_parts(a_log_handle, n_head),
                BufferArg::from_raw_parts(dt_bias_handle, n_head),
                BufferArg::from_raw_parts(beta_out_handle, total),
                BufferArg::from_raw_parts(decay_out_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Z-gating kernel (Issue 599 GPU-resident forward)
// ---------------------------------------------------------------------------

/// CubeCL kernel: apply z-gating to the recurrent output.
///
/// Computes `output[i] *= z[i] * sigmoid(z[i])` = `output[i] *= silu(z[i])`.
///
/// One thread per element. Dispatch: `CubeCount::Static(ceil(n/128), 1, 1)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_z_gating_f32(
    output: &mut [f32],
    z: &[f32],
    params: &[f32],
) {
    let n = params[0usize] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= n {
        terminate!();
    }

    let z_val = z[idx];
    // silu(z) = z * sigmoid(z)
    let neg_z = f32::new(0.0f32) - z_val;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_z.exp());
    output[idx] = output[idx] * z_val * sig;
}

/// Launch z-gating kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltanetZGatingCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DeltanetZGatingCubeCL {
    /// Apply z-gating in-place: `output[i] *= silu(z[i])`.
    ///
    /// # Safety
    /// `output` and `z` handles must have at least `n` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        output_handle: Handle,
        z_handle: Handle,
        n: usize,
    ) {
        // Metal grid guard (Issue 730, 2026-08-19): n/128 workgroups exceeds
        // Metal's 65535 x-cap from n >= 8.4M elements (p * z_dim at p=1366
        // with Bonsai's z_dim 6144 → 98304 at p=2048). Pure elementwise —
        // chunk on element batches, slice both handles, adjust the bound.
        // Bit-identical per-element arithmetic.
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let elems_per_chunk = (MAX_WG_X as usize) * wg_size as usize;
        let mut e0 = 0usize;
        while e0 < n {
            let ec = elems_per_chunk.min(n - e0);
            let params: [f32; 1] = [ec as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (ec as u32).div_ceil(wg_size).max(1);
            unsafe {
                deltanet_z_gating_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(
                        output_handle.clone().offset_start((e0 * 4) as u64),
                        ec,
                    ),
                    BufferArg::from_raw_parts(z_handle.clone().offset_start((e0 * 4) as u64), ec),
                    BufferArg::from_raw_parts(params_handle, 1),
                );
            }
            e0 += ec;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-head L2 normalization kernel (Issue 599 GPU-resident forward)
// ---------------------------------------------------------------------------

/// CubeCL kernel: L2-normalize each head's slice of the qkv buffer.
///
/// Operates on the Q and K sections of the qkv buffer (first 2/3).
/// Layout: `[q (n_head × hd), k (n_head × hd), v (n_head × hd)]`.
/// Normalizes q[h, :] and k[h, :] to unit L2 norm.
///
/// Simple O(n_head × head_dim) per-thread approach — each thread computes
/// the full head norm then normalizes its element. Correctness first;
/// shared-memory reduction optimization deferred.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn l2_normalize_heads_f32(
    qkv: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let head_dim = params[1usize] as usize;
    let total_qk = n_head * head_dim;
    let total_elems = 2usize * total_qk;

    let idx = ABSOLUTE_POS;

    if idx >= total_elems {
        terminate!();
    }

    // Determine section offset without branching (avoids NativeExpand type issues):
    // section 0 (Q): offset = 0, section 1 (K): offset = total_qk
    let section_idx = idx / total_qk; // 0 for Q, 1 for K
    let local_idx = idx % total_qk; // index within the section
    let head = local_idx / head_dim;
    let col = local_idx % head_dim;
    let head_off = section_idx * total_qk + head * head_dim;

    // Compute L2 norm for this head (each thread redundantly computes)
    let mut sq_sum = f32::new(0.0f32);
    for c in 0..head_dim {
        let val = qkv[head_off + c];
        sq_sum += val * val;
    }
    // Zero-norm guard (Issue 673 Bug D): an all-zero head would give
    // 1/sqrt(0) = inf and 0·inf = NaN, poisoning the recurrent state. The
    // CPU reference guards `if sum_sq > 0.0`; multiplying by 0 instead is
    // bit-compatible with that guard for the all-zero case (±0·0 = ±0).
    let inv_norm = if sq_sum > f32::new(0.0f32) {
        f32::new(1.0f32) / sq_sum.sqrt()
    } else {
        f32::new(0.0f32)
    };

    // Normalize this element
    qkv[head_off + col] = qkv[head_off + col] * inv_norm;
}

/// Launch per-head L2 normalization kernel for Q and K in the qkv buffer.
#[cfg(feature = "cubecl_runtime")]
pub struct L2NormalizeHeadsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl L2NormalizeHeadsCubeCL {
    /// L2-normalize Q and K heads in-place within the qkv buffer.
    ///
    /// # Safety
    /// `qkv_handle` must have `3 * n_head * head_dim` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_handle: Handle,
        n_head: usize,
        head_dim: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, head_dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let total_elems = 2u32 * n_head as u32 * head_dim as u32;
        let wg_size = 128u32;
        let num_wg = total_elems.div_ceil(wg_size).max(1);

        unsafe {
            l2_normalize_heads_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(qkv_handle, 3 * n_head * head_dim),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fused head expansion + L2 normalization kernel (Issue 604 T5)
// ---------------------------------------------------------------------------

/// CubeCL kernel: expand Q/K heads (tiled broadcast) + L2-normalize, copy V.
///
/// The CPU reference (`expand_heads_into` + `l2_normalize` in
/// `riir-engine/src/deltanet/forward.rs`) expands Q/K from `n_k_heads` to
/// `n_v_heads` via tiled broadcast (v-head `j` maps to k-head `j % n_k_heads`)
/// BEFORE L2-normalizing each head. The recurrence kernel then expects the
/// expanded layout `[Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]`.
///
/// This kernel fuses the expansion + L2-normalization + V copy into one
/// dispatch, reading from the compact GEMV output and writing to the expanded
/// buffer that the recurrence kernel consumes.
///
/// ## Layout
///
/// - `compact`: `[Q(n_k×hd) | K(n_k×hd) | V(n_v×hd)]` — the GEMV output
/// - `expanded`: `[Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]` — consumed by recurrence
/// - `params`: `[n_k_heads, n_v_heads, head_dim]`
///
/// Each thread processes one element of the expanded output. For Q/K
/// sections, the source head is `j % n_k_heads` (tiled broadcast); the thread
/// redundantly computes the head's L2 norm then writes its normalized element.
/// For V, the thread copies the corresponding element verbatim.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn expand_and_l2_normalize_heads_f32(
    compact: &[f32],
    expanded: &mut [f32],
    params: &[f32],
) {
    let n_k_heads = params[0usize] as usize;
    let n_v_heads = params[1usize] as usize;
    let head_dim = params[2usize] as usize;
    let v_compact_off = 2usize * n_k_heads * head_dim;
    let qk_expanded_block = n_v_heads * head_dim;
    let total_elems = 3usize * n_v_heads * head_dim;

    let idx = ABSOLUTE_POS;
    if idx >= total_elems {
        terminate!();
    }

    // Determine which section this element belongs to in the EXPANDED buffer:
    //   0 → Q_expanded, 1 → K_expanded, 2 → V_copy
    let section = idx / qk_expanded_block; // 0, 1, or 2
    let local_idx = idx % qk_expanded_block; // index within this section
    let out_head = local_idx / head_dim; // which output head (0..n_v_heads)
    let col = local_idx % head_dim; // which column within the head

    if section < 2usize {
        // ── Q or K section: expand + L2-normalize ──
        // Source head in the compact buffer: tiled broadcast j % n_k_heads.
        // Compact layout: [Q(n_k×hd) | K(n_k×hd) | ...], so section 0 (Q) starts
        // at offset 0, section 1 (K) starts at n_k_heads * head_dim.
        let src_head = out_head % n_k_heads;
        let compact_section_off = section * n_k_heads * head_dim;
        let src_head_off = compact_section_off + src_head * head_dim;

        // Redundantly compute the L2 norm of the source head (each thread in
        // this head computes the same value — matches l2_normalize_heads_f32).
        let mut sq_sum = f32::new(0.0f32);
        for c in 0..head_dim {
            let val = compact[src_head_off + c];
            sq_sum += val * val;
        }
        // Zero-norm guard (Issue 673 Bug D) — see l2_normalize_heads_f32.
        let inv_norm = if sq_sum > f32::new(0.0f32) {
            f32::new(1.0f32) / sq_sum.sqrt()
        } else {
            f32::new(0.0f32)
        };

        // Write normalized value to expanded buffer
        expanded[idx] = compact[src_head_off + col] * inv_norm;
    } else {
        // ── V section: copy verbatim from compact V ──
        // V is already n_v_heads in the compact buffer; just copy element-by-element.
        expanded[idx] = compact[v_compact_off + local_idx];
    }
}

/// Launch fused head-expansion + L2-normalization + V-copy kernel.
///
/// Reads the compact GEMV output `[Q(n_k×hd) | K(n_k×hd) | V(n_v×hd)]` and
/// writes the expanded buffer `[Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]` consumed
/// by `DeltanetRecurrenceCubeCL`. Q/K heads are L2-normalized during the
/// expansion; V is copied as-is.
///
/// This is the Issue 604 T5 fix for the GPU-resident forward path — the
/// missing step that caused the G1 correctness failure.
///
/// # Safety
/// - `compact_handle` must have `(2 * n_k_heads + n_v_heads) * head_dim` f32 elements.
/// - `expanded_handle` must have `3 * n_v_heads * head_dim` f32 elements.
/// - `n_v_heads` must be a multiple of `n_k_heads` (the tiled broadcast requires it).
//
// Gate: the only non-test consumer is `ternary_deltanet_gpu_forward` (needs
// `ternary_gemv`); tests under `cubecl_runtime` also exercise it. Without
// `ternary_gemv` and without `test`, the struct + `launch` have zero callers,
// so they are gated out to avoid `dead_code` warnings.
#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
pub struct ExpandAndL2NormalizeHeadsCubeCL;

#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
impl ExpandAndL2NormalizeHeadsCubeCL {
    /// Launch the head-expansion + L2-normalize kernel.
    ///
    /// # Safety
    /// - `compact_handle` must have `(2 * n_k_heads + n_v_heads) * head_dim` f32 elements.
    /// - `expanded_handle` must have `3 * n_v_heads * head_dim` f32 elements.
    /// - `n_v_heads` must be a multiple of `n_k_heads` (the tiled broadcast requires it).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        compact_handle: Handle,
        expanded_handle: Handle,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) {
        debug_assert!(
            n_v_heads.is_multiple_of(n_k_heads),
            "n_v_heads ({n_v_heads}) must be a multiple of n_k_heads ({n_k_heads})"
        );

        let params: [f32; 3] = [n_k_heads as f32, n_v_heads as f32, head_dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let total_elems = 3u32 * n_v_heads as u32 * head_dim as u32;
        let wg_size = 128u32;
        let num_wg = total_elems.div_ceil(wg_size).max(1);

        let compact_len = (2 * n_k_heads + n_v_heads) * head_dim;
        let expanded_len = 3 * n_v_heads * head_dim;

        unsafe {
            expand_and_l2_normalize_heads_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(compact_handle, compact_len),
                BufferArg::from_raw_parts(expanded_handle, expanded_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Batched head-expansion kernel (Issue 637 T5)
// ---------------------------------------------------------------------------

/// CubeCL kernel: `expand_and_l2_normalize_heads_f32` over `p` tokens in one
/// dispatch.
///
/// Each output element depends only on its own token's compact row, so the
/// tokens are independent and batching is a wider index space plus a per-token
/// base offset. Same motivation as the batched beta/decay above.
///
/// ## Layout
///
/// - `compact`: `[p, 2*n_k*hd + n_v*hd]` row-major.
/// - `expanded`: `[p, 3*n_v*hd]` row-major.
/// - `params`: `[n_k_heads, n_v_heads, head_dim, total]`, `total = p * 3*n_v*hd`.
#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
#[cube(launch_unchecked)]
fn expand_and_l2_normalize_heads_batched_f32(
    compact: &[f32],
    expanded: &mut [f32],
    params: &[f32],
) {
    let n_k_heads = params[0usize] as usize;
    let n_v_heads = params[1usize] as usize;
    let head_dim = params[2usize] as usize;
    let total = params[3usize] as usize;

    let idx = ABSOLUTE_POS;
    if idx >= total {
        terminate!();
    }

    let qk_expanded_block = n_v_heads * head_dim;
    let per_token_out = 3usize * qk_expanded_block;
    let per_token_in = (2usize * n_k_heads + n_v_heads) * head_dim;

    // Which token this element belongs to, and where that token's rows start.
    let t = idx / per_token_out;
    let local = idx % per_token_out;
    let in_base = t * per_token_in;
    let v_compact_off = in_base + 2usize * n_k_heads * head_dim;

    let section = local / qk_expanded_block; // 0 -> Q, 1 -> K, 2 -> V
    let local_idx = local % qk_expanded_block;
    let out_head = local_idx / head_dim;
    let col = local_idx % head_dim;

    if section < 2usize {
        let src_head = out_head % n_k_heads;
        let compact_section_off = in_base + section * n_k_heads * head_dim;
        let src_head_off = compact_section_off + src_head * head_dim;

        let mut sq_sum = f32::new(0.0f32);
        for c in 0..head_dim {
            let val = compact[src_head_off + c];
            sq_sum += val * val;
        }
        // Zero-norm guard (Issue 673 Bug D) — see l2_normalize_heads_f32.
        let inv_norm = if sq_sum > f32::new(0.0f32) {
            f32::new(1.0f32) / sq_sum.sqrt()
        } else {
            f32::new(0.0f32)
        };

        expanded[idx] = compact[src_head_off + col] * inv_norm;
    } else {
        expanded[idx] = compact[v_compact_off + local_idx];
    }
}

/// Launch batched head-expansion + L2-normalization over `p` tokens.
#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
pub struct ExpandAndL2NormalizeHeadsBatchedCubeCL;

#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
impl ExpandAndL2NormalizeHeadsBatchedCubeCL {
    /// Bit-identical to `p` sequential [`ExpandAndL2NormalizeHeadsCubeCL::launch`]
    /// calls — each thread's arithmetic is unchanged, only its base offset is.
    ///
    /// # Safety
    /// - `compact`: >= `p * (2*n_k_heads + n_v_heads) * head_dim` f32 elements.
    /// - `expanded`: >= `p * 3 * n_v_heads * head_dim` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        compact_handle: Handle,
        expanded_handle: Handle,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
        p: usize,
    ) {
        debug_assert!(
            n_v_heads.is_multiple_of(n_k_heads),
            "n_v_heads ({n_v_heads}) must be a multiple of n_k_heads ({n_k_heads})"
        );

        // Metal grid guard (Issue 726, 2026-08-19): 3*n_v_heads*head_dim/128
        // workgroups per token (144 at Bonsai-27B dims) blows Metal's 65535
        // x-dimension cap from p >= 456. Chunk on token boundaries — the
        // kernel derives its token from the LOCAL flat index against
        // per-chunk params, and sliced handles make local indexes land at
        // the chunk's absolute offset. Bit-identical per-element arithmetic.
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let per_token_out = 3 * n_v_heads * head_dim;
        let per_token_in = (2 * n_k_heads + n_v_heads) * head_dim;
        let wg_per_token = (per_token_out as u32).div_ceil(wg_size).max(1);
        let tokens_per_chunk = (MAX_WG_X / wg_per_token).max(1) as usize;
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let expanded_len = tc * per_token_out;
            let compact_len = tc * per_token_in;
            let params: [f32; 4] = [
                n_k_heads as f32,
                n_v_heads as f32,
                head_dim as f32,
                expanded_len as f32,
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (expanded_len as u32).div_ceil(wg_size).max(1);
            unsafe {
                expand_and_l2_normalize_heads_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(
                        compact_handle
                            .clone()
                            .offset_start((t0 * per_token_in * 4) as u64),
                        compact_len,
                    ),
                    BufferArg::from_raw_parts(
                        expanded_handle
                            .clone()
                            .offset_start((t0 * per_token_out * 4) as u64),
                        expanded_len,
                    ),
                    BufferArg::from_raw_parts(params_handle, 4),
                );
            }
            t0 += tc;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU-resident state buffers
// ---------------------------------------------------------------------------

/// GPU-resident recurrent state for all DeltaNet layers.
///
/// Pre-allocated once, reused across all decode steps.
/// Layout: [n_deltanet_layers × n_head × state_dim_per_head]
/// Total size for Qwen 3.5-0.8B (18 DeltaNet layers): 18 × 16 × 16384 × 4 bytes = 18 MB.
#[cfg(feature = "cubecl_runtime")]
pub struct DeltaNetStateBuffers {
    /// Per-layer recurrent state handles.
    /// One Handle per DeltaNet layer, each containing [n_head × state_dim_per_head] floats.
    /// Empty handles for Attention layers (unused). None for layers without state.
    pub layer_states: Vec<Option<Handle>>,
    /// Number of linear attention heads (for dispatch).
    pub n_head: usize,
    /// State dimension per head (head_dim × head_dim).
    pub state_dim_per_head: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl DeltaNetStateBuffers {
    /// Create zero-initialized state buffers for all layers.
    ///
    /// Pre-allocates GPU memory for DeltaNet layer states. Attention layers get `None`.
    /// Call once at model load time — buffers persist for the entire decode session.
    pub fn new(
        client: &ComputeClient<ActiveRuntime>,
        n_layer: usize,
        layer_types: &[riir_infer_core::types::DeltaNetLayerType],
        n_head: usize,
        state_dim_per_head: usize,
    ) -> Self {
        use riir_infer_core::types::DeltaNetLayerType;

        let zeros = vec![0.0f32; n_head * state_dim_per_head];

        let layer_states: Vec<Option<Handle>> = (0..n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::DeltaNet {
                    Some(client.create_from_slice(f32::as_bytes(&zeros)))
                } else {
                    None
                }
            })
            .collect();

        Self {
            layer_states,
            n_head,
            state_dim_per_head,
        }
    }
}

// ---------------------------------------------------------------------------
// T9 GOAT proof — GPU kernel correctness tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use crate::cubecl_runtime::ActiveRuntime;

    use super::*;
    use crate::cubecl_runtime::CubeCLContext;

    /// Tolerance for GPU vs CPU comparison.
    /// GPU uses f32 throughout so 1e-3 is conservative.
    const TOL: f32 = 1e-3;

    // ── Recurrence kernel (T5) ──────────────────────────────────────────────

    /// GOAT proof: `deltanet_recurrence_f32` GPU output matches CPU
    /// `gated_deltanet_step` reference implementation.
    ///
    /// Tests with the production dimensions (16 heads × head_dim 128).
    /// Test data uses per-head scaling to expose head-indexing bugs.
    #[test]
    fn test_deltanet_recurrence_matches_reference() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 16;
        let head_dim: usize = 128;

        // ── Build synthetic test data ──
        // q[h, c] = 0.01 * (h+1)   — distinct per head to catch head-index bugs
        // k[h, c] = 0.01            — uniform
        // v[h, c] = 0.01            — uniform
        // beta[h]  = 0.5
        // decay[h] = 0.95
        let mut q = vec![0.0f32; n_head * head_dim];
        let mut k = vec![0.0f32; n_head * head_dim];
        let mut v = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            for c in 0..head_dim {
                q[h * head_dim + c] = 0.01 * (h as f32 + 1.0);
                k[h * head_dim + c] = 0.01;
                v[h * head_dim + c] = 0.01;
            }
        }
        let betas: Vec<f32> = (0..n_head).map(|_| 0.5).collect();
        let decays: Vec<f32> = (0..n_head).map(|_| 0.95).collect();

        // Interleave q|k|v into flat qkv for GPU
        let mut qkv = vec![0.0f32; 3 * n_head * head_dim];
        qkv[0..n_head * head_dim].copy_from_slice(&q);
        qkv[n_head * head_dim..2 * n_head * head_dim].copy_from_slice(&k);
        qkv[2 * n_head * head_dim..3 * n_head * head_dim].copy_from_slice(&v);

        // ── GPU path ──
        let state_len = n_head * head_dim * head_dim;
        let output_len = n_head * head_dim;
        let state_zeros = vec![0.0f32; state_len];

        let qkv_handle = client.create_from_slice(f32::as_bytes(&qkv));
        let state_handle = client.create_from_slice(f32::as_bytes(&state_zeros));
        let output_handle = client.empty(output_len * std::mem::size_of::<f32>());

        // Clone state handle so we can read it back after the kernel mutates it.
        let state_readback = state_handle.clone();

        unsafe {
            DeltanetRecurrenceCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_handle,
                state_handle,
                output_handle.clone(),
                n_head,
                head_dim,
                &betas,
                &decays,
            );
        }

        // Read GPU output + state
        let gpu_output_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_output_bytes);
        let gpu_state_bytes = client.read_one(state_readback).expect("read state");
        let gpu_state = f32::from_bytes(&gpu_state_bytes);

        // ── CPU reference path ──
        let mut cpu_state = state_zeros.clone();
        let cpu_output = riir_infer_core::deltanet::forward::gated_deltanet_step(
            &q,
            &k,
            &v,
            &mut cpu_state,
            &betas,
            &decays,
            n_head,
            head_dim,
            head_dim,
        );

        // ── Compare outputs ──
        assert_eq!(gpu_output.len(), cpu_output.len(), "output length mismatch");
        let mut max_output_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c).abs();
            max_output_diff = max_output_diff.max(diff);
            assert!(
                diff < TOL,
                "output[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        // ── Compare states ──
        assert_eq!(gpu_state.len(), cpu_state.len(), "state length mismatch");
        let mut max_state_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_state.iter().zip(cpu_state.iter()).enumerate() {
            let diff = (g - c).abs();
            max_state_diff = max_state_diff.max(diff);
            assert!(
                diff < TOL,
                "state[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        eprintln!(
            "deltanet recurrence GOAT ✓ max_output_diff={max_output_diff:.6} max_state_diff={max_state_diff:.6}"
        );
    }

    // ── Row-parallel recurrence G1 (Issue 619 T3) ───────────────────────────

    /// Issue 619 T3 tolerance. Tighter than the module-wide 1e-3, matching the
    /// CUDA counterpart's measured ~9e-5 (Issue 617). The gate is **vs the CPU
    /// reference**, not vs the legacy kernel — `plane_sum` and the smem tree
    /// reduce in different orders, so neither GPU kernel is bit-identical to the
    /// other and comparing them would gate on the wrong invariant.
    #[cfg(feature = "deltanet_recurrence_rowpar")]
    const ROWPAR_TOL: f32 = 1e-4;

    /// Deterministic, **column-varying** test data.
    ///
    /// The existing `test_deltanet_recurrence_matches_reference` uses uniform
    /// `k` and `v`, which cannot detect a wrong column→lane mapping — and the
    /// column mapping is precisely what the row-parallel kernel changes (lane
    /// `L` owns columns `L, L+32, L+64, L+96` instead of `col = UNIT_POS`).
    /// Every element here varies in both `h` and `c` so a mis-indexed lane
    /// produces a visible mismatch.
    #[cfg(feature = "deltanet_recurrence_rowpar")]
    fn rowpar_step_data(
        step: usize,
        n_head: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let s = step as f32;
        let mut q = vec![0.0f32; n_head * head_dim];
        let mut k = vec![0.0f32; n_head * head_dim];
        let mut v = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            for c in 0..head_dim {
                let (hf, cf) = (h as f32, c as f32);
                let i = h * head_dim + c;
                q[i] = 0.02 * ((hf + 1.0) * 0.37 + (cf * 0.11 + s).sin());
                k[i] = 0.02 * ((hf * 0.23 + 1.0) + (cf * 0.07 + s).cos());
                v[i] = 0.02 * ((hf + 1.0) * 0.19 + (cf * 0.13 + s).sin());
            }
        }
        (q, k, v)
    }

    /// G1 for `deltanet_recurrence_f32_rowpar` (Issue 619 T3).
    ///
    /// Runs **two sequential decode steps**. This matters: with a zero initial
    /// state the decay path is mathematically a no-op (`0 * decay == 0`), so a
    /// single-step test never exercises step 1 at all. Step 2 runs against a
    /// non-zero state and does.
    ///
    /// Reports the legacy kernel's error against the same CPU reference for
    /// context, but only gates on the row-parallel kernel.
    #[cfg(feature = "deltanet_recurrence_rowpar")]
    #[test]
    fn test_deltanet_recurrence_rowpar_matches_reference() {
        const STEPS: usize = 2;

let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 16;
        let head_dim: usize = 128;

        assert!(
            DeltanetRecurrenceRowParCubeCL::supports(head_dim),
            "rowpar kernel must support the production head_dim"
        );

        let betas: Vec<f32> =
            (0..n_head).map(|h| 0.3 + 0.4 * (h as f32 / n_head as f32)).collect();
        let decays: Vec<f32> =
            (0..n_head).map(|h| 0.90 + 0.09 * (h as f32 / n_head as f32)).collect();

        let state_len = n_head * head_dim * head_dim;
        let output_len = n_head * head_dim;
        let out_bytes_len = output_len * std::mem::size_of::<f32>();

        // ── CPU reference (ground truth) ──
        let mut cpu_state = vec![0.0f32; state_len];
        let mut cpu_output = Vec::new();
        for step in 0..STEPS {
            let (q, k, v) = rowpar_step_data(step, n_head, head_dim);
            cpu_output = riir_infer_core::deltanet::forward::gated_deltanet_step(
                &q,
                &k,
                &v,
                &mut cpu_state,
                &betas,
                &decays,
                n_head,
                head_dim,
                head_dim,
            );
        }

        // ── Row-parallel GPU path ──
        let rp_state = client.create_from_slice(f32::as_bytes(&vec![0.0f32; state_len]));
        let beta_h = client.create_from_slice(f32::as_bytes(&betas));
        let decay_h = client.create_from_slice(f32::as_bytes(&decays));
        let mut rp_output: Vec<f32> = Vec::new();
        for step in 0..STEPS {
            let (q, k, v) = rowpar_step_data(step, n_head, head_dim);
            let mut qkv = vec![0.0f32; 3 * n_head * head_dim];
            qkv[..output_len].copy_from_slice(&q);
            qkv[output_len..2 * output_len].copy_from_slice(&k);
            qkv[2 * output_len..].copy_from_slice(&v);

            let qkv_h = client.create_from_slice(f32::as_bytes(&qkv));
            let out_h = client.empty(out_bytes_len);
            unsafe {
                DeltanetRecurrenceRowParCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                    &client,
                    qkv_h,
                    beta_h.clone(),
                    decay_h.clone(),
                    rp_state.clone(),
                    out_h.clone(),
                    n_head,
                    head_dim,
                );
            }
            rp_output = f32::from_bytes(&client.read_one(out_h).expect("read rowpar out")).to_vec();
        }
        let rp_state_final =
            f32::from_bytes(&client.read_one(rp_state).expect("read rowpar state")).to_vec();

        // ── Legacy GPU path (context only, not the gate) ──
        let lg_state = client.create_from_slice(f32::as_bytes(&vec![0.0f32; state_len]));
        let mut lg_output: Vec<f32> = Vec::new();
        for step in 0..STEPS {
            let (q, k, v) = rowpar_step_data(step, n_head, head_dim);
            let mut qkv = vec![0.0f32; 3 * n_head * head_dim];
            qkv[..output_len].copy_from_slice(&q);
            qkv[output_len..2 * output_len].copy_from_slice(&k);
            qkv[2 * output_len..].copy_from_slice(&v);

            let qkv_h = client.create_from_slice(f32::as_bytes(&qkv));
            let out_h = client.empty(out_bytes_len);
            unsafe {
                DeltanetRecurrenceCubeCL::launch::<ActiveRuntime>(
                    &client,
                    qkv_h,
                    lg_state.clone(),
                    out_h.clone(),
                    n_head,
                    head_dim,
                    &betas,
                    &decays,
                );
            }
            lg_output = f32::from_bytes(&client.read_one(out_h).expect("read legacy out")).to_vec();
        }
        let lg_state_final =
            f32::from_bytes(&client.read_one(lg_state).expect("read legacy state")).to_vec();

        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };

        let rp_out_diff = max_abs(&rp_output, &cpu_output);
        let rp_state_diff = max_abs(&rp_state_final, &cpu_state);
        let lg_out_diff = max_abs(&lg_output, &cpu_output);
        let lg_state_diff = max_abs(&lg_state_final, &cpu_state);
        // Informational: the two GPU kernels differ from each other by reduction
        // order alone. Reported, never asserted.
        let cross_out_diff = max_abs(&rp_output, &lg_output);

        eprintln!(
            "Issue 619 T3 ({STEPS} steps, {n_head}x{head_dim}, column-varying data)\n\
             \x20 rowpar vs CPU : out={rp_out_diff:.3e} state={rp_state_diff:.3e}\n\
             \x20 legacy vs CPU : out={lg_out_diff:.3e} state={lg_state_diff:.3e}\n\
             \x20 rowpar vs legacy (informational): out={cross_out_diff:.3e}"
        );

        assert_eq!(rp_output.len(), cpu_output.len(), "output length mismatch");
        assert_eq!(rp_state_final.len(), cpu_state.len(), "state length mismatch");
        assert!(
            rp_out_diff < ROWPAR_TOL,
            "rowpar output vs CPU: max diff {rp_out_diff:.3e} >= {ROWPAR_TOL:.0e} \
             (legacy was {lg_out_diff:.3e})"
        );
        assert!(
            rp_state_diff < ROWPAR_TOL,
            "rowpar state vs CPU: max diff {rp_state_diff:.3e} >= {ROWPAR_TOL:.0e} \
             (legacy was {lg_state_diff:.3e})"
        );
    }

    /// Issue 604 T1: prove the recurrence kernel is parameterized for any
    /// `n_head` — not just the Qwen3.5-0.8B default of 16. Bonsai-27B uses
    /// `n_v_heads=48` after Q/K head expansion, so this test exercises the
    /// production Bonsai head count + head_dim 128.
    ///
    /// Per-head scaling exposes any head-index regression from the
    /// hardcoded→parameterized transition.
    #[test]
    fn test_deltanet_recurrence_n_head_48_bonsai() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 48; // Bonsai-27B n_v_heads
        let head_dim: usize = 128;

        let mut q = vec![0.0f32; n_head * head_dim];
        let mut k = vec![0.0f32; n_head * head_dim];
        let mut v = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            for c in 0..head_dim {
                // Distinct per head + a small col gradient so head-index bugs
                // (e.g. truncation to 16) are immediately visible.
                q[h * head_dim + c] = 0.001 * (h as f32 + 1.0) + 0.0001 * c as f32;
                k[h * head_dim + c] = 0.001 * (h as f32 + 1.0);
                v[h * head_dim + c] = 0.001 * (h as f32 + 1.0);
            }
        }
        let betas: Vec<f32> = (0..n_head).map(|h| 0.4 + 0.01 * h as f32).collect();
        let decays: Vec<f32> = (0..n_head).map(|h| 0.9 - 0.005 * h as f32).collect();

        let mut qkv = vec![0.0f32; 3 * n_head * head_dim];
        qkv[0..n_head * head_dim].copy_from_slice(&q);
        qkv[n_head * head_dim..2 * n_head * head_dim].copy_from_slice(&k);
        qkv[2 * n_head * head_dim..3 * n_head * head_dim].copy_from_slice(&v);

        let state_len = n_head * head_dim * head_dim;
        let output_len = n_head * head_dim;
        let state_zeros = vec![0.0f32; state_len];

        let qkv_handle = client.create_from_slice(f32::as_bytes(&qkv));
        let state_handle = client.create_from_slice(f32::as_bytes(&state_zeros));
        let output_handle = client.empty(output_len * std::mem::size_of::<f32>());

        let state_readback = state_handle.clone();

        unsafe {
            DeltanetRecurrenceCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_handle,
                state_handle,
                output_handle.clone(),
                n_head,
                head_dim,
                &betas,
                &decays,
            );
        }

        let gpu_output_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_output_bytes);
        let gpu_state_bytes = client.read_one(state_readback).expect("read state");
        let gpu_state = f32::from_bytes(&gpu_state_bytes);

        let mut cpu_state = state_zeros.clone();
        let cpu_output = riir_infer_core::deltanet::forward::gated_deltanet_step(
            &q,
            &k,
            &v,
            &mut cpu_state,
            &betas,
            &decays,
            n_head,
            head_dim,
            head_dim,
        );

        assert_eq!(gpu_output.len(), cpu_output.len());
        let mut max_output_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c).abs();
            max_output_diff = max_output_diff.max(diff);
            assert!(
                diff < TOL,
                "output[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        assert_eq!(gpu_state.len(), cpu_state.len());
        let mut max_state_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_state.iter().zip(cpu_state.iter()).enumerate() {
            let diff = (g - c).abs();
            max_state_diff = max_state_diff.max(diff);
            assert!(
                diff < TOL,
                "state[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        eprintln!(
            "deltanet recurrence n_head=48 GOAT ✓ max_output_diff={max_output_diff:.6} max_state_diff={max_state_diff:.6}"
        );
    }

    /// Issue 610 regression test: prove the recurrence kernel's dot-product
    /// reductions sum across the FULL workgroup (head_dim=128 threads), not
    /// just the first plane (plane_dim=32 on Metal).
    ///
    /// **Why this test exists.** The pre-Issue-610 code used `plane_sum` for
    /// the readout + kv_mem dot products. `plane_sum` is a subgroup-level
    /// reduction, so on Metal it only summed 32 of 128 terms. The bug was
    /// masked by the other recurrence tests because they used tiny uniform-ish
    /// inputs where the absolute error stayed under the 0.001 tolerance.
    ///
    /// This test catches the bug by using:
    /// 1. **Small magnitudes** (~0.001 range — matches the original masking).
    /// 2. **A linear column gradient** in `q` — so Σ(q[0..32]) ≠ Σ(q[32..64])
    ///    ≠ Σ(q[96..128]). With the bug, only the first gradient slice is
    ///    summed; without the bug, all four slices contribute.
    /// 3. **Two decode steps** with non-zero state — exercises BOTH the
    ///    readout dot (step 5) and the kv_mem dot (step 2-3).
    ///
    /// Diagnostic: if the bug regresses, the readout output ratio
    /// (CPU/GPU) equals Σ(q[0..128]) / Σ(q[0..32]) — a value that depends
    /// on the gradient but is always ≠ 1.0.
    #[test]
    fn test_deltanet_recurrence_full_workgroup_reduction_issue_610() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 4;
        let head_dim: usize = 128; // > plane_dim (32) — exercises cross-plane reduction

        // ── Small-magnitude inputs with a column gradient ──
        // q[h, c] = 0.001 + 0.0001 * c   — linear ramp, distinct per column
        // k[h, c] = 0.001                  — uniform per head
        // v[h, c] = 0.002                  — uniform per head
        //
        // With this gradient: Σ(q[0..128]) = 128*0.001 + 0.0001*Σ(0..128)
        //                                  = 0.128 + 0.0001*8128 = 0.9408
        //                          Σ(q[0..32])  = 32*0.001  + 0.0001*Σ(0..32)
        //                                  = 0.032  + 0.0001*496  = 0.0816
        //                  ratio = 0.9408 / 0.0816 = 11.53
        //
        // If the plane_sum bug returns, GPU readout ≈ CPU/11.53 — an order-
        // of-magnitude error that blows past any tolerance, even with these
        // small input magnitudes.
        let mut q = vec![0.0f32; n_head * head_dim];
        let mut k = vec![0.0f32; n_head * head_dim];
        let mut v = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            for c in 0..head_dim {
                q[h * head_dim + c] = 0.001 + 0.0001 * c as f32;
                k[h * head_dim + c] = 0.001;
                v[h * head_dim + c] = 0.002;
            }
        }
        let betas: Vec<f32> = vec![0.5; n_head];
        let decays: Vec<f32> = vec![0.95; n_head];

        // Interleave q|k|v into flat qkv for GPU
        let mut qkv = vec![0.0f32; 3 * n_head * head_dim];
        qkv[0..n_head * head_dim].copy_from_slice(&q);
        qkv[n_head * head_dim..2 * n_head * head_dim].copy_from_slice(&k);
        qkv[2 * n_head * head_dim..3 * n_head * head_dim].copy_from_slice(&v);

        // ── Step 1: zero state, run recurrence ──
        let state_len = n_head * head_dim * head_dim;
        let output_len = n_head * head_dim;
        let state_zeros = vec![0.0f32; state_len];

        let qkv_handle = client.create_from_slice(f32::as_bytes(&qkv));
        let state_handle = client.create_from_slice(f32::as_bytes(&state_zeros));
        let output_handle = client.empty(output_len * std::mem::size_of::<f32>());

        unsafe {
            DeltanetRecurrenceCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_handle.clone(),
                state_handle.clone(),
                output_handle.clone(),
                n_head,
                head_dim,
                &betas,
                &decays,
            );
        }

        // ── Step 2: re-use mutated state, run recurrence AGAIN ──
        // This exercises the kv_mem dot product (step 2-3) — on step 1 the
        // state was zero so kv_mem was trivially 0; on step 2 the state is
        // non-zero so kv_mem must reduce across all 128 threads.
        unsafe {
            DeltanetRecurrenceCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_handle,
                state_handle.clone(),
                output_handle.clone(),
                n_head,
                head_dim,
                &betas,
                &decays,
            );
        }

        let gpu_output_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_output_bytes);
        let gpu_state_bytes = client.read_one(state_handle).expect("read state");
        let gpu_state = f32::from_bytes(&gpu_state_bytes);

        // ── CPU reference: run gated_deltanet_step TWICE with the same qkv ──
        let mut cpu_state = state_zeros.clone();
        let _cpu_out1 = riir_infer_core::deltanet::forward::gated_deltanet_step(
            &q, &k, &v, &mut cpu_state, &betas, &decays, n_head, head_dim, head_dim,
        );
        let cpu_output = riir_infer_core::deltanet::forward::gated_deltanet_step(
            &q, &k, &v, &mut cpu_state, &betas, &decays, n_head, head_dim, head_dim,
        );

        // ── Compare outputs (after 2 steps) ──
        assert_eq!(gpu_output.len(), cpu_output.len());
        let mut max_output_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c).abs();
            max_output_diff = max_output_diff.max(diff);
            assert!(
                diff < TOL,
                "issue-610 regression: output[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL} (full-workgroup reduction broken?)"
            );
        }

        // ── Compare states (after 2 steps) ──
        assert_eq!(gpu_state.len(), cpu_state.len());
        let mut max_state_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_state.iter().zip(cpu_state.iter()).enumerate() {
            let diff = (g - c).abs();
            max_state_diff = max_state_diff.max(diff);
            assert!(
                diff < TOL,
                "issue-610 regression: state[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL} (full-workgroup reduction broken?)"
            );
        }

        eprintln!(
            "issue-610 full-workgroup reduction regression ✓ (2 decode steps) max_output_diff={max_output_diff:.6} max_state_diff={max_state_diff:.6}"
        );
    }

    // ── Fused expand + L2-normalize kernel (Issue 604 T5) ──────────────────

    /// GOAT proof: `expand_and_l2_normalize_heads_f32` GPU matches the CPU
    /// reference (`expand_heads_into` + `l2_normalize`).
    ///
    /// Tests with the Bonsai-27B production dimensions:
    /// - n_k_heads = 16 (compact Q/K)
    /// - n_v_heads = 48 (expanded Q/K + V)
    /// - head_dim = 128
    /// - repeat_factor = 3
    ///
    /// The CPU reference expands Q/K via tiled broadcast (v-head j → k-head
    /// j % n_k_heads) then L2-normalizes each expanded head. This test verifies
    /// the GPU kernel produces bit-identical results.
    #[test]
    fn test_expand_and_l2_normalize_matches_reference() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_k_heads: usize = 16;
        let n_v_heads: usize = 48;
        let head_dim: usize = 128;

        // ── Build compact qkv with distinct per-head values ──
        // q[h, c] = 0.01 * (h+1) + 0.001 * c  — distinct per head + col gradient
        // k[h, c] = 0.02 * (h+1)              — distinct per head only
        // v[h, c] = 0.1 * (h+1) + 0.01 * c    — distinct per head + col
        let q_dim = n_k_heads * head_dim;
        let k_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let compact_len = q_dim + k_dim + v_dim;

        let mut compact = vec![0.0f32; compact_len];
        for h in 0..n_k_heads {
            for c in 0..head_dim {
                compact[h * head_dim + c] = 0.01 * (h as f32 + 1.0) + 0.001 * c as f32;
                compact[q_dim + h * head_dim + c] = 0.02 * (h as f32 + 1.0);
            }
        }
        for h in 0..n_v_heads {
            for c in 0..head_dim {
                compact[q_dim + k_dim + h * head_dim + c] = 0.1 * (h as f32 + 1.0) + 0.01 * c as f32;
            }
        }

        // ── GPU path ──
        let compact_handle = client.create_from_slice(f32::as_bytes(&compact));
        let expanded_len = 3 * n_v_heads * head_dim;
        let expanded_handle = client.empty(expanded_len * std::mem::size_of::<f32>());

        unsafe {
            ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                &client,
                compact_handle,
                expanded_handle.clone(),
                n_k_heads,
                n_v_heads,
                head_dim,
            );
        }

        let gpu_bytes = client
            .read_one(expanded_handle)
            .expect("read expanded buffer");
        let gpu_expanded = f32::from_bytes(&gpu_bytes);

        // ── CPU reference: expand_heads_into + l2_normalize ──
        let mut cpu_expanded = vec![0.0f32; expanded_len];

        // Expand Q (tiled broadcast): each k-head repeated repeat_factor times
        let repeat = n_v_heads / n_k_heads;
        let block = n_k_heads * head_dim;
        for r in 0..repeat {
            cpu_expanded[r * block..r * block + block]
                .copy_from_slice(&compact[..block]);
        }
        // Expand K
        let k_expanded_off = n_v_heads * head_dim;
        for r in 0..repeat {
            cpu_expanded[k_expanded_off + r * block..k_expanded_off + r * block + block]
                .copy_from_slice(&compact[q_dim..q_dim + block]);
        }
        // Copy V as-is
        let v_expanded_off = 2 * n_v_heads * head_dim;
        cpu_expanded[v_expanded_off..v_expanded_off + v_dim]
            .copy_from_slice(&compact[q_dim + k_dim..]);

        // L2-normalize each Q and K expanded head
        for h in 0..n_v_heads {
            let off = h * head_dim;
            let q_slice = &mut cpu_expanded[off..off + head_dim];
            let sum_sq: f32 = q_slice.iter().map(|x| x * x).sum();
            if sum_sq > 0.0 {
                let inv = 1.0 / sum_sq.sqrt();
                for x in q_slice.iter_mut() {
                    *x *= inv;
                }
            }
            let k_off = k_expanded_off + off;
            let k_slice = &mut cpu_expanded[k_off..k_off + head_dim];
            let sum_sq: f32 = k_slice.iter().map(|x| x * x).sum();
            if sum_sq > 0.0 {
                let inv = 1.0 / sum_sq.sqrt();
                for x in k_slice.iter_mut() {
                    *x *= inv;
                }
            }
        }

        // ── Compare ──
        assert_eq!(
            gpu_expanded.len(),
            cpu_expanded.len(),
            "expanded length mismatch"
        );
        let mut max_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_expanded.iter().zip(cpu_expanded.iter()).enumerate() {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < TOL,
                "expanded[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        eprintln!("expand + L2 normalize GOAT ✓ max_diff={max_diff:.6}");
    }

    /// Composition test: expand+L2-norm → recurrence pipeline.
    ///
    /// This is the exact pipeline the GPU forward will use once Issue 604 T5
    /// is wired in:
    ///   compact_qkv → ExpandAndL2NormalizeHeadsCubeCL → expanded_qkv
    ///             → DeltanetRecurrenceCubeCL → output + state update
    ///
    /// **Issue 610 fix verified here** — the recurrence kernel's `plane_sum`
    /// was replaced with a full-workgroup shared-memory tree reduction.
    #[test]
    fn test_expand_norm_then_recurrence_pipeline() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_k_heads: usize = 16;
        let n_v_heads: usize = 48;
        let head_dim: usize = 128;

        // ── Build compact qkv ──
        let q_dim = n_k_heads * head_dim;
        let k_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let compact_len = q_dim + k_dim + v_dim;

        let mut compact = vec![0.0f32; compact_len];
        for h in 0..n_k_heads {
            for c in 0..head_dim {
                compact[h * head_dim + c] = 0.01 * (h as f32 + 1.0) + 0.001 * c as f32;
                compact[q_dim + h * head_dim + c] = 0.02 * (h as f32 + 1.0);
            }
        }
        for h in 0..n_v_heads {
            for c in 0..head_dim {
                compact[q_dim + k_dim + h * head_dim + c] = 0.1 * (h as f32 + 1.0) + 0.01 * c as f32;
            }
        }

        let betas: Vec<f32> = (0..n_v_heads).map(|h| 0.4 + 0.01 * h as f32).collect();
        let decays: Vec<f32> = (0..n_v_heads).map(|h| 0.9 - 0.005 * h as f32).collect();

        // ── GPU path: expand+norm → recurrence ──
        let compact_handle = client.create_from_slice(f32::as_bytes(&compact));
        let expanded_len = 3 * n_v_heads * head_dim;
        let expanded_handle = client.empty(expanded_len * std::mem::size_of::<f32>());

        unsafe {
            ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                &client,
                compact_handle,
                expanded_handle.clone(),
                n_k_heads,
                n_v_heads,
                head_dim,
            );
        }

        // Now run recurrence on the expanded buffer
        let state_len = n_v_heads * head_dim * head_dim;
        let output_len = n_v_heads * head_dim;
        let state_zeros = vec![0.0f32; state_len];
        let state_handle = client.create_from_slice(f32::as_bytes(&state_zeros));
        let output_handle = client.empty(output_len * std::mem::size_of::<f32>());
        let beta_handle = client.create_from_slice(f32::as_bytes(&betas));
        let decay_handle = client.create_from_slice(f32::as_bytes(&decays));
        let state_readback = state_handle.clone();

        unsafe {
            DeltanetRecurrenceCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                &client,
                expanded_handle,
                beta_handle,
                decay_handle,
                state_handle,
                output_handle.clone(),
                n_v_heads,
                head_dim,
            );
        }

        let gpu_output_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_output_bytes);
        let gpu_state_bytes = client.read_one(state_readback).expect("read state");
        let gpu_state = f32::from_bytes(&gpu_state_bytes);

        // ── CPU reference: expand + normalize + recurrence ──
        let repeat = n_v_heads / n_k_heads;
        let block = n_k_heads * head_dim;
        let mut q_expanded = vec![0.0f32; n_v_heads * head_dim];
        let mut k_expanded = vec![0.0f32; n_v_heads * head_dim];
        let v_expanded: Vec<f32> = compact[q_dim + k_dim..].to_vec();

        for r in 0..repeat {
            q_expanded[r * block..r * block + block].copy_from_slice(&compact[..block]);
            k_expanded[r * block..r * block + block].copy_from_slice(&compact[q_dim..q_dim + block]);
        }

        // L2-normalize each expanded head
        for h in 0..n_v_heads {
            let off = h * head_dim;
            for buf in [&mut q_expanded, &mut k_expanded] {
                let slice = &mut buf[off..off + head_dim];
                let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
                if sum_sq > 0.0 {
                    let inv = 1.0 / sum_sq.sqrt();
                    for x in slice.iter_mut() {
                        *x *= inv;
                    }
                }
            }
        }

        let mut cpu_state = state_zeros.clone();
        let cpu_output = riir_infer_core::deltanet::forward::gated_deltanet_step(
            &q_expanded,
            &k_expanded,
            &v_expanded,
            &mut cpu_state,
            &betas,
            &decays,
            n_v_heads,
            head_dim,
            head_dim,
        );

        let mut max_output_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c).abs();
            max_output_diff = max_output_diff.max(diff);
            assert!(
                diff < TOL,
                "pipeline output[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        assert_eq!(gpu_state.len(), cpu_state.len());
        let mut max_state_diff = 0.0f32;
        for (i, (&g, &c)) in gpu_state.iter().zip(cpu_state.iter()).enumerate() {
            let diff = (g - c).abs();
            max_state_diff = max_state_diff.max(diff);
            assert!(
                diff < TOL,
                "pipeline state[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }

        eprintln!(
            "expand→norm→recurrence pipeline GOAT ✓ max_output_diff={max_output_diff:.6} max_state_diff={max_state_diff:.6}"
        );
    }

    // ── Conv1d kernel (T6) ─────────────────────────────────────────────────

    /// GOAT proof: `deltanet_conv1d_f32` GPU matches CPU `causal_conv1d_update`.
    ///
    /// Uses small dimensions (conv_dim=4, kernel_size=3) — the kernel is
    /// fully parameterized via the params array.
    #[test]
    fn test_deltanet_conv1d_matches_reference() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let conv_dim: usize = 4;
        let kernel_size: usize = 3;

        // Input: one value per channel
        let input: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5];
        // Depthwise weights: conv_dim × kernel_size
        let conv_weight: Vec<f32> = vec![
            0.1, 0.2, 0.3, // ch 0
            0.4, 0.5, 0.6, // ch 1
            0.7, 0.8, 0.9, // ch 2
            1.0, 0.0, -0.5, // ch 3
        ];
        // Conv state (sliding window): starts with old values, will be shifted
        let conv_state: Vec<f32> = vec![
            0.0, 0.0, 0.0, // ch 0 window
            0.0, 0.0, 0.0, // ch 1 window
            0.0, 0.0, 0.0, // ch 2 window
            0.0, 0.0, 0.0, // ch 3 window
        ];

        // ── GPU path ──
        let input_handle = client.create_from_slice(f32::as_bytes(&input));
        let conv_weight_handle = client.create_from_slice(f32::as_bytes(&conv_weight));
        let conv_state_handle = client.create_from_slice(f32::as_bytes(&conv_state));

        let state_readback = conv_state_handle.clone();
        let input_readback = input_handle.clone();

        unsafe {
            DeltanetConv1dCubeCL::launch::<ActiveRuntime>(
                &client,
                input_handle,
                conv_weight_handle,
                conv_state_handle,
                conv_dim,
                kernel_size,
            );
        }

        let gpu_input_bytes = client.read_one(input_readback).expect("read input");
        let gpu_input = f32::from_bytes(&gpu_input_bytes);
        let gpu_state_bytes = client.read_one(state_readback).expect("read conv_state");
        let gpu_state = f32::from_bytes(&gpu_state_bytes);

        // ── CPU reference ──
        let mut cpu_input = input.clone();
        let mut cpu_conv_state = conv_state.clone();
        riir_infer_core::deltanet::forward::causal_conv1d_update(
            &mut cpu_input,
            &conv_weight,
            &mut cpu_conv_state,
            conv_dim,
            kernel_size,
        );

        // ── Compare output (modified input) ──
        assert_eq!(gpu_input.len(), cpu_input.len());
        for (i, (&g, &c)) in gpu_input.iter().zip(cpu_input.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff < TOL,
                "conv1d output[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6}"
            );
        }

        // ── Compare conv_state ──
        assert_eq!(gpu_state.len(), cpu_conv_state.len());
        for (i, (&g, &c)) in gpu_state.iter().zip(cpu_conv_state.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff < TOL,
                "conv_state[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6}"
            );
        }

        eprintln!("deltanet conv1d GOAT ✓");
    }

    // ── Gating kernel (T7) ─────────────────────────────────────────────────

    /// GOAT proof: `deltanet_gating_f32` GPU matches CPU `silu(gate) * up`.
    ///
    /// Uses n=8 elements — the kernel is fully parameterized.
    #[test]
    fn test_deltanet_gating_matches_reference() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n: usize = 8;

        let gate: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
        let up: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // ── GPU path ──
        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let up_handle = client.create_from_slice(f32::as_bytes(&up));
        let output_handle = client.empty(n * std::mem::size_of::<f32>());

        unsafe {
            DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_handle,
                up_handle,
                output_handle.clone(),
                n,
            );
        }

        let gpu_bytes = client.read_one(output_handle).expect("read output");
        let gpu_output = f32::from_bytes(&gpu_bytes);

        // ── CPU reference: silu(g) * u where silu(x) = x * sigmoid(x) ──
        let cpu_output: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let sig = 1.0 / (1.0 + (-g).exp());
                g * sig * u
            })
            .collect();

        // ── Compare ──
        assert_eq!(gpu_output.len(), cpu_output.len());
        for (i, (&g, &c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
            let diff = (g - c).abs();
            assert!(
                diff < TOL,
                "gating[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6}"
            );
        }

        eprintln!("deltanet gating GOAT ✓");
    }

    // ── Issue 642 F2: fused SwiGLU from concatenated gate_up buffer ──

    /// Verify DeltanetGatingConcatCubeCL matches DeltanetGatingCubeCL on the
    /// same gate/up values.
    #[test]
    fn test_deltanet_gating_concat_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n: usize = 128; // realistic head_dim-sized FFN slice

        let mut seed = 42u32;
        let mut lcg = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 4.0 - 2.0
        };
        let gate: Vec<f32> = (0..n).map(|_| lcg()).collect();
        let up: Vec<f32> = (0..n).map(|_| lcg()).collect();

        // Separate path: DeltanetGatingCubeCL
        let gate_h = client.create_from_slice(f32::as_bytes(&gate));
        let up_h = client.create_from_slice(f32::as_bytes(&up));
        let sep_out = client.empty(n * std::mem::size_of::<f32>());
        unsafe {
            DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_h,
                up_h,
                sep_out.clone(),
                n,
            );
        }
        let sep_bytes = client.read_one(sep_out).expect("read separate output");
        let sep_output = f32::from_bytes(&sep_bytes);

        // Concatenated path: build [gate | up] buffer, then DeltanetGatingConcatCubeCL
        let mut gate_up = gate.clone();
        gate_up.extend_from_slice(&up);
        let gate_up_h = client.create_from_slice(f32::as_bytes(&gate_up));
        let concat_out = client.empty(n * std::mem::size_of::<f32>());
        unsafe {
            DeltanetGatingConcatCubeCL::launch::<ActiveRuntime>(
                &client,
                gate_up_h,
                concat_out.clone(),
                n,
            );
        }
        let concat_bytes = client.read_one(concat_out).expect("read concat output");
        let concat_output = f32::from_bytes(&concat_bytes);

        // Compare
        assert_eq!(concat_output.len(), sep_output.len());
        let mut max_err = 0.0f32;
        for (i, (&s, &c)) in sep_output.iter().zip(concat_output.iter()).enumerate() {
            let err = (s - c).abs();
            max_err = max_err.max(err);
            assert!(
                err < 1e-6,
                "gating[{i}] separate={s:.6} concat={c:.6} err={err:.2e}"
            );
        }
        println!("fused concat SwiGLU (n={n}): max_err vs separate = {max_err}");
    }

    // ── Issue 673 guards ─────────────────────────────────────────────────

    /// Issue 673 Bug B: the legacy recurrence kernel's `cube_size` + smem tree
    /// are hardcoded to 128 — `supports()` must reflect that. Pre-fix, a
    /// head_dim≠128 launch silently misassigned heads (head_dim=64:
    /// double-assigns head 0, skips half the heads, reads uninitialized smem)
    /// instead of failing.
    #[test]
    fn test_legacy_recurrence_supports_issue673() {
        assert!(DeltanetRecurrenceCubeCL::supports(128));
        assert!(!DeltanetRecurrenceCubeCL::supports(64));
        assert!(!DeltanetRecurrenceCubeCL::supports(96));
        assert!(!DeltanetRecurrenceCubeCL::supports(256));
    }

    /// Issue 673 Bug D: an all-zero head must stay zero, not become NaN.
    /// Pre-fix: `inv_norm = 1/sqrt(0) = inf` and `0·inf = NaN` poisoned the
    /// recurrent state. The CPU reference guards `if sum_sq > 0.0`.
    #[test]
    fn test_l2_normalize_issue673_zero_head() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 4;
        let head_dim: usize = 128;
        let total = 3 * n_head * head_dim;

        // qkv: head 1 of Q all zeros; everything else generic nonzero.
        let mut qkv: Vec<f32> = (0..total)
            .map(|i| 0.01 + 0.001 * (i as f32 % 17.0))
            .collect();
        for c in 0..head_dim {
            qkv[head_dim + c] = 0.0;
        }

        // CPU expectation: zero head stays zero; Q/K heads unit-norm.
        let mut expected = qkv.clone();
        for section in 0..2 {
            for h in 0..n_head {
                let off = section * n_head * head_dim + h * head_dim;
                let norm: f32 = expected[off..off + head_dim]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();
                if norm > 0.0 {
                    for c in 0..head_dim {
                        expected[off + c] /= norm;
                    }
                }
            }
        }

        let qkv_h = client.create_from_slice(f32::as_bytes(&qkv));
        unsafe {
            L2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(&client, qkv_h.clone(), n_head, head_dim);
        }
        let bytes = client.read_one(qkv_h).expect("read qkv");
        let gpu = f32::from_bytes(&bytes);

        for (i, (&g, &e)) in gpu.iter().zip(expected.iter()).enumerate() {
            assert!(g.is_finite(), "qkv[{i}] = {g} (Issue 673 Bug D NaN)");
            assert!((g - e).abs() < 1e-5, "qkv[{i}] GPU={g:.6} CPU={e:.6}");
        }
        // The zero head must stay exactly zero.
        for c in 0..head_dim {
            assert_eq!(gpu[head_dim + c].to_bits(), 0u32, "zero head must stay zero");
        }
    }

    /// Issue 673 Bug D (expand variant): an all-zero source head must expand
    /// to all-zero heads, not NaN. Pre-fix: same missing zero-norm guard in
    /// `expand_and_l2_normalize_heads_f32`.
    #[test]
    fn test_expand_l2_normalize_issue673_zero_head() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_k_heads: usize = 8;
        let n_v_heads: usize = 16; // repeat_factor = 2
        let head_dim: usize = 64;

        let q_dim = n_k_heads * head_dim;
        let k_dim = n_k_heads * head_dim;
        let compact_len = q_dim + k_dim + n_v_heads * head_dim;

        // compact: [Q | K | V] — k-head 2 all zeros, rest generic.
        let mut compact = vec![0.0f32; compact_len];
        for h in 0..n_k_heads {
            for c in 0..head_dim {
                compact[h * head_dim + c] = 0.01 * (h as f32 + 1.0) + 0.001 * c as f32;
                compact[q_dim + h * head_dim + c] = 0.02 * (h as f32 + 1.0);
            }
        }
        for c in 0..head_dim {
            compact[q_dim + 2 * head_dim + c] = 0.0;
        }
        for h in 0..n_v_heads {
            for c in 0..head_dim {
                compact[q_dim + k_dim + h * head_dim + c] = 0.1 * (h as f32 + 1.0);
            }
        }

        // GPU.
        let compact_h = client.create_from_slice(f32::as_bytes(&compact));
        let expanded_len = 3 * n_v_heads * head_dim;
        let expanded_h = client.empty(expanded_len * std::mem::size_of::<f32>());
        unsafe {
            ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                &client,
                compact_h,
                expanded_h.clone(),
                n_k_heads,
                n_v_heads,
                head_dim,
            );
        }
        let bytes = client.read_one(expanded_h).expect("read expanded");
        let gpu = f32::from_bytes(&bytes);

        // Finite everywhere.
        for (i, &g) in gpu.iter().enumerate() {
            assert!(g.is_finite(), "expanded[{i}] = {g} (Issue 673 Bug D NaN)");
        }

        let qk_block = n_v_heads * head_dim;
        // Expanded K heads 2 and 10 (source k-head 2, all zeros) stay exactly zero.
        for j in [2usize, 10usize] {
            for c in 0..head_dim {
                let val = gpu[qk_block + j * head_dim + c];
                assert_eq!(
                    val.to_bits(),
                    0u32,
                    "expanded K head {j} (zero source) must stay zero, got {val}"
                );
            }
        }
        // A nonzero expanded K head (source k-head 0) is unit-norm.
        let norm: f32 = (0..head_dim)
            .map(|c| {
                let v = gpu[qk_block + c];
                v * v
            })
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expanded K head 0 norm = {norm}, expected 1.0");
        // V copied verbatim.
        for h in 0..n_v_heads {
            for c in 0..head_dim {
                let got = gpu[2 * qk_block + h * head_dim + c];
                let want = compact[q_dim + k_dim + h * head_dim + c];
                assert_eq!(got.to_bits(), want.to_bits(), "V[{h},{c}] must copy verbatim");
            }
        }
    }
}
