//! Issue 615 T5 — DeltaNet recurrence CUDA kernels (cudarc + nvrtc).
//!
//! Ports the CubeCL DeltaNet linear recurrence path to raw CUDA:
//! - `conv1d_f32` — depthwise conv1d + SiLU preprocessing
//! - `beta_decay_f32` — per-head beta/decay from raw a/b projections
//! - `expand_and_l2_normalize_heads_f32` — expand Q/K heads + L2-norm + V copy
//! - `recurrence_f32` — gated delta-rule recurrence (the hard part)
//! - `z_gating_f32` — z-gated output
//!
//! ## Why this is the hardest part
//!
//! The recurrence has **data dependencies between heads**: each head maintains
//! a `[head_dim × head_dim]` state matrix, and the delta-rule update reads
//! the current state to compute a delta before writing back. The algorithm is:
//!
//! ```text
//! S_t = diag(decay) * S_{t-1}                    (decay)
//! kv_mem = S_t * k                                (retrieve)
//! delta = beta * (v - kv_mem)                     (delta)
//! S_t += delta ⊗ k^T                              (update — depends on kv_mem)
//! output = (S_t * q) / sqrt(d)                    (read)
//! ```
//!
//! The retrieve + update + read all involve per-row dot products over head_dim,
//! requiring workgroup-wide reductions. Each row is processed serially.
//!
//! ## Dispatch summary
//!
//! | Kernel | Grid | Block | smem |
//! |---|---|---|---|
//! | `conv1d_f32` | `ceil(conv_dim / 256)` | 256 | 0 |
//! | `beta_decay_f32` | 1 | `n_head` | 0 |
//! | `expand_and_l2_normalize_heads_f32` | `ceil(3*n_v*hd / 256)` | 256 | 0 |
//! | `expand_and_l2_normalize_heads_rows_f32` | `ceil(p*3*n_v*hd / 256)` | 256 | 0 |
//! | `expand_and_l2_normalize_heads_rows_v2_f32` | `(p, 3)` | 1024 | `n_k × 4` |
//! | `recurrence_f32` | `n_v_heads` | `head_dim` | `head_dim × 4` |
//! | `z_gating_f32` | `ceil(n / 256)` | 256 | 0 |

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, DevicePtr, LaunchConfig};
use cudarc::driver::PushKernelArg;

/// Plan 603 R2 — the half-precision recurrent-state residency format.
/// The state lives as 16-bit halves (half the per-token DRAM traffic); the
/// recurrence compute stays f32 (hardware cvt on load — exact; `cvt.rn` on
/// store — round-to-nearest-even, deterministic). f16 = 11-bit mantissa +
/// saturating clamp at ±65504; bf16 = f32 range at 8-bit mantissa.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfStateFmt {
    F16,
    Bf16,
}

/// Combined CUDA source for all DeltaNet recurrence kernels.
///
/// Uses dynamic shared memory (`extern __shared__`) for the recurrence kernel's
/// per-row dot-product reductions.
const DELTANET_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// Conv1d + SiLU kernel (depthwise, single-token)
// ---------------------------------------------------------------------------

extern "C" __global__ void conv1d_f32(
    float* __restrict__ input,         // [conv_dim] (in-place: conv output)
    const float* __restrict__ weight,  // [conv_dim * kernel_size]
    float* __restrict__ conv_state,    // [conv_dim * kernel_size] (sliding window)
    const int conv_dim,
    const int kernel_size)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int state_off = ch * kernel_size;

    // Shift conv_state left by 1, append new input
    for (int k = 0; k < kernel_size - 1; k++) {
        conv_state[state_off + k] = conv_state[state_off + k + 1];
    }
    conv_state[state_off + kernel_size - 1] = input[ch];

    // Depthwise convolution
    const int weight_off = ch * kernel_size;
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        sum += conv_state[state_off + k] * weight[weight_off + k];
    }

    // SiLU: x / (1 + exp(-x))
    input[ch] = sum / (1.0f + expf(-sum));
}

// ---------------------------------------------------------------------------
// Beta/decay kernel (one thread per head)
// ---------------------------------------------------------------------------

extern "C" __global__ void beta_decay_f32(
    const float* __restrict__ a_raw,     // [n_head]
    const float* __restrict__ b_raw,     // [n_head]
    const float* __restrict__ a_log,     // [n_head]
    const float* __restrict__ dt_bias,   // [n_head]
    float* __restrict__ beta_out,        // [n_head]
    float* __restrict__ decay_out,       // [n_head]
    const int n_head)
{
    const int h = threadIdx.x;
    if (h >= n_head) return;

    // beta = sigmoid(b_raw[h])
    float b_val = b_raw[h];
    beta_out[h] = 1.0f / (1.0f + expf(-b_val));

    // g = a_log[h] * softplus(a_raw[h] + dt_bias[h])
    float a_val = a_raw[h] + dt_bias[h];
    float sp;
    if (a_val > 20.0f) {
        sp = a_val;
    } else if (a_val < -20.0f) {
        sp = 0.0f;
    } else {
        sp = logf(1.0f + expf(a_val));
    }
    float g = a_log[h] * sp;
    decay_out[h] = expf(g);
}

// ---------------------------------------------------------------------------
// Expand Q/K heads (tiled broadcast) + L2-normalize + copy V
// ---------------------------------------------------------------------------

extern "C" __global__ void expand_and_l2_normalize_heads_f32(
    const float* __restrict__ compact,    // [Q(n_k×hd) | K(n_k×hd) | V(n_v×hd)]
    float* __restrict__ expanded,         // [Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]
    const int n_k_heads,
    const int n_v_heads,
    const int head_dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int qk_expanded_block = n_v_heads * head_dim;
    const int total_elems = 3 * qk_expanded_block;
    if (idx >= total_elems) return;

    const int section = idx / qk_expanded_block;  // 0=Q, 1=K, 2=V
    const int local_idx = idx % qk_expanded_block;
    const int out_head = local_idx / head_dim;
    const int col = local_idx % head_dim;

    if (section < 2) {
        // Q or K: expand + L2-normalize
        const int src_head = out_head % n_k_heads;
        const int compact_section_off = section * n_k_heads * head_dim;
        const int src_head_off = compact_section_off + src_head * head_dim;

        // Redundantly compute L2 norm (same as CubeCL l2_normalize_heads_f32).
        // Guard against zero vectors: CPU l2_normalize skips normalization for
        // sum_sq == 0 (outputs zeros). We match that behavior to avoid
        // 1/sqrt(0) = Inf which produces 0*Inf = NaN.
        float sq_sum = 0.0f;
        for (int c = 0; c < head_dim; c++) {
            float val = compact[src_head_off + c];
            sq_sum += val * val;
        }
        if (sq_sum > 0.0f) {
            float inv_norm = 1.0f / sqrtf(sq_sum);
            expanded[idx] = compact[src_head_off + col] * inv_norm;
        } else {
            expanded[idx] = 0.0f;
        }
    } else {
        // V: copy verbatim
        const int v_compact_off = 2 * n_k_heads * head_dim;
        expanded[idx] = compact[v_compact_off + local_idx];
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet recurrence (the hard part)
// ---------------------------------------------------------------------------

/// One block per head. blockDim.x = head_dim threads per block.
///
/// Algorithm per head (serial over rows, parallel over columns within a row):
/// 1. Decay: `S[row, col] *= decay[h]`
/// 2. For each row:
///    a. Retrieve: `kv_mem[row] = Σ_c S[row, c] * k[c]` (parallel dot reduction)
///    b. Delta: `delta[row] = beta[h] * (v[row] - kv_mem[row])` (thread 0, broadcast via smem)
///    c. Update: `S[row, c] += k[c] * delta[row]` (parallel — each thread updates its column)
/// 3. Read: for each row, `output[row] = (Σ_c S[row, c] * q[c]) / sqrt(d)`
///
/// Issue 610 ported: full workgroup reduction via shared memory (plane_sum
/// only reduces within a warp on Metal). Issue 612 ported: smem sized to
/// head_dim via dynamic `extern __shared__`.
extern "C" __global__ void recurrence_f32(
    const float* __restrict__ qkv,       // [3 * n_head * head_dim]
    const float* __restrict__ beta,      // [n_head]
    const float* __restrict__ decay,     // [n_head]
    float* __restrict__ state,           // [n_head * head_dim * head_dim] (persistent)
    float* __restrict__ output,          // [n_head * head_dim]
    const int head_dim,
    const int n_head)
{
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int cube_size = blockDim.x;  // = head_dim

    if (head >= n_head) return;

    const int head_off_q = head * head_dim;
    const int head_off_k = n_head * head_dim + head * head_dim;
    const int head_off_v = 2 * n_head * head_dim + head * head_dim;
    const int state_off = head * head_dim * head_dim;

    const float beta_val = beta[head];
    const float decay_val = decay[head];

    const float k_col = qkv[head_off_k + col];
    const float q_col = qkv[head_off_q + col];

    // Dynamic shared memory for per-row dot-product reductions
    extern __shared__ float smem[];

    // Step 1: Decay
    for (int row = 0; row < head_dim; row++) {
        int idx = state_off + row * head_dim + col;
        state[idx] *= decay_val;
    }

    // Steps 2-4: Retrieve + delta + update (serial over rows)
    for (int row = 0; row < head_dim; row++) {
        int idx = state_off + row * head_dim + col;
        float s_val = state[idx];
        float partial = s_val * k_col;

        // Parallel reduction
        smem[col] = partial;
        __syncthreads();

        for (int stride = cube_size / 2; stride >= 32; stride >>= 1) {
            if (col < stride) {
                smem[col] += smem[col + stride];
            }
            __syncthreads();
        }
        // Final warp (warp-synchronous, no syncthreads needed)
        if (col < 32) {
            volatile float* vsmem = smem;
            if (col < 16) vsmem[col] += vsmem[col + 16];
            if (col < 8)  vsmem[col] += vsmem[col + 8];
            if (col < 4)  vsmem[col] += vsmem[col + 4];
            if (col < 2)  vsmem[col] += vsmem[col + 2];
            if (col < 1)  vsmem[0] += vsmem[1];
        }
        __syncthreads();

        float kv_mem_row = smem[0];

        // Compute delta in thread 0, broadcast via smem[0].
        // Issue 616: the original port (Issue 610) did the ENTIRE state update
        // (S[row, :] += k[:] * delta_row) serially in thread 0 — 128 serial
        // global memory writes per row, with 127/128 threads idle. This was
        // the single biggest bottleneck in the cudarc forward (71% of GPU time
        // was in the pre_gemv section dominated by this kernel).
        //
        // Fix: thread 0 computes delta, broadcasts via smem[0], then ALL threads
        // update their own column in parallel — 1 global write per thread
        // instead of head_dim serial writes.
        if (col == 0) {
            float v_row = qkv[head_off_v + row];
            smem[0] = beta_val * (v_row - kv_mem_row);
        }
        __syncthreads();

        // Parallel state update: S[row, col] += k[col] * delta_row
        float delta_row = smem[0];
        state[state_off + row * head_dim + col] += k_col * delta_row;

        // Need sync before next row (all threads wrote state[row, :])
        __syncthreads();
    }

    // Step 5 (read): output[head, row] = Σ_c S[head, row, c] * q[head, c] / sqrt(d)
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int row = 0; row < head_dim; row++) {
        int idx = state_off + row * head_dim + col;
        float s_val = state[idx];
        float partial = s_val * q_col;

        smem[col] = partial;
        __syncthreads();

        for (int stride = cube_size / 2; stride >= 32; stride >>= 1) {
            if (col < stride) {
                smem[col] += smem[col + stride];
            }
            __syncthreads();
        }
        if (col < 32) {
            volatile float* vsmem = smem;
            if (col < 16) vsmem[col] += vsmem[col + 16];
            if (col < 8)  vsmem[col] += vsmem[col + 8];
            if (col < 4)  vsmem[col] += vsmem[col + 4];
            if (col < 2)  vsmem[col] += vsmem[col + 2];
            if (col < 1)  vsmem[0] += vsmem[1];
        }
        __syncthreads();

        float dot = smem[0];

        if (col == 0) {
            output[head * head_dim + row] = dot * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Z-gating kernel (elementwise)
// ---------------------------------------------------------------------------

extern "C" __global__ void z_gating_f32(
    float* __restrict__ output,    // [n] (in-place)
    const float* __restrict__ z,   // [n]
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float z_val = z[idx];
    // silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    float silu = z_val / (1.0f + expf(-z_val));
    output[idx] *= silu;
}

// ---------------------------------------------------------------------------
// Row-parallel recurrence (Issue 617)
//
// The original `recurrence_f32` uses one block per head (blockDim.x = head_dim)
// and iterates rows serially with 3+ syncthreads per row. For head_dim=128,
// that's 128 rows * 4 syncs = 512 sync barriers per head, plus 128 syncs in
// the read step = ~1024 barriers/head. For Ternary-Bonsai-27B (768 heads),
// that's ~786K sync barriers per token — latency-bound on barrier overhead.
//
// Key insight: rows are TRULY INDEPENDENT. Step 2c for row R writes S[R,:];
// step 2a for row R+1 reads S[R+1,:]. No overlap. The serial loop is
// artificial.
//
// This kernel uses a 2D grid: (n_head, head_dim). One warp (32 threads)
// per block, one row per block. Each thread handles head_dim/32 columns.
//
// Benefits:
//   - ZERO __syncthreads() in the kernel (single warp = self-synchronized
//     via implicit ordering + __syncwarp where cross-lane ordering matters)
//   - Warp-level reduction via __shfl_xor (5 ops, zero smem)
//   - 16 heads * 128 rows = 2048 blocks per layer — plenty of parallelism
//
// Correctness: bit-identical to `recurrence_f32` because the math is the
// same; only the parallelization pattern changes (rows parallel vs serial).
// Requires head_dim % 32 == 0.
// ---------------------------------------------------------------------------

extern "C" __global__ void recurrence_f32_parallel(
    const float* __restrict__ qkv,       // [3 * n_head * head_dim]
    const float* __restrict__ beta,      // [n_head]
    const float* __restrict__ decay,     // [n_head]
    float* __restrict__ state,           // [n_head * head_dim * head_dim] (persistent)
    float* __restrict__ output,          // [n_head * head_dim]
    const int head_dim,
    const int n_head)
{
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31
    const int WARP_SIZE = 32;
    const int COLS_PER_LANE = head_dim / WARP_SIZE;  // 4 for head_dim=128

    if (head >= n_head) return;
    if (row >= head_dim) return;

    const int head_off_q = head * head_dim;
    const int head_off_k = n_head * head_dim + head * head_dim;
    const int head_off_v = 2 * n_head * head_dim + head * head_dim;
    const int row_off = (head * head_dim + row) * head_dim;  // S[head][row][:]

    const float beta_val = beta[head];
    const float decay_val = decay[head];

    // ── Step 1: Decay — each thread decays its COLS_PER_LANE columns ──
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * WARP_SIZE + lane;
        state[row_off + col] *= decay_val;
    }
    __syncwarp();

    // ── Step 2: Retrieve kv_mem[row] = Σ_c S[row, c] * k[c] ──
    // Each thread accumulates its 4 column-wise partial products.
    float partial = 0.0f;
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * WARP_SIZE + lane;
        partial += state[row_off + col] * qkv[head_off_k + col];
    }

    // Warp reduction via __shfl_xor (no smem, no syncthreads).
    partial += __shfl_xor_sync(0xffffffff, partial, 16);
    partial += __shfl_xor_sync(0xffffffff, partial, 8);
    partial += __shfl_xor_sync(0xffffffff, partial, 4);
    partial += __shfl_xor_sync(0xffffffff, partial, 2);
    partial += __shfl_xor_sync(0xffffffff, partial, 1);
    // Now every lane has kv_mem[row].

    // ── Step 3: Delta = beta * (v[row] - kv_mem[row]) ──
    // Every lane computes it independently (uniform value across the warp).
    float v_row = qkv[head_off_v + row];
    float delta_row = beta_val * (v_row - partial);

    // ── Step 4: Update S[row, c] += k[c] * delta ──
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * WARP_SIZE + lane;
        state[row_off + col] += qkv[head_off_k + col] * delta_row;
    }
    __syncwarp();

    // ── Step 5: Read output[row] = (Σ_c S[row, c] * q[c]) / sqrt(d) ──
    partial = 0.0f;
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * WARP_SIZE + lane;
        partial += state[row_off + col] * qkv[head_off_q + col];
    }
    partial += __shfl_xor_sync(0xffffffff, partial, 16);
    partial += __shfl_xor_sync(0xffffffff, partial, 8);
    partial += __shfl_xor_sync(0xffffffff, partial, 4);
    partial += __shfl_xor_sync(0xffffffff, partial, 2);
    partial += __shfl_xor_sync(0xffffffff, partial, 1);
    float dot = partial;

    const float scale = 1.0f / sqrtf((float)head_dim);
    if (lane == 0) {
        output[head * head_dim + row] = dot * scale;
    }
}

// Fused single-pass recurrence (Issue 742)
//
// The row-parallel kernel makes 5 state accesses per call (3R + 2W across
// 4 loop phases: decay RW, kv-read, update RW, output-read). The fusion
// keeps the proven (head, row) grid + warp-per-row shape but touches the
// state ONCE per direction:
//   read  s             -> s_dec = s * decay
//   reduce sum(s_dec*k) -> kv_mem  (same butterfly order)
//   delta = beta * (v[row] - kv_mem)
//   write s_dec + k*delta          (decay;update composed in registers)
//   reduce sum(s_new*q) -> output  (same butterfly order)
//
// Bit-identity argument: per element the op sequence is IDENTICAL to
// recurrence_f32_parallel — FMUL decay, FFMA(s_dec, k, delta) update,
// FFMA accumulations in the same c order, the same __shfl_xor butterfly
// (16/8/4/2/1), and `dot * (1/sqrt(d))` output scaling. The only change
// is WHERE intermediates live (registers vs state-memory round-trips).
//
// TEMPLATE SPECIALIZATION (the Issue-706 lesson, live here): the first
// draft took head_dim as a runtime kernel arg — COLS_PER_LANE then a
// runtime value, so the per-lane k/q/s_dec arrays were dynamically
// indexed and demoted to LOCAL MEMORY (the unroll cannot happen; every
// array access is a local-memory round-trip). Measured: 0.72x vs the
// parallel kernel. The bound is a LAUNCHER fact, not a kernel input:
// head_dim comes from config at every call site, so the body is a
// template<int HD> instantiated at 64/128/256 with constexpr COLS ->
// full unroll, register-resident arrays.
template <int HD>
__device__ __forceinline__ void recurrence_fused_body(
    const float* __restrict__ qkv,       // [3 * n_head * HD]
    const float* __restrict__ beta,      // [n_head]
    const float* __restrict__ decay,     // [n_head]
    float* __restrict__ state,           // [n_head * HD * HD] (persistent)
    float* __restrict__ output,          // [n_head * HD]
    const int n_head)
{
    constexpr int COLS_PER_LANE = HD / 32;  // compile-time -> full unroll
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31

    if (head >= n_head) return;
    if (row >= HD) return;

    const int head_off_q = head * HD;
    const int head_off_k = n_head * HD + head * HD;
    const int head_off_v = 2 * n_head * HD + head * HD;
    const int row_off = (head * HD + row) * HD;  // S[head][row][:]

    const float beta_val = beta[head];
    const float decay_val = decay[head];

    // k[c] / q[c] per lane — L2-hot across the head's row blocks.
    float k_c[COLS_PER_LANE];
    float q_c[COLS_PER_LANE];
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * 32 + lane;
        k_c[c] = qkv[head_off_k + col];
        q_c[c] = qkv[head_off_q + col];
    }

    // ── Single read pass: decay in registers + kv_mem partial ──
    float partial = 0.0f;
    float s_dec[COLS_PER_LANE];
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * 32 + lane;
        float s = state[row_off + col];
        s_dec[c] = s * decay_val;
        partial += s_dec[c] * k_c[c];
    }
    partial += __shfl_xor_sync(0xffffffff, partial, 16);
    partial += __shfl_xor_sync(0xffffffff, partial, 8);
    partial += __shfl_xor_sync(0xffffffff, partial, 4);
    partial += __shfl_xor_sync(0xffffffff, partial, 2);
    partial += __shfl_xor_sync(0xffffffff, partial, 1);
    const float kv_mem = partial;

    const float v_row = qkv[head_off_v + row];
    const float delta_row = beta_val * (v_row - kv_mem);

    // ── Single write pass: updated state + output partial in one sweep ──
    float partial2 = 0.0f;
    #pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * 32 + lane;
        float s_new = s_dec[c] + k_c[c] * delta_row;
        state[row_off + col] = s_new;
        partial2 += s_new * q_c[c];
    }
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 16);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 8);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 4);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 2);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 1);

    const float scale = 1.0f / sqrtf((float)HD);
    if (lane == 0) {
        output[head * HD + row] = partial2 * scale;
    }
}

extern "C" __global__ void recurrence_f32_fused_hd64(
    const float* __restrict__ qkv, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ output, const int n_head)
{
    recurrence_fused_body<64>(qkv, beta, decay, state, output, n_head);
}

extern "C" __global__ void recurrence_f32_fused_hd128(
    const float* __restrict__ qkv, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ output, const int n_head)
{
    recurrence_fused_body<128>(qkv, beta, decay, state, output, n_head);
}

extern "C" __global__ void recurrence_f32_fused_hd256(
    const float* __restrict__ qkv, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ output, const int n_head)
{
    recurrence_fused_body<256>(qkv, beta, decay, state, output, n_head);
}

// ---------------------------------------------------------------------------
// Plan 603 R2 — half-precision recurrent state residency (f16/bf16 BITS,
// the Issue-734-T5 wscale pattern). The persistent state lives as 16-bit
// halves of the per-token DRAM traffic (0.302 → 0.151 GB/token at Bonsai-2's
// 48×48×128²); the COMPUTE stays f32 — load via hardware cvt (exact for
// normals AND subnormals — the gemv_q4k Issue-593 lesson), store via
// cvt.rn (round-to-nearest-even, deterministic across runs/processes).
// Per-element op sequence is otherwise IDENTICAL to recurrence_fused_body,
// so the only numerics delta vs the f32 lane is the state STORE rounding
// (f16: ≤2^-11 relative per step; bf16: ≤2^-8, f32 range — no clamp).
// NVRTC compiles this source standalone (no CUDA headers) — inline PTX wraps
// the converters, same as gemv_ternary_cuda_raw's f16_bits_to_f32.
__device__ __forceinline__ float h16_bits_to_f32(unsigned short b)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}
__device__ __forceinline__ unsigned short f32_to_h16_bits_rn(float f)
{
    // Saturation instead of inf: a pathological |s| > 65504 degrades the
    // state rather than poisoning it (deterministic clamp pre-cvt).
    const float clamped = fminf(fmaxf(f, -65504.0f), 65504.0f);
    unsigned short b;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(b) : "f"(clamped));
    return b;
}
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short b)
{
    float f;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}
__device__ __forceinline__ unsigned short f32_to_bf16_bits_rn(float f)
{
    unsigned short b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(f));
    return b;
}

template <int HD, int FMT>  // FMT: 0 = f16, 1 = bf16
__device__ __forceinline__ void recurrence_fused_half_body(
    const float* __restrict__ qkv,       // [3 * n_head * HD]
    const float* __restrict__ beta,      // [n_head]
    const float* __restrict__ decay,     // [n_head]
    unsigned short* __restrict__ state,  // [n_head * HD * HD] persistent, half bits
    float* __restrict__ output,          // [n_head * HD]
    const int n_head)
{
    constexpr int COLS_PER_LANE = HD / 32;
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31

    if (head >= n_head) return;
    if (row >= HD) return;

    const int head_off_q = head * HD;
    const int head_off_k = n_head * HD + head * HD;
    const int head_off_v = 2 * n_head * HD + head * HD;
    const int row_off = (head * HD + row) * HD;  // S[head][row][:]

    const float beta_val = beta[head];
    const float decay_val = decay[head];

    float k_c[COLS_PER_LANE];
    float q_c[COLS_PER_LANE];
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * 32 + lane;
        k_c[c] = qkv[head_off_k + col];
        q_c[c] = qkv[head_off_q + col];
    }

    // Single read pass: exact widen to f32, decay in registers, kv_mem
    // partial — the same op order as recurrence_fused_body.
    float partial = 0.0f;
    float s_dec[COLS_PER_LANE];
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * 32 + lane;
        const unsigned short sb = state[row_off + col];
        const float s = (FMT == 0) ? h16_bits_to_f32(sb) : bf16_bits_to_f32(sb);
        s_dec[c] = s * decay_val;
        partial += s_dec[c] * k_c[c];
    }
    partial += __shfl_xor_sync(0xffffffff, partial, 16);
    partial += __shfl_xor_sync(0xffffffff, partial, 8);
    partial += __shfl_xor_sync(0xffffffff, partial, 4);
    partial += __shfl_xor_sync(0xffffffff, partial, 2);
    partial += __shfl_xor_sync(0xffffffff, partial, 1);
    const float kv_mem = partial;

    const float v_row = qkv[head_off_v + row];
    const float delta_row = beta_val * (v_row - kv_mem);

    // Single write pass: RN-round the new state to half bits + the output
    // partial computed from the PRE-rounding f32 s_new (the value the f32
    // lane's output would see — keeps the recurrence readout identical
    // given the same stored state).
    float partial2 = 0.0f;
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        int col = c * 32 + lane;
        const float s_new = s_dec[c] + k_c[c] * delta_row;
        state[row_off + col] =
            (FMT == 0) ? f32_to_h16_bits_rn(s_new) : f32_to_bf16_bits_rn(s_new);
        partial2 += s_new * q_c[c];
    }
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 16);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 8);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 4);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 2);
    partial2 += __shfl_xor_sync(0xffffffff, partial2, 1);

    const float scale = 1.0f / sqrtf((float)HD);
    if (lane == 0) {
        output[head * HD + row] = partial2 * scale;
    }
}

extern "C" __global__ void recurrence_half_fused_hd128_f16(
    const float* __restrict__ qkv, const float* __restrict__ beta,
    const float* __restrict__ decay, unsigned short* __restrict__ state,
    float* __restrict__ output, const int n_head)
{
    recurrence_fused_half_body<128, 0>(qkv, beta, decay, state, output, n_head);
}

extern "C" __global__ void recurrence_half_fused_hd128_bf16(
    const float* __restrict__ qkv, const float* __restrict__ beta,
    const float* __restrict__ decay, unsigned short* __restrict__ state,
    float* __restrict__ output, const int n_head)
{
    recurrence_fused_half_body<128, 1>(qkv, beta, decay, state, output, n_head);
}

// ---------------------------------------------------------------------------
// Issue 641 — DeltaNet BPTT backward with integrated forward recompute.
//
// Processes ALL T timesteps for ONE head. Each block handles one head.
// blockDim.x = head_dim (one thread per column), matching `recurrence_f32`.
//
// Two phases:
//   Phase 1 (forward recompute, t=0..T-1): reconstruct state trajectory from
//     qkv/beta/decay. Saves state_decayed, state, kv_mem, delta per timestep
//     to global memory (allocated by caller).
//   Phase 2 (BPTT backward, t=T-1..0): compute grad_q, grad_k, grad_v,
//     grad_beta, grad_decay per timestep using the saved trajectory.
//
// grad_S (the accumulated state gradient) lives in global memory
// (n_head * head_dim * head_dim * 4B = 48*128*128*4 = 3.1 MB for Bonsai).
// Shared memory layout: smem_reduce[head_dim] || smem_grad_delta[head_dim]
// — 2 * head_dim * 4 = 1024 bytes for head_dim=128. Used for reductions +
// storing per-row grad_delta for the second-pass grad_k_s2 computation.
//
// Memory layout: all [T][n_head][...] tensors are indexed as
//   [t * n_head * stride + head * stride + ...]
// ---------------------------------------------------------------------------

extern "C" __global__ void deltanet_bptt_recompute_f32(
    // ── Inputs (saved during forward training pass) ──
    const float* __restrict__ qkv_seq,       // [T * n_head * 3 * head_dim] — Q,K,V per timestep
    const float* __restrict__ beta_seq,      // [T * n_head]
    const float* __restrict__ decay_seq,     // [T * n_head]
    const float* __restrict__ grad_out_seq,  // [T * n_head * head_dim] — gradient of recurrence output
    // ── Outputs (gradients) ──
    float* __restrict__ grad_qkv_seq,        // [T * n_head * 3 * head_dim] — grad Q,K,V
    float* __restrict__ grad_beta_seq,       // [T * n_head]
    float* __restrict__ grad_decay_seq,      // [T * n_head]
    // ── Scratch buffers (allocated by caller, zeroed before launch) ──
    float* __restrict__ state_decayed_seq,   // [T * n_head * head_dim * head_dim]
    float* __restrict__ state_seq,           // [T * n_head * head_dim * head_dim]
    float* __restrict__ kv_mem_seq,          // [T * n_head * head_dim] (scratch: written in Phase 1)
    float* __restrict__ delta_seq,           // [T * n_head * head_dim] (scratch: written in Phase 1)
    float* __restrict__ grad_s,              // [n_head * head_dim * head_dim] — MUST be zeroed
    // ── Config ──
    const int T,
    const int head_dim,
    const int n_head)
{
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    if (head >= n_head) return;

    const int state_per_head = head_dim * head_dim;
    const int qkv_per_head = 3 * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Offsets for this head within the per-timestep QKV layout [3*n_head*head_dim]
    const int q_off = head * head_dim;                    // Q
    const int k_off = n_head * head_dim + head * head_dim; // K
    const int v_off = 2 * n_head * head_dim + head * head_dim; // V

    extern __shared__ float smem[];
    float* smem_reduce = smem;              // [head_dim] — reduction scratch
    float* smem_gdelta = smem + head_dim;   // [head_dim] — per-row grad_delta

    // ═══════════════════════════════════════════════════════════════════
    // Phase 1: Forward recompute (t = 0 .. T-1)
    // Reconstruct state_decayed, state, kv_mem, delta from qkv/beta/decay.
    // For t=0, state_prev = 0 (no prior state).
    // ═══════════════════════════════════════════════════════════════════

    for (int t = 0; t < T; t++) {
        const int t_head = t * n_head + head;
        const float beta_val = beta_seq[t_head];
        const float decay_val = decay_seq[t_head];

        const float* qkv_t = &qkv_seq[t * n_head * qkv_per_head];
        const float k_col = qkv_t[k_off + col];

        float* state_d = &state_decayed_seq[t * n_head * state_per_head + head * state_per_head];
        float* state_s = &state_seq[t * n_head * state_per_head + head * state_per_head];
        float* kv_mem = &kv_mem_seq[t * n_head * head_dim + head * head_dim];
        float* delta_arr = &delta_seq[t * n_head * head_dim + head * head_dim];

        // Step 1: Decay — state_decayed = decay * state_prev (or 0 for t=0)
        if (t == 0) {
            for (int row = 0; row < head_dim; row++)
                state_d[row * head_dim + col] = 0.0f;
        } else {
            const float* state_prev = &state_seq[(t-1) * n_head * state_per_head + head * state_per_head];
            for (int row = 0; row < head_dim; row++)
                state_d[row * head_dim + col] = decay_val * state_prev[row * head_dim + col];
        }
        __syncthreads();

        // Steps 2-4: Retrieve + delta + update (serial over rows)
        for (int row = 0; row < head_dim; row++) {
            float partial = state_d[row * head_dim + col] * k_col;
            smem_reduce[col] = partial;
            __syncthreads();

            for (int stride = head_dim / 2; stride >= 32; stride >>= 1) {
                if (col < stride) smem_reduce[col] += smem_reduce[col + stride];
                __syncthreads();
            }
            if (col < 32) {
                volatile float* v = smem_reduce;
                if (col < 16) v[col] += v[col + 16];
                if (col < 8)  v[col] += v[col + 8];
                if (col < 4)  v[col] += v[col + 4];
                if (col < 2)  v[col] += v[col + 2];
                if (col < 1)  v[0]    += v[1];
            }
            __syncthreads();

            float kv_row = smem_reduce[0];
            if (col == 0) {
                float v_row = qkv_t[v_off + row];
                kv_mem[row] = kv_row;
                float d = beta_val * (v_row - kv_row);
                delta_arr[row] = d;
                smem_reduce[0] = d;
            }
            __syncthreads();

            float delta_row = smem_reduce[0];
            state_s[row * head_dim + col] = state_d[row * head_dim + col] + k_col * delta_row;
            __syncthreads();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: BPTT backward (t = T-1 .. 0)
    // grad_s (global, zeroed by caller) accumulates dL/dS.
    // ═══════════════════════════════════════════════════════════════════

    float* gs = &grad_s[head * state_per_head];

    for (int t = T - 1; t >= 0; t--) {
        const int t_head = t * n_head + head;
        const float beta_val = beta_seq[t_head];
        const float decay_val = decay_seq[t_head];

        const float* qkv_t = &qkv_seq[t * n_head * qkv_per_head];
        const float k_col = qkv_t[k_off + col];
        const float q_col = qkv_t[q_off + col];
        const float* dy = &grad_out_seq[t * n_head * head_dim + head * head_dim];

        const float* state_d = &state_decayed_seq[t * n_head * state_per_head + head * state_per_head];
        const float* state_s = &state_seq[t * n_head * state_per_head + head * state_per_head];
        const float* kv_mem = &kv_mem_seq[t * n_head * head_dim + head * head_dim];
        const float* delta_arr = &delta_seq[t * n_head * head_dim + head * head_dim];

        float* grad_qkv_t = &grad_qkv_seq[t * n_head * qkv_per_head];

        // ── Step 5 backward: y_t = S_t @ q_t * scale ──
        // grad_S[r,c] += dy[r] * q[c] * scale
        // grad_q[c] = sum_r S_t[r,c] * dy[r] * scale
        float grad_q_col = 0.0f;
        for (int row = 0; row < head_dim; row++) {
            float dy_r = dy[row] * scale;
            gs[row * head_dim + col] += dy_r * q_col;
            grad_q_col += state_s[row * head_dim + col] * dy_r;
        }
        grad_qkv_t[q_off + col] = grad_q_col;

        // ── Pass 1 (rows): compute grad_delta per row, accumulate grad_k_s4,
        //    grad_beta, grad_v, and the grad_S update from step 2 backward.
        //    Store grad_delta[row] in smem_gdelta for the grad_k_s2 pass.
        float grad_k_s4 = 0.0f;
        for (int row = 0; row < head_dim; row++) {
            float gs_row = gs[row * head_dim + col];
            float partial = gs_row * k_col;  // grad_delta partial
            smem_reduce[col] = partial;
            __syncthreads();

            for (int stride = head_dim / 2; stride >= 32; stride >>= 1) {
                if (col < stride) smem_reduce[col] += smem_reduce[col + stride];
                __syncthreads();
            }
            if (col < 32) {
                volatile float* v = smem_reduce;
                if (col < 16) v[col] += v[col + 16];
                if (col < 8)  v[col] += v[col + 8];
                if (col < 4)  v[col] += v[col + 4];
                if (col < 2)  v[col] += v[col + 2];
                if (col < 1)  v[0]    += v[1];
            }
            __syncthreads();

            float grad_delta_row = smem_reduce[0];
            if (col == 0) smem_gdelta[row] = grad_delta_row;  // store for pass 2
            // grad_k_s4[c] += delta[r] * grad_S[r,c]
            grad_k_s4 += delta_arr[row] * gs_row;

            // Step 3 backward: grad_v = beta * grad_delta; grad_kv = -beta * grad_delta
            float grad_kv_row = -beta_val * grad_delta_row;
            if (col == 0) {
                float v_row = qkv_t[v_off + row];
                grad_qkv_t[v_off + row] = beta_val * grad_delta_row;
                grad_beta_seq[t_head] += grad_delta_row * (v_row - kv_mem[row]);
            }

            // Step 2 backward: grad_S += outer(grad_kv, k)
            gs[row * head_dim + col] += grad_kv_row * k_col;

            // Close the read-write window on smem_reduce[0]: every thread
            // reads grad_delta_row = smem_reduce[0] above, and without this
            // barrier thread 0 may overwrite slot 0 with the NEXT row's
            // `partial` before warps 1-3 have issued their read (a latent
            // cross-warp race — never observed to fire, but Phase 1's row
            // loop and recurrence_f32 both close with this barrier; this
            // restores the symmetry).
            __syncthreads();
        }
        __syncthreads();  // smem_gdelta fully written

        // ── Pass 2: grad_k_s2[c] = sum_r S'_t[r,c] * grad_kv[r]
        //    where grad_kv[r] = -beta * grad_delta[r] (stored in smem_gdelta)
        float grad_k_s2 = 0.0f;
        for (int row = 0; row < head_dim; row++) {
            grad_k_s2 += state_d[row * head_dim + col] * (-beta_val * smem_gdelta[row]);
        }
        grad_qkv_t[k_off + col] = grad_k_s4 + grad_k_s2;

        // ── Step 1 backward: S'_t = decay * S_{t-1} ──
        // grad_decay = <grad_S', S_{t-1}> (Frobenius inner product)
        // grad_S_{t-1} = decay * grad_S'  (scale for next iteration)
        if (t > 0) {
            const float* state_prev = &state_seq[(t-1) * n_head * state_per_head + head * state_per_head];
            float g_partial = 0.0f;
            for (int row = 0; row < head_dim; row++)
                g_partial += gs[row * head_dim + col] * state_prev[row * head_dim + col];

            smem_reduce[col] = g_partial;
            __syncthreads();
            for (int stride = head_dim / 2; stride >= 32; stride >>= 1) {
                if (col < stride) smem_reduce[col] += smem_reduce[col + stride];
                __syncthreads();
            }
            if (col < 32) {
                volatile float* v = smem_reduce;
                if (col < 16) v[col] += v[col + 16];
                if (col < 8)  v[col] += v[col + 8];
                if (col < 4)  v[col] += v[col + 4];
                if (col < 2)  v[col] += v[col + 2];
                if (col < 1)  v[0]    += v[1];
            }
            __syncthreads();
            if (col == 0) grad_decay_seq[t_head] = smem_reduce[0];
            for (int row = 0; row < head_dim; row++)
                gs[row * head_dim + col] *= decay_val;
        } else {
            if (col == 0) grad_decay_seq[t_head] = 0.0f;
        }
    }
}

// ===========================================================================
// Issue 742 T9.9 - the p-ROW batched verify family (Q4_K verify port).
// Sequential-state kernels (conv window, recurrence state) keep the
// per-position op order VERBATIM with the state carried in registers
// between positions (f32 store/load is bit-preserving, so the register
// values are exactly what the per-position kernel would read back) - the
// chunk results are bit-identical to p sequential decode steps, with ONE
// launch and ONE state read/write per layer instead of p.
// ===========================================================================

// Batched conv1d: thread per channel, kernel_size-wide window in registers
// (KS is compile-time - the Batch-49 lesson: a runtime-bound loop over a
// per-thread array demotes it to local memory). Per position the shift +
// append + ascending-k dot + SiLU are VERBATIM conv1d_f32; the final window
// write leaves exactly the state p sequential launches would leave.
template <int KS>
__device__ __forceinline__ void conv1d_rows_body(
    float* __restrict__ input,         // [p, conv_dim] (in-place)
    const float* __restrict__ weight,  // [conv_dim * KS]
    float* __restrict__ conv_state,    // [conv_dim * KS]
    const int conv_dim,
    const int p)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    const int state_off = ch * KS;
    const int weight_off = ch * KS;
    float st[KS];
#pragma unroll
    for (int k = 0; k < KS; k++) st[k] = conv_state[state_off + k];
    for (int t = 0; t < p; t++) {
        // Shift window left by 1, append the new input (same order).
#pragma unroll
        for (int k = 0; k < KS - 1; k++) st[k] = st[k + 1];
        st[KS - 1] = input[(long)t * conv_dim + ch];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < KS; k++) {
            sum += st[k] * weight[weight_off + k];
        }
        input[(long)t * conv_dim + ch] = sum / (1.0f + expf(-sum));
    }
#pragma unroll
    for (int k = 0; k < KS; k++) conv_state[state_off + k] = st[k];
}

extern "C" __global__ void conv1d_rows_f32(
    float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ conv_state,
    const int conv_dim,
    const int kernel_size,
    const int p)
{
    if (kernel_size == 4) {
        conv1d_rows_body<4>(input, weight, conv_state, conv_dim, p);
    } else if (kernel_size == 8) {
        conv1d_rows_body<8>(input, weight, conv_state, conv_dim, p);
    } else {
        // Unsupported kernel size - the launcher contract excludes it.
    }
}

// Batched beta/decay: one block of n_head threads; per (row, head) the ops
// are VERBATIM beta_decay_f32 (the a_log/dt_bias weights are shared).
extern "C" __global__ void beta_decay_rows_f32(
    const float* __restrict__ a_raw,     // [p, n_head]
    const float* __restrict__ b_raw,     // [p, n_head]
    const float* __restrict__ a_log,     // [n_head]
    const float* __restrict__ dt_bias,   // [n_head]
    float* __restrict__ beta_out,        // [p, n_head]
    float* __restrict__ decay_out,       // [p, n_head]
    const int n_head,
    const int p)
{
    const int h = threadIdx.x;
    if (h >= n_head) return;
    for (int t = 0; t < p; t++) {
        const long i = (long)t * n_head + h;
        float b_val = b_raw[i];
        beta_out[i] = 1.0f / (1.0f + expf(-b_val));

        float a_val = a_raw[i] + dt_bias[h];
        float sp;
        if (a_val > 20.0f) {
            sp = a_val;
        } else if (a_val < -20.0f) {
            sp = 0.0f;
        } else {
            sp = logf(1.0f + expf(a_val));
        }
        float g = a_log[h] * sp;
        decay_out[i] = expf(g);
    }
}

// Batched expand + L2-normalize: per (row, element) VERBATIM the decode
// kernel's math with row offsets on the compact/expanded bases.
extern "C" __global__ void expand_and_l2_normalize_heads_rows_f32(
    const float* __restrict__ compact,    // [p, compact_len]
    float* __restrict__ expanded,         // [p, 3 * n_v_heads * head_dim]
    const int n_k_heads,
    const int n_v_heads,
    const int head_dim,
    const int p)
{
    const int qk_expanded_block = n_v_heads * head_dim;
    const long per_row_out = 3L * qk_expanded_block;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)p * per_row_out) return;

    const int row = (int)(idx / per_row_out);
    const int local = (int)(idx % per_row_out);
    const int section = local / qk_expanded_block;  // 0=Q, 1=K, 2=V
    const int local_idx = local % qk_expanded_block;
    const int out_head = local_idx / head_dim;
    const int col = local_idx % head_dim;

    const long cbase = (long)row * (2L * n_k_heads * head_dim + n_v_heads * head_dim);

    if (section < 2) {
        const int src_head = out_head % n_k_heads;
        const long compact_section_off = cbase + section * n_k_heads * head_dim;
        const long src_head_off = compact_section_off + src_head * head_dim;
        float sq_sum = 0.0f;
        for (int c = 0; c < head_dim; c++) {
            float val = compact[src_head_off + c];
            sq_sum += val * val;
        }
        if (sq_sum > 0.0f) {
            float inv_norm = 1.0f / sqrtf(sq_sum);
            expanded[idx] = compact[src_head_off + col] * inv_norm;
        } else {
            expanded[idx] = 0.0f;
        }
    } else {
        const long v_compact_off = cbase + 2L * n_k_heads * head_dim;
        expanded[idx] = compact[v_compact_off + local_idx];
    }
}

// Issue 772 T2 residue (Bench 803): the Q/K half above had EVERY element
// thread re-read its full head_dim source row serially to compute sq_sum —
// n_v*head_dim threads x head_dim loads per (row, section) = 786k loads per
// section per row at the qwen38 GDN dims (n_v=48, hd=128), LSU/broadcast-
// bound at ~126 GB/s. V2 computes each source row's sum ONCE per
// (row, section): thread s < n_k_heads accumulates its source row in the
// EXACT scalar order (float4 groups loaded whole, accumulated .x->.y->.z->.w
// — the identical FP add sequence as the c-loop, so the staged sum is
// bit-identical to every legacy thread's), stages the n_k sums in shared
// memory, then every element thread normalizes from the staged sum with the
// same ops (`val * (1/sqrtf(sq_sum))`, the same `sq_sum > 0` branch). Output
// writes and the V copy are index-for-index identical to the legacy kernel.
// Grid (p, 3); block 1024; caller guards n_k_heads <= 1024 (launcher falls
// back to the legacy kernel otherwise).
extern "C" __global__ void expand_and_l2_normalize_heads_rows_v2_f32(
    const float* __restrict__ compact,    // [p, compact_len]
    float* __restrict__ expanded,         // [p, 3 * n_v_heads * head_dim]
    const int n_k_heads,
    const int n_v_heads,
    const int head_dim,
    const int p)
{
    const int row = blockIdx.x;
    const int section = blockIdx.y;  // 0=Q, 1=K, 2=V
    const int tid = threadIdx.x;

    const int qk_expanded_block = n_v_heads * head_dim;
    const long per_row_out = 3L * qk_expanded_block;
    const long cbase = (long)row * (2L * n_k_heads * head_dim + n_v_heads * head_dim);

    if (section == 2) {
        const long v_compact_off = cbase + 2L * n_k_heads * head_dim;
        float* out_v = expanded + (long)row * per_row_out + 2L * qk_expanded_block;
        for (int e = tid; e < qk_expanded_block; e += blockDim.x) {
            out_v[e] = compact[v_compact_off + e];
        }
        return;
    }

    __shared__ float s_sq[1024];
    if (tid < n_k_heads) {
        const long src_off = cbase + (long)section * n_k_heads * head_dim + (long)tid * head_dim;
        float sq_sum = 0.0f;
        if ((head_dim & 3) == 0) {
            // float4 groups accumulated .x->.y->.z->.w: the same add order
            // (and values, FMA contraction included) as the scalar c-loop.
            const float4* src4 = reinterpret_cast<const float4*>(compact + src_off);
            for (int c = 0; c < head_dim; c += 4) {
                const float4 v = src4[c >> 2];
                sq_sum += v.x * v.x;
                sq_sum += v.y * v.y;
                sq_sum += v.z * v.z;
                sq_sum += v.w * v.w;
            }
        } else {
            for (int c = 0; c < head_dim; c++) {
                const float val = compact[src_off + c];
                sq_sum += val * val;
            }
        }
        s_sq[tid] = sq_sum;
    }
    __syncthreads();

    for (int e = tid; e < qk_expanded_block; e += blockDim.x) {
        const int out_head = e / head_dim;
        const int col = e % head_dim;
        const int src_head = out_head % n_k_heads;
        const long src_off = cbase + (long)section * n_k_heads * head_dim + (long)src_head * head_dim;
        const long idx = (long)row * per_row_out + (long)section * qk_expanded_block + e;
        const float sq_sum = s_sq[src_head];
        if (sq_sum > 0.0f) {
            const float inv_norm = 1.0f / sqrtf(sq_sum);
            expanded[idx] = compact[src_off + col] * inv_norm;
        } else {
            expanded[idx] = 0.0f;
        }
    }
}

// Batched fused recurrence: the (head, row) warp-per-row grid VERBATIM with
// an outer position loop - the state slice stays in registers between
// positions (the exact values the per-position kernel would store/load) and
// every per-position op sequence is IDENTICAL to recurrence_fused_body, so
// the chunk is bit-identical to p sequential launches while touching the
// persistent state ONCE per direction per chunk.
extern "C" __global__ void recurrence_f32_fused_rows_hd128(
    const float* __restrict__ qkv,   // [p, 3 * n_head * 128]
    const float* __restrict__ beta,  // [p, n_head]
    const float* __restrict__ decay, // [p, n_head]
    float* __restrict__ state,       // [n_head * 128 * 128] persistent
    float* __restrict__ output,      // [p, n_head * 128]
    const int n_head,
    const int p)
{
    constexpr int HD = 128;
    constexpr int COLS_PER_LANE = HD / 32;  // compile-time -> full unroll
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;  // 0..31

    if (head >= n_head) return;
    if (row >= HD) return;

    const int head_off_q = head * HD;
    const int head_off_k = n_head * HD + head * HD;
    const int head_off_v = 2 * n_head * HD + head * HD;
    const int row_off = (head * HD + row) * HD;  // S[head][row][:]

    float s_c[COLS_PER_LANE];
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        s_c[c] = state[row_off + c * 32 + lane];
    }

    const long qkv_stride = 3L * n_head * HD;
    const float scale = 1.0f / sqrtf((float)HD);
    for (int t = 0; t < p; t++) {
        const float beta_val = beta[t * n_head + head];
        const float decay_val = decay[t * n_head + head];
        const float* qkv_t = qkv + t * qkv_stride;

        float k_c[COLS_PER_LANE];
        float q_c[COLS_PER_LANE];
#pragma unroll
        for (int c = 0; c < COLS_PER_LANE; c++) {
            int col = c * 32 + lane;
            k_c[c] = qkv_t[head_off_k + col];
            q_c[c] = qkv_t[head_off_q + col];
        }

        // Single read pass: decay in registers + kv_mem partial (verbatim).
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < COLS_PER_LANE; c++) {
            const float sd = s_c[c] * decay_val;
            s_c[c] = sd;
            partial += sd * k_c[c];
        }
        partial += __shfl_xor_sync(0xffffffffu, partial, 16);
        partial += __shfl_xor_sync(0xffffffffu, partial, 8);
        partial += __shfl_xor_sync(0xffffffffu, partial, 4);
        partial += __shfl_xor_sync(0xffffffffu, partial, 2);
        partial += __shfl_xor_sync(0xffffffffu, partial, 1);
        const float kv_mem = partial;

        const float v_row = qkv_t[head_off_v + row];
        const float delta_row = beta_val * (v_row - kv_mem);

        // Single write pass: updated state + output partial (verbatim).
        float partial2 = 0.0f;
#pragma unroll
        for (int c = 0; c < COLS_PER_LANE; c++) {
            const float s_new = s_c[c] + k_c[c] * delta_row;
            s_c[c] = s_new;
            partial2 += s_new * q_c[c];
        }
        partial2 += __shfl_xor_sync(0xffffffffu, partial2, 16);
        partial2 += __shfl_xor_sync(0xffffffffu, partial2, 8);
        partial2 += __shfl_xor_sync(0xffffffffu, partial2, 4);
        partial2 += __shfl_xor_sync(0xffffffffu, partial2, 2);
        partial2 += __shfl_xor_sync(0xffffffffu, partial2, 1);

        if (lane == 0) {
            output[(long)t * n_head * HD + head * HD + row] = partial2 * scale;
        }
    }
#pragma unroll
    for (int c = 0; c < COLS_PER_LANE; c++) {
        state[row_off + c * 32 + lane] = s_c[c];
    }
}
"#;

/// Holds compiled CUDA kernels for the DeltaNet recurrence path.
pub struct DeltanetKernels {
    conv1d: CudaFunction,
    beta_decay: CudaFunction,
    expand_l2: CudaFunction,
    recurrence: CudaFunction,
    /// Issue 617: row-parallel recurrence. One warp per row, 2D grid.
    /// Bit-identical to `recurrence`; ~4-8× faster (no syncthreads).
    recurrence_parallel: CudaFunction,
    /// Issue 742: fused single-pass recurrence — 1R+1W over the state
    /// vs the parallel kernel's 3R+2W. Bit-identical to
    /// `recurrence_parallel`. Template-specialized per head_dim (the
    /// Issue-706 runtime-bound lesson).
    recurrence_fused_hd64: CudaFunction,
    recurrence_fused_hd128: CudaFunction,
    recurrence_fused_hd256: CudaFunction,
    /// Plan 603 R2 — half-precision-state twins of the fused hd128 kernel
    /// (f16 / bf16 state bits; compute f32). Loaded unconditionally; the
    /// ternary decode lane dispatches them under `RIIR_GDN_STATE_HALF`.
    recurrence_half_hd128_f16: CudaFunction,
    recurrence_half_hd128_bf16: CudaFunction,
    z_gating: CudaFunction,
    /// Issue 641: BPTT backward with integrated forward recompute.
    /// One block per head, processes all T timesteps (forward recompute +
    /// backward BPTT). The production GPU backward kernel.
    bptt_recompute: CudaFunction,
    /// Issue 742 T9.9 — the p-row batched verify family (Q4_K verify port):
    /// per (row, element) arithmetic VERBATIM the decode kernels', with the
    /// sequential state (conv window, recurrence S-matrix slice) carried in
    /// registers between positions — bit-identical to p sequential launches.
    conv1d_rows: CudaFunction,
    beta_decay_rows: CudaFunction,
    expand_l2_rows: CudaFunction,
    /// Issue 772 T2 residue (Bench 803 successor): per-(row, section) staged
    /// sq_sum — bit-identical to `expand_l2_rows` (see the kernel comment).
    expand_l2_rows_v2: CudaFunction,
    recurrence_fused_rows_hd128: CudaFunction,
    _module: Arc<CudaModule>,
}

/// Issue 772 T2 residue (Bench 803 → 809): V2 rows-kernel dispatch. DEFAULT
/// ON — the staged-sum kernel is bit-identical to the legacy per-element
/// kernel (same FP add sequence, same ops, index-for-index identical
/// writes), so the env is a KILL-SWITCH: `RIIR_EXPAND_L2_ROWS_LEGACY=1`
/// restores the legacy kernel. The launch counter is the vacuous guard —
/// bit-identical outputs mean ONLY the counter proves V2 actually ran
/// (the Bench-768 lesson).
static EXPAND_L2_ROWS_V2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static EXPAND_L2_ROWS_V2_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static EXPAND_L2_ROWS_V2_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn expand_l2_rows_v2_enabled() -> bool {
    // Env read exactly once (first caller wins the OnceLock); the AtomicBool
    // is the live value and the setter below is authoritative after.
    if EXPAND_L2_ROWS_V2_ENV
        .set(!matches!(
            std::env::var("RIIR_EXPAND_L2_ROWS_LEGACY")
                .unwrap_or_default()
                .to_lowercase()
                .as_str(),
            "1" | "on" | "true",
        ))
        .is_ok()
    {
        EXPAND_L2_ROWS_V2.store(
            EXPAND_L2_ROWS_V2_ENV.get().copied().unwrap_or(true),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    EXPAND_L2_ROWS_V2.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime override for one-binary A/B: `Some(false)` forces the legacy
/// kernel, `Some(true)` forces V2, `None` restores env/default resolution.
/// Test/bench-only (the production kill-switch is the env var).
#[cfg_attr(not(test), allow(dead_code))]
fn set_expand_l2_rows_v2(override_value: Option<bool>) {
    let v = override_value
        .or_else(|| EXPAND_L2_ROWS_V2_ENV.get().copied())
        .unwrap_or(true);
    EXPAND_L2_ROWS_V2.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Launches dispatched through the V2 staged-sum kernel (the vacuous guard).
#[cfg_attr(not(test), allow(dead_code))]
fn expand_l2_rows_v2_launches() -> usize {
    EXPAND_L2_ROWS_V2_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

impl DeltanetKernels {
    /// Compile all DeltaNet recurrence kernels via nvrtc.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, super::CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            DELTANET_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| super::CudarcKernelError::Compile(e.to_string()))?;

        let conv1d = module
            .load_function("conv1d_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let beta_decay = module
            .load_function("beta_decay_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let expand_l2 = module
            .load_function("expand_and_l2_normalize_heads_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence = module
            .load_function("recurrence_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_parallel = module
            .load_function("recurrence_f32_parallel")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_fused_hd64 = module
            .load_function("recurrence_f32_fused_hd64")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_fused_hd128 = module
            .load_function("recurrence_f32_fused_hd128")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_fused_hd256 = module
            .load_function("recurrence_f32_fused_hd256")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_half_hd128_f16 = module
            .load_function("recurrence_half_fused_hd128_f16")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_half_hd128_bf16 = module
            .load_function("recurrence_half_fused_hd128_bf16")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let z_gating = module
            .load_function("z_gating_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let bptt_recompute = module
            .load_function("deltanet_bptt_recompute_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let conv1d_rows = module
            .load_function("conv1d_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let beta_decay_rows = module
            .load_function("beta_decay_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let expand_l2_rows = module
            .load_function("expand_and_l2_normalize_heads_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let expand_l2_rows_v2 = module
            .load_function("expand_and_l2_normalize_heads_rows_v2_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let recurrence_fused_rows_hd128 = module
            .load_function("recurrence_f32_fused_rows_hd128")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;

        Ok(Self {
            conv1d,
            beta_decay,
            expand_l2,
            recurrence,
            recurrence_parallel,
            recurrence_fused_hd64,
            recurrence_fused_hd128,
            recurrence_fused_hd256,
            recurrence_half_hd128_f16,
            recurrence_half_hd128_bf16,
            z_gating,
            bptt_recompute,
            conv1d_rows,
            beta_decay_rows,
            expand_l2_rows,
            expand_l2_rows_v2,
            recurrence_fused_rows_hd128,
            _module: module,
        })
    }

    /// Depthwise conv1d + SiLU preprocessing.
    ///
    /// Shifts the conv state left by 1, appends the new input, applies
    /// the depthwise convolution, then SiLU activation. Modifies `input`
    /// in-place and `conv_state` in-place.
    pub fn launch_conv1d(
        &self,
        stream: &CudaStream,
        input: &cudarc::driver::safe::CudaSlice<f32>,
        weight: &cudarc::driver::safe::CudaSlice<f32>,
        conv_state: &cudarc::driver::safe::CudaSlice<f32>,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let grid_x = (conv_dim as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let conv_dim_i32 = conv_dim as i32;
        let kernel_size_i32 = kernel_size as i32;
        unsafe {
            stream
                .launch_builder(&self.conv1d)
                .arg(input)
                .arg(weight)
                .arg(conv_state)
                .arg(&conv_dim_i32)
                .arg(&kernel_size_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Compute per-head beta and decay from raw a/b projections.
    ///
    /// - `beta_out[h] = sigmoid(b_raw[h])`
    /// - `decay_out[h] = exp(a_log[h] * softplus(a_raw[h] + dt_bias[h]))`
    pub fn launch_beta_decay(
        &self,
        stream: &CudaStream,
        a_raw: &cudarc::driver::safe::CudaSlice<f32>,
        b_raw: &cudarc::driver::safe::CudaSlice<f32>,
        a_log: &cudarc::driver::safe::CudaSlice<f32>,
        dt_bias: &cudarc::driver::safe::CudaSlice<f32>,
        beta_out: &cudarc::driver::safe::CudaSlice<f32>,
        decay_out: &cudarc::driver::safe::CudaSlice<f32>,
        n_head: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let n_head_i32 = n_head as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n_head.max(1) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.beta_decay)
                .arg(a_raw)
                .arg(b_raw)
                .arg(a_log)
                .arg(dt_bias)
                .arg(beta_out)
                .arg(decay_out)
                .arg(&n_head_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Expand Q/K heads (tiled broadcast) + L2-normalize + copy V.
    ///
    /// Reads compact GEMV output `[Q(n_k×hd) | K(n_k×hd) | V(n_v×hd)]` and
    /// writes expanded `[Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]`.
    pub fn launch_expand_and_l2_normalize(
        &self,
        stream: &CudaStream,
        compact: &cudarc::driver::safe::CudaSlice<f32>,
        expanded: &cudarc::driver::safe::CudaSlice<f32>,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) -> Result<(), super::CudarcKernelError> {
        debug_assert!(
            n_v_heads.is_multiple_of(n_k_heads),
            "n_v_heads ({n_v_heads}) must be a multiple of n_k_heads ({n_k_heads})"
        );
        let total_elems = 3 * n_v_heads * head_dim;
        let grid_x = (total_elems as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_k_i32 = n_k_heads as i32;
        let n_v_i32 = n_v_heads as i32;
        let head_dim_i32 = head_dim as i32;
        unsafe {
            stream
                .launch_builder(&self.expand_l2)
                .arg(compact)
                .arg(expanded)
                .arg(&n_k_i32)
                .arg(&n_v_i32)
                .arg(&head_dim_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Gated DeltaNet recurrence (decode step).
    ///
    /// One block per head (`n_v_heads` blocks), `head_dim` threads per block.
    /// Updates the persistent `state` buffer in-place and writes `output`.
    ///
    /// - `qkv`: `[3 * n_head * head_dim]` (expanded layout)
    /// - `state`: `[n_head * head_dim * head_dim]` (persistent, read-write)
    /// - `output`: `[n_head * head_dim]` (write-only)
    pub fn launch_recurrence(
        &self,
        stream: &CudaStream,
        qkv: &cudarc::driver::safe::CudaSlice<f32>,
        beta: &cudarc::driver::safe::CudaSlice<f32>,
        decay: &cudarc::driver::safe::CudaSlice<f32>,
        state: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim * 4) as u32, // dynamic smem = head_dim floats
        };
        unsafe {
            stream
                .launch_builder(&self.recurrence)
                .arg(qkv)
                .arg(beta)
                .arg(decay)
                .arg(state)
                .arg(output)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 617 — Row-parallel recurrence. Same math as `launch_recurrence`,
    /// but uses a 2D grid (n_head, head_dim) with one warp per block. Each
    /// warp handles one row of the state matrix, eliminating the serial row
    /// loop and ~1024 syncthreads per head.
    ///
    /// Requires `head_dim % 32 == 0` (warp size). For head_dim=128, each lane
    /// handles 4 columns. Bit-identical to `launch_recurrence`.
    ///
    /// - `qkv`: `[3 * n_head * head_dim]` (expanded layout)
    /// - `state`: `[n_head * head_dim * head_dim]` (persistent, read-write)
    /// - `output`: `[n_head * head_dim]` (write-only)
    pub fn launch_recurrence_parallel(
        &self,
        stream: &CudaStream,
        qkv: &cudarc::driver::safe::CudaSlice<f32>,
        beta: &cudarc::driver::safe::CudaSlice<f32>,
        decay: &cudarc::driver::safe::CudaSlice<f32>,
        state: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
    ) -> Result<(), super::CudarcKernelError> {
        debug_assert!(
            head_dim.is_multiple_of(32),
            "recurrence_f32_parallel requires head_dim % 32 == 0; got {head_dim}"
        );
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let cfg = LaunchConfig {
            // 2D grid: (n_head, head_dim) — one block per (head, row).
            grid_dim: (n_head as u32, head_dim as u32, 1),
            // Single warp per block.
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0, // no smem — uses __shfl_xor
        };
        unsafe {
            stream
                .launch_builder(&self.recurrence_parallel)
                .arg(qkv)
                .arg(beta)
                .arg(decay)
                .arg(state)
                .arg(output)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 — Fused single-pass recurrence. Same math and grid shape
    /// as `launch_recurrence_parallel`, but touches the persistent state
    /// ONCE per direction (1R + 1W vs 3R + 2W) by keeping the decayed state
    /// in registers between the kv reduction and the update store.
    /// Bit-identical to `launch_recurrence_parallel` (identical per-element
    /// op order + identical butterfly reduction order).
    ///
    /// head_dim is TEMPLATE-SPECIALIZED (64/128/256 — the Issue-706
    /// runtime-bound lesson: a runtime head_dim demotes the per-lane
    /// arrays to local memory). Other head_dims: callers fall back to
    /// `launch_recurrence_parallel`.
    pub fn launch_recurrence_fused(
        &self,
        stream: &CudaStream,
        qkv: &cudarc::driver::safe::CudaSlice<f32>,
        beta: &cudarc::driver::safe::CudaSlice<f32>,
        decay: &cudarc::driver::safe::CudaSlice<f32>,
        state: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let n_head_i32 = n_head as i32;
        let cfg = LaunchConfig {
            // Same 2D grid as `recurrence_f32_parallel`: (n_head, head_dim),
            // one warp per block, one row per block.
            grid_dim: (n_head as u32, head_dim as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0, // no smem — registers + __shfl_xor
        };
        let f = match head_dim {
            64 => &self.recurrence_fused_hd64,
            128 => &self.recurrence_fused_hd128,
            256 => &self.recurrence_fused_hd256,
            other => {
                return Err(super::CudarcKernelError::InvalidArg(format!(
                    "recurrence_f32_fused is specialized for head_dim \
                     {{64,128,256}}; got {other} (use recurrence_parallel)"
                )))
            }
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(qkv)
                .arg(beta)
                .arg(decay)
                .arg(state)
                .arg(output)
                .arg(&n_head_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Plan 603 R2 — the half-precision-state twin of
    /// [`launch_recurrence_fused`] at head_dim 128: same grid shape, same
    /// per-element f32 op order; the persistent state is 16-bit half bits
    /// (widened exactly on load, RN-rounded on store). Only the numerics of
    /// the STORE differ from the f32 lane — see the kernel comment.
    pub fn launch_recurrence_fused_half_hd128(
        &self,
        stream: &CudaStream,
        qkv: &cudarc::driver::safe::CudaSlice<f32>,
        beta: &cudarc::driver::safe::CudaSlice<f32>,
        decay: &cudarc::driver::safe::CudaSlice<f32>,
        state: &cudarc::driver::safe::CudaSlice<u16>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        n_head: usize,
        fmt: HalfStateFmt,
    ) -> Result<(), super::CudarcKernelError> {
        let n_head_i32 = n_head as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 128, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = match fmt {
            HalfStateFmt::F16 => &self.recurrence_half_hd128_f16,
            HalfStateFmt::Bf16 => &self.recurrence_half_hd128_bf16,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(qkv)
                .arg(beta)
                .arg(decay)
                .arg(state)
                .arg(output)
                .arg(&n_head_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Apply z-gating in-place: `output[i] *= silu(z[i])`.
    pub fn launch_z_gating(
        &self,
        stream: &CudaStream,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        z: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let grid_x = (n as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        unsafe {
            stream
                .launch_builder(&self.z_gating)
                .arg(output)
                .arg(z)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 641 — Launch the BPTT recompute backward kernel.
    ///
    /// One block per head (grid = n_head). BlockDim = head_dim.
    /// Shared memory = 2 * head_dim * 4 bytes (smem_reduce + smem_gdelta).
    ///
    /// # Arguments
    /// - `qkv_seq`: `[T * n_head * 3 * head_dim]` — Q, K, V per timestep
    /// - `beta_seq`: `[T * n_head]`
    /// - `decay_seq`: `[T * n_head]`
    /// - `grad_out_seq`: `[T * n_head * head_dim]` — gradient of recurrence output
    /// - `grad_qkv_seq`: `[T * n_head * 3 * head_dim]` — output grad Q, K, V
    /// - `grad_beta_seq`: `[T * n_head]` — output, MUST be zeroed before launch
    /// - `grad_decay_seq`: `[T * n_head]` — output
    /// - `state_decayed_seq`, `state_seq`, `kv_mem_seq`, `delta_seq`: scratch,
    ///   sizes as annotated in the kernel doc. Content doesn't matter pre-launch.
    /// - `grad_s`: `[n_head * head_dim * head_dim]` — MUST be zeroed before launch
    /// - `T`, `head_dim`, `n_head`: dims
    #[allow(clippy::too_many_arguments)]
    pub fn launch_bptt_recompute(
        &self,
        stream: &CudaStream,
        qkv_seq: &cudarc::driver::safe::CudaSlice<f32>,
        beta_seq: &cudarc::driver::safe::CudaSlice<f32>,
        decay_seq: &cudarc::driver::safe::CudaSlice<f32>,
        grad_out_seq: &cudarc::driver::safe::CudaSlice<f32>,
        grad_qkv_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        grad_beta_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        grad_decay_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        state_decayed_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        state_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        kv_mem_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        delta_seq: &mut cudarc::driver::safe::CudaSlice<f32>,
        grad_s: &mut cudarc::driver::safe::CudaSlice<f32>,
        t_len: usize,
        head_dim: usize,
        n_head: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let t_i32 = t_len as i32;
        let hd_i32 = head_dim as i32;
        let nh_i32 = n_head as i32;
        // smem = 2 * head_dim floats (reduce + grad_delta)
        let smem_bytes = (2 * head_dim * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        unsafe {
            stream
                .launch_builder(&self.bptt_recompute)
                .arg(qkv_seq)
                .arg(beta_seq)
                .arg(decay_seq)
                .arg(grad_out_seq)
                .arg(grad_qkv_seq)
                .arg(grad_beta_seq)
                .arg(grad_decay_seq)
                .arg(state_decayed_seq)
                .arg(state_seq)
                .arg(kv_mem_seq)
                .arg(delta_seq)
                .arg(grad_s)
                .arg(&t_i32)
                .arg(&hd_i32)
                .arg(&nh_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    // ── Issue 742 T9.9: the p-row batched verify launchers ────────────────

    /// Batched conv1d + SiLU: processes `p` positions through the sliding
    /// window in registers — per position VERBATIM [`Self::launch_conv1d`]'s
    /// arithmetic; ONE state read/write per chunk. `kernel_size` must be 4
    /// or 8 (the template instantiations).
    ///
    /// # Safety
    ///
    /// Caller guarantees `input` covers `p * conv_dim`, `weight` covers
    /// `conv_dim * kernel_size`, `conv_state` covers `conv_dim * kernel_size`.
    /// Plan 556 Stage 1 — `input`/`conv_state` widened to [`DeviceSlice`]
    /// (slice or zero-copy view): the lane path passes per-lane views.
    pub fn launch_conv1d_rows(
        &self,
        stream: &CudaStream,
        input: &impl DevicePtr<f32>,
        weight: &impl DevicePtr<f32>,
        conv_state: &impl DevicePtr<f32>,
        conv_dim: usize,
        kernel_size: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        if kernel_size != 4 && kernel_size != 8 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "conv1d_rows: kernel_size {kernel_size} not in {{4, 8}}"
            )));
        }
        let grid_x = (conv_dim as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (cd_i, ks_i, p_i) = (conv_dim as i32, kernel_size as i32, p as i32);
        // Plan 556 — buffer params widened to `DevicePtr` (slice or zero-copy
        // view): the lane path passes per-lane views of the lane arenas. The
        // raw pointers ride the launch args directly (the `SyncOnDrop` guards
        // are no-ops unless the context manages stream synchronization).
        let (in_ptr, _sync_in) = input.device_ptr(stream);
        let (w_ptr, _sync_w) = weight.device_ptr(stream);
        let (st_ptr, _sync_st) = conv_state.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.conv1d_rows)
                .arg(&in_ptr)
                .arg(&w_ptr)
                .arg(&st_ptr)
                .arg(&cd_i)
                .arg(&ks_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched beta/decay over `p` rows (one block, n_head threads; the
    /// a_log/dt_bias weights are shared across rows).
    ///
    /// # Safety
    ///
    /// Caller guarantees `a_raw`/`b_raw`/`beta_out`/`decay_out` cover
    /// `p * n_head` and `a_log`/`dt_bias` cover `n_head`.
    pub fn launch_beta_decay_rows(
        &self,
        stream: &CudaStream,
        a_raw: &cudarc::driver::safe::CudaSlice<f32>,
        b_raw: &cudarc::driver::safe::CudaSlice<f32>,
        a_log: &cudarc::driver::safe::CudaSlice<f32>,
        dt_bias: &cudarc::driver::safe::CudaSlice<f32>,
        beta_out: &cudarc::driver::safe::CudaSlice<f32>,
        decay_out: &cudarc::driver::safe::CudaSlice<f32>,
        n_head: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n_head.max(1) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nh_i, p_i) = (n_head as i32, p as i32);
        unsafe {
            stream
                .launch_builder(&self.beta_decay_rows)
                .arg(a_raw)
                .arg(b_raw)
                .arg(a_log)
                .arg(dt_bias)
                .arg(beta_out)
                .arg(decay_out)
                .arg(&nh_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched expand + L2-normalize over `p` rows.
    ///
    /// # Safety
    ///
    /// Caller guarantees `compact` covers
    /// `p * (2 * n_k_heads + n_v_heads) * head_dim` and `expanded` covers
    /// `p * 3 * n_v_heads * head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_expand_l2_rows(
        &self,
        stream: &CudaStream,
        compact: &cudarc::driver::safe::CudaSlice<f32>,
        expanded: &cudarc::driver::safe::CudaSlice<f32>,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let total = p * 3 * n_v_heads * head_dim;
        if total == 0 {
            return Ok(());
        }
        // V2 staged-sum dispatch (Issue 772 T2 residue): bit-identical to the
        // legacy path below; n_k_heads > 1024 cannot stage its sums in the
        // 1024-wide shared tile, so those configs stay on the legacy kernel.
        if expand_l2_rows_v2_enabled() && n_k_heads <= 1024 {
            let cfg = LaunchConfig {
                grid_dim: (p as u32, 3, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            let (nk_i, nv_i, hd_i, p_i) = (
                n_k_heads as i32,
                n_v_heads as i32,
                head_dim as i32,
                p as i32,
            );
            unsafe {
                stream
                    .launch_builder(&self.expand_l2_rows_v2)
                    .arg(compact)
                    .arg(expanded)
                    .arg(&nk_i)
                    .arg(&nv_i)
                    .arg(&hd_i)
                    .arg(&p_i)
                    .launch(cfg)
                    .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
            }
            EXPAND_L2_ROWS_V2_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        let grid_x = (total as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nk_i, nv_i, hd_i, p_i) = (
            n_k_heads as i32,
            n_v_heads as i32,
            head_dim as i32,
            p as i32,
        );
        unsafe {
            stream
                .launch_builder(&self.expand_l2_rows)
                .arg(compact)
                .arg(expanded)
                .arg(&nk_i)
                .arg(&nv_i)
                .arg(&hd_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched fused recurrence for head_dim=128: the (head, row) warp grid
    /// with an outer position loop — per position VERBATIM
    /// [`Self::launch_recurrence_fused`]'s arithmetic with the state slice
    /// register-carried between positions (bit-identical to p sequential
    /// launches; ONE state read/write per chunk).
    ///
    /// # Safety
    ///
    /// Caller guarantees `qkv` covers `p * 3 * n_head * 128`, `beta`/`decay`
    /// cover `p * n_head`, `state` covers `n_head * 128 * 128`, `output`
    /// covers `p * n_head * 128`, and head_dim == 128.
    pub fn launch_recurrence_fused_rows_hd128(
        &self,
        stream: &CudaStream,
        qkv: &impl DevicePtr<f32>,
        beta: &impl DevicePtr<f32>,
        decay: &impl DevicePtr<f32>,
        state: &impl DevicePtr<f32>,
        output: &impl DevicePtr<f32>,
        n_head: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 128, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nh_i, p_i) = (n_head as i32, p as i32);
        // DevicePtr-widened (see conv1d_rows) — the lane path passes views.
        let (q_ptr, _sync_q) = qkv.device_ptr(stream);
        let (b_ptr, _sync_b) = beta.device_ptr(stream);
        let (d_ptr, _sync_d) = decay.device_ptr(stream);
        let (s_ptr, _sync_s) = state.device_ptr(stream);
        let (o_ptr, _sync_o) = output.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.recurrence_fused_rows_hd128)
                .arg(&q_ptr)
                .arg(&b_ptr)
                .arg(&d_ptr)
                .arg(&s_ptr)
                .arg(&o_ptr)
                .arg(&nh_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::safe::CudaContext;

    const TOL: f32 = 1e-4;

    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    // ── CPU references ──

    fn cpu_conv1d(
        input: &mut [f32],
        weight: &[f32],
        conv_state: &mut [f32],
        kernel_size: usize,
    ) {
        for (ch, input_val) in input.iter_mut().enumerate() {
            let state_off = ch * kernel_size;
            for k in 0..kernel_size - 1 {
                conv_state[state_off + k] = conv_state[state_off + k + 1];
            }
            conv_state[state_off + kernel_size - 1] = *input_val;
            let weight_off = ch * kernel_size;
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                sum += conv_state[state_off + k] * weight[weight_off + k];
            }
            *input_val = sum / (1.0 + (-sum).exp());
        }
    }

    fn cpu_beta_decay(
        a_raw: &[f32],
        b_raw: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        n_head: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut beta = vec![0.0f32; n_head];
        let mut decay = vec![0.0f32; n_head];
        for h in 0..n_head {
            beta[h] = 1.0 / (1.0 + (-b_raw[h]).exp());
            let a_val = a_raw[h] + dt_bias[h];
            let sp = if a_val > 20.0 {
                a_val
            } else if a_val < -20.0 {
                0.0
            } else {
                (1.0 + a_val.exp()).ln()
            };
            let g = a_log[h] * sp;
            decay[h] = g.exp();
        }
        (beta, decay)
    }

    fn cpu_recurrence(
        qkv: &[f32],
        beta: &[f32],
        decay: &[f32],
        state: &mut [f32],
        n_head: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n_head * head_dim];

        for head in 0..n_head {
            let head_off_q = head * head_dim;
            let head_off_k = n_head * head_dim + head * head_dim;
            let head_off_v = 2 * n_head * head_dim + head * head_dim;
            let state_off = head * head_dim * head_dim;
            let beta_val = beta[head];
            let decay_val = decay[head];

            // Step 1: decay
            for row in 0..head_dim {
                for col in 0..head_dim {
                    let idx = state_off + row * head_dim + col;
                    state[idx] *= decay_val;
                }
            }

            // Steps 2-4: retrieve + delta + update
            for row in 0..head_dim {
                let mut kv_mem_row = 0.0f32;
                for col in 0..head_dim {
                    let idx = state_off + row * head_dim + col;
                    kv_mem_row += state[idx] * qkv[head_off_k + col];
                }
                let v_row = qkv[head_off_v + row];
                let delta_row = beta_val * (v_row - kv_mem_row);
                for c in 0..head_dim {
                    let update_idx = state_off + row * head_dim + c;
                    state[update_idx] += qkv[head_off_k + c] * delta_row;
                }
            }

            // Step 5: read
            for row in 0..head_dim {
                let mut dot = 0.0f32;
                for col in 0..head_dim {
                    let idx = state_off + row * head_dim + col;
                    dot += state[idx] * qkv[head_off_q + col];
                }
                output[head * head_dim + row] = dot * scale;
            }
        }
        output
    }

    fn cpu_expand_and_l2(
        compact: &[f32],
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let qk_expanded_block = n_v_heads * head_dim;
        let total_elems = 3 * qk_expanded_block;
        let mut expanded = vec![0.0f32; total_elems];

        for (idx, expanded_val) in expanded.iter_mut().enumerate() {
            let section = idx / qk_expanded_block;
            let local_idx = idx % qk_expanded_block;
            let out_head = local_idx / head_dim;
            let col = local_idx % head_dim;

            if section < 2 {
                let src_head = out_head % n_k_heads;
                let compact_section_off = section * n_k_heads * head_dim;
                let src_head_off = compact_section_off + src_head * head_dim;
                let mut sq_sum = 0.0f32;
                for c in 0..head_dim {
                    let val = compact[src_head_off + c];
                    sq_sum += val * val;
                }
                let inv_norm = 1.0 / sq_sum.sqrt();
                *expanded_val = compact[src_head_off + col] * inv_norm;
            } else {
                let v_compact_off = 2 * n_k_heads * head_dim;
                *expanded_val = compact[v_compact_off + local_idx];
            }
        }
        expanded
    }

    // ── Tests ──

    #[test]
    fn test_conv1d_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // Bonsai-27B: conv_dim = 3 * n_v_heads * head_dim = 3 * 48 * 128 = 18432
        // Use a small subset for testing.
        let conv_dim = 512usize;
        let kernel_size = 4usize;

        let input: Vec<f32> = (0..conv_dim).map(|i| (i as f32) * 0.01 - 2.5).collect();
        let weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|i| (i as f32) * 0.001 - 0.5)
            .collect();
        let conv_state = vec![0.5f32; conv_dim * kernel_size];

        // CPU reference
        let mut input_cpu = input.clone();
        let mut state_cpu = conv_state.clone();
        cpu_conv1d(&mut input_cpu, &weight, &mut state_cpu, kernel_size);

        // GPU
        let input_dev = stream.clone_htod(&input).unwrap();
        let weight_dev = stream.clone_htod(&weight).unwrap();
        let state_dev = stream.clone_htod(&conv_state).unwrap();

        kernels
            .launch_conv1d(&stream, &input_dev, &weight_dev, &state_dev, conv_dim, kernel_size)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_input = vec![0f32; conv_dim];
        let mut gpu_state = vec![0f32; conv_dim * kernel_size];
        stream.memcpy_dtoh(&input_dev, &mut gpu_input).unwrap();
        stream.memcpy_dtoh(&state_dev, &mut gpu_state).unwrap();

        // Compare conv output (input buffer, post-SiLU)
        let mut max_rel = 0f32;
        for i in 0..conv_dim {
            let denom = input_cpu[i].abs().max(1e-6);
            let rel = (gpu_input[i] - input_cpu[i]).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        // Compare state
        let mut max_state_rel = 0f32;
        for i in 0..conv_dim * kernel_size {
            let denom = state_cpu[i].abs().max(1e-6);
            let rel = (gpu_state[i] - state_cpu[i]).abs() / denom;
            max_state_rel = max_state_rel.max(rel);
        }
        eprintln!(
            "[conv1d] conv_dim={conv_dim}, kernel={kernel_size}: input max_rel={max_rel:.4e}, state max_rel={max_state_rel:.4e}"
        );
        assert!(max_rel < 1e-4, "conv1d input max_rel {max_rel:.4e}");
        assert!(max_state_rel < 1e-4, "conv1d state max_rel {max_state_rel:.4e}");
    }

    #[test]
    fn test_beta_decay_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        let n_head = 48usize; // Bonsai-27B n_v_heads
        let a_raw: Vec<f32> = (0..n_head).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let b_raw: Vec<f32> = (0..n_head).map(|i| (i as f32) * 0.02 + 0.1).collect();
        let a_log: Vec<f32> = (0..n_head).map(|i| -0.1 - (i as f32) * 0.001).collect();
        let dt_bias: Vec<f32> = (0..n_head).map(|i| 0.05 + (i as f32) * 0.002).collect();

        let (beta_cpu, decay_cpu) = cpu_beta_decay(&a_raw, &b_raw, &a_log, &dt_bias, n_head);

        let a_raw_dev = stream.clone_htod(&a_raw).unwrap();
        let b_raw_dev = stream.clone_htod(&b_raw).unwrap();
        let a_log_dev = stream.clone_htod(&a_log).unwrap();
        let dt_bias_dev = stream.clone_htod(&dt_bias).unwrap();
        let beta_dev = stream.alloc_zeros::<f32>(n_head).unwrap();
        let decay_dev = stream.alloc_zeros::<f32>(n_head).unwrap();

        kernels
            .launch_beta_decay(
                &stream,
                &a_raw_dev,
                &b_raw_dev,
                &a_log_dev,
                &dt_bias_dev,
                &beta_dev,
                &decay_dev,
                n_head,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut beta_gpu = vec![0f32; n_head];
        let mut decay_gpu = vec![0f32; n_head];
        stream.memcpy_dtoh(&beta_dev, &mut beta_gpu).unwrap();
        stream.memcpy_dtoh(&decay_dev, &mut decay_gpu).unwrap();

        let mut max_beta_rel = 0f32;
        let mut max_decay_rel = 0f32;
        for i in 0..n_head {
            let denom = beta_cpu[i].abs().max(1e-6);
            max_beta_rel = max_beta_rel.max((beta_gpu[i] - beta_cpu[i]).abs() / denom);
            let denom = decay_cpu[i].abs().max(1e-6);
            max_decay_rel = max_decay_rel.max((decay_gpu[i] - decay_cpu[i]).abs() / denom);
        }
        eprintln!(
            "[beta_decay] n_head={n_head}: beta max_rel={max_beta_rel:.4e}, decay max_rel={max_decay_rel:.4e}"
        );
        assert!(max_beta_rel < 1e-5, "beta max_rel {max_beta_rel:.4e}");
        assert!(max_decay_rel < 1e-5, "decay max_rel {max_decay_rel:.4e}");
    }

    #[test]
    fn test_expand_and_l2_normalize_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // Bonsai-27B: n_k_heads=16, n_v_heads=48, head_dim=128.
        // Use smaller subset to keep test fast.
        let n_k_heads = 4usize;
        let n_v_heads = 12usize; // 3× expansion
        let head_dim = 128usize;

        // Compact layout: [Q(n_k×hd) | K(n_k×hd) | V(n_v×hd)]
        let compact_len = 2 * n_k_heads * head_dim + n_v_heads * head_dim;
        let compact: Vec<f32> = (0..compact_len).map(|i| (i as f32) * 0.001 - 0.5).collect();

        let cpu_expanded = cpu_expand_and_l2(&compact, n_k_heads, n_v_heads, head_dim);

        let compact_dev = stream.clone_htod(&compact).unwrap();
        let expanded_len = 3 * n_v_heads * head_dim;
        let expanded_dev = stream.alloc_zeros::<f32>(expanded_len).unwrap();

        kernels
            .launch_expand_and_l2_normalize(
                &stream,
                &compact_dev,
                &expanded_dev,
                n_k_heads,
                n_v_heads,
                head_dim,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_expanded = vec![0f32; expanded_len];
        stream.memcpy_dtoh(&expanded_dev, &mut gpu_expanded).unwrap();

        let mut max_rel = 0f32;
        for i in 0..expanded_len {
            let denom = cpu_expanded[i].abs().max(1e-6);
            let rel = (gpu_expanded[i] - cpu_expanded[i]).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!(
            "[expand_l2] n_k={n_k_heads}, n_v={n_v_heads}, hd={head_dim}: max_rel={max_rel:.4e}"
        );
        assert!(max_rel < 1e-4, "expand_l2 max_rel {max_rel:.4e}");
    }

    /// Issue 772 T2 residue (Bench 809): the V2 staged-sum rows kernel must
    /// be BIT-identical to the legacy per-element kernel — same FP add
    /// sequence for sq_sum, same ops, index-for-index identical writes.
    /// Poisoned output buffers prove full coverage (any unwritten element
    /// keeps the sentinel and fails); one fully-zero source row exercises
    /// the `sq_sum == 0` branch; run-twice re-launch pins V2 determinism;
    /// the launch counter is the vacuous guard.
    #[test]
    fn test_expand_l2_rows_v2_bit_identical_to_legacy() {
        const P_ROWS: &[usize] = &[1, 3, 8];
const POISON: f32 = 12_345.5;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // (n_k, n_v, hd): production dims + tiny + hd%4!=0 + single-head.
        const DIMS: &[(usize, usize, usize)] = &[
            (16, 48, 128),
            (2, 3, 16),
            (4, 4, 10),
            (1, 1, 128),
            (5, 7, 6),
        ];

        let launches_before = expand_l2_rows_v2_launches();
        let mut v2_launches_expected = 0usize;

        for &(n_k, n_v, hd) in DIMS {
            let compact_row = 2 * n_k * hd + n_v * hd;
            let expanded_len = 3 * n_v * hd;
            for &p in P_ROWS {
                let compact_len = p * compact_row;
                // Deterministic LCG data with mixed signs and magnitudes.
                let mut seed = 0x2545_F491_4F6C_DD1Du64 ^ (compact_len as u64);
                let mut compact: Vec<f32> = (0..compact_len)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let r = ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5;
                        r * 6.0
                    })
                    .collect();
                // Zero one full row (row p/2): every section's sq_sum on that
                // row becomes exactly 0 — exercises the `sq_sum > 0`
                // else-branch identically on both paths.
                let zero_row = p / 2;
                compact[zero_row * compact_row..(zero_row + 1) * compact_row].fill(0.0);
                let compact_dev = stream.clone_htod(&compact).unwrap();
                // Poisoned destination: outputs live in [-1, 1] (normalized)
                // or the compact range (V copy) — the sentinel can never be a
                // legit value, so a surviving sentinel = a coverage gap.
                let poison: Vec<f32> = vec![POISON; compact_len.max(expanded_len * p)];

                let run = |v2: bool| -> Vec<f32> {
                    set_expand_l2_rows_v2(Some(v2));
                    let expanded_dev = stream.clone_htod(&poison[..expanded_len * p]).unwrap();
                    kernels
                        .launch_expand_l2_rows(&stream, &compact_dev, &expanded_dev, n_k, n_v, hd, p)
                        .expect("launch");
                    stream.synchronize().expect("sync");
                    let mut out = vec![0f32; expanded_len * p];
                    stream.memcpy_dtoh(&expanded_dev, &mut out).unwrap();
                    out
                };

                let legacy = run(false);
                let v2_a = run(true);
                let v2_b = run(true);
                v2_launches_expected += 2;

                let mismatches: Vec<usize> = legacy
                    .iter()
                    .zip(v2_a.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                    .map(|(i, _)| i)
                    .collect();
                assert!(
                    mismatches.is_empty(),
                    "n_k={n_k} n_v={n_v} hd={hd} p={p}: {} bit mismatches, first at {:?} (legacy={:?} v2={:?})",
                    mismatches.len(),
                    mismatches.first(),
                    mismatches.first().map(|&i| legacy[i]),
                    mismatches.first().map(|&i| v2_a[i]),
                );
                // Full coverage on BOTH paths: no sentinel survived.
                assert!(
                    legacy.iter().chain(v2_a.iter()).all(|v| v.to_bits() != POISON.to_bits()),
                    "sentinel survived (coverage gap) n_k={n_k} n_v={n_v} hd={hd} p={p}",
                );
                // Run-twice determinism of the V2 kernel.
                assert!(
                    v2_a.iter().zip(v2_b.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "V2 run-twice nondeterminism at n_k={n_k} n_v={n_v} hd={hd} p={p}",
                );
                // The zero row normalized to exact zeros on the legacy path.
                let zout = zero_row * expanded_len;
                assert!(legacy[zout..zout + expanded_len].iter().all(|v| *v == 0.0));
            }
        }
        set_expand_l2_rows_v2(None);
        // Vacuous guard: every forced `Some(true)` launch routes the V2 branch
        // and increments the counter. `>=` because sibling lib tests in the
        // same binary may exercise the rows launcher concurrently (shared
        // static — an exact count would race).
        assert!(
            expand_l2_rows_v2_launches() - launches_before >= v2_launches_expected,
            "vacuous guard: V2 launch counter did not advance as expected",
        );
    }

    /// Kernel-only A/B at the production GDN dims (n_k=16, n_v=48, hd=128,
    /// p=2048) — the Bench 809 verdict instrument. Bench 803 attributed the
    /// 32 ms expand bucket to the legacy kernel's redundant serial sq_sum
    /// re-reads (~126 GB/s, "3-5× headroom"); the real traffic is 235 MB per
    /// layer (84 read + 151 written), so ~350 GB/s — memory-floor, and the
    /// redundancy was already free via warp broadcast. This probe measures
    /// legacy vs V2 interleaved so the attribution is settled on numbers.
    #[test]
    #[ignore = "kernel-only timing probe — needs an exclusive GPU window"]
    fn test_expand_l2_rows_v2_timing_probe() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        let (n_k, n_v, hd, p) = (16usize, 48usize, 128usize, 2048usize);
        let compact_row = 2 * n_k * hd + n_v * hd;
        let compact_len = p * compact_row;
        let expanded_len = p * 3 * n_v * hd;
        let mut seed = 0xDEFA_CE01u64;
        let compact: Vec<f32> = (0..compact_len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5
            })
            .collect();
        let compact_dev = stream.clone_htod(&compact).unwrap();
        let expanded_dev = stream.alloc_zeros::<f32>(expanded_len).unwrap();

        let read_bytes = (compact_len * 4) as f64;
        let write_bytes = (expanded_len * 4) as f64;
        let traffic_gb = (read_bytes + write_bytes) / 1e9;
        eprintln!(
            "[expand-timing] traffic per call: {traffic_gb:.3} GB (read {:.1} MB + write {:.1} MB)",
            read_bytes / 1e6,
            write_bytes / 1e6,
        );

        let timed = |v2: bool| -> f64 {
            set_expand_l2_rows_v2(Some(v2));
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            kernels
                .launch_expand_l2_rows(&stream, &compact_dev, &expanded_dev, n_k, n_v, hd, p)
                .expect("launch");
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        };

        // Warm both paths (NVRTC + caches).
        for v2 in [false, true] {
            timed(v2);
        }
        // Interleaved rounds, median-of-5 per arm.
        let mut legacy_t = Vec::with_capacity(5);
        let mut v2_t = Vec::with_capacity(5);
        for _ in 0..5 {
            legacy_t.push(timed(false));
            v2_t.push(timed(true));
        }
        legacy_t.sort_by(|a, b| a.total_cmp(b));
        v2_t.sort_by(|a, b| a.total_cmp(b));
        let (lm, vm) = (legacy_t[2], v2_t[2]);
        eprintln!(
            "[expand-timing] legacy median {lm:.3} ms ({:.0} GB/s) | v2 median {vm:.3} ms ({:.0} GB/s) | v2/legacy {:.3}",
            traffic_gb / (lm / 1e3),
            traffic_gb / (vm / 1e3),
            vm / lm,
        );
        set_expand_l2_rows_v2(None);
    }

    #[test]
    fn test_recurrence_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // Small but representative: n_head=4, head_dim=128 (CubeCL uses 128).
        // Test with head_dim=128 (the actual DeltaNet head_dim for Bonsai).
        let n_head = 4usize;
        let head_dim = 128usize;

        // qkv: [Q(n*hd) | K(n*hd) | V(n*hd)]
        let qkv_len = 3 * n_head * head_dim;
        let qkv: Vec<f32> = (0..qkv_len)
            .map(|i| {
                let s = (i as f32) * 0.001 - 0.3;
                // Mix signs for nontrivial dot products
                if i.is_multiple_of(7) { -s } else { s }
            })
            .collect();

        let beta: Vec<f32> = (0..n_head).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let decay: Vec<f32> = (0..n_head).map(|i| 0.9 - (i as f32) * 0.01).collect();

        let state_len = n_head * head_dim * head_dim;
        let state_init: Vec<f32> = (0..state_len)
            .map(|i| ((i % 1024) as f32) * 0.0001 - 0.05)
            .collect();

        // CPU reference
        let mut state_cpu = state_init.clone();
        let cpu_output = cpu_recurrence(&qkv, &beta, &decay, &mut state_cpu, n_head, head_dim);

        // GPU
        let qkv_dev = stream.clone_htod(&qkv).unwrap();
        let beta_dev = stream.clone_htod(&beta).unwrap();
        let decay_dev = stream.clone_htod(&decay).unwrap();
        let state_dev = stream.clone_htod(&state_init).unwrap();
        let output_dev = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();

        kernels
            .launch_recurrence(
                &stream,
                &qkv_dev,
                &beta_dev,
                &decay_dev,
                &state_dev,
                &output_dev,
                head_dim,
                n_head,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_output = vec![0f32; n_head * head_dim];
        stream.memcpy_dtoh(&output_dev, &mut gpu_output).unwrap();

        // Compare output
        let mut max_diff = 0f32;
        let mut worst = 0;
        for i in 0..n_head * head_dim {
            let diff = (gpu_output[i] - cpu_output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst = i;
            }
        }
        eprintln!(
            "[recurrence] n_head={n_head}, hd={head_dim}: max_diff={max_diff:.6e} (worst idx={worst}: gpu={:.6}, cpu={:.6})",
            gpu_output[worst], cpu_output[worst]
        );
        assert!(
            max_diff <= TOL,
            "recurrence max_diff={max_diff:.6} > {TOL}"
        );

        // Also compare state (the persistent recurrent state after the step)
        let mut gpu_state = vec![0f32; state_len];
        stream.memcpy_dtoh(&state_dev, &mut gpu_state).unwrap();
        // Use absolute difference for state — near-zero values have high relative error
        let mut max_state_abs = 0f32;
        for i in 0..state_len {
            let diff = (gpu_state[i] - state_cpu[i]).abs();
            max_state_abs = max_state_abs.max(diff);
        }
        eprintln!("[recurrence] state max_abs={max_state_abs:.4e}");
        assert!(max_state_abs < 1e-4, "recurrence state max_abs {max_state_abs:.4e}");
    }

    /// Issue 617: the row-parallel recurrence kernel must be FP-equivalent
    /// to both the CPU reference and the baseline `recurrence_f32` kernel.
    /// It's NOT bit-identical to the baseline (the reduction order differs:
    /// smem tree vs __shfl_xor) — but it's within the existing recurrence
    /// tolerance budget (TOL = 1e-4).
    #[test]
    fn test_recurrence_parallel_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // Production shape: n_head=16, head_dim=128 (Ternary-Bonsai-27B).
        let n_head = 16usize;
        let head_dim = 128usize;

        let qkv_len = 3 * n_head * head_dim;
        let qkv: Vec<f32> = (0..qkv_len)
            .map(|i| {
                let s = (i as f32) * 0.001 - 0.3;
                if i.is_multiple_of(7) { -s } else { s }
            })
            .collect();
        let beta: Vec<f32> = (0..n_head).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let decay: Vec<f32> = (0..n_head).map(|i| 0.9 - (i as f32) * 0.01).collect();
        let state_len = n_head * head_dim * head_dim;
        let state_init: Vec<f32> = (0..state_len)
            .map(|i| ((i % 1024) as f32) * 0.0001 - 0.05)
            .collect();

        // CPU reference
        let mut state_cpu = state_init.clone();
        let cpu_output = cpu_recurrence(&qkv, &beta, &decay, &mut state_cpu, n_head, head_dim);

        // GPU parallel kernel
        let qkv_dev = stream.clone_htod(&qkv).unwrap();
        let beta_dev = stream.clone_htod(&beta).unwrap();
        let decay_dev = stream.clone_htod(&decay).unwrap();
        let state_dev = stream.clone_htod(&state_init).unwrap();
        let output_dev = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();

        kernels
            .launch_recurrence_parallel(
                &stream, &qkv_dev, &beta_dev, &decay_dev,
                &state_dev, &output_dev, head_dim, n_head,
            )
            .expect("launch parallel");
        stream.synchronize().expect("sync");

        let mut gpu_output = vec![0f32; n_head * head_dim];
        stream.memcpy_dtoh(&output_dev, &mut gpu_output).unwrap();

        let mut max_diff = 0f32;
        for i in 0..n_head * head_dim {
            max_diff = max_diff.max((gpu_output[i] - cpu_output[i]).abs());
        }
        eprintln!(
            "[recurrence_parallel] n_head={n_head}, hd={head_dim}: max_diff={max_diff:.6e}"
        );
        // Use 10×TOL = 1e-3 to accommodate reduction-order differences
        // (smem tree vs __shfl_xor) — the existing TOL is 1e-4 vs CPU.
        assert!(
            max_diff <= 1e-3,
            "recurrence_parallel max_diff={max_diff:.6} > 1e-3"
        );
    }

    #[test]
    fn test_recurrence_fused_bit_identical_to_parallel() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // qwen38 GDN production shape (48 v-heads x 128) + two off-shapes.
        let geometries: [(usize, usize); 3] = [(48, 128), (16, 128), (4, 64)];
        for &(n_head, head_dim) in &geometries {
            let qkv_len = 3 * n_head * head_dim;
            // LCG-driven magnitudes (the deterministic-repro rule).
            let mut seed: u64 = 0x853c_49e4_7f13_2a51;
            let mut next = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };
            let qkv: Vec<f32> = (0..qkv_len).map(|_| next()).collect();
            let beta: Vec<f32> = (0..n_head).map(|_| 0.2 + 0.6 * next().abs()).collect();
            let decay: Vec<f32> = (0..n_head).map(|_| 0.7 + 0.29 * next().abs()).collect();
            let state_len = n_head * head_dim * head_dim;
            let state_init: Vec<f32> = (0..state_len).map(|_| next() * 0.1).collect();

            // 5 sequential steps on BOTH kernels; every state element AND
            // every output element must stay bit-identical at every step
            // (state evolution compounding check, not a single-shot one).
            let state_par = stream.clone_htod(&state_init).unwrap();
            let state_fus = stream.clone_htod(&state_init).unwrap();
            let out_par = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();
            let out_fus = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();
            let qkv_dev = stream.clone_htod(&qkv).unwrap();
            let beta_dev = stream.clone_htod(&beta).unwrap();
            let decay_dev = stream.clone_htod(&decay).unwrap();

            let mut host_par = vec![0f32; state_len];
            let mut host_fus = vec![0f32; state_len];
            let mut opar = vec![0f32; n_head * head_dim];
            let mut ofus = vec![0f32; n_head * head_dim];
            for step in 0..5 {
                kernels
                    .launch_recurrence_parallel(
                        &stream, &qkv_dev, &beta_dev, &decay_dev, &state_par, &out_par,
                        head_dim, n_head,
                    )
                    .expect("launch parallel");
                kernels
                    .launch_recurrence_fused(
                        &stream, &qkv_dev, &beta_dev, &decay_dev, &state_fus, &out_fus,
                        head_dim, n_head,
                    )
                    .expect("launch fused");
                stream.synchronize().expect("sync");
                stream.memcpy_dtoh(&state_par, &mut host_par).unwrap();
                stream.memcpy_dtoh(&state_fus, &mut host_fus).unwrap();
                stream.memcpy_dtoh(&out_par, &mut opar).unwrap();
                stream.memcpy_dtoh(&out_fus, &mut ofus).unwrap();
                for i in 0..state_len {
                    assert!(
                        host_par[i].to_bits() == host_fus[i].to_bits(),
                        "state diverged at step {step} geom ({n_head},{head_dim}) idx {i}: \
                         par={:e} fus={:e}",
                        host_par[i],
                        host_fus[i]
                    );
                }
                for i in 0..n_head * head_dim {
                    assert!(
                        opar[i].to_bits() == ofus[i].to_bits(),
                        "output diverged at step {step} geom ({n_head},{head_dim}) idx {i}: \
                         par={:e} fus={:e}",
                        opar[i],
                        ofus[i]
                    );
                }
            }
            eprintln!(
                "[recurrence_fused] bit-identical to parallel across 5 steps at \
                 ({n_head},{head_dim})"
            );
        }
    }

    #[test]
    fn test_recurrence_fused_matches_cpu() {
        // 3-way confidence: fused vs the independent scalar CPU reference
        // (same 1e-3 tolerance as the parallel kernel's CPU gate — the
        // reduction-order class vs serial CPU accumulation).
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // qwen38 GDN production shape.
        let n_head = 48usize;
        let head_dim = 128usize;

        let qkv_len = 3 * n_head * head_dim;
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let qkv: Vec<f32> = (0..qkv_len).map(|_| next()).collect();
        let beta: Vec<f32> = (0..n_head).map(|_| 0.2 + 0.6 * next().abs()).collect();
        let decay: Vec<f32> = (0..n_head).map(|_| 0.7 + 0.29 * next().abs()).collect();
        let state_len = n_head * head_dim * head_dim;
        let state_init: Vec<f32> = (0..state_len).map(|_| next() * 0.1).collect();

        let mut state_cpu = state_init.clone();
        let cpu_output = cpu_recurrence(&qkv, &beta, &decay, &mut state_cpu, n_head, head_dim);

        let qkv_dev = stream.clone_htod(&qkv).unwrap();
        let beta_dev = stream.clone_htod(&beta).unwrap();
        let decay_dev = stream.clone_htod(&decay).unwrap();
        let state_dev = stream.clone_htod(&state_init).unwrap();
        let output_dev = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();

        kernels
            .launch_recurrence_fused(
                &stream, &qkv_dev, &beta_dev, &decay_dev, &state_dev, &output_dev,
                head_dim, n_head,
            )
            .expect("launch fused");
        stream.synchronize().expect("sync");

        let mut gpu_output = vec![0f32; n_head * head_dim];
        stream.memcpy_dtoh(&output_dev, &mut gpu_output).unwrap();
        let mut gpu_state = vec![0f32; state_len];
        stream.memcpy_dtoh(&state_dev, &mut gpu_state).unwrap();

        let mut max_diff = 0f32;
        for i in 0..n_head * head_dim {
            max_diff = max_diff.max((gpu_output[i] - cpu_output[i]).abs());
        }
        let mut max_state_diff = 0f32;
        for i in 0..state_len {
            max_state_diff = max_state_diff.max((gpu_state[i] - state_cpu[i]).abs());
        }
        eprintln!(
            "[recurrence_fused] n_head={n_head}, hd={head_dim}: out max_diff={max_diff:.6e} \
             state max_diff={max_state_diff:.6e}"
        );
        assert!(
            max_diff <= 1e-3,
            "recurrence_fused output max_diff={max_diff:.6} > 1e-3"
        );
        assert!(
            max_state_diff <= 1e-3,
            "recurrence_fused state max_diff={max_state_diff:.6} > 1e-3"
        );
    }

    /// Plan 603 R2 — the half-state kernel vs the CPU reference computed
    /// over the SAME quantized initial state: the output must match the f32
    /// CPU reference class (same tolerance — the compute is f32 identical;
    /// only the initial state was rounded), and the STORED state must be
    /// exactly the RN-quantized CPU state (the `half` crate as the rounding
    /// reference — production never depends on it, tests do).
    #[test]
    fn test_recurrence_half_fused_matches_cpu_on_quantized_state() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        let n_head = 48usize;
        let head_dim = 128usize;
        let qkv_len = 3 * n_head * head_dim;
        let mut seed: u64 = 0xa409_3822_299f_31d0;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let qkv: Vec<f32> = (0..qkv_len).map(|_| next()).collect();
        let beta: Vec<f32> = (0..n_head).map(|_| 0.2 + 0.6 * next().abs()).collect();
        let decay: Vec<f32> = (0..n_head).map(|_| 0.7 + 0.29 * next().abs()).collect();
        let state_len = n_head * head_dim * head_dim;
        let state_init: Vec<f32> = (0..state_len).map(|_| next() * 0.1).collect();

        for fmt in [HalfStateFmt::F16, HalfStateFmt::Bf16] {
            // Quantize the initial state exactly as the device lane holds it.
            let quantize = |x: &f32| -> f32 {
                match fmt {
                    HalfStateFmt::F16 => {
                        half::f16::from_f32(x.clamp(-65504.0, 65504.0)).to_f32()
                    }
                    HalfStateFmt::Bf16 => half::bf16::from_f32(*x).to_f32(),
                }
            };
            let state_q: Vec<f32> = state_init.iter().map(quantize).collect();
            let mut state_cpu = state_q.clone();
            let cpu_output = cpu_recurrence(&qkv, &beta, &decay, &mut state_cpu, n_head, head_dim);

            let qkv_dev = stream.clone_htod(&qkv).unwrap();
            let beta_dev = stream.clone_htod(&beta).unwrap();
            let decay_dev = stream.clone_htod(&decay).unwrap();
            let state_bits: Vec<u16> = state_q
                .iter()
                .map(|&x| match fmt {
                    HalfStateFmt::F16 => half::f16::from_f32(x).to_bits(),
                    HalfStateFmt::Bf16 => half::bf16::from_f32(x).to_bits(),
                })
                .collect();

            // The BIT-EXACT reference is the f32 fused kernel run on the SAME
            // quantized initial state (identical f32 op order to the half
            // kernel — only the STORE rounds); a serial CPU reference differs
            // at last-ULP (reduction-order class), which flips RN boundaries
            // for a ~0.03% slice of entries and is NOT the contract here.
            let state_f32_dev = stream.clone_htod(&state_q).unwrap();
            let out_f32_dev = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();
            kernels
                .launch_recurrence_fused(
                    &stream, &qkv_dev, &beta_dev, &decay_dev, &state_f32_dev, &out_f32_dev,
                    head_dim, n_head,
                )
                .expect("launch f32 fused reference");
            let mut state_f32_ref = vec![0f32; state_len];
            stream
                .memcpy_dtoh(&state_f32_dev, &mut state_f32_ref)
                .unwrap();

            let state_dev = stream.clone_htod(&state_bits).unwrap();
            let output_dev = stream.alloc_zeros::<f32>(n_head * head_dim).unwrap();

            kernels
                .launch_recurrence_fused_half_hd128(
                    &stream, &qkv_dev, &beta_dev, &decay_dev, &state_dev, &output_dev,
                    n_head, fmt,
                )
                .expect("launch half fused");
            stream.synchronize().expect("sync");

            let mut gpu_output = vec![0f32; n_head * head_dim];
            stream.memcpy_dtoh(&output_dev, &mut gpu_output).unwrap();
            let mut gpu_state_bits = vec![0u16; state_len];
            stream.memcpy_dtoh(&state_dev, &mut gpu_state_bits).unwrap();

            // The half kernel's OUTPUT must equal the f32 kernel's output
            // bit-for-bit (computed from the pre-rounding f32 s_new).
            let mut out_bit_mismatches = 0usize;
            let mut out_f32_ref = vec![0f32; n_head * head_dim];
            stream.memcpy_dtoh(&out_f32_dev, &mut out_f32_ref).unwrap();
            for i in 0..n_head * head_dim {
                if gpu_output[i].to_bits() != out_f32_ref[i].to_bits() {
                    out_bit_mismatches += 1;
                }
            }
            // The stored state must be the RN-quantized F32-KERNEL state,
            // exactly (the only numerics delta is the store rounding).
            let mut bit_mismatches = 0usize;
            let mut cpu_max_diff = 0f32;
            for i in 0..state_len {
                let want = match fmt {
                    HalfStateFmt::F16 => half::f16::from_f32(state_f32_ref[i].clamp(-65504.0, 65504.0)).to_bits(),
                    HalfStateFmt::Bf16 => half::bf16::from_f32(state_f32_ref[i]).to_bits(),
                };
                if gpu_state_bits[i] != want {
                    bit_mismatches += 1;
                }
                cpu_max_diff = cpu_max_diff.max((state_f32_ref[i] - state_cpu[i]).abs());
            }
            let mut cpu_out_diff = 0f32;
            for i in 0..n_head * head_dim {
                cpu_out_diff = cpu_out_diff.max((gpu_output[i] - cpu_output[i]).abs());
            }
            eprintln!(
                "[recurrence_half_{fmt:?}] out-vs-f32kernel bit mismatches={out_bit_mismatches} state bit mismatches={bit_mismatches}/{state_len} (vs serial CPU: out {cpu_out_diff:.2e} state {cpu_max_diff:.2e})"
            );
            assert_eq!(
                out_bit_mismatches, 0,
                "half kernel output differs from the f32 kernel output ({fmt:?})"
            );
            assert_eq!(
                bit_mismatches, 0,
                "stored state is not the RN-quantized f32-kernel state ({fmt:?})"
            );
            // And the f32-kernel-vs-CPU sanity stays in the serial-reference
            // tolerance class (the reduction-order class).
            assert!(
                cpu_out_diff <= 1e-3,
                "half recurrence output vs CPU max_diff={cpu_out_diff:.6} > 1e-3 ({fmt:?})"
            );
        }
    }

    #[test]
    fn test_recurrence_fused_vs_parallel_timing() {
        // A/B wall-clock at the qwen38 GDN production geometry. Production
        // realism: the state is COLD (DRAM) each call — between tokens the
        // weight GEMVs churn L2 — modeled by rotating 24 state buffers
        // (24 x 3.1 MB = 75 MB > 72 MB L2, so each call's state is evicted
        // before it is reused).
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        let n_head = 48usize;
        let head_dim = 128usize;
        let state_len = n_head * head_dim * head_dim;
        let qkv_len = 3 * n_head * head_dim;

        let mut seed: u64 = 0x243f_6a88_85a3_08d3;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let qkv: Vec<f32> = (0..qkv_len).map(|_| next()).collect();
        let beta: Vec<f32> = (0..n_head).map(|_| 0.5).collect();
        let decay: Vec<f32> = (0..n_head).map(|_| 0.95).collect();
        let state_init: Vec<f32> = (0..state_len).map(|_| next() * 0.1).collect();

        let qkv_dev = stream.clone_htod(&qkv).unwrap();
        let beta_dev = stream.clone_htod(&beta).unwrap();
        let decay_dev = stream.clone_htod(&decay).unwrap();
        let n_buf = 24usize;
        let states: Vec<_> = (0..n_buf)
            .map(|_| stream.clone_htod(&state_init).unwrap())
            .collect();
        let outputs: Vec<_> = (0..n_buf)
            .map(|_| stream.alloc_zeros::<f32>(n_head * head_dim).unwrap())
            .collect();

        let run_batch = |kernels: &DeltanetKernels, fused: bool, iters: usize| -> f64 {
            for i in 0..iters {
                let b = i % n_buf;
                if fused {
                    kernels
                        .launch_recurrence_fused(
                            &stream, &qkv_dev, &beta_dev, &decay_dev, &states[b], &outputs[b],
                            head_dim, n_head,
                        )
                        .unwrap();
                } else {
                    kernels
                        .launch_recurrence_parallel(
                            &stream, &qkv_dev, &beta_dev, &decay_dev, &states[b], &outputs[b],
                            head_dim, n_head,
                        )
                        .unwrap();
                }
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for i in 0..iters {
                let b = i % n_buf;
                if fused {
                    kernels
                        .launch_recurrence_fused(
                            &stream, &qkv_dev, &beta_dev, &decay_dev, &states[b], &outputs[b],
                            head_dim, n_head,
                        )
                        .unwrap();
                } else {
                    kernels
                        .launch_recurrence_parallel(
                            &stream, &qkv_dev, &beta_dev, &decay_dev, &states[b], &outputs[b],
                            head_dim, n_head,
                        )
                        .unwrap();
                }
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64 * 1e3 // ms/call
        };

        // Warmup (compile/load paths) then interleaved batches.
        let _ = run_batch(&kernels, false, 48);
        let _ = run_batch(&kernels, true, 48);
        let mut par_best = f64::MAX;
        let mut fus_best = f64::MAX;
        for _ in 0..5 {
            par_best = par_best.min(run_batch(&kernels, false, 240));
            fus_best = fus_best.min(run_batch(&kernels, true, 240));
        }
        let state_mb = state_len as f64 * 4.0 / 1e6;
        eprintln!(
            "[recurrence timing] n_head={n_head} hd={head_dim} state={state_mb:.2} MB \
             (x{n_buf} rotating): parallel={par_best:.4} ms/call ({:.0} GB/s effective), \
             fused={fus_best:.4} ms/call ({:.0} GB/s), speedup={:.2}x",
            2.0 * state_mb / (par_best / 1e3) / 1e3,
            2.0 * state_mb / (fus_best / 1e3) / 1e3,
            par_best / fus_best
        );
        // No hard perf gate here (GPU-state-dependent); the e2e rows in
        // qwen38_dense_forward_cudarc carry the production gate.
    }

    #[test]
    fn test_z_gating_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        let n = 512usize; // n_v_heads * head_dim (subset)
        let output: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 5.0).collect();
        let z: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003).collect();

        // CPU reference
        let cpu_out: Vec<f32> = (0..n)
            .map(|i| {
                let z_val = z[i];
                let silu = z_val / (1.0 + (-z_val).exp());
                output[i] * silu
            })
            .collect();

        let output_dev = stream.clone_htod(&output).unwrap();
        let z_dev = stream.clone_htod(&z).unwrap();

        kernels
            .launch_z_gating(&stream, &output_dev, &z_dev, n)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; n];
        stream.memcpy_dtoh(&output_dev, &mut gpu_out).unwrap();

        let mut max_rel = 0f32;
        for i in 0..n {
            let denom = cpu_out[i].abs().max(1e-6);
            let rel = (gpu_out[i] - cpu_out[i]).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!("[z_gating] n={n}: max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-5, "z_gating max_rel {max_rel:.4e}");
    }

    // ── Issue 641: BPTT backward CPU reference ──
    // Mirrors `gated_deltanet_bptt` from riir-engine but self-contained
    // (no dep on riir-engine in the test). Processes T timesteps in reverse.
    // Returns (grad_q, grad_k, grad_v, grad_beta, grad_decay) per timestep.
    type DeltanetGrads = (
        Vec<f32>, // grad_q [T * n_head * head_dim]
        Vec<f32>, // grad_k
        Vec<f32>, // grad_v
        Vec<f32>, // grad_beta [T * n_head]
        Vec<f32>, // grad_decay
    );
    // Returns (grad_q, grad_k, grad_v, grad_beta, grad_decay) per timestep.
    fn cpu_bptt(
        qkv_seq: &[f32],   // [T * n_head * 3 * head_dim]
        beta_seq: &[f32],  // [T * n_head]
        decay_seq: &[f32], // [T * n_head]
        grad_out: &[f32],  // [T * n_head * head_dim]
        t_len: usize,
        n_head: usize,
        head_dim: usize,
    ) -> DeltanetGrads {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let sph = head_dim * head_dim; // state per head
        let qkv_ph = 3 * head_dim; // qkv per head

        // Forward recompute: build state trajectory
        let mut state_decayed = vec![0.0f32; t_len * n_head * sph];
        let mut state = vec![0.0f32; t_len * n_head * sph];
        let mut kv_mem = vec![0.0f32; t_len * n_head * head_dim];
        let mut delta = vec![0.0f32; t_len * n_head * head_dim];

        for t in 0..t_len {
            for h in 0..n_head {
                let _q_off = h * head_dim;
                let k_off = n_head * head_dim + h * head_dim;
                let v_off = 2 * n_head * head_dim + h * head_dim;
                let qkv_t = &qkv_seq[t * n_head * qkv_ph..];
                let beta_val = beta_seq[t * n_head + h];
                let decay_val = decay_seq[t * n_head + h];
                let sd_off = t * n_head * sph + h * sph;
                let ss_off = sd_off;
                let km_off = t * n_head * head_dim + h * head_dim;
                let dl_off = km_off;

                // Decay
                if t == 0 {
                    for i in 0..sph { state_decayed[sd_off + i] = 0.0; }
                } else {
                    let prev_off = (t - 1) * n_head * sph + h * sph;
                    for i in 0..sph { state_decayed[sd_off + i] = decay_val * state[prev_off + i]; }
                }

                // Retrieve + delta + update
                for row in 0..head_dim {
                    let mut kv = 0.0f32;
                    for c in 0..head_dim {
                        kv += state_decayed[sd_off + row * head_dim + c] * qkv_t[k_off + c];
                    }
                    let v_row = qkv_t[v_off + row];
                    let d = beta_val * (v_row - kv);
                    kv_mem[km_off + row] = kv;
                    delta[dl_off + row] = d;
                    for c in 0..head_dim {
                        state[ss_off + row * head_dim + c] = state_decayed[sd_off + row * head_dim + c] + qkv_t[k_off + c] * d;
                    }
                }
            }
        }

        // Backward BPTT
        let mut grad_q = vec![0.0f32; t_len * n_head * head_dim];
        let mut grad_k = vec![0.0f32; t_len * n_head * head_dim];
        let mut grad_v = vec![0.0f32; t_len * n_head * head_dim];
        let mut grad_beta = vec![0.0f32; t_len * n_head];
        let mut grad_decay = vec![0.0f32; t_len * n_head];
        let mut grad_s = vec![0.0f32; n_head * sph];

        for t in (0..t_len).rev() {
            for h in 0..n_head {
                let q_off = h * head_dim;
                let k_off = n_head * head_dim + h * head_dim;
                let v_off = 2 * n_head * head_dim + h * head_dim;
                let qkv_t = &qkv_seq[t * n_head * qkv_ph..];
                let beta_val = beta_seq[t * n_head + h];
                let decay_val = decay_seq[t * n_head + h];
                let sd_off = t * n_head * sph + h * sph;
                let ss_off = sd_off;
                let km_off = t * n_head * head_dim + h * head_dim;
                let dl_off = km_off;
                let dy = &grad_out[t * n_head * head_dim + h * head_dim..];
                // Issue 981: the per-head slices are bounded to EXACTLY the
                // head's span. The pre-fix `gs` slice ran to the end of
                // grad_s, and a 385c9f34d clippy-heal rewrite of the decay
                // loop (`for i in 0..sph { gs[i] *= decay_val; }` →
                // `gs.iter_mut()`) silently multiplied EVERY LATER HEAD's slab
                // by this head's decay — h=0 stayed exact, h≥1's gs carried a
                // spurious Π decay(t',h'<h) factor per backward step, and the
                // GPU-vs-CPU cross-validation failed deterministically (the
                // kernel was never wrong). Bounding the slices makes the
                // iter_mut form semantically identical to the index form.
                let gs = &mut grad_s[h * sph..][..sph];
                let gq = &mut grad_q[t * n_head * head_dim + h * head_dim..][..head_dim];
                let gk = &mut grad_k[t * n_head * head_dim + h * head_dim..][..head_dim];
                let gv = &mut grad_v[t * n_head * head_dim + h * head_dim..][..head_dim];

                // Step 5 backward: grad_S += outer(dy, q) * scale; grad_q = S^T @ dy * scale
                for row in 0..head_dim {
                    let dy_r = dy[row] * scale;
                    for c in 0..head_dim {
                        gs[row * head_dim + c] += dy_r * qkv_t[q_off + c];
                    }
                }
                for c in 0..head_dim {
                    let mut s = 0.0f32;
                    for row in 0..head_dim {
                        s += state[ss_off + row * head_dim + c] * dy[row] * scale;
                    }
                    gq[c] = s;
                }

                // Steps 4-2 backward
                let mut grad_delta_row = vec![0.0f32; head_dim];
                for row in 0..head_dim {
                    let mut gd = 0.0f32;
                    for c in 0..head_dim {
                        gd += gs[row * head_dim + c] * qkv_t[k_off + c];
                    }
                    grad_delta_row[row] = gd;
                }
                // grad_k_s4[c] = sum_row delta[row] * gs[row, c]
                let mut grad_k_s4 = vec![0.0f32; head_dim];
                for c in 0..head_dim {
                    let mut s = 0.0f32;
                    for row in 0..head_dim {
                        s += delta[dl_off + row] * gs[row * head_dim + c];
                    }
                    grad_k_s4[c] = s;
                }

                // grad_beta, grad_v, grad_kv
                let mut grad_kv = vec![0.0f32; head_dim];
                for row in 0..head_dim {
                    let gd = grad_delta_row[row];
                    grad_beta[t * n_head + h] += gd * (qkv_t[v_off + row] - kv_mem[km_off + row]);
                    gv[row] = beta_val * gd;
                    grad_kv[row] = -beta_val * gd;
                }

                // Step 2 backward: grad_S += outer(grad_kv, k)
                for row in 0..head_dim {
                    for c in 0..head_dim {
                        gs[row * head_dim + c] += grad_kv[row] * qkv_t[k_off + c];
                    }
                }

                // grad_k_s2[c] = sum_row state_decayed[row,c] * grad_kv[row]
                let mut grad_k_s2 = vec![0.0f32; head_dim];
                for c in 0..head_dim {
                    let mut s = 0.0f32;
                    for row in 0..head_dim {
                        s += state_decayed[sd_off + row * head_dim + c] * grad_kv[row];
                    }
                    grad_k_s2[c] = s;
                }

                for c in 0..head_dim {
                    gk[c] = grad_k_s4[c] + grad_k_s2[c];
                }

                // Step 1 backward
                if t > 0 {
                    let prev_off = (t - 1) * n_head * sph + h * sph;
                    let mut gg = 0.0f32;
                    for i in 0..sph {
                        gg += gs[i] * state[prev_off + i];
                    }
                    grad_decay[t * n_head + h] = gg;
                    for g in gs.iter_mut() {
                        *g *= decay_val;
                    }
                } else {
                    grad_decay[t * n_head + h] = 0.0;
                }
            }
        }

        (grad_q, grad_k, grad_v, grad_beta, grad_decay)
    }

    /// Issue 641: cross-validate the GPU BPTT kernel against the CPU reference.
    #[test]
    fn test_bptt_recompute_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        // ⚠ Do NOT run while another GPU process is active (Issue 649 contention).
        // Check nvidia-smi first if running manually.
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        let t_len = 4usize;
        let n_head = 4usize;
        let head_dim = 128usize;

        let qkv_ph = 3 * head_dim;
        let qkv_len = t_len * n_head * qkv_ph;
        let mut qkv: Vec<f32> = (0..qkv_len)
            .map(|i| {
                let s = (i as f32) * 0.001 - 0.3;
                if i.is_multiple_of(7) { -s } else { s }
            })
            .collect();
        // Issue 981: L2-normalize q and k per (t, head) — exactly what the
        // production caller's expand_and_l2_normalize feeds this kernel.
        // The raw fixture drove beta*||k||^2 to 5..30 (eigenvalues of the
        // (I - beta k k^T) update in -4..-29), exploding the state to ~1.5e7
        // and the gradients to ~1e11 — magnitudes no production input ever
        // reaches, and which only passed the relative tolerance because the
        // huge denominators masked real divergence. Same hygiene as the
        // probe-fixture fix in 19bf64ed4 ("un-normalized synthetic q/k
        // explode the recurrence state and poison the absolute correctness
        // side"). The root cause of the deterministic failure was the
        // cpu_bptt decay-slice leak fixed above — this makes the fixture
        // representative on top.
        for t in 0..t_len {
            for h in 0..n_head {
                for section in 0..2 {
                    let base = t * n_head * qkv_ph + section * n_head * head_dim + h * head_dim;
                    let mut sq = 0f32;
                    for c in 0..head_dim {
                        sq += qkv[base + c] * qkv[base + c];
                    }
                    if sq > 0.0 {
                        let inv = 1.0f32 / sq.sqrt();
                        for c in 0..head_dim {
                            qkv[base + c] *= inv;
                        }
                    }
                }
            }
        }
        let beta: Vec<f32> = (0..t_len * n_head).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let decay: Vec<f32> = (0..t_len * n_head).map(|i| 0.9 - (i as f32) * 0.01).collect();
        let grad_out: Vec<f32> = (0..t_len * n_head * head_dim)
            .map(|i| (i as f32) * 0.002 - 0.5)
            .collect();

        // CPU reference
        let (cpu_gq, _cpu_gk, cpu_gv, cpu_gb, _cpu_gd) =
            cpu_bptt(&qkv, &beta, &decay, &grad_out, t_len, n_head, head_dim);

        // GPU
        let qkv_dev = stream.clone_htod(&qkv).unwrap();
        let beta_dev = stream.clone_htod(&beta).unwrap();
        let decay_dev = stream.clone_htod(&decay).unwrap();
        let grad_out_dev = stream.clone_htod(&grad_out).unwrap();

        let sph = head_dim * head_dim;
        let mut grad_qkv_dev = stream.alloc_zeros::<f32>(qkv_len).unwrap();
        let mut grad_beta_dev = stream.alloc_zeros::<f32>(t_len * n_head).unwrap();
        let mut grad_decay_dev = stream.alloc_zeros::<f32>(t_len * n_head).unwrap();
        let mut sd_dev = stream.alloc_zeros::<f32>(t_len * n_head * sph).unwrap();
        let mut ss_dev = stream.alloc_zeros::<f32>(t_len * n_head * sph).unwrap();
        let mut km_dev = stream.alloc_zeros::<f32>(t_len * n_head * head_dim).unwrap();
        let mut dl_dev = stream.alloc_zeros::<f32>(t_len * n_head * head_dim).unwrap();
        let mut gs_dev = stream.alloc_zeros::<f32>(n_head * sph).unwrap();

        kernels
            .launch_bptt_recompute(
                &stream,
                &qkv_dev,
                &beta_dev,
                &decay_dev,
                &grad_out_dev,
                &mut grad_qkv_dev,
                &mut grad_beta_dev,
                &mut grad_decay_dev,
                &mut sd_dev,
                &mut ss_dev,
                &mut km_dev,
                &mut dl_dev,
                &mut gs_dev,
                t_len,
                head_dim,
                n_head,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_gqkv = vec![0f32; qkv_len];
        stream.memcpy_dtoh(&grad_qkv_dev, &mut gpu_gqkv).unwrap();
        let mut gpu_gb = vec![0f32; t_len * n_head];
        stream.memcpy_dtoh(&grad_beta_dev, &mut gpu_gb).unwrap();
        let mut gpu_gd = vec![0f32; t_len * n_head];
        stream.memcpy_dtoh(&grad_decay_dev, &mut gpu_gd).unwrap();

        // Compare grad_q (first third of qkv)
        let mut max_gq = 0f32;
        for h in 0..n_head {
            for t in 0..t_len {
                let q_off = t * n_head * qkv_ph + h * head_dim;
                let cpu_off = t * n_head * head_dim + h * head_dim;
                for c in 0..head_dim {
                    let diff = (gpu_gqkv[q_off + c] - cpu_gq[cpu_off + c]).abs();
                    max_gq = max_gq.max(diff);
                }
            }
        }
        eprintln!("[bptt] grad_q max_diff={max_gq:.6e}");

        // Compare grad_v (third third of qkv)
        let mut max_gv = 0f32;
        for h in 0..n_head {
            for t in 0..t_len {
                let v_off = t * n_head * qkv_ph + 2 * n_head * head_dim + h * head_dim;
                let cpu_off = t * n_head * head_dim + h * head_dim;
                for c in 0..head_dim {
                    let diff = (gpu_gqkv[v_off + c] - cpu_gv[cpu_off + c]).abs();
                    max_gv = max_gv.max(diff);
                }
            }
        }
        eprintln!("[bptt] grad_v max_diff={max_gv:.6e}");

        // Compare grad_beta
        let mut max_gb = 0f32;
        for i in 0..t_len * n_head {
            max_gb = max_gb.max((gpu_gb[i] - cpu_gb[i]).abs());
        }
        eprintln!("[bptt] grad_beta max_diff={max_gb:.6e}");

        assert!(max_gq < 1e-3 * cpu_gq.iter().cloned().map(|x| x.abs()).fold(0f32, f32::max).max(1.0),
            "grad_q max_diff {max_gq:.6e} (rel error too large)");
        assert!(max_gv < 1e-3 * cpu_gv.iter().cloned().map(|x| x.abs()).fold(0f32, f32::max).max(1.0),
            "grad_v max_diff {max_gv:.6e} (rel error too large)");
        assert!(max_gb < 1e-3 * cpu_gb.iter().cloned().map(|x| x.abs()).fold(0f32, f32::max).max(1.0),
            "grad_beta max_diff {max_gb:.6e} (rel error too large)");
    }

    /// Issue 470 T2-B: production-shape recurrence-backward ceiling probe for
    /// the Bonsai RLVR trainer wiring decision. Times the CPU reference
    /// (`cpu_bptt`) against `launch_bptt_recompute` at the Bonsai deltanet
    /// dims (n_head=16, head_dim=128) for T in {64, 256}, and re-checks the
    /// GPU/CPU gradient agreement at production scale (the T=4 unit test does
    /// not exercise the T-loop tail behavior at realistic lengths).
    ///
    /// `#[ignore]`d: CUDA-only (skips on macOS builds by construction),
    /// release-only timing, and it consumes real GPU seconds — run manually
    /// on the 4090 with
    /// `cargo test --release -p riir-gpu --features ternary_gemv_cuda_raw
    ///  --lib probe_470 -- --ignored --nocapture`.
    /// ⚠ Issue 649: do not run while another GPU compute process is active.
    #[test]
    #[ignore = "4090 timing probe (Issue 470 T2-B): release + CUDA only, consumes GPU seconds"]
    fn probe_470_bptt_production_shapes_timing() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = DeltanetKernels::new(ctx).expect("compile");

        // Bonsai-27B deltanet-linear dims (GGUF metadata: 16 heads x 128).
        let n_head = 16usize;
        let head_dim = 128usize;
        let iters = 3usize;

        for t_len in [64usize, 256usize] {
            let qkv_ph = 3 * head_dim;
            let qkv_len = t_len * n_head * qkv_ph;
            let mut qkv: Vec<f32> = (0..qkv_len)
                .map(|i| {
                    let s = (i as f32) * 0.001 - 0.3;
                    if i.is_multiple_of(7) { -s } else { s }
                })
                .collect();
            // Production feeds this kernel the `expand_and_l2_normalize_heads`
            // output — q/k are L2-normalized per (t, head). Synthetic q/k that
            // skip the normalization explode the recurrence state over long T
            // (1e30-class values at T>=64), which poisons the absolute side of
            // the correctness check. Normalize to match the real caller.
            for h in 0..n_head {
                for t in 0..t_len {
                    for which in 0..2usize {
                        let off = t * n_head * qkv_ph + h * qkv_ph + which * head_dim;
                        let norm: f32 = qkv[off..off + head_dim]
                            .iter()
                            .map(|x| x * x)
                            .sum::<f32>()
                            .sqrt();
                        if norm > 0.0 {
                            for x in &mut qkv[off..off + head_dim] {
                                *x /= norm;
                            }
                        }
                    }
                }
            }
            let beta: Vec<f32> = (0..t_len * n_head)
                .map(|i| 0.5 + ((i % n_head) as f32) * 0.01)
                .collect();
            // Decay must stay in (0, 1) like the real beta/decay projection —
            // a naive 0.9 − 0.01·i over the flattened (t·n_head) index drives
            // |decay| past 1 within ~190 entries and compounds the state to
            // 1e30-class garbage at production T. Index per-head instead:
            // per-head decay values in [0.74, 0.9].
            let decay: Vec<f32> = (0..t_len * n_head)
                .map(|i| 0.9 - ((i % n_head) as f32) * 0.01)
                .collect();
            let grad_out: Vec<f32> = (0..t_len * n_head * head_dim)
                .map(|i| (i as f32) * 0.002 - 0.5)
                .collect();

            // ── CPU arm ──
            // Warm + correctness reference.
            let cpu_start = std::time::Instant::now();
            let (cpu_gq, _cpu_gk, cpu_gv, cpu_gb, _cpu_gd) =
                cpu_bptt(&qkv, &beta, &decay, &grad_out, t_len, n_head, head_dim);
            let cpu_first = cpu_start.elapsed();
            let mut cpu_total = std::time::Duration::ZERO;
            for _ in 0..iters.saturating_sub(1) {
                let s = std::time::Instant::now();
                let _ = cpu_bptt(&qkv, &beta, &decay, &grad_out, t_len, n_head, head_dim);
                cpu_total += s.elapsed();
            }
            let cpu_per_iter = if iters > 1 {
                cpu_total / (iters - 1) as u32
            } else {
                cpu_first
            };

            // ── GPU arm ──
            let qkv_dev = stream.clone_htod(&qkv).unwrap();
            let beta_dev = stream.clone_htod(&beta).unwrap();
            let decay_dev = stream.clone_htod(&decay).unwrap();
            let grad_out_dev = stream.clone_htod(&grad_out).unwrap();

            let sph = head_dim * head_dim;
            let mut grad_qkv_dev = stream.alloc_zeros::<f32>(qkv_len).unwrap();
            let mut grad_beta_dev = stream.alloc_zeros::<f32>(t_len * n_head).unwrap();
            let mut grad_decay_dev = stream.alloc_zeros::<f32>(t_len * n_head).unwrap();
            let mut sd_dev = stream.alloc_zeros::<f32>(t_len * n_head * sph).unwrap();
            let mut ss_dev = stream.alloc_zeros::<f32>(t_len * n_head * sph).unwrap();
            let mut km_dev = stream.alloc_zeros::<f32>(t_len * n_head * head_dim).unwrap();
            let mut dl_dev = stream.alloc_zeros::<f32>(t_len * n_head * head_dim).unwrap();
            let mut gs_dev = stream.alloc_zeros::<f32>(n_head * sph).unwrap();

            let launch = |grad_qkv_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          grad_beta_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          grad_decay_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          sd_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          ss_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          km_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          dl_dev: &mut cudarc::driver::safe::CudaSlice<f32>,
                          gs_dev: &mut cudarc::driver::safe::CudaSlice<f32>| {
                // Outputs must be zeroed before each launch (kernel accumulates).
                stream.memset_zeros(grad_qkv_dev).unwrap();
                stream.memset_zeros(grad_beta_dev).unwrap();
                stream.memset_zeros(grad_decay_dev).unwrap();
                stream.memset_zeros(gs_dev).unwrap();
                kernels
                    .launch_bptt_recompute(
                        &stream,
                        &qkv_dev,
                        &beta_dev,
                        &decay_dev,
                        &grad_out_dev,
                        grad_qkv_dev,
                        grad_beta_dev,
                        grad_decay_dev,
                        sd_dev,
                        ss_dev,
                        km_dev,
                        dl_dev,
                        gs_dev,
                        t_len,
                        head_dim,
                        n_head,
                    )
                    .expect("launch");
            };

            // Warm launch (also the correctness arm at production shape).
            launch(&mut grad_qkv_dev, &mut grad_beta_dev,
                   &mut grad_decay_dev, &mut sd_dev, &mut ss_dev,
                   &mut km_dev, &mut dl_dev, &mut gs_dev);
            stream.synchronize().expect("sync");

            let gpu_start = std::time::Instant::now();
            for _ in 0..iters {
                launch(&mut grad_qkv_dev, &mut grad_beta_dev,
                       &mut grad_decay_dev, &mut sd_dev, &mut ss_dev,
                       &mut km_dev, &mut dl_dev, &mut gs_dev);
                stream.synchronize().expect("sync");
            }
            let gpu_per_iter = gpu_start.elapsed() / iters as u32;

            // Correctness at production shape — offset-aware (GPU packs
            // [q|k|v] per (t,h) at qkv_ph stride; CPU returns separate
            // contiguous arrays) + the same relative tolerance family as the
            // T=4 unit test above.
            let mut gpu_gqkv = vec![0f32; qkv_len];
            stream.memcpy_dtoh(&grad_qkv_dev, &mut gpu_gqkv).unwrap();
            let mut gpu_gb = vec![0f32; t_len * n_head];
            stream.memcpy_dtoh(&grad_beta_dev, &mut gpu_gb).unwrap();
            let mut max_gq = 0f32;
            let mut max_gv = 0f32;
            for h in 0..n_head {
                for t in 0..t_len {
                    let cpu_off = t * n_head * head_dim + h * head_dim;
                    let q_off = t * n_head * qkv_ph + h * head_dim;
                    let v_off = t * n_head * qkv_ph + 2 * n_head * head_dim + h * head_dim;
                    for c in 0..head_dim {
                        max_gq = max_gq.max((gpu_gqkv[q_off + c] - cpu_gq[cpu_off + c]).abs());
                        max_gv = max_gv.max((gpu_gqkv[v_off + c] - cpu_gv[cpu_off + c]).abs());
                    }
                }
            }
            let max_gb = cpu_gb
                .iter()
                .zip(gpu_gb.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let rel = |max_diff: f32, cpu: &[f32]| {
                let scale = cpu.iter().cloned().map(|x| x.abs()).fold(0f32, f32::max).max(1.0);
                max_diff < 1e-3 * scale
            };

            eprintln!(
                "[470-T2B] T={t_len} n_head={n_head} hd={head_dim}: \
                 cpu {cpu_per_iter:?}/iter (first {cpu_first:?}) | \
                 gpu {gpu_per_iter:?}/iter (sync incl.) | speedup {:.2}x | \
                 max_diff gq={max_gq:.3e} gv={max_gv:.3e} gb={max_gb:.3e}",
                cpu_per_iter.as_secs_f64() / gpu_per_iter.as_secs_f64()
            );

            assert!(rel(max_gq, &cpu_gq), "T={t_len} grad_q max_diff {max_gq:.3e} (rel too large)");
            assert!(rel(max_gv, &cpu_gv), "T={t_len} grad_v max_diff {max_gv:.3e} (rel too large)");
            assert!(rel(max_gb, &cpu_gb), "T={t_len} grad_beta max_diff {max_gb:.3e} (rel too large)");
        }
    }
}
