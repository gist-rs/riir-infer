//! Issue 980 T4 — cudarc activation-side Hadamard rotation kernels for the
//! Bonsai-2 folded-ternary runtime (Plan 600 Phase B).
//!
//! The CPU contract lives in `riir-infer-core::deltanet::rotation`
//! (`rotate_forward_inplace` / `rotate_inverse_inplace` /
//! `permute_gdn_v_grouped_inplace`); every kernel here is a numeric twin of
//! those functions, dispatched on the shared cudarc stream so the folded
//! matmuls consume rotated activations GPU-resident (the CPU lane's
//! memoization posture: the rotation runs between the RMSNorm and the int8
//! activation quantize — it commutes with neither).
//!
//! Kernel set (all compiled once at construction from [`ROTATION_CUDA_SRC`],
//! only when the loaded model declares `prism.hadamard`):
//!
//! | kernel | CPU twin | used at |
//! |---|---|---|
//! | `fwht_rotate_forward_f32` | `rotate_forward_inplace` | every folded matmul input (sign→FWHT) |
//! | `fwht_rotate_inverse_f32` | `rotate_inverse_inplace` | token-embedding lookup (FWHT→sign) |
//! | `gdn_v_permute_f32` | `permute_gdn_v_grouped_inplace` | `ssm_out` input (tiled→grouped heads) |
//! | `gate_silu_f32` | `silu(z) *` gating | `ssm_out` input (split from the fused Issue-627 kernel) |
//! | `gate_sigmoid_f32` | `sigmoid(gate) *` gating | attention `wo` input (split from Issue-626) |
//! | `gemv_dense_f32` | `GateProjWeights::matvec_into` dense arm | the Bonsai-2 dense `ssm_alpha/beta` escape set |
//!
//! Numeric parity: the FWHT applies the per-butterfly `1/√2` (the CPU/fork
//! unitary map), not a single end-stage `1/√n` — matching the CPU lane's
//! rounding shape. Like every cudarc-path numeric, results live inside the
//! lane's established int8-activation-quantization envelope (Bench 706:
//! logits max_rel 2.7e-3, argmax-identical).
//!
//! Dense a/b: the escape set is deliberately full-precision (whitepaper A.2)
//! and consumes the PRIMAL normed input, so it bypasses the int8/dp4a GEMV
//! path entirely — `gemv_dense_f32` is a plain fp32 row GEMV (48×5120 per
//! DeltaNet layer; ~0.25 MFLOP, launch-bound).
//!
//! NOT wired here (refused at construction until landed): the CUDA-graph
//! (`_devpos`) decode lane — the graph path captures device-pointer launches
//! and needs the devpos launcher twins; the eager path is the T4 validation
//! surface (G1 + G2). LoRA-on-folded and the training forwards also refuse
//! (adapter math and the minimal-activation-cache contract are
//! primal-basis-shaped).

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;
use riir_infer_core::deltanet::rotation::TernaryRotationConfig;

use crate::cudarc_kernels::CudarcKernelError;

pub(crate) const ROTATION_CUDA_SRC: &str = r#"
// One CUDA block per block_size segment; 256 threads butterfly-pair through
// shared memory. Sign FIRST (absolute index within the width vector), then
// the unitary FWHT (per-butterfly 1/sqrt2) — the CPU twin's exact order.
extern "C" __global__ void fwht_rotate_forward_f32(
    float* x, const float* signs, int n, int block_size)
{
    __shared__ float smem[1024];
    int base = blockIdx.x * block_size;
    if (base >= n) return;
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        float v = x[base + i];
        smem[i] = v * signs[base + i];
    }
    __syncthreads();
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 2; step <= block_size; step <<= 1) {
        int half = step >> 1;
        for (int i = threadIdx.x; i < block_size / 2; i += blockDim.x) {
            int blk = (i / half) * step;
            int off = i % half;
            float a = smem[blk + off];
            float b = smem[blk + off + half];
            smem[blk + off] = (a + b) * inv_sqrt2;
            smem[blk + off + half] = (a - b) * inv_sqrt2;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        x[base + i] = smem[i];
    }
}

// Embedding-inverse twin: Hadamard FIRST, sign SECOND (fork build_inp_embd
// `h = s * (H z)`; composing with the forward gives S·H·H·S = I per block).
extern "C" __global__ void fwht_rotate_inverse_f32(
    float* x, const float* signs, int n, int block_size)
{
    __shared__ float smem[1024];
    int base = blockIdx.x * block_size;
    if (base >= n) return;
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        smem[i] = x[base + i];
    }
    __syncthreads();
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 2; step <= block_size; step <<= 1) {
        int half = step >> 1;
        for (int i = threadIdx.x; i < block_size / 2; i += blockDim.x) {
            int blk = (i / half) * step;
            int off = i % half;
            float a = smem[blk + off];
            float b = smem[blk + off + half];
            smem[blk + off] = (a + b) * inv_sqrt2;
            smem[blk + off + half] = (a - b) * inv_sqrt2;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        x[base + i] = smem[i] * signs[base + i];
    }
}

// gdn_v_grouped head permute, tiled [hd, nk, rep] -> grouped [hd, rep, nk].
// Whole-head-block gather through a scratch copy (no in-place race): the
// caller uploads x to tmp first (or dispatches a copy kernel); this kernel
// gathers dst from tmp.
extern "C" __global__ void gdn_v_permute_f32(
    const float* tmp, float* x, int len, int hd, int n_k, int rep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    int head = i / hd;          // grouped head index: nk * rep + r
    int off = i % hd;
    int nk = head / rep;
    int r = head % rep;
    int src_head = r * n_k + nk; // tiled head index
    x[i] = tmp[src_head * hd + off];
}

// Bare gate kernels (the fused Issue-626/627 kernels quantize in the same
// pass; the rotation must run between gate and quantize, so the gate stage
// is split out here — same math, f32 out).
extern "C" __global__ void gate_silu_f32(float* x, const float* gate, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    x[i] *= g / (1.0f + expf(-g));
}

extern "C" __global__ void gate_sigmoid_f32(float* x, const float* gate, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

// Dense fp32 row GEMV for the Bonsai-2 ssm_alpha/beta escape set (48x5120):
// one block per row, 512 threads strided dot + shared reduction. Reads the
// PRIMAL (unrotated) normed input; full precision — never int8.
extern "C" __global__ void gemv_dense_f32(
    float* out, const float* w, const float* x, int cols)
{
    int row = blockIdx.x;
    const float* wr = w + (size_t)row * (size_t)cols;
    extern __shared__ float smem[];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += wr[i] * x[i];
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = smem[0];
}

// ===========================================================================
// Issue 980 T4 escalation — FUSED rotation kernels (Bench 940: the split
// path's ~614 extra launches/token at the ~3 µs in-graph dispatch floor was
// the whole +1.90 ms/token decode cost; these restore the Issue-623/626/627
// fusion shapes with sign+FWHT inserted between the norm/gate stage and the
// quantize stage, at ZERO extra launches vs the pre-rotation baseline).
//
// Bit-identity contract (asserted by the bitexact unit tests): every fused
// kernel reproduces the split path's arithmetic VERBATIM —
//   - rmsnorm reduction: rmsnorm_f32's exact 256-thread strided partials +
//     256-slot tree (each fused block re-runs the FULL-row reduction
//     redundantly; identical instruction sequence over identical inputs
//     gives a bit-identical inv_rms with no cross-block barrier at all);
//   - per-head norm: rmsnorm_batched_f32's 256-slot tree pairing, expressed
//     as the smem fold sequence (s = 128..1 within the head region; the
//     +0.0 folds of the 256-slot original are bit-neutral and skipped);
//   - transform value order: ((x*inv_rms)*gamma)*sign — the exact order the
//     split path computed across two kernels;
//   - butterfly: fwht_rotate_forward_f32 verbatim ((a±b)*inv_sqrt2 per step);
//   - quantize: per-16 absmax (fmaxf is commutative — group shape does not
//     matter), d = absmax*(1/127), roundf(x/d), clamp ±[−128,127].
// ===========================================================================

// Shared body helpers (device inline) — one Hadamard block per CUDA block,
// 256 threads. `base` = the block's offset within the width vector.
__device__ __forceinline__ void rot_reduction_inv_rms_256(
    const float* x, int dim, float inv_dim, float eps,
    float* red /* [256] shared */, float* inv_rms_out)
{
    const int tid = threadIdx.x;
    float partial_sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = x[i];
        partial_sq += v * v;
    }
    red[tid] = partial_sq;
    __syncthreads();
    if (tid < 128) red[tid] += red[tid + 128]; __syncthreads();
    if (tid < 64)  red[tid] += red[tid + 64];  __syncthreads();
    if (tid < 32)  red[tid] += red[tid + 32];  __syncthreads();
    if (tid < 16)  red[tid] += red[tid + 16];  __syncthreads();
    if (tid < 8)   red[tid] += red[tid + 8];   __syncthreads();
    if (tid < 4)   red[tid] += red[tid + 4];   __syncthreads();
    if (tid < 2)   red[tid] += red[tid + 2];   __syncthreads();
    if (tid < 1)   red[0] += red[1];
    __syncthreads();
    if (tid == 0) {
        float mean_sq = red[0] * inv_dim;
        red[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    *inv_rms_out = red[0];
}

// Butterfly one Hadamard block in `buf` (hblock elements, blockDim threads).
__device__ __forceinline__ void rot_butterfly_1024(float* buf, int hblock)
{
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 2; step <= hblock; step <<= 1) {
        int half = step >> 1;
        for (int i = threadIdx.x; i < hblock / 2; i += blockDim.x) {
            int blk = (i / half) * step;
            int off = i % half;
            float a = buf[blk + off];
            float b = buf[blk + off + half];
            buf[blk + off] = (a + b) * inv_sqrt2;
            buf[blk + off + half] = (a - b) * inv_sqrt2;
        }
        __syncthreads();
    }
}

// Quantize one Hadamard block from smem: per-16 absmax via shfl_xor within
// 16-lane groups, d = absmax*(1/127), roundf + clamp. Writes i8 + ascale at
// GLOBAL indices (base + j) / ((base + j)/16).
__device__ __forceinline__ void rot_quantize_16(
    const float* buf, int base, int hblock,
    signed char* out, float* ascale)
{
    const int tid = threadIdx.x;
    for (int g0 = 0; g0 < hblock; g0 += blockDim.x) {
        int j = g0 + tid;
        float v = (j < hblock) ? buf[j] : 0.0f;
        float a = fabsf(v);
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 8));
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 4));
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 2));
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 1));
        // All 16 lanes of a group hold the same absmax; lanes outside hblock
        // contributed +0.0 to their group's max only when the whole group is
        // a tail group — matching the split quantize kernel's zero-fill.
        float d = (a > 0.0f) ? (a * (1.0f / 127.0f)) : 1.0f;
        if (j < hblock && (tid % 16) == 0) {
            ascale[(base + j) / 16] = d;
        }
        float inv_d = 1.0f / d;
        if (j < hblock) {
            float q = roundf(v * inv_d);
            q = fmaxf(-128.0f, fminf(127.0f, q));
            out[base + j] = (signed char)q;
        }
    }
}

// K1 — fused rmsnorm + sign + FWHT + quantize, one Hadamard block per CUDA
// block, 256 threads. Replaces launch_rmsnorm + memcpy_dtod +
// fwht_rotate_forward + launch_quantize (4 launches -> 1).
extern "C" __global__ void rmsnorm_sign_fwht_quantize_f32(
    const float* __restrict__ x,        // [dim] raw (residual stream)
    const float* __restrict__ gamma,    // [dim] norm weights
    const float* __restrict__ signs,    // [dim] +1/-1
    signed char* __restrict__ out,      // [dim] int8
    float* __restrict__ ascale,         // [dim/16]
    float inv_dim, float eps,
    int dim, int hblock)
{
    __shared__ float red[256];
    __shared__ float buf[1024];
    float inv_rms;
    rot_reduction_inv_rms_256(x, dim, inv_dim, eps, red, &inv_rms);
    const int base = blockIdx.x * hblock;
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        float v = x[base + i] * inv_rms * gamma[base + i];
        buf[i] = v * signs[base + i];
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

// K2 — GDN input stage, heterogeneous single launch: blocks [0, dim/hblock)
// run the K1 body (rotated+quantized activation for the qkv/z dp4a GEMVs);
// blocks [dim/hblock, +2*rows) are dense fp32 GEMV rows for the escape-set
// a/b on the PRIMAL normed input — each GEMV block re-runs the full-row
// rmsnorm reduction redundantly (bit-identical inv_rms, no cross-block
// dependency at all) and folds the normalization INLINE into the dot:
// acc += w[i] * ((x[i]*inv_rms)*gamma[i]) — the exact value order the split
// path computed via the stored norm_x. 512 threads (uniform blockDim).
// Replaces launch_rmsnorm + memcpy + fwht + quantize + gemv_a + gemv_b
// (6 launches -> 1).
extern "C" __global__ void gdn_input_rotabq_f32(
    const float* __restrict__ x,        // [dim] raw
    const float* __restrict__ gamma,    // [dim] input_norm
    const float* __restrict__ signs,    // [dim]
    const float* __restrict__ wa,       // [rows x dim] dense ssm_alpha
    const float* __restrict__ wb,       // [rows x dim] dense ssm_beta
    float* __restrict__ out_a,          // [rows]
    float* __restrict__ out_b,          // [rows]
    signed char* __restrict__ qout,     // [dim] int8
    float* __restrict__ ascale,         // [dim/16]
    float inv_dim, float eps,
    int dim, int hblock, int rows)
{
    const int nq = dim / hblock;
    if ((int)blockIdx.x < nq) {
        __shared__ float red[256];
        __shared__ float buf[1024];
        float inv_rms;
        if (threadIdx.x < 256) {
            // rmsnorm_f32's exact reduction, run by the first 256 threads.
            float partial_sq = 0.0f;
            for (int i = threadIdx.x; i < dim; i += 256) {
                float v = x[i];
                partial_sq += v * v;
            }
            red[threadIdx.x] = partial_sq;
        }
        __syncthreads();
        if (threadIdx.x < 128) red[threadIdx.x] += red[threadIdx.x + 128]; __syncthreads();
        if (threadIdx.x < 64)  red[threadIdx.x] += red[threadIdx.x + 64];  __syncthreads();
        if (threadIdx.x < 32)  red[threadIdx.x] += red[threadIdx.x + 32];  __syncthreads();
        if (threadIdx.x < 16)  red[threadIdx.x] += red[threadIdx.x + 16];  __syncthreads();
        if (threadIdx.x < 8)   red[threadIdx.x] += red[threadIdx.x + 8];   __syncthreads();
        if (threadIdx.x < 4)   red[threadIdx.x] += red[threadIdx.x + 4];   __syncthreads();
        if (threadIdx.x < 2)   red[threadIdx.x] += red[threadIdx.x + 2];   __syncthreads();
        if (threadIdx.x < 1)   red[0] += red[1];
        __syncthreads();
        if (threadIdx.x == 0) {
            float mean_sq = red[0] * inv_dim;
            red[0] = 1.0f / sqrtf(mean_sq + eps);
        }
        __syncthreads();
        inv_rms = red[0];
        const int base = blockIdx.x * hblock;
        for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
            float v = x[base + i] * inv_rms * gamma[base + i];
            buf[i] = v * signs[base + i];
        }
        __syncthreads();
        rot_butterfly_1024(buf, hblock);
        rot_quantize_16(buf, base, hblock, qout, ascale);
    } else {
        const int row = blockIdx.x - nq;
        const bool is_a = row < rows;
        const int r = is_a ? row : (row - rows);
        const float* w = is_a ? wa : wb;
        const float* wr = w + (size_t)r * (size_t)dim;
        __shared__ float red[256];
        __shared__ float gred[512];
        float inv_rms;
        if (threadIdx.x < 256) {
            float partial_sq = 0.0f;
            for (int i = threadIdx.x; i < dim; i += 256) {
                float v = x[i];
                partial_sq += v * v;
            }
            red[threadIdx.x] = partial_sq;
        }
        __syncthreads();
        if (threadIdx.x < 128) red[threadIdx.x] += red[threadIdx.x + 128]; __syncthreads();
        if (threadIdx.x < 64)  red[threadIdx.x] += red[threadIdx.x + 64];  __syncthreads();
        if (threadIdx.x < 32)  red[threadIdx.x] += red[threadIdx.x + 32];  __syncthreads();
        if (threadIdx.x < 16)  red[threadIdx.x] += red[threadIdx.x + 16];  __syncthreads();
        if (threadIdx.x < 8)   red[threadIdx.x] += red[threadIdx.x + 8];   __syncthreads();
        if (threadIdx.x < 4)   red[threadIdx.x] += red[threadIdx.x + 4];   __syncthreads();
        if (threadIdx.x < 2)   red[threadIdx.x] += red[threadIdx.x + 2];   __syncthreads();
        if (threadIdx.x < 1)   red[0] += red[1];
        __syncthreads();
        if (threadIdx.x == 0) {
            float mean_sq = red[0] * inv_dim;
            red[0] = 1.0f / sqrtf(mean_sq + eps);
        }
        __syncthreads();
        inv_rms = red[0];
        // gemv_dense_f32's exact strided dot + 512-slot tree, with the stored
        // normed value folded inline ((x*inv_rms)*gamma — same two multiplies
        // in the same order rmsnorm_f32 applied before the store).
        float acc = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            float v = x[i] * inv_rms * gamma[i];
            acc += wr[i] * v;
        }
        gred[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) gred[threadIdx.x] += gred[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (is_a) out_a[r] = gred[0]; else out_b[r] = gred[0];
        }
    }
}

// K3 — GDN output chain fused: per-head RMSNorm + silu(z) gate + tiled->
// grouped head permute + sign + FWHT + quantize. LAUNCHED WITH blockDim ==
// hblock (one thread per element of the block's Hadamard window) so the
// direct tid indexing stays in-window. The head norm replicates
// rmsnorm_batched_f32's 256-slot tree via the smem fold sequence within each
// contiguous head region (the +0.0 folds of the original are bit-neutral);
// the gate is applied at the SOURCE head index (pre-permute, the split
// path's pairing); the sign+butterfly run at the DESTINATION index.
// Replaces launch_rmsnorm_batched + gate_silu + memcpy_dtod + gdn_v_permute
// + fwht_rotate_forward + launch_quantize (6 launches -> 1).
extern "C" __global__ void gdn_out_norm_gate_permute_rotq_f32(
    const float* __restrict__ x_in,     // [n_heads*head_dim] pre-norm recurrent_out (read-only)
    const float* __restrict__ z,        // [n_heads*head_dim] z_buf (source-indexed)
    const float* __restrict__ gamma,    // [head_dim] linear_norm
    const float* __restrict__ signs,    // [v_dim]
    signed char* __restrict__ out,      // [v_dim] int8 (destination order)
    float* __restrict__ ascale,         // [v_dim/16]
    float inv_head_dim, float eps,
    int head_dim, int n_k, int rep, int hblock)
{
    const int tid = threadIdx.x;        // == index within the Hadamard window
    const int base = blockIdx.x * hblock;
    const int gj = base + tid;          // destination element index
    const int H = gj / head_dim;        // destination head
    const int off = gj % head_dim;
    const int nk = H / rep;
    const int r = H % rep;
    const int S = r * n_k + nk;         // source head (identity when rep==1)
    const int src = S * head_dim + off;

    __shared__ float buf[1024];         // butterfly window
    __shared__ float hsq[1024];         // per-head sumsq scratch (head region)

    float xv = x_in[src];
    float zv = z[src];
    hsq[tid] = xv * xv;
    __syncthreads();

    // rmsnorm_batched_f32's tree pairing, folded within the head region:
    // s = 128, 64, ..., 1 (the 256-slot original folds +0 above head_dim,
    // which is bit-neutral — skipped here). Requires head_dim <= 256.
    for (int s = 128; s >= 1; s >>= 1) {
        if (off < s && (off + s) < head_dim) {
            hsq[(H % (hblock / head_dim)) * head_dim + off] +=
                hsq[(H % (hblock / head_dim)) * head_dim + off + s];
        }
        __syncthreads();
    }
    const int hreg = (H % (hblock / head_dim)) * head_dim;
    float inv_rms;
    if (off == 0) {
        float mean_sq = hsq[hreg] * inv_head_dim;
        hsq[hreg] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = hsq[hreg];

    // rmsnorm value order then gate (gate_silu_f32: x *= g/(1+e^-g)).
    float normed = xv * inv_rms * gamma[off];
    float gated = normed * (zv / (1.0f + expf(-zv)));
    buf[tid] = gated * signs[gj];
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

// K4 — fused sigmoid gate + sign + FWHT + quantize (the attention wo input;
// gate_sigmoid_f32's expression verbatim). 3 launches -> 1.
extern "C" __global__ void gate_sigmoid_sign_fwht_quantize_f32(
    const float* __restrict__ x,        // [dim] attn_out
    const float* __restrict__ gate,     // [dim] attn_gate
    const float* __restrict__ signs,    // [dim]
    signed char* __restrict__ out,      // [dim] int8
    float* __restrict__ ascale,         // [dim/16]
    int hblock)
{
    __shared__ float buf[1024];
    const int base = blockIdx.x * hblock;
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        float g = gate[base + i];
        float v = x[base + i] * (1.0f / (1.0f + expf(-g)));
        buf[i] = v * signs[base + i];
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

// K5 — fused SwiGLU + sign + FWHT + quantize (the FFN down input; swiglu_f32's
// expression verbatim: silu_g * up). 3 launches -> 1.
extern "C" __global__ void swiglu_sign_fwht_quantize_f32(
    const float* __restrict__ gate,     // [dim] ffn_gate
    const float* __restrict__ up,       // [dim] ffn_up
    const float* __restrict__ signs,    // [dim]
    signed char* __restrict__ out,      // [dim] int8
    float* __restrict__ ascale,         // [dim/16]
    int hblock)
{
    __shared__ float buf[1024];
    const int base = blockIdx.x * hblock;
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        float g = gate[base + i];
        float silu_g = g / (1.0f + expf(-g));
        float v = silu_g * up[base + i];
        buf[i] = v * signs[base + i];
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

// ===========================================================================
// Issue 980 T4 prefill lane — BATCHED rotation kernels ([p rows x width]
// activations; sign index = element index % width). Prefill is compute-
// dominated (launches amortize inside the captured layer-loop graph), so
// these are the SPLIT-path batched twins — no fusion heroics needed.
// ===========================================================================

// Batched forward rotation: `x <- S⊙(H·x)` per (row, hblock) segment. One
// CUDA block per hblock segment; segments never straddle rows (width %
// hblock == 0, asserted at the wrapper — the fold geometry).
extern "C" __global__ void fwht_rotate_forward_batched_f32(
    float* x, const float* signs, long long total, int width, int hblock)
{
    __shared__ float smem[1024];
    long long base = (long long)blockIdx.x * hblock;
    if (base >= total) return;
    int row_off = (int)(base % (long long)width);  // segment stays in-row
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        smem[i] = x[base + i] * signs[row_off + i];
    }
    __syncthreads();
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 2; step <= hblock; step <<= 1) {
        int half = step >> 1;
        for (int i = threadIdx.x; i < hblock / 2; i += blockDim.x) {
            int blk = (i / half) * step;
            int off = i % half;
            float a = smem[blk + off];
            float b = smem[blk + off + half];
            smem[blk + off] = (a + b) * inv_sqrt2;
            smem[blk + off + half] = (a - b) * inv_sqrt2;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        x[base + i] = smem[i];
    }
}

// Batched inverse rotation (the embedding lookup twin): Hadamard FIRST,
// sign SECOND, per row.
extern "C" __global__ void fwht_rotate_inverse_batched_f32(
    float* x, const float* signs, long long total, int width, int hblock)
{
    __shared__ float smem[1024];
    long long base = (long long)blockIdx.x * hblock;
    if (base >= total) return;
    int row_off = (int)(base % (long long)width);
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        smem[i] = x[base + i];
    }
    __syncthreads();
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 2; step <= hblock; step <<= 1) {
        int half = step >> 1;
        for (int i = threadIdx.x; i < hblock / 2; i += blockDim.x) {
            int blk = (i / half) * step;
            int off = i % half;
            float a = smem[blk + off];
            float b = smem[blk + off + half];
            smem[blk + off] = (a + b) * inv_sqrt2;
            smem[blk + off + half] = (a - b) * inv_sqrt2;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        x[base + i] = smem[i] * signs[row_off + i];
    }
}

// Issue 980 T4-ALT — batched COPY-rotate (the whole-prefill staging twin):
// reads the PRIMAL normed input `src`, writes the rotated copy `dst`. One
// memory pass where memcpy_dtod + in-place rotate costs two, no &mut alias
// on the destination (CUDA-graph-safe kernel node), and mathematically
// identical to memcpy+fwht_rotate_forward_batched (sign first, Hadamard
// second — the forward transform's order, applied while staging).
extern "C" __global__ void fwht_forward_copy_batched_f32(
    const float* src, float* dst, const float* signs, long long total, int width, int hblock)
{
    __shared__ float smem[1024];
    long long base = (long long)blockIdx.x * hblock;
    if (base >= total) return;
    int row_off = (int)(base % (long long)width);
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        smem[i] = src[base + i] * signs[row_off + i];
    }
    __syncthreads();
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 2; step <= hblock; step <<= 1) {
        int half = step >> 1;
        for (int i = threadIdx.x; i < hblock / 2; i += blockDim.x) {
            int blk = (i / half) * step;
            int off = i % half;
            float a = smem[blk + off];
            float b = smem[blk + off + half];
            smem[blk + off] = (a + b) * inv_sqrt2;
            smem[blk + off + half] = (a - b) * inv_sqrt2;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < hblock; i += blockDim.x) {
        dst[base + i] = smem[i];
    }
}

// Batched tiled->grouped head permute over [p x v_dim]: one thread per
// element; the per-row mapping is gdn_v_permute_f32's verbatim.
extern "C" __global__ void gdn_v_permute_batched_f32(
    const float* __restrict__ tmp, float* __restrict__ x,
    long long total, int hd, int n_k, int rep, int v_dim)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int j = (int)(i % (long long)v_dim);   // index within the row
    int r = (int)(i / (long long)v_dim);   // row
    int head = j / hd;          // grouped head index: nk * rep + rr
    int off = j % hd;
    int nk = head / rep;
    int rr = head % rep;
    int src_head = rr * n_k + nk; // tiled head index
    x[i] = tmp[(long long)r * v_dim + src_head * hd + off];
}

// Batched dense escape-set GEMM: out_{a,b}[p x rows] = x[p x n] @
// w_{a,b}^T[rows x n]. v4 (Issue 985 rung 1, 2026-09-19 session 3) — the
// fast path is REGISTER-TILED: threads < 96 each own a 4-token x 4-col
// output tile in 16 scalar accumulators; per k, 8 LDS operand loads -> 16
// FMAs (v3's scattered (t,c) mapping paid 6 loads -> 3 FMAs — ~4x the LDS
// traffic cut on the LDS-bound kernel, the named 25-30 ms/chunk residue).
// The 512-thread BLOCK stays for staging — a pure 96-thread cut measured
// SLOWER e2e (3302 vs 3402): global->smem staging needs the wide block's
// memory-level parallelism. NUMERICS: the per-output accumulation order is
// IDENTICAL to v3 — 4 chains of 16 consecutive kk per 64-wide k chunk,
// tree-summed (s0+s1)+(s2+s3), added to the running accumulator across
// chunks — so the folded-lane pins (argmax 248046 / FNV 8b1ec1f8eec78d23)
// hold EXACT (bench-verified). NVRTC TRAP (measured this session): the
// array form (float s[4][4]) MISCOMPILES under default options — s[m][nn]
// collapses across nn (out[t][c] = dot(x[t], SUM of the col-quad's w rows));
// all hot-loop state is scalars for that reason. v3 heritage: the
// stride-65 bank pad, the grid.y a/b split, and the 32-token tiles stay.
// Generic tail path (p % 32 != 0, or rows != 48) keeps the v1 loop verbatim.
extern "C" __global__ void gemm_dense_ab_batched_f32(
    const float* __restrict__ x,      // [p x n] primal normed
    const float* __restrict__ wa,     // [rows x n]
    const float* __restrict__ wb,     // [rows x n]
    float* __restrict__ out_a,        // [p x rows]
    float* __restrict__ out_b,        // [p x rows]
    int p, int rows, int n)
{
    const float* w = (blockIdx.y == 0) ? wa : wb;
    float* out = (blockIdx.y == 0) ? out_a : out_b;
    const int T0 = blockIdx.x * 32;            // first token of the tile (v3: 32-token tiles — grid (p/32, 2) puts one block on EVERY SM at p=2048; the v2 64-token grid left half the part idle)
    const int t_count = min(32, p - T0);       // live tokens in the tile
    __shared__ float xs[32][65];
    __shared__ float ws[48][65];

    if (t_count == 32 && rows == 48) {
        // v4 register-tiled fast path: threads (ti, tj) = (tid/12, tid%12)
        // for tid < 96 each own a 4-token x 4-col output tile in 16 scalar
        // accumulators. The 512-thread BLOCK is kept for the staging phase —
        // the first 96-thread cut measured SLOWER e2e (3302 vs 3402): staging
        // from global with 96 threads loses the memory-level parallelism the
        // 512-thread block provides, and the staging loss ate the LDS win.
        // Threads 96..511 skip the compute and wait at the barrier.
        // All compute state in SCALARS — the array form (float s[4][4])
        // miscompiles under NVRTC's default options, measured to collapse
        // s[m][nn] across nn (out[t][c] returned dot(x[t], SUM of the
        // col-quad's w rows); the scalar form is correct on the same data —
        // Issue 985 debug session 2026-09-19, the flat-probe bisect).
        const int ti = threadIdx.x / 12;
        const int tj = threadIdx.x % 12;
        // acc_mn: running output accumulators (v3 acc semantics; dead for
        // tid >= 96 — the branch below gates every use).
        float a00=0.f, a01=0.f, a02=0.f, a03=0.f;
        float a10=0.f, a11=0.f, a12=0.f, a13=0.f;
        float a20=0.f, a21=0.f, a22=0.f, a23=0.f;
        float a30=0.f, a31=0.f, a32=0.f, a33=0.f;
        for (int k0 = 0; k0 < n; k0 += 64) {
            for (int idx = threadIdx.x; idx < 32 * 64; idx += blockDim.x) {
                int t = idx / 64, kk = idx % 64;
                int k = k0 + kk;
                xs[t][kk] = (k < n) ? x[(long long)(T0 + t) * n + k] : 0.0f;
            }
            for (int idx = threadIdx.x; idx < rows * 64; idx += blockDim.x) {
                int c = idx / 64, kk = idx % 64;
                int k = k0 + kk;
                ws[c][kk] = (k < n) ? w[(long long)c * n + k] : 0.0f;
            }
            __syncthreads();
            if (threadIdx.x < 96) {
            // The v3 per-output order, held exactly: per 64-wide k chunk,
            // 4 chains of 16 consecutive kk (q = 0..3), tree-summed per
            // output as (s_q0 + s_q1) + (s_q2 + s_q3), then ONE add into the
            // running acc. The l/r halves carry (s0+s1) and (s2+s3): the
            // l accumulators seed with the q=0 partial (0.0f + s0 == s0
            // bitwise — a 16-FMA chain from +0.0f can never produce -0.0f)
            // and add the q=1 partial (l = s0 + s1, one rounding — the
            // v3 (s0+s1) verbatim); r likewise for q=2/q=3. The final
            // acc += l + r is v3's acc += (s0+s1)+(s2+s3) verbatim.
            float l00=0.f, l01=0.f, l02=0.f, l03=0.f;
            float l10=0.f, l11=0.f, l12=0.f, l13=0.f;
            float l20=0.f, l21=0.f, l22=0.f, l23=0.f;
            float l30=0.f, l31=0.f, l32=0.f, l33=0.f;
            float r00=0.f, r01=0.f, r02=0.f, r03=0.f;
            float r10=0.f, r11=0.f, r12=0.f, r13=0.f;
            float r20=0.f, r21=0.f, r22=0.f, r23=0.f;
            float r30=0.f, r31=0.f, r32=0.f, r33=0.f;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                float s00=0.f, s01=0.f, s02=0.f, s03=0.f;
                float s10=0.f, s11=0.f, s12=0.f, s13=0.f;
                float s20=0.f, s21=0.f, s22=0.f, s23=0.f;
                float s30=0.f, s31=0.f, s32=0.f, s33=0.f;
                #pragma unroll
                for (int k16 = 0; k16 < 16; k16++) {
                    const int kk = q * 16 + k16;
                    const float x0 = xs[ti * 4 + 0][kk];
                    const float x1 = xs[ti * 4 + 1][kk];
                    const float x2 = xs[ti * 4 + 2][kk];
                    const float x3 = xs[ti * 4 + 3][kk];
                    const float w0 = ws[tj * 4 + 0][kk];
                    const float w1 = ws[tj * 4 + 1][kk];
                    const float w2 = ws[tj * 4 + 2][kk];
                    const float w3 = ws[tj * 4 + 3][kk];
                    s00 = fmaf(x0, w0, s00); s01 = fmaf(x0, w1, s01); s02 = fmaf(x0, w2, s02); s03 = fmaf(x0, w3, s03);
                    s10 = fmaf(x1, w0, s10); s11 = fmaf(x1, w1, s11); s12 = fmaf(x1, w2, s12); s13 = fmaf(x1, w3, s13);
                    s20 = fmaf(x2, w0, s20); s21 = fmaf(x2, w1, s21); s22 = fmaf(x2, w2, s22); s23 = fmaf(x2, w3, s23);
                    s30 = fmaf(x3, w0, s30); s31 = fmaf(x3, w1, s31); s32 = fmaf(x3, w2, s32); s33 = fmaf(x3, w3, s33);
                }
                if (q == 0) {
                    l00 = s00; l01 = s01; l02 = s02; l03 = s03;
                    l10 = s10; l11 = s11; l12 = s12; l13 = s13;
                    l20 = s20; l21 = s21; l22 = s22; l23 = s23;
                    l30 = s30; l31 = s31; l32 = s32; l33 = s33;
                } else if (q == 1) {
                    l00 += s00; l01 += s01; l02 += s02; l03 += s03;
                    l10 += s10; l11 += s11; l12 += s12; l13 += s13;
                    l20 += s20; l21 += s21; l22 += s22; l23 += s23;
                    l30 += s30; l31 += s31; l32 += s32; l33 += s33;
                } else if (q == 2) {
                    r00 = s00; r01 = s01; r02 = s02; r03 = s03;
                    r10 = s10; r11 = s11; r12 = s12; r13 = s13;
                    r20 = s20; r21 = s21; r22 = s22; r23 = s23;
                    r30 = s30; r31 = s31; r32 = s32; r33 = s33;
                } else {
                    r00 += s00; r01 += s01; r02 += s02; r03 += s03;
                    r10 += s10; r11 += s11; r12 += s12; r13 += s13;
                    r20 += s20; r21 += s21; r22 += s22; r23 += s23;
                    r30 += s30; r31 += s31; r32 += s32; r33 += s33;
                }
            }
            a00 += l00 + r00; a01 += l01 + r01; a02 += l02 + r02; a03 += l03 + r03;
            a10 += l10 + r10; a11 += l11 + r11; a12 += l12 + r12; a13 += l13 + r13;
            a20 += l20 + r20; a21 += l21 + r21; a22 += l22 + r22; a23 += l23 + r23;
            a30 += l30 + r30; a31 += l31 + r31; a32 += l32 + r32; a33 += l33 + r33;
            }
            __syncthreads();
        }
        if (threadIdx.x < 96) {
        out[(long long)(T0 + ti * 4 + 0) * rows + (tj * 4 + 0)] = a00;
        out[(long long)(T0 + ti * 4 + 0) * rows + (tj * 4 + 1)] = a01;
        out[(long long)(T0 + ti * 4 + 0) * rows + (tj * 4 + 2)] = a02;
        out[(long long)(T0 + ti * 4 + 0) * rows + (tj * 4 + 3)] = a03;
        out[(long long)(T0 + ti * 4 + 1) * rows + (tj * 4 + 0)] = a10;
        out[(long long)(T0 + ti * 4 + 1) * rows + (tj * 4 + 1)] = a11;
        out[(long long)(T0 + ti * 4 + 1) * rows + (tj * 4 + 2)] = a12;
        out[(long long)(T0 + ti * 4 + 1) * rows + (tj * 4 + 3)] = a13;
        out[(long long)(T0 + ti * 4 + 2) * rows + (tj * 4 + 0)] = a20;
        out[(long long)(T0 + ti * 4 + 2) * rows + (tj * 4 + 1)] = a21;
        out[(long long)(T0 + ti * 4 + 2) * rows + (tj * 4 + 2)] = a22;
        out[(long long)(T0 + ti * 4 + 2) * rows + (tj * 4 + 3)] = a23;
        out[(long long)(T0 + ti * 4 + 3) * rows + (tj * 4 + 0)] = a30;
        out[(long long)(T0 + ti * 4 + 3) * rows + (tj * 4 + 1)] = a31;
        out[(long long)(T0 + ti * 4 + 3) * rows + (tj * 4 + 2)] = a32;
        out[(long long)(T0 + ti * 4 + 3) * rows + (tj * 4 + 3)] = a33;
        }
        return;
    }

    // Generic tail path (v1 shape — reached when p % 32 != 0 or rows != 48;
    // acc[16] bounds the worst case at any blockDim in [96, 512]).
    {
        const int cols = rows;
        float acc[16];
        int n_acc = t_count * cols / blockDim.x
                  + ((threadIdx.x < t_count * cols % blockDim.x) ? 1 : 0);
        #pragma unroll
        for (int q = 0; q < 16; q++) acc[q] = 0.0f;
        for (int k0 = 0; k0 < n; k0 += 64) {
            for (int idx = threadIdx.x; idx < t_count * 64; idx += blockDim.x) {
                int t = idx / 64, kk = idx % 64;
                int k = k0 + kk;
                xs[t][kk] = (k < n) ? x[(long long)(T0 + t) * n + k] : 0.0f;
            }
            for (int idx = threadIdx.x; idx < cols * 64; idx += blockDim.x) {
                int c = idx / 64, kk = idx % 64;
                int k = k0 + kk;
                ws[c][kk] = (k < n) ? w[(long long)c * n + k] : 0.0f;
            }
            __syncthreads();
            for (int q = 0; q < n_acc; q++) {
                int idx = q * blockDim.x + threadIdx.x;
                if (idx >= t_count * cols) break;
                int t = idx / cols, c = idx % cols;
                float s = 0.0f;
                for (int kk = 0; kk < 64; kk++) {
                    s = fmaf(xs[t][kk], ws[c][kk], s);
                }
                acc[q] += s;
            }
            __syncthreads();
        }
        for (int q = 0; q < n_acc; q++) {
            int idx = q * blockDim.x + threadIdx.x;
            if (idx >= t_count * cols) break;
            int t = idx / cols, c = idx % cols;
            out[(long long)(T0 + t) * rows + c] = acc[q];
        }
    }
}

// Issue 980 C0.5 — the fused ROTATE+QUANTIZE (the prefill escalation):
// one block per token row, the row resident in dynamic smem. Sign+FWHT
// (bit-identical op order to fwht_forward_copy_batched), then the q8
// single-plane quantize VERBATIM from the mma module's QBODY_Q8(Full)
// (whole-row absmax -> s = div_full(m,127) -> rne(x/s) clamp pack) — so
// the fused output is BYTE-IDENTICAL to copy-rotate + launch_quantize_q8_div
// (the escalation's bitexact contract; the pins hold across the fusion).
__device__ __forceinline__ float qrot_div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float qrot_rne_f32(float x)
{
    float r;
    asm("cvt.rni.f32.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
// C0.5 rung-2 reopen (2026-09-19 session 2): ptxas CONTRACTS mul.rn.f32 +
// add.rn.f32 pairs into FFMA at the SASS level — invisible in PTX, value-
// changing (single- vs double-rounding, data-dependent 1 ULP). Measured:
// the production warp kernel compiled with 130 FFMAs vs the block chain's
// ZERO — the registered 1–2 ULP row-scale divergence (rows whose max
// element crossed a contracted pair). The block kernel cannot contract
// (its adds read smem loads, not products). Inline asm is the only barrier
// ptxas cannot see through (the qrot_div_full precedent) — every FP op in
// the warp butterfly goes through these guards.
__device__ __forceinline__ float qrot_add(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float qrot_sub(float a, float b)
{
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float qrot_mul(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#define QROT_BODY(NAME, PERMUTE)                                                \
extern "C" __global__ void NAME(                                              \
    const float* __restrict__ input,  /* [p * n] primal */                     \
    const float* __restrict__ signs,  /* [n] per-width sign vector */          \
    unsigned int* __restrict__ q_w,    /* [p * (n/4)] q8 words */               \
    float* __restrict__ s_out,         /* [p] row scales */                     \
    int n, int hblock,                                                          \
    int hd, int n_k)   /* permute geometry (PERMUTE only) */                    \
{                                                                              \
    extern __shared__ __align__(16) float smem[];  /* row [n] + red [256] */   \
    float* red = smem + n;                                                     \
    const int row = blockIdx.x;                                                \
    const int tid = threadIdx.x;                                               \
    const long base = (long)row * n;                                           \
                                                                               \
    if (PERMUTE) {                                                             \
        const int rep = (n / hd) / n_k;                                        \
        for (int j = tid; j < n; j += blockDim.x) {                            \
            int head = j / hd;                                                 \
            int off = j % hd;                                                  \
            int nk = head / rep;                                               \
            int rr = head % rep;                                               \
            int src_head = rr * n_k + nk;                                      \
            smem[j] = input[base + (long)src_head * hd + off];                 \
        }                                                                      \
    } else {                                                                   \
        for (int c = tid; c < n; c += blockDim.x) smem[c] = input[base + c];   \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    const float inv_sqrt2 = 0.7071067811865476f;                               \
    for (int b = 0; b < n / hblock; ++b) {                                     \
        const int off = b * hblock;                                            \
        for (int i = tid; i < hblock; i += blockDim.x)                         \
            smem[off + i] *= signs[off + i];                                   \
        __syncthreads();                                                       \
        for (int step = 2; step <= hblock; step <<= 1) {                       \
            int half = step >> 1;                                              \
            for (int i = tid; i < hblock / 2; i += blockDim.x) {               \
                int blk = (i / half) * step;                                   \
                int o = i % half;                                              \
                float a = smem[off + blk + o];                                 \
                float bv = smem[off + blk + o + half];                          \
                smem[off + blk + o] = (a + bv) * inv_sqrt2;                    \
                smem[off + blk + o + half] = (a - bv) * inv_sqrt2;             \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
    }                                                                          \
                                                                               \
    float mx = 0.0f;                                                           \
    for (int c = tid; c < n; c += blockDim.x) {                                \
        float a = smem[c] < 0.0f ? -smem[c] : smem[c];                         \
        mx = a > mx ? a : mx;                                                  \
    }                                                                          \
    red[tid] = mx;                                                             \
    __syncthreads();                                                           \
    float m = 0.0f;                                                            \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < 256; ++i) {                                            \
        m = red[i] > m ? red[i] : m;                                           \
    }                                                                          \
    const float s = m > 0.0f ? qrot_div_full(m, 127.0f) : 1.0f;                \
    if (tid == 0) s_out[row] = s;                                              \
                                                                               \
    const int words = n / 4;                                                   \
    for (int w = tid; w < words; w += blockDim.x) {                            \
        unsigned int wh = 0u;                                                 \
        _Pragma("unroll")                                                     \
        for (int j = 0; j < 4; ++j) {                                          \
            float x = smem[(long)w * 4 + j];                                   \
            float xf = qrot_div_full(x, s);                                    \
            float qh = qrot_rne_f32(xf);                                       \
            qh = qh > 127.0f ? 127.0f : (qh < -127.0f ? -127.0f : qh);         \
            wh |= ((unsigned int)(int)qh & 0xFFu) << (8 * j);                  \
        }                                                                      \
        q_w[(long)row * words + w] = wh;                                       \
    }                                                                          \
}

QROT_BODY(quantize_rotate_q8_full_f32, 0)
QROT_BODY(quantize_permute_rotate_q8_full_f32, 1)

// Issue 980 C0.5 rung 2 — the WARP-PARALLEL register-resident variant:
// each warp owns ONE 1024-chunk (32 elements per lane), the butterfly runs
// bits 0..4 via __shfl_xor (5 stages, warp-synchronous — no block syncs)
// and bits 5..9 REGISTER-LOCAL (k^b pairs in-lane) — the block's only
// __syncthreads is the single row-absmax reduce. Same op order as the
// block kernel (stage sequence + (a±b) assignment) → BIT-IDENTICAL output.
// Eligible when hblock == 1024 and n/1024 <= 8 (one chunk per warp — v[]
// stays register-resident through the quantize); wider rows dispatch to the
// block kernel (slower, correct).
#define QROT_WARP_BODY(NAME, PERMUTE)                                          \
extern "C" __global__ void NAME(                                              \
    const float* __restrict__ input, const float* __restrict__ signs,         \
    unsigned int* __restrict__ q_w, float* __restrict__ s_out,                \
    int n, int hblock, int hd, int n_k)                                        \
{                                                                              \
    __shared__ float red[8];                                                   \
    const int row = blockIdx.x;                                                \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int warp = tid >> 5;                                                 \
    const long base = (long)row * n;                                           \
    const int nchunks = n / hblock;                                            \
    const float inv_sqrt2 = 0.7071067811865476f;                               \
    float v[32];                                                               \
    float mx = 0.0f;                                                           \
    const int my_chunk = warp;                                                 \
    if (my_chunk < nchunks) {                                                  \
        const int coff = my_chunk * hblock;                                     \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k) {                                         \
            const int e = coff + k * 32 + lane;                                \
            long src = e;                                                      \
            if (PERMUTE) {                                                     \
                const int rep = (n / hd) / n_k;                                \
                const int head = e / hd;                                       \
                const int off = e % hd;                                        \
                const int nk = head / rep;                                     \
                const int rr = head % rep;                                     \
                src = (long)(rr * n_k + nk) * hd + off;                        \
            }                                                                  \
            v[k] = qrot_mul(input[base + src], signs[e]);                      \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int sb = 1; sb <= 16; sb <<= 1) {                                 \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 32; ++k) {                                     \
                const float p = __shfl_xor_sync(0xffffffffu, v[k], sb);        \
                /* the LOWER element (bit clear) keeps (a+b); the UPPER gets \
                 * (a-b) = (partner - self) — the original kernel's assignment \
                 * (blk+off → +, blk+off+half → −) viewed from the upper lane. \
                 * qrot_* asm guards: ptxas cannot contract these (the rung-2 \
                 * FFMA divergence — see the launcher comment). */               \
                v[k] = ((lane & sb) == 0)                                      \
                    ? qrot_mul(qrot_add(v[k], p), inv_sqrt2)                   \
                    : qrot_mul(qrot_sub(p, v[k]), inv_sqrt2);                  \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int b = 1; b <= 16; b <<= 1) {                                    \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 32; ++k) {                                     \
                if ((k & b) == 0) {                                            \
                    const float a = v[k];                                      \
                    const float b2 = v[k | b];                                 \
                    v[k] = qrot_mul(qrot_add(a, b2), inv_sqrt2);               \
                    v[k | b] = qrot_mul(qrot_sub(a, b2), inv_sqrt2);           \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k) {                                         \
            const float a = v[k] < 0.0f ? -v[k] : v[k];                        \
            mx = a > mx ? a : mx;                                              \
        }                                                                      \
    }                                                                          \
    float wm = mx;                                                             \
    _Pragma("unroll")                                                          \
    for (int d = 16; d >= 1; d >>= 1)                                          \
        wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, d));                   \
    if (lane == 0) red[warp] = wm;                                             \
    __syncthreads();                                                           \
    float m = 0.0f;                                                            \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 8; ++i) m = red[i] > m ? red[i] : m;                   \
    const float s = m > 0.0f ? qrot_div_full(m, 127.0f) : 1.0f;                \
    if (tid == 0) s_out[row] = s;                                              \
    if (my_chunk < nchunks) {                                                  \
        const int coff = my_chunk * hblock;                                    \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k) {                                         \
            const float xf = qrot_div_full(v[k], s);                           \
            float qh = qrot_rne_f32(xf);                                       \
            qh = qh > 127.0f ? 127.0f : (qh < -127.0f ? -127.0f : qh);         \
            v[k] = qh;                                                         \
        }                                                                      \
        const int lbase = lane & ~3;                                           \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k) {                                         \
            const unsigned int qb = (unsigned int)(int)v[k] & 0xFFu;           \
            const unsigned int b1 = __shfl_sync(0xffffffffu, qb, lbase + 1);   \
            const unsigned int b2 = __shfl_sync(0xffffffffu, qb, lbase + 2);   \
            const unsigned int b3 = __shfl_sync(0xffffffffu, qb, lbase + 3);   \
            if ((lane & 3) == 0) {                                             \
                const int w = coff / 4 + k * 8 + (lane >> 2);                   \
                q_w[(long)row * (n / 4) + w] =                                  \
                    qb | (b1 << 8) | (b2 << 16) | (b3 << 24);                  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

QROT_WARP_BODY(quantize_rotate_q8_warp_f32, 0)
QROT_WARP_BODY(quantize_permute_rotate_q8_warp_f32, 1)

// C0.5 rung-2 ISOLATION PROBE — the QROT_WARP_BODY load + butterflies
// VERBATIM (track it on any edit), tail replaced: snapshots the owned
// chunk to dbg[row][stage][e] after the sign-load (stage 0) and after each
// of the 10 butterfly stages (1..5 = shuffle distances 1..16, 6..10 =
// register distances 32..512). No absmax, no quantize — pure FP forensics
// for the disabled dispatch (compare stage-by-stage vs the host oracle to
// localize the first divergent stage). Test instrument, not on any
// production path.
#define QROT_WARP_ISO_BODY(NAME, PERMUTE)                                      \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ input, const float* __restrict__ signs,          \
    float* __restrict__ dbg, int n, int hblock, int hd, int n_k)               \
{                                                                              \
    const int row = blockIdx.x;                                                \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int warp = tid >> 5;                                                 \
    const long base = (long)row * n;                                           \
    const int nchunks = n / hblock;                                            \
    const float inv_sqrt2 = 0.7071067811865476f;                               \
    float v[32];                                                               \
    const int my_chunk = warp;                                                 \
    if (my_chunk < nchunks) {                                                  \
        const int coff = my_chunk * hblock;                                     \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k) {                                         \
            const int e = coff + k * 32 + lane;                                \
            long src = e;                                                      \
            if (PERMUTE) {                                                     \
                const int rep = (n / hd) / n_k;                                \
                const int head = e / hd;                                       \
                const int off = e % hd;                                        \
                const int nk = head / rep;                                     \
                const int rr = head % rep;                                     \
                src = (long)(rr * n_k + nk) * hd + off;                        \
            }                                                                  \
            v[k] = qrot_mul(input[base + src], signs[e]);                      \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k)                                           \
            dbg[((long)row * 12 + 0) * n + coff + k * 32 + lane] = v[k];       \
        _Pragma("unroll")                                                      \
        for (int sb = 1; sb <= 16; sb <<= 1) {                                 \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 32; ++k) {                                     \
                const float p = __shfl_xor_sync(0xffffffffu, v[k], sb);        \
                v[k] = ((lane & sb) == 0)                                      \
                    ? qrot_mul(qrot_add(v[k], p), inv_sqrt2)                   \
                    : qrot_mul(qrot_sub(p, v[k]), inv_sqrt2);                  \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 32; ++k)                                       \
                dbg[((long)row * 12 + (sb == 1 ? 1 : sb == 2 ? 2 : sb == 4 ? 3 : sb == 8 ? 4 : 5)) * n \
                    + coff + k * 32 + lane] = v[k];                            \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int b = 1; b <= 16; b <<= 1) {                                    \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 32; ++k) {                                     \
                if ((k & b) == 0) {                                            \
                    const float a = v[k];                                      \
                    const float b2 = v[k | b];                                 \
                    v[k] = qrot_mul(qrot_add(a, b2), inv_sqrt2);               \
                    v[k | b] = qrot_mul(qrot_sub(a, b2), inv_sqrt2);           \
                }                                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int k = 0; k < 32; ++k)                                       \
                dbg[((long)row * 12 + 5 + (b == 1 ? 1 : b == 2 ? 2 : b == 4 ? 3 : b == 8 ? 4 : 5)) * n \
                    + coff + k * 32 + lane] = v[k];                            \
        }                                                                      \
        /* session-2 tail probe: the PRODUCTION abs + warp-reduce (character-  \
         * identical to QROT_WARP_BODY) — the per-chunk max lands in slot 11   \
         * at [row][11][chunk] (one scalar per chunk). Localizes which chunk   \
         * (and via stage-10, which element) owns the diverging max. */        \
        float mx2 = 0.0f;                                                      \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < 32; ++k) {                                         \
            const float a = v[k] < 0.0f ? -v[k] : v[k];                        \
            mx2 = a > mx2 ? a : mx2;                                           \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int d = 16; d >= 1; d >>= 1)                                      \
            mx2 = fmaxf(mx2, __shfl_xor_sync(0xffffffffu, mx2, d));            \
        if (lane == 0) dbg[((long)row * 12 + 11) * n + my_chunk] = mx2;        \
    }                                                                          \
}

QROT_WARP_ISO_BODY(quantize_permute_rotate_q8_warp_iso_f32, 1)
"#;

/// Shared-mem capacity bound of the FWHT kernels: `block_size` f32. The
/// loader accepts power-of-2 block sizes; anything above 1024 would exceed
/// the default static shared budget and is refused at construction
/// (Bonsai-2 ships 1024 — the only validated geometry).
const MAX_FWHT_BLOCK: usize = 1024;
const FWHT_THREADS: u32 = 256;
const GATE_THREADS: u32 = 256;
const GEMV_DENSE_THREADS: u32 = 512;
/// C0.5 — the fused rotate+quantize's row-resident smem bound: the widest
/// folded matmul input width (Bonsai-2 ffn_down, 17408) + the 256-slot
/// absmax reduction. The launcher REFUSES wider rows (no known geometry).
const MAX_QROT_ROW: usize = 17408;
const QROT_REDUCTION: usize = 256;
const QROT_THREADS: u32 = 256;

/// Compiled rotation kernel set. Built once in `with_shared` when the model
/// declares `prism.hadamard`; `None` everywhere otherwise (the guard refuses
/// folded models before this point when the kernels are absent).
pub struct RotationKernels {
    fwht_forward: CudaFunction,
    fwht_inverse: CudaFunction,
    gdn_permute: CudaFunction,
    gate_silu: CudaFunction,
    gate_sigmoid: CudaFunction,
    gemv_dense: CudaFunction,
    // Issue 980 T4 escalation — the fused launch-count-reduction set (Bench 940).
    rmsnorm_rotq: CudaFunction,
    gdn_input_rotabq: CudaFunction,
    gdn_out_rotq: CudaFunction,
    gate_sigmoid_rotq: CudaFunction,
    swiglu_rotq: CudaFunction,
    // Issue 980 T4 prefill lane — batched ([p x width]) twins.
    fwht_forward_batched: CudaFunction,
    fwht_inverse_batched: CudaFunction,
    fwht_forward_copy_batched: CudaFunction,
    gdn_permute_batched: CudaFunction,
    gemm_dense_ab_batched: CudaFunction,
    // Issue 980 C0.5 — the fused rotate(+permute)+quantize (the prefill
    // escalation): rotation rides the quantize pass, byte-identical to the
    // unfused chain (see the kernel doc). The *_warp pair is the rung-2
    // shuffle/register variant dispatched at Bonsai-2's geometry.
    quantize_rot_q8: CudaFunction,
    quantize_permute_rot_q8: CudaFunction,
    quantize_rot_q8_warp: CudaFunction,
    quantize_permute_rot_q8_warp: CudaFunction,
    // C0.5 rung-2 isolation probe (see the kernel doc) — test instrument
    // (loaded unconditionally: the kernel ships in ROTATION_CUDA_SRC; the
    // launcher runs from the tests mod).
    #[allow(dead_code)]
    quantize_permute_rot_q8_warp_iso: CudaFunction,
    _module: Arc<CudaModule>,
}

impl RotationKernels {
    /// Compile against the shared context. ~1 nvrtc pass (~10 ms — six tiny
    /// kernels, no templates).
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx(ROTATION_CUDA_SRC)
            .map_err(|e| CudarcKernelError::Compile(format!("rotation: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| CudarcKernelError::Compile(format!("rotation: {e}")))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| CudarcKernelError::Compile(format!("rotation.{name}: {e}")))
        };
        let ret = Self {
            fwht_forward: load("fwht_rotate_forward_f32")?,
            fwht_inverse: load("fwht_rotate_inverse_f32")?,
            gdn_permute: load("gdn_v_permute_f32")?,
            gate_silu: load("gate_silu_f32")?,
            gate_sigmoid: load("gate_sigmoid_f32")?,
            gemv_dense: load("gemv_dense_f32")?,
            rmsnorm_rotq: load("rmsnorm_sign_fwht_quantize_f32")?,
            gdn_input_rotabq: load("gdn_input_rotabq_f32")?,
            gdn_out_rotq: load("gdn_out_norm_gate_permute_rotq_f32")?,
            gate_sigmoid_rotq: load("gate_sigmoid_sign_fwht_quantize_f32")?,
            swiglu_rotq: load("swiglu_sign_fwht_quantize_f32")?,
            fwht_forward_batched: load("fwht_rotate_forward_batched_f32")?,
            fwht_inverse_batched: load("fwht_rotate_inverse_batched_f32")?,
            fwht_forward_copy_batched: load("fwht_forward_copy_batched_f32")?,
            gdn_permute_batched: load("gdn_v_permute_batched_f32")?,
            gemm_dense_ab_batched: load("gemm_dense_ab_batched_f32")?,
            quantize_rot_q8: load("quantize_rotate_q8_full_f32")?,
            quantize_permute_rot_q8: load("quantize_permute_rotate_q8_full_f32")?,
            quantize_rot_q8_warp: load("quantize_rotate_q8_warp_f32")?,
            quantize_permute_rot_q8_warp: load("quantize_permute_rotate_q8_warp_f32")?,
            quantize_permute_rot_q8_warp_iso: load("quantize_permute_rotate_q8_warp_iso_f32")?,
            _module: module,
        };
        // C0.5 — the fused quantize kernels keep the row resident in
        // dynamic smem (row f32 + 256 reduction slots). The widest folded
        // width (Bonsai-2's ffn_down input, 17408) needs 71, 680 B > the
        // 48 KB default — opt in once at load (a MAX; narrower rows launch
        // below it). Wider rows REFUSE at the launcher (folded models with
        // rows above this are not a known geometry).
        for f in [&ret.quantize_rot_q8, &ret.quantize_permute_rot_q8] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                ((MAX_QROT_ROW + QROT_REDUCTION) * core::mem::size_of::<f32>()) as i32,
            )
            .map_err(|e| CudarcKernelError::Compile(format!("qrot smem opt-in: {e}")))?;
        }
        Ok(ret)
    }

    /// `x ← (1/√n)·H_n·(S⊙x)` per block, in place (the folded-matmul input
    /// transform).
    pub(crate) fn fwht_rotate_forward(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        n: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(block_size <= MAX_FWHT_BLOCK, "FWHT block {block_size} exceeds the {MAX_FWHT_BLOCK}-element shared-mem kernel");
        assert_eq!(x.len(), n, "rotation width must match the slice");
        let n_i = n as i32;
        let bs_i = block_size as i32;
        let grid = (n / block_size).max(1) as u32;
        unsafe {
            stream
                .launch_builder(&self.fwht_forward)
                .arg(x)
                .arg(signs)
                .arg(&n_i)
                .arg(&bs_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// `z ← S⊙((1/√n)·H_n·z)` per block, in place (the embedding-inverse).
    pub(crate) fn fwht_rotate_inverse(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        n: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(block_size <= MAX_FWHT_BLOCK);
        let n_i = n as i32;
        let bs_i = block_size as i32;
        let grid = (n / block_size).max(1) as u32;
        unsafe {
            stream
                .launch_builder(&self.fwht_inverse)
                .arg(x)
                .arg(signs)
                .arg(&n_i)
                .arg(&bs_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// tiled→grouped head permute. `tmp` must already hold a copy of `x`
    /// (dispatched right before by `memcpy_dtod`); this gathers `x` from it.
    pub(crate) fn gdn_v_permute(
        &self,
        stream: &CudaStream,
        tmp: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        len: usize,
        n_v: usize,
        n_k: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(n_k > 0 && n_v.is_multiple_of(n_k) && len.is_multiple_of(n_v));
        let hd = len / n_v;
        let rep = n_v / n_k;
        let len_i = len as i32;
        let hd_i = hd as i32;
        let nk_i = n_k as i32;
        let rep_i = rep as i32;
        unsafe {
            stream
                .launch_builder(&self.gdn_permute)
                .arg(tmp)
                .arg(x)
                .arg(&len_i)
                .arg(&hd_i)
                .arg(&nk_i)
                .arg(&rep_i)
                .launch(LaunchConfig {
                    grid_dim: (len.div_ceil(GATE_THREADS as usize) as u32, 1, 1),
                    block_dim: (GATE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// `x ← silu(gate)⊙x` (ssm_out gating stage, split from Issue 627).
    pub(crate) fn gate_silu(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        let n_i = n as i32;
        unsafe {
            stream
                .launch_builder(&self.gate_silu)
                .arg(x)
                .arg(gate)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (n.div_ceil(GATE_THREADS as usize) as u32, 1, 1),
                    block_dim: (GATE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// `x ← sigmoid(gate)⊙x` (attention wo gating stage, split from 626).
    pub(crate) fn gate_sigmoid(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        let n_i = n as i32;
        unsafe {
            stream
                .launch_builder(&self.gate_sigmoid)
                .arg(x)
                .arg(gate)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (n.div_ceil(GATE_THREADS as usize) as u32, 1, 1),
                    block_dim: (GATE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Dense fp32 GEMV: `out[..rows] = W[..rows×cols] @ x[..cols]` (the
    /// escape-set a/b matvec on the primal normed input).
    pub(crate) fn gemv_dense(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(cols <= MAX_FWHT_BLOCK * MAX_FWHT_BLOCK, "dense GEMV row stride must fit size_t-checked addressing");
        assert_eq!(w.len(), rows * cols, "dense weight extent must be rows*cols");
        let cols_i = cols as i32;
        unsafe {
            stream
                .launch_builder(&self.gemv_dense)
                .arg(out)
                .arg(w)
                .arg(x)
                .arg(&cols_i)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (GEMV_DENSE_THREADS, 1, 1),
                    shared_mem_bytes: (GEMV_DENSE_THREADS as usize * std::mem::size_of::<f32>()) as u32,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    // ── Issue 980 T4 escalation — fused wrappers (Bench 940: +1.90 ms/token
    // was ~614 launches at the in-graph dispatch floor, not GPU work). Each
    // documents the split-path launches it replaces; the bit-exact unit test
    // pins byte-identity against that split path.

    /// K1 — `rmsnorm(x,γ) → S⊙· → FWHT → int8+ascale`, one Hadamard block per
    /// CUDA block (256 threads). Replaces launch_rmsnorm + memcpy_dtod +
    /// fwht_rotate_forward + launch_quantize (4 launches → 1). `dim` must be
    /// a multiple of `block_size` and `block_size ∈ [16, 1024]` — otherwise
    /// the caller must keep the split path (the geometry guard at the wiring).
    pub(crate) fn rmsnorm_rotate_quantize(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        out_i8: &CudaSlice<i8>,
        ascale: &CudaSlice<f32>,
        dim: usize,
        eps: f32,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            dim.is_multiple_of(block_size) && (16..=MAX_FWHT_BLOCK).contains(&block_size),
            "fused rotation geometry: dim {dim} vs hblock {block_size}"
        );
        assert_eq!(x.len(), dim);
        let inv_dim = 1.0f32 / dim as f32;
        let dim_i = dim as i32;
        let hb_i = block_size as i32;
        let grid = (dim / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_rotq)
                .arg(x)
                .arg(gamma)
                .arg(signs)
                .arg(out_i8)
                .arg(ascale)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// K2 — the whole GDN input stage in ONE heterogeneous launch: rotated +
    /// quantized activation for the qkv/z dp4a GEMVs (blocks `[0, dim/hblock)`)
    /// AND the dense fp32 escape-set a/b GEMVs on the primal normed input
    /// (blocks `[dim/hblock, +2*rows)`, normalization folded inline — each
    /// block re-runs the full-row reduction redundantly, so no cross-block
    /// dependency exists). 512 threads. Replaces launch_rmsnorm + memcpy +
    /// fwht + quantize + gemv_a + gemv_b (6 launches → 1).
    pub(crate) fn gdn_input_fused(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        wa: &CudaSlice<f32>,
        wb: &CudaSlice<f32>,
        out_a: &CudaSlice<f32>,
        out_b: &CudaSlice<f32>,
        rows: usize,
        out_i8: &CudaSlice<i8>,
        ascale: &CudaSlice<f32>,
        dim: usize,
        eps: f32,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            dim.is_multiple_of(block_size) && (16..=MAX_FWHT_BLOCK).contains(&block_size),
            "fused rotation geometry: dim {dim} vs hblock {block_size}"
        );
        assert_eq!(wa.len(), rows * dim);
        assert_eq!(wb.len(), rows * dim);
        let inv_dim = 1.0f32 / dim as f32;
        let dim_i = dim as i32;
        let hb_i = block_size as i32;
        let rows_i = rows as i32;
        let grid = (dim / block_size + 2 * rows) as u32;
        unsafe {
            stream
                .launch_builder(&self.gdn_input_rotabq)
                .arg(x)
                .arg(gamma)
                .arg(signs)
                .arg(wa)
                .arg(wb)
                .arg(out_a)
                .arg(out_b)
                .arg(out_i8)
                .arg(ascale)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i)
                .arg(&hb_i)
                .arg(&rows_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (GEMV_DENSE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// K3 — the GDN output chain in ONE kernel: per-head RMSNorm + silu(z)
    /// gate + tiled→grouped head permute + sign + FWHT + quantize. Launched
    /// with `blockDim == block_size` (one thread per window element).
    /// `rep = n_v/n_k` (pass 1 when `!gdn_v_grouped` — the mapping degenerates
    /// to identity). Replaces launch_rmsnorm_batched + gate_silu + memcpy +
    /// gdn_v_permute + fwht + quantize (6 launches → 1). Geometry guard:
    /// `v_dim % block_size == 0 && block_size % head_dim == 0 && head_dim <= 256`.
    pub(crate) fn gdn_out_fused(
        &self,
        stream: &CudaStream,
        x_in: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        out_i8: &CudaSlice<i8>,
        ascale: &CudaSlice<f32>,
        v_dim: usize,
        head_dim: usize,
        n_k: usize,
        rep: usize,
        eps: f32,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            v_dim.is_multiple_of(block_size)
                && block_size.is_multiple_of(head_dim)
                && head_dim <= 256
                && block_size <= MAX_FWHT_BLOCK,
            "fused GDN-out geometry: v_dim {v_dim}, hblock {block_size}, head_dim {head_dim}"
        );
        assert_eq!(x_in.len(), v_dim);
        assert_eq!(z.len(), v_dim);
        let inv_hd = 1.0f32 / head_dim as f32;
        let hd_i = head_dim as i32;
        let nk_i = n_k as i32;
        let rep_i = rep as i32;
        let hb_i = block_size as i32;
        let grid = (v_dim / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.gdn_out_rotq)
                .arg(x_in)
                .arg(z)
                .arg(gamma)
                .arg(signs)
                .arg(out_i8)
                .arg(ascale)
                .arg(&inv_hd)
                .arg(&eps)
                .arg(&hd_i)
                .arg(&nk_i)
                .arg(&rep_i)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// K4 — `x·sigmoid(gate) → S⊙· → FWHT → int8+ascale` (the attention wo
    /// input). Replaces gate_sigmoid + fwht + quantize (3 launches → 1).
    pub(crate) fn gate_sigmoid_rotate_quantize(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        out_i8: &CudaSlice<i8>,
        ascale: &CudaSlice<f32>,
        dim: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            dim.is_multiple_of(block_size) && (16..=MAX_FWHT_BLOCK).contains(&block_size),
            "fused rotation geometry: dim {dim} vs hblock {block_size}"
        );
        let hb_i = block_size as i32;
        let grid = (dim / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.gate_sigmoid_rotq)
                .arg(x)
                .arg(gate)
                .arg(signs)
                .arg(out_i8)
                .arg(ascale)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// K5 — `silu(gate)·up → S⊙· → FWHT → int8+ascale` (the FFN down input).
    /// Replaces launch_swiglu + fwht + quantize (3 launches → 1).
    pub(crate) fn swiglu_rotate_quantize(
        &self,
        stream: &CudaStream,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        out_i8: &CudaSlice<i8>,
        ascale: &CudaSlice<f32>,
        dim: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            dim.is_multiple_of(block_size) && (16..=MAX_FWHT_BLOCK).contains(&block_size),
            "fused rotation geometry: dim {dim} vs hblock {block_size}"
        );
        let hb_i = block_size as i32;
        let grid = (dim / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.swiglu_rotq)
                .arg(gate)
                .arg(up)
                .arg(signs)
                .arg(out_i8)
                .arg(ascale)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    // ── Issue 980 T4 prefill lane — batched ([p rows x width]) twins. The
    // prefill layer loop amortizes launches inside the captured graph, so
    // these are the SPLIT-path shapes (no fused variants needed).
    // NOTE: allow(dead_code) until the whole_prefill wiring lands (the next
    // Issue-980 unit) — landed ahead of it as tested substrate.
    /// Batched forward rotation over `x[..rows*width]`: per-row sign+FWHT.
    /// `width` must be a multiple of `block_size` (Hadamard segments never
    /// straddle rows — the fold geometry).
    #[allow(dead_code)]
    pub(crate) fn fwht_rotate_forward_batched(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            width.is_multiple_of(block_size) && block_size <= MAX_FWHT_BLOCK,
            "batched rotation geometry: width {width} vs hblock {block_size}"
        );
        let total = (rows * width) as i64;
        let width_i = width as i32;
        let hb_i = block_size as i32;
        let grid = (rows * width / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.fwht_forward_batched)
                .arg(x)
                .arg(signs)
                .arg(&total)
                .arg(&width_i)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched inverse rotation (the prefill embedding-lookup twin):
    /// Hadamard FIRST, sign SECOND, per row.
    #[allow(dead_code)]
    pub(crate) fn fwht_rotate_inverse_batched(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            width.is_multiple_of(block_size) && block_size <= MAX_FWHT_BLOCK,
            "batched rotation geometry: width {width} vs hblock {block_size}"
        );
        let total = (rows * width) as i64;
        let width_i = width as i32;
        let hb_i = block_size as i32;
        let grid = (rows * width / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.fwht_inverse_batched)
                .arg(x)
                .arg(signs)
                .arg(&total)
                .arg(&width_i)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 980 T4-ALT — batched COPY-rotate: `dst ← R(src)` over `[rows ×
    /// width]` in one pass. The whole-prefill staging transform (identical
    /// math to memcpy_dtod + [`Self::fwht_rotate_forward_batched`], one
    /// memory pass instead of two, and no `&mut` destination — a plain
    /// kernel node, CUDA-graph-safe and share-capturable).
    #[allow(dead_code)]
    pub(crate) fn fwht_rotate_copy_batched(
        &self,
        stream: &CudaStream,
        src: &CudaSlice<f32>,
        dst: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            width.is_multiple_of(block_size) && block_size <= MAX_FWHT_BLOCK,
            "batched copy-rotation geometry: width {width} vs hblock {block_size}"
        );
        // Capacity semantics (>=, not ==): the dst is the shared grow-only
        // rot_scratch — ONE buffer serving every rotated width (5120 AND
        // 6144 at Bonsai-2), so after a wide growth the narrower calls see a
        // LARGER dst. The kernel touches exactly rows*width elements.
        assert!(src.len() >= rows * width, "copy-rotate src too small");
        assert!(dst.len() >= rows * width, "copy-rotate dst too small");
        let total = (rows * width) as i64;
        let width_i = width as i32;
        let hb_i = block_size as i32;
        let grid = (rows * width / block_size) as u32;
        unsafe {
            stream
                .launch_builder(&self.fwht_forward_copy_batched)
                .arg(src)
                .arg(dst)
                .arg(signs)
                .arg(&total)
                .arg(&width_i)
                .arg(&hb_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (FWHT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 980 C0.5 — the fused ROTATE+QUANTIZE (one pass, row-resident
    /// smem): rotation rides the existing q8 quantize shape — byte-identical
    /// to [`Self::fwht_rotate_copy_batched`] into a staging buffer + the
    /// mma module's `launch_prefill_quantize` q8 arm reading it (the
    /// escalation's bitexact contract — the folded pins hold across the
    /// fusion). Writes the SAME scratch layout (`q_w` words `[p*(n/4)]` +
    /// `s_t` row scales `[p]`) the prefill GEMMs consume.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_rotate_q8(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        q_w: &CudaSlice<u32>,
        s_t: &CudaSlice<f32>,
        n: usize,
        p: usize,
        block_size: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            n.is_multiple_of(block_size) && block_size <= MAX_FWHT_BLOCK,
            "qrot geometry: width {n} vs hblock {block_size}"
        );
        assert!(n <= MAX_QROT_ROW, "qrot row {n} exceeds the smem budget (max {MAX_QROT_ROW})");
        assert_eq!(input.len(), p * n, "qrot input shape");
        let n_i = n as i32;
        let hb_i = block_size as i32;
        // C0.5 rung 2 — the warp-parallel variant (hblock 1024, ≤ 8
        // chunks/row), DISPATCHED (2026-09-19 session 2: the recorded ULP
        // divergence was ptxas FFMA CONTRACTION — mul.rn+add.rn pairs fused
        // at the SASS level, 130 FFMAs vs the block chain's 0 — measured via
        // the stage-by-stage probe `qrot_warp_stage_isolation` + SASS census;
        // every butterfly op now rides qrot_add/sub/mul inline-asm guards,
        // SASS FFMA count 0, bitexact vs the unfused chain, pins hold).
        // Perf-neutral at the fused sites (2671 vs 2664-2669 hybrid — the
        // butterfly syncs were NOT the folded lane's cost; the remaining
        // −29% vs the old lane lives in the gdn/other trace buckets).
        const QROT_WARP_ENABLED: bool = true;
        if QROT_WARP_ENABLED && block_size == 1024 && n / 1024 <= 8 {
            let smem = (8usize * core::mem::size_of::<f32>()) as u32;
            unsafe {
                stream
                    .launch_builder(&self.quantize_rot_q8_warp)
                    .arg(input)
                    .arg(signs)
                    .arg(q_w)
                    .arg(s_t)
                    .arg(&n_i)
                    .arg(&hb_i)
                    .arg(&0)
                    .arg(&0)
                    .launch(LaunchConfig {
                        grid_dim: (p as u32, 1, 1),
                        block_dim: (QROT_THREADS, 1, 1),
                        shared_mem_bytes: smem,
                    })
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            return Ok(());
        }
        let smem = ((n + QROT_REDUCTION) * core::mem::size_of::<f32>()) as u32;
        unsafe {
            stream
                .launch_builder(&self.quantize_rot_q8)
                .arg(input)
                .arg(signs)
                .arg(q_w)
                .arg(s_t)
                .arg(&n_i)
                .arg(&hb_i)
                .arg(&0)
                .arg(&0)
                .launch(LaunchConfig {
                    grid_dim: (p as u32, 1, 1),
                    block_dim: (QROT_THREADS, 1, 1),
                    shared_mem_bytes: smem,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// The GDN-out variant: PERMUTE (tiled→grouped V-heads) + rotate +
        /// quantize in ONE pass — replaces the unfused permute + rotate +
        /// quantize chain at the `out_proj` input site (byte-identical to
        /// it; `hd`/`n_k` = the config's head_dim / `gdn_k_groups`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_permute_rotate_q8(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        q_w: &CudaSlice<u32>,
        s_t: &CudaSlice<f32>,
        n: usize,
        p: usize,
        block_size: usize,
        head_dim: usize,
        n_k: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            n.is_multiple_of(block_size) && block_size <= MAX_FWHT_BLOCK,
            "qrot-permute geometry: width {n} vs hblock {block_size}"
        );
        assert!(n <= MAX_QROT_ROW, "qrot row {n} exceeds the smem budget (max {MAX_QROT_ROW})");
        assert!(
            n_k > 0 && n.is_multiple_of(head_dim) && (n / head_dim).is_multiple_of(n_k),
            "qrot-permute geometry: n {n}, hd {head_dim}, n_k {n_k}"
        );
        assert!(n <= MAX_QROT_ROW, "qrot row {n} exceeds the smem budget (max {MAX_QROT_ROW})");
        assert_eq!(input.len(), p * n, "qrot-permute input shape");
        let n_i = n as i32;
        let hb_i = block_size as i32;
        // C0.5 rung 2 — see quantize_rotate_q8: dispatched (FFMA-contraction
        // fixed via asm guards; `qrot_warp_stage_isolation` is the gate).
        const QROT_WARP_ENABLED: bool = true;
        if QROT_WARP_ENABLED && block_size == 1024 && n / 1024 <= 8 {
            let smem = (8usize * core::mem::size_of::<f32>()) as u32;
            let hd_i = head_dim as i32;
            let nk_i = n_k as i32;
            unsafe {
                stream
                    .launch_builder(&self.quantize_permute_rot_q8_warp)
                    .arg(input)
                    .arg(signs)
                    .arg(q_w)
                    .arg(s_t)
                    .arg(&n_i)
                    .arg(&hb_i)
                    .arg(&hd_i)
                    .arg(&nk_i)
                    .launch(LaunchConfig {
                        grid_dim: (p as u32, 1, 1),
                        block_dim: (QROT_THREADS, 1, 1),
                        shared_mem_bytes: smem,
                    })
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            return Ok(());
        }
        let hd_i = head_dim as i32;
        let nk_i = n_k as i32;
        let smem = ((n + QROT_REDUCTION) * core::mem::size_of::<f32>()) as u32;
        unsafe {
            stream
                .launch_builder(&self.quantize_permute_rot_q8)
                .arg(input)
                .arg(signs)
                .arg(q_w)
                .arg(s_t)
                .arg(&n_i)
                .arg(&hb_i)
                .arg(&hd_i)
                .arg(&nk_i)
                .launch(LaunchConfig {
                    grid_dim: (p as u32, 1, 1),
                    block_dim: (QROT_THREADS, 1, 1),
                    shared_mem_bytes: smem,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// C0.5 rung-2 ISOLATION PROBE launcher (see the kernel doc): dumps the
    /// warp butterfly's per-stage snapshots (`[p][12][n]` f32; slots 0-10 =
    /// post-load + the 10 stages, slot 11 = per-chunk maxima) to `dbg` —
    /// the FFMA-forensics instrument that localized the rung-2 divergence
    /// (ptxas FFMA contraction, 2026-09-19 session 2). Test instrument; not
    /// on any production path. Geometry mirrors the warp arm (hblock 1024,
    /// ≤ 8 chunks/row).
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn warp_butterfly_stage_dump(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        signs: &CudaSlice<f32>,
        dbg: &CudaSlice<f32>,
        n: usize,
        p: usize,
        block_size: usize,
        head_dim: usize,
        n_k: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            block_size == 1024 && n / 1024 <= 8,
            "warp-iso geometry: hblock {block_size}, n {n}"
        );
        assert_eq!(input.len(), p * n, "warp-iso input shape");
        assert_eq!(dbg.len(), p * 12 * n, "warp-iso dbg shape");
        let n_i = n as i32;
        let hb_i = block_size as i32;
        let hd_i = head_dim as i32;
        let nk_i = n_k as i32;
        unsafe {
            stream
                .launch_builder(&self.quantize_permute_rot_q8_warp_iso)
                .arg(input)
                .arg(signs)
                .arg(dbg)
                .arg(&n_i)
                .arg(&hb_i)
                .arg(&hd_i)
                .arg(&nk_i)
                .launch(LaunchConfig {
                    grid_dim: (p as u32, 1, 1),
                    block_dim: (QROT_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched tiled→grouped head permute over `[p x v_dim]`. `tmp` must
    /// already hold a device copy of `x` (memcpy_dtod dispatched right
    /// before). `rep` = v_dim/head_dim/n_k (1 when ungrouped → identity).
    #[allow(dead_code)]
    pub(crate) fn gdn_v_permute_batched(
        &self,
        stream: &CudaStream,
        tmp: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        p: usize,
        v_dim: usize,
        head_dim: usize,
        n_k: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(
            n_k > 0 && v_dim.is_multiple_of(head_dim) && (v_dim / head_dim).is_multiple_of(n_k),
            "batched permute geometry: v_dim {v_dim}, head_dim {head_dim}, n_k {n_k}"
        );
        let total = (p * v_dim) as i64;
        let hd_i = head_dim as i32;
        let nk_i = n_k as i32;
        let rep_i = (v_dim / head_dim / n_k).max(1) as i32;
        let vd_i = v_dim as i32;
        let grid = (p * v_dim).div_ceil(GATE_THREADS as usize) as u32;
        unsafe {
            stream
                .launch_builder(&self.gdn_permute_batched)
                .arg(tmp)
                .arg(x)
                .arg(&total)
                .arg(&hd_i)
                .arg(&nk_i)
                .arg(&rep_i)
                .arg(&vd_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (GATE_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched dense escape-set GEMM: `out_{a,b}[p x rows] = x[p x n] @
    /// w_{a,b}^T`. `rows <= 48`; the v4 register-tiled fast path additionally
    /// requires `rows == 48` (exact 4×4 thread tiles; anything else takes the
    /// generic tail). 512-thread blocks: staging uses every thread (the MLP
    /// the 96-thread cut lost), the register-tiled compute runs on threads
    /// < 96 (Issue 985 rung 1).
    #[allow(dead_code)]
    pub(crate) fn gemm_dense_ab_batched(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        wa: &CudaSlice<f32>,
        wb: &CudaSlice<f32>,
        out_a: &CudaSlice<f32>,
        out_b: &CudaSlice<f32>,
        p: usize,
        rows: usize,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        assert!(rows <= 48, "dense ab GEMM tile supports rows <= 48 (got {rows})");
        assert_eq!(wa.len(), rows * n);
        assert_eq!(wb.len(), rows * n);
        assert_eq!(x.len(), p * n);
        let p_i = p as i32;
        let rows_i = rows as i32;
        let n_i = n as i32;
        let grid = p.div_ceil(32) as u32;
        // v2: grid.y splits a (0) / b (1) — one weight matrix per block
        // (2x the blocks on a 128-SM part; the v1 single grid left 75% of
        // the SMs idle at p=2048).
        unsafe {
            stream
                .launch_builder(&self.gemm_dense_ab_batched)
                .arg(x)
                .arg(wa)
                .arg(wb)
                .arg(out_a)
                .arg(out_b)
                .arg(&p_i)
                .arg(&rows_i)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 2, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }
}

/// Per-width device sign tables + the geometry the forward needs. The three
/// Bonsai-2 widths (5120 = in_proj/gate/up/lm_head/attn-qkv inputs,
/// 6144 = attn_wo/ssm_out inputs, 17408 = ffn_down input) are resolved by
/// lookup; a folded matmul at an untabled width is a loader bug (the loader
/// validated the set at load).
pub struct RotationTables {
    pub kernels: RotationKernels,
    pub block_size: usize,
    pub signs: Vec<(usize, CudaSlice<f32>)>,
    pub gdn_v_grouped: bool,
    pub gdn_v_heads: usize,
    pub gdn_k_groups: usize,
    pub inverse_embedding: bool,
}

impl RotationTables {
    /// Build from the parsed CPU config: compile the kernels + upload the
    /// sign vectors as f32 (+1/-1).
    pub(crate) fn build(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        cfg: &TernaryRotationConfig,
    ) -> Result<Self, CudarcKernelError> {
        if cfg.block_size > MAX_FWHT_BLOCK {
            return Err(CudarcKernelError::Compile(format!(
                "prism.hadamard block_size {} exceeds the GPU FWHT kernel's {}-element shared-memory bound",
                cfg.block_size, MAX_FWHT_BLOCK
            )));
        }
        let kernels = RotationKernels::new(ctx)?;
        let mut signs = Vec::with_capacity(cfg.signs.len());
        for (width, v) in &cfg.signs {
            let row: Vec<f32> = v.iter().map(|&s| s as f32).collect();
            debug_assert_eq!(row.len(), *width);
            signs.push((*width, stream.clone_htod(&row).map_err(alloc_err)?));
        }
        Ok(Self {
            kernels,
            block_size: cfg.block_size,
            signs,
            gdn_v_grouped: cfg.gdn_v_grouped,
            gdn_v_heads: cfg.gdn_v_heads,
            gdn_k_groups: cfg.gdn_k_groups,
            inverse_embedding: cfg.inverse_embedding,
        })
    }

    pub(crate) fn signs_for_width(&self, width: usize) -> &CudaSlice<f32> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, s)| s)
            .unwrap_or_else(|| panic!("rotation sign table missing width {width} (loader validated the set — this is a wiring bug)"))
    }

    /// Issue 980 T4 escalation — whether the fused K1/K4/K5 geometry guards
    /// hold at `dim` (the split path stays the fallback otherwise; Bonsai-2's
    /// 1024-block geometry always takes the fused path).
    pub(crate) fn fused_geometry_ok(&self, dim: usize) -> bool {
        dim.is_multiple_of(self.block_size)
            && (16..=MAX_FWHT_BLOCK).contains(&self.block_size)
    }

    /// The K3 (GDN output chain) guard — additionally requires the Hadamard
    /// window to tile whole heads and the per-head reduction replication's
    /// head_dim bound. `rep` = n_v_heads / gdn_k_groups (1 when ungrouped).
    pub(crate) fn gdn_out_fused_ok(&self, v_dim: usize, head_dim: usize) -> bool {
        v_dim.is_multiple_of(self.block_size)
            && self.block_size.is_multiple_of(head_dim)
            && head_dim <= 256
            && self.block_size <= MAX_FWHT_BLOCK
    }
}

fn alloc_err(e: cudarc::driver::DriverError) -> CudarcKernelError {
    CudarcKernelError::Launch(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cudarc_kernels::{AttentionKernels, ElementwiseKernels};

    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    /// Deterministic LCG — mixed-magnitude signed values, no external deps.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((self.0 >> 33) as f32) / (u32::MAX >> 1) as f32;
            lo + bits * (hi - lo)
        }
    }

    fn signs_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..n).map(|_| if lcg.next_f32(-1.0, 1.0) >= 0.0 { 1.0 } else { -1.0 }).collect()
    }

    /// K1 — fused rmsnorm+sign+FWHT+quantize must be byte-identical to the
    /// split production path (launch_rmsnorm → memcpy → fwht → launch_quantize).
    #[test]
    fn k1_rmsnorm_rotate_quantize_bitexact() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let ew = ElementwiseKernels::new(ctx.clone()).expect("compile elementwise");
        const EPS: f32 = 1e-5;

        for &(dim, hblock) in &[(5120usize, 1024usize), (6144, 1024), (5120, 512)] {
            let mut lcg = Lcg(0x980 + dim as u64);
            let x: Vec<f32> = (0..dim).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
            let gamma: Vec<f32> = (0..dim).map(|_| lcg.next_f32(0.5, 1.5)).collect();
            let signs = signs_vec(dim, 0x5117 + dim as u64);
            let ablocks = dim / 16;

            // Split reference.
            let x_a = stream.clone_htod(&x).unwrap();
            let gamma_a = stream.clone_htod(&gamma).unwrap();
            let norm_x = stream.alloc_zeros::<f32>(dim).unwrap();
            let mut scratch = stream.alloc_zeros::<f32>(dim).unwrap();
            let signs_a = stream.clone_htod(&signs).unwrap();
            let ref_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
            let ref_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
            ew.launch_rmsnorm(&stream, &x_a, &gamma_a, &norm_x, dim, EPS).unwrap();
            stream.memcpy_dtod(&norm_x, &mut scratch).unwrap();
            rot.fwht_rotate_forward(&stream, &scratch, &signs_a, dim, hblock).unwrap();
            ew.launch_quantize(&stream, &scratch, &ref_i8, &ref_as, dim).unwrap();

            // Fused.
            let x_b = stream.clone_htod(&x).unwrap();
            let gamma_b = stream.clone_htod(&gamma).unwrap();
            let signs_b = stream.clone_htod(&signs).unwrap();
            let fus_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
            let fus_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
            rot.rmsnorm_rotate_quantize(
                &stream, &x_b, &gamma_b, &signs_b, &fus_i8, &fus_as, dim, EPS, hblock,
            )
            .unwrap();
            stream.synchronize().unwrap();

            let mut a_i8 = vec![0i8; dim];
            let mut b_i8 = vec![0i8; dim];
            let mut a_as = vec![0f32; ablocks];
            let mut b_as = vec![0f32; ablocks];
            stream.memcpy_dtoh(&ref_i8, &mut a_i8).unwrap();
            stream.memcpy_dtoh(&fus_i8, &mut b_i8).unwrap();
            stream.memcpy_dtoh(&ref_as, &mut a_as).unwrap();
            stream.memcpy_dtoh(&fus_as, &mut b_as).unwrap();
            let i8_mm = a_i8.iter().zip(&b_i8).filter(|(p, q)| p != q).count();
            let as_mm = a_as.iter().zip(&b_as).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            eprintln!("[k1] dim={dim} hblock={hblock}: {i8_mm} i8 mismatches, {as_mm} ascale bit mismatches");
            assert_eq!(i8_mm, 0, "dim {dim} hblock {hblock}: fused K1 i8 differs");
            assert_eq!(as_mm, 0, "dim {dim} hblock {hblock}: fused K1 ascale differs");
        }
    }

    /// K2 — the heterogeneous GDN-input launch must be byte-identical to the
    /// split path INCLUDING the inline-normalized dense a/b GEMVs.
    #[test]
    fn k2_gdn_input_bitexact() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let ew = ElementwiseKernels::new(ctx.clone()).expect("compile elementwise");
        const EPS: f32 = 1e-5;
        let (dim, hblock, rows) = (5120usize, 1024usize, 48usize);

        let mut lcg = Lcg(0xAB12);
        let x: Vec<f32> = (0..dim).map(|_| lcg.next_f32(-1.5, 1.5)).collect();
        let gamma: Vec<f32> = (0..dim).map(|_| lcg.next_f32(0.6, 1.4)).collect();
        let wa: Vec<f32> = (0..rows * dim).map(|_| lcg.next_f32(-0.05, 0.05)).collect();
        let wb: Vec<f32> = (0..rows * dim).map(|_| lcg.next_f32(-0.05, 0.05)).collect();
        let signs = signs_vec(dim, 0xC0DE);
        let ablocks = dim / 16;

        // Split reference.
        let x_a = stream.clone_htod(&x).unwrap();
        let gamma_a = stream.clone_htod(&gamma).unwrap();
        let wa_d = stream.clone_htod(&wa).unwrap();
        let wb_d = stream.clone_htod(&wb).unwrap();
        let norm_x = stream.alloc_zeros::<f32>(dim).unwrap();
        let mut scratch = stream.alloc_zeros::<f32>(dim).unwrap();
        let signs_a = stream.clone_htod(&signs).unwrap();
        let ref_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
        let ref_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
        let ref_a = stream.alloc_zeros::<f32>(rows).unwrap();
        let ref_b = stream.alloc_zeros::<f32>(rows).unwrap();
        ew.launch_rmsnorm(&stream, &x_a, &gamma_a, &norm_x, dim, EPS).unwrap();
        stream.memcpy_dtod(&norm_x, &mut scratch).unwrap();
        rot.fwht_rotate_forward(&stream, &scratch, &signs_a, dim, hblock).unwrap();
        ew.launch_quantize(&stream, &scratch, &ref_i8, &ref_as, dim).unwrap();
        rot.gemv_dense(&stream, &wa_d, &norm_x, &ref_a, rows, dim).unwrap();
        rot.gemv_dense(&stream, &wb_d, &norm_x, &ref_b, rows, dim).unwrap();

        // Fused.
        let x_b = stream.clone_htod(&x).unwrap();
        let gamma_b = stream.clone_htod(&gamma).unwrap();
        let signs_b = stream.clone_htod(&signs).unwrap();
        let fus_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
        let fus_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
        let fus_a = stream.alloc_zeros::<f32>(rows).unwrap();
        let fus_b = stream.alloc_zeros::<f32>(rows).unwrap();
        rot.gdn_input_fused(
            &stream, &x_b, &gamma_b, &signs_b, &wa_d, &wb_d,
            &fus_a, &fus_b, rows, &fus_i8, &fus_as, dim, EPS, hblock,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut a_i8 = vec![0i8; dim];
        let mut b_i8 = vec![0i8; dim];
        let mut a_as = vec![0f32; ablocks];
        let mut b_as = vec![0f32; ablocks];
        let mut a_raw = vec![0f32; rows];
        let mut b_raw = vec![0f32; rows];
        let mut f_a = vec![0f32; rows];
        let mut f_b = vec![0f32; rows];
        stream.memcpy_dtoh(&ref_i8, &mut a_i8).unwrap();
        stream.memcpy_dtoh(&fus_i8, &mut b_i8).unwrap();
        stream.memcpy_dtoh(&ref_as, &mut a_as).unwrap();
        stream.memcpy_dtoh(&fus_as, &mut b_as).unwrap();
        stream.memcpy_dtoh(&ref_a, &mut a_raw).unwrap();
        stream.memcpy_dtoh(&ref_b, &mut b_raw).unwrap();
        stream.memcpy_dtoh(&fus_a, &mut f_a).unwrap();
        stream.memcpy_dtoh(&fus_b, &mut f_b).unwrap();
        let i8_mm = a_i8.iter().zip(&b_i8).filter(|(p, q)| p != q).count();
        let as_mm = a_as.iter().zip(&b_as).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let a_mm = a_raw.iter().zip(&f_a).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let b_mm = b_raw.iter().zip(&f_b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        eprintln!("[k2] dim={dim}: {i8_mm} i8, {as_mm} ascale, {a_mm}+{b_mm} a/b bit mismatches");
        assert_eq!(i8_mm + as_mm + a_mm + b_mm, 0, "fused K2 differs from the split GDN input");
    }

    /// K3 — the fused GDN output chain must be byte-identical to the split
    /// path (rmsnorm_batched → gate_silu → memcpy → permute → fwht → quantize),
    /// for both the grouped geometry and the rep=1 identity.
    #[test]
    fn k3_gdn_out_bitexact() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let ew = ElementwiseKernels::new(ctx.clone()).expect("compile elementwise");
        let at = AttentionKernels::new(ctx).expect("compile attention");
        const EPS: f32 = 1e-5;
        let head_dim = 128usize;
        let hblock = 1024usize;

        for &n_heads in &[48usize, 32usize] {
            // Bonsai-2 grouped shape (48 heads) + a second geometry.
            let n_k = if n_heads == 48 { 16 } else { 32 };
            let rep = n_heads / n_k;
            let v_dim = n_heads * head_dim;
            let mut lcg = Lcg(0x600D + n_heads as u64);
            let x: Vec<f32> = (0..v_dim).map(|_| lcg.next_f32(-1.2, 1.2)).collect();
            let z: Vec<f32> = (0..v_dim).map(|_| lcg.next_f32(-0.8, 0.8)).collect();
            let gamma: Vec<f32> = (0..head_dim).map(|_| lcg.next_f32(0.7, 1.3)).collect();
            let signs = signs_vec(v_dim, 0x31337);
            let ablocks = v_dim / 16;

            // Split reference (in-place on recurrent_out, the production shape).
            let x_a = stream.clone_htod(&x).unwrap();
            let z_a = stream.clone_htod(&z).unwrap();
            let gamma_a = stream.clone_htod(&gamma).unwrap();
            let signs_a = stream.clone_htod(&signs).unwrap();
            let mut ptmp = stream.alloc_zeros::<f32>(v_dim).unwrap();
            let ref_i8 = stream.alloc_zeros::<i8>(v_dim).unwrap();
            let ref_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
            at.launch_rmsnorm_batched(&stream, &x_a, &gamma_a, &x_a, n_heads, head_dim, EPS).unwrap();
            rot.gate_silu(&stream, &x_a, &z_a, v_dim).unwrap();
            stream.memcpy_dtod(&x_a, &mut ptmp).unwrap();
            rot.gdn_v_permute(&stream, &ptmp, &x_a, v_dim, n_heads, n_k).unwrap();
            rot.fwht_rotate_forward(&stream, &x_a, &signs_a, v_dim, hblock).unwrap();
            ew.launch_quantize(&stream, &x_a, &ref_i8, &ref_as, v_dim).unwrap();

            // Fused.
            let x_b = stream.clone_htod(&x).unwrap();
            let z_b = stream.clone_htod(&z).unwrap();
            let gamma_b = stream.clone_htod(&gamma).unwrap();
            let signs_b = stream.clone_htod(&signs).unwrap();
            let fus_i8 = stream.alloc_zeros::<i8>(v_dim).unwrap();
            let fus_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
            rot.gdn_out_fused(
                &stream, &x_b, &z_b, &gamma_b, &signs_b, &fus_i8, &fus_as,
                v_dim, head_dim, n_k, rep, EPS, hblock,
            )
            .unwrap();
            stream.synchronize().unwrap();

            let mut a_i8 = vec![0i8; v_dim];
            let mut b_i8 = vec![0i8; v_dim];
            let mut a_as = vec![0f32; ablocks];
            let mut b_as = vec![0f32; ablocks];
            stream.memcpy_dtoh(&ref_i8, &mut a_i8).unwrap();
            stream.memcpy_dtoh(&fus_i8, &mut b_i8).unwrap();
            stream.memcpy_dtoh(&ref_as, &mut a_as).unwrap();
            stream.memcpy_dtoh(&fus_as, &mut b_as).unwrap();
            let i8_mm = a_i8.iter().zip(&b_i8).filter(|(p, q)| p != q).count();
            let as_mm = a_as.iter().zip(&b_as).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            eprintln!("[k3] heads={n_heads} nk={n_k} rep={rep}: {i8_mm} i8, {as_mm} ascale bit mismatches");
            assert_eq!(i8_mm + as_mm, 0, "fused K3 differs from the split GDN output chain");
        }
    }

    /// K4 — fused sigmoid-gate+sign+FWHT+quantize vs the split path.
    #[test]
    fn k4_gate_sigmoid_bitexact() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let ew = ElementwiseKernels::new(ctx.clone()).expect("compile elementwise");
        let (dim, hblock) = (6144usize, 1024usize);
        let mut lcg = Lcg(0x404);
        let x: Vec<f32> = (0..dim).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let gate: Vec<f32> = (0..dim).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs = signs_vec(dim, 0x506);
        let ablocks = dim / 16;

        let x_a = stream.clone_htod(&x).unwrap();
        let g_a = stream.clone_htod(&gate).unwrap();
        let signs_a = stream.clone_htod(&signs).unwrap();
        let ref_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
        let ref_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
        rot.gate_sigmoid(&stream, &x_a, &g_a, dim).unwrap();
        rot.fwht_rotate_forward(&stream, &x_a, &signs_a, dim, hblock).unwrap();
        ew.launch_quantize(&stream, &x_a, &ref_i8, &ref_as, dim).unwrap();

        let x_b = stream.clone_htod(&x).unwrap();
        let g_b = stream.clone_htod(&gate).unwrap();
        let signs_b = stream.clone_htod(&signs).unwrap();
        let fus_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
        let fus_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
        rot.gate_sigmoid_rotate_quantize(
            &stream, &x_b, &g_b, &signs_b, &fus_i8, &fus_as, dim, hblock,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut a_i8 = vec![0i8; dim];
        let mut b_i8 = vec![0i8; dim];
        let mut a_as = vec![0f32; ablocks];
        let mut b_as = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8, &mut a_i8).unwrap();
        stream.memcpy_dtoh(&fus_i8, &mut b_i8).unwrap();
        stream.memcpy_dtoh(&ref_as, &mut a_as).unwrap();
        stream.memcpy_dtoh(&fus_as, &mut b_as).unwrap();
        let i8_mm = a_i8.iter().zip(&b_i8).filter(|(p, q)| p != q).count();
        let as_mm = a_as.iter().zip(&b_as).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        eprintln!("[k4] dim={dim}: {i8_mm} i8, {as_mm} ascale bit mismatches");
        assert_eq!(i8_mm + as_mm, 0, "fused K4 differs from the split attention-out path");
    }

    /// K5 — fused SwiGLU+sign+FWHT+quantize vs the split path, at the real
    /// FFN-down width (17408).
    #[test]
    fn k5_swiglu_bitexact() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let ew = ElementwiseKernels::new(ctx.clone()).expect("compile elementwise");
        let (dim, hblock) = (17408usize, 1024usize);
        let mut lcg = Lcg(0x711);
        let gate: Vec<f32> = (0..dim).map(|_| lcg.next_f32(-3.0, 3.0)).collect();
        let up: Vec<f32> = (0..dim).map(|_| lcg.next_f32(-2.0, 2.0)).collect();
        let signs = signs_vec(dim, 0x815);
        let ablocks = dim / 16;

        let g_a = stream.clone_htod(&gate).unwrap();
        let u_a = stream.clone_htod(&up).unwrap();
        let hid = stream.alloc_zeros::<f32>(dim).unwrap();
        let signs_a = stream.clone_htod(&signs).unwrap();
        let ref_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
        let ref_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
        ew.launch_swiglu(&stream, &g_a, &u_a, &hid, dim).unwrap();
        rot.fwht_rotate_forward(&stream, &hid, &signs_a, dim, hblock).unwrap();
        ew.launch_quantize(&stream, &hid, &ref_i8, &ref_as, dim).unwrap();

        let g_b = stream.clone_htod(&gate).unwrap();
        let u_b = stream.clone_htod(&up).unwrap();
        let signs_b = stream.clone_htod(&signs).unwrap();
        let fus_i8 = stream.alloc_zeros::<i8>(dim).unwrap();
        let fus_as = stream.alloc_zeros::<f32>(ablocks).unwrap();
        rot.swiglu_rotate_quantize(
            &stream, &g_b, &u_b, &signs_b, &fus_i8, &fus_as, dim, hblock,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let mut a_i8 = vec![0i8; dim];
        let mut b_i8 = vec![0i8; dim];
        let mut a_as = vec![0f32; ablocks];
        let mut b_as = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8, &mut a_i8).unwrap();
        stream.memcpy_dtoh(&fus_i8, &mut b_i8).unwrap();
        stream.memcpy_dtoh(&ref_as, &mut a_as).unwrap();
        stream.memcpy_dtoh(&fus_as, &mut b_as).unwrap();
        let i8_mm = a_i8.iter().zip(&b_i8).filter(|(p, q)| p != q).count();
        let as_mm = a_as.iter().zip(&b_as).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        eprintln!("[k5] dim={dim}: {i8_mm} i8, {as_mm} ascale bit mismatches");
        assert_eq!(i8_mm + as_mm, 0, "fused K5 differs from the split FFN-down path");
    }

    /// Batched fwht forward/inverse — byte-identical to the single-row
    /// kernels applied per row (p=3 rows of width 6144, hblock 1024).
    #[test]
    fn batched_fwht_matches_per_row() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let (p, width, hblock) = (3usize, 6144usize, 1024usize);
        let mut lcg = Lcg(0xBADC0DE);
        let x: Vec<f32> = (0..p * width).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs = signs_vec(width, 0xF00D);

        // Reference: per-row single-row kernels (each row uploaded on its own
        // slice — no subrange API games).
        let signs_a = stream.clone_htod(&signs).unwrap();
        let mut ref_fwd = Vec::with_capacity(p * width);
        for r in 0..p {
            let row_dev = stream.clone_htod(&x[r * width..(r + 1) * width]).unwrap();
            rot.fwht_rotate_forward(&stream, &row_dev, &signs_a, width, hblock).unwrap();
            let mut row = vec![0f32; width];
            stream.memcpy_dtoh(&row_dev, &mut row).unwrap();
            ref_fwd.extend_from_slice(&row);
        }

        // Fused batched.
        let x_b = stream.clone_htod(&x).unwrap();
        rot.fwht_rotate_forward_batched(&stream, &x_b, &signs_a, p, width, hblock).unwrap();
        let mut got_fwd = vec![0f32; p * width];
        stream.memcpy_dtoh(&x_b, &mut got_fwd).unwrap();
        let fwd_mm = ref_fwd.iter().zip(&got_fwd).filter(|(a, b)| a.to_bits() != b.to_bits()).count();

        // Inverse: reference = inverse applied per row to the ORIGINAL x.
        let mut inv_rows = Vec::with_capacity(p * width);
        for r in 0..p {
            let row_dev = stream.clone_htod(&x[r * width..(r + 1) * width]).unwrap();
            rot.fwht_rotate_inverse(&stream, &row_dev, &signs_a, width, hblock).unwrap();
            let mut row = vec![0f32; width];
            stream.memcpy_dtoh(&row_dev, &mut row).unwrap();
            inv_rows.extend_from_slice(&row);
        }
        let x_d = stream.clone_htod(&x).unwrap();
        rot.fwht_rotate_inverse_batched(&stream, &x_d, &signs_a, p, width, hblock).unwrap();
        let mut got_inv = vec![0f32; p * width];
        stream.memcpy_dtoh(&x_d, &mut got_inv).unwrap();
        let inv_mm = inv_rows.iter().zip(&got_inv).filter(|(a, b)| a.to_bits() != b.to_bits()).count();

        eprintln!("[batched_fwht] p={p} w={width}: fwd {fwd_mm}, inv {inv_mm} bit mismatches");
        assert_eq!(fwd_mm + inv_mm, 0, "batched fwht differs from per-row");
    }

    /// Batched gdn_v permute — byte-identical to the single-row kernel applied
    /// per row (p=3, v_dim 6144, hd 128, n_k 16).
    #[test]
    fn batched_permute_matches_per_row() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let (p, v_dim, hd, n_k) = (3usize, 6144usize, 128usize, 16usize);
        let mut lcg = Lcg(0x600D + 7);
        let x: Vec<f32> = (0..p * v_dim).map(|_| lcg.next_f32(-1.0, 1.0)).collect();

        // Reference: per-row permute (memcpy to tmp, gather back).
        let mut ref_rows = Vec::new();
        for r in 0..p {
            let row = &x[r * v_dim..(r + 1) * v_dim];
            let slice = stream.clone_htod(row).unwrap();
            let tmp = stream.clone_htod(row).unwrap();
            rot.gdn_v_permute(&stream, &tmp, &slice, v_dim, v_dim / hd, n_k).unwrap();
            let mut out = vec![0f32; v_dim];
            stream.memcpy_dtoh(&slice, &mut out).unwrap();
            ref_rows.extend_from_slice(&out);
        }

        // Batched.
        let x_b = stream.clone_htod(&x).unwrap();
        let tmp_b = stream.clone_htod(&x).unwrap();
        rot.gdn_v_permute_batched(&stream, &tmp_b, &x_b, p, v_dim, hd, n_k).unwrap();
        let mut got = vec![0f32; p * v_dim];
        stream.memcpy_dtoh(&x_b, &mut got).unwrap();
        let mm = ref_rows.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("[batched_permute] p={p}: {mm} bit mismatches");
        assert_eq!(mm, 0, "batched permute differs from per-row");
    }

    /// Batched dense a/b GEMM vs a CPU f64 reference (fp32 FMA order differs
    /// from CPU — tolerance check; determinism is the pin-relevant property).
    #[test]
    fn batched_dense_gemm_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        // Odd p exercises the tail tile (p = 130 = 2*64 + 2).
        let (p, rows, n) = (130usize, 48usize, 512usize);
        let mut lcg = Lcg(0xD15EA5E);
        let x: Vec<f32> = (0..p * n).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let wa: Vec<f32> = (0..rows * n).map(|_| lcg.next_f32(-0.05, 0.05)).collect();
        let wb: Vec<f32> = (0..rows * n).map(|_| lcg.next_f32(-0.05, 0.05)).collect();

        let x_d = stream.clone_htod(&x).unwrap();
        let wa_d = stream.clone_htod(&wa).unwrap();
        let wb_d = stream.clone_htod(&wb).unwrap();
        let out_a = stream.alloc_zeros::<f32>(p * rows).unwrap();
        let out_b = stream.alloc_zeros::<f32>(p * rows).unwrap();
        rot.gemm_dense_ab_batched(&stream, &x_d, &wa_d, &wb_d, &out_a, &out_b, p, rows, n)
            .unwrap();
        let mut got_a = vec![0f32; p * rows];
        let mut got_b = vec![0f32; p * rows];
        stream.memcpy_dtoh(&out_a, &mut got_a).unwrap();
        stream.memcpy_dtoh(&out_b, &mut got_b).unwrap();

        let check = |got: &[f32], w: &[f32], name: &str| {
            // Condition-scaled bound: |got-ref| <= tol * Σ|x·w| (relative error
            // explodes at near-zero dots — cancellation, not a kernel bug).
            let mut worst_rel = 0f64;
            let mut worst_abs = 0f64;
            for t in 0..p {
                for r in 0..rows {
                    let mut acc = 0f64;
                    let mut abs_sum = 0f64;
                    for k in 0..n {
                        let term = x[t * n + k] as f64 * w[r * n + k] as f64;
                        acc += term;
                        abs_sum += term.abs();
                    }
                    let got_v = got[t * rows + r] as f64;
                    let diff = (got_v - acc).abs();
                    worst_abs = worst_abs.max(diff);
                    worst_rel = worst_rel.max(diff / abs_sum.max(1e-12));
                }
            }
            eprintln!("[batched_gemm:{name}] worst abs {worst_abs:.3e}, cond-scaled {worst_rel:.3e}");
            worst_rel
        };
        let ea = check(&got_a, &wa, "a");
        let eb = check(&got_b, &wb, "b");
        eprintln!("[batched_gemm] p={p} rows={rows} n={n}: cond-scaled err a={ea:.3e} b={eb:.3e}");
        assert!(ea < 1e-4 && eb < 1e-4, "dense ab GEMM outside fp32 tolerance: {ea:.3e}/{eb:.3e}");

        // Determinism: a second run must be bit-identical (the pin property).
        rot.gemm_dense_ab_batched(&stream, &x_d, &wa_d, &wb_d, &out_a, &out_b, p, rows, n)
            .unwrap();
        let mut again_a = vec![0f32; p * rows];
        stream.memcpy_dtoh(&out_a, &mut again_a).unwrap();
        let dmm = got_a.iter().zip(&again_a).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(dmm, 0, "dense ab GEMM nondeterministic");
    }

    /// Issue 980 T4-ALT — the copy-rotate staging twin must be BIT-IDENTICAL
    /// to memcpy + the in-place forward rotate (the whole-prefill sites swap
    /// that pair for this single-pass kernel; identical bytes is the
    /// contract, not tolerance).
    #[test]
    fn batched_copy_rotate_bitexact_vs_memcpy_plus_inplace() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let (rows, width, block) = (130usize, 4096usize, 1024usize);
        let mut lcg = Lcg(0xC0FFEE5);
        let src: Vec<f32> = (0..rows * width).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs: Vec<f32> = (0..width).map(|i| if i % 3 == 0 { -1.0f32 } else { 1.0f32 }).collect();

        let src_d = stream.clone_htod(&src).unwrap();
        let signs_d = stream.clone_htod(&signs).unwrap();
        // Reference: memcpy + in-place forward rotate.
        let mut ref_d = stream.alloc_zeros::<f32>(rows * width).unwrap();
        stream.memcpy_dtod(&src_d, &mut ref_d).unwrap();
        rot.fwht_rotate_forward_batched(&stream, &ref_d, &signs_d, rows, width, block)
            .unwrap();
        // Candidate: the one-pass copy-rotate.
        let dst_d = stream.alloc_zeros::<f32>(rows * width).unwrap();
        rot.fwht_rotate_copy_batched(&stream, &src_d, &dst_d, &signs_d, rows, width, block)
            .unwrap();

        let mut got_ref = vec![0f32; rows * width];
        let mut got_dst = vec![0f32; rows * width];
        stream.memcpy_dtoh(&ref_d, &mut got_ref).unwrap();
        stream.memcpy_dtoh(&dst_d, &mut got_dst).unwrap();
        let diff = got_ref
            .iter()
            .zip(&got_dst)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(diff, 0, "copy-rotate differs from memcpy+in-place at {diff} of {} elements", rows * width);
        // And the source must be untouched (the PRIMAL normx stays primal).
        let mut got_src = vec![0f32; rows * width];
        stream.memcpy_dtoh(&src_d, &mut got_src).unwrap();
        assert_eq!(
            got_src.iter().zip(&src).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
            0,
            "copy-rotate mutated its source"
        );
    }

    /// Issue 980 C0.5 — the fused rotate(+permute)+quantize must be
    /// BYTE-IDENTICAL to the unfused chain it replaces (copy-rotate into a
    /// staging buffer + the mma module's q8 quantize reading it): same q8
    /// words + same row scales. The folded prefill pins ride this contract.
    #[cfg(feature = "prefill_q8_act")]
    #[test]
    fn fused_qrot_bitexact_vs_unfused_chain() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        use crate::gemm_ternary_i8_mma_cuda_raw::{GemmTernaryI8MmaCuda, QuantDiv};
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let mma = GemmTernaryI8MmaCuda::new(ctx.clone()).expect("compile mma");
        let (p, n, block) = (5usize, 4096usize, 1024usize);
        let mut lcg = Lcg(0x900DC0DE);
        let x: Vec<f32> = (0..p * n).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { -1.0f32 } else { 1.0f32 }).collect();
        let x_d = stream.clone_htod(&x).unwrap();
        let signs_d = stream.clone_htod(&signs).unwrap();
        let scratch = mma.alloc_scratch(&stream, n, p).unwrap();

        // Reference: copy-rotate into staging + the q8 quantize (Full div).
        let stage = stream.alloc_zeros::<f32>(p * n).unwrap();
        rot.fwht_rotate_copy_batched(&stream, &x_d, &stage, &signs_d, p, n, block)
            .unwrap();
        mma.launch_quantize_q8_div(&stream, &stage, &scratch, n, p, QuantDiv::Full)
            .unwrap();
        let words = p * (n / 4);
        let mut words_ref = vec![0u32; words];
        let mut s_ref = vec![0f32; p];
        stream.memcpy_dtoh(&scratch.q_hi_w, &mut words_ref).unwrap();
        stream.memcpy_dtoh(&scratch.s_t, &mut s_ref).unwrap();

        // Candidate A: the fused rotate+quantize.
        rot.quantize_rotate_q8(
            &stream, &x_d, &signs_d, &scratch.q_hi_w, &scratch.s_t, n, p, block,
        )
        .unwrap();
        let mut words_a = vec![0u32; words];
        let mut s_a = vec![0f32; p];
        stream.memcpy_dtoh(&scratch.q_hi_w, &mut words_a).unwrap();
        stream.memcpy_dtoh(&scratch.s_t, &mut s_a).unwrap();
        assert_eq!(
            words_a.iter().zip(&words_ref).filter(|(a, b)| a != b).count(),
            0,
            "fused qrot q8 words differ from the unfused chain"
        );
        assert_eq!(
            s_a.iter().zip(&s_ref).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
            0,
            "fused qrot row scales differ from the unfused chain"
        );

        // Candidate B: the permute variant vs its unfused chain
        // (permute into tmp + rotate + quantize). hd/n_k chosen so rep=2.
        let (hd, n_k) = (128usize, 16usize); // heads = 4096/128 = 32, rep = 32/16 = 2
        let mut perm_ref = vec![0u32; words];
        let mut perm_s_ref = vec![0f32; p];
        {
            let tmp = stream.alloc_zeros::<f32>(p * n).unwrap();
            rot.gdn_v_permute_batched(&stream, &x_d, &tmp, p, n, hd, n_k)
                .unwrap();
            rot.fwht_rotate_forward_batched(&stream, &tmp, &signs_d, p, n, block)
                .unwrap();
            mma.launch_quantize_q8_div(&stream, &tmp, &scratch, n, p, QuantDiv::Full)
                .unwrap();
            stream.memcpy_dtoh(&scratch.q_hi_w, &mut perm_ref).unwrap();
            stream.memcpy_dtoh(&scratch.s_t, &mut perm_s_ref).unwrap();
            // Host oracle (2026-09-19 evidence): permute + sign + blockwise
            // FWHT in the kernel's exact op order — the device BLOCK chain
            // matches this element-for-element (0 diffs, the rung-2 warp
            // variant diverged 1–2 ULP in the row scale on this same data —
            // why its dispatch is disabled; see the launcher comment).
            let mut tmp_host = vec![0f32; p * n];
            stream.memcpy_dtoh(&tmp, &mut tmp_host).unwrap();
            let rep = (n / hd) / n_k;
            let mut host_rot = vec![0f32; p * n];
            for r in 0..p {
                for j in 0..n {
                    let head = j / hd;
                    let off = j % hd;
                    let nk = head / rep;
                    let rr = head % rep;
                    host_rot[r * n + j] = x[r * n + (rr * n_k + nk) * hd + off];
                }
                for b in 0..n / 1024 {
                    let coff = b * 1024;
                    for i in 0..1024 {
                        host_rot[r * n + coff + i] *= signs[coff + i];
                    }
                    let mut step = 2usize;
                    while step <= 1024 {
                        let half = step >> 1;
                        for ii in 0..512 {
                            let blk = (ii / half) * step;
                            let off2 = ii % half;
                            let i0 = r * n + coff + blk + off2;
                            let i1 = i0 + half;
                            let a = host_rot[i0];
                            let bv = host_rot[i1];
                            host_rot[i0] = (a + bv) * std::f32::consts::FRAC_1_SQRT_2;
                            host_rot[i1] = (a - bv) * std::f32::consts::FRAC_1_SQRT_2;
                        }
                        step <<= 1;
                    }
                }
            }
            let dev_host_diff = (0..p * n)
                .filter(|&i| tmp_host[i].to_bits() != host_rot[i].to_bits())
                .count();
            assert_eq!(
                dev_host_diff, 0,
                "the unfused device chain drifted from exact host FP arithmetic"
            );
        }
        rot.quantize_permute_rotate_q8(
            &stream, &x_d, &signs_d, &scratch.q_hi_w, &scratch.s_t, n, p, block, hd, n_k,
        )
        .unwrap();
        let mut words_b = vec![0u32; words];
        let mut s_b = vec![0f32; p];
        stream.memcpy_dtoh(&scratch.q_hi_w, &mut words_b).unwrap();
        stream.memcpy_dtoh(&scratch.s_t, &mut s_b).unwrap();
        assert_eq!(
            words_b.iter().zip(&perm_ref).filter(|(a, b)| a != b).count(),
            0,
            "fused permute+qrot q8 words differ from the unfused chain"
        );
        assert_eq!(
            s_b.iter().zip(&perm_s_ref).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
            0,
            "fused permute+qrot row scales differ from the unfused chain"
        );

        // The wide-row arm (12 chunks — above the warp variant's 8-chunk
        // budget even if it were enabled): byte-identical via the same
        // block kernel at a different width.
        let n2 = 12288usize;
        let x2: Vec<f32> = (0..p * n2).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs2: Vec<f32> = (0..n2).map(|i| if i % 7 == 0 { -1.0f32 } else { 1.0f32 }).collect();
        let x2_d = stream.clone_htod(&x2).unwrap();
        let signs2_d = stream.clone_htod(&signs2).unwrap();
        let scratch2 = mma.alloc_scratch(&stream, n2, p).unwrap();
        let stage2 = stream.alloc_zeros::<f32>(p * n2).unwrap();
        rot.fwht_rotate_copy_batched(&stream, &x2_d, &stage2, &signs2_d, p, n2, block)
            .unwrap();
        mma.launch_quantize_q8_div(&stream, &stage2, &scratch2, n2, p, QuantDiv::Full)
            .unwrap();
        let mut w2_ref = vec![0u32; p * (n2 / 4)];
        let mut s2_ref = vec![0f32; p];
        stream.memcpy_dtoh(&scratch2.q_hi_w, &mut w2_ref).unwrap();
        stream.memcpy_dtoh(&scratch2.s_t, &mut s2_ref).unwrap();
        rot.quantize_rotate_q8(
            &stream, &x2_d, &signs2_d, &scratch2.q_hi_w, &scratch2.s_t, n2, p, block,
        )
        .unwrap();
        let mut w2_got = vec![0u32; p * (n2 / 4)];
        let mut s2_got = vec![0f32; p];
        stream.memcpy_dtoh(&scratch2.q_hi_w, &mut w2_got).unwrap();
        stream.memcpy_dtoh(&scratch2.s_t, &mut s2_got).unwrap();
        assert_eq!(
            w2_got.iter().zip(&w2_ref).filter(|(a, b)| a != b).count(),
            0,
            "block-kernel qrot q8 words differ from the unfused chain"
        );
        assert_eq!(
            s2_got.iter().zip(&s2_ref).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
            0,
            "block-kernel qrot row scales differ from the unfused chain"
        );
    }

    /// Session instrument (C0.5 rung-2 reopen): dump the compiled PTX to a
    /// file for offline diffing of the block vs warp butterflies. Not a
    /// gate — run explicitly with `RIIR_ROTATION_PTX_DUMP=<path>`.
    #[test]
    #[ignore = "instrument: set RIIR_ROTATION_PTX_DUMP=<path> to dump PTX"]
    fn dump_rotation_ptx() {
        let Ok(path) = std::env::var("RIIR_ROTATION_PTX_DUMP") else {
            panic!("set RIIR_ROTATION_PTX_DUMP=<path>");
        };
        let ptx = cudarc::nvrtc::compile_ptx(ROTATION_CUDA_SRC).expect("compile");
        std::fs::write(&path, ptx.to_src()).expect("write");
        eprintln!("wrote {path}");
    }

    /// C0.5 rung-2 reopen — the warp butterfly vs the host oracle, STAGE BY
    /// STAGE: stage 0 = post gather+sign, 1..5 = shuffle distances 1..16,
    /// 6..10 = register distances 32..512. The host oracle is the exact-FP
    /// simulation the block chain was proven identical to
    /// (`fused_qrot_bitexact_vs_unfused_chain`), re-simulated with a
    /// snapshot after every stage. On divergence: per-stage diff counts +
    /// the first divergent element's address decomposition (chunk/k/lane)
    /// — the forensic report that localizes the ULP source for the disabled
    /// dispatch. When rung 2 is repaired this is its regression gate.
    #[test]
    fn qrot_warp_stage_isolation() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let (p, n, block, hd, n_k) = (5usize, 4096usize, 1024usize, 128usize, 16usize);
        let mut lcg = Lcg(0x900DC0DE);
        let x: Vec<f32> = (0..p * n).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { -1.0f32 } else { 1.0f32 }).collect();
        let x_d = stream.clone_htod(&x).unwrap();
        let signs_d = stream.clone_htod(&signs).unwrap();
        let dbg_d = stream.alloc_zeros::<f32>(p * 12 * n).unwrap();
        rot.warp_butterfly_stage_dump(&stream, &x_d, &signs_d, &dbg_d, n, p, block, hd, n_k)
            .unwrap();
        let mut got = vec![0f32; p * 12 * n];
        stream.memcpy_dtoh(&dbg_d, &mut got).unwrap();

        // Host oracle with per-stage snapshots — identical op order to the
        // kernels (gather -> sign -> butterfly distances ascending 1..512).
        // Layout mirrors the kernel: slot s at [row*12 + s], s in 0..10.
        let rep = (n / hd) / n_k;
        let nchunks = n / 1024;
        let mut host = vec![0f32; p * 12 * n];
        let c = std::f32::consts::FRAC_1_SQRT_2;
        for r in 0..p {
            for j in 0..n {
                let head = j / hd;
                let off = j % hd;
                let nk = head / rep;
                let rr = head % rep;
                let v = x[r * n + (rr * n_k + nk) * hd + off] * signs[j];
                host[(r * 12) * n + j] = v;
            }
            let mut stage = 1usize;
            let mut dist = 1usize;
            while dist < 1024 {
                for b in 0..nchunks {
                    let coff = b * 1024;
                    for blk in 0..(1024 / (2 * dist)) {
                        let base0 = coff + blk * 2 * dist;
                        for o in 0..dist {
                            let i0 = (r * 12 + stage) * n + base0 + o;
                            let i1 = i0 + dist;
                            let a = host[i0 - n]; // previous stage's value
                            let bv = host[i1 - n];
                            host[i0] = (a + bv) * c;
                            host[i1] = (a - bv) * c;
                        }
                    }
                }
                stage += 1;
                dist <<= 1;
            }
        }

        let stage_desc = |s: usize| match s {
            0 => "post gather+sign".to_string(),
            1..=5 => format!("shuffle dist {}", 1usize << (s - 1)),
            6..=10 => format!("register dist {}", 32usize << (s - 6)),
            _ => unreachable!(),
        };
        let mut first: Option<(usize, usize, u32, u32, i64)> = None;
        let mut counts = [0usize; 11];
        let mut stage0_report: Vec<(usize, usize, u32, u32)> = Vec::new();
        for r in 0..p {
            for (s, count) in counts.iter_mut().enumerate() {
                for b in 0..nchunks {
                    let coff = b * 1024;
                    for e in coff..coff + 1024 {
                        let i = (r * 12 + s) * n + e;
                        let (hg, hh) = (got[i].to_bits(), host[i].to_bits());
                        if hg != hh {
                            *count += 1;
                            if s == 0 && stage0_report.len() < 8 {
                                stage0_report.push((r, e, hg, hh));
                            }
                            if first.is_none() {
                                let ulp = (hg as i32).wrapping_sub(hh as i32).abs() as i64;
                                first = Some((r, s, hg, hh, ulp));
                            }
                        }
                    }
                }
            }
        }
        for (r, e, hg, hh) in &stage0_report {
            let (chunk, within) = (e / 1024, e % 1024);
            let (k, lane) = (within / 32, within % 32);
            eprintln!(
                "stage-0 diff: row {r} e={e} (chunk {chunk} k {k} lane {lane}) got=0x{hg:08X} ({}) host=0x{hh:08X} ({})",
                f32::from_bits(*hg), f32::from_bits(*hh)
            );
        }
        eprintln!("warp-vs-host per-stage diff counts (of {} elems/stage):", p * n);
        for (s, count) in counts.iter().enumerate() {
            eprintln!("  stage {s:2} ({}): {}", stage_desc(s), count);
        }
        // Session-2 tail probe: per-chunk maxima (slot 11) vs host per-chunk
        // maxima computed from the (probe-exact) stage-10 values.
        let mut chunk_bad = 0usize;
        for r in 0..p {
            for b in 0..nchunks {
                let host_chunk_max = (0..1024)
                    .map(|o| host[(r * 12 + 10) * n + b * 1024 + o].abs())
                    .fold(0f32, |m, a| if a > m { a } else { m });
                let got_chunk_max = got[(r * 12 + 11) * n + b];
                if got_chunk_max.to_bits() != host_chunk_max.to_bits() {
                    chunk_bad += 1;
                    eprintln!(
                        "  CHUNK-MAX row {r} chunk {b}: probe={:016X} host={:016X} ({} ulp)",
                        got_chunk_max.to_bits(),
                        host_chunk_max.to_bits(),
                        (got_chunk_max.to_bits() as i32).wrapping_sub(host_chunk_max.to_bits() as i32)
                    );
                }
            }
        }
        eprintln!("chunk-max diffs: {chunk_bad}");
        if let Some((r, s, hg, hh, ulp)) = first {
            panic!(
                "warp butterfly diverged from the host oracle: row {r} stage {s} ({}) \
                 got=0x{hg:08X} host=0x{hh:08X} ulp={ulp} — first of {} divergent at that stage",
                stage_desc(s),
                counts[s]
            );
        }
    }

    /// C0.5 rung-2 forensics (session 2): the production warp kernel's row
    /// scales diverge 1–2 ULP on this data (words match) while the stage
    /// probe is host-exact — recover WHICH rows diverge and whether the
    /// production absmax `m` itself is wrong (vs the host oracle's m from
    /// the stage-10 snapshot) or the div/write is. Prints per-row bits.
    #[cfg(feature = "prefill_q8_act")]
    #[test]
    fn qrot_warp_scale_forensics() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let rot = RotationKernels::new(&ctx).expect("compile rotation");
        let (p, n, block, hd, n_k) = (5usize, 4096usize, 1024usize, 128usize, 16usize);
        let mut lcg = Lcg(0x900DC0DE);
        let x: Vec<f32> = (0..p * n).map(|_| lcg.next_f32(-1.0, 1.0)).collect();
        let signs: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { -1.0f32 } else { 1.0f32 }).collect();
        let x_d = stream.clone_htod(&x).unwrap();
        let signs_d = stream.clone_htod(&signs).unwrap();

        // Ground-truth per-row absmax: host oracle over the probe's stage-10
        // values (the probe is host-exact at every stage — proven by
        // `qrot_warp_stage_isolation` on this same data).
        let rep = (n / hd) / n_k;
        let c = std::f32::consts::FRAC_1_SQRT_2;
        let mut m_host = vec![0f32; p];
        {
            let mut h = vec![0f32; p * n];
            for r in 0..p {
                for j in 0..n {
                    let head = j / hd;
                    let off = j % hd;
                    let nk = head / rep;
                    let rr = head % rep;
                    h[r * n + j] = x[r * n + (rr * n_k + nk) * hd + off] * signs[j];
                }
                for b in 0..n / 1024 {
                    let coff = b * 1024;
                    let mut dist = 1usize;
                    while dist < 1024 {
                        for blk in 0..(1024 / (2 * dist)) {
                            let base0 = r * n + coff + blk * 2 * dist;
                            for o in 0..dist {
                                let a = h[base0 + o];
                                let bv = h[base0 + o + dist];
                                h[base0 + o] = (a + bv) * c;
                                h[base0 + o + dist] = (a - bv) * c;
                            }
                        }
                        dist <<= 1;
                    }
                }
                m_host[r] = h[r * n..(r + 1) * n].iter().fold(0f32, |m, v| {
                    let a = if *v < 0.0 { -*v } else { *v };
                    if a > m { a } else { m }
                });
            }
        }

        // Production warp kernel scales (the dispatch is armed in this
        // session) vs the block qrot scales — same inputs.
        let q_w = stream.alloc_zeros::<u32>(p * (n / 4)).unwrap();
        let s_t = stream.alloc_zeros::<f32>(p).unwrap();
        rot.quantize_permute_rotate_q8(&stream, &x_d, &signs_d, &q_w, &s_t, n, p, block, hd, n_k)
            .unwrap();
        let mut s_warp = vec![0f32; p];
        stream.memcpy_dtoh(&s_t, &mut s_warp).unwrap();

        // BLOCK kernel head-to-head (direct launch — bypasses the armed
        // warp dispatch so both kernels run on identical inputs).
        let q_w3 = stream.alloc_zeros::<u32>(p * (n / 4)).unwrap();
        let s_t3 = stream.alloc_zeros::<f32>(p).unwrap();
        {
            let n_i = n as i32;
            let hb_i = block as i32;
            let hd_i = hd as i32;
            let nk_i = n_k as i32;
            let smem = ((n + QROT_REDUCTION) * core::mem::size_of::<f32>()) as u32;
            unsafe {
                stream
                    .launch_builder(&rot.quantize_permute_rot_q8)
                    .arg(&x_d)
                    .arg(&signs_d)
                    .arg(&q_w3)
                    .arg(&s_t3)
                    .arg(&n_i)
                    .arg(&hb_i)
                    .arg(&hd_i)
                    .arg(&nk_i)
                    .launch(LaunchConfig {
                        grid_dim: (p as u32, 1, 1),
                        block_dim: (QROT_THREADS, 1, 1),
                        shared_mem_bytes: smem,
                    })
                    .unwrap();
            }
        }
        let mut s_blockp = vec![0f32; p];
        stream.memcpy_dtoh(&s_t3, &mut s_blockp).unwrap();

        // The unfused chain (mma quantize on the host-exact staging).
        let tmp = stream.alloc_zeros::<f32>(p * n).unwrap();
        rot.gdn_v_permute_batched(&stream, &x_d, &tmp, p, n, hd, n_k).unwrap();
        rot.fwht_rotate_forward_batched(&stream, &tmp, &signs_d, p, n, block).unwrap();
        let mut tmp_host = vec![0f32; p * n];
        stream.memcpy_dtoh(&tmp, &mut tmp_host).unwrap();
        // host-exact staging ground truth for m
        let mut m_true = vec![0f32; p];
        for r in 0..p {
            m_true[r] = tmp_host[r * n..(r + 1) * n]
                .iter()
                .fold(0f32, |m, v| { let a = v.abs(); if a > m { a } else { m } });
        }
        let mut s_mma = vec![0f32; p];
        {
            use crate::gemm_ternary_i8_mma_cuda_raw::{GemmTernaryI8MmaCuda, QuantDiv};
            let mma = GemmTernaryI8MmaCuda::new(ctx.clone()).expect("compile mma");
            let scratch = mma.alloc_scratch(&stream, n, p).unwrap();
            mma.launch_quantize_q8_div(&stream, &tmp, &scratch, n, p, QuantDiv::Full).unwrap();
            stream.memcpy_dtoh(&scratch.s_t, &mut s_mma).unwrap();
        }

        let ulp = |a: f32, b: f32| (a.to_bits() as i32).wrapping_sub(b.to_bits() as i32);
        // Recover the production warp kernel's ACTUAL m by inverting
        // div.full on device: candidates m_true ± k ULP — find the k whose
        // div.full(c,127) reproduces s_warp bit-exactly (div.full is
        // monotonic; the preimage of one s is 1-2 adjacent floats).
        {
            const DIVSRC: &str = r#"
extern "C" __global__ void divfull_probe(const float* __restrict__ in, float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(in[i]), "f"(127.0f));
    out[i] = r;
}
"#;
            let ptx = cudarc::nvrtc::compile_ptx(DIVSRC).expect("compile probe");
            let module = ctx.load_module(ptx).expect("load probe");
            let f = module.load_function("divfull_probe").expect("fn");
            for r in 0..p {
                let mut cands = Vec::new();
                for k in -8i32..=8 {
                    let bits = (m_true[r].to_bits() as i32).wrapping_add(k) as u32;
                    cands.push(f32::from_bits(bits));
                }
                let cd = stream.clone_htod(&cands).unwrap();
                let od = stream.alloc_zeros::<f32>(cands.len()).unwrap();
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&cd)
                        .arg(&od)
                        .arg(&(cands.len() as i32))
                        .launch(LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                let mut outs = vec![0f32; cands.len()];
                stream.memcpy_dtoh(&od, &mut outs).unwrap();
                let hits: Vec<i32> = (0..cands.len())
                    .filter(|&i| outs[i].to_bits() == s_warp[r].to_bits())
                    .map(|i| (i as i32) - 8)
                    .collect();
                eprintln!(
                    "row {r}: m_true={:016X} s_warp={:016X} matches k={hits:?} (k=0 => m exact, div/write path at fault; k!=0 => chunk/reduce codegen at fault)",
                    m_true[r].to_bits(), s_warp[r].to_bits()
                );
            }
        }
        eprintln!("row | s_warp vs s_block(head-to-head) | s_warp vs s_mma | s_block vs s_mma | m_true");
        let mut bad_wb = 0usize;
        for r in 0..p {
            let dwb = ulp(s_warp[r], s_blockp[r]);
            let dwm = ulp(s_warp[r], s_mma[r]);
            let dbm = ulp(s_blockp[r], s_mma[r]);
            if dwb != 0 { bad_wb += 1; }
            eprintln!(
                "  {r} | warp={:016X} block={:016X} mma={:016X} | w-b {dwb:+5} w-m {dwm:+5} b-m {dbm:+5} | m_true={:e}",
                s_warp[r].to_bits(), s_blockp[r].to_bits(), s_mma[r].to_bits(), m_true[r]
            );
        }
        assert_eq!(bad_wb, 0, "{bad_wb} row scale(s) diverged warp-vs-block head-to-head");
    }
}
