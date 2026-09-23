//! Issue 615 — CUDA kernels for the full cudarc forward port.
//!
//! Raw CUDA source compiled via `cudarc::nvrtc` at init time. Each kernel is
//! a `const &str` compiled once; dispatches use `stream.launch_builder`.
//!
//! ## Why this exists
//!
//! Issue 608 T2 Phase B proved the dp4a GEMV kernel's 4.7× speedup is
//! uncapturable in mixed mode (CPU elementwise + GPU GEMV) — the per-GEMV
//! transfer overhead cancels the kernel gain (14.42 tok/s vs 15.00 CubeCL
//! resident, `.benchmarks/608`). The only way to capture the speedup is a
//! full GPU-resident forward where ALL ops (including elementwise) run on
//! the same CUDA stream, sharing device buffers without CPU round-trips.
//!
//! These kernels replace the CubeCL elementwise ops (`RmsNormCubeCL`,
//! `ResidualAddCubeCL`, `DeltanetGatingCubeCL`) with raw CUDA equivalents
//! that run on the same `cudarc::driver::safe::CudaStream` as the dp4a GEMV.
//!
//! ## Kernel inventory
//!
//! | Kernel | Replaces | LOC | Status |
//! |---|---|---|---|
//! | `rmsnorm_f32` | `RmsNormCubeCL` | ~40 | T1 (this file) |
//! | `residual_add_f32` | `ResidualAddCubeCL` | ~15 | T2 |
//! | `swiglu_f32` | `DeltanetGatingCubeCL` | ~20 | T3 |
//! | `rope_partial_f32` | `QwenRopePartialCubeCL` | ~30 | T4 (`attention.rs`) |
//! | `split_qg_f32` | `QwenSplitQgCubeCL` | ~15 | T4 |
//! | `rmsnorm_batched_f32` | `RmsNormBatchedCubeCL` | ~40 | T4 |
//! | `kv_cache_append_f32` | `QwenKvCacheAppendCubeCL` | ~15 | T4 |
//! | `attention_decode_f32` | `QwenAttentionDecodeCubeCL` | ~90 | T4 |
//! | `output_gate_f32` | `QwenOutputGateCubeCL` | ~10 | T4 |
//! | `conv1d_f32` | `DeltanetConv1dCubeCL` | ~25 | T5 (`deltanet.rs`) |
//! | `beta_decay_f32` | `DeltanetBetaDecayCubeCL` | ~25 | T5 |
//! | `expand_and_l2_normalize_heads_f32` | `ExpandAndL2NormalizeHeadsCubeCL` | ~40 | T5 |
//! | `recurrence_f32` | `DeltanetRecurrenceCubeCL` | ~100 | T5 |
//! | `z_gating_f32` | `DeltanetZGatingCubeCL` | ~10 | T5 |
//! | `dequant_wte_row_f32` | `DequantWteRowCubeCL` | ~25 | T6 (`embedding.rs`) |

#![allow(clippy::too_many_arguments)]

pub mod attention; // Issue 742 T9.2: AttentionKernels consumed by qwen38_dense_cudarc
pub mod attention_score_mma; // Plan 551 / Issue 773 T2: 3xtf32 score-phase scaffold (unwired, GPU-gated)
pub mod deltanet; // Issue 742 T9.2: DeltanetKernels consumed by qwen38_dense_cudarc
mod embedding;
mod lora;

pub use attention::{AttentionKernels, SPLITGQA_QG_ROWS_MAX_P};
pub use attention_score_mma::AttentionScoreMmaKernels;
pub use deltanet::{DeltanetKernels, HalfStateFmt};
pub use embedding::EmbeddingDequantKernels;
pub use lora::{LoraDecodeKernels, QvLoraGpuCudarc};

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

/// Combined CUDA source for all elementwise kernels. Compiled as a single
/// PTX module — each kernel is a `__global__` function within it.
const ELEMENTWISE_CUDA_SRC: &str = r#"
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ input,   // [dim]
    const float* __restrict__ gamma,   // [dim]
    float* __restrict__ output,        // [dim]
    float inv_dim,
    float eps,
    int dim)
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // 256

    // ── Phase 1: Strided accumulation of x² ──
    float partial_sq = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        float x = input[i];
        partial_sq += x * x;
    }

    // ── Phase 2: Shared memory parallel reduction ──
    __shared__ float smem[256];
    smem[tid] = partial_sq;
    __syncthreads();

    if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
    if (tid < 64)  smem[tid] += smem[tid + 64];  __syncthreads();
    if (tid < 32)  smem[tid] += smem[tid + 32];  __syncthreads();
    if (tid < 16)  smem[tid] += smem[tid + 16];  __syncthreads();
    if (tid < 8)   smem[tid] += smem[tid + 8];   __syncthreads();
    if (tid < 4)   smem[tid] += smem[tid + 4];   __syncthreads();
    if (tid < 2)   smem[tid] += smem[tid + 2];   __syncthreads();
    if (tid < 1)   smem[0] += smem[1];
    __syncthreads();

    // ── Phase 3: Compute inv_rms ──
    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    // ── Phase 4: Normalize and apply gamma ──
    for (int j = tid; j < dim; j += block_size) {
        output[j] = input[j] * inv_rms * gamma[j];
    }
}

extern "C" __global__ void residual_add_f32(
    const float* __restrict__ a,   // [n]
    const float* __restrict__ b,   // [n]
    float* __restrict__ output,    // [n]
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,   // [mlp_hidden]
    const float* __restrict__ up,     // [mlp_hidden]
    float* __restrict__ output,       // [mlp_hidden]
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * up[idx];
    }
}

// ---------------------------------------------------------------------------
// Activation quantize kernel: f32 → int8 + per-block ascale.
// Feeds the dp4a GEMV kernel (Issue 608 T1) on the same stream — keeping the
// activation path entirely GPU-resident (Issue 615 T7) rather than downloading
// to CPU for quantization, which would replicate the Issue 608 Phase B
// mixed-mode failure.
// ---------------------------------------------------------------------------

extern "C" __global__ void quantize_f32_to_i8(
    const float* __restrict__ x,        // [n]
    signed char* __restrict__ out,      // [n]
    float* __restrict__ ascale,         // [ablocks]
    const int n,
    const int ablock,                   // ACTIVATION_BLOCK (16)
    const int ablocks)
{
    // One block per activation block. blockDim.x == ablock (typically 16).
    const int blk = blockIdx.x;
    if (blk >= ablocks) return;
    const int tid = threadIdx.x;

    const int start = blk * ablock;
    const int len = min(ablock, n - start);  // elements valid in this block

    // Each thread reads one element (or zero if out-of-range for the tail).
    float x_val = (tid < len) ? x[start + tid] : 0.0f;
    float a = fabsf(x_val);

    // Warp reduction for max (blockDim.x ≤ 32, so one warp covers the block).
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 1));
    const float absmax = __shfl_sync(0xFFFFFFFFu, a, 0);

    // d = absmax / 127 (or 1.0 if all-zero to avoid div-by-zero).
    const float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
    if (tid == 0) {
        ascale[blk] = d;
    }
    const float inv_d = 1.0f / d;

    if (tid < len) {
        float q = roundf(x_val * inv_d);
        // Clamp to int8 range (just in case absmax round-trip overshoots).
        q = fmaxf(-128.0f, fminf(127.0f, q));
        out[start + tid] = (signed char)q;
    }
}

// ---------------------------------------------------------------------------
// Issue 623 — Fused RMSNorm + quantize kernel.
// Replaces the separate rmsnorm_f32 + quantize_f32_to_i8 pair when the RMSNorm
// output is immediately consumed by a GEMV (which reads int8, not f32).
// Eliminates the intermediate norm_x write + read (~2.6 MB/token saved).
// ---------------------------------------------------------------------------

extern "C" __global__ void rmsnorm_quantize_f32(
    const float* __restrict__ input,   // [dim]
    const float* __restrict__ gamma,   // [dim]
    signed char* __restrict__ out,     // [dim] int8 quantized
    float* __restrict__ ascale,        // [ablocks]
    float inv_dim,
    float eps,
    int dim,
    int ablock,                        // 16
    int ablocks,                       // dim / 16
    int block_size)                    // blockDim.x (next pow2 >= ablocks)
{
    const int tid = threadIdx.x;

    // Each active thread handles 16 contiguous elements (1 activation block).
    const int start = tid * ablock;
    const bool active = (tid < ablocks);

    // ── Phase 1: Load elements + gamma into registers + compute partial sum of squares ──
    // Issue 706: gamma is prefetched here (loads overlap the reduction barrier) so
    // phase 4 computes from registers — eliminating the dependent second global-load
    // phase (the old phase 4 re-read input + gamma, ~40 KB after the barrier). Same
    // values, same multiply order → bit-identical outputs; only the load schedule changed.
    //
    // The loop bound is the compile-time ACTIVATION_BLOCK (16 — the launcher's Rust
    // const, never anything else at any call site), NOT the runtime `ablock` arg: a
    // runtime bound blocks unrolling, which forces vals[]/gam[] into local memory
    // (indexed addressing has no register form). `#pragma unroll` + constant bound
    // keeps both arrays in registers.
    float vals[16];  // ACTIVATION_BLOCK = 16 (compile-time)
    float gam[16];   // Issue 706 — gamma prefetched alongside vals
    float partial_sq = 0.0f;
    if (active) {
        const int len = (start + ablock <= dim) ? ablock : (dim - start);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const bool ok = (i < len);
            float v = ok ? input[start + i] : 0.0f;
            vals[i] = v;
            gam[i] = ok ? gamma[start + i] : 0.0f;
            partial_sq += v * v;
        }
    }

    // ── Phase 2: Shared memory reduction for total sum of squares ──
    // pad to block_size (power of 2) for clean tree reduction.
    __shared__ float smem[1024];  // max block_size = 1024
    smem[tid] = partial_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    // The loop exits when s=32, but we need one more iteration to fold
    // smem[32..63] into smem[0..31] before the warp-level reduction.
    if (block_size > 32 && tid < 32) {
        smem[tid] += smem[tid + 32];
    }
    __syncthreads();
    // Warp-level reduction (final 32 threads, no sync needed within a warp).
    if (tid < 32) {
        volatile float* vsmem = smem;
        if (tid < 16) vsmem[tid] += vsmem[tid + 16];
        if (tid < 8)  vsmem[tid] += vsmem[tid + 8];
        if (tid < 4)  vsmem[tid] += vsmem[tid + 4];
        if (tid < 2)  vsmem[tid] += vsmem[tid + 2];
        if (tid < 1)  vsmem[tid] += vsmem[tid + 1];
    }
    __syncthreads();

    // ── Phase 3: Compute inv_rms ──
    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    // ── Phase 4: Normalize + quantize (Issue 706: from registers — no global re-read) ──
    if (active) {
        const int len = (start + ablock <= dim) ? ablock : (dim - start);

        // Apply RMSNorm + gamma, find absmax for this activation block.
        float absmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float v;
            if (i < len) {
                v = vals[i] * inv_rms * gam[i];
            } else {
                v = 0.0f;
            }
            vals[i] = v;
            absmax = fmaxf(absmax, fabsf(v));
        }

        // Quantize: d = absmax / 127, int8 = round(v / d)
        float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
        ascale[tid] = d;
        float inv_d = 1.0f / d;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i < len) {
                float q = roundf(vals[i] * inv_d);
                q = fmaxf(-128.0f, fminf(127.0f, q));
                out[start + i] = (signed char)q;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 734 T5 — multi-block RMSNorm + quantize. The 1-block kernel above
// pins the whole reduction to ONE SM (measured 4.66 µs avg for dim=5120 —
// 5.2% of decode time across 129 launches/token). This variant spreads the
// leaves across `block_size/32` one-warp blocks and reproduces the 1-block
// kernel's smem-tree reduction pairing EXACTLY over the same leaf order,
// producing a BIT-IDENTICAL inv_rms:
//
// - thread<->ablock mapping is unchanged (global slot t owns elems
//   [t*16, t*16+16); inactive slots publish +0.0 exactly like the 1-block
//   kernel's inactive threads);
// - the cross-block tree replays the same level sequence (s = block_size/2
//   halving to >32, then the t<32 fold, then the warp tree) over the same
//   leaf array, and +0.0 folds are bit-neutral for the non-negative leaves;
// - the quantize phase copies the formula verbatim (roundf + clamp).
//
// Cross-block coordination: plain-store leaf partials + a two-counter
// software barrier whose counters SELF-RESET at kernel exit — the state is
// {0,0} on entry of every launch, including CUDA-graph replays (graph
// replay runs no host code, so the reset must live in the kernel). All
// blocks are one warp and grid ≤ 32 blocks for every supported dim, so the
// spin can never deadlock (all blocks co-resident).
// ---------------------------------------------------------------------------

extern "C" __global__ void rmsnorm_quantize_multiblock_f32(
    const float* __restrict__ input,   // [dim]
    const float* __restrict__ gamma,   // [dim]
    signed char* __restrict__ out,     // [dim] int8 quantized
    float* __restrict__ ascale,        // [ablocks]
    float* __restrict__ partials,      // [block_size] leaf partials (plain stores)
    int* __restrict__ arrived,         // [2] — {0,0} on entry; self-resets on exit
    float inv_dim,
    float eps,
    int dim,
    int ablock,                        // 16
    int ablocks,                       // dim / 16
    int block_size)                    // next_pow2 >= ablocks (same as 1-block)
{
    const int lane = threadIdx.x;      // blockDim.x == 32
    const int t = blockIdx.x * 32 + lane;
    const bool active = t < ablocks;

    float vals[16];
    float gam[16];
    float partial_sq = 0.0f;
    if (active) {
        const int start = t * ablock;
        const int len = (start + ablock <= dim) ? ablock : (dim - start);
        if (len == 16) {
            // Vectorized load path (dim multiple of 16 — the production shape).
            // The partial-sum order is the same element sequence as the scalar
            // loop (i = 0..15 in order), so the leaf value is bit-identical.
            const float4* in4 = (const float4*)(input + start);
            const float4* ga4 = (const float4*)(gamma + start);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = in4[i];
                const float4 g = ga4[i];
                vals[4*i+0] = v.x; vals[4*i+1] = v.y; vals[4*i+2] = v.z; vals[4*i+3] = v.w;
                gam[4*i+0] = g.x; gam[4*i+1] = g.y; gam[4*i+2] = g.z; gam[4*i+3] = g.w;
                partial_sq += v.x * v.x;
                partial_sq += v.y * v.y;
                partial_sq += v.z * v.z;
                partial_sq += v.w * v.w;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const bool ok = (i < len);
                const float v = ok ? input[start + i] : 0.0f;
                vals[i] = v;
                gam[i] = ok ? gamma[start + i] : 0.0f;
                partial_sq += v * v;
            }
        }
    }

    // Publish this block's 32 leaves. Inactive slots store +0.0 — exactly
    // what the 1-block kernel's inactive threads left in smem.
    partials[t] = partial_sq;
    if (lane == 0) {
        __threadfence();
        atomicAdd(&arrived[0], 1);
        // Spin until every block published. grid ≤ 32 one-warp blocks — all
        // co-resident by construction, so this cannot deadlock.
        while (((volatile int*)arrived)[0] < (int)gridDim.x) { }
    }
    __syncwarp();
    __threadfence();  // acquire: partials writes from all blocks are visible

    // ── Replicate the 1-block smem tree over the SAME leaf array ──
    __shared__ float sm[512];  // max block_size (launcher guards block_size <= 512)
    // Zero-fill the warp-tree read range beyond the valid leaves (the 1-block
    // kernel reads uninitialized smem there for block_size < 32 — UB it never
    // hits in production; zeroing makes every shape deterministic).
    if (lane + block_size < 32) sm[lane + block_size] = 0.0f;
    for (int i = lane; i < block_size; i += 32) sm[i] = partials[i];
    __syncwarp();
    for (int s = block_size / 2; s > 32; s >>= 1) {
        for (int t2 = lane; t2 < s; t2 += 32) sm[t2] += sm[t2 + s];
        __syncwarp();
    }
    if (block_size > 32) {
        sm[lane] += sm[lane + 32];   // the explicit t<32 fold (lane < 32 always)
    }
    __syncwarp();
    {
        volatile float* vsm = sm;
        if (lane < 16) vsm[lane] += vsm[lane + 16];
        if (lane < 8)  vsm[lane] += vsm[lane + 8];
        if (lane < 4)  vsm[lane] += vsm[lane + 4];
        if (lane < 2)  vsm[lane] += vsm[lane + 2];
        if (lane < 1)  vsm[lane] += vsm[lane + 1];
    }
    __syncwarp();
    const float sumsq = sm[0];
    const float inv_rms = 1.0f / sqrtf(sumsq * inv_dim + eps);

    // ── Normalize + quantize (formula copied verbatim from the 1-block kernel) ──
    if (active) {
        const int start = t * ablock;
        const int len = (start + ablock <= dim) ? ablock : (dim - start);
        float absmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v;
            if (i < len) {
                v = vals[i] * inv_rms * gam[i];
            } else {
                v = 0.0f;
            }
            vals[i] = v;
            absmax = fmaxf(absmax, fabsf(v));
        }

        float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
        ascale[t] = d;
        float inv_d = 1.0f / d;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < len) {
                float q = roundf(vals[i] * inv_d);
                q = fmaxf(-128.0f, fminf(127.0f, q));
                out[start + i] = (signed char)q;
            }
        }
    }

    // ── Exit barrier + scratch reset (graph-replay safe) ──
    __syncwarp();
    if (lane == 0) {
        __threadfence();
        const int old = atomicAdd(&arrived[1], 1);
        if (old == (int)gridDim.x - 1) {
            // Last block: restore {0,0} for the next launch/replay. partials
            // needs no reset — every slot < block_size is plain-stored each run.
            arrived[0] = 0;
            arrived[1] = 0;
            __threadfence();
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 634 — Sibling of `rmsnorm_quantize_f32` that ALSO writes the f32
// `norm_x` to a side buffer. Same kernel body, same launch cost, one extra
// write per element. Used only by `forward_token_with_final_hidden`
// on the cudarc path so the lm_head LoRA precompute can read `norm_x`.
//
// The hot-path `rmsnorm_quantize_f32` (Issue 623) is unchanged — keeps the
// Issue 631 G1 (dead-buffer removal) bit-identical.
// ---------------------------------------------------------------------------

extern "C" __global__ void rmsnorm_quantize_with_norm_x_f32(
    const float* __restrict__ input,   // [dim]
    const float* __restrict__ gamma,   // [dim]
    signed char* __restrict__ out,     // [dim] int8 quantized
    float* __restrict__ ascale,        // [ablocks]
    float* __restrict__ norm_x_out,    // [dim] f32 RMSNorm output (side buffer)
    float inv_dim,
    float eps,
    int dim,
    int ablock,                        // 16
    int ablocks,                       // dim / 16
    int block_size)                    // blockDim.x (next pow2 >= ablocks)
{
    const int tid = threadIdx.x;

    const int start = tid * ablock;
    const bool active = (tid < ablocks);

    // Issue 706 — single-pass: gamma prefetched in phase 1 (same load-schedule fix as
    // rmsnorm_quantize_f32); phase 4 computes from registers. Bit-identical outputs.
    // Compile-time loop bound (ACTIVATION_BLOCK = 16) + #pragma unroll keeps vals/gam
    // in registers (a runtime `ablock` bound forces local-memory indexing).
    float vals[16];
    float gam[16];
    float partial_sq = 0.0f;
    if (active) {
        const int len = (start + ablock <= dim) ? ablock : (dim - start);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const bool ok = (i < len);
            float v = ok ? input[start + i] : 0.0f;
            vals[i] = v;
            gam[i] = ok ? gamma[start + i] : 0.0f;
            partial_sq += v * v;
        }
    }

    __shared__ float smem[1024];
    smem[tid] = partial_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (block_size > 32 && tid < 32) {
        smem[tid] += smem[tid + 32];
    }
    __syncthreads();
    if (tid < 32) {
        volatile float* vsmem = smem;
        if (tid < 16) vsmem[tid] += vsmem[tid + 16];
        if (tid < 8)  vsmem[tid] += vsmem[tid + 8];
        if (tid < 4)  vsmem[tid] += vsmem[tid + 4];
        if (tid < 2)  vsmem[tid] += vsmem[tid + 2];
        if (tid < 1)  vsmem[tid] += vsmem[tid + 1];
    }
    __syncthreads();

    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    if (active) {
        const int len = (start + ablock <= dim) ? ablock : (dim - start);

        float absmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float v;
            if (i < len) {
                v = vals[i] * inv_rms * gam[i];
            } else {
                v = 0.0f;
            }
            vals[i] = v;
            absmax = fmaxf(absmax, fabsf(v));
            // Issue 634 — write the f32 norm_x to the side buffer alongside
            // the quantize path. Same store the un-fused rmsnorm_f32 would emit.
            if (i < len) {
                norm_x_out[start + i] = v;
            }
        }

        float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
        ascale[tid] = d;
        float inv_d = 1.0f / d;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i < len) {
                float q = roundf(vals[i] * inv_d);
                q = fmaxf(-128.0f, fminf(127.0f, q));
                out[start + i] = (signed char)q;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 625 — Fused SwiGLU + quantize kernel.
// Replaces the separate swiglu_f32 + quantize_f32_to_i8 pair when the SwiGLU
// output is immediately consumed by a GEMV (down_proj reads int8, not f32).
// Eliminates the intermediate ffn_hidden write + read (~115 KB/layer saved).
//
// Grid: same as quantize_f32_to_i8 — ablocks blocks × 16 threads.
// Each thread reads gate[i] + up[i], computes silu(gate)*up, then participates
// in a warp-shuffle absmax reduction + int8 quantization.
// ---------------------------------------------------------------------------

extern "C" __global__ void swiglu_quantize_f32(
    const float* __restrict__ gate,      // [n]
    const float* __restrict__ up,        // [n]
    signed char* __restrict__ out,       // [n] int8 quantized SwiGLU output
    float* __restrict__ ascale,          // [ablocks]
    const int n,
    const int ablock,                    // ACTIVATION_BLOCK (16)
    const int ablocks)
{
    const int blk = blockIdx.x;
    if (blk >= ablocks) return;
    const int tid = threadIdx.x;

    const int start = blk * ablock;
    const int len = min(ablock, n - start);

    // Compute SwiGLU: silu(gate) * up for this thread's element.
    float val = 0.0f;
    if (tid < len) {
        float g = gate[start + tid];
        float silu_g = g / (1.0f + expf(-g));
        val = silu_g * up[start + tid];
    }

    // Warp reduction for absmax (blockDim.x ≤ 32, so one warp covers the block).
    float a = fabsf(val);
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 1));
    const float absmax = __shfl_sync(0xFFFFFFFFu, a, 0);

    const float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
    if (tid == 0) {
        ascale[blk] = d;
    }
    const float inv_d = 1.0f / d;

    if (tid < len) {
        float q = roundf(val * inv_d);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        out[start + tid] = (signed char)q;
    }
}

// ---------------------------------------------------------------------------
// Issue 626 — Fused output-gate + quantize kernels.
//
// Both layer types gate their output before the out_proj GEMV:
//   - DeltaNet:   recurrent_out[i] *= silu(z[i])       (z-gating)
//   - Attention:  attn_out[i]    *= sigmoid(gate[i])   (output-gating)
//
// Previously the gate ran as a separate kernel writing f32, then `gemv_into`
// re-quantized before dispatching the dp4a GEMV. These two kernels merge the
// gate + absmax + int8 quantize into one pass, eliminating 1 kernel launch
// per layer + the intermediate gated f32 round-trip through HBM.
//
// Grid + reduction: identical to swiglu_quantize_f32 (ablocks blocks × 16
// threads, warp-shuffle absmax).
// ---------------------------------------------------------------------------

extern "C" __global__ void gate_silu_quantize_f32(
    const float* __restrict__ x,         // [n]   — input (e.g. recurrent_out)
    const float* __restrict__ gate,      // [n]   — gate vector (e.g. z_buf)
    signed char* __restrict__ out,       // [n]   int8 quantized gated output
    float* __restrict__ ascale,          // [ablocks]
    const int n,
    const int ablock,                    // ACTIVATION_BLOCK (16)
    const int ablocks)
{
    const int blk = blockIdx.x;
    if (blk >= ablocks) return;
    const int tid = threadIdx.x;

    const int start = blk * ablock;
    const int len = min(ablock, n - start);

    // silu(gate) * x
    float val = 0.0f;
    if (tid < len) {
        float g = gate[start + tid];
        float silu_g = g / (1.0f + expf(-g));
        val = silu_g * x[start + tid];
    }

    float a = fabsf(val);
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 1));
    const float absmax = __shfl_sync(0xFFFFFFFFu, a, 0);

    const float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
    if (tid == 0) {
        ascale[blk] = d;
    }
    const float inv_d = 1.0f / d;

    if (tid < len) {
        float q = roundf(val * inv_d);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        out[start + tid] = (signed char)q;
    }
}

extern "C" __global__ void gate_sigmoid_quantize_f32(
    const float* __restrict__ x,         // [n]   — input (e.g. attn_out)
    const float* __restrict__ gate,      // [n]   — gate vector (e.g. attn_gate)
    signed char* __restrict__ out,       // [n]   int8 quantized gated output
    float* __restrict__ ascale,          // [ablocks]
    const int n,
    const int ablock,                    // ACTIVATION_BLOCK (16)
    const int ablocks)
{
    const int blk = blockIdx.x;
    if (blk >= ablocks) return;
    const int tid = threadIdx.x;

    const int start = blk * ablock;
    const int len = min(ablock, n - start);

    // sigmoid(gate) * x
    float val = 0.0f;
    if (tid < len) {
        float g = gate[start + tid];
        float sig = 1.0f / (1.0f + expf(-g));
        val = sig * x[start + tid];
    }

    float a = fabsf(val);
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_down_sync(0xFFFFFFFFu, a, 1));
    const float absmax = __shfl_sync(0xFFFFFFFFu, a, 0);

    const float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
    if (tid == 0) {
        ascale[blk] = d;
    }
    const float inv_d = 1.0f / d;

    if (tid < len) {
        float q = roundf(val * inv_d);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        out[start + tid] = (signed char)q;
    }
}

// ---------------------------------------------------------------------------
// Issue 627 — Fused per-head RMSNorm + silu-gate + quantize for DeltaNet.
//
// DeltaNet step 9 (per-head RMSNorm on recurrent_out) + steps 10-11
// (z-gating + quantize for out_proj GEMV) merged into a single kernel.
// Eliminates 1 kernel launch per DeltaNet layer (48 launches/token) + the
// intermediate f32 normalized recurrent_out HBM round-trip.
//
// The RMSNorm reduction uses the identical 256-thread shared-memory tree as
// rmsnorm_batched_f32, producing bit-identical sum_sq → inv_rms for
// head_dim ≤ 256. The quantize step uses butterfly shfl_xor within groups
// of 16 (fmaxf is order-independent, so absmax is bit-identical regardless
// of reduction method).
//
// Grid:  n_v_heads blocks (one per head)
// Block: 256 threads (matches rmsnorm_batched_f32)
// Constraint: head_dim must be a multiple of ablock (16), head_dim ≤ 256.
// ---------------------------------------------------------------------------

extern "C" __global__ void rmsnorm_gate_silu_quantize_f32(
    const float* __restrict__ x,         // [n_v_heads * head_dim] — pre-norm
    const float* __restrict__ gate,      // [n_v_heads * head_dim] — z_buf
    const float* __restrict__ gamma,     // [head_dim] — linear_norm weights
    signed char* __restrict__ out,       // [n_v_heads * head_dim] int8
    float* __restrict__ ascale,          // [n_v_heads * head_dim / ablock]
    const float inv_head_dim,            // 1.0f / head_dim
    const float eps,
    const int head_dim,                  // e.g. 64
    const int ablock)                    // ACTIVATION_BLOCK (16)
{
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;   // 256
    const int base = head_idx * head_dim;
    const int groups_per_head = head_dim / ablock;

    // ── Phase 1: RMSNorm sum-of-squares (matches rmsnorm_batched_f32 tree) ──
    float partial_sq = 0.0f;
    for (int i = tid; i < head_dim; i += block_size) {
        float xv = x[base + i];
        partial_sq += xv * xv;
    }

    __shared__ float smem[256];
    smem[tid] = partial_sq;
    __syncthreads();

    // Tree reduction (identical to rmsnorm_batched_f32)
    if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
    if (tid < 64)  smem[tid] += smem[tid + 64];  __syncthreads();
    if (tid < 32)  smem[tid] += smem[tid + 32];  __syncthreads();
    if (tid < 16)  smem[tid] += smem[tid + 16];  __syncthreads();
    if (tid < 8)   smem[tid] += smem[tid + 8];   __syncthreads();
    if (tid < 4)   smem[tid] += smem[tid + 4];   __syncthreads();
    if (tid < 2)   smem[tid] += smem[tid + 2];   __syncthreads();
    if (tid < 1)   smem[0] += smem[1];
    __syncthreads();

    // Compute inv_rms (matches rmsnorm_batched_f32)
    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_head_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    // ── Phase 2: compute gated value (matches separate path float order) ──
    // rmsnorm writes: output = input * inv_rms * gamma (left-to-right)
    // gate_silu reads: val = silu(gate) * normed_output
    float val = 0.0f;
    if (tid < head_dim) {
        float xv = x[base + tid];
        float gv = gate[base + tid];
        float silu_g = gv / (1.0f + expf(-gv));
        float x_norm = xv * inv_rms * gamma[tid];
        val = silu_g * x_norm;
    }

    // ── Phase 3: quantize — absmax per group of ablock=16 ──
    // Butterfly shfl_xor reduces within 16-lane groups without crossing
    // boundaries (unlike shfl_down which would cross at delta=8 within a
    // 32-lane warp).
    float a = fabsf(val);
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 1));

    const int group = tid / ablock;
    const int ascale_idx = head_idx * groups_per_head + group;

    const float d = (a > 0.0f) ? (a * (1.0f / 127.0f)) : 1.0f;
    if ((tid % ablock) == 0 && tid < head_dim) {
        ascale[ascale_idx] = d;
    }
    const float inv_d = 1.0f / d;

    if (tid < head_dim) {
        float q = roundf(val * inv_d);
        q = fmaxf(-128.0f, fminf(127.0f, q));
        out[base + tid] = (signed char)q;
    }
}

// ---------------------------------------------------------------------------
// Issue 697 — GPU-side argmax with CPU-exact tie-breaking.
//
// The decode loop's greedy next-token selection previously downloaded the
// full vocab-sized logits vector (~1 MB for Bonsai) and argmaxed on the CPU
// (~380 µs/token of dtoh + host work, Bench 674 stage profile). This kernel
// reduces on-GPU; the host downloads 8 bytes.
//
// Tie-break semantics match the CPU `argmax` (strict `>`, first index wins):
// - within a thread: indices are visited in increasing order, strict `>` keeps
//   the first occurrence;
// - across threads/blocks: the reduction prefers higher value, then SMALLER
//   index on exact float equality;
// - the global atomicMax packs (ordered-float-key << 32) | (~index) so a larger
//   packed value = higher logit, or equal logit with a smaller index.
//   +/-0.0 are normalized to +0.0 before keying so the CPU `(+0.0 > -0.0) ==
//   false` semantics are preserved.
// ---------------------------------------------------------------------------
extern "C" __global__ void argmax_first_f32(
    const float* __restrict__ values,     // [n]
    int n,
    unsigned long long* __restrict__ result)  // [1], zeroed before launch
{
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // -INFINITY via bit pattern (nvrtc has no INFINITY macro in default mode).
    float m = __int_as_float(0xff800000);
    int mi = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = values[i];
        if (v > m) { m = v; mi = i; }
    }

    __shared__ float sm[256];
    __shared__ int si[256];
    sm[tid] = m;
    si[tid] = mi;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm[tid + s] > sm[tid] || (sm[tid + s] == sm[tid] && si[tid + s] < si[tid])) {
                sm[tid] = sm[tid + s];
                si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned int bits = __float_as_uint(sm[0]);
        // Normalize +/-0.0 to the same key (CPU comparison treats them equal).
        if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;
        // Ordered-float key: monotone map from f32 bits to unsigned.
        const unsigned int key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long packed =
            ((unsigned long long)key << 32) | (unsigned int)(~(unsigned int)si[0]);
        atomicMax(result, packed);
    }
}
"#;

/// Error type for cudarc kernel operations.
#[derive(Debug)]
pub enum CudarcKernelError {
    /// CUDA context creation failed.
    CudaInit(String),
    /// NVRTC compilation failed.
    Compile(String),
    /// Kernel launch failed.
    Launch(String),
    /// Caller passed an invalid argument (e.g. `set_lora` targeting a
    /// non-DeltaNet layer — Issue 722 H2; was `debug_assert!`-only, i.e. a
    /// silent no-op adapter in release builds).
    InvalidArg(String),
}

impl std::fmt::Display for CudarcKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CudaInit(s) => write!(f, "CUDA init failed: {s}"),
            Self::Compile(s) => write!(f, "NVRTC compile failed: {s}"),
            Self::Launch(s) => write!(f, "kernel launch failed: {s}"),
            Self::InvalidArg(s) => write!(f, "invalid argument: {s}"),
        }
    }
}

impl std::error::Error for CudarcKernelError {}

/// Holds compiled CUDA kernels for elementwise ops.
///
/// Compiled once at construction; all kernels share the same CUDA module.
/// The caller provides the stream — kernels are launched on whatever stream
/// the caller specifies, allowing sharing with the dp4a GEMV handler.
pub struct ElementwiseKernels {
    rmsnorm: CudaFunction,
    residual_add: CudaFunction,
    swiglu: CudaFunction,
    quantize: CudaFunction,
    rmsnorm_quantize: CudaFunction,
    /// Issue 634 — sibling of `rmsnorm_quantize` that also writes the f32
    /// `norm_x` to a side buffer. Only used by `forward_token_with_final_hidden`
    /// on the cudarc path; the hot path keeps using `rmsnorm_quantize`.
    rmsnorm_quantize_with_norm_x: CudaFunction,
    /// Issue 734 T5 — multi-block rmsnorm_quantize (bit-identical reduction,
    /// leaves spread across one-warp blocks instead of 1 SM). Used by
    /// `launch_rmsnorm_quantize` when the scratch is initialized.
    rmsnorm_quantize_mb: CudaFunction,
    /// Issue 734 T5 — leaf-partial scratch for the multi-block kernel.
    /// Set once via [`Self::init_rmsnorm_mb_scratch`] on the forward's stream,
    /// BEFORE any graph capture (addresses bake into captured graphs).
    rmsnorm_mb_partials: std::sync::OnceLock<cudarc::driver::safe::CudaSlice<f32>>,
    /// Issue 734 T5 — arrival counters for the multi-block kernel's software
    /// barrier (`{0,0}` at rest; the kernel self-resets them on exit so every
    /// graph replay starts clean).
    rmsnorm_mb_arrived: std::sync::OnceLock<cudarc::driver::safe::CudaSlice<i32>>,
    swiglu_quantize: CudaFunction,
    gate_silu_quantize: CudaFunction,
    gate_sigmoid_quantize: CudaFunction,
    rmsnorm_gate_silu_quantize: CudaFunction,
    /// Issue 697 — GPU-side argmax (first-index tie-break, CPU-exact).
    argmax_first: CudaFunction,
    _module: Arc<CudaModule>,
}

impl ElementwiseKernels {
    /// Compile all elementwise kernels via nvrtc.
    ///
    /// Uses `sm_89` (Ada Lovelace / RTX 4090). The kernels only use basic
    /// CUDA features (shared memory, `__syncthreads`) so any sm_60+ GPU works.
    ///
    /// Takes `Arc<CudaContext>` because `CudaContext::load_module` requires
    /// `self: &Arc<Self>` (cudarc 0.19 ARC-self pattern).
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            ELEMENTWISE_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| CudarcKernelError::Compile(e.to_string()))?;

        let rmsnorm = module
            .load_function("rmsnorm_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let residual_add = module
            .load_function("residual_add_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let swiglu = module
            .load_function("swiglu_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let quantize = module
            .load_function("quantize_f32_to_i8")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let rmsnorm_quantize = module
            .load_function("rmsnorm_quantize_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        // Issue 634 — sibling kernel that also emits the f32 norm_x side buffer.
        let rmsnorm_quantize_with_norm_x = module
            .load_function("rmsnorm_quantize_with_norm_x_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        // Issue 734 T5 — multi-block variant (bit-identical reduction).
        let rmsnorm_quantize_mb = module
            .load_function("rmsnorm_quantize_multiblock_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let swiglu_quantize = module
            .load_function("swiglu_quantize_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let gate_silu_quantize = module
            .load_function("gate_silu_quantize_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let gate_sigmoid_quantize = module
            .load_function("gate_sigmoid_quantize_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let rmsnorm_gate_silu_quantize = module
            .load_function("rmsnorm_gate_silu_quantize_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        // Issue 697 — GPU-side argmax for the decode loop.
        let argmax_first = module
            .load_function("argmax_first_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        Ok(Self {
            rmsnorm,
            residual_add,
            swiglu,
            quantize,
            rmsnorm_quantize,
            rmsnorm_quantize_with_norm_x,
            rmsnorm_quantize_mb,
            rmsnorm_mb_partials: std::sync::OnceLock::new(),
            rmsnorm_mb_arrived: std::sync::OnceLock::new(),
            swiglu_quantize,
            gate_silu_quantize,
            gate_sigmoid_quantize,
            rmsnorm_gate_silu_quantize,
            argmax_first,
            _module: module,
        })
    }

    /// Launch RMSNorm: `output[i] = input[i] * inv_rms * gamma[i]`.
    ///
    /// - `input`, `gamma`, `output`: device slices of length `dim`
    /// - `inv_dim`: `1.0 / dim` (precomputed on CPU)
    /// - `eps`: RMS norm epsilon
    ///
    /// Dispatch: 1 block × 256 threads (single-block shared-memory reduction).
    pub fn launch_rmsnorm(
        &self,
        stream: &CudaStream,
        input: &cudarc::driver::safe::CudaSlice<f32>,
        gamma: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        dim: usize,
        eps: f32,
    ) -> Result<(), CudarcKernelError> {
        let inv_dim = 1.0f32 / dim as f32;
        let dim_i32 = dim as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4, // 256 floats of shared memory
        };
        unsafe {
            stream
                .launch_builder(&self.rmsnorm)
                .arg(input)
                .arg(gamma)
                .arg(output)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 734 T5 — allocate the multi-block rmsnorm scratch on the given
    /// stream (leaf partials `[max_leaves]` f32 + arrival counters `[2]` i32,
    /// zero-initialized). Call once at forward construction, BEFORE any graph
    /// capture — the kernel addresses bake into captured graphs. Idempotent:
    /// later calls are no-ops.
    ///
    /// `max_leaves` must cover the largest `block_size` (= next_pow2(dim/16),
    /// capped at 512 — the launcher falls back to the 1-block kernel above
    /// that, so 512 slots suffice for dim ≤ 8192).
    pub fn init_rmsnorm_mb_scratch(
        &self,
        stream: &std::sync::Arc<CudaStream>,
        max_leaves: usize,
    ) -> Result<(), CudarcKernelError> {
        if self.rmsnorm_mb_partials.get().is_some() {
            return Ok(());
        }
        let partials = stream
            .alloc_zeros::<f32>(max_leaves)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let arrived = stream
            .alloc_zeros::<i32>(2)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let _ = self.rmsnorm_mb_partials.set(partials);
        let _ = self.rmsnorm_mb_arrived.set(arrived);
        Ok(())
    }

    /// Launch fused RMSNorm + quantize: computes `norm_x = rmsnorm(input, gamma)`
    /// and quantizes to int8 in a single pass, writing directly to `out_i8` +
    /// `ascale` without the intermediate f32 `norm_x` buffer.
    ///
    /// Issue 623 — eliminates 1 kernel launch + ~2 × dim × 4 B intermediate
    /// memory traffic per call (read + write of norm_x).
    ///
    /// - `input`, `gamma`: device slices of length `dim`
    /// - `out_i8`: device slice of length `dim` (int8 quantized output)
    /// - `ascale`: device slice of length `dim / 16` (per-block activation scale)
    ///
    /// Dispatch (Issue 734 T5): multi-block — `block_size/32` one-warp blocks
    /// with a software barrier + the 1-block kernel's exact reduction pairing
    /// (bit-identical `inv_rms`; the 1-block shape pinned the whole reduction
    /// to a single SM at 4.66 µs avg). Falls back to the 1-block kernel when
    /// the scratch is uninitialized (unit-test paths) or `block_size` exceeds
    /// the scratch/shared-memory bound of 512 (dim > 8192).
    pub fn launch_rmsnorm_quantize(
        &self,
        stream: &CudaStream,
        input: &cudarc::driver::safe::CudaSlice<f32>,
        gamma: &cudarc::driver::safe::CudaSlice<f32>,
        out_i8: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        dim: usize,
        eps: f32,
    ) -> Result<(), CudarcKernelError> {
        const ABLOCK: usize = 16;
        let ablocks = dim.div_ceil(ABLOCK);
        // Next power of 2 >= ablocks, capped at 1024.
        let block_size = {
            let mut bs = 1usize;
            while bs < ablocks {
                bs <<= 1;
            }
            bs.min(1024)
        };
        let inv_dim = 1.0f32 / dim as f32;
        let dim_i32 = dim as i32;
        let ablock_i32 = ABLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let block_size_i32 = block_size as i32;

        // Issue 734 T5 — multi-block path when scratch is ready and the shape
        // fits (block_size ≤ 512 leaves ≤ sm[512]; grid ≤ 16 one-warp blocks).
        if block_size <= 512
            && let (Some(partials), Some(arrived)) =
                (self.rmsnorm_mb_partials.get(), self.rmsnorm_mb_arrived.get())
        {
            let grid = (block_size / 32).max(1) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&self.rmsnorm_quantize_mb)
                    .arg(input)
                    .arg(gamma)
                    .arg(out_i8)
                    .arg(ascale)
                    .arg(partials)
                    .arg(arrived)
                    .arg(&inv_dim)
                    .arg(&eps)
                    .arg(&dim_i32)
                    .arg(&ablock_i32)
                    .arg(&ablocks_i32)
                    .arg(&block_size_i32)
                    .launch(cfg)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32, // block_size floats
        };
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_quantize)
                .arg(input)
                .arg(gamma)
                .arg(out_i8)
                .arg(ascale)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .arg(&block_size_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch fused RMSNorm + quantize + norm_x side-buffer write. Sibling of
    /// [`launch_rmsnorm_quantize`](Self::launch_rmsnorm_quantize) that ALSO
    /// writes the f32 `norm_x = input[i] * inv_rms * gamma[i]` to
    /// `norm_x_out` alongside the int8 + ascale outputs. Same launch cost,
    /// same dispatch, one extra elementwise write.
    ///
    /// Issue 634 — used only by `TernaryDeltanetGpuForwardCudarc::
    /// forward_token_with_final_hidden` so the lm_head LoRA precompute can
    /// consume the f32 `norm_x`. The hot path keeps using
    /// `launch_rmsnorm_quantize` (no side-buffer write, no regression).
    ///
    /// - `input`, `gamma`: device slices of length `dim`
    /// - `out_i8`: device slice of length `dim` (int8 quantized output)
    /// - `ascale`: device slice of length `dim / 16` (per-block activation scale)
    /// - `norm_x_out`: device slice of length `dim` (f32 RMSNorm output)
    ///
    /// Dispatch: 1 block × `next_pow2(dim/16)` threads (same as
    /// `launch_rmsnorm_quantize`).
    pub fn launch_rmsnorm_quantize_with_norm_x(
        &self,
        stream: &CudaStream,
        input: &cudarc::driver::safe::CudaSlice<f32>,
        gamma: &cudarc::driver::safe::CudaSlice<f32>,
        out_i8: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        norm_x_out: &cudarc::driver::safe::CudaSlice<f32>,
        dim: usize,
        eps: f32,
    ) -> Result<(), CudarcKernelError> {
        const ABLOCK: usize = 16;
        let ablocks = dim.div_ceil(ABLOCK);
        let block_size = {
            let mut bs = 1usize;
            while bs < ablocks {
                bs <<= 1;
            }
            bs.min(1024)
        };
        let inv_dim = 1.0f32 / dim as f32;
        let dim_i32 = dim as i32;
        let ablock_i32 = ABLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let block_size_i32 = block_size as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_quantize_with_norm_x)
                .arg(input)
                .arg(gamma)
                .arg(out_i8)
                .arg(ascale)
                .arg(norm_x_out)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .arg(&block_size_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch fused SwiGLU + quantize: computes `silu(gate) * up` and quantizes
    /// to int8 in a single pass, writing directly to `out_i8` + `ascale`
    /// without the intermediate f32 `ffn_hidden` buffer.
    ///
    /// Issue 625 — eliminates 1 kernel launch + ~2 × n × 4 B intermediate
    /// memory traffic per call (read + write of ffn_hidden).
    ///
    /// - `gate`, `up`: device slices of length `n`
    /// - `out_i8`: device slice of length `n` (int8 quantized SwiGLU output)
    /// - `ascale`: device slice of length `n / 16` (per-block activation scale)
    ///
    /// Dispatch: `ablocks` blocks × 16 threads (same grid as `launch_quantize`).
    pub fn launch_swiglu_quantize(
        &self,
        stream: &CudaStream,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        up: &cudarc::driver::safe::CudaSlice<f32>,
        out_i8: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        const ABLOCK: usize = 16;
        let ablocks = n.div_ceil(ABLOCK);
        let n_i32 = n as i32;
        let ablock_i32 = ABLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (ablocks as u32, 1, 1),
            block_dim: (ABLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.swiglu_quantize)
                .arg(gate)
                .arg(up)
                .arg(out_i8)
                .arg(ascale)
                .arg(&n_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch fused silu-gate + quantize: computes `silu(gate) * x` and
    /// quantizes to int8 in a single pass. Issue 626 — used by the DeltaNet
    /// output path to merge z-gating into the `out_proj` GEMV input quantize.
    ///
    /// - `x`, `gate`: device slices of length `n`
    /// - `out_i8`: device slice of length `n` (int8 quantized gated output)
    /// - `ascale`: device slice of length `n / 16`
    ///
    /// Dispatch: `ablocks` blocks × 16 threads (same grid as `launch_quantize`).
    pub fn launch_gate_silu_quantize(
        &self,
        stream: &CudaStream,
        x: &cudarc::driver::safe::CudaSlice<f32>,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        out_i8: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        const ABLOCK: usize = 16;
        let ablocks = n.div_ceil(ABLOCK);
        let n_i32 = n as i32;
        let ablock_i32 = ABLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (ablocks as u32, 1, 1),
            block_dim: (ABLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.gate_silu_quantize)
                .arg(x)
                .arg(gate)
                .arg(out_i8)
                .arg(ascale)
                .arg(&n_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch fused sigmoid-gate + quantize: computes `sigmoid(gate) * x` and
    /// quantizes to int8 in a single pass. Issue 626 — used by the attention
    /// output path to merge output-gating into the `wo` GEMV input quantize.
    ///
    /// - `x`, `gate`: device slices of length `n`
    /// - `out_i8`: device slice of length `n` (int8 quantized gated output)
    /// - `ascale`: device slice of length `n / 16`
    ///
    /// Dispatch: `ablocks` blocks × 16 threads (same grid as `launch_quantize`).
    pub fn launch_gate_sigmoid_quantize(
        &self,
        stream: &CudaStream,
        x: &cudarc::driver::safe::CudaSlice<f32>,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        out_i8: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        const ABLOCK: usize = 16;
        let ablocks = n.div_ceil(ABLOCK);
        let n_i32 = n as i32;
        let ablock_i32 = ABLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (ablocks as u32, 1, 1),
            block_dim: (ABLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.gate_sigmoid_quantize)
                .arg(x)
                .arg(gate)
                .arg(out_i8)
                .arg(ascale)
                .arg(&n_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch fused per-head RMSNorm + silu-gate + quantize (Issue 627).
    ///
    /// Computes, in a single kernel per head:
    ///   `inv_rms = 1 / sqrt(mean(x_head²) + eps)`
    ///   `val = silu(gate) * (x * inv_rms * gamma)`
    ///   `out_i8, ascale = quantize(val)`
    ///
    /// - `x`, `gate`: device slices of length `n_v_heads * head_dim`
    /// - `gamma`: device slice of length `head_dim`
    /// - `out_i8`: device slice of length `n_v_heads * head_dim` (int8)
    /// - `ascale`: device slice of length `n_v_heads * head_dim / 16`
    ///
    /// Dispatch: `n_v_heads` blocks × 256 threads (one block per head).
    /// Constraint: `head_dim` must be a multiple of 16 and ≤ 256.
    pub fn launch_rmsnorm_gate_silu_quantize(
        &self,
        stream: &CudaStream,
        x: &cudarc::driver::safe::CudaSlice<f32>,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        gamma: &cudarc::driver::safe::CudaSlice<f32>,
        out_i8: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        n_v_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), CudarcKernelError> {
        const ABLOCK: usize = 16;
        debug_assert!(
            head_dim.is_multiple_of(ABLOCK),
            "Issue 627: head_dim must be a multiple of {ABLOCK}"
        );
        debug_assert!(
            head_dim <= 256,
            "Issue 627: head_dim must be ≤ 256 (one thread per element)"
        );
        let inv_head_dim = 1.0f32 / head_dim as f32;
        let head_dim_i32 = head_dim as i32;
        let ablock_i32 = ABLOCK as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_v_heads as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * core::mem::size_of::<f32>() as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_gate_silu_quantize)
                .arg(x)
                .arg(gate)
                .arg(gamma)
                .arg(out_i8)
                .arg(ascale)
                .arg(&inv_head_dim)
                .arg(&eps)
                .arg(&head_dim_i32)
                .arg(&ablock_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch residual add: `output[i] = a[i] + b[i]`.
    ///
    /// Dispatch: `ceil(n/256)` blocks × 256 threads.
    pub fn launch_residual_add(
        &self,
        stream: &CudaStream,
        a: &cudarc::driver::safe::CudaSlice<f32>,
        b: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        let n_i32 = n as i32;
        let grid_x = (n as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.residual_add)
                .arg(a)
                .arg(b)
                .arg(output)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 697 — GPU-side argmax over `values[0..n]` with CPU-exact
    /// first-index tie-breaking (see `argmax_first_f32` kernel docs).
    ///
    /// `result` MUST be zeroed before each launch (stream memset is
    /// graph-capturable). Reads back as: index = `!(packed as u32)`.
    ///
    /// Dispatch: `ceil(n/256)` blocks × 256 threads (one element per thread
    /// for vocab-sized vectors).
    pub fn launch_argmax_first(
        &self,
        stream: &CudaStream,
        values: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
        result: &cudarc::driver::safe::CudaSlice<u64>,
    ) -> Result<(), CudarcKernelError> {
        let n_i32 = n as i32;
        let grid_x = (n as u32).div_ceil(256).min(4096);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * (core::mem::size_of::<f32>() + core::mem::size_of::<i32>())
                as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.argmax_first)
                .arg(values)
                .arg(&n_i32)
                .arg(result)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch SwiGLU: `output[i] = silu(gate[i]) * up[i]`.
    ///
    /// where `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
    ///
    /// Dispatch: `ceil(n/256)` blocks × 256 threads.
    pub fn launch_swiglu(
        &self,
        stream: &CudaStream,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        up: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        let n_i32 = n as i32;
        let grid_x = (n as u32).div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.swiglu)
                .arg(gate)
                .arg(up)
                .arg(output)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the activation quantize kernel: per-block (16-element) absmax +
    /// round(x / d) → int8.
    ///
    /// Feeds the dp4a GEMV kernel on the same stream — keeping activations
    /// GPU-resident (Issue 615 T7). Mirrors the CPU quantize loop in
    /// `gemv_ternary_cuda_raw::TernaryGemmCudaRaw::forward`.
    ///
    /// - `x`: `[n]` f32 activation (input)
    /// - `out`: `[n]` int8 quantized activation (output)
    /// - `ascale`: `[ceil(n / ACTIVATION_BLOCK)]` f32 per-block scales (output)
    ///
    /// Dispatch: `ablocks` blocks × `ACTIVATION_BLOCK` (16) threads.
    pub fn launch_quantize(
        &self,
        stream: &CudaStream,
        x: &cudarc::driver::safe::CudaSlice<f32>,
        out: &cudarc::driver::safe::CudaSlice<i8>,
        ascale: &cudarc::driver::safe::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        // Must match `ACTIVATION_BLOCK` in `gemv_ternary_cuda_raw.rs` (16).
        const ABLOCK: usize = 16;
        let ablocks = n.div_ceil(ABLOCK);
        let n_i32 = n as i32;
        let ablock_i32 = ABLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let cfg = LaunchConfig {
            grid_dim: (ablocks as u32, 1, 1),
            block_dim: (ABLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.quantize)
                .arg(x)
                .arg(out)
                .arg(ascale)
                .arg(&n_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::safe::CudaContext;

    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    #[test]
    fn test_rmsnorm_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");

        let dim = 5120usize; // n_embd for Bonsai-27B
        let eps = 1e-6f32;

        // Input: simple pattern [1.0, 2.0, 3.0, ...] mod 10
        let input: Vec<f32> = (0..dim).map(|i| ((i % 10) as f32) + 1.0).collect();
        let gamma: Vec<f32> = vec![1.0; dim]; // identity gamma

        // CPU reference
        let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let cpu_out: Vec<f32> = input.iter().map(|&x| x * inv_rms * 1.0).collect();

        // GPU
        let input_dev = stream.clone_htod(&input).unwrap();
        let gamma_dev = stream.clone_htod(&gamma).unwrap();
        let output_dev = stream.alloc_zeros::<f32>(dim).unwrap();

        kernels
            .launch_rmsnorm(&stream, &input_dev, &gamma_dev, &output_dev, dim, eps)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; dim];
        stream.memcpy_dtoh(&output_dev, &mut gpu_out).unwrap();

        // Compare
        let mut max_rel = 0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            let denom = a.abs().max(1e-6);
            let rel = (a - b).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!("[rmsnorm] dim={dim}: max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-4, "RMSNorm max_rel {max_rel:.4e} > 1e-4");
    }

    #[test]
    fn test_residual_add_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");

        let n = 5120usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2 - 100.0).collect();

        let a_dev = stream.clone_htod(&a).unwrap();
        let b_dev = stream.clone_htod(&b).unwrap();
        let out_dev = stream.alloc_zeros::<f32>(n).unwrap();

        kernels
            .launch_residual_add(&stream, &a_dev, &b_dev, &out_dev, n)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; n];
        stream.memcpy_dtoh(&out_dev, &mut gpu_out).unwrap();

        for i in 0..n {
            let expected = a[i] + b[i];
            assert!(
                (gpu_out[i] - expected).abs() < 1e-5,
                "residual_add mismatch at {i}: got {}, expected {}",
                gpu_out[i],
                expected
            );
        }
        eprintln!("[residual_add] n={n}: all match");
    }

    #[test]
    fn test_swiglu_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");

        let n = 17408usize; // mlp_hidden for Bonsai-27B
        let gate: Vec<f32> = (0..n).map(|i| ((i as f32) / n as f32 - 0.5) * 4.0).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i as f32) / n as f32) * 2.0).collect();

        // CPU reference: silu(g) * up
        let cpu_out: Vec<f32> = (0..n)
            .map(|i| {
                let g = gate[i];
                let silu = g / (1.0 + (-g).exp());
                silu * up[i]
            })
            .collect();

        let gate_dev = stream.clone_htod(&gate).unwrap();
        let up_dev = stream.clone_htod(&up).unwrap();
        let out_dev = stream.alloc_zeros::<f32>(n).unwrap();

        kernels
            .launch_swiglu(&stream, &gate_dev, &up_dev, &out_dev, n)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; n];
        stream.memcpy_dtoh(&out_dev, &mut gpu_out).unwrap();

        let mut max_rel = 0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            let denom = a.abs().max(1e-6);
            let rel = (a - b).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!("[swiglu] n={n}: max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-4, "SwiGLU max_rel {max_rel:.4e} > 1e-4");
    }

    /// GPU quantize kernel must match the CPU reference bit-exactly (the dp4a
    /// kernel reads back the int8 codes + ascale and reconstructs f32 — any
    /// divergence in quantization propagates directly into the GEMV output).
    #[test]
    fn test_quantize_matches_cpu() {
        const ABLOCK: usize = 16;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");
        let n = 5120usize; // n_embd for Bonsai-27B
        let ablocks = n.div_ceil(ABLOCK);

        // Mixed-magnitude input: alternating small/large values so each block
        // has a different absmax (exercises per-block scale independence).
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let blk = i / ABLOCK;
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.1 * ((i % 7) as f32 + 1.0);
                sign * mag
            })
            .collect();

        // CPU reference quantize — mirrors `TernaryGemmCudaRaw::forward`.
        let mut cpu_i8 = vec![0i8; n];
        let mut cpu_ascale = vec![0f32; ablocks];
        for (blk, ascale_val) in cpu_ascale.iter_mut().enumerate() {
            let start = blk * ABLOCK;
            let end = (start + ABLOCK).min(n);
            let mut absmax = 0.0f32;
            for &xv in &x[start..end] {
                absmax = absmax.max(xv.abs());
            }
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            *ascale_val = d;
            let inv_d = 1.0 / d;
            for i in start..end {
                let q = (x[i] * inv_d).round().clamp(-128.0, 127.0);
                cpu_i8[i] = q as i8;
            }
        }

        let x_dev = stream.clone_htod(&x).unwrap();
        let i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_quantize(&stream, &x_dev, &i8_dev, &ascale_dev, n)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_i8 = vec![0i8; n];
        let mut gpu_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&i8_dev, &mut gpu_i8).unwrap();
        stream.memcpy_dtoh(&ascale_dev, &mut gpu_ascale).unwrap();

        // ascale must match bit-exactly (same div-by-127 path).
        let mut max_scale_diff = 0f32;
        for blk in 0..ablocks {
            let diff = (gpu_ascale[blk] - cpu_ascale[blk]).abs();
            max_scale_diff = max_scale_diff.max(diff);
        }
        eprintln!("[quantize] ascale: max_diff={max_scale_diff:.4e}");
        assert!(
            max_scale_diff < 1e-6,
            "ascale max_diff {max_scale_diff:.4e}"
        );

        // int8 codes must match exactly (round-to-nearest is deterministic).
        let mut mismatches = 0usize;
        for i in 0..n {
            if gpu_i8[i] != cpu_i8[i] {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "  mismatch at {i}: gpu={}, cpu={}, x={:.4}",
                        gpu_i8[i], cpu_i8[i], x[i]
                    );
                }
            }
        }
        eprintln!("[quantize] n={n}: {mismatches} int8 mismatches");
        assert!(mismatches == 0, "{mismatches} int8 code mismatches");
    }

    /// Issue 623 — verify the fused rmsnorm_quantize kernel produces bit-identical
    /// int8 + ascale output to the separate rmsnorm → quantize path.
    #[test]
    fn test_rmsnorm_quantize_matches_separate() {
        const ABLOCK: usize = 16;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");
        let dim = 5120usize; // n_embd for Bonsai-27B
        let ablocks = dim.div_ceil(ABLOCK);
        let eps = 1e-6f32;

        // Mixed-magnitude input + non-trivial gamma.
        let input: Vec<f32> = (0..dim)
            .map(|i| {
                let blk = i / ABLOCK;
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.1 * ((i % 7) as f32 + 1.0);
                sign * mag
            })
            .collect();
        let gamma: Vec<f32> = (0..dim).map(|i| 0.8 + 0.4 * ((i as f32) / dim as f32)).collect();

        // ── Reference path: separate rmsnorm → quantize ──
        let input_dev = stream.clone_htod(&input).unwrap();
        let gamma_dev = stream.clone_htod(&gamma).unwrap();
        let norm_x_dev = stream.alloc_zeros::<f32>(dim).unwrap();
        let ref_i8_dev = stream.alloc_zeros::<i8>(dim).unwrap();
        let ref_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_rmsnorm(&stream, &input_dev, &gamma_dev, &norm_x_dev, dim, eps)
            .expect("rmsnorm launch");
        kernels
            .launch_quantize(&stream, &norm_x_dev, &ref_i8_dev, &ref_ascale_dev, dim)
            .expect("quantize launch");
        stream.synchronize().expect("sync");

        let mut ref_i8 = vec![0i8; dim];
        let mut ref_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8_dev, &mut ref_i8).unwrap();
        stream.memcpy_dtoh(&ref_ascale_dev, &mut ref_ascale).unwrap();

        // ── Fused path: rmsnorm_quantize in one kernel ──
        let fused_i8_dev = stream.alloc_zeros::<i8>(dim).unwrap();
        let fused_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_rmsnorm_quantize(
                &stream, &input_dev, &gamma_dev, &fused_i8_dev, &fused_ascale_dev, dim, eps,
            )
            .expect("fused launch");
        stream.synchronize().expect("sync");

        let mut fused_i8 = vec![0i8; dim];
        let mut fused_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&fused_i8_dev, &mut fused_i8).unwrap();
        stream.memcpy_dtoh(&fused_ascale_dev, &mut fused_ascale).unwrap();

        // ── Compare int8 codes (must be bit-exact) ──
        let mut i8_mismatches = 0usize;
        for i in 0..dim {
            if fused_i8[i] != ref_i8[i] {
                i8_mismatches += 1;
                if i8_mismatches <= 5 {
                    eprintln!(
                        "  i8 mismatch at {i}: fused={}, ref={}",
                        fused_i8[i], ref_i8[i]
                    );
                }
            }
        }

        // ── Compare ascale (must match within float precision) ──
        let mut max_scale_diff = 0f32;
        for blk in 0..ablocks {
            let diff = (fused_ascale[blk] - ref_ascale[blk]).abs();
            max_scale_diff = max_scale_diff.max(diff);
        }

        eprintln!(
            "[rmsnorm_quantize] dim={dim}: {i8_mismatches} i8 mismatches, ascale max_diff={max_scale_diff:.4e}"
        );
        assert!(i8_mismatches == 0, "{i8_mismatches} int8 mismatches");
        assert!(max_scale_diff < 1e-5, "ascale max_diff {max_scale_diff:.4e}");
    }

    /// Issue 625 — verify the fused swiglu_quantize kernel produces bit-identical
    /// int8 + ascale output to the separate swiglu → quantize path.
    #[test]
    fn test_swiglu_quantize_matches_separate() {
        const ABLOCK: usize = 16;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");
        let n = 14336usize; // mlp_hidden for Bonsai-27B
        let ablocks = n.div_ceil(ABLOCK);

        // Mixed-magnitude gate + up inputs.
        let gate: Vec<f32> = (0..n)
            .map(|i| {
                let blk = i / ABLOCK;
                let sign = if i.is_multiple_of(3) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.15 * ((i % 5) as f32 + 1.0);
                sign * mag
            })
            .collect();
        let up: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = 0.5 + 0.3 * ((i as f32) / n as f32);
                sign * mag
            })
            .collect();

        // ── Reference path: separate swiglu → quantize ──
        let gate_dev = stream.clone_htod(&gate).unwrap();
        let up_dev = stream.clone_htod(&up).unwrap();
        let ffn_hidden_dev = stream.alloc_zeros::<f32>(n).unwrap();
        let ref_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let ref_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_swiglu(&stream, &gate_dev, &up_dev, &ffn_hidden_dev, n)
            .expect("swiglu launch");
        kernels
            .launch_quantize(&stream, &ffn_hidden_dev, &ref_i8_dev, &ref_ascale_dev, n)
            .expect("quantize launch");
        stream.synchronize().expect("sync");

        let mut ref_i8 = vec![0i8; n];
        let mut ref_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8_dev, &mut ref_i8).unwrap();
        stream.memcpy_dtoh(&ref_ascale_dev, &mut ref_ascale).unwrap();

        // ── Fused path: swiglu_quantize in one kernel ──
        let fused_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let fused_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_swiglu_quantize(
                &stream, &gate_dev, &up_dev, &fused_i8_dev, &fused_ascale_dev, n,
            )
            .expect("fused launch");
        stream.synchronize().expect("sync");

        let mut fused_i8 = vec![0i8; n];
        let mut fused_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&fused_i8_dev, &mut fused_i8).unwrap();
        stream.memcpy_dtoh(&fused_ascale_dev, &mut fused_ascale).unwrap();

        // ── Compare int8 codes (must be bit-exact) ──
        let mut i8_mismatches = 0usize;
        for i in 0..n {
            if fused_i8[i] != ref_i8[i] {
                i8_mismatches += 1;
                if i8_mismatches <= 5 {
                    eprintln!(
                        "  i8 mismatch at {i}: fused={}, ref={}",
                        fused_i8[i], ref_i8[i]
                    );
                }
            }
        }

        // ── Compare ascale (must match within float precision) ──
        let mut max_scale_diff = 0f32;
        for blk in 0..ablocks {
            let diff = (fused_ascale[blk] - ref_ascale[blk]).abs();
            max_scale_diff = max_scale_diff.max(diff);
        }

        eprintln!(
            "[swiglu_quantize] n={n}: {i8_mismatches} i8 mismatches, ascale max_diff={max_scale_diff:.4e}"
        );
        assert!(i8_mismatches == 0, "{i8_mismatches} int8 mismatches");
        assert!(max_scale_diff < 1e-5, "ascale max_diff {max_scale_diff:.4e}");
    }

    /// Issue 626 — verify the fused `gate_silu_quantize_f32` kernel produces
    /// bit-identical int8 + ascale output to the separate GPU path
    /// (`launch_z_gating` → `launch_quantize`). This is the actual production
    /// reference path: both use the same nvrtc `expf`, so the result is
    /// bit-exact (no CPU↔GPU math-library divergence).
    #[test]
    fn test_gate_silu_quantize_matches_separate() {
        const ABLOCK: usize = 16;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx.clone()).expect("compile elem");
        let dn_kernels = super::DeltanetKernels::new(ctx).expect("compile dn");
        // DeltaNet z_dim for Bonsai-27B (n_v_heads * head_dim = 128 * 64).
        let n = 8192usize;
        let ablocks = n.div_ceil(ABLOCK);

        // Mixed-magnitude x (recurrent_out) + gate (z_buf).
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let blk = i / ABLOCK;
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.1 * ((i % 7) as f32 + 1.0);
                sign * mag
            })
            .collect();
        let gate: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i.is_multiple_of(3) { 1.0 } else { -1.0 };
                let mag = 0.6 + 0.4 * ((i as f32) / n as f32);
                sign * mag
            })
            .collect();

        // ── Reference path: separate z_gating → quantize (all on GPU) ──
        // z_gating is in-place on `output`, so copy x into gated_dev first.
        let gate_dev_a = stream.clone_htod(&gate).unwrap();
        let gated_dev = stream.clone_htod(&x).unwrap();
        let ref_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let ref_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        dn_kernels
            .launch_z_gating(&stream, &gated_dev, &gate_dev_a, n)
            .expect("z_gating launch");
        kernels
            .launch_quantize(&stream, &gated_dev, &ref_i8_dev, &ref_ascale_dev, n)
            .expect("quantize launch");
        stream.synchronize().expect("sync");

        let mut ref_i8 = vec![0i8; n];
        let mut ref_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8_dev, &mut ref_i8).unwrap();
        stream.memcpy_dtoh(&ref_ascale_dev, &mut ref_ascale).unwrap();

        // ── Fused path: gate_silu_quantize in one kernel ──
        let x_dev_b = stream.clone_htod(&x).unwrap();
        let gate_dev_b = stream.clone_htod(&gate).unwrap();
        let fused_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let fused_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_gate_silu_quantize(
                &stream, &x_dev_b, &gate_dev_b, &fused_i8_dev, &fused_ascale_dev, n,
            )
            .expect("fused launch");
        stream.synchronize().expect("sync");

        let mut fused_i8 = vec![0i8; n];
        let mut fused_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&fused_i8_dev, &mut fused_i8).unwrap();
        stream.memcpy_dtoh(&fused_ascale_dev, &mut fused_ascale).unwrap();

        // ── Compare int8 codes (must be bit-exact) ──
        let mut i8_mismatches = 0usize;
        for i in 0..n {
            if fused_i8[i] != ref_i8[i] {
                i8_mismatches += 1;
                if i8_mismatches <= 5 {
                    eprintln!(
                        "  i8 mismatch at {i}: fused={}, ref={}, x={:.4}, gate={:.4}",
                        fused_i8[i], ref_i8[i], x[i], gate[i]
                    );
                }
            }
        }
        let mut max_scale_diff = 0f32;
        for blk in 0..ablocks {
            let diff = (fused_ascale[blk] - ref_ascale[blk]).abs();
            max_scale_diff = max_scale_diff.max(diff);
        }

        eprintln!(
            "[gate_silu_quantize] n={n}: {i8_mismatches} i8 mismatches, ascale max_diff={max_scale_diff:.4e}"
        );
        assert!(i8_mismatches == 0, "{i8_mismatches} int8 mismatches");
        assert!(max_scale_diff < 1e-5, "ascale max_diff {max_scale_diff:.4e}");
    }

    /// Issue 626 — verify the fused `gate_sigmoid_quantize_f32` kernel produces
    /// bit-identical int8 + ascale output to the separate GPU path
    /// (`launch_output_gate` → `launch_quantize`).
    #[test]
    fn test_gate_sigmoid_quantize_matches_separate() {
        const ABLOCK: usize = 16;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx.clone()).expect("compile elem");
        let attn_kernels = super::AttentionKernels::new(ctx).expect("compile attn");
        // Attention q_dim for Bonsai-27B (n_head * head_dim = 40 * 128).
        let n = 5120usize;
        let ablocks = n.div_ceil(ABLOCK);

        let x: Vec<f32> = (0..n)
            .map(|i| {
                let blk = i / ABLOCK;
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.1 * ((i % 7) as f32 + 1.0);
                sign * mag
            })
            .collect();
        let gate: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i.is_multiple_of(3) { 1.0 } else { -1.0 };
                let mag = 0.6 + 0.4 * ((i as f32) / n as f32);
                sign * mag
            })
            .collect();

        // ── Reference path: separate output_gate → quantize (all on GPU) ──
        // output_gate is in-place on `attn_out`, so copy x into gated_dev first.
        let gate_dev_a = stream.clone_htod(&gate).unwrap();
        let gated_dev = stream.clone_htod(&x).unwrap();
        let ref_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let ref_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        attn_kernels
            .launch_output_gate(&stream, &gated_dev, &gate_dev_a, n)
            .expect("output_gate launch");
        kernels
            .launch_quantize(&stream, &gated_dev, &ref_i8_dev, &ref_ascale_dev, n)
            .expect("quantize launch");
        stream.synchronize().expect("sync");

        let mut ref_i8 = vec![0i8; n];
        let mut ref_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8_dev, &mut ref_i8).unwrap();
        stream.memcpy_dtoh(&ref_ascale_dev, &mut ref_ascale).unwrap();

        // ── Fused path: gate_sigmoid_quantize in one kernel ──
        let x_dev_b = stream.clone_htod(&x).unwrap();
        let gate_dev_b = stream.clone_htod(&gate).unwrap();
        let fused_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let fused_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_gate_sigmoid_quantize(
                &stream, &x_dev_b, &gate_dev_b, &fused_i8_dev, &fused_ascale_dev, n,
            )
            .expect("fused launch");
        stream.synchronize().expect("sync");

        let mut fused_i8 = vec![0i8; n];
        let mut fused_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&fused_i8_dev, &mut fused_i8).unwrap();
        stream.memcpy_dtoh(&fused_ascale_dev, &mut fused_ascale).unwrap();

        let mut i8_mismatches = 0usize;
        for i in 0..n {
            if fused_i8[i] != ref_i8[i] {
                i8_mismatches += 1;
                if i8_mismatches <= 5 {
                    eprintln!(
                        "  i8 mismatch at {i}: fused={}, ref={}, x={:.4}, gate={:.4}",
                        fused_i8[i], ref_i8[i], x[i], gate[i]
                    );
                }
            }
        }
        let mut max_scale_diff = 0f32;
        for blk in 0..ablocks {
            let diff = (fused_ascale[blk] - ref_ascale[blk]).abs();
            max_scale_diff = max_scale_diff.max(diff);
        }

        eprintln!(
            "[gate_sigmoid_quantize] n={n}: {i8_mismatches} i8 mismatches, ascale max_diff={max_scale_diff:.4e}"
        );
        assert!(i8_mismatches == 0, "{i8_mismatches} int8 mismatches");
        assert!(max_scale_diff < 1e-5, "ascale max_diff {max_scale_diff:.4e}");
    }

    /// Issue 627 — verify the fused `rmsnorm_gate_silu_quantize_f32` kernel
    /// produces bit-identical int8 + ascale output to the separate GPU path
    /// (`launch_rmsnorm_batched` → `launch_z_gating` → `launch_quantize`).
    ///
    /// The RMSNorm tree reduction is identical to `rmsnorm_batched_f32`, the
    /// silu computation uses the same nvrtc `expf`, and the quantize absmax
    /// uses `fmaxf` (order-independent). So the result should be bit-exact.
    #[test]
    fn test_rmsnorm_gate_silu_quantize_matches_separate() {
        const ABLOCK: usize = 16;

let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx.clone()).expect("compile elem");
        let attn_kernels = super::AttentionKernels::new(ctx.clone()).expect("compile attn");
        let dn_kernels = super::DeltanetKernels::new(ctx).expect("compile dn");
        // DeltaNet config for Bonsai-27B: n_v_heads=128, head_dim=64.
        let n_v_heads = 128usize;
        let head_dim = 64usize;
        let n = n_v_heads * head_dim; // 8192
        let ablocks = n.div_ceil(ABLOCK); // 512
        let eps = 1e-5f32;

        // Mixed-magnitude x (pre-norm recurrent_out) + gate (z_buf).
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let blk = i / ABLOCK;
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.1 * ((i % 7) as f32 + 1.0);
                sign * mag
            })
            .collect();
        let gate: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i.is_multiple_of(3) { 1.0 } else { -1.0 };
                let mag = 0.6 + 0.4 * ((i as f32) / n as f32);
                sign * mag
            })
            .collect();
        // gamma (linear_norm) — per-head_dim weights, shared across heads.
        let gamma: Vec<f32> = (0..head_dim)
            .map(|j| 0.8 + 0.1 * ((j as f32) / head_dim as f32))
            .collect();

        // ── Reference path: rmsnorm_batched → z_gating → quantize (all GPU) ──
        let gamma_dev = stream.clone_htod(&gamma).unwrap();
        // rmsnorm_batched is in-place at the CUDA level (input == output buffer).
        // Both params are &CudaSlice<f32> (immutable handles), so this is safe.
        let normed_dev = stream.clone_htod(&x).unwrap();
        attn_kernels
            .launch_rmsnorm_batched(
                &stream, &normed_dev, &gamma_dev, &normed_dev, n_v_heads, head_dim, eps,
            )
            .expect("rmsnorm_batched launch");
        let gate_dev_a = stream.clone_htod(&gate).unwrap();
        dn_kernels
            .launch_z_gating(&stream, &normed_dev, &gate_dev_a, n)
            .expect("z_gating launch");
        let ref_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let ref_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();
        kernels
            .launch_quantize(&stream, &normed_dev, &ref_i8_dev, &ref_ascale_dev, n)
            .expect("quantize launch");
        stream.synchronize().expect("sync");

        let mut ref_i8 = vec![0i8; n];
        let mut ref_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&ref_i8_dev, &mut ref_i8).unwrap();
        stream.memcpy_dtoh(&ref_ascale_dev, &mut ref_ascale).unwrap();

        // ── Fused path: rmsnorm_gate_silu_quantize in one kernel ──
        let x_dev_b = stream.clone_htod(&x).unwrap();
        let gate_dev_b = stream.clone_htod(&gate).unwrap();
        let gamma_dev_b = stream.clone_htod(&gamma).unwrap();
        let fused_i8_dev = stream.alloc_zeros::<i8>(n).unwrap();
        let fused_ascale_dev = stream.alloc_zeros::<f32>(ablocks).unwrap();

        kernels
            .launch_rmsnorm_gate_silu_quantize(
                &stream, &x_dev_b, &gate_dev_b, &gamma_dev_b,
                &fused_i8_dev, &fused_ascale_dev,
                n_v_heads, head_dim, eps,
            )
            .expect("fused launch");
        stream.synchronize().expect("sync");

        let mut fused_i8 = vec![0i8; n];
        let mut fused_ascale = vec![0f32; ablocks];
        stream.memcpy_dtoh(&fused_i8_dev, &mut fused_i8).unwrap();
        stream.memcpy_dtoh(&fused_ascale_dev, &mut fused_ascale).unwrap();

        // ── Compare int8 codes (must be bit-exact) ──
        let mut i8_mismatches = 0usize;
        for i in 0..n {
            if fused_i8[i] != ref_i8[i] {
                i8_mismatches += 1;
                if i8_mismatches <= 5 {
                    eprintln!(
                        "  i8 mismatch at {i}: fused={}, ref={}, x={:.4}, gate={:.4}",
                        fused_i8[i], ref_i8[i], x[i], gate[i]
                    );
                }
            }
        }
        let mut max_scale_diff = 0f32;
        for blk in 0..ablocks {
            let diff = (fused_ascale[blk] - ref_ascale[blk]).abs();
            max_scale_diff = max_scale_diff.max(diff);
        }

        eprintln!(
            "[rmsnorm_gate_silu_quantize] n_v_heads={n_v_heads}, head_dim={head_dim}: {i8_mismatches} i8 mismatches, ascale max_diff={max_scale_diff:.4e}"
        );
        assert!(i8_mismatches == 0, "{i8_mismatches} int8 mismatches");
        assert!(max_scale_diff < 1e-5, "ascale max_diff {max_scale_diff:.4e}");
    }

    /// Issue 697 — GPU argmax must match the CPU first-index tie-break argmax
    /// EXACTLY (strict `>`, first index wins among equal maxima). Covers:
    /// negatives, duplicated maxima, and a vocab-scale length.
    #[test]
    fn test_argmax_first_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = ElementwiseKernels::new(ctx).expect("compile");

        // Vocab-scale length (248320 for Bonsai-27B) with LCG values + planted
        // duplicates of the max at several indices.
        let n = 248_320usize;
        let mut state: u32 = 0xFEEDF00D;
        let mut v: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65535.0 - 0.5) * 4.0
            })
            .collect();
        // Plant the max (with duplicates — the FIRST index must win). Plant a
        // value strictly above every LCG value so the planted indices are the
        // true maxima.
        let lcg_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let planted = lcg_max + 1.0f32;
        v[100_000] = planted;
        v[200_001] = planted;
        v[3] = planted;
        v[77] = planted;
        // Also plant a -0.0 (must not beat a +0.0 at a LOWER index).
        v[50] = -0.0f32;
        v[150_000] = 0.0f32;

        // CPU reference: strict `>`, first index wins.
        let cpu = {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &l) in v.iter().enumerate() {
                if l > best_v {
                    best_v = l;
                    best = i;
                }
            }
            best
        };
        assert_eq!(cpu, 3, "test setup: expected the planted first max at 3");

        let v_dev = stream.clone_htod(&v).unwrap();
        let mut result_dev = stream.alloc_zeros::<u64>(1).unwrap();
        for _ in 0..3 {
            // Repeat 3× — the caller must zero the buffer per launch; the
            // decode path does this via a capturable memset node.
            stream.memset_zeros(&mut result_dev).unwrap();
            kernels
                .launch_argmax_first(&stream, &v_dev, n, &result_dev)
                .expect("launch");
            stream.synchronize().expect("sync");
            let mut packed = [0u64; 1];
            stream.memcpy_dtoh(&result_dev, &mut packed).unwrap();
            let idx = (!(packed[0] as u32)) as usize;
            assert_eq!(idx, cpu, "GPU argmax diverged from CPU first-index argmax");
        }
        eprintln!("[argmax_first] n={n}: gpu == cpu == {cpu} (with duplicate maxima)");
    }
}
