//! Issue 742 T9.2 — full 64-layer GPU decode forward for the dense
//! `dbirks/Qwen3.8-27B-W4A16-AutoRound` (qwen35 arch, Q4_K_M GGUF) on cudarc.
//!
//! Composes the T9.3 dp4a dequant GEMV kernels (Q4_K/Q6_K-resident weights,
//! 742 GB/s — the 46.9 tok/s base ceiling) with the existing qwen3_5-family
//! decode kernels (`cudarc_kernels::{deltanet, attention, Elementwise}` —
//! shipped for the Bonsai ternary twin, parametric in every dim this model
//! needs) into a single-stream whole-model decode: embedding row dequant →
//! 48 GDN layers + 16 full-attention layers (interval 4) → SwiGLU MLP →
//! final norm → Q6_K lm_head GEMV → device-side argmax (8-byte readback).
//!
//! # Weight residency
//!
//! Quantized tensors upload VERBATIM (the Batch-37 rule: on-disk format ==
//! device format, never dequant-requantize) — 14.72 GiB for the 497-tensor
//! decode set. Small F32 tensors (norms, conv1d, ssm_a/dt_bias, alpha/beta)
//! upload as f32 directly; F16/BF16 norms convert host-side (trivial sizes).
//!
//! # Kernel provenance (consume, not rebuild)
//!
//! - `conv1d_f32` / `beta_decay_f32` / `expand_and_l2_normalize_heads_f32` /
//!   `recurrence_f32_parallel` / `z_gating_f32` — `cudarc_kernels::deltanet`,
//!   bit-matched to `riir_infer_core::deltanet::forward` (the Issue-594-hardened
//!   CPU reference: tiled-modulo head broadcast, per-head gated RMSNorm,
//!   a_log = -exp(A_log) direct use).
//! - `rope_partial_f32` / `split_qg_f32` / `rmsnorm_batched_f32` /
//!   `kv_cache_append_f32` / `attention_decode_f32` / `output_gate_f32` —
//!   `cudarc_kernels::attention` (partial RoPE rotary_dim=64 of 256, theta
//!   1e7; gated-Q interleaved split; GQA 24Q/4KV).
//! - `rmsnorm_f32` / `swiglu_f32` / `residual_add_f32` / `argmax_first_f32`
//!   — `cudarc_kernels::Elementwise`.
//! - `qwen38_quant_x_q8` / `qwen38_gemv_q4k_q8x` / `qwen38_gemv_q6k_q8x` —
//!   the T9.3 kernels (moved here from the test file; the test keeps the
//!   f32 ladder + sweep, this module takes the dp4a production path).
//! - `qwen38_rmsnorm_quant_x_q8` — Issue 742 T9.7 follow-on (Bench 735):
//!   fused rmsnorm→q8-quantize for the 129 plain-rmsnorm sites/token (all
//!   feed ONLY the quantizer). Bit-identical to the `rmsnorm_f32` +
//!   `qwen38_quant_x_q8` pair (phases 1-3 verbatim rmsnorm_f32; phase 4 the
//!   quant ops on the exact expression it stores). −129 launches/token,
//!   −77 µs/token kernel exec (Bench 735). `QWEN38_RNQ_FUSED=0` hatch.
//! - `qwen38_rmsnorm_quant_x_q8_s5120` — T9.7c (Bench 736): constexpr
//!   specialized twin of the fused kernel for dim=5120 (all 129 sites) —
//!   NITER=20/GTRIPS=2 compile-time so both data loops fully unroll (the
//!   generic kernel's runtime-bound loops serialize the strided loads
//!   through ~5 DRAM round-trips; its 7.36 µs/site is memory latency,
//!   ~63x the bandwidth floor). Verbatim numerics — bit-identical.
//!   `QWEN38_RNQ_FAST=0` hatch.
//! - `qwen38_rmsnorm_zgate_quant_x_q8` — Issue 742 lever-2 (Bench 737):
//!   fused per-head RMSNorm + silu z-gate + q8-quantize for the GDN
//!   post-recurrence chain — replaces the `rmsnorm_batched_f32` →
//!   `z_gating_f32` → `qwen38_quant_x_q8` trio at the 48 GDN layers
//!   (144 launches/token → 48; each eliminated kernel saves its whole
//!   ~2-3.5 µs invocation floor per the Bench-736 floor model).
//!   Bit-identical (phase 1 verbatim rmsnorm_batched_f32; phase 2 the exact
//!   gated bits; phase 3 qwen38_quant_x_q8's ops). `QWEN38_ZGQ_FUSED=0` hatch.
//! - `qwen38_residual_norm_quant_x_q8_s5120[_rows]` — Issue 755 (MTPLX PR
//!   #335 row 48 + corpus B63 `deferred-residual-fused-into-next-layer-norm`):
//!   the fused layer-BOUNDARY chain — residual_add + snapshot copy + the
//!   NEXT site's rmsnorm+q8-quantize in ONE kernel (128 sites/token decode,
//!   ×p in verify; −2 launches + no x DRAM round-trip per site).
//!   Bit-identical by construction (same single f32 add, same stored bits,
//!   verbatim reduction/quant phases). `QWEN38_RES_NQ_FUSED=0` hatch.
//!
//! Rides `ternary_gemv_cuda_raw` (dep:cudarc; CUDA-only, non-macOS).
#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
#![allow(clippy::too_many_arguments)]

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use riir_infer_core::gguf_loader::{GgufFile, GgmlType};

use crate::cudarc_kernels::attention::AttentionKernels;
use crate::cudarc_kernels::attention_score_mma::AttentionScoreMmaKernels;
use crate::cudarc_kernels::deltanet::DeltanetKernels;
use crate::cudarc_kernels::ElementwiseKernels;
use crate::qwen38_verify_mma::VerifyMmaKernels;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA source: the T9.3 dp4a kernels + the T9.2 additions (embedding row
// dequant, small f32 GEMV, elementwise copy).
// ─────────────────────────────────────────────────────────────────────────────

pub const QWEN38_DENSE_CUDA_SRC: &str = r#"
// Exact IEEE f16 -> f32 conversion (NVRTC has no cuda_fp16.h in its default
// include path). Case-based so subnormals are exact (Issue 593 class).
__device__ __forceinline__ float f16_to_f32_exact(unsigned short h) {
    const unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    const unsigned int exp = (h >> 10) & 0x1Fu;
    const unsigned int frac = h & 0x03FFu;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign);
        const float v = (float)frac * 5.9604644775390625e-08f;
        return sign ? -v : v;
    }
    if (exp == 31) {
        return __int_as_float(sign | 0x7F800000u | (frac << 13));
    }
    return __int_as_float(sign | ((exp + 112u) << 23) | (frac << 13));
}

extern "C" __global__ void qwen38_quant_x_q8(
    const float* __restrict__ x,
    signed char* __restrict__ xq,
    float* __restrict__ xs,
    int* __restrict__ xsum,
    const int n)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = b << 4;
    if (base >= n) return;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[base + i]));
    const float s = amax > 0.f ? amax / 127.f : 1.f;
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int q = (int)__float2int_rn(x[base + i] / s);
        q = max(-127, min(127, q));
        xq[base + i] = (signed char)q;
        sum += q;
    }
    xs[b] = s;
    xsum[b] = sum;
}

// Issue 742 T9.7 follow-on (Bench 734 ranked lever 1): fused RMSNorm +
// q8 quantize for the 129 plain-rmsnorm decode sites (every one feeds ONLY
// the quantizer — GDN/attn input_norm, MLP post_attn_norm, final output_norm;
// the batched norms feed f32 consumers and stay unfused). Replaces the
// `rmsnorm_f32` + `qwen38_quant_x_q8` pair with one kernel: same reduction
// tree (phases 1-3 are VERBATIM `rmsnorm_f32` — the strided accumulation and
// the 256-thread smem tree, so `inv_rms` is bit-identical), then phase 4
// quantizes per-16 groups with `qwen38_quant_x_q8`'s exact ops applied to
// `v = input[j] * inv_rms * gamma[j]` — the exact expression AND multiply
// order `rmsnorm_f32` stores to x_norm (f32 stores/loads are bit-preserving,
// so the register value equals what the quantizer would have read back).
// Eliminates one kernel launch + the x_norm DRAM round-trip (~40 KB) per site.
extern "C" __global__ void qwen38_rmsnorm_quant_x_q8(
    const float* __restrict__ input,   // [dim] raw input
    const float* __restrict__ gamma,   // [dim]
    signed char* __restrict__ xq,      // [dim]
    float* __restrict__ xs,            // [dim/16]
    int* __restrict__ xsum,            // [dim/16]
    float inv_dim,
    float eps,
    int dim,
    int groups)                        // dim / 16
{
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // 256 (launcher contract)

    // ── Phase 1: strided accumulation of x² (VERBATIM rmsnorm_f32) ──
    float partial_sq = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        float x = input[i];
        partial_sq += x * x;
    }

    // ── Phase 2: shared memory parallel reduction (VERBATIM rmsnorm_f32) ──
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

    // ── Phase 3: compute inv_rms (VERBATIM rmsnorm_f32) ──
    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    // ── Phase 4: normalize + quantize per-16 group (qwen38_quant_x_q8 ops
    // on the exact value rmsnorm_f32 would store). vals[] stays in registers
    // (compile-time 16 + unroll — the Issue-706/Batch-49 lesson). Guard: only
    // full groups — the launcher asserts dim % 16 == 0 (5120 at every site).
    for (int g = tid; g < groups; g += block_size) {
        const int base = g << 4;
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const float v = input[base + i] * inv_rms * gamma[base + i];
            vals[i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float s = amax > 0.f ? amax / 127.f : 1.f;
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int q = (int)__float2int_rn(vals[i] / s);
            q = max(-127, min(127, q));
            xq[base + i] = (signed char)q;
            sum += q;
        }
        xs[g] = s;
        xsum[g] = sum;
    }
}

// Issue 742 T9.7c (Bench 736): constexpr-specialized twin of the fused
// kernel above for dim=5120 — the dim of ALL 129 production sites. NITER=20
// and GTRIPS=2 are compile-time, so both data loops FULLY UNROLL: the
// generic kernel's runtime-bound loops serialize the strided loads through
// ~5 DRAM round-trips (its measured 7.36 µs/site is memory LATENCY, ~63x
// the bandwidth floor of the ~60 KB it touches); unrolling issues every
// load back-to-back (~1-2 rounds in flight).
//
// BIT-IDENTICAL by construction: the per-lane ascending `partial_sq += x*x`
// chain, the 8-step 256-thread smem tree, the inv_rms expression, and the
// phase-4 group mapping + expression are all VERBATIM (unrolling preserves
// statement order — the FMA chain per lane is the same sequence). No
// cross-block state — trivially CUDA-graph safe (no barrier/reset protocol,
// unlike a multiblock design).
//
// Considered and rejected: stashing x in a 20 KB smem buffer during phase 1
// so phase 4 skips the input re-read — thread-per-group reads sx[16g+i]
// which is a 16-WAY bank conflict (~4096 extra smem transactions ≈ the
// entire win). The L2 re-read is cheaper.
extern "C" __global__ void qwen38_rmsnorm_quant_x_q8_s5120(
    const float* __restrict__ input,   // [5120] raw input
    const float* __restrict__ gamma,   // [5120]
    signed char* __restrict__ xq,      // [5120]
    float* __restrict__ xs,            // [320]
    int* __restrict__ xsum,            // [320]
    float inv_dim,
    float eps)
{
    const int tid = threadIdx.x;       // blockDim.x == 256 (launcher contract)
    constexpr int DIM = 5120;
    constexpr int BLOCK = 256;         // DIM % BLOCK == 0 -> no tail guard
    constexpr int NITER = DIM / BLOCK;             // 20
    constexpr int GROUPS = DIM / 16;               // 320
    constexpr int GTRIPS = (GROUPS + BLOCK - 1) / BLOCK;  // 2

    // ── Phase 1: strided accumulation of x² (verbatim lane mapping + ascending
    // per-lane order; unrolled so all 20 loads are in flight) ──
    float partial_sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < NITER; ++k) {
        const int i = tid + k * BLOCK;
        const float x = input[i];
        partial_sq += x * x;
    }

    // ── Phase 2: shared memory parallel reduction (VERBATIM) ──
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

    // ── Phase 3: compute inv_rms (VERBATIM) ──
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

    // ── Phase 4: normalize + quantize per-16 group (expression verbatim;
    // group trips unrolled so both groups' loads issue together) ──
    #pragma unroll
    for (int r = 0; r < GTRIPS; ++r) {
        const int g = tid + r * BLOCK;
        if (g < GROUPS) {
            const int base = g << 4;
            float vals[16];
            float amax = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const float v = input[base + i] * inv_rms * gamma[base + i];
                vals[i] = v;
                amax = fmaxf(amax, fabsf(v));
            }
            const float s = amax > 0.f ? amax / 127.f : 1.f;
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int q = (int)__float2int_rn(vals[i] / s);
                q = max(-127, min(127, q));
                xq[base + i] = (signed char)q;
                sum += q;
            }
            xs[g] = s;
            xsum[g] = sum;
        }
    }
}

// Issue 742 lever-2 (Bench 737): fused per-head RMSNorm + silu z-gate +
// q8-quantize for the GDN post-recurrence chain — replaces the
// `rmsnorm_batched_f32` (attention module) → `z_gating_f32` (deltanet
// module) → `qwen38_quant_x_q8` trio at the 48 GDN layers (3 kernels × 48
// = 144 launches/token → 48). The Bench-736 floor model: each eliminated
// tiny kernel saves its WHOLE ~2-3.5 µs device-side invocation floor.
//
// BIT-IDENTICAL by construction (all three parent modules compile under the
// identical nvrtc options — arch sm_89 + defaults — so the same source text
// lowers to the same instructions):
// - Phase 1 (strided sum-of-squares + 256-thread smem tree + inv_rms
//   expression) is VERBATIM `rmsnorm_batched_f32` → the same inv_rms bits
//   (incl. the FMA-contracted `partial_sq += x*x` chain under the default
//   -fmad=true, identical in both modules).
// - The gated value: `rmsnorm_batched_f32` stores `input*inv_rms*gamma`
//   (left-to-right; f32 store/load is bit-preserving), `z_gating_f32` reads
//   it back and computes `out = stored * silu(z)` with silu = z/(1+expf(-z))
//   — the same nvrtc expf. The fused form computes the same expression in
//   registers; IEEE multiply is commutative, so `x_norm * silu` is the same
//   bits the trio stores into rec_normed.
// - The quantize step applies `qwen38_quant_x_q8`'s EXACT ops to `val`:
//   amax via butterfly shfl_xor within each 16-lane group (fmaxf is
//   order-independent → the same amax as the sequential chain),
//   s = amax>0 ? amax/127 : 1 (the same bits), q = __float2int_rn(val/s)
//   clamped [-127,127] (the same ops), and the group integer sum via the
//   same butterfly (integer add is exactly associative → the same xsum).
// Grid: n_heads blocks (one per head), 256 threads. Constraints (launcher
// asserts): head_dim % 16 == 0 keeps every 16-lane group inside one head and
// inside the active lane region; head_dim <= 256. No cross-block state —
// trivially CUDA-graph safe.
extern "C" __global__ void qwen38_rmsnorm_zgate_quant_x_q8(
    const float* __restrict__ input,   // [n_heads * head_dim] — pre-norm (rec_out)
    const float* __restrict__ z,       // [n_heads * head_dim] — gate (z_buf)
    const float* __restrict__ gamma,   // [head_dim]
    signed char* __restrict__ xq,      // [n_heads * head_dim]
    float* __restrict__ xs,            // [n_heads * head_dim / 16]
    int* __restrict__ xsum,            // [n_heads * head_dim / 16]
    const float inv_dim,
    const float eps,
    const int head_dim)
{
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // 256 (launcher contract)
    const int base = head_idx * head_dim;

    // ── Phase 1: VERBATIM rmsnorm_batched_f32 sum-of-squares tree ──
    float partial_sq = 0.0f;
    for (int i = tid; i < head_dim; i += block_size) {
        float x = input[base + i];
        partial_sq += x * x;
    }
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

    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    // ── Phase 2: the gated value — the exact bits the trio produces ──
    float val = 0.0f;
    if (tid < head_dim) {
        float x_norm = input[base + tid] * inv_rms * gamma[tid];
        float zv = z[base + tid];
        float silu = zv / (1.0f + expf(-zv));
        val = x_norm * silu;
    }

    // ── Phase 3: quantize per-16 group (qwen38_quant_x_q8's exact ops).
    // All shfl calls execute for every lane of the warp (unguarded — no
    // divergence); head_dim % 16 == 0 keeps each 16-lane group entirely
    // inside or outside the active region. ──
    float a = fabsf(val);
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 1));

    const float s = a > 0.f ? a / 127.f : 1.f;
    int q = 0;
    if (tid < head_dim) {
        q = (int)__float2int_rn(val / s);
        q = max(-127, min(127, q));
        xq[base + tid] = (signed char)q;
    }
    int sum = q;
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);

    const int group = tid >> 4;
    const int groups_per_head = head_dim >> 4;
    if ((tid & 15) == 0 && tid < head_dim) {
        xs[head_idx * groups_per_head + group] = s;
        xsum[head_idx * groups_per_head + group] = sum;
    }
}

extern "C" __global__ void qwen38_gemv_q4k_q8x(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xs,
    const int* __restrict__ xsum,
    float* __restrict__ y,
    const int m, const int n, const int blocks_per_row)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= m) return;
    const int lane = threadIdx.x & 31;
    const unsigned char* wrow = w + (size_t)warp_id * (size_t)blocks_per_row * 144;
    const int n_sub = n >> 5;
    float acc = 0.f;
    for (int sb = lane; sb < n_sub; sb += 32) {
        const int blk = sb >> 3;
        const int j = sb & 7;
        const unsigned char* b = wrow + (size_t)blk * 144;
        const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b));
        const float dmin = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 2));
        const unsigned char* s = b + 4;
        float sc, mn;
        if (j < 4) {
            sc = (float)(s[j] & 63);
            mn = (float)(s[j + 4] & 63);
        } else {
            sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
            mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        }
        const unsigned char* q = b + 16 + ((j >> 1) << 5);
        const int xbase = sb << 5;
        const unsigned int nib_shift = (j & 1) ? 4 : 0;
        float sub = 0.f;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            const int b16 = ((xbase >> 4) + half);
            int dot = 0;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const unsigned int wu = *reinterpret_cast<const unsigned int*>(q + half * 16 + u * 4);
                const unsigned int nib = (wu >> nib_shift) & 0x0F0F0F0Fu;
                const unsigned int nib8 = __vsub4(nib, 0x08080808u);
                const int xw = *reinterpret_cast<const int*>(xq + xbase + half * 16 + u * 4);
                dot = __dp4a((int)nib8, xw, dot);
            }
            const int isum = xsum[b16];
            sub += xs[b16] * ((d * sc) * (float)(dot + 8 * isum) - (dmin * mn) * (float)isum);
        }
        acc += sub;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

extern "C" __global__ void qwen38_gemv_q6k_q8x(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xs,
    const int* __restrict__ xsum,
    float* __restrict__ y,
    const int m, const int n, const int blocks_per_row)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= m) return;
    const int lane = threadIdx.x & 31;
    const unsigned char* wrow = w + (size_t)warp_id * (size_t)blocks_per_row * 210;
    const int n_grp = n >> 4;
    float acc = 0.f;
    for (int g = lane; g < n_grp; g += 32) {
        const int blk = g >> 4;
        const int h = (g >> 3) & 1;
        const int slot = (g >> 1) & 3;
        const int l0 = (g & 1) << 4;
        const unsigned char* b = wrow + (size_t)blk * 210;
        const unsigned char* ql = b + ((h << 6) + ((slot & 1) << 5));
        const unsigned char* qh = b + 128 + (h << 5);
        const float sc = (float)(signed char)b[192 + (h << 3) + (g & 1) + (slot << 1)];
        const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 208));
        const int xbase = g << 4;
        const unsigned int nib_shift = (slot < 2) ? 0 : 4;
        const unsigned int qh_shift = (unsigned int)(slot << 1);
        int dot = 0;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            #define Q6_U32(p) (((unsigned int)*(const unsigned short*)(p)) | \
                              ((unsigned int)*(const unsigned short*)((p) + 2) << 16))
            const unsigned int qlw = Q6_U32(ql + l0 + u * 4);
            const unsigned int qhw = Q6_U32(qh + l0 + u * 4);
            const unsigned int assembled =
                ((qlw >> nib_shift) & 0x0F0F0F0Fu) | (((qhw >> qh_shift) & 0x03030303u) << 4);
            const unsigned int q6s = __vsub4(assembled, 0x20202020u);
            const int xw = *reinterpret_cast<const int*>(xq + xbase + u * 4);
            dot = __dp4a((int)q6s, xw, dot);
            #undef Q6_U32
        }
        const int isum = xsum[g];
        acc += xs[g] * ((d * sc) * (float)dot);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// T9.2: dequantize one Q4_K row of the token embedding into f32 (the
// embedding "lookup" — one row = the token's hidden vector). Element mapping
// identical to qwen38_gemv_q4k (pinned to dequantize_row_q4_k).
extern "C" __global__ void qwen38_dequant_q4k_row(
    const unsigned char* __restrict__ w,   // [rows * blocks_per_row * 144]
    float* __restrict__ out,               // [n]
    const int row,
    const int n,
    const int blocks_per_row)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 144;
    const int j = (e >> 5) & 7;   // sub-block 0..7 within the 256-block
    const int blk = e >> 8;       // 256-element block
    const unsigned char* b = wrow + (size_t)blk * 144;
    const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b));
    const float dmin = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 2));
    const unsigned char* s = b + 4;
    float sc, mn;
    if (j < 4) {
        sc = (float)(s[j] & 63);
        mn = (float)(s[j + 4] & 63);
    } else {
        sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
    const unsigned char* q = b + 16 + ((j >> 1) << 5);
    const int l = e & 31;
    const int nib = (j & 1) ? (q[l] >> 4) : (q[l] & 0x0F);
    out[e] = (d * sc) * (float)nib - dmin * mn;
}

// T9.5 graph path: device-pointer variant — row read from a device buffer at
// kernel runtime (CUDA-graph capture: the scalar would bake into the graph).
extern "C" __global__ void qwen38_dequant_q4k_row_devpos(
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    const int* __restrict__ row_dev,
    const int n,
    const int blocks_per_row)
{
    const int row = *row_dev;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 144;
    const int j = (e >> 5) & 7;
    const int blk = e >> 8;
    const unsigned char* b = wrow + (size_t)blk * 144;
    const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b));
    const float dmin = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 2));
    const unsigned char* sc = b + 4;
    float sc_v, mn;
    if (j < 4) {
        sc_v = (float)(sc[j] & 63);
        mn = (float)(sc[j + 4] & 63);
    } else {
        sc_v = (float)((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4));
        mn = (float)((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
    const unsigned char* q = b + 16 + ((j >> 1) << 5);
    const int l = e & 31;
    const int nib = (j & 1) ? (q[l] >> 4) : (q[l] & 0x0F);
    out[e] = (d * sc_v) * (float)nib - dmin * mn;
}

// T9.2: elementwise copy (residual snapshot + trace taps).
extern "C" __global__ void qwen38_copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ===========================================================================
// Issue 742 T9.9 - the p-ROW batched Q4_K GEMV, shape B (MEASURED NEGATIVE
// 2026-08-23: 45-103 GB/s at the production shapes vs the strict arm's
// 322-492 - the broadcast weight loads + 8 serial row blocks per sub-block
// destroy memory-level parallelism; kept behind QWEN38_VERIFY_GEMV=shapeb
// as the A/B artifact. The design note below records the intent.):
// lane = (input row r, half) FIXED per lane; one warp owns R_OUT consecutive
// OUTPUT rows. The weight words are BROADCAST loads (same address across
// lanes - 1 transaction each) shared by all 16 input rows; the x word and
// the two correction scalars are loaded ONCE per lane per sub-block and
// reused for all R_OUT rows. This is the x-side L1 fix: the v2 warp-per-row
// shape re-reads the full 16-row x set once per output row (459 GB of L1
// traffic per chunk - the measured ~450 GB/s plateau); shape B shares every
// x load R_OUT-fold, dropping the x-side to 459/R_OUT GB.
//
// NUMERICS: per (input row r, output row, sub-block sb) the terms are
// bit-identical to the single-row kernel (same weight bytes -> the same
// nibble words, the same 4-dp4a chain per half, the same correction
// expression and scalar values). The SUMMATION ORDER over sub-blocks
// differs (sequential over all sb + a final half-pair merge, vs the single
// kernel's lane-strided partials + butterfly) - the reassociation class,
// the T1.8-accepted tolerance (near-tie argmax flips ~1/512 measured on the
// Bonsai chunked path). The strict bit-identical arm stays available via
// QWEN38_VERIFY_GEMV=rows (the v2 kernel above... retained as
// qwen38_gemv_q4k_q8x_rows_strict).
// ===========================================================================
#define QW38_ROWS_B_ROUT 8

extern "C" __global__ void qwen38_gemv_q4k_q8x_rows(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16]
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int row0 = warp_id * QW38_ROWS_B_ROUT;
    if (row0 >= m) return;
    const int nrows = (m - row0) < QW38_ROWS_B_ROUT ? (m - row0) : QW38_ROWS_B_ROUT;
    const int lane = threadIdx.x & 31;
    const int r = lane >> 1;        // input row this lane owns
    const int half = lane & 1;      // sub-word half this lane owns
    const int n_sub = n >> 5;
    const int groups = n >> 4;

    float acc[QW38_ROWS_B_ROUT];
#pragma unroll
    for (int o = 0; o < QW38_ROWS_B_ROUT; o++) acc[o] = 0.f;

    for (int sb = 0; sb < n_sub; sb++) {
        const int blk = sb >> 3;
        const int j = sb & 7;
        const int xbase = sb << 5;
        const unsigned int nib_shift = (j & 1) ? 4 : 0;
        // per-lane x word + scalars (loaded ONCE, reused by all nrows rows)
        int4 xw4 = make_int4(0, 0, 0, 0);
        float xs_v = 0.f;
        int isum = 0;
        if (r < p) {
            xw4 = *reinterpret_cast<const int4*>(xq + (size_t)r * n + xbase + half * 16);
            const int b16 = r * groups + ((xbase >> 4) + half);
            xs_v = xs[b16];
            isum = xsum[b16];
        }
        const int* xw = reinterpret_cast<const int*>(&xw4);
        // the R output rows: broadcast weight words + scales, shared by all lanes
        for (int o = 0; o < nrows; o++) {
            const int row = row0 + o;
            const unsigned char* b = w + (size_t)row * (size_t)blocks_per_row * 144
                                    + (size_t)blk * 144;
            const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b));
            const float dmin = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 2));
            const unsigned char* s = b + 4;
            float sc, mn;
            if (j < 4) {
                sc = (float)(s[j] & 63);
                mn = (float)(s[j + 4] & 63);
            } else {
                sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
                mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
            }
            const unsigned char* q = b + 16 + ((j >> 1) << 5);
            // this lane's half's 4 words (broadcast across lanes)
            int dot = 0;
            {
                const int4 w4 = *reinterpret_cast<const int4*>(q + half * 16);
                const int* ww = reinterpret_cast<const int*>(&w4);
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const unsigned int nib =
                        ((unsigned int)ww[u] >> nib_shift) & 0x0F0F0F0Fu;
                    dot = __dp4a((int)__vsub4(nib, 0x08080808u), xw[u], dot);
                }
            }
            const float sub =
                xs_v * ((d * sc) * (float)(dot + 8 * isum) - (dmin * mn) * (float)isum);
            acc[o] += sub;
        }
    }
    // merge the two half-lanes per input row; even lanes hold the row sums
#pragma unroll
    for (int o = 0; o < QW38_ROWS_B_ROUT; o++) {
        float a = acc[o];
        a += __shfl_xor_sync(0xFFFFFFFFu, a, 1);
        if (!(lane & 1) && r < p) {
            y[(size_t)r * m + row0 + o] = a;
        }
    }
}

// ===========================================================================
// Issue 742 T9.9 - the p-ROW batched GEMV family (the Q4_K verify port).
//
// R=2 ROW-BLOCKED shape (v2, the scalar-traffic fix): one warp owns TWO
// consecutive output rows. The p input rows ride the inner loop with the
// 2x8 dequantized weight words in registers (dequant ONCE per sub-block for
// all p rows) and the x-side words loaded as one 16-byte int4 per half.
// The per-(row, sub-block) correction scalars xs/xsum are loaded ONCE and
// shared by BOTH rows - halving the dominant scattered-scalar sector
// traffic (the v1 single-row-warp shape measured 213-338 GB/s, scalar-
// bound; the x-side scalars are ~8x the weight bytes in L1 sectors at p=16
// when unshared). Weights are read exactly once for all p rows.
//
// BIT-IDENTITY per (row r, output row): the sub-block iteration is the same
// lane-strided order, the per-half dot is the same 4-dp4a chain over the
// same nibble values, the correction expression is textually identical, and
// acc[r] accumulates in the same sub-block order with the same final
// butterfly - so every output element equals the single-row kernel's bits.
// ===========================================================================
extern "C" __global__ void qwen38_gemv_q4k_q8x_rows_strict(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16]
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int warp2 = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // row pair id
    const int row0 = warp2 * 2;
    if (row0 >= m) return;
    const int row1 = (row0 + 1 < m) ? row0 + 1 : row0;  // m even (launcher contract)
    const int lane = threadIdx.x & 31;
    const unsigned char* wrow0 = w + (size_t)row0 * (size_t)blocks_per_row * 144;
    const unsigned char* wrow1 = w + (size_t)row1 * (size_t)blocks_per_row * 144;
    const int n_sub = n >> 5;
    const int groups = n >> 4;
    float acc0[16];
    float acc1[16];
#pragma unroll
    for (int r = 0; r < 16; r++) { acc0[r] = 0.f; acc1[r] = 0.f; }
    for (int sb = lane; sb < n_sub; sb += 32) {
        const int blk = sb >> 3;
        const int j = sb & 7;
        const unsigned char* b0 = wrow0 + (size_t)blk * 144;
        const unsigned char* b1 = wrow1 + (size_t)blk * 144;
        // Issue 742 T9.16 - every weight-side load below is __ldcs
        // (evict-first streaming): weights are read EXACTLY ONCE per chunk
        // (registers hold them for all p rows), so letting the ~7.6 GB/chunk
        // weight stream ALLOCATE normally in L1 evicts the ~121 KB x/xs/xsum
        // working set between re-reads - the measured 322-492 GB/s p=16
        // plateau (T9.9). Values are untouched (cache policy only).
        const float d0 =
            f16_to_f32_exact(__ldcs(reinterpret_cast<const unsigned short*>(b0)));
        const float dmin0 =
            f16_to_f32_exact(__ldcs(reinterpret_cast<const unsigned short*>(b0 + 2)));
        const float d1 =
            f16_to_f32_exact(__ldcs(reinterpret_cast<const unsigned short*>(b1)));
        const float dmin1 =
            f16_to_f32_exact(__ldcs(reinterpret_cast<const unsigned short*>(b1 + 2)));
        const unsigned char* s0 = b0 + 4;
        const unsigned char* s1 = b1 + 4;
        float sc0, mn0, sc1, mn1;
        if (j < 4) {
            sc0 = (float)(__ldcs((const unsigned char*)(s0 + j)) & 63);
            mn0 = (float)(__ldcs((const unsigned char*)(s0 + j + 4)) & 63);
            sc1 = (float)(__ldcs((const unsigned char*)(s1 + j)) & 63);
            mn1 = (float)(__ldcs((const unsigned char*)(s1 + j + 4)) & 63);
        } else {
            sc0 = (float)((__ldcs((const unsigned char*)(s0 + j + 4)) & 0x0F) |
                          ((__ldcs((const unsigned char*)(s0 + j - 4)) >> 6) << 4));
            mn0 = (float)((__ldcs((const unsigned char*)(s0 + j + 4)) >> 4) |
                          ((__ldcs((const unsigned char*)(s0 + j)) >> 6) << 4));
            sc1 = (float)((__ldcs((const unsigned char*)(s1 + j + 4)) & 0x0F) |
                          ((__ldcs((const unsigned char*)(s1 + j - 4)) >> 6) << 4));
            mn1 = (float)((__ldcs((const unsigned char*)(s1 + j + 4)) >> 4) |
                          ((__ldcs((const unsigned char*)(s1 + j)) >> 6) << 4));
        }
        const unsigned char* q0 = b0 + 16 + ((j >> 1) << 5);
        const unsigned char* q1 = b1 + 16 + ((j >> 1) << 5);
        const int xbase = sb << 5;
        const unsigned int nib_shift = (j & 1) ? 4 : 0;
        unsigned int nib0[8];
        unsigned int nib1[8];
#pragma unroll
        for (int half = 0; half < 2; half++) {
            // T9.16 - one __ldcs int4 per (row, half) replaces the 4 u32
            // loads (the 16 nibble words are 4 consecutive u32s; q0/q1 are
            // 16B-aligned: 144-byte blocks + 32B sub-block offsets). Same
            // words, same order - bit-identical.
            const int4 w40 = __ldcs(reinterpret_cast<const int4*>(q0 + half * 16));
            const int4 w41 = __ldcs(reinterpret_cast<const int4*>(q1 + half * 16));
            const unsigned int* ww0 = reinterpret_cast<const unsigned int*>(&w40);
            const unsigned int* ww1 = reinterpret_cast<const unsigned int*>(&w41);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                nib0[half * 4 + u] =
                    __vsub4((ww0[u] >> nib_shift) & 0x0F0F0F0Fu, 0x08080808u);
                nib1[half * 4 + u] =
                    __vsub4((ww1[u] >> nib_shift) & 0x0F0F0F0Fu, 0x08080808u);
            }
        }
        const int b16_pair = xbase >> 4;  // == 2*sb — even index, 8B-aligned pair
#pragma unroll
        for (int r = 0; r < 16; r++) {
            if (r < p) {
                const signed char* xr = xq + (size_t)r * n;
                // ONE coalesced float2 per buffer covers BOTH halves' scalars
                // (b16 = 2*sb and 2*sb+1 are adjacent) — halves use .x/.y.
                const float2 xs2 =
                    *reinterpret_cast<const float2*>(xs + r * groups + b16_pair);
                const float2 is2 =
                    *reinterpret_cast<const float2*>(xsum + r * groups + b16_pair);
                float sub0 = 0.f;
                float sub1 = 0.f;
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int4 xw4 =
                        *reinterpret_cast<const int4*>(xr + xbase + half * 16);
                    const int* xw = reinterpret_cast<const int*>(&xw4);
                    int dot0 = 0;
                    int dot1 = 0;
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        dot0 = __dp4a((int)nib0[half * 4 + u], xw[u], dot0);
                        dot1 = __dp4a((int)nib1[half * 4 + u], xw[u], dot1);
                    }
                    const int isum =
                        __float_as_int(half == 0 ? is2.x : is2.y);
                    const float xs_v = (half == 0) ? xs2.x : xs2.y;
                    sub0 += xs_v * ((d0 * sc0) * (float)(dot0 + 8 * isum)
                                    - (dmin0 * mn0) * (float)isum);
                    sub1 += xs_v * ((d1 * sc1) * (float)(dot1 + 8 * isum)
                                    - (dmin1 * mn1) * (float)isum);
                }
                acc0[r] += sub0;
                acc1[r] += sub1;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; r++) {
        if (r < p) {
            float a0 = acc0[r];
            float a1 = acc1[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a0 += __shfl_down_sync(0xFFFFFFFFu, a0, off);
                a1 += __shfl_down_sync(0xFFFFFFFFu, a1, off);
            }
            if (lane == 0) {
                y[(size_t)r * m + row0] = a0;
                y[(size_t)r * m + row1] = a1;
            }
        }
    }
}

extern "C" __global__ void qwen38_gemv_q6k_q8x_rows(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16]
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int warp2 = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // row pair id
    const int row0 = warp2 * 2;
    if (row0 >= m) return;
    const int row1 = (row0 + 1 < m) ? row0 + 1 : row0;
    const int lane = threadIdx.x & 31;
    const unsigned char* wrow0 = w + (size_t)row0 * (size_t)blocks_per_row * 210;
    const unsigned char* wrow1 = w + (size_t)row1 * (size_t)blocks_per_row * 210;
    const int n_grp = n >> 4;
    const int groups = n >> 4;
    float acc0[16];
    float acc1[16];
#pragma unroll
    for (int r = 0; r < 16; r++) { acc0[r] = 0.f; acc1[r] = 0.f; }
    for (int gpos = lane; gpos < n_grp; gpos += 32) {
        const int blk = gpos >> 4;
        const int h = (gpos >> 3) & 1;
        const int slot = (gpos >> 1) & 3;
        const int l0 = (gpos & 1) << 4;
        const unsigned char* b0 = wrow0 + (size_t)blk * 210;
        const unsigned char* b1 = wrow1 + (size_t)blk * 210;
        const unsigned char* ql0p = b0 + ((h << 6) + ((slot & 1) << 5));
        const unsigned char* qh0p = b0 + 128 + (h << 5);
        const unsigned char* ql1p = b1 + ((h << 6) + ((slot & 1) << 5));
        const unsigned char* qh1p = b1 + 128 + (h << 5);
        // Issue 742 T9.16 - weight-side loads are __ldcs (evict-first
        // streaming; read exactly once - see the Q4 kernel's note). Q6_K's
        // 210-byte blocks stay 2-aligned, so the u16-pair assembly is the
        // max vectorization (the T9.3 lesson) - only the hint changes.
        const float sc0 =
            (float)(signed char)__ldcs((const unsigned char*)(b0 + 192 + (h << 3) + (gpos & 1) + (slot << 1)));
        const float d0 =
            f16_to_f32_exact(__ldcs(reinterpret_cast<const unsigned short*>(b0 + 208)));
        const float sc1 =
            (float)(signed char)__ldcs((const unsigned char*)(b1 + 192 + (h << 3) + (gpos & 1) + (slot << 1)));
        const float d1 =
            f16_to_f32_exact(__ldcs(reinterpret_cast<const unsigned short*>(b1 + 208)));
        const unsigned int nib_shift = (slot < 2) ? 0 : 4;
        const unsigned int qh_shift = (unsigned int)(slot << 1);
        int q6s0[4];
        int q6s1[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
#define Q6_U32(p_) (((unsigned int)__ldcs((const unsigned short*)(p_))) | \
                    ((unsigned int)__ldcs((const unsigned short*)((p_) + 2)) << 16))
            const unsigned int qlw0 = Q6_U32(ql0p + l0 + u * 4);
            const unsigned int qhw0 = Q6_U32(qh0p + l0 + u * 4);
            const unsigned int as0 =
                ((qlw0 >> nib_shift) & 0x0F0F0F0Fu) | (((qhw0 >> qh_shift) & 0x03030303u) << 4);
            q6s0[u] = (int)__vsub4(as0, 0x20202020u);
            const unsigned int qlw1 = Q6_U32(ql1p + l0 + u * 4);
            const unsigned int qhw1 = Q6_U32(qh1p + l0 + u * 4);
            const unsigned int as1 =
                ((qlw1 >> nib_shift) & 0x0F0F0F0Fu) | (((qhw1 >> qh_shift) & 0x03030303u) << 4);
            q6s1[u] = (int)__vsub4(as1, 0x20202020u);
#undef Q6_U32
        }
#pragma unroll
        for (int r = 0; r < 16; r++) {
            if (r < p) {
                const int xw_idx = r * groups + gpos;
                const float xs_v = xs[xw_idx];  // shared by both rows
                const int4 xw4 =
                    *reinterpret_cast<const int4*>(xq + (size_t)r * n + (gpos << 4));
                const int* xw = reinterpret_cast<const int*>(&xw4);
                int dot0 = 0;
                int dot1 = 0;
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    dot0 = __dp4a(q6s0[u], xw[u], dot0);
                    dot1 = __dp4a(q6s1[u], xw[u], dot1);
                }
                acc0[r] += xs_v * ((d0 * sc0) * (float)dot0);
                acc1[r] += xs_v * ((d1 * sc1) * (float)dot1);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; r++) {
        if (r < p) {
            float a0 = acc0[r];
            float a1 = acc1[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a0 += __shfl_down_sync(0xFFFFFFFFu, a0, off);
                a1 += __shfl_down_sync(0xFFFFFFFFu, a1, off);
            }
            if (lane == 0) {
                y[(size_t)r * m + row0] = a0;
                y[(size_t)r * m + row1] = a1;
            }
        }
    }
}

// Batched embedding lookup: dequantize the token rows[t] row of the Q4_K
// embedding into out[t, n] - the element mapping is VERBATIM
// qwen38_dequant_q4k_row.
extern "C" __global__ void qwen38_dequant_q4k_rows(
    const unsigned char* __restrict__ w,   // [rows * blocks_per_row * 144]
    float* __restrict__ out,               // [p, n]
    const int* __restrict__ tokens,        // [p]
    const int n,
    const int blocks_per_row,
    const int p)
{
    const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long)p * n) return;
    const int r = (int)(e / n);
    const int i = (int)(e % n);
    const int row = tokens[r];
    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 144;
    const int j = (i >> 5) & 7;
    const int blk = i >> 8;
    const unsigned char* b = wrow + (size_t)blk * 144;
    const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b));
    const float dmin = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 2));
    const unsigned char* s = b + 4;
    float sc, mn;
    if (j < 4) {
        sc = (float)(s[j] & 63);
        mn = (float)(s[j + 4] & 63);
    } else {
        sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
    const unsigned char* q = b + 16 + ((j >> 1) << 5);
    const int l = i & 31;
    const int nib = (j & 1) ? (q[l] >> 4) : (q[l] & 0x0F);
    out[e] = (d * sc) * (float)nib - dmin * mn;
}

// Batched argmax (first-index tie-break, CPU-exact - the Issue-697 packed
// u64 reduction) over p vocab-sized logits rows; one output per row.
extern "C" __global__ void qwen38_argmax_first_rows(
    const float* __restrict__ values,   // [p, n]
    int n,
    int p,
    unsigned long long* __restrict__ result)  // [p], zeroed before launch
{
    const int row = blockIdx.y;
    const float* vals = values + (long)row * n;
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    float m = __int_as_float(0xff800000);
    int mi = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = vals[i];
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
        if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;
        const unsigned int key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long packed =
            ((unsigned long long)key << 32) | (unsigned int)(~(unsigned int)si[0]);
        atomicMax(result + row, packed);
    }
}

// Issue 742 T9.9 - the ROWS twins of the fused rmsnorm+q8 kernels: one
// block per row (grid p), the body VERBATIM the single-row kernel's with
// the row offset added to every buffer index. Bit-identical per row.
extern "C" __global__ void qwen38_rmsnorm_quant_x_q8_rows(
    const float* __restrict__ input,   // [p, dim] raw input
    const float* __restrict__ gamma,   // [dim]
    signed char* __restrict__ xq,      // [p, dim]
    float* __restrict__ xs,            // [p, dim/16]
    int* __restrict__ xsum,            // [p, dim/16]
    float inv_dim,
    float eps,
    int dim,
    int groups)                        // dim / 16
{
    const long row = blockIdx.x;
    const float* input_r = input + row * dim;
    signed char* xq_r = xq + row * dim;
    float* xs_r = xs + row * groups;
    int* xsum_r = xsum + row * groups;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // 256 (launcher contract)

    // Phase 1: strided accumulation of x^2 (VERBATIM).
    float partial_sq = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        float x = input_r[i];
        partial_sq += x * x;
    }

    // Phase 2: shared memory parallel reduction (VERBATIM).
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

    // Phase 3: compute inv_rms (VERBATIM).
    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    // Phase 4: normalize + quantize per-16 group (VERBATIM).
    for (int g = tid; g < groups; g += block_size) {
        const int base = g << 4;
        float vals[16];
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const float v = input_r[base + i] * inv_rms * gamma[base + i];
            vals[i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float s = amax > 0.f ? amax / 127.f : 1.f;
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            int q = (int)__float2int_rn(vals[i] / s);
            q = max(-127, min(127, q));
            xq_r[base + i] = (signed char)q;
            sum += q;
        }
        xs_r[g] = s;
        xsum_r[g] = sum;
    }
}

// The constexpr-specialized twin (dim=5120) with the row offset - VERBATIM
// qwen38_rmsnorm_quant_x_q8_s5120's body per row.
extern "C" __global__ void qwen38_rmsnorm_quant_x_q8_s5120_rows(
    const float* __restrict__ input,   // [p, 5120] raw input
    const float* __restrict__ gamma,   // [5120]
    signed char* __restrict__ xq,      // [p, 5120]
    float* __restrict__ xs,            // [p, 320]
    int* __restrict__ xsum,            // [p, 320]
    float inv_dim,
    float eps)
{
    const long row = blockIdx.x;
    const float* input_r = input + row * 5120;
    signed char* xq_r = xq + row * 5120;
    float* xs_r = xs + row * 320;
    int* xsum_r = xsum + row * 320;
    const int tid = threadIdx.x;       // blockDim.x == 256 (launcher contract)
    constexpr int DIM = 5120;
    constexpr int BLOCK = 256;
    constexpr int NITER = DIM / BLOCK;             // 20
    constexpr int GROUPS = DIM / 16;               // 320
    constexpr int GTRIPS = (GROUPS + BLOCK - 1) / BLOCK;  // 2

    float partial_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < NITER; ++k) {
        const int i = tid + k * BLOCK;
        const float x = input_r[i];
        partial_sq += x * x;
    }

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

    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

#pragma unroll
    for (int r = 0; r < GTRIPS; ++r) {
        const int g = tid + r * BLOCK;
        if (g < GROUPS) {
            const int base = g << 4;
            float vals[16];
            float amax = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float v = input_r[base + i] * inv_rms * gamma[base + i];
                vals[i] = v;
                amax = fmaxf(amax, fabsf(v));
            }
            const float s = amax > 0.f ? amax / 127.f : 1.f;
            int sum = 0;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                int q = (int)__float2int_rn(vals[i] / s);
                q = max(-127, min(127, q));
                xq_r[base + i] = (signed char)q;
                sum += q;
            }
            xs_r[g] = s;
            xsum_r[g] = sum;
        }
    }
}

// Issue 755 — the fused layer-BOUNDARY kernel: residual_add + snapshot
// copy + the NEXT site's rmsnorm+q8-quantize in ONE launch. Provenance:
// MTPLX PR #335 row 48 (fused residual/RMSNorm boundary chain) + corpus
// B63 deferred-residual-fused-into-next-layer-norm; Issue 755 follow-on.
// Bit-identical by construction (see phase comments). Eliminates the
// residual_add + copy_f32 launches + the x round-trip per site; 128
// sites/token decode, ×p in verify.
//
// BIT-IDENTITY vs the unfused 3-kernel chain (residual_add_f32 →
// qwen38_copy_f32 → qwen38_rmsnorm_quant_x_q8_s5120):
// - `x = res[i] + y[i]` is exactly residual_add_f32's `a[idx] + b[idx]`.
// - `x_out[i] = x` is what residual_add stored into the residual stream and
//   `res_out[i] = x` is what the following copy_f32 produced (f32
//   store→load is bit-preserving — the T9.7 argument), so every later read
//   of `x_out` sees bit-identical bits to the unfused norm's read of `x`.
// - The per-lane ascending `partial_sq += x*x` chain, the 8-step
//   256-thread smem tree, the inv_rms expression, and the phase-4 group
//   mapping + expression are VERBATIM qwen38_rmsnorm_quant_x_q8_s5120.
//
// ALIASING CONTRACT: `res_out` MAY alias `res` (the call sites pass the
// SAME x_res/xb_res buffer — this kernel reads the pre-site snapshot and
// overwrites it with the post-site value for the NEXT site). Safe ONLY
// because (a) phase 1 is element-wise: thread t exclusively owns indices
// {t, t+256, ...} and reads res[i] before writing res_out[i] — no
// cross-thread reads of res; and (b) phase 4 reads `x_out`, NEVER `res` —
// if it read res[base+i] it would read the OVERWRITTEN value (sum+y —
// wrong). Preserve both when editing.
//
// Phase 4's re-read of x_out crosses threads (thread t quantizes groups
// built from OTHER threads' phase-1 stores); the __syncthreads() pair
// between the phases orders them — same visibility the template gets from
// being a separate launch.
extern "C" __global__ void qwen38_residual_norm_quant_x_q8_s5120(
    const float* __restrict__ res,     // [5120] pre-site residual snapshot
    const float* __restrict__ y,       // [5120] site output (block or MLP)
    const float* __restrict__ gamma,   // [5120]
    float* __restrict__ x_out,         // [5120] residual stream (post-site)
    float* __restrict__ res_out,       // [5120] next snapshot (MAY ALIAS res)
    signed char* __restrict__ xq,      // [5120]
    float* __restrict__ xs,            // [320]
    int* __restrict__ xsum,            // [320]
    float inv_dim,
    float eps)
{
    const int tid = threadIdx.x;       // blockDim.x == 256 (launcher contract)
    constexpr int DIM = 5120;
    constexpr int BLOCK = 256;         // DIM % BLOCK == 0 -> no tail guard
    constexpr int NITER = DIM / BLOCK;             // 20
    constexpr int GROUPS = DIM / 16;               // 320
    constexpr int GTRIPS = (GROUPS + BLOCK - 1) / BLOCK;  // 2

    // ── Phase 1: element-wise residual add + dual store + strided x²
    // accumulation. The add + dual stores replace residual_add_f32 +
    // copy_f32 (same single f32 add, same stored bits — see the bit-identity
    // block above); the lane mapping + ascending per-lane accumulation
    // order are verbatim the template. ──
    float partial_sq = 0.0f;
    #pragma unroll
    for (int k = 0; k < NITER; ++k) {
        const int i = tid + k * BLOCK;
        const float x = res[i] + y[i];
        x_out[i] = x;
        res_out[i] = x;   // may alias res[i] — same-thread read-before-write
        partial_sq += x * x;
    }

    // ── Phase 2: shared memory parallel reduction (VERBATIM) ──
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

    // ── Phase 3: compute inv_rms (VERBATIM) ──
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

    // ── Phase 4: normalize + quantize per-16 group — expression VERBATIM;
    // the read is `x_out` (the stored residual), NEVER `res` (which
    // res_out may have overwritten — the aliasing contract above). ──
    #pragma unroll
    for (int r = 0; r < GTRIPS; ++r) {
        const int g = tid + r * BLOCK;
        if (g < GROUPS) {
            const int base = g << 4;
            float vals[16];
            float amax = 0.f;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                const float v = x_out[base + i] * inv_rms * gamma[base + i];
                vals[i] = v;
                amax = fmaxf(amax, fabsf(v));
            }
            const float s = amax > 0.f ? amax / 127.f : 1.f;
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int q = (int)__float2int_rn(vals[i] / s);
                q = max(-127, min(127, q));
                xq[base + i] = (signed char)q;
                sum += q;
            }
            xs[g] = s;
            xsum[g] = sum;
        }
    }
}

// The p-row twin (grid `p`, one block per row; every buffer index gets
// `row * 5120` added — mirrors qwen38_rmsnorm_quant_x_q8_s5120_rows):
// per row bit-identical to the single-row kernel above, i.e. to the
// unfused residual_add(n*p) → copy_f32(n*p) → rows-norm chain.
// Same ALIASING contract: res_out may alias res (row-disjoint per block).
extern "C" __global__ void qwen38_residual_norm_quant_x_q8_s5120_rows(
    const float* __restrict__ res,     // [p, 5120] pre-site residual snapshot
    const float* __restrict__ y,       // [p, 5120] site output
    const float* __restrict__ gamma,   // [5120]
    float* __restrict__ x_out,         // [p, 5120] residual stream (post-site)
    float* __restrict__ res_out,       // [p, 5120] next snapshot (MAY ALIAS res)
    signed char* __restrict__ xq,      // [p, 5120]
    float* __restrict__ xs,            // [p, 320]
    int* __restrict__ xsum,            // [p, 320]
    float inv_dim,
    float eps)
{
    const long row = blockIdx.x;
    const float* res_r = res + row * 5120;
    const float* y_r = y + row * 5120;
    float* x_out_r = x_out + row * 5120;
    float* res_out_r = res_out + row * 5120;
    signed char* xq_r = xq + row * 5120;
    float* xs_r = xs + row * 320;
    int* xsum_r = xsum + row * 320;
    const int tid = threadIdx.x;       // blockDim.x == 256 (launcher contract)
    constexpr int DIM = 5120;
    constexpr int BLOCK = 256;
    constexpr int NITER = DIM / BLOCK;             // 20
    constexpr int GROUPS = DIM / 16;               // 320
    constexpr int GTRIPS = (GROUPS + BLOCK - 1) / BLOCK;  // 2

    float partial_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < NITER; ++k) {
        const int i = tid + k * BLOCK;
        const float x = res_r[i] + y_r[i];
        x_out_r[i] = x;
        res_out_r[i] = x;
        partial_sq += x * x;
    }

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

    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

#pragma unroll
    for (int r = 0; r < GTRIPS; ++r) {
        const int g = tid + r * BLOCK;
        if (g < GROUPS) {
            const int base = g << 4;
            float vals[16];
            float amax = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float v = x_out_r[base + i] * inv_rms * gamma[base + i];
                vals[i] = v;
                amax = fmaxf(amax, fabsf(v));
            }
            const float s = amax > 0.f ? amax / 127.f : 1.f;
            int sum = 0;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                int q = (int)__float2int_rn(vals[i] / s);
                q = max(-127, min(127, q));
                xq_r[base + i] = (signed char)q;
                sum += q;
            }
            xs_r[g] = s;
            xsum_r[g] = sum;
        }
    }
}
"#;

/// Compiled T9.2/T9.3 dense kernels.
pub struct DenseKernels {
    quant_x: CudaFunction,
    rmsnorm_quant_x: CudaFunction,
    rmsnorm_quant_x_s5120: CudaFunction,
    rmsnorm_zgate_quant_x: CudaFunction,
    q4k_q8x: CudaFunction,
    q6k_q8x: CudaFunction,
    dequant_q4k_row: CudaFunction,
    dequant_q4k_row_devpos: CudaFunction,
    copy_f32: CudaFunction,
    q4k_q8x_rows: CudaFunction,
    q4k_q8x_rows_strict: CudaFunction,
    q6k_q8x_rows: CudaFunction,
    dequant_q4k_rows: CudaFunction,
    argmax_rows: CudaFunction,
    rmsnorm_quant_rows: CudaFunction,
    rmsnorm_quant_rows_s5120: CudaFunction,
    residual_norm_quant_x_s5120: CudaFunction,
    residual_norm_quant_x_s5120_rows: CudaFunction,
    _module: Arc<CudaModule>,
}

impl DenseKernels {
    /// Public so integration tests can drive the shipped kernels directly
    /// (the bit-identity gate for `qwen38_rmsnorm_quant_x_q8` composes this
    /// with `ElementwiseKernels::launch_rmsnorm`).
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            QWEN38_DENSE_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| format!("{name}: {e}"))
        };
        Ok(Self {
            quant_x: f("qwen38_quant_x_q8")?,
            rmsnorm_quant_x: f("qwen38_rmsnorm_quant_x_q8")?,
            rmsnorm_quant_x_s5120: f("qwen38_rmsnorm_quant_x_q8_s5120")?,
            rmsnorm_zgate_quant_x: f("qwen38_rmsnorm_zgate_quant_x_q8")?,
            q4k_q8x: f("qwen38_gemv_q4k_q8x")?,
            q6k_q8x: f("qwen38_gemv_q6k_q8x")?,
            dequant_q4k_row: f("qwen38_dequant_q4k_row")?,
            dequant_q4k_row_devpos: f("qwen38_dequant_q4k_row_devpos")?,
            copy_f32: f("qwen38_copy_f32")?,
            q4k_q8x_rows: f("qwen38_gemv_q4k_q8x_rows")?,
            q4k_q8x_rows_strict: f("qwen38_gemv_q4k_q8x_rows_strict")?,
            q6k_q8x_rows: f("qwen38_gemv_q6k_q8x_rows")?,
            dequant_q4k_rows: f("qwen38_dequant_q4k_rows")?,
            argmax_rows: f("qwen38_argmax_first_rows")?,
            rmsnorm_quant_rows: f("qwen38_rmsnorm_quant_x_q8_rows")?,
            rmsnorm_quant_rows_s5120: f("qwen38_rmsnorm_quant_x_q8_s5120_rows")?,
            residual_norm_quant_x_s5120: f("qwen38_residual_norm_quant_x_q8_s5120")?,
            residual_norm_quant_x_s5120_rows: f("qwen38_residual_norm_quant_x_q8_s5120_rows")?,
            _module: module,
        })
    }

    /// Launch the fused rmsnorm+q8-quantize kernel (Issue 742 T9.7 follow-on).
    ///
    /// Bit-identical to `ElementwiseKernels::launch_rmsnorm` followed by
    /// [`Self::launch_quant_x_q8`] on the normed output (phases 1-3 verbatim
    /// `rmsnorm_f32`; phase 4 `qwen38_quant_x_q8`'s exact ops on the same
    /// expression `rmsnorm_f32` stores).
    ///
    /// # Safety
    ///
    /// Caller guarantees `input`/`gamma` cover `dim` f32 each, `xq` covers
    /// `dim` i8, `xs`/`xsum` cover `dim/16`, and `dim % 16 == 0` (one block;
    /// the phase-4 group loop covers any `dim/16`).
    pub unsafe fn launch_rmsnorm_quant_x_q8(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        assert!(dim.is_multiple_of(16), "fused rnq: dim % 16 == 0 (got {dim})");
        let inv_dim = 1.0f32 / dim as f32;
        let (dim_i, groups_i) = (dim as i32, (dim / 16) as i32);
        // SAFETY: caller contract above (buffer extents + divisibility).
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_quant_x)
                .arg(input)
                .arg(gamma)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i)
                .arg(&groups_i)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 256 * 4,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the constexpr-specialized fused rmsnorm+q8 kernel for
    /// dim=5120 (Issue 742 T9.7c, Bench 736) — the dim of every production
    /// site. Numerically identical to [`Self::launch_rmsnorm_quant_x_q8`]
    /// at dim 5120 (verbatim lane chains, reduction tree, and phase-4
    /// expression; see the kernel's doc comment), but both data loops are
    /// fully unrolled via compile-time trip counts — the generic kernel's
    /// runtime-bound loops serialize the strided loads through ~5 DRAM
    /// round-trips, which is the measured 7.36 µs/site latency wall.
    ///
    /// # Safety
    ///
    /// Caller guarantees `input`/`gamma` cover 5120 f32 each, `xq` covers
    /// 5120 i8, and `xs`/`xsum` cover 320.
    pub unsafe fn launch_rmsnorm_quant_x_q8_s5120(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        eps: f32,
    ) -> Result<(), String> {
        let inv_dim = 1.0f32 / 5120.0f32;
        // SAFETY: caller contract above (fixed 5120/320 extents).
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_quant_x_s5120)
                .arg(input)
                .arg(gamma)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(&inv_dim)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the Issue-755 fused layer-BOUNDARY kernel for dim=5120:
    /// `x = res + y`, dual-store (`x_out` = residual stream, `res_out` = the
    /// next site's snapshot), and the next site's rmsnorm+q8-quantize of
    /// `x` — ONE launch replacing residual_add_f32 + copy_f32 +
    /// rmsnorm_quant_x (3 launches + a full x DRAM round-trip per boundary;
    /// 128 sites/token decode, ×p in verify).
    ///
    /// Bit-identical to the 3-kernel chain by construction (kernel doc:
    /// same f32 add, same stored bits, verbatim reduction/quant phases).
    ///
    /// # Safety
    ///
    /// Caller guarantees `res`/`y`/`gamma`/`x_out`/`res_out` cover 5120 f32
    /// each, `xq` covers 5120 i8, and `xs`/`xsum` cover 320. ALIASING
    /// contract: `res_out` MAY alias (even equal) `res` — phase 1 is
    /// element-wise (thread t exclusively owns {t, t+256, ..}, reads
    /// `res[i]` before writing `res_out[i]`), and phase 4 reads `x_out`
    /// ONLY, never `res`. `x_out` must NOT alias `res` or `y`.
    pub unsafe fn launch_residual_norm_quant_x_q8_s5120(
        &self,
        stream: &CudaStream,
        res: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        x_out: &CudaSlice<f32>,
        res_out: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        eps: f32,
    ) -> Result<(), String> {
        let inv_dim = 1.0f32 / 5120.0f32;
        // SAFETY: caller contract above (fixed 5120/320 extents + aliasing).
        unsafe {
            stream
                .launch_builder(&self.residual_norm_quant_x_s5120)
                .arg(res)
                .arg(y)
                .arg(gamma)
                .arg(x_out)
                .arg(res_out)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(&inv_dim)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Rows twin of [`Self::launch_residual_norm_quant_x_q8_s5120`] (one
    /// block per row, grid `p`) — per row bit-identical to the single-row
    /// kernel, i.e. to the unfused residual_add(n*p) → copy_f32(n*p) →
    /// rows-norm chain (the verify/ingest boundary sites).
    ///
    /// # Safety
    ///
    /// Caller guarantees `res`/`y`/`x_out`/`res_out` cover `p * 5120` f32
    /// each, `gamma` covers 5120, `xq` covers `p * 5120` i8, `xs`/`xsum`
    /// cover `p * 320`, and `p >= 1`. Same aliasing contract as the
    /// single-row launcher: `res_out` may alias `res` (row-disjoint per
    /// block — block `row` exclusively owns `[row*5120, (row+1)*5120)`);
    /// phase 4 reads `x_out` only; `x_out` must not alias `res`/`y`.
    pub unsafe fn launch_residual_norm_quant_x_q8_s5120_rows(
        &self,
        stream: &CudaStream,
        res: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        x_out: &CudaSlice<f32>,
        res_out: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        eps: f32,
        p: usize,
    ) -> Result<(), String> {
        assert!(p >= 1, "res-nq rows: p >= 1");
        let inv_dim = 1.0f32 / 5120.0f32;
        // SAFETY: caller contract above (row-contiguous buffers, grid p).
        unsafe {
            stream
                .launch_builder(&self.residual_norm_quant_x_s5120_rows)
                .arg(res)
                .arg(y)
                .arg(gamma)
                .arg(x_out)
                .arg(res_out)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(&inv_dim)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (p as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the fused per-head RMSNorm + silu z-gate + q8-quantize kernel
    /// (Issue 742 lever-2, Bench 737) — the GDN post-recurrence chain in one
    /// kernel.
    ///
    /// Bit-identical to `AttentionKernels::launch_rmsnorm_batched` →
    /// `DeltanetKernels::launch_z_gating` → [`Self::launch_quant_x_q8`] on
    /// the gated output: phase 1 is verbatim `rmsnorm_batched_f32` (same
    /// strided loop + smem tree + expressions → the same inv_rms bits),
    /// phase 2 computes the exact gated value the trio stores into
    /// rec_normed (f32 store/load is bit-preserving; IEEE multiply is
    /// commutative; both paths run the same nvrtc expf under identical
    /// compile options), and phase 3 applies `qwen38_quant_x_q8`'s exact ops
    /// — amax via 16-lane butterfly shfl_xor (fmaxf is order-independent),
    /// the same s expression, `__float2int_rn` clamped [-127,127], and the
    /// group integer sum via the same butterfly (integer add is exactly
    /// associative).
    ///
    /// # Safety
    ///
    /// Caller guarantees `input`/`z` cover `n_heads * head_dim` f32 each,
    /// `gamma` covers `head_dim` f32, `xq` covers `n_heads * head_dim` i8,
    /// `xs`/`xsum` cover `n_heads * head_dim / 16`, `head_dim % 16 == 0`,
    /// and `head_dim <= 256` (one block per head).
    pub unsafe fn launch_rmsnorm_zgate_quant_x_q8(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        assert!(
            head_dim.is_multiple_of(16) && head_dim <= 256,
            "fused zgq: head_dim % 16 == 0 && head_dim <= 256 (got {head_dim})"
        );
        let inv_dim = 1.0f32 / head_dim as f32;
        let head_dim_i = head_dim as i32;
        let grid = n_heads.max(1) as u32;
        // SAFETY: caller contract above (buffer extents + head_dim gates).
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_zgate_quant_x)
                .arg(input)
                .arg(z)
                .arg(gamma)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&head_dim_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 256 * 4,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the standalone q8 quantizer (the second half of the unfused
    /// pair; public so the bit-identity gate can drive both parents).
    ///
    /// # Safety
    ///
    /// Caller guarantees `x` covers `n` f32, `xq` covers `n` i8, `xs`/`xsum`
    /// cover `n/16`, and `n % 16 == 0`.
    pub unsafe fn launch_quant_x_q8(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        n: usize,
    ) -> Result<(), String> {
        assert!(n.is_multiple_of(16), "quant_x: n % 16 == 0 (got {n})");
        let n_i = n as i32;
        let grid = (n / 16).div_ceil(256).max(1) as u32;
        // SAFETY: caller contract above (buffer extents + divisibility).
        unsafe {
            stream
                .launch_builder(&self.quant_x)
                .arg(x)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }


    /// The single-row q8 GEMV (the production decode path's kernel — the
    /// rows twins' bit-identity reference). Public so integration tests can
    /// drive the shipped kernel directly (the `launch_rmsnorm_quant_x_q8`
    /// precedent).
    ///
    /// # Safety
    ///
    /// Caller guarantees `xq` covers `n` i8, `xs`/`xsum` cover `n/16`, `y`
    /// covers `m` f32, and `n % 32 == 0` (Q4_K) / `n % 16 == 0` (Q6_K).
    pub unsafe fn launch_gemv_q8x_single(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<u8>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        y: &CudaSlice<f32>,
        m: usize,
        n: usize,
        blocks_per_row: usize,
        q4: bool,
    ) -> Result<(), String> {
        let (m_i, n_i, bpr_i) = (m as i32, n as i32, blocks_per_row as i32);
        let grid = m.div_ceil(8).max(1) as u32;
        let func = if q4 {
            &self.q4k_q8x
        } else {
            &self.q6k_q8x
        };
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(func)
                .arg(w)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The single-row q8 GEMV over SLICED VIEWS (the `loop` verify arm: p
    /// back-to-back launches of the production kernel, one per draft
    /// position — per-row pointer arithmetic via CudaView offsets). Same
    /// kernel and numerics as [`Self::launch_gemv_q8x_single`].
    ///
    /// # Safety
    ///
    /// Caller guarantees each view covers exactly `n` i8 / `n/16` scalars /
    /// `m` f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_gemv_q8x_views(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<u8>,
        xq: &cudarc::driver::safe::CudaView<'_, i8>,
        xs: &cudarc::driver::safe::CudaView<'_, f32>,
        xsum: &cudarc::driver::safe::CudaView<'_, i32>,
        y: &cudarc::driver::safe::CudaView<'_, f32>,
        m: usize,
        n: usize,
        blocks_per_row: usize,
        q4: bool,
    ) -> Result<(), String> {
        let (m_i, n_i, bpr_i) = (m as i32, n as i32, blocks_per_row as i32);
        let grid = m.div_ceil(8).max(1) as u32;
        let func = if q4 {
            &self.q4k_q8x
        } else {
            &self.q6k_q8x
        };
        // SAFETY: caller contract above (exact-size views).
        unsafe {
            stream
                .launch_builder(func)
                .arg(w)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    // ── Issue 742 T9.9: the p-row batched verify launchers ────────────────

    /// Rows twin of [`Self::launch_rmsnorm_quant_x_q8`]: one block per row
    /// (grid `p`), per row bit-identical to the single-row kernel.
    ///
    /// # Safety
    ///
    /// Caller guarantees `input` covers `p * dim` f32, `gamma` covers `dim`,
    /// `xq` covers `p * dim` i8, `xs`/`xsum` cover `p * dim/16`, and
    /// `dim % 16 == 0`.
    pub unsafe fn launch_rmsnorm_quant_x_q8_rows(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        dim: usize,
        eps: f32,
        p: usize,
    ) -> Result<(), String> {
        assert!(dim.is_multiple_of(16), "rnq rows: dim % 16 == 0");
        let inv_dim = 1.0f32 / dim as f32;
        let (dim_i, groups_i) = (dim as i32, (dim / 16) as i32);
        // SAFETY: caller contract above (row-contiguous buffers, grid p).
        unsafe {
            if dim == 5120 {
                stream
                    .launch_builder(&self.rmsnorm_quant_rows_s5120)
                    .arg(input)
                    .arg(gamma)
                    .arg(xq)
                    .arg(xs)
                    .arg(xsum)
                    .arg(&inv_dim)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (p as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| e.to_string())?;
            } else {
                stream
                    .launch_builder(&self.rmsnorm_quant_rows)
                    .arg(input)
                    .arg(gamma)
                    .arg(xq)
                    .arg(xs)
                    .arg(xsum)
                    .arg(&inv_dim)
                    .arg(&eps)
                    .arg(&dim_i)
                    .arg(&groups_i)
                    .launch(LaunchConfig {
                        grid_dim: (p as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 256 * 4,
                    })
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    /// The p-row batched Q4_K GEMV: `y[r, row] = dot(w_row, x_r)` for all
    /// `p` rows — weights read ONCE, per element bit-identical to
    /// `qwen38_gemv_q4k_q8x` (see the kernel doc). Grid: one warp per
    /// output row.
    ///
    /// # Safety
    ///
    /// Caller guarantees `xq` covers `p * n` i8, `xs`/`xsum` cover
    /// `p * n/16`, `y` covers `p * m` f32, `p <= 16`, and `n % 32 == 0`.
    pub unsafe fn launch_gemv_q4k_rows_strict(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<u8>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        y: &CudaSlice<f32>,
        m: usize,
        n: usize,
        blocks_per_row: usize,
        p: usize,
    ) -> Result<(), String> {
        assert!(p <= 16 && n.is_multiple_of(32), "gemv rows: p<=16, n%32==0");
        let (m_i, n_i, bpr_i, p_i) =
            (m as i32, n as i32, blocks_per_row as i32, p as i32);
        // The bit-identical arm: warp per row PAIR (R=2), lane-strided
        // sub-blocks + butterfly — per element equals the single-row kernel.
        let grid = m.div_ceil(16).max(1) as u32;
        let func = &self.q4k_q8x_rows_strict;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(func)
                .arg(w)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The p-row batched Q4_K GEMV, shape B (the FAST default): lane =
    /// (input row, half), 8 output rows per warp — every x load and scalar
    /// load shared 8-fold (the x-side L1 fix). Per (row, output) the
    /// per-sub-block terms are bit-identical to the single kernel; the
    /// sub-block SUMMATION ORDER differs (reassociation class — see the
    /// kernel doc). `launch_gemv_q4k_rows_strict` is the bit-identical arm.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_gemv_q4k_rows_strict`].
    pub unsafe fn launch_gemv_q4k_rows(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<u8>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        y: &CudaSlice<f32>,
        m: usize,
        n: usize,
        blocks_per_row: usize,
        p: usize,
    ) -> Result<(), String> {
        assert!(p <= 16 && n.is_multiple_of(32), "gemv rows: p<=16, n%32==0");
        let (m_i, n_i, bpr_i, p_i) =
            (m as i32, n as i32, blocks_per_row as i32, p as i32);
        // Shape B: 8 output rows per warp (grid = m/64 blocks of 8 warps).
        let grid = m.div_ceil(64).max(1) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.q4k_q8x_rows)
                .arg(w)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The p-row batched Q6_K GEMV (the Q4 twin's shape; per element
    /// bit-identical to `qwen38_gemv_q6k_q8x`).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_gemv_q4k_rows`].
    pub unsafe fn launch_gemv_q6k_rows(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<u8>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsum: &CudaSlice<i32>,
        y: &CudaSlice<f32>,
        m: usize,
        n: usize,
        blocks_per_row: usize,
        p: usize,
    ) -> Result<(), String> {
        assert!(p <= 16 && n.is_multiple_of(16), "gemv rows: p<=16, n%16==0");
        let (m_i, n_i, bpr_i, p_i) =
            (m as i32, n as i32, blocks_per_row as i32, p as i32);
        // R=2 row-blocked warps: one warp per TWO output rows, 8 warps/block.
        let grid = m.div_ceil(16).max(1) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.q6k_q8x_rows)
                .arg(w)
                .arg(xq)
                .arg(xs)
                .arg(xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Batched embedding lookup: dequantize the `tokens[t]` row of the Q4_K
    /// embedding into `out[t, n]` (per element VERBATIM the single-row
    /// kernel's mapping).
    ///
    /// # Safety
    ///
    /// Caller guarantees `out` covers `p * n` and `tokens` covers `p`.
    pub unsafe fn launch_dequant_q4k_rows(
        &self,
        stream: &CudaStream,
        w: &CudaSlice<u8>,
        out: &CudaSlice<f32>,
        tokens: &CudaSlice<i32>,
        n: usize,
        blocks_per_row: usize,
        p: usize,
    ) -> Result<(), String> {
        let (n_i, bpr_i, p_i) = (n as i32, blocks_per_row as i32, p as i32);
        let total = (p * n) as u32;
        let grid = total.div_ceil(256).max(1);
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.dequant_q4k_rows)
                .arg(w)
                .arg(out)
                .arg(tokens)
                .arg(&n_i)
                .arg(&bpr_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Batched argmax with the Issue-697 first-index tie-break over `p`
    /// vocab-sized rows; `result[r]` holds the packed (key, ~idx) — the
    /// token is `!(packed as u32)`. Zero `result` before every launch.
    ///
    /// # Safety
    ///
    /// Caller guarantees `values` covers `p * n`, `result` covers `p`, and
    /// that `result` was zeroed since the last launch.
    pub unsafe fn launch_argmax_rows(
        &self,
        stream: &CudaStream,
        values: &CudaSlice<f32>,
        n: usize,
        p: usize,
        result: &CudaSlice<u64>,
    ) -> Result<(), String> {
        let (n_i, p_i) = (n as i32, p as i32);
        // ~2 blocks of 256 per row (matches the decode argmax shape).
        let blocks_per_row = (n as u32).div_ceil(512).max(1);
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.argmax_rows)
                .arg(values)
                .arg(&n_i)
                .arg(&p_i)
                .arg(result)
                .launch(LaunchConfig {
                    grid_dim: (blocks_per_row, p as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config (parsed from GGUF metadata — mirrors
// `qwen35_deltanet_config_from_gguf_metadata`, which is engine-private; this
// module stays self-contained so no engine change is needed).
// ─────────────────────────────────────────────────────────────────────────────

/// Layer type: GDN (linear attention) vs full attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen38LayerType {
    Deltanet,
    Attention,
}

/// Dense qwen35 model config (the dbirks 27B values in brackets).
#[derive(Clone, Debug)]
pub struct Qwen38DenseConfig {
    pub n_embd: usize,      // 5120
    pub n_layer: usize,     // 64 (main stack; MTP/nextn excluded)
    pub n_head: usize,      // 24
    pub n_kv_head: usize,   // 4
    pub head_dim: usize,    // 256
    pub rotary_dim: usize,  // 64 (partial RoPE)
    pub rope_theta: f32,    // 1e7
    pub vocab_size: usize,  // 248320
    pub rms_norm_eps: f32,
    pub mlp_hidden: usize,  // 17408
    pub n_k_heads: usize,   // 16 (ssm.group_count)
    pub n_v_heads: usize,   // 48 (ssm.time_step_rank)
    pub head_k_dim: usize,  // 128 (ssm.state_size)
    pub head_v_dim: usize,  // 128 (d_inner / n_v_heads)
    pub d_inner: usize,     // 6144 (ssm.inner_size)
    pub conv_kernel: usize, // 4
    pub layer_types: Vec<Qwen38LayerType>,
}

impl Qwen38DenseConfig {
    /// Parse from GGUF metadata. Fails on non-qwen35 arch.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.architecture().unwrap_or("unknown");
        if arch != "qwen35" {
            return Err(format!("expected qwen35 architecture, got '{arch}'"));
        }
        let p = "qwen35.";
        let u = |k: &str| gguf.metadata_u64(&format!("{p}{k}"));
        let f = |k: &str| gguf.metadata_f64(&format!("{p}{k}"));
        let n_embd = u("embedding_length").ok_or("embedding_length")? as usize;
        let n_layer_all = u("block_count").ok_or("block_count")? as usize;
        let mlp_hidden = u("feed_forward_length").ok_or("feed_forward_length")? as usize;
        let n_head = u("attention.head_count").ok_or("head_count")? as usize;
        let n_kv_head = u("attention.head_count_kv").ok_or("head_count_kv")? as usize;
        let head_dim = u("attention.key_length").ok_or("key_length")? as usize;
        let rms_norm_eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32;
        let conv_kernel = u("ssm.conv_kernel").unwrap_or(4) as usize;
        let head_k_dim = u("ssm.state_size").unwrap_or(128) as usize;
        let n_v_heads = u("ssm.time_step_rank").ok_or("time_step_rank")? as usize;
        let n_k_heads = u("ssm.group_count").ok_or("group_count")? as usize;
        let d_inner = u("ssm.inner_size").ok_or("inner_size")? as usize;
        let rotary_dim = u("rope.dimension_count").unwrap_or(0) as usize;
        let rope_theta = f("rope.freq_base").unwrap_or(10_000.0) as f32;
        let interval = u("full_attention_interval").unwrap_or(4) as usize;
        let nextn = u("nextn_predict_layers").unwrap_or(0) as usize;
        let n_layer = n_layer_all
            .checked_sub(nextn)
            .ok_or("nextn_predict_layers > block_count")?;

        let layer_types = (0..n_layer)
            .map(|i| {
                if !(i + 1).is_multiple_of(interval) {
                    Qwen38LayerType::Deltanet
                } else {
                    Qwen38LayerType::Attention
                }
            })
            .collect();
        let head_v_dim = d_inner.checked_div(n_v_heads).unwrap_or(head_k_dim);
        let rotary_dim = if rotary_dim == 0 { head_dim } else { rotary_dim };
        let vocab_size = gguf
            .tensor_info("token_embd.weight")
            .and_then(|i| i.shape.last().copied())
            .unwrap_or(248_320);
        Ok(Self {
            n_embd,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            rotary_dim,
            rope_theta,
            vocab_size,
            rms_norm_eps,
            mlp_hidden,
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            d_inner,
            conv_kernel,
            layer_types,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device-side weight storage
// ─────────────────────────────────────────────────────────────────────────────

/// A quantized GEMV weight tensor resident on device (verbatim bytes).
pub struct QuantW {
    pub dev: CudaSlice<u8>,
    pub rows: usize,
    pub n: usize,
    pub blocks_per_row: usize,
    pub q4: bool,
}

/// A small f32 tensor resident on device.
pub struct F32W {
    pub dev: CudaSlice<f32>,
    pub len: usize,
}

/// Per-layer device weights (one variant arm populated per layer type).
pub struct Qwen38LayerGpu {
    pub input_norm: F32W,        // [n_embd]
    pub post_attn_norm: F32W,    // [n_embd]
    // GDN-only
    pub qkv: Option<QuantW>,     // [q+k+v, n_embd] = [10240, 5120]
    pub z: Option<QuantW>,       // [d_inner, n_embd]
    pub alpha: Option<QuantW>,   // [n_v_heads, n_embd] (Q4_K in this GGUF)
    pub beta: Option<QuantW>,    // [n_v_heads, n_embd]
    pub conv1d: Option<F32W>,    // [conv_dim * kernel]
    pub a_log: Option<F32W>,     // [n_v_heads]
    pub dt_bias: Option<F32W>,   // [n_v_heads]
    pub ssm_norm: Option<F32W>,  // [head_v_dim]
    pub ssm_out: Option<QuantW>, // [n_embd, d_inner]
    // attention-only
    pub wq: Option<QuantW>,      // [2*q_dim, n_embd]
    pub wk: Option<QuantW>,      // [kv_dim, n_embd]
    pub wv: Option<QuantW>,      // [kv_dim, n_embd]
    pub wo: Option<QuantW>,      // [n_embd, q_dim]
    pub q_norm: Option<F32W>,    // [head_dim]
    pub k_norm: Option<F32W>,    // [head_dim]
    // MLP (both)
    pub ffn_gate: QuantW,        // [mlp_hidden, n_embd]
    pub ffn_up: QuantW,          // [mlp_hidden, n_embd]
    pub ffn_down: QuantW,        // [n_embd, mlp_hidden]
}

/// Whole-model GPU-resident weights.
pub struct Qwen38DenseWeightsGpu {
    /// Q4_K token embedding, verbatim bytes [vocab * (n_embd/256) * 144].
    pub token_embd: QuantW,
    pub output_norm: F32W, // [n_embd]
    /// Q6_K (or Q4_K) lm_head [vocab, n_embd].
    pub lm_head: QuantW,
    pub layers: Vec<Qwen38LayerGpu>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Load + upload
// ─────────────────────────────────────────────────────────────────────────────

/// Exact host-side f16 → f32 (mirrors the kernel helper; no `half` dep).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x03ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let v = (frac as f32) * 5.960_464_5e-8; // frac * 2^-24
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (frac << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f32_bytes_from_gguf(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| format!("tensor '{name}' not found"))?;
    let bytes = gguf
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor '{name}' has no data"))?;
    let n = info.n_elements();
    let v = match info.ggml_type {
        GgmlType::F32 => bytemuck::cast_slice(bytes).to_vec(),
        GgmlType::F16 => {
            let h: &[u16] = bytemuck::cast_slice(bytes);
            h.iter().map(|&b| f16_bits_to_f32(b)).collect()
        }
        GgmlType::BF16 => {
            let h: &[u16] = bytemuck::cast_slice(bytes);
            h.iter().map(|&b| bf16_bits_to_f32(b)).collect()
        }
        _ => {
            return Err(format!(
                "tensor '{name}': expected F32/F16/BF16 small tensor, got {:?}",
                ggml_type_name(info.ggml_type)
            ))
        }
    };
    if v.len() != n {
        return Err(format!("tensor '{name}': len {} != {n}", v.len()));
    }
    Ok(v)
}

fn ggml_type_name(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::BF16 => "BF16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q8_1 => "Q8_1",
        GgmlType::Q2_K => "Q2_K",
        GgmlType::Q3_K => "Q3_K",
        GgmlType::Q4_K => "Q4_K",
        GgmlType::Q5_K => "Q5_K",
        GgmlType::Q6_K => "Q6_K",
        GgmlType::Q8_K => "Q8_K",
        GgmlType::Q2_0 => "Q2_0",
        GgmlType::PTQ1_0 => "PTQ1_0",
        GgmlType::I8 => "I8",
        GgmlType::I16 => "I16",
        GgmlType::I32 => "I32",
        GgmlType::I64 => "I64",
        GgmlType::F64 => "F64",
    }
}

fn upload_f32(stream: &Arc<CudaStream>, gguf: &GgufFile, name: &str) -> Result<F32W, String> {
    let host = f32_bytes_from_gguf(gguf, name)?;
    let len = host.len();
    let dev = stream
        .clone_htod(&host)
        .map_err(|e| format!("upload '{name}': {e}"))?;
    Ok(F32W { dev, len })
}

fn upload_quant(
    stream: &Arc<CudaStream>,
    gguf: &GgufFile,
    name: &str,
    rows: usize,
    n: usize,
) -> Result<QuantW, String> {
    let info = gguf
        .tensor_info(name)
        .ok_or_else(|| format!("tensor '{name}' not found"))?;
    let bytes = gguf
        .tensor_slice(name)
        .ok_or_else(|| format!("tensor '{name}' has no data"))?;
    let (q4, block_bytes) = match info.ggml_type {
        GgmlType::Q4_K => (true, 144usize),
        GgmlType::Q6_K => (false, 210usize),
        t => {
            return Err(format!(
                "tensor '{name}': expected Q4_K/Q6_K, got {}",
                ggml_type_name(t)
            ))
        }
    };
    let blocks_per_row = n / 256;
    let expected = rows * blocks_per_row * block_bytes;
    if bytes.len() != expected {
        return Err(format!(
            "tensor '{name}': {} bytes != expected {expected} (rows={rows} n={n})",
            bytes.len()
        ));
    }
    let dev = stream
        .clone_htod(bytes)
        .map_err(|e| format!("upload '{name}': {e}"))?;
    Ok(QuantW {
        dev,
        rows,
        n,
        blocks_per_row,
        q4,
    })
}

/// Load all decode-path weights from the GGUF mmap onto the device.
///
/// Uploads ~14.7 GiB (quantized tensors verbatim + small f32). Assumes the
/// GPU is otherwise idle (correctness-of-measurement + VRAM budget).
pub fn load_weights_gpu(
    gguf: &GgufFile,
    cfg: &Qwen38DenseConfig,
    stream: &Arc<CudaStream>,
) -> Result<Qwen38DenseWeightsGpu, String> {
    let n = cfg.n_embd;
    let q_dim = cfg.n_head * cfg.head_dim;
    let kvd = cfg.n_kv_head * cfg.head_dim;
    // qkv layout (pinned by the real tensor): [q(n_k*hd) | k(n_k*hd) | v(n_v*hd)]
    // = 2048 + 2048 + 6144 = 10240 for the dbirks 27B (the weights-struct
    // doc comment's formula has the x2 on the wrong term — the CPU forward's
    // q_dim/k_dim/v_dim split is the truth).
    let l_qkv_out = 2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
    let l_z_out = cfg.n_v_heads * cfg.head_v_dim;

    let token_embd = upload_quant(stream, gguf, "token_embd.weight", cfg.vocab_size, n)?;
    let output_norm = upload_f32(stream, gguf, "output_norm.weight")?;
    let lm_head = upload_quant(stream, gguf, "output.weight", cfg.vocab_size, n)?;

    let mut layers = Vec::with_capacity(cfg.n_layer);
    for i in 0..cfg.n_layer {
        let is_linear = cfg.layer_types[i] == Qwen38LayerType::Deltanet;
        let blk = format!("blk.{i}.");
        let input_norm = upload_f32(stream, gguf, &format!("{blk}attn_norm.weight"))?;
        let post_attn_norm = upload_f32(
            stream,
            gguf,
            &format!("{blk}post_attention_norm.weight"),
        )?;
        let ffn_gate = upload_quant(
            stream,
            gguf,
            &format!("{blk}ffn_gate.weight"),
            cfg.mlp_hidden,
            n,
        )?;
        let ffn_up = upload_quant(
            stream,
            gguf,
            &format!("{blk}ffn_up.weight"),
            cfg.mlp_hidden,
            n,
        )?;
        let ffn_down = upload_quant(stream, gguf, &format!("{blk}ffn_down.weight"), n, cfg.mlp_hidden)?;
        let layer = if is_linear {
            Qwen38LayerGpu {
                input_norm,
                post_attn_norm,
                qkv: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}attn_qkv.weight"),
                    l_qkv_out,
                    n,
                )?),
                z: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}attn_gate.weight"),
                    l_z_out,
                    n,
                )?),
                alpha: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}ssm_alpha.weight"),
                    cfg.n_v_heads,
                    n,
                )?),
                beta: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}ssm_beta.weight"),
                    cfg.n_v_heads,
                    n,
                )?),
                conv1d: Some(upload_f32(stream, gguf, &format!("{blk}ssm_conv1d.weight"))?),
                a_log: Some(upload_f32(stream, gguf, &format!("{blk}ssm_a"))?),
                dt_bias: Some(upload_f32(stream, gguf, &format!("{blk}ssm_dt.bias"))?),
                ssm_norm: Some(upload_f32(stream, gguf, &format!("{blk}ssm_norm.weight"))?),
                ssm_out: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}ssm_out.weight"),
                    n,
                    l_z_out,
                )?),
                wq: None,
                wk: None,
                wv: None,
                wo: None,
                q_norm: None,
                k_norm: None,
                ffn_gate,
                ffn_up,
                ffn_down,
            }
        } else {
            Qwen38LayerGpu {
                input_norm,
                post_attn_norm,
                qkv: None,
                z: None,
                alpha: None,
                beta: None,
                conv1d: None,
                a_log: None,
                dt_bias: None,
                ssm_norm: None,
                ssm_out: None,
                wq: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}attn_q.weight"),
                    2 * q_dim,
                    n,
                )?),
                wk: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}attn_k.weight"),
                    kvd,
                    n,
                )?),
                wv: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}attn_v.weight"),
                    kvd,
                    n,
                )?),
                wo: Some(upload_quant(
                    stream,
                    gguf,
                    &format!("{blk}attn_output.weight"),
                    n,
                    q_dim,
                )?),
                q_norm: Some(upload_f32(stream, gguf, &format!("{blk}attn_q_norm.weight"))?),
                k_norm: Some(upload_f32(stream, gguf, &format!("{blk}attn_k_norm.weight"))?),
                ffn_gate,
                ffn_up,
                ffn_down,
            }
        };
        layers.push(layer);
    }
    Ok(Qwen38DenseWeightsGpu {
        token_embd,
        output_norm,
        lm_head,
        layers,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode state + the forward
// ─────────────────────────────────────────────────────────────────────────────

const ZERO_U64: [u64; 1] = [0];
const ZERO_U64_16: [u64; 16] = [0; 16];

/// GPU-resident decode state: KV caches (attention layers), recurrent +
/// conv states (GDN layers).
pub struct Qwen38DecodeState {
    /// [attn_layer][ctx_len * kvd] key caches.
    pub keys: Vec<CudaSlice<f32>>,
    /// [attn_layer][ctx_len * kvd] value caches.
    pub values: Vec<CudaSlice<f32>>,
    /// [gdn_layer][n_v_heads * head_k_dim * head_v_dim].
    pub recurrent: Vec<CudaSlice<f32>>,
    /// [gdn_layer][l_qkv_out * conv_kernel].
    pub conv: Vec<CudaSlice<f32>>,
}

/// Issue 742 T9.9 — scratch for the p-row batched verify chunk (p <=
/// [`QWEN38_VERIFY_MAX_P`]). All buffers row-major `[p, dim]`, allocated
/// once at construction (~210 MB + the GDN snapshot ~154 MB at the dbirks
/// dims); the attention partials are sized for the full context.
pub struct Qwen38VerifyScratch {
    pub tokens_dev: CudaSlice<i32>, // [16]
    pub xb: CudaSlice<f32>,         // [16][n_embd] residual stream
    pub xb_res: CudaSlice<f32>,     // [16][n_embd] residual snapshot
    pub xq: CudaSlice<i8>,          // [16][mlp_hidden] quantized rows
    pub xs: CudaSlice<f32>,         // [16][mlp_hidden/16]
    pub xsum: CudaSlice<i32>,       // [16][mlp_hidden/16]
    pub y: CudaSlice<f32>,          // [16][mlp_hidden] GEMV outputs
    pub qkv: CudaSlice<f32>,        // [16][l_qkv_out]
    pub qkv_exp: CudaSlice<f32>,    // [16][l_exp]
    pub z: CudaSlice<f32>,          // [16][d_inner]
    pub rec_out: CudaSlice<f32>,    // [16][d_inner]
    pub a_raw: CudaSlice<f32>,      // [16][n_v_heads]
    pub b_raw: CudaSlice<f32>,
    pub beta: CudaSlice<f32>,
    pub decay: CudaSlice<f32>,
    pub qg: CudaSlice<f32>,         // [16][2*q_dim]
    pub q: CudaSlice<f32>,          // [16][q_dim]
    pub gate: CudaSlice<f32>,       // [16][q_dim]
    pub q_normed: CudaSlice<f32>,   // [16][q_dim]
    pub k: CudaSlice<f32>,          // [16][kvd]
    pub k_normed: CudaSlice<f32>,   // [16][kvd]
    pub vv: CudaSlice<f32>,         // [16][kvd]
    pub attn_out: CudaSlice<f32>,   // [16][q_dim]
    pub mlp_gate: CudaSlice<f32>,   // [16][mlp_hidden]
    pub mlp_up: CudaSlice<f32>,     // [16][mlp_hidden]
    pub mlp_hidden: CudaSlice<f32>, // [16][mlp_hidden]
    pub logits: CudaSlice<f32>,     // [16][vocab]
    pub argmax_res: CudaSlice<u64>, // [16]
    /// Attention split partials (rows layout): [16][n_head][n_chunks].
    pub part_m: CudaSlice<f32>,
    pub part_l: CudaSlice<f32>,
    /// [16][n_head][n_chunks][head_dim].
    pub part_out: CudaSlice<f32>,
    /// Issue 742 T9.14 — two-pass verify-attention scratch: per-tile
    /// flash stats `[16][n_head][n_stat_tiles]` (m, s) + the merged
    /// frozen (M, L) `[16][n_head]`. Allocated once at construction —
    /// BEFORE any capture region (the T1.7 rule).
    pub stat_m: CudaSlice<f32>,
    pub stat_l: CudaSlice<f32>,
    pub mrg_m: CudaSlice<f32>,
    pub mrg_l: CudaSlice<f32>,
    /// GDN rollback snapshots (same shapes as the state buffers).
    pub snap_recurrent: Vec<CudaSlice<f32>>,
    pub snap_conv: Vec<CudaSlice<f32>>,
}

/// The verify chunk's maximum row count (the loop's K cap).
///
/// ⚠ **Do not raise this expecting wider chunks to work.** It was raised to 64
/// once (Issue 754) on the theory that the `_rows` kernels take `p` as a
/// runtime argument, so widening `p` would buy context-ingestion throughput
/// with no new kernel. [Bench 753] **refuted that at p=32**: the qwen38 Q4_K
/// verify stack has SIX independent `p <= 16` caps in three mechanisms, and
/// the default arm's is not a relaxable assert —
///
/// 1. `launch_gemv_q4k_rows_strict` (**default**) — `float acc0[16]; acc1[16]`
///    fixed register arrays; `lane` strides sub-blocks, not rows, so `p` lives
///    entirely in those 32 accumulators. p=64 would need 128 regs/thread
///    against a 255 limit, before weights/scalars/addresses.
/// 2. `launch_gemv_q4k_rows` (shape B) — `r = lane >> 1`, a lane→row map.
/// 3. `launch_gemv_q6k_q8x_rows` — as (1).
///
/// 4./5. The Q4_K + Q6_K mma verify GEMMs — exact-fold-order pins 64 f32
///    regs/thread (109 regs, 33% occupancy), already 1.3-1.7x SLOWER here.
/// 6. The fast q-group attention arm (`..._rows_qg`, the T9.11-T9.16 winner at
///    long context) — WIDENED by Issue 754 T6 to p <= 64 via grid.z 16-row
///    slices ([`riir_gpu::SPLITGQA_QG_ROWS_MAX_P`]; bit-identical at
///    p <= 16, grid.z == 1). See [`Self::verify_use_qg`].
///
/// Widening also worsens a second measured wall: T9.10 pinned the mma arm's
/// loss to a ~490 KB/SM x working set against ~112 KB L1, and the strict GEMV
/// spends `__ldcs` on every weight load (T9.16) precisely to protect its
/// ~121 KB x set. At p=64 that set is ~484 KB — the regime already measured as
/// fatal.
///
/// Raising this constant alone is therefore **inert**: it changes scratch
/// sizing only, and no arm consumes `p > 16`. See Issue 754 T1 for the shape
/// that could (a p-tiled row-group loop *inside* the existing kernel, which
/// keeps accumulators at 32 while amortizing the weight read).
///
/// [Bench 753]: ../../../.benchmarks/753_issue754_wide_p_ingest.md
pub const QWEN38_VERIFY_MAX_P: usize = 16;

/// Issue 754 P4-b — the wide-ingest chunk's row count.
///
/// The wide path pins p to EXACTLY this value: the T5 GEMM kernels stage all
/// 64 x rows per k-tile (Bench 772/783's gated contract), so every projection
/// site's quantized rows must be freshly written at full width — which a
/// full 64-row chunk guarantees. Ragged prompt tails ride the p <= 16 strict
/// path (the caller's split duty — see the fill helper in the P4-b bench).
/// Decode/verify are untouched: the T4 scope rule keeps their 0-bit-diff
/// pin, and the tolerance class lives inside the ingest path only.
pub const QWEN38_INGEST_WIDE_P: usize = 64;

/// Issue 755 T2 — the DFlash2 drafter's target tap layers (0-indexed
/// OUTPUTS; llama.cpp's GGUF `dflash.target_layers = [6,20,34,48,62]` is the
/// same taps expressed as the next layer's INPUT — see
/// `qwen38_dflash2`'s module doc). The single source of truth for BOTH
/// `forward_token_capture` prompt fills and the Issue-755 T2 verify-chunk
/// taps, so the ring the drafter sees is construction-identical between the
/// prompt phase and the loop phase.
pub const QWEN38_DFLASH2_TAP_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

/// The tap slot (ascending index into [`QWEN38_DFLASH2_TAP_LAYERS`) for a
/// layer, `None` when the layer is not a tap layer.
fn dflash2_tap_slot(layer: usize) -> Option<usize> {
    QWEN38_DFLASH2_TAP_LAYERS
        .iter()
        .position(|&t| t == layer)
}

/// Issue 755 T4 — the per-GDN-layer replay journal. One buffer per GDN layer,
/// sized `[QWEN38_VERIFY_MAX_P]` rows:
/// - `qkv`: the PRE-conv in_proj rows (captured before `conv1d_rows` — its
///   output is in-place), so the replay can re-run the conv-state shift;
/// - `exp`/`beta`/`decay`: the recurrence inputs after `expand_l2_rows`, so
///   the replay can re-run the state update.
///
/// The replay re-launches the SAME `conv1d_rows` + `recurrence_fused_rows`
/// kernels on the first `j` journaled rows — both evolve their state in
/// registers with a sequential per-token loop, so a `j`-row call is
/// bit-identical to the first `j` iterations of the `p`-row call that
/// captured the journal. No weight reads.
pub struct Qwen38GdnJournal {
    /// [gdn_layer][MAX_P * l_qkv_out] pre-conv qkv rows.
    pub qkv: Vec<CudaSlice<f32>>,
    /// [gdn_layer][MAX_P * l_exp] post-expand recurrence input rows.
    pub exp: Vec<CudaSlice<f32>>,
    /// [gdn_layer][MAX_P * n_v_heads] beta rows.
    pub beta: Vec<CudaSlice<f32>>,
    /// [gdn_layer][MAX_P * n_v_heads] decay rows.
    pub decay: Vec<CudaSlice<f32>>,
}

/// Plan 556 Stage 1 (Issue 780 U2) — per-lane decode state arenas. One flat
/// buffer per layer, `n` lane slots each; the per-lane kernels consume the
/// lane's slot through a zero-copy [`CudaSlice::slice`] view, so lane mode
/// needs NO staging copies (the plan's dtod staging estimate assumed
/// separate journal-style buffers — lane state lives where the kernels read
/// it). F32 KV only in stage 1: the f16 hatch is a solo-path bake (the PTX
/// module + buffer sizing both key off it) and lanes are a new substrate.
///
/// Site 3 of katgpt-core's `GraphStablePool` family (Issue 800 C1): verdict
/// is discipline-adopted, storage-not-re-pointed — this set already IS the
/// lifetime-scoped flat-arena pattern (allocate-once `alloc_zeros`, fixed
/// `n`, no free list, no slot churn), and the captured-graph cache lives
/// INSIDE the set so whole-set replacement drops the baked pointers with
/// the arenas — the property a pool wrapper could not own.
pub struct Qwen38LaneSet {
    /// Lane count == the packed row count of every lane forward.
    pub n: usize,
    /// Per-lane KV capacity in positions (every lane's `pos + 1` must stay
    /// within it; lanes are decode-only sessions, typically far shorter
    /// than the solo ctx budget, so the arenas stay affordable —
    /// `n * lane_ctx * kvd * 2 * 4` bytes per attention layer ≈ 8.4 MB
    /// per lane per layer at lane_ctx 1024 on the dbirks dims).
    pub lane_ctx: usize,
    /// [attn_layer][n * lane_ctx * kvd] per-lane key caches.
    pub keys: Vec<CudaSlice<f32>>,
    /// [attn_layer][n * lane_ctx * kvd] per-lane value caches.
    pub values: Vec<CudaSlice<f32>>,
    /// [gdn_layer][n * n_v_heads * head_k_dim * head_v_dim].
    pub recurrent: Vec<CudaSlice<f32>>,
    /// [gdn_layer][n * l_qkv_out * conv_kernel].
    pub conv: Vec<CudaSlice<f32>>,
    /// Stage 2 — the pre-chunk GDN snapshot arenas (the rollback medium of
    /// the fail-closed commit contract), allocated lazily by
    /// [`Qwen38DenseForward::enable_lanes_snapshots`]. Same shapes as the
    /// live arenas; a whole-arena (or per-lane-slice) dtod restores every
    /// lane to its chunk's base state.
    pub snap_recurrent: Option<Vec<CudaSlice<f32>>>,
    /// Stage 2 — per-GDN-layer conv snapshot, same contract.
    pub snap_conv: Option<Vec<CudaSlice<f32>>>,
    /// Stage 3 — per-lane base positions (`[n] i32`), uploaded per cycle
    /// before the graph launch; the captured lane graphs read them at kernel
    /// runtime via the `_devpos` twins (a scalar position would bake into
    /// the graph — the solo T9.12 contract, per lane).
    pub(crate) pos_dev: CudaSlice<i32>,
    /// Stage 3 (mechanism F) — the captured per-lane verify graphs, keyed by
    /// `(n, k)`. A captured graph bakes DEVICE POINTERS (the lane arenas,
    /// the verify scratch, `pos_dev`), so the cache lives INSIDE the lane
    /// set: `enable_lanes` replacing a set (or [`Self::disable_lanes`]) drops
    /// the graphs together with the pointers they bake — replay against
    /// freed pointers is structurally impossible.
    pub(crate) graphs: std::collections::HashMap<(usize, usize), cudarc::driver::safe::CudaGraph>,
    /// Keys whose capture failed — permanent eager fallback for those (reset
    /// with the lane set: a new set may capture fine).
    pub(crate) graph_failed: std::collections::HashSet<(usize, usize)>,
}

/// Issue 742 T9.12 - the verify chunk's position source. `Live` passes
/// `base_pos` as a scalar kernel arg (the eager path, bit-identical to the
/// pre-T9.12 sequence); `Dev` reads it from `pos_dev` at kernel runtime via
/// the `_devpos` kernel twins (the captured-graph path - the scalar would
/// bake into the graph). In `Dev` mode the attention grid is pinned to the
/// fixed max [`Self::attn_n_chunks`] (dead chunks write neutral partials).
#[derive(Clone, Copy)]
enum VerifyPos {
    Live(usize),
    Dev,
}

/// Plan 556 Stage 3 — the lane verify chunk's position source. `Live` passes
/// each lane's `base_pos` as a scalar kernel arg (the eager Stage-2 path);
/// `Dev` reads per-lane positions from the lane set's `pos_dev` `[n] i32`
/// buffer via the `_devpos` kernel twins (`pos_dev.slice(l..l+1)` per lane
/// launch) — the captured-graph path, where a scalar would bake. `Dev`
/// additionally pins every lane to the ROWS attention arm at the FIXED max
/// `attn_n_chunks` grid (dead chunks write neutral partials, the T9.6
/// contract) — the capture guard (`lane_ctx·kvd·8B < 96 MB`) makes the qg
/// arm unreachable for the lane set's whole lifetime, so the pinned arm is
/// the arm.
enum LaneVerifyPos<'a> {
    Live(&'a [usize]),
    Dev(&'a CudaSlice<i32>),
}

fn alloc_verify_scratch(
    stream: &Arc<CudaStream>,
    cfg: &Qwen38DenseConfig,
    l_qkv_out: usize,
    l_exp: usize,
    q_dim: usize,
    kvd: usize,
    attn_n_chunks: usize,
    n_stat_tiles: usize,
    // Issue 754 P4-b — row count (16 for the verify scratch, 64 for the
    // wide-ingest scratch; every buffer is row-major [p, dim]).
    p: usize,
) -> Result<Qwen38VerifyScratch, String> {
    let n = cfg.n_embd;
    let mlp = cfg.mlp_hidden;
    let a_f32 = |len: usize| -> Result<CudaSlice<f32>, String> {
        stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| format!("alloc: {e}"))
    };
    let v = Qwen38VerifyScratch {
        tokens_dev: stream
            .alloc_zeros::<i32>(p)
            .map_err(|e| format!("alloc: {e}"))?,
        xb: a_f32(p * n)?,
        xb_res: a_f32(p * n)?,
        xq: stream
            .alloc_zeros::<i8>(p * mlp)
            .map_err(|e| format!("alloc: {e}"))?,
        xs: a_f32(p * mlp.div_ceil(16))?,
        xsum: stream
            .alloc_zeros::<i32>(p * mlp.div_ceil(16))
            .map_err(|e| format!("alloc: {e}"))?,
        y: a_f32(p * mlp)?,
        qkv: a_f32(p * l_qkv_out)?,
        qkv_exp: a_f32(p * l_exp)?,
        z: a_f32(p * cfg.d_inner)?,
        rec_out: a_f32(p * cfg.d_inner)?,
        a_raw: a_f32(p * cfg.n_v_heads)?,
        b_raw: a_f32(p * cfg.n_v_heads)?,
        beta: a_f32(p * cfg.n_v_heads)?,
        decay: a_f32(p * cfg.n_v_heads)?,
        qg: a_f32(p * 2 * q_dim)?,
        q: a_f32(p * q_dim)?,
        gate: a_f32(p * q_dim)?,
        q_normed: a_f32(p * q_dim)?,
        k: a_f32(p * kvd)?,
        k_normed: a_f32(p * kvd)?,
        vv: a_f32(p * kvd)?,
        attn_out: a_f32(p * q_dim)?,
        mlp_gate: a_f32(p * mlp)?,
        mlp_up: a_f32(p * mlp)?,
        mlp_hidden: a_f32(p * mlp)?,
        logits: a_f32(p * cfg.vocab_size)?,
        argmax_res: stream
            .alloc_zeros::<u64>(p)
            .map_err(|e| format!("alloc: {e}"))?,
        part_m: a_f32(p * cfg.n_head * attn_n_chunks)?,
        part_l: a_f32(p * cfg.n_head * attn_n_chunks)?,
        part_out: a_f32(p * cfg.n_head * attn_n_chunks * cfg.head_dim)?,
        stat_m: a_f32(p * cfg.n_head * n_stat_tiles)?,
        stat_l: a_f32(p * cfg.n_head * n_stat_tiles)?,
        mrg_m: a_f32(p * cfg.n_head)?,
        mrg_l: a_f32(p * cfg.n_head)?,
        snap_recurrent: {
            let mut v = Vec::new();
            for _ in 0..cfg.layer_types.iter().filter(|t| **t == Qwen38LayerType::Deltanet).count() {
                v.push(a_f32(cfg.n_v_heads * cfg.head_k_dim * cfg.head_v_dim)?);
            }
            v
        },
        snap_conv: {
            let mut v = Vec::new();
            for _ in 0..cfg.layer_types.iter().filter(|t| **t == Qwen38LayerType::Deltanet).count() {
                v.push(a_f32(l_qkv_out * cfg.conv_kernel)?);
            }
            v
        },
    };
    Ok(v)
}

/// Traced forward result: (argmax token, per-layer hidden taps, final logits).
pub type TracedForward = (u32, Vec<Vec<f32>>, Vec<f32>);

/// Whole-model dense decode forward on one CUDA stream.
///
/// One `forward_token` call enqueues the full 64-layer stack and returns the
/// greedy argmax token (device-side argmax, one 8-byte readback + sync).
pub struct Qwen38DenseForward {
    pub cfg: Qwen38DenseConfig,
    pub weights: Qwen38DenseWeightsGpu,
    pub state: Qwen38DecodeState,
    pub ctx_len: usize,
    /// Issue 753 — KV cache dtype arm. `true` = f16 halves (2× ctx headroom,
    /// halved KV traffic; tolerance-class numerics change, gates declared in
    /// Issue 753). `false` = f32 (byte-identical pre-753 default). Resolved
    /// ONCE per process from `QWEN38_KV_DTYPE` — the PTX module and the KV
    /// buffer sizing both bake it.
    pub kv_f16: bool,
    stream: Arc<CudaStream>,
    dense: DenseKernels,
    dn: DeltanetKernels,
    attn: AttentionKernels,
    ew: ElementwiseKernels,
    // scratch (persistent, sized at construction)
    x: CudaSlice<f32>,          // [n_embd] residual stream
    x_res: CudaSlice<f32>,      // [n_embd] residual snapshot
    x_norm: CudaSlice<f32>,     // [n_embd] normed input
    y: CudaSlice<f32>,          // [n_embd] GEMV output
    xq: CudaSlice<i8>,          // [mlp_hidden] quantized activation (max dim)
    xs: CudaSlice<f32>,         // [mlp_hidden/16 + 1]
    xsum: CudaSlice<i32>,       // [mlp_hidden/16 + 1]
    qkv: CudaSlice<f32>,        // [l_qkv_out] conv input/output
    qkv_exp: CudaSlice<f32>,    // [3 * n_v_heads * head_k_dim]
    z_buf: CudaSlice<f32>,      // [d_inner]
    rec_out: CudaSlice<f32>,    // [d_inner]
    rec_normed: CudaSlice<f32>, // [d_inner]
    a_raw: CudaSlice<f32>,      // [n_v_heads]
    b_raw: CudaSlice<f32>,      // [n_v_heads]
    beta: CudaSlice<f32>,       // [n_v_heads]
    decay: CudaSlice<f32>,      // [n_v_heads]
    qg: CudaSlice<f32>,         // [2 * q_dim]
    q: CudaSlice<f32>,          // [q_dim]
    gate: CudaSlice<f32>,       // [q_dim]
    q_normed: CudaSlice<f32>,   // [q_dim]
    k: CudaSlice<f32>,          // [kvd]
    k_normed: CudaSlice<f32>,   // [kvd]
    v: CudaSlice<f32>,          // [kvd]
    attn_out: CudaSlice<f32>,   // [q_dim]
    // Issue 742 split-KV flash decode: partial scratch (sized at construction
    // for the fixed graph grid; every slot rewritten on every launch).
    attn_part_m: CudaSlice<f32>,   // [n_head * n_chunks_max]
    attn_part_l: CudaSlice<f32>,   // [n_head * n_chunks_max]
    attn_part_out: CudaSlice<f32>, // [n_head * n_chunks_max * head_dim]
    mlp_gate: CudaSlice<f32>,   // [mlp_hidden]
    mlp_up: CudaSlice<f32>,     // [mlp_hidden]
    mlp_hidden: CudaSlice<f32>, // [mlp_hidden]
    logits: CudaSlice<f32>,     // [vocab]
    argmax_res: CudaSlice<u64>, // [1]
    // T5 graph path: device-side token/pos + the captured decode graph.
    token_dev: CudaSlice<i32>,
    pos_dev: CudaSlice<i32>,
    /// True while capturing (kernels use the _devpos variants + token_dev).
    capturing: bool,
    graph: Option<cudarc::driver::safe::CudaGraph>,
    /// Split-KV decode: positions per chunk (multiple of head_dim, so tile
    /// boundaries align with chunk boundaries — the launcher contract).
    attn_chunk_len: usize,
    /// Fixed chunk-column count for the graph grid: ceil(ctx_len / chunk).
    attn_n_chunks: usize,
    /// Issue 742 T9.13 - the verify path's tile-level split (QG ARM ONLY -
    /// the arm where the attn-only sweep measured the win): the serial
    /// per-chunk tile walk subdivided across subx more blocks at
    /// `attn_chunk_len / QWEN38_VERIFY_ATTN_SUB`. The decode path's
    /// `attn_chunk_len` (and its stream pins) is UNTOUCHED; the rows arm
    /// (short ctx) stays at `attn_chunk_len` (measured flat).
    verify_attn_chunk_len: usize,
    /// Fixed verify chunk-column count (the T9.12 graph pin): ceil(ctx_len /
    /// verify_attn_chunk_len).
    verify_attn_n_chunks: usize,
    /// Issue 742 T9.14 — the two-pass flash restructure of the qg arm's
    /// verify-attention walk (pass A tile-parallel stats → merge → pass
    /// B frozen-max weighted V-accumulate → the unchanged combine).
    /// MEASURED NEGATIVE at 20K (+18% ms/chunk — the duplicated score
    /// phase; Bench 745 T9.14) — OPT-IN `QWEN38_VERIFY_ATTN_2P=1`
    /// artifact (the T9.10 mma-GEMM precedent); the T9.13 qg kernels
    /// stay the default. Tolerance-class vs qg (frozen-max
    /// reassociation, ~2.2e-5 max_rel unit-measured) — gated by
    /// `qwen38_verify_2p_g1` (max-rel + argmax identity + devpos
    /// bit-identity).
    verify_attn_2p: bool,
    /// Fixed stat-tile column count (the T9.14 graph pin):
    /// ceil(ctx_len / 32) — the pass-A grid AND the stat stride.
    verify_stat_n_tiles: usize,
    /// Issue 742 T9.15 — the multi-accumulator dot arms (DOTMA=1 kernels):
    /// the score-phase 256-deep serial-FMA chain broken into FOUR
    /// independent 64-deep partial accumulators (the T9.14 re-diagnosis
    /// lever — the walk is per-lane-serial-FMA-bound, NOT running-max-
    /// chain-bound). Applies to BOTH verify arms (rows = short ctx,
    /// qg = long ctx). Tolerance class (fold-order reassociation ~1e-6,
    /// the T9.13 finer-chunk precedent) — gated by `qwen38_verify_qgma_g1`
    /// together with the model-level 0/256 argmax and loop-stream gates.
    /// resolves ONCE at construction (a captured graph bakes the arm).
    verify_attn_dotma: bool,
    /// A/B escape hatch (`QWEN38_ATTN_SPLIT=0` routes back to the serial
    /// kernel — the parity/wall-clock comparison apparatus).
    attn_split: bool,
    /// GQA-fused split decode (Bench 732 §6 headroom levers: coalesced Q·K +
    /// GQA-amortized V). Default ON; `QWEN38_ATTN_GQA=0` routes back to the
    /// per-head split. Requires GQA group ≤ 8 + head_dim ≤ 256 (else falls
    /// back to the per-head split automatically).
    attn_gqa: bool,
    /// Fused single-pass GDN recurrence (Issue 742 base-decode lever:
    /// 1R+1W state traffic vs the parallel kernel's 3R+2W). Default ON;
    /// `QWEN38_DN_FUSED=0` routes back to `recurrence_f32_parallel`.
    /// Requires head_k_dim % 32 == 0 (else falls back).
    dn_fused: bool,
    /// Fused rmsnorm→q8-quantize (Issue 742 T9.7 follow-on, Bench 734 ranked
    /// lever 1): one kernel replaces the `rmsnorm_f32` + `qwen38_quant_x_q8`
    /// pair at every plain-rmsnorm site (all 129 feed ONLY the quantizer).
    /// Bit-identical (phases 1-3 verbatim `rmsnorm_f32`, phase 4 the exact
    /// quant ops on the same stored expression). Default ON;
    /// `QWEN38_RNQ_FUSED=0` restores the two-kernel pair.
    rnq_fused: bool,
    /// T9.7c (Bench 736): constexpr-specialized fused kernel at n==5120
    /// (all 129 sites) — fully-unrolled data loops kill the rolled-loop
    /// latency serialization (the generic fused kernel's 7.36 µs/site is
    /// memory latency, ~63x its bandwidth floor). Bit-identical;
    /// `QWEN38_RNQ_FAST=0` restores the generic fused kernel.
    rnq_fast: bool,
    /// Issue 742 lever-2 (Bench 737): the GDN post-recurrence chain
    /// (per-head rmsnorm + silu z-gate + q8 quantize) fused into one kernel
    /// — 144 launches/token -> 48. Bit-identical by construction (see the
    /// kernel doc). Default ON; `QWEN38_ZGQ_FUSED=0` restores the 3-kernel
    /// trio. Also OFF when head_v_dim fails the kernel's shape gates
    /// (% 16 != 0 or > 256).
    zgq_fused: bool,
    /// Issue 755 — the fused layer-BOUNDARY kernel (MTPLX PR #335 row 48 +
    /// corpus B63 `deferred-residual-fused-into-next-layer-norm`):
    /// residual_add + snapshot copy + the next site's rmsnorm+q8-quantize
    /// in ONE kernel at every decode/verify boundary — kills 2 of 3
    /// launches + the full x DRAM round-trip per site (128 sites/token
    /// decode, ×p in verify). Bit-identical by construction (kernel doc).
    /// Default ON; `QWEN38_RES_NQ_FUSED=0` restores the 3-kernel chain.
    /// Requires n_embd == 5120 (the constexpr-specialized kernel; all
    /// decode/verify norm sites are n_embd=5120 on this model).
    res_nq_fused: bool,
    /// Issue 742 T9.9 - the p-row verify scratch + GDN snapshots.
    verify: Qwen38VerifyScratch,
    /// Issue 755 T2 — the verify-chunk feature taps (the DFlash2 ring feed):
    /// `[QWEN38_VERIFY_MAX_P][5*n_embd]` f32, row-major `[p][25600]` with one
    /// `n_embd` slot per tap layer in ascending layer order (the exact layout
    /// `DFlash2GpuDrafter::inject_positions` consumes). Written ONLY by the
    /// eager tap chunk ([`Self::forward_verify_chunk_taps`]) — the captured
    /// graph arm does not tap (Issue 755 T2's slice-B exemption). Allocated
    /// once by [`Self::enable_verify_taps`].
    verify_taps_dev: Option<CudaSlice<f32>>,
    /// Issue 755 T4 (the reopen lever) — per-GDN-layer journal of the chunk's
    /// recurrence inputs, captured by [`Self::verify_gdn_layer`] when enabled
    /// via [`Self::enable_gdn_journal`]. Consumed by
    /// [`Self::verify_advance_gdn_replay`] to advance the GDN state by the
    /// accepted prefix WITHOUT re-running the weight-reading stack (the
    /// lucebox `gdn-transition-replay-log-commit-many` pattern — the rewind
    /// advance's ~31 ms fixed cost becomes ~1 ms of state math).
    gdn_journal: Option<Qwen38GdnJournal>,
    /// Plan 556 Stage 1 (Issue 780 U2) — per-lane decode state arenas,
    /// allocated by [`Self::enable_lanes`]. Lane mode is a SEPARATE decode
    /// substrate: the packed row-agnostic ops ride the verify `_rows`
    /// scratch at p = n, the per-lane stateful ops launch the solo p=1
    /// kernels on slice views of these arenas. `self.state` (the solo
    /// request's KV/GDN) is untouched by lane mode, and vice versa.
    lanes: Option<Qwen38LaneSet>,
    /// Plan 556 Stage 2 — `Some(k)` between a successful
    /// [`Self::forward_lanes_verify`] and its [`Self::lanes_commit`]: the
    /// per-lane journal + snapshot are live and MUST be consumed (commit or
    /// the next verify's own re-snapshot) before any other chunk runs. The
    /// gate is fail-closed against the forgotten-commit protocol violation
    /// (a second chunk on un-rolled-back state would snapshot the
    /// ALREADY-ADVANCED state and silently mis-commit).
    lanes_commit_ready: Option<usize>,
    /// Issue 742 T9.10 - the m8n8k16-s8 mma verify GEMM (the x-side L1
    /// wall fix; bit-identical to the strict rows GEMVs).
    verify_mma: VerifyMmaKernels,
    /// Plan 551 / Issue 773 T2 — the tensor-core score-phase arm (opt-in
    /// `QWEN38_VERIFY_ATTN_MMA=1`): replaces the qg verify-attention SCORE
    /// phase with the 3xtf32 mma kernel (everything else verbatim; the
    /// combine unchanged). None unless the knob resolved on AND the
    /// geometry passed the kernel's gates (head_dim == 256, g <= 8).
    /// Resolved ONCE at construction (a captured graph bakes the arm).
    attn_mma: Option<AttentionScoreMmaKernels>,
    /// The knob's resolved state (drives the verify match arms; true
    /// implies attn_mma is Some). Arm: 1 = T-a6, 2 = T-a24.
    verify_attn_mma: bool,
    /// The arm selector (read at construction; only meaningful when
    /// verify_attn_mma).
    verify_attn_mma_arm24: bool,
    /// Issue 742 T9.12 - the captured verify-chunk graphs, keyed by
    /// (p, use_qg): ONE capture replays at EVERY context position (pos
    /// rides `pos_dev`, the attention grid is the fixed max
    /// `verify_attn_n_chunks` with dead chunks writing neutral partials).
    /// Key is `(p, use_qg, ingest)` — the third axis is Issue 754 T2's ingest
    /// tail (lm_head last-row-only), so ingest and verify graphs stay
    /// distinct captures. Bounded by 2 * 2 * 16 keys.
    verify_graphs: std::collections::HashMap<
        (usize, bool, bool),
        cudarc::driver::safe::CudaGraph,
    >,
    /// Keys whose capture failed - permanent eager fallback for those.
    verify_graph_failed: std::collections::HashSet<(usize, bool, bool)>,
    /// Issue 742 T4 - the single-stream prefix KV+GDN cache (whole-prefix
    /// GDN checkpoints keyed by token prefix; the live KV buffers stay in
    /// place so the captured graphs remain valid - see the module doc).
    prefix_cache: crate::qwen38_prefix_cache::Qwen38PrefixCache,
    /// Issue 754 P4-b - the 64-row wide-ingest scratch, allocated lazily on
    /// the first wide chunk (outside any capture region; the wide path is
    /// eager-only in P4-b, so there is no capture interaction). `None` until
    /// then; while a wide chunk runs, the p=16 scratch parks here and
    /// `self.verify` holds the wide buffers (swapped back on every exit
    /// path).
    wide_scratch: Option<Qwen38VerifyScratch>,
    /// True while the wide scratch is swapped into `self.verify`. Routes
    /// [`Self::gemv_quant_rows`] to the T5 tolerance-class GEMMs (the
    /// ingest-path-only arm — the T4 scope rule: decode/verify keep the
    /// strict bit-identical GEMVs). Never true during graph capture or
    /// decode.
    wide_ingest_active: bool,
}



/// Issue 753 — host-side f16↔f32 bit conversion (RN-even), the exact twin
/// of the CUDA `kv_f16_to_f32` / `kv_f32_to_f16` helpers in
/// `cudarc_kernels/attention.rs` (cudarc's nvrtc has no include dirs, so
/// both sides ship manual bit math). The GPU parity gates pin the two
/// implementations to bit-for-bit agreement.
pub mod kv_f16_bits {
    /// f16 bits → f32 (subnormals normalized; NaN quieted).
    pub fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h as u32) & 0x8000) << 16;
        let exp = ((h as u32) >> 10) & 0x1f;
        let mut mant = (h as u32) & 0x3ff;
        if exp == 0x1f {
            let payload = if mant != 0 { 0x400_000 | (mant << 13) } else { 0 };
            return f32::from_bits(sign | 0x7f80_0000 | payload);
        }
        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign);
            }
            let mut s = 0;
            while mant & 0x400 == 0 {
                mant <<= 1;
                s += 1;
            }
            return f32::from_bits(sign | ((113 - s) << 23) | ((mant & 0x3ff) << 13));
        }
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }

    /// f32 → f16 bits, round-to-nearest-even (matches `__float2half_rn`).
    pub fn f32_to_f16(f: f32) -> u16 {
        let x = f.to_bits();
        let sign = (x >> 16) & 0x8000;
        let e = ((x >> 23) & 0xff) as i32;
        let m = x & 0x007f_ffff;
        if e == 0xff {
            let payload = if m != 0 { 0x0200 | (m >> 13) } else { 0 };
            return (sign | 0x7c00 | payload) as u16;
        }
        if e == 0 {
            // f32 zero/subnormal — below the f16 subnormal rounding floor.
            return sign as u16;
        }
        let he = e - 112;
        if he >= 0x1f {
            return (sign | 0x7c00) as u16; // overflow → Inf
        }
        if he <= 0 {
            if he < -10 {
                return sign as u16; // < 2^-25 → 0 (tie rounds to even = 0)
            }
            let mf = m | 0x0080_0000;
            let sh = (14 - he) as u32;
            let n = mf >> sh;
            let rem = mf & ((1u32 << sh) - 1);
            let half = 1u32 << (sh - 1);
            let nr = n
                + u32::from(rem > half || (rem == half && (n & 1) == 1));
            return (sign | nr) as u16;
        }
        let mut h = (sign | ((he as u32) << 10) | (m >> 13)) as u16;
        let rem = m & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1; // carry into the exponent is correct RN (up to Inf at 65520)
        }
        h
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Round-trip: every f16 bit pattern decodes and re-encodes exactly.
        #[test]
        fn f16_bits_round_trip_exact() {
            for h in 0u16..=u16::MAX {
                let f = f16_to_f32(h);
                // NaNs/Inf don't re-encode; skip exp==0x1f payloads
                if ((h >> 10) & 0x1f) == 0x1f {
                    continue;
                }
                assert_eq!(f32_to_f16(f), h, "h={h:#06x} f={f:e}");
            }
        }

        /// The f16 RN lattice: midpoints round to even, in-range values are
        /// exact, overflow carries to Inf exactly at 65520.
        #[test]
        fn f32_to_f16_rn_lattice() {
            assert_eq!(f32_to_f16(0.0), 0x0000);
            assert_eq!(f32_to_f16(-0.0), 0x8000);
            assert_eq!(f32_to_f16(1.0), 0x3c00);
            assert_eq!(f32_to_f16(-1.0), 0xbc00);
            // smallest subnormal + tie
            assert_eq!(f32_to_f16(2.0f32.powi(-24)), 0x0001);
            assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0x0000); // tie → even = 0
            assert_eq!(f32_to_f16(1.5 * 2.0f32.powi(-25)), 0x0001);
            // min normal, and subnormal carry to min normal
            assert_eq!(f32_to_f16(2.0f32.powi(-14)), 0x0400);
            assert_eq!(f32_to_f16(65504.0), 0x7bff);
            assert_eq!(f32_to_f16(65512.0), 0x7bff); // rounds DOWN to 65504
            assert_eq!(f32_to_f16(65520.0), 0x7c00); // tie → Inf
            assert_eq!(f32_to_f16(65536.0), 0x7c00); // overflow → Inf
            assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
            assert!(f32_to_f16(f32::NAN) & 0x7c00 == 0x7c00);
            // midpoint ties round to even in the normal range too
            assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00); // tie → even
            assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02); // tie → even
            assert_eq!(f32_to_f16(1.0 + 2.0 * 2.0f32.powi(-11)), 0x3c01); // exact
        }
    }
}

impl Qwen38DenseForward {
    /// Construct: load weights + allocate state + scratch on a fresh
    /// context/stream. GPU-heavy (~15 GiB upload); assumes an idle GPU.
    pub fn new(gguf_path: &Path, ctx_len: usize) -> Result<Self, String> {
        let gguf = GgufFile::open(gguf_path).map_err(|e| format!("open gguf: {e}"))?;
        let cfg = Qwen38DenseConfig::from_gguf(&gguf)?;
        let ctx = CudaContext::new(0).map_err(|e| format!("cuda init: {e}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("stream: {e}"))?;
        Self::with_parts(&gguf, cfg, ctx, stream, ctx_len)
    }

    /// Caller-provided context/stream variant (composability with the T1
    /// verify machinery later).
    pub fn with_parts(
        gguf: &GgufFile,
        cfg: Qwen38DenseConfig,
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        ctx_len: usize,
    ) -> Result<Self, String> {
        let weights = load_weights_gpu(gguf, &cfg, &stream)?;

        // Issue 753 — the KV dtype hatch: resolved ONCE per process (the PTX
        // module selection and the KV buffer sizing both bake it; a captured
        // graph would pin whatever arm constructed it).
        let kv_f16 = Self::kv_dtype_f16();

        let dense = DenseKernels::new(&ctx)?;
        let verify_mma = VerifyMmaKernels::new(&ctx)?;
        // Plan 551 / Issue 773 T2 — the mma score-phase arm: knob resolved
        // ONCE here (a captured graph bakes the arm), kernels compiled ONLY
        // when opted in (the extra nvrtc pass is seconds, not worth paying
        // on every construction). =1 T-a6 (warp-per-m-frag), =2 T-a24
        // (warp-per-(m-frag x n-frag)) — the plan's two pre-declared
        // tilings, picked by measured ms.
        let verify_attn_mma_arm = std::env::var("QWEN38_VERIFY_ATTN_MMA")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|v| *v == 1 || *v == 2);
        let verify_attn_mma = verify_attn_mma_arm.is_some()
            && cfg.head_dim == 256
            && cfg.n_head.is_multiple_of(cfg.n_kv_head)
            && cfg.n_head / cfg.n_kv_head <= 8;
        let attn_mma = if verify_attn_mma {
            Some(
                AttentionScoreMmaKernels::new(Arc::clone(&ctx), kv_f16)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let dn = DeltanetKernels::new(Arc::clone(&ctx)).map_err(|e| e.to_string())?;
        let attn =
            AttentionKernels::new_with_kv_dtype(Arc::clone(&ctx), kv_f16)
                .map_err(|e| e.to_string())?;
        let ew = ElementwiseKernels::new(Arc::clone(&ctx)).map_err(|e| e.to_string())?;

        let n = cfg.n_embd;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let mlp = cfg.mlp_hidden;
        // qkv layout (pinned by the real tensor): [q(n_k*hd) | k(n_k*hd) | v(n_v*hd)]
    // = 2048 + 2048 + 6144 = 10240 for the dbirks 27B (the weights-struct
    // doc comment's formula has the x2 on the wrong term — the CPU forward's
    // q_dim/k_dim/v_dim split is the truth).
    let l_qkv_out = 2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
        let l_exp = 3 * cfg.n_v_heads * cfg.head_k_dim;

        macro_rules! alloc_f32 {
            ($len:expr) => {
                stream
                    .alloc_zeros::<f32>($len)
                    .map_err(|e| format!("alloc: {e}"))?
            };
        }

        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut recurrent = Vec::new();
        let mut conv = Vec::new();
        for lt in &cfg.layer_types {
            match lt {
                Qwen38LayerType::Attention => {
                    // Issue 753 — f16 KV halves the cache bytes. The slice
                    // keeps the f32 type (device memory is bytes; the f16
                    // PTX module reinterprets), holding ctx_len*kvd HALVES =
                    // ctx_len*kvd/2 f32 words. kvd is n_kv_head*head_dim —
                    // always even for every GQA geometry this engine serves.
                    let kv_words = if kv_f16 { ctx_len * kvd / 2 } else { ctx_len * kvd };
                    keys.push(alloc_f32!(kv_words));
                    values.push(alloc_f32!(kv_words));
                }
                Qwen38LayerType::Deltanet => {
                    recurrent.push(alloc_f32!(
                        cfg.n_v_heads * cfg.head_k_dim * cfg.head_v_dim
                    ));
                    conv.push(alloc_f32!(l_qkv_out * cfg.conv_kernel));
                }
            }
        }
        let state = Qwen38DecodeState {
            keys,
            values,
            recurrent,
            conv,
        };

        // Allocate all scratch BEFORE the struct literal — `stream` moves into
        // the struct on construction, and the field macros reference it.
        let x = alloc_f32!(n);
        let x_res = alloc_f32!(n);
        let x_norm = alloc_f32!(n);
        let y = alloc_f32!(n);
        let xq = stream
            .alloc_zeros::<i8>(mlp)
            .map_err(|e| format!("alloc: {e}"))?;
        let xs = alloc_f32!(mlp / 16 + 1);
        let xsum = stream
            .alloc_zeros::<i32>(mlp / 16 + 1)
            .map_err(|e| format!("alloc: {e}"))?;
        let qkv = alloc_f32!(l_qkv_out);
        let qkv_exp = alloc_f32!(l_exp);
        let z_buf = alloc_f32!(cfg.d_inner);
        let rec_out = alloc_f32!(cfg.d_inner);
        let rec_normed = alloc_f32!(cfg.d_inner);
        let a_raw = alloc_f32!(cfg.n_v_heads);
        let b_raw = alloc_f32!(cfg.n_v_heads);
        let beta = alloc_f32!(cfg.n_v_heads);
        let decay = alloc_f32!(cfg.n_v_heads);
        let qg = alloc_f32!(2 * q_dim);
        let q = alloc_f32!(q_dim);
        let gate = alloc_f32!(q_dim);
        let q_normed = alloc_f32!(q_dim);
        let k = alloc_f32!(kvd);
        let k_normed = alloc_f32!(kvd);
        let v = alloc_f32!(kvd);
        let attn_out = alloc_f32!(q_dim);
        let mlp_gate = alloc_f32!(mlp);
        let mlp_up = alloc_f32!(mlp);
        let mlp_hidden = alloc_f32!(mlp);
        let logits = alloc_f32!(cfg.vocab_size);
        let argmax_res = stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| format!("alloc: {e}"))?;
        let token_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc: {e}"))?;
        let pos_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc: {e}"))?;

        // Issue 742 split-KV flash decode: config (env resolved ONCE at
        // construction — never per dispatch) + the partial scratch.
        let attn_chunk_len = {
            let raw = std::env::var("QWEN38_ATTN_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256);
            raw.div_ceil(cfg.head_dim).max(1) * cfg.head_dim
        };
        let attn_n_chunks = ctx_len.div_ceil(attn_chunk_len).max(1);
        // Issue 742 T9.13 - the verify-attention tile-level split: the
        // verify launches run the SAME kernels at `attn_chunk_len / sub`
        // (subx more blocks, each walking subx fewer serial tiles; the
        // partial layout + the combine simply carry the finer column
        // count). Tolerance-class at kernel level (finer reassociation
        // boundaries) - the model-level G1 (0/256 argmax + loop stream)
        // is the gate; the knob resolves ONCE here, never per dispatch.
        // Clamped so the effective chunk stays a multiple of the 32-pos
        // tile (the launcher contract).
        let verify_attn_sub = {
            let raw = std::env::var("QWEN38_VERIFY_ATTN_SUB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2);
            raw.clamp(1, (attn_chunk_len / 32).max(1))
        };
        // Round down to a 32-position multiple (the launcher contract:
        // chunk_len % 32 == 0 - a non-power-of-2 sub like 3 would otherwise
        // yield 85 and be rejected at launch time).
        let verify_attn_chunk_len =
            ((attn_chunk_len / verify_attn_sub) / 32).max(1) * 32;
        let verify_attn_n_chunks = ctx_len.div_ceil(verify_attn_chunk_len).max(1);
        // Issue 742 T9.14 - the two-pass flash restructure (the qg arm's
        // serial running-max chain removed): pass A computes per-tile
        // flash stats FULLY tile-parallel (one block per (kv_head,
        // 32-tile)); the merge folds them into the frozen (M, L); pass B
        // accumulates p = expf(score - M) * V with NO per-tile rescale.
        // MEASURED NEGATIVE at 20K (Bench 745 T9.14: 89.71 vs 75.85
        // ms/chunk, +18% - the score-phase dots are done TWICE and the
        // walk is per-lane-serial-FMA-bound, NOT running-max-chain-bound;
        // the T9.11/T9.13 latency framing is refuted at kernel level).
        // OPT-IN artifact (=1); the T9.13 qg kernels stay the default.
        // Resolved ONCE here, never per dispatch (a captured graph bakes
        // the arm).
        let verify_attn_2p = std::env::var("QWEN38_VERIFY_ATTN_2P").is_ok_and(|v| v == "1")
            && cfg.head_dim == 256
            && cfg.n_head.is_multiple_of(cfg.n_kv_head)
            && cfg.n_head / cfg.n_kv_head <= 8;
        // Issue 742 T9.15 - the multi-accumulator dot arms (both verify
        // arms): the score dot's 256-deep serial-FMA chain broken into
        // FOUR independent 64-deep partials. Opt-in (=1) since Bench 759
        // (Issue 755 T2): the 4x64 partial association DIFFERS from the
        // decode kernels' serial dot, so the verify chunk stopped being
        // bit-identical to sequential greedy — silent until a chat-lane
        // near-tie (pos 96) flipped argmax (287/384-class G1 divergence;
        // the sibling n_chunks fix f19bc6045 was real but secondary —
        // DOTMA=0 alone restores 0/384 bit-identity). The A/B win (Bench
        // 745 T9.15: -5.0% verify @20K, -2.0% @8K) is a tolerance-class
        // trade the strict G1 contract cannot take as a DEFAULT; opt
        // back in per-run where argmax near-ties are absent (doc-repro
        // long-ctx). Resolved ONCE here, never per dispatch (a captured
        // graph bakes the arm).
        let verify_attn_dotma = std::env::var("QWEN38_VERIFY_ATTN_DOTMA").is_ok_and(|v| v == "1")
            && cfg.head_dim == 256
            && cfg.n_head.is_multiple_of(cfg.n_kv_head)
            && cfg.n_head / cfg.n_kv_head <= 8;
        let verify_stat_n_tiles = ctx_len.div_ceil(32).max(1);
        let attn_split = std::env::var("QWEN38_ATTN_SPLIT").map_or(true, |v| v != "0");
        // GQA-fused split (Issue 742 headroom): needs an even GQA group ≤ 8
        // (one score warp per head) + head_dim ≤ 256 (block budget). The
        // chunk normalization above already forces chunk_len % 32 == 0.
        let attn_gqa = std::env::var("QWEN38_ATTN_GQA").map_or(true, |v| v != "0")
            && cfg.n_head.is_multiple_of(cfg.n_kv_head)
            && cfg.n_head / cfg.n_kv_head <= 8
            && cfg.head_dim <= 256;
        // Issue 742 fused GDN recurrence: single-pass (1R+1W) state update,
        // bit-identical math order. A/B hatch: QWEN38_DN_FUSED=0 routes back
        // to recurrence_parallel (resolved ONCE here, never per dispatch).
        let dn_fused = std::env::var("QWEN38_DN_FUSED").map_or(true, |v| v != "0")
            && cfg.head_k_dim.is_multiple_of(32);
        // Issue 742 rmsnorm→quant fusion (Bench 734 ranked lever 1): one
        // kernel for the rmsnorm + q8-quantize pair at every plain-rmsnorm
        // site, bit-identical by construction. Requires n_embd % 16 == 0
        // (5120 ✓). Resolved ONCE here, never per dispatch.
        let rnq_fused = std::env::var("QWEN38_RNQ_FUSED").map_or(true, |v| v != "0")
            && cfg.n_embd.is_multiple_of(16);
        // Issue 742 T9.7c (Bench 736): the constexpr-specialized fused kernel
        // at n_embd==5120 — fully-unrolled data loops (the generic fused
        // kernel's rolled loops serialize the strided loads; its 7.36 µs/site
        // is memory latency, not bandwidth). Resolved ONCE here, never per
        // dispatch.
        let rnq_fast = std::env::var("QWEN38_RNQ_FAST").map_or(true, |v| v != "0")
            && cfg.n_embd == 5120;
        // Issue 742 lever-2 (Bench 737): the GDN post-recurrence chain fused
        // into one kernel — 144 launches/token -> 48 (each eliminated kernel
        // saves its whole ~2-3.5 µs invocation floor per the Bench-736 floor
        // model). Bit-identical by construction; requires head_v_dim % 16 == 0
        // and <= 256 (128 for this model). Resolved ONCE here, never per
        // dispatch.
        let zgq_fused = std::env::var("QWEN38_ZGQ_FUSED").map_or(true, |v| v != "0")
            && cfg.head_v_dim.is_multiple_of(16)
            && cfg.head_v_dim <= 256;
        // Issue 755: the fused layer-boundary chain (residual + snapshot +
        // next site's norm+quant). Resolved ONCE here, never per dispatch
        // (the env-once lesson). Requires n_embd == 5120 (the constexpr
        // twin; every decode/verify norm site is 5120 on this model).
        let res_nq_fused = std::env::var("QWEN38_RES_NQ_FUSED").map_or(true, |v| v != "0")
            && cfg.n_embd == 5120;
        let attn_part_m = alloc_f32!(cfg.n_head * attn_n_chunks);
        let attn_part_l = alloc_f32!(cfg.n_head * attn_n_chunks);
        let attn_part_out = alloc_f32!(cfg.n_head * attn_n_chunks * cfg.head_dim);

        let verify = alloc_verify_scratch(
            &stream,
            &cfg,
            l_qkv_out,
            l_exp,
            q_dim,
            kvd,
            verify_attn_n_chunks,
            verify_stat_n_tiles,
            QWEN38_VERIFY_MAX_P,
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            weights,
            state,
            ctx_len,
            kv_f16,
            stream,
            dense,
            dn,
            attn,
            ew,
            lanes_commit_ready: None,
            x,
            x_res,
            x_norm,
            y,
            xq,
            xs,
            xsum,
            qkv,
            qkv_exp,
            z_buf,
            rec_out,
            rec_normed,
            a_raw,
            b_raw,
            beta,
            decay,
            qg,
            q,
            gate,
            q_normed,
            k,
            k_normed,
            v,
            attn_out,
            mlp_gate,
            mlp_up,
            mlp_hidden,
            logits,
            argmax_res,
            token_dev,
            pos_dev,
            attn_part_m,
            attn_part_l,
            attn_part_out,
            capturing: false,
            graph: None,
            attn_chunk_len,
            attn_n_chunks,
            verify_attn_chunk_len,
            verify_attn_n_chunks,
            verify_attn_2p,
            verify_stat_n_tiles,
            verify_attn_dotma,
            attn_split,
            attn_gqa,
            dn_fused,
            rnq_fused,
            rnq_fast,
            zgq_fused,
            res_nq_fused,
            verify,
            wide_scratch: None,
            wide_ingest_active: false,
            verify_taps_dev: None,
            gdn_journal: None,
            lanes: None,
            verify_mma,
            attn_mma,
            verify_attn_mma,
            verify_attn_mma_arm24: verify_attn_mma_arm == Some(2),
            verify_graphs: std::collections::HashMap::new(),
            verify_graph_failed: std::collections::HashSet::new(),
            prefix_cache: {
                // Issue 742 T4 — resolved ONCE here, never per call (the
                // construction-time env-resolution rule). ~159 MB of device
                // snapshots per checkpoint on the 27B (48×3.1 MB recurrent
                // + 48×160 KB conv).
                let max = std::env::var("QWEN38_PREFIX_CACHE_MAX")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4);
                crate::qwen38_prefix_cache::Qwen38PrefixCache::new(max)
            },
        })
    }

    // ── kernel helpers ──

    fn copy_f32(
        &self,
        src: &CudaSlice<f32>,
        dst: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let n_i = n as i32;
        let grid = n.div_ceil(256).max(1) as u32;
        unsafe {
            stream
                .launch_builder(&self.dense.copy_f32)
                .arg(src)
                .arg(dst)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn quantize_x(&self, x: &CudaSlice<f32>, n: usize) -> Result<(), String> {
        let stream = &self.stream;
        let n_i = n as i32;
        let grid = (n / 16).div_ceil(256).max(1) as u32;
        unsafe {
            stream
                .launch_builder(&self.dense.quant_x)
                .arg(x)
                .arg(&self.xq)
                .arg(&self.xs)
                .arg(&self.xsum)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Whether the Issue-755 fused boundary chain is active (the
    /// `QWEN38_RES_NQ_FUSED=0` A/B hatch — asserted by the bench_757 G1 gate
    /// so each arm is non-vacuous).
    pub fn res_nq_fused(&self) -> bool {
        self.res_nq_fused
    }

    /// Issue 755 — the fused layer-boundary kernel (decode site):
    /// `x = x_res + y`, snapshot (`x_res = x`), and the NEXT site's
    /// rmsnorm+q8-quantize of `x` in ONE launch — replaces the
    /// residual_add_f32 + copy_f32 + rmsnorm_quant_x trio (2 fewer
    /// launches + no x round-trip per boundary; 128 sites/token).
    /// Bit-identical by construction (kernel doc).
    ///
    /// SAFETY note: `xq`/`xs`/`xsum` are sized for mlp_hidden (17408) ⊇
    /// 5120/320 (the same contract as [`Self::rmsnorm_quant_x`]); the
    /// launcher is `n_embd == 5120`-gated by `res_nq_fused`.
    fn residual_norm_quant_x(&self, y: &CudaSlice<f32>, gamma: &CudaSlice<f32>) -> Result<(), String> {
        // SAFETY: x_res/y/gamma cover 5120; x_out (self.x) covers 5120;
        // res_out ALIASES res (both self.x_res) — safe per the kernel's
        // ownership discipline (phase-1 same-thread read-before-write;
        // phase 4 reads x_out only; see the launcher's # Safety).
        unsafe {
            self.dense.launch_residual_norm_quant_x_q8_s5120(
                &self.stream,
                &self.x_res,
                y,
                gamma,
                &self.x,
                &self.x_res,
                &self.xq,
                &self.xs,
                &self.xsum,
                self.cfg.rms_norm_eps,
            )
        }
    }

    /// Issue 755 — the rows twin (verify/ingest boundary site): per row
    /// `xb = xb_res + y`, snapshot (`xb_res = xb`), and the next site's
    /// rmsnorm+q8-quantize in ONE launch (grid `p`, one block per row).
    /// Bit-identical to the unfused residual_add(n*p) → copy_f32(n*p) →
    /// rows-norm chain per row.
    ///
    /// SAFETY note: `v.xq`/`v.xs`/`v.xsum` are sized for mlp_hidden * p
    /// (17408·p) ⊇ 5120·p / 320·p (the same contract as
    /// [`Self::rmsnorm_quant_x_rows`]).
    fn residual_norm_quant_x_rows(
        &self,
        y: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        p: usize,
    ) -> Result<(), String> {
        let v = &self.verify;
        // SAFETY: xb_res/y cover p*5120; xb covers p*5120; res_out ALIASES
        // res (both v.xb_res) — row-disjoint per block, phase-4 reads xb
        // only; see the launcher's # Safety.
        unsafe {
            self.dense.launch_residual_norm_quant_x_q8_s5120_rows(
                &self.stream,
                &v.xb_res,
                y,
                gamma,
                &v.xb,
                &v.xb_res,
                &v.xq,
                &v.xs,
                &v.xsum,
                self.cfg.rms_norm_eps,
                p,
            )
        }
    }

    /// rmsnorm(x, gamma) → xq/xs/xsum in one kernel (Issue 742 T9.7 follow-on;
    /// the `QWEN38_RNQ_FUSED=0` hatch restores the two-kernel pair).
    fn rmsnorm_quant_x(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        if self.rnq_fused {
            if self.rnq_fast && n == 5120 {
                // T9.7c: constexpr-specialized twin (Bench 736) — identical
                // numerics, fully-unrolled data loops. SAFETY: n == 5120 and
                // the forward's persistent buffers cover it; xq/xs/xsum sized
                // for mlp_hidden (17408) ⊇ 5120/320.
                unsafe {
                    self.dense.launch_rmsnorm_quant_x_q8_s5120(
                        &self.stream,
                        x,
                        gamma,
                        &self.xq,
                        &self.xs,
                        &self.xsum,
                        eps,
                    )
                }
            } else {
                // SAFETY: x/gamma cover `n` (the forward's persistent buffers);
                // xq/xs/xsum are sized for the max call dim (mlp_hidden) which
                // covers every rmsnorm site's n_embd; dim % 16 asserted in-launch.
                unsafe {
                    self.dense.launch_rmsnorm_quant_x_q8(
                        &self.stream,
                        x,
                        gamma,
                        &self.xq,
                        &self.xs,
                        &self.xsum,
                        n,
                        eps,
                    )
                }
            }
        } else {
            self.ew
                .launch_rmsnorm(&self.stream, x, gamma, &self.x_norm, n, eps)
                .map_err(|e| e.to_string())?;
            self.quantize_x(&self.x_norm, n)
        }
    }

    /// rmsnorm(x, gamma) → silu(z)-gate → xq/xs/xsum in one kernel (Issue 742
    /// lever-2, Bench 737; the `QWEN38_ZGQ_FUSED=0` hatch restores the
    /// 3-kernel trio).
    fn rmsnorm_zgate_quant_x(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        if self.zgq_fused {
            // SAFETY: x/z cover n_heads*head_dim (the persistent rec_out /
            // z_buf buffers, sized for d_inner = n_v_heads*head_v_dim);
            // gamma covers head_dim (ssm_norm); xq/xs/xsum are sized for
            // mlp_hidden (17408) ⊇ d_inner (6144) and its groups; the
            // head_dim gates were checked at construction (zgq_fused).
            unsafe {
                self.dense.launch_rmsnorm_zgate_quant_x_q8(
                    &self.stream,
                    x,
                    z,
                    gamma,
                    &self.xq,
                    &self.xs,
                    &self.xsum,
                    n_heads,
                    head_dim,
                    eps,
                )
            }
        } else {
            let n = n_heads * head_dim;
            self.attn
                .launch_rmsnorm_batched(
                    &self.stream,
                    x,
                    gamma,
                    &self.rec_normed,
                    n_heads,
                    head_dim,
                    eps,
                )
                .map_err(|e| e.to_string())?;
            self.dn
                .launch_z_gating(&self.stream, &self.rec_normed, z, n)
                .map_err(|e| e.to_string())?;
            self.quantize_x(&self.rec_normed, n)
        }
    }

    fn gemv_quant(&self, w: &QuantW, y: &CudaSlice<f32>) -> Result<(), String> {
        let stream = &self.stream;
        let (m_i, n_i, bpr_i) = (w.rows as i32, w.n as i32, w.blocks_per_row as i32);
        let grid = w.rows.div_ceil(8).max(1) as u32;
        let func = if w.q4 {
            &self.dense.q4k_q8x
        } else {
            &self.dense.q6k_q8x
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(&w.dev)
                .arg(&self.xq)
                .arg(&self.xs)
                .arg(&self.xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn embed_row(&self, token: u32) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let n_i = cfg.n_embd as i32;
        let bpr = self.weights.token_embd.blocks_per_row as i32;
        let grid = cfg.n_embd.div_ceil(256).max(1) as u32;
        unsafe {
            if self.capturing {
                stream
                    .launch_builder(&self.dense.dequant_q4k_row_devpos)
                    .arg(&self.weights.token_embd.dev)
                    .arg(&self.x)
                    .arg(&self.token_dev)
                    .arg(&n_i)
                    .arg(&bpr)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| e.to_string())?;
            } else {
                let row = token as i32;
                stream
                    .launch_builder(&self.dense.dequant_q4k_row)
                    .arg(&self.weights.token_embd.dev)
                    .arg(&self.x)
                    .arg(&row)
                    .arg(&n_i)
                    .arg(&bpr)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    // ── layer forwards ──

    #[allow(clippy::too_many_lines)]
    fn forward_gdn_layer(&self, layer_idx: usize, gdn_idx: usize) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let n_v = cfg.n_v_heads;

        if !self.res_nq_fused {
            self.rmsnorm_quant_x(&self.x, &lw.input_norm.dev, n, cfg.rms_norm_eps)?;
        }
        // fused path: the leading norm was done by the previous site's
        // Issue-755 fused boundary kernel (run_layer_stack's site B).
        self.gemv_quant(lw.qkv.as_ref().unwrap(), &self.qkv)?;
        self.gemv_quant(lw.z.as_ref().unwrap(), &self.z_buf)?;
        self.gemv_quant(lw.alpha.as_ref().unwrap(), &self.a_raw)?;
        self.gemv_quant(lw.beta.as_ref().unwrap(), &self.b_raw)?;
        self.dn
            .launch_beta_decay(
                stream,
                &self.a_raw,
                &self.b_raw,
                &lw.a_log.as_ref().unwrap().dev,
                &lw.dt_bias.as_ref().unwrap().dev,
                &self.beta,
                &self.decay,
                n_v,
            )
            .map_err(|e| e.to_string())?;
        {
            let conv_w = lw.conv1d.as_ref().unwrap();
            let conv_state = &self.state.conv[gdn_idx];
            self.dn
                .launch_conv1d(
                    stream,
                    &self.qkv,
                    &conv_w.dev,
                    conv_state,
                    conv_w.len / cfg.conv_kernel,
                    cfg.conv_kernel,
                )
                .map_err(|e| e.to_string())?;
        }
        self.dn
            .launch_expand_and_l2_normalize(
                stream,
                &self.qkv,
                &self.qkv_exp,
                cfg.n_k_heads,
                cfg.n_v_heads,
                cfg.head_k_dim,
            )
            .map_err(|e| e.to_string())?;
        {
            let state = &self.state.recurrent[gdn_idx];
            // Issue 742: fused single-pass recurrence (default) with the
            // parallel kernel as the QWEN38_DN_FUSED=0 A/B hatch. Both are
            // bit-identical (same op order per element + same butterfly).
            let launch = if self.dn_fused {
                self.dn.launch_recurrence_fused(
                    stream,
                    &self.qkv_exp,
                    &self.beta,
                    &self.decay,
                    state,
                    &self.rec_out,
                    cfg.head_k_dim,
                    cfg.n_v_heads,
                )
            } else {
                self.dn.launch_recurrence_parallel(
                    stream,
                    &self.qkv_exp,
                    &self.beta,
                    &self.decay,
                    state,
                    &self.rec_out,
                    cfg.head_k_dim,
                    cfg.n_v_heads,
                )
            };
            launch.map_err(|e| e.to_string())?;
        }
        // Lever-2 (Bench 737): the post-recurrence chain fused into one
        // kernel by default (bit-identical; see rmsnorm_zgate_quant_x).
        self.rmsnorm_zgate_quant_x(
            &self.rec_out,
            &self.z_buf,
            &lw.ssm_norm.as_ref().unwrap().dev,
            cfg.n_v_heads,
            cfg.head_v_dim,
            cfg.rms_norm_eps,
        )?;
        self.gemv_quant(lw.ssm_out.as_ref().unwrap(), &self.y)?;
        if !self.res_nq_fused {
            // fused path: the residual is folded into the next site's
            // Issue-755 fused boundary kernel (run_layer_stack's site A).
            self.ew
                .launch_residual_add(stream, &self.x_res, &self.y, &self.x, n)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn forward_attn_layer(&self, layer_idx: usize, attn_idx: usize, pos: usize) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;

        if !self.res_nq_fused {
            self.rmsnorm_quant_x(&self.x, &lw.input_norm.dev, n, cfg.rms_norm_eps)?;
        }
        // fused path: norm done by the previous site's fused boundary
        // kernel; residual folded into the next site's.
        self.gemv_quant(lw.wq.as_ref().unwrap(), &self.qg)?;
        self.attn
            .launch_split_qg(
                stream,
                &self.qg,
                &self.q,
                &self.gate,
                cfg.head_dim,
                cfg.n_head,
            )
            .map_err(|e| e.to_string())?;
        self.gemv_quant(lw.wk.as_ref().unwrap(), &self.k)?;
        self.gemv_quant(lw.wv.as_ref().unwrap(), &self.v)?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &self.q,
                &lw.q_norm.as_ref().unwrap().dev,
                &self.q_normed,
                cfg.n_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &self.k,
                &lw.k_norm.as_ref().unwrap().dev,
                &self.k_normed,
                cfg.n_kv_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        if self.capturing {
            self.attn
                .launch_rope_devpos(
                    stream,
                    &self.q_normed,
                    &self.k_normed,
                    cfg.rotary_dim,
                    cfg.head_dim,
                    cfg.n_head,
                    cfg.n_kv_head,
                    &self.pos_dev,
                    cfg.rope_theta,
                )
                .map_err(|e| e.to_string())?;
            let kc = &self.state.keys[attn_idx];
            let vc = &self.state.values[attn_idx];
            self.attn
                .launch_kv_cache_append_devpos(
                    stream, &self.k_normed, &self.v, kc, vc, kvd, &self.pos_dev,
                )
                .map_err(|e| e.to_string())?;
            if self.attn_gqa {
                self.attn
                    .launch_attention_decode_split_gqa_devpos(
                        stream,
                        &self.q_normed,
                        kc,
                        vc,
                        &self.attn_part_m,
                        &self.attn_part_l,
                        &self.attn_part_out,
                        &self.attn_out,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        self.attn_chunk_len,
                        self.attn_n_chunks,
                        &self.pos_dev,
                    )
                    .map_err(|e| e.to_string())?;
            } else if self.attn_split {
                self.attn
                    .launch_attention_decode_split_devpos(
                        stream,
                        &self.q_normed,
                        kc,
                        vc,
                        &self.attn_part_m,
                        &self.attn_part_l,
                        &self.attn_part_out,
                        &self.attn_out,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        self.attn_chunk_len,
                        self.attn_n_chunks,
                        &self.pos_dev,
                    )
                    .map_err(|e| e.to_string())?;
            } else {
                self.attn
                    .launch_attention_decode_devpos(
                        stream,
                        &self.q_normed,
                        kc,
                        vc,
                        &self.attn_out,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        &self.pos_dev,
                    )
                    .map_err(|e| e.to_string())?;
            }
        } else {
            self.attn
                .launch_rope(
                    stream,
                    &self.q_normed,
                    &self.k_normed,
                    cfg.rotary_dim,
                    cfg.head_dim,
                    cfg.n_head,
                    cfg.n_kv_head,
                    pos,
                    cfg.rope_theta,
                )
                .map_err(|e| e.to_string())?;
            let kc = &self.state.keys[attn_idx];
            let vc = &self.state.values[attn_idx];
            self.attn
                .launch_kv_cache_append(stream, &self.k_normed, &self.v, kc, vc, kvd, pos)
                .map_err(|e| e.to_string())?;
            if self.attn_gqa {
                self.attn
                    .launch_attention_decode_split_gqa(
                        stream,
                        &self.q_normed,
                        kc,
                        vc,
                        &self.attn_part_m,
                        &self.attn_part_l,
                        &self.attn_part_out,
                        &self.attn_out,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        self.attn_chunk_len,
                        pos + 1,
                    )
                    .map_err(|e| e.to_string())?;
            } else if self.attn_split {
                self.attn
                    .launch_attention_decode_split(
                        stream,
                        &self.q_normed,
                        kc,
                        vc,
                        &self.attn_part_m,
                        &self.attn_part_l,
                        &self.attn_part_out,
                        &self.attn_out,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        self.attn_chunk_len,
                        pos + 1,
                    )
                    .map_err(|e| e.to_string())?;
            } else {
                self.attn
                    .launch_attention_decode(
                        stream,
                        &self.q_normed,
                        kc,
                        vc,
                        &self.attn_out,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        pos + 1,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        self.attn
            .launch_output_gate(stream, &self.attn_out, &self.gate, q_dim)
            .map_err(|e| e.to_string())?;
        self.quantize_x(&self.attn_out, q_dim)?;
        self.gemv_quant(lw.wo.as_ref().unwrap(), &self.y)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel.
            self.ew
                .launch_residual_add(stream, &self.x_res, &self.y, &self.x, n)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn forward_mlp(&self, layer_idx: usize) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        if !self.res_nq_fused {
            self.rmsnorm_quant_x(&self.x, &lw.post_attn_norm.dev, n, cfg.rms_norm_eps)?;
        }
        // fused path: norm done by the previous site's fused boundary
        // kernel (run_layer_stack's site A); residual folded into the
        // next site's.
        self.gemv_quant(&lw.ffn_gate, &self.mlp_gate)?;
        self.gemv_quant(&lw.ffn_up, &self.mlp_up)?;
        self.ew
            .launch_swiglu(
                stream,
                &self.mlp_gate,
                &self.mlp_up,
                &self.mlp_hidden,
                cfg.mlp_hidden,
            )
            .map_err(|e| e.to_string())?;
        self.quantize_x(&self.mlp_hidden, cfg.mlp_hidden)?;
        self.gemv_quant(&lw.ffn_down, &self.y)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel (or the plain last-layer add).
            self.ew
                .launch_residual_add(stream, &self.x_res, &self.y, &self.x, n)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn run_layer_stack(&mut self, token: u32, pos: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        self.embed_row(token)?;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 — the fused boundary chain. Prologue: the layer-0
            // snapshot + layer-0 input norm, plain (no previous site to
            // fold them into); then EVERY per-layer boundary runs ONE
            // fused residual+snapshot+next-norm kernel instead of
            // residual_add + copy_f32 + rmsnorm_quant_x. The block fns and
            // forward_mlp skip their own leading norm + trailing residual
            // (the `!res_nq_fused` conditionals inside them).
            self.copy_f32(&self.x, &self.x_res, n)?;
            self.rmsnorm_quant_x(
                &self.x,
                &self.weights.layers[0].input_norm.dev,
                n,
                cfg.rms_norm_eps,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.forward_gdn_layer(i, gdn_idx)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.forward_attn_layer(i, attn_idx, pos)?;
                        attn_idx += 1;
                    }
                }
                // Site A: residual + next snapshot + this layer's MLP
                // post_attn norm in one kernel.
                self.residual_norm_quant_x(
                    &self.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                )?;
                self.forward_mlp(i)?;
                if i + 1 < cfg.n_layer {
                    // Site B: residual + next snapshot + the NEXT layer's
                    // input norm in one kernel.
                    self.residual_norm_quant_x(
                        &self.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                    )?;
                } else {
                    // Last layer: plain residual — the outer code (e.g.
                    // forward_token) owns output_norm + lm_head, and leaving
                    // x_res at the pre-MLP snapshot is fine (the next
                    // token's prologue copy overwrites it).
                    self.ew
                        .launch_residual_add(&self.stream, &self.x_res, &self.y, &self.x, n)
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.x, &self.x_res, n)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.forward_gdn_layer(i, gdn_idx)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.forward_attn_layer(i, attn_idx, pos)?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.x, &self.x_res, n)?;
                self.forward_mlp(i)?;
            }
        }
        Ok(())
    }

    /// One decode step. Enqueues the whole layer stack; returns the greedy
    /// argmax token. `pos` is the sequence position of `token` (0-based).
    pub fn forward_token(&mut self, token: u32, pos: usize) -> Result<u32, String> {
        // T4: this write clobbers KV rows [pos, ..) — prune prefix
        // checkpoints whose KV tail it could hit (the lineage rule).
        self.prefix_cache.note_write(pos);
        let n = self.cfg.n_embd;
        let eps = self.cfg.rms_norm_eps;
        self.run_layer_stack(token, pos)?;
        let stream = &self.stream;
        self.rmsnorm_quant_x(&self.x, &self.weights.output_norm.dev, n, eps)?;
        self.gemv_quant(&self.weights.lm_head, &self.logits)?;
        stream
            .memcpy_htod(ZERO_U64.as_slice(), &mut self.argmax_res)
            .map_err(|e| e.to_string())?;
        self.ew
            .launch_argmax_first(stream, &self.logits, self.cfg.vocab_size, &self.argmax_res)
            .map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let packed = stream
            .clone_dtoh(&self.argmax_res)
            .map_err(|e| e.to_string())?[0];
        Ok(!(packed as u32))
    }
    /// **Per-layer residual-stream capture** (Issue 742 T3 — the DFlash2
    /// drafter-feature extraction hook; the Bonsai-cudarc
    /// `forward_token_with_layer_capture` pattern ported to the qwen38 dense
    /// forward).
    ///
    /// After each layer listed in `capture_layers` completes (post-MLP
    /// residual stream — `self.x`, the SAME tap `forward_token_traced` uses,
    /// i.e. the INPUT of the next layer = llama.cpp's
    /// `embeddings_layer_inp` semantics), the row `[0..n_embd]` is
    /// downloaded into `capture_out[i][..n_embd]`. `capture_layers` must be
    /// strictly ascending; `capture_out` one buffer per layer, each
    /// `>= n_embd`.
    ///
    /// Each capture adds a sync + a 20 KB dtoh; with the 5 DFlash2 target
    /// layers that is 5 syncs/token — diagnostic-only cost, never a hot
    /// path. Returns the greedy argmax token (the same first-index
    /// convention as [`Self::forward_token`]).
    pub fn forward_token_capture(
        &mut self,
        token: u32,
        pos: usize,
        capture_layers: &[usize],
        capture_out: &mut [Vec<f32>],
    ) -> Result<u32, String> {
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        assert_eq!(capture_layers.len(), capture_out.len());
        for w in capture_out.iter() {
            assert!(w.len() >= n, "capture buffers must be >= n_embd");
        }
        for w in capture_layers.windows(2) {
            assert!(w[0] < w[1], "capture_layers must be strictly ascending");
        }

        let stream = &self.stream;
        self.embed_row(token)?;
        let mut cap_i = 0usize;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 fused boundary chain (same sequence as
            // run_layer_stack's fused arm). The capture tap sits AFTER the
            // per-layer residual is materialized (site B / the plain
            // last-layer add), so `self.x` holds the same post-layer value
            // the unfused path captures.
            self.copy_f32(&self.x, &self.x_res, n)?;
            self.rmsnorm_quant_x(
                &self.x,
                &self.weights.layers[0].input_norm.dev,
                n,
                cfg.rms_norm_eps,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.forward_gdn_layer(i, gdn_idx)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.forward_attn_layer(i, attn_idx, pos)?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x(
                    &self.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                )?;
                self.forward_mlp(i)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x(
                        &self.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                    )?;
                } else {
                    self.ew
                        .launch_residual_add(stream, &self.x_res, &self.y, &self.x, n)
                        .map_err(|e| e.to_string())?;
                }
                if cap_i < capture_layers.len() && capture_layers[cap_i] == i {
                    stream.synchronize().map_err(|e| e.to_string())?;
                    let row = stream.clone_dtoh(&self.x).map_err(|e| e.to_string())?;
                    capture_out[cap_i][..n].copy_from_slice(&row[..n]);
                    cap_i += 1;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.x, &self.x_res, n)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.forward_gdn_layer(i, gdn_idx)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.forward_attn_layer(i, attn_idx, pos)?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.x, &self.x_res, n)?;
                self.forward_mlp(i)?;
                if cap_i < capture_layers.len() && capture_layers[cap_i] == i {
                    stream.synchronize().map_err(|e| e.to_string())?;
                    let row = stream.clone_dtoh(&self.x).map_err(|e| e.to_string())?;
                    capture_out[cap_i][..n].copy_from_slice(&row[..n]);
                    cap_i += 1;
                }
            }
        }

        self.rmsnorm_quant_x(&self.x, &self.weights.output_norm.dev, n, cfg.rms_norm_eps)?;
        self.gemv_quant(&self.weights.lm_head, &self.logits)?;
        stream
            .memcpy_htod(ZERO_U64.as_slice(), &mut self.argmax_res)
            .map_err(|e| e.to_string())?;
        self.ew
            .launch_argmax_first(stream, &self.logits, self.cfg.vocab_size, &self.argmax_res)
            .map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let packed = stream
            .clone_dtoh(&self.argmax_res)
            .map_err(|e| e.to_string())?[0];
        Ok(!(packed as u32))
    }

    /// lm_head rows for an EXTERNAL caller (Issue 742 T3 — the DFlash2
    /// drafter's noise block consumes the target's output projection; the
    /// shared-weights contract — the drafter GGUF ships no `output` tensor).
    ///
    /// `hidden_rows` is `[n_rows][n_embd]` flattened, each row ALREADY
    /// final-normed by the caller (the drafter applies its OWN output_norm;
    /// the PR-27342 graph pipes that straight into the target's lm_head with
    /// no second target-side norm). Each row is quantized + GEMV'd through
    /// the SAME production lm_head path as [`Self::forward_token`] (q8
    /// x-quant + the Q6_K dp4a GEMV — bit-faithful to the target's own
    /// logits), returning `[n_rows][vocab]` flattened. One sync per row —
    /// diagnostic cadence (the eval harness calls this once per ~8-row
    /// draft block).
    pub fn lm_head_rows(&mut self, hidden_rows: &[f32], n_rows: usize) -> Result<Vec<f32>, String> {
        let n = self.cfg.n_embd;
        let vocab = self.cfg.vocab_size;
        assert_eq!(hidden_rows.len(), n_rows * n);
        let stream = &self.stream;
        let mut out = Vec::with_capacity(n_rows * vocab);
        for r in 0..n_rows {
            stream
                .memcpy_htod(&hidden_rows[r * n..(r + 1) * n], &mut self.x)
                .map_err(|e| e.to_string())?;
            unsafe {
                self.dense
                    .launch_quant_x_q8(stream, &self.x, &self.xq, &self.xs, &self.xsum, n)?;
            }
            self.gemv_quant(&self.weights.lm_head, &self.logits)?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let row = stream.clone_dtoh(&self.logits).map_err(|e| e.to_string())?;
            out.extend_from_slice(&row[..vocab]);
        }
        Ok(out)
    }

    /// Issue 755 T2 — the BATCHED lm_head rows (the loop's drafter head):
    /// `hidden_rows` `[n_rows][n_embd]` (each row ALREADY final-normed by the
    /// caller — the DFlash2 contract, see [`Self::lm_head_rows`]) projected
    /// through the production rows path in ONE weight read: upload →
    /// `launch_quant_x_q8` (flat n_rows·n) → `gemv_quant_rows(lm_head, p)` →
    /// sync → dtoh `[n_rows][vocab]`. Per-row bit-identical to the verify
    /// chunk's own tail (the T9.9 rows fold-order contract — the same class
    /// [`Self::lm_head_rows`] carries vs the single-row GEMV); replaces the
    /// per-row helper's n_rows separate lm_head reads, which would dominate
    /// the drafter cycle at n_rows = 7.
    pub fn lm_head_rows_batched(
        &mut self,
        hidden_rows: &[f32],
        n_rows: usize,
    ) -> Result<Vec<f32>, String> {
        let n = self.cfg.n_embd;
        let vocab = self.cfg.vocab_size;
        if n_rows == 0 || n_rows > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "lm_head_rows_batched: n_rows must be in 1..={QWEN38_VERIFY_MAX_P} (got {n_rows})"
            ));
        }
        if hidden_rows.len() != n_rows * n {
            return Err(format!(
                "lm_head_rows_batched: hidden_rows.len() {} != n_rows {n_rows} * n_embd {n}",
                hidden_rows.len()
            ));
        }
        let stream = &self.stream;
        stream
            .memcpy_htod(hidden_rows, &mut self.verify.xb)
            .map_err(|e| e.to_string())?;
        // SAFETY: xb covers p*n f32 (p<=16); xq/xs/xsum are sized for
        // p*mlp_hidden ⊇ p*n; n_rows*n % 16 == 0 (n % 16 asserted at every
        // other site; n_embd = 5120 on this model).
        unsafe {
            self.dense.launch_quant_x_q8(
                stream,
                &self.verify.xb,
                &self.verify.xq,
                &self.verify.xs,
                &self.verify.xsum,
                n_rows * n,
            )?;
        }
        self.gemv_quant_rows(&self.weights.lm_head, &self.verify.logits, n_rows)?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let view = self.verify.logits.slice(0..n_rows * vocab);
        stream
            .clone_dtoh(&view)
            .map_err(|e| e.to_string())
    }

    /// Issue 755 T2 — the TARGET's token-embedding rows, dequantized ON
    /// DEVICE and downloaded (host `[p][n_embd]`). The DFlash2 block input
    /// uses the target's `token_embd` (shared weights — the drafter ships
    /// none); the same `launch_dequant_q4k_rows` kernel the verify chunk uses
    /// for its own embedding stage, so the anchor/mask rows the drafter sees
    /// are construction-identical to the chunk's. p <= 16. One sync.
    pub fn embed_rows_host(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let n = self.cfg.n_embd;
        let p = tokens.len();
        if p == 0 || p > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "embed_rows_host: p must be in 1..={QWEN38_VERIFY_MAX_P} (got {p})"
            ));
        }
        let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
        for (i, &t) in tokens.iter().enumerate() {
            toks[i] = t as i32;
        }
        let stream = &self.stream;
        stream
            .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
            .map_err(|e| e.to_string())?;
        let w = &self.weights.token_embd;
        // SAFETY: out (verify.xb) covers p*n; tokens covers p — the verify
        // chunk's own embedding-stage contract.
        unsafe {
            self.dense.launch_dequant_q4k_rows(
                stream,
                &w.dev,
                &self.verify.xb,
                &self.verify.tokens_dev,
                n,
                w.blocks_per_row,
                p,
            )?;
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        let view = self.verify.xb.slice(0..p * n);
        stream.clone_dtoh(&view).map_err(|e| e.to_string())
    }

    /// The recorded launch sequence for graph capture (devpos variants —
    /// token/pos read from device buffers at kernel runtime).
    fn run_capture_body(&mut self) -> Result<(), String> {
        let n = self.cfg.n_embd;
        let eps = self.cfg.rms_norm_eps;
        self.run_layer_stack(0, 0)?;
        let stream = &self.stream;
        self.rmsnorm_quant_x(&self.x, &self.weights.output_norm.dev, n, eps)?;
        self.gemv_quant(&self.weights.lm_head, &self.logits)?;
        stream
            .memset_zeros(&mut self.argmax_res)
            .map_err(|e| e.to_string())?;
        self.ew
            .launch_argmax_first(
                stream,
                &self.logits,
                self.cfg.vocab_size,
                &self.argmax_res,
            )
            .map_err(|e| e.to_string())
    }

    /// T5 graph path: capture the whole decode step ONCE, then one
    /// `graph.launch()` per token (the ~1450-kernel-launch WDDM overhead
    /// collapses to a single graph launch; token/pos ride device buffers —
    /// the _devpos kernel variants — and the argmax zero rides a capturable
    /// memset).
    ///
    /// The capture follows the decode-path SAFETY contract (single stream,
    /// event tracking disabled — `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`
    /// otherwise; persistent buffers only, no allocs inside the arm).
    /// During capture the kernels are RECORDED, not executed — the first
    /// real step is the first `launch()`.
    pub fn forward_token_graph(&mut self, token: u32, pos: usize) -> Result<u32, String> {
        // T4 lineage rule (see forward_token).
        self.prefix_cache.note_write(pos);
        if self.graph.is_none() {
            {
                let stream = &self.stream;
                // SAFETY: this struct is single-threaded and single-stream; the
                // capture arm performs no cross-stream ops (the sibling
                // prefill-probe contract, Issue 742 T1.2).
                unsafe { stream.context().disable_event_tracking(); }
                stream
                    .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
                    .map_err(|e| format!("begin_capture: {e}"))?;
            }
            self.capturing = true;
            let cap_result = self.run_capture_body();
            self.capturing = false;
            let end = {
                let stream = &self.stream;
                stream.end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
            };
            match (cap_result, end) {
                (Ok(()), Ok(Some(graph))) => {
                    graph
                        .upload()
                        .map_err(|e| format!("graph upload: {e}"))?;
                    self.stream.synchronize().map_err(|e| e.to_string())?;
                    self.graph = Some(graph);
                }
                (r, e) => {
                    let run_err = r.err().map(|e| e.to_string()).unwrap_or_default();
                    let end_err = e.err().map(|e| e.to_string()).unwrap_or_default();
                    return Err(format!(
                        "graph capture failed (run err: {run_err}, end_capture err: {end_err})"
                    ));
                }
            }
        }
        // device params + launch + readback
        {
            let probe = std::env::var("QWEN38_GRAPH_PROBE").is_ok();
            let stream = &self.stream;
            let t0 = std::time::Instant::now();
            stream
                .memcpy_htod(&[token as i32], &mut self.token_dev)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(&[pos as i32], &mut self.pos_dev)
                .map_err(|e| e.to_string())?;
            let t1 = std::time::Instant::now();
            let graph = self.graph.as_ref().expect("graph captured above");
            graph.launch().map_err(|e| format!("graph launch: {e}"))?;
            let t2 = std::time::Instant::now();
            stream.synchronize().map_err(|e| e.to_string())?;
            let t3 = std::time::Instant::now();
            let packed = stream
                .clone_dtoh(&self.argmax_res)
                .map_err(|e| e.to_string())?[0];
            if probe {
                eprintln!(
                    "[graph-probe] htod={:?} launch={:?} sync={:?} dtoh={:?}",
                    t1 - t0,
                    t2 - t1,
                    t3 - t2,
                    t3.elapsed()
                );
            }
            Ok(!(packed as u32))
        }
    }

    /// Trace variant: returns the hidden state after EVERY layer (the G1 gate
    /// tap; trace[0] = embedding, trace[i+1] = after layer i, + the final
    /// logits). Syncs per tap — diagnostic only, never the perf path.
    pub fn forward_token_traced(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<TracedForward, String> {
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        let mut trace = Vec::with_capacity(cfg.n_layer + 1);

        self.embed_row(token)?;
        trace.push(self.tap());

        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 fused boundary chain (same sequence as
            // run_layer_stack's fused arm). The tap sits AFTER the per-layer
            // residual is materialized, so trace[i+1] is the same post-layer
            // value the unfused path records.
            self.copy_f32(&self.x, &self.x_res, n)?;
            self.rmsnorm_quant_x(
                &self.x,
                &self.weights.layers[0].input_norm.dev,
                n,
                cfg.rms_norm_eps,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.forward_gdn_layer(i, gdn_idx)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.forward_attn_layer(i, attn_idx, pos)?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x(
                    &self.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                )?;
                self.forward_mlp(i)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x(
                        &self.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                    )?;
                } else {
                    self.ew
                        .launch_residual_add(&self.stream, &self.x_res, &self.y, &self.x, n)
                        .map_err(|e| e.to_string())?;
                }
                trace.push(self.tap());
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.x, &self.x_res, n)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.forward_gdn_layer(i, gdn_idx)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.forward_attn_layer(i, attn_idx, pos)?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.x, &self.x_res, n)?;
                self.forward_mlp(i)?;
                trace.push(self.tap());
            }
        }

        let stream = &self.stream;
        self.rmsnorm_quant_x(&self.x, &self.weights.output_norm.dev, n, cfg.rms_norm_eps)?;
        self.gemv_quant(&self.weights.lm_head, &self.logits)?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let logits = stream
            .clone_dtoh(&self.logits)
            .map_err(|e| e.to_string())?;
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > best_v {
                best_v = l;
                best = i;
            }
        }
        Ok((best as u32, trace, logits))
    }

    fn tap(&self) -> Vec<f32> {
        let stream = &self.stream;
        let _ = stream.synchronize();
        stream.clone_dtoh(&self.x).unwrap_or_default()
    }


    // ─────────────────────────────────────────────────────────────────────────
    // Issue 742 T9.9 — the p-row batched speculative VERIFY chunk (the Q4_K
    // verify port). `forward_verify_chunk` feeds `p <= 16` draft tokens at
    // `base_pos` through the whole 64-layer stack on the rows-variant
    // kernels (each VERBATIM the decode kernel's arithmetic per row — see
    // the kernel docs) and returns the greedy argmax after every position.
    // The chunk mutates the same KV/GDN state the decode path uses; the
    // GDN state is snapshot/rollback-able for the loop's mismatch path (KV
    // rows >= base_pos are overwritten-before-read by any re-feed, so they
    // need no snapshot — the T1.8 lesson).
    // ─────────────────────────────────────────────────────────────────────────

    /// Snapshot the GDN state (recurrent + conv, dtod on the compute
    /// stream) so a rejected verify chunk can be rolled back.
    pub fn verify_snapshot_gdn(&mut self) -> Result<(), String> {
        let stream = &self.stream;
        for (s, r) in self
            .verify
            .snap_recurrent
            .iter_mut()
            .zip(self.state.recurrent.iter())
        {
            stream
                .memcpy_dtod(r, s)
                .map_err(|e| format!("snap rec dtod: {e}"))?;
        }
        for (s, c) in self.verify.snap_conv.iter_mut().zip(self.state.conv.iter()) {
            stream
                .memcpy_dtod(c, s)
                .map_err(|e| format!("snap conv dtod: {e}"))?;
        }
        Ok(())
    }

    /// Restore the GDN state from the snapshot (dtod, stream-ordered).
    pub fn verify_rollback_gdn(&mut self) -> Result<(), String> {
        let stream = &self.stream;
        for (s, r) in self
            .verify
            .snap_recurrent
            .iter()
            .zip(self.state.recurrent.iter_mut())
        {
            stream
                .memcpy_dtod(s, r)
                .map_err(|e| format!("rollback rec dtod: {e}"))?;
        }
        for (s, c) in self.verify.snap_conv.iter().zip(self.state.conv.iter_mut()) {
            stream
                .memcpy_dtod(s, c)
                .map_err(|e| format!("rollback conv dtod: {e}"))?;
        }
        Ok(())
    }

    /// Zero the GDN state (the construction state) — the Phase-C loop's
    /// re-fill reset: after this, re-feeding the prompt reproduces the
    /// post-prompt state exactly (all prompt kernels are deterministic).
    pub fn verify_reset_gdn(&mut self) -> Result<(), String> {
        let stream = &self.stream;
        for r in self.state.recurrent.iter_mut() {
            stream.memset_zeros(r).map_err(|e| e.to_string())?;
        }
        for c in self.state.conv.iter_mut() {
            stream.memset_zeros(c).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 755 T4 — allocate the per-GDN-layer replay journal (~82 MB at
    /// the dbirks dims: 44 layers × (16×10240 + 16×18432 + 2×16×48) f32).
    /// Call once before the first journaled chunk; idempotent. While enabled,
    /// every eager verify chunk pays 4 small copy-kernel launches per GDN
    /// layer (~0.3 ms at p=8, ~0.7% of the chunk) to capture its recurrence
    /// inputs — measure with that included (it is the mechanism's honest
    /// cost, kept in the same `verify_ms` window by stream ordering).
    pub fn enable_gdn_journal(&mut self) -> Result<(), String> {
        if self.gdn_journal.is_some() {
            return Ok(());
        }
        let n_gdn = self
            .cfg
            .layer_types
            .iter()
            .filter(|t| **t == Qwen38LayerType::Deltanet)
            .count();
        let l_qkv_out =
            2 * self.cfg.n_k_heads * self.cfg.head_k_dim + self.cfg.n_v_heads * self.cfg.head_v_dim;
        let l_exp = 3 * self.cfg.n_v_heads * self.cfg.head_k_dim;
        let p = QWEN38_VERIFY_MAX_P;
        let mut qkv = Vec::with_capacity(n_gdn);
        let mut exp = Vec::with_capacity(n_gdn);
        let mut beta = Vec::with_capacity(n_gdn);
        let mut decay = Vec::with_capacity(n_gdn);
        for _ in 0..n_gdn {
            let a = |len: usize| -> Result<CudaSlice<f32>, String> {
                self.stream
                    .alloc_zeros::<f32>(len)
                    .map_err(|e| format!("journal alloc: {e}"))
            };
            qkv.push(a(p * l_qkv_out)?);
            exp.push(a(p * l_exp)?);
            beta.push(a(p * self.cfg.n_v_heads)?);
            decay.push(a(p * self.cfg.n_v_heads)?);
        }
        self.gdn_journal = Some(Qwen38GdnJournal {
            qkv,
            exp,
            beta,
            decay,
        });
        Ok(())
    }

    /// Issue 755 T4 — advance the GDN state by the first `j` journaled rows
    /// WITHOUT re-running the weight-reading stack. Precondition: the caller
    /// ran [`Self::verify_rollback_gdn`] (state at the chunk's base) AND the
    /// journal was captured by the chunk being rewound (no intervening verify
    /// chunk overwrites it). Re-launches the same `conv1d_rows` +
    /// `recurrence_fused_rows_hd128` kernels the chunk used, on `j` rows:
    /// both evolve their state in registers over a sequential per-token loop
    /// and store once at the end, so the `j`-row call reproduces the first
    /// `j` iterations of the capturing `p`-row call BIT-IDENTICALLY (same
    /// kernels, same inputs, same op order — the state-equivalence is gated
    /// directly in the Bench-759 RP diag arm via `dump_gdn_state`).
    ///
    /// Cost at the dbirks dims: 44 layers × (2 launches, ~4.3 MB slice dtod,
    /// and the 6 MB state load/store per recurrence) ≈ 0.3–1 ms total —
    /// replacing the ~31 ms weight-reading advance chunk.
    pub fn verify_advance_gdn_replay(&mut self, j: usize) -> Result<(), String> {
        let Some(jr) = self.gdn_journal.as_ref() else {
            return Err("verify_advance_gdn_replay: journal not enabled".into());
        };
        if j == 0 || j > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "verify_advance_gdn_replay: j must be in 1..={QWEN38_VERIFY_MAX_P} (got {j})"
            ));
        }
        let cfg = &self.cfg;
        let l_qkv_out = 2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
        let l_exp = 3 * cfg.n_v_heads * cfg.head_k_dim;
        let n_v = cfg.n_v_heads;
        let mut gdn_idx = 0usize;
        for layer_idx in 0..cfg.n_layer {
            if !matches!(cfg.layer_types[layer_idx], Qwen38LayerType::Deltanet) {
                continue;
            }
            let conv_w = self.weights.layers[layer_idx]
                .conv1d
                .as_ref()
                .expect("gdn layer conv1d weight");
            {
                let stream = &self.stream;
                let n = j * l_qkv_out;
                let src = jr.qkv[gdn_idx].slice(0..n);
                let mut dst = self.verify.qkv.slice_mut(0..n);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| format!("replay qkv dtod: {e}"))?;
            }
            self.dn
                .launch_conv1d_rows(
                    &self.stream,
                    &self.verify.qkv,
                    &conv_w.dev,
                    &self.state.conv[gdn_idx],
                    conv_w.len / cfg.conv_kernel,
                    cfg.conv_kernel,
                    j,
                )
                .map_err(|e| format!("replay conv: {e}"))?;
            {
                let stream = &self.stream;
                let n = j * l_exp;
                let src = jr.exp[gdn_idx].slice(0..n);
                let mut dst = self.verify.qkv_exp.slice_mut(0..n);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| format!("replay exp dtod: {e}"))?;
                let n = j * n_v;
                let src = jr.beta[gdn_idx].slice(0..n);
                let mut dst = self.verify.beta.slice_mut(0..n);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| format!("replay beta dtod: {e}"))?;
                let src = jr.decay[gdn_idx].slice(0..n);
                let mut dst = self.verify.decay.slice_mut(0..n);
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| format!("replay decay dtod: {e}"))?;
            }
            self.dn
                .launch_recurrence_fused_rows_hd128(
                    &self.stream,
                    &self.verify.qkv_exp,
                    &self.verify.beta,
                    &self.verify.decay,
                    &self.state.recurrent[gdn_idx],
                    &self.verify.rec_out,
                    n_v,
                    j,
                )
                .map_err(|e| format!("replay recurrence: {e}"))?;
            gdn_idx += 1;
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Plan 556 Stage 1 (Issue 780 U2) — decode-only lanes. ONE packed
    // forward over `n` independent sequences: the row-agnostic ops ride the
    // existing `_rows` kernels at p = n (mechanism D's lane dimension is
    // free — the port is the lane dimension, not kernels); the per-lane
    // stateful ops (GDN conv + recurrence, attention rope + KV-append +
    // scores) launch the SOLO p=1 kernels per lane on zero-copy slice
    // views of the lane arenas. No staging copies. The G1 property (each
    // lane's stream bit-identical to solo `forward_verify_chunk(p=1)` runs
    // of the same sequence) holds by construction: per-lane ops ARE the
    // solo kernels at p=1 on lane-owned state, and the packed ops are
    // row-independent (the chunk's own 0/256 argmax pin).
    // ─────────────────────────────────────────────────────────────────────

    /// Allocate the lane arenas: `n` sequences × `lane_ctx` positions of KV
    /// each + per-lane GDN state. F32 KV only (the f16 hatch is a solo-path
    /// bake). Replaces any previous lane set (device memory frees on drop).
    pub fn enable_lanes(&mut self, n: usize, lane_ctx: usize) -> Result<(), String> {
        if n == 0 || n > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "enable_lanes: n must be in 1..={QWEN38_VERIFY_MAX_P} (got {n})"
            ));
        }
        if lane_ctx == 0 || lane_ctx > self.ctx_len {
            return Err(format!(
                "enable_lanes: lane_ctx must be in 1..={} (got {lane_ctx})",
                self.ctx_len
            ));
        }
        if self.kv_f16 {
            return Err("enable_lanes: the f16 KV hatch is unsupported in lane mode (stage 1)".into());
        }
        let cfg = &self.cfg;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let l_qkv_out = 2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
        let n_attn = cfg.layer_types.iter().filter(|t| **t == Qwen38LayerType::Attention).count();
        let n_gdn = cfg.layer_types.iter().filter(|t| **t == Qwen38LayerType::Deltanet).count();
        let mut keys = Vec::with_capacity(n_attn);
        let mut values = Vec::with_capacity(n_attn);
        let mut recurrent = Vec::with_capacity(n_gdn);
        let mut conv = Vec::with_capacity(n_gdn);
        for lt in &cfg.layer_types {
            match lt {
                Qwen38LayerType::Attention => {
                    keys.push(
                        self.stream
                            .alloc_zeros::<f32>(n * lane_ctx * kvd)
                            .map_err(|e| format!("lanes keys alloc: {e}"))?,
                    );
                    values.push(
                        self.stream
                            .alloc_zeros::<f32>(n * lane_ctx * kvd)
                            .map_err(|e| format!("lanes values alloc: {e}"))?,
                    );
                }
                Qwen38LayerType::Deltanet => {
                    recurrent.push(
                        self.stream
                            .alloc_zeros::<f32>(
                                n * cfg.n_v_heads * cfg.head_k_dim * cfg.head_v_dim,
                            )
                            .map_err(|e| format!("lanes recurrent alloc: {e}"))?,
                    );
                    conv.push(
                        self.stream
                            .alloc_zeros::<f32>(n * l_qkv_out * cfg.conv_kernel)
                            .map_err(|e| format!("lanes conv alloc: {e}"))?,
                    );
                }
            }
        }
        self.lanes = Some(Qwen38LaneSet {
            n,
            lane_ctx,
            keys,
            values,
            recurrent,
            conv,
            snap_recurrent: None,
            snap_conv: None,
            pos_dev: self
                .stream
                .alloc_zeros::<i32>(n)
                .map_err(|e| format!("lanes pos_dev alloc: {e}"))?,
            graphs: std::collections::HashMap::new(),
            graph_failed: std::collections::HashSet::new(),
        });
        // The boundary discards any pending commit of the REPLACED set (the
        // old journal rows belong to the old arenas; the new set's snapshots
        // are not allocated yet, so a stale flag could only ever error —
        // clear it so the boundary is a clean protocol reset).
        self.lanes_commit_ready = None;
        Ok(())
    }

    /// Plan 556 Stage 3 (mechanism I) — the request-boundary teardown: drop
    /// the lane arenas, snapshots, and the captured lane graphs, and clear
    /// the commit state. cudarc frees each arena with `cuMemFreeAsync` back
    /// to the driver's async mempool (no internal block cache), so the drop
    /// IS the trim — the next `enable_lanes` of the same shape reuses the
    /// pool without new device reservations. The teardown SYNCHRONIZES
    /// before returning: the stream-ordered frees must be COMPLETE before
    /// the boundary returns — an immediate re-`enable_lanes` on a still-
    /// draining stream failed with `CUDA_ERROR_INVALID_VALUE` on the GDN
    /// arena re-alloc (measured, the Stage-3 I-rider gate) — the sync makes
    /// the boundary deterministic. Idempotent (a second call is a no-op).
    /// Any uncommitted pending chunk is DISCARDED — this is the protocol's
    /// abort path, not a mid-protocol operation.
    pub fn disable_lanes(&mut self) {
        let had_lanes = self.lanes.is_some();
        self.lanes = None;
        self.lanes_commit_ready = None;
        if had_lanes {
            // Drain the frees the drop enqueued (see the doc above).
            let _ = self.stream.synchronize();
        }
    }

    /// `(n, lane_ctx)` when lanes are enabled (harness/diagnostic use).
    pub fn lanes_enabled(&self) -> Option<(usize, usize)> {
        self.lanes.as_ref().map(|l| (l.n, l.lane_ctx))
    }

    /// Zero the lane arenas — fresh sequences without a re-alloc (the
    /// construction-zeroed state is the model's expected t=0 init).
    pub fn zero_lanes(&mut self) -> Result<(), String> {
        let Some(lanes) = self.lanes.as_mut() else {
            return Err("zero_lanes: lanes not enabled".into());
        };
        for k in lanes.keys.iter_mut() {
            self.stream.memset_zeros(k).map_err(|e| e.to_string())?;
        }
        for v in lanes.values.iter_mut() {
            self.stream.memset_zeros(v).map_err(|e| e.to_string())?;
        }
        for r in lanes.recurrent.iter_mut() {
            self.stream.memset_zeros(r).map_err(|e| e.to_string())?;
        }
        for c in lanes.conv.iter_mut() {
            self.stream.memset_zeros(c).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Zero the SOLO decode state (KV + GDN) — fresh single-sequence runs
    /// without an engine re-construction (the G1 harness's per-sequence
    /// reset). Semantically identical to construction-time zeros: all-zero
    /// f32 bits, stream-ordered, no re-alloc (the ternary engine's
    /// reset_state memset pattern).
    pub fn zero_state(&mut self) -> Result<(), String> {
        for k in self.state.keys.iter_mut() {
            self.stream.memset_zeros(k).map_err(|e| e.to_string())?;
        }
        for v in self.state.values.iter_mut() {
            self.stream.memset_zeros(v).map_err(|e| e.to_string())?;
        }
        for r in self.state.recurrent.iter_mut() {
            self.stream.memset_zeros(r).map_err(|e| e.to_string())?;
        }
        for c in self.state.conv.iter_mut() {
            self.stream.memset_zeros(c).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Plan 556 Stage 1 — ONE packed decode step for `n` independent
    /// sequences: lane `l` consumes `tokens[l]` at `positions[l]`; the
    /// return is the greedy argmax per lane. The caller owns position
    /// bookkeeping (the same contract as
    /// [`Self::forward_verify_chunk`]'s `base_pos`). Each lane's KV/GDN
    /// state lives in the lane arenas — `self.state` and the prefix cache
    /// are untouched.
    ///
    /// Stage-1 scope cuts, each erroring loudly: graph capture (eager
    /// only), f16 KV, the exotic attention knobs (mma/2p/dotma — measured-
    /// negative opt-in artifacts with zero lane consumers; the queue's
    /// "default-arm" arbitration keeps them out of lane mode), and the
    /// prefix cache (lane KV is lane-owned; the lineage rule concerns
    /// `self.state` only). The plain qg attention arm IS supported per lane
    /// — it is the default long-ctx arm and the solo chunk resolves it per
    /// chunk the same way. Stage 2: the GDN journal is no longer an error
    /// here (lane verify needs it) — a decode step merely INVALIDATES any
    /// pending verify commit (the journal is only valid for the chunk that
    /// captured it).
    #[allow(clippy::too_many_lines)]
    pub fn forward_lanes_decode(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<u32>, String> {
        let (n, lane_ctx) = {
            let Some(lanes) = self.lanes.as_ref() else {
                return Err("forward_lanes_decode: call enable_lanes() first".into());
            };
            (lanes.n, lanes.lane_ctx)
        };
        if tokens.len() != n || positions.len() != n {
            return Err(format!(
                "forward_lanes_decode: tokens/positions must be len {n} (got {} / {})",
                tokens.len(),
                positions.len()
            ));
        }
        if self.capturing {
            return Err("forward_lanes_decode: eager-only in stage 1 (no graph capture)".into());
        }
        // Stage 2: a decode step invalidates any pending verify commit — the
        // per-lane journal is only valid for the chunk that captured it.
        self.lanes_commit_ready = None;
        if self.verify_attn_mma || self.verify_attn_2p || self.verify_attn_dotma {
            return Err("forward_lanes_decode: exotic attention knobs (mma/2p/dotma) unsupported in lane mode (stage 1)".into());
        }
        for (l, &pos) in positions.iter().enumerate() {
            if pos + 1 > lane_ctx {
                return Err(format!(
                    "forward_lanes_decode: lane {l} pos {pos} + 1 exceeds lane_ctx {lane_ctx}"
                ));
            }
        }
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd;
        let p = n;
        // G4 (the chunk's rule): stack staging — p <= QWEN38_VERIFY_MAX_P,
        // so the i32 upload buffer never touches the allocator.
        let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
        for (i, &t) in tokens.iter().enumerate() {
            toks[i] = t as i32;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
        }
        // Embedding rows (row-agnostic — the packed D dimension).
        {
            let w = &self.weights.token_embd;
            let v = &self.verify;
            // SAFETY: out covers p*n; tokens covers p.
            unsafe {
                self.dense.launch_dequant_q4k_rows(
                    &self.stream,
                    &w.dev,
                    &v.xb,
                    &v.tokens_dev,
                    n_embd,
                    w.blocks_per_row,
                    p,
                )
            }?;
        }
        // Per-lane q-group resolution (the solo chunk resolves once per
        // chunk; lanes have per-lane positions, so per-lane arms).
        let mut use_qg = [false; QWEN38_VERIFY_MAX_P];
        for (l, &pos) in positions.iter().enumerate() {
            use_qg[l] = self.verify_use_qg(pos, 1);
        }
        let lanes = self.lanes.as_ref().unwrap();
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 — the fused boundary chain, p = n (row-agnostic).
            // VERBATIM the verify_chunk_eager fused arm's site order.
            self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
            self.rmsnorm_quant_x_rows(
                &self.verify.xb,
                &self.weights.layers[0].input_norm.dev,
                n_embd,
                cfg.rms_norm_eps,
                p,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.lanes_gdn_layer(lanes, i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.lanes_attn_layer(lanes, i, attn_idx, positions, &use_qg, p)?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x_rows(
                    &self.verify.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                    p,
                )?;
                self.verify_mlp(i, p)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x_rows(
                        &self.verify.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                        p,
                    )?;
                } else {
                    // Last layer: plain residual — the tail below owns
                    // output_norm + lm_head.
                    self.ew
                        .launch_residual_add(
                            &self.stream,
                            &self.verify.xb_res,
                            &self.verify.y,
                            &self.verify.xb,
                            n_embd * p,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.lanes_gdn_layer(lanes, i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.lanes_attn_layer(lanes, i, attn_idx, positions, &use_qg, p)?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
                self.verify_mlp(i, p)?;
            }
        }
        // Final norm + lm_head + per-lane argmax (verbatim the chunk tail).
        {
            let v = &self.verify;
            self.rmsnorm_quant_x_rows(
                &v.xb,
                &self.weights.output_norm.dev,
                n_embd,
                cfg.rms_norm_eps,
                p,
            )?;
            self.gemv_quant_rows(&self.weights.lm_head, &v.logits, p)?;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(ZERO_U64_16.as_slice(), &mut self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            // SAFETY: logits covers p*vocab; argmax_res covers p (zeroed).
            unsafe {
                self.dense.launch_argmax_rows(
                    stream,
                    &self.verify.logits,
                    cfg.vocab_size,
                    p,
                    &self.verify.argmax_res,
                )
            }?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let packed = stream
                .clone_dtoh(&self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            Ok(packed[..p].iter().map(|&pk| !(pk as u32)).collect())
        }
    }

    /// The lane GDN layer: packed row-agnostic ops at p = n (GEMVs,
    /// beta/decay, expand_l2, zgate, ssm_out) with the PER-LANE stateful
    /// ops (conv, recurrence) launched per lane at p=1 on slice views of
    /// the lane arenas — the solo chunk's kernels and op order verbatim for
    /// every lane (the G1 property). The conv writes its output in place
    /// over the lane's packed qkv row (the kernel sees the sliced view as
    /// its row 0); the packed expand then reads all conv outputs in one
    /// launch; the recurrence reads the lane's expanded row and evolves the
    /// lane's arena state.
    fn lanes_gdn_layer(
        &self,
        lanes: &Qwen38LaneSet,
        layer_idx: usize,
        gdn_idx: usize,
        p: usize,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let n_v = cfg.n_v_heads;
        let l_exp = 3 * n_v * cfg.head_k_dim;
        let conv_dim = 2 * cfg.n_k_heads * cfg.head_k_dim + n_v * cfg.head_v_dim;
        let rec_len = n_v * cfg.head_k_dim * cfg.head_v_dim;
        let v = &self.verify;
        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.input_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        self.gemv_quant_rows(lw.qkv.as_ref().unwrap(), &v.qkv, p)?;
        self.gemv_quant_rows(lw.z.as_ref().unwrap(), &v.z, p)?;
        self.gemv_quant_rows(lw.alpha.as_ref().unwrap(), &v.a_raw, p)?;
        self.gemv_quant_rows(lw.beta.as_ref().unwrap(), &v.b_raw, p)?;
        self.dn
            .launch_beta_decay_rows(
                stream,
                &v.a_raw,
                &v.b_raw,
                &lw.a_log.as_ref().unwrap().dev,
                &lw.dt_bias.as_ref().unwrap().dev,
                &v.beta,
                &v.decay,
                n_v,
                p,
            )
            .map_err(|e| e.to_string())?;
        // Per-lane conv (state rides the lane arena slice).
        let conv_w = lw.conv1d.as_ref().unwrap();
        for l in 0..p {
            let off = l * conv_dim;
            let input = v.qkv.slice(off..off + conv_dim);
            let soff = l * conv_dim * cfg.conv_kernel;
            let state = lanes.conv[gdn_idx].slice(soff..soff + conv_dim * cfg.conv_kernel);
            self.dn
                .launch_conv1d_rows(stream, &input, &conv_w.dev, &state, conv_dim,
                    cfg.conv_kernel, 1)
                .map_err(|e| e.to_string())?;
        }
        // Packed expand (row-agnostic over the conv outputs).
        self.dn
            .launch_expand_l2_rows(stream, &v.qkv, &v.qkv_exp, cfg.n_k_heads, n_v,
                cfg.head_k_dim, p)
            .map_err(|e| e.to_string())?;
        // Per-lane recurrence (the state slice view IS the lane's state).
        for l in 0..p {
            let off = l * l_exp;
            let qkv_in = v.qkv_exp.slice(off..off + l_exp);
            let boff = l * n_v;
            let beta = v.beta.slice(boff..boff + n_v);
            let decay = v.decay.slice(boff..boff + n_v);
            let soff = l * rec_len;
            let state = lanes.recurrent[gdn_idx].slice(soff..soff + rec_len);
            let ooff = l * cfg.d_inner;
            let out = v.rec_out.slice(ooff..ooff + cfg.d_inner);
            self.dn
                .launch_recurrence_fused_rows_hd128(stream, &qkv_in, &beta, &decay, &state,
                    &out, n_v, 1)
                .map_err(|e| e.to_string())?;
        }
        // SAFETY: rec_out/z cover p*d_inner; xq/xs/xsum cover p*(d_inner/16).
        unsafe {
            self.dense.launch_rmsnorm_zgate_quant_x_q8(
                stream,
                &v.rec_out,
                &v.z,
                &lw.ssm_norm.as_ref().unwrap().dev,
                &v.xq,
                &v.xs,
                &v.xsum,
                p * n_v,
                cfg.head_v_dim,
                cfg.rms_norm_eps,
            )
        }?;
        self.gemv_quant_rows(lw.ssm_out.as_ref().unwrap(), &v.y, p)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel.
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The lane attention layer: packed projections/norms at p = n, then
    /// PER-LANE rope → KV-append → split-GQA scores at p=1 (each lane's
    /// position and KV region are its own; the solo chunk's arms verbatim —
    /// the non-qg arm pins the FIXED `attn_n_chunks` count per the
    /// Bench-759 lesson, the qg arm's live count matches the solo Live arm).
    #[allow(clippy::too_many_lines)]
    fn lanes_attn_layer(
        &self,
        lanes: &Qwen38LaneSet,
        layer_idx: usize,
        attn_idx: usize,
        positions: &[usize],
        use_qg: &[bool; QWEN38_VERIFY_MAX_P],
        p: usize,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let lane_kv = lanes.lane_ctx * kvd;
        let v = &self.verify;
        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.input_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        self.gemv_quant_rows(lw.wq.as_ref().unwrap(), &v.qg, p)?;
        self.attn
            .launch_split_qg_rows(stream, &v.qg, &v.q, &v.gate, cfg.head_dim, cfg.n_head, p)
            .map_err(|e| e.to_string())?;
        self.gemv_quant_rows(lw.wk.as_ref().unwrap(), &v.k, p)?;
        self.gemv_quant_rows(lw.wv.as_ref().unwrap(), &v.vv, p)?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &v.q,
                &lw.q_norm.as_ref().unwrap().dev,
                &v.q_normed,
                p * cfg.n_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &v.k,
                &lw.k_norm.as_ref().unwrap().dev,
                &v.k_normed,
                p * cfg.n_kv_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        // Per-lane rope FIRST (the cache must receive the ROPED k — the
        // solo chunk's site order), then per-lane KV append into the lane's
        // cache region, then per-lane scores.
        for (l, &pos) in positions.iter().enumerate().take(p) {
            let qoff = l * q_dim;
            let qrow = v.q_normed.slice(qoff..qoff + q_dim);
            let koff = l * kvd;
            let krow = v.k_normed.slice(koff..koff + kvd);
            self.attn
                .launch_rope_rows(stream, &qrow, &krow, cfg.rotary_dim, cfg.head_dim,
                    cfg.n_head, cfg.n_kv_head, pos, 1, cfg.rope_theta)
                .map_err(|e| e.to_string())?;
        }
        for (l, &pos) in positions.iter().enumerate().take(p) {
            let koff = l * kvd;
            let krow = v.k_normed.slice(koff..koff + kvd);
            let voff = l * kvd;
            let vrow = v.vv.slice(voff..voff + kvd);
            let coff = l * lane_kv;
            let kc = lanes.keys[attn_idx].slice(coff..coff + lane_kv);
            let vc = lanes.values[attn_idx].slice(coff..coff + lane_kv);
            self.attn
                .launch_kv_append_rows(stream, &krow, &vrow, &kc, &vc, kvd, pos, 1)
                .map_err(|e| e.to_string())?;
        }
        for l in 0..p {
            let coff = l * lane_kv;
            let kc = lanes.keys[attn_idx].slice(coff..coff + lane_kv);
            let vc = lanes.values[attn_idx].slice(coff..coff + lane_kv);
            let qoff = l * q_dim;
            let qrow = v.q_normed.slice(qoff..qoff + q_dim);
            // The scratch partials are sized [MAX_P][n_head][verify_attn_n_chunks]
            // (the max column count); each lane's slice is its row slot.
            let pstride = cfg.n_head * self.verify_attn_n_chunks;
            let pm = v.part_m.slice(l * pstride..(l + 1) * pstride);
            let pl = v.part_l.slice(l * pstride..(l + 1) * pstride);
            let postride = pstride * cfg.head_dim;
            let po = v.part_out.slice(l * postride..(l + 1) * postride);
            let aoff = l * q_dim;
            let arow = v.attn_out.slice(aoff..aoff + q_dim);
            let attn_res = if use_qg[l] {
                let n_chunks = (positions[l] + 1).div_ceil(self.verify_attn_chunk_len).max(1);
                self.attn.launch_attention_splitgqa_rows_qg(
                    stream, &qrow, &kc, &vc, &pm, &pl, &po, &arow, cfg.head_dim,
                    cfg.n_head, cfg.n_kv_head, self.verify_attn_chunk_len, n_chunks,
                    positions[l], 1,
                )
            } else {
                self.attn.launch_attention_splitgqa_rows(
                    stream, &qrow, &kc, &vc, &pm, &pl, &po, &arow, cfg.head_dim,
                    cfg.n_head, cfg.n_kv_head, self.attn_chunk_len, self.attn_n_chunks,
                    positions[l], 1,
                )
            };
            attn_res.map_err(|e| e.to_string())?;
        }
        self.attn
            .launch_output_gate(stream, &v.attn_out, &v.gate, q_dim * p)
            .map_err(|e| e.to_string())?;
        // SAFETY: flat buffers, q_dim*p multiple of 16.
        unsafe {
            self.dense
                .launch_quant_x_q8(stream, &v.attn_out, &v.xq, &v.xs, &v.xsum, q_dim * p)
        }?;
        self.gemv_quant_rows(lw.wo.as_ref().unwrap(), &v.y, p)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel.
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Plan 556 Stage 2 (Issue 780 U2) — per-lane VERIFY chunks + the B/G/H
    // commit contract. Each lane runs its own K-wide speculative chunk at
    // its own base position; the packed row-agnostic ops ride the verify
    // scratch at p = lanes*K (ONE weight pass per cycle — lanes*K ≤ 16),
    // the per-lane stateful ops launch the solo K-row kernels on slice
    // views of the lane arenas (the Stage-1 pattern at K rows instead of
    // 1 — G1 holds by the same construction: per-lane ops ARE the solo
    // K-row kernels on lane-owned state, the packed ops are row-agnostic).
    //
    // Commit contract, the lucebox G/H mapping: the GDN snapshot is taken
    // BEFORE the chunk's first mutation (the staged-append rollback
    // medium — G); KV needs NO rollback (the Issue 746 full-length
    // append-only immunity: rejected rows are repaired by the next chunk's
    // write-before-read); the accept advance replays the per-lane journal
    // prefix WITHOUT re-running the weight stack (mechanism B —
    // `gdn-transition-replay-log-commit-many`, landed solo as Issue 755
    // T4). Fail-closed (H): every check runs before the first mutation;
    // once the chunk mutates, an error leaves the lanes TORN and the only
    // recovery is zero_lanes() + refill — no partial repair is attempted
    // or promised.
    // ─────────────────────────────────────────────────────────────────────

    /// Stage 2 — allocate the per-lane GDN snapshot arenas (the rollback
    /// medium of the commit contract). Required before the first
    /// [`Self::forward_lanes_verify`]. Idempotent. ~0.6 GB at C4/K4 on the
    /// dbirks dims (the C16 decode lane set needs none — decode never
    /// commits).
    pub fn enable_lanes_snapshots(&mut self) -> Result<(), String> {
        let Some(ls) = self.lanes.as_mut() else {
            return Err("enable_lanes_snapshots: call enable_lanes() first".into());
        };
        if ls.snap_recurrent.is_some() {
            return Ok(());
        }
        let cfg = &self.cfg;
        let n = ls.n;
        let rec_len = cfg.n_v_heads * cfg.head_k_dim * cfg.head_v_dim;
        let l_qkv_out =
            2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
        let conv_len = l_qkv_out * cfg.conv_kernel;
        let n_gdn = cfg
            .layer_types
            .iter()
            .filter(|t| **t == Qwen38LayerType::Deltanet)
            .count();
        let mut snap_recurrent = Vec::with_capacity(n_gdn);
        let mut snap_conv = Vec::with_capacity(n_gdn);
        for _ in 0..n_gdn {
            snap_recurrent.push(
                self.stream
                    .alloc_zeros::<f32>(n * rec_len)
                    .map_err(|e| format!("lane snap recurrent alloc: {e}"))?,
            );
            snap_conv.push(
                self.stream
                    .alloc_zeros::<f32>(n * conv_len)
                    .map_err(|e| format!("lane snap conv alloc: {e}"))?,
            );
        }
        ls.snap_recurrent = Some(snap_recurrent);
        ls.snap_conv = Some(snap_conv);
        Ok(())
    }

    /// Snapshot-arena presence (harness/diagnostic use).
    pub fn lanes_snapshots_enabled(&self) -> bool {
        self.lanes
            .as_ref()
            .is_some_and(|l| l.snap_recurrent.is_some())
    }

    /// Snapshot the lane GDN arenas (dtod, stream-ordered). INTERNAL: called
    /// at [`Self::forward_lanes_verify`] entry, after validation and before
    /// the chunk's first mutation (the G-half of the contract — a rollback
    /// medium that predates every mutation).
    fn lanes_snapshot_gdn_inner(&mut self) -> Result<(), String> {
        let Some(ls) = self.lanes.as_mut() else {
            return Err("lane snapshot: lanes not enabled".into());
        };
        let Some(sr) = ls.snap_recurrent.as_mut() else {
            return Err("lane snapshot: call enable_lanes_snapshots() first".into());
        };
        let Some(sc) = ls.snap_conv.as_mut() else {
            return Err("lane snapshot: call enable_lanes_snapshots() first".into());
        };
        for (snap, live) in sr.iter_mut().zip(ls.recurrent.iter()) {
            self.stream
                .memcpy_dtod(live, snap)
                .map_err(|e| format!("lane snap rec dtod: {e}"))?;
        }
        for (snap, live) in sc.iter_mut().zip(ls.conv.iter()) {
            self.stream
                .memcpy_dtod(live, snap)
                .map_err(|e| format!("lane snap conv dtod: {e}"))?;
        }
        Ok(())
    }

    /// Plan 556 Stage 2 — per-lane K-wide verify chunk. Lane `l` consumes
    /// `tokens[l][0..k]` at positions `base_positions[l] .. +k`; the return
    /// is the greedy argmax per row per lane (`[n][k]`). The packed ops run
    /// at p = n*k rows (ONE weight pass — the row budget lanes*k ≤ 16); the
    /// per-lane stateful ops (conv, recurrence, rope, KV-append, scores)
    /// launch the solo K-row kernels on slice views of the lane arenas.
    ///
    /// The GDN snapshot is taken at entry (after validation, before the
    /// first mutation). The caller compares drafts vs argmaxes, computes
    /// the per-lane accepted prefix, and commits via [`Self::lanes_commit`]
    /// BEFORE the next chunk (the `lanes_commit_ready` gate enforces it).
    ///
    /// Stage-2 scope: default attention arms only per lane (qg resolved
    /// per-lane exactly like the solo chunk; rows arm for short ctx). The
    /// exotic knobs (mma/2p/dotma), f16 KV, graph capture, and the prefix
    /// cache error loudly — measured-negative opt-in artifacts with zero
    /// lane consumers (the queue's "default-arm" arbitration).
    #[allow(clippy::too_many_lines)]
    pub fn forward_lanes_verify(
        &mut self,
        tokens: &[Vec<u32>],
        base_positions: &[usize],
    ) -> Result<Vec<Vec<u32>>, String> {
        // ── H: validate EVERYTHING before the first mutation ──
        let (n, _lane_ctx, k, p) =
            self.lanes_verify_precheck(tokens, base_positions, "forward_lanes_verify")?;
        // ── G: the rollback medium predates every mutation ──
        self.lanes_snapshot_gdn_inner()?;
        self.lanes_commit_ready = None;

        let cfg = &self.cfg;
        let n_embd = cfg.n_embd;
        // G4 (the chunk's rule): stack staging — p ≤ QWEN38_VERIFY_MAX_P.
        let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
        for (i, t) in tokens.iter().flatten().enumerate() {
            toks[i] = *t as i32;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
        }
        // Embedding rows (row-agnostic — the packed D dimension).
        {
            let w = &self.weights.token_embd;
            let v = &self.verify;
            // SAFETY: out covers p*n; tokens covers p.
            unsafe {
                self.dense.launch_dequant_q4k_rows(
                    &self.stream,
                    &w.dev,
                    &v.xb,
                    &v.tokens_dev,
                    n_embd,
                    w.blocks_per_row,
                    p,
                )
            }?;
        }
        // Per-lane q-group resolution (the solo chunk resolves once per
        // chunk; lanes have per-lane positions).
        let mut use_qg = [false; QWEN38_VERIFY_MAX_P];
        for (l, &bp) in base_positions.iter().enumerate() {
            use_qg[l] = self.verify_use_qg(bp, k);
        }
        let lanes = self.lanes.as_ref().unwrap();
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 — the fused boundary chain, p = n*k (row-agnostic).
            // VERBATIM the verify_chunk_eager fused arm's site order.
            self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
            self.rmsnorm_quant_x_rows(
                &self.verify.xb,
                &self.weights.layers[0].input_norm.dev,
                n_embd,
                cfg.rms_norm_eps,
                p,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.lanes_verify_gdn_layer(lanes, i, gdn_idx, p, k)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.lanes_verify_attn_layer(
                            lanes,
                            i,
                            attn_idx,
                            LaneVerifyPos::Live(base_positions),
                            &use_qg,
                            p,
                            k,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x_rows(
                    &self.verify.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                    p,
                )?;
                self.verify_mlp(i, p)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x_rows(
                        &self.verify.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                        p,
                    )?;
                } else {
                    self.ew
                        .launch_residual_add(
                            &self.stream,
                            &self.verify.xb_res,
                            &self.verify.y,
                            &self.verify.xb,
                            n_embd * p,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.lanes_verify_gdn_layer(lanes, i, gdn_idx, p, k)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.lanes_verify_attn_layer(
                            lanes,
                            i,
                            attn_idx,
                            LaneVerifyPos::Live(base_positions),
                            &use_qg,
                            p,
                            k,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
                self.verify_mlp(i, p)?;
            }
        }
        // Final norm + lm_head + per-row argmax (verbatim the chunk tail).
        {
            let v = &self.verify;
            self.rmsnorm_quant_x_rows(
                &v.xb,
                &self.weights.output_norm.dev,
                n_embd,
                cfg.rms_norm_eps,
                p,
            )?;
            self.gemv_quant_rows(&self.weights.lm_head, &v.logits, p)?;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(ZERO_U64_16.as_slice(), &mut self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            // SAFETY: logits covers p*vocab; argmax_res covers p (zeroed).
            unsafe {
                self.dense.launch_argmax_rows(
                    stream,
                    &self.verify.logits,
                    cfg.vocab_size,
                    p,
                    &self.verify.argmax_res,
                )
            }?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let packed = stream
                .clone_dtoh(&self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            let out: Vec<u32> = packed[..p].iter().map(|&pk| !(pk as u32)).collect();
            self.lanes_commit_ready = Some(k);
            Ok((0..n)
                .map(|l| out[l * k..(l + 1) * k].to_vec())
                .collect())
        }
    }

    /// The shared lanes-verify validation (the H half of the fail-closed
    /// contract), used verbatim by the eager chunk AND the Stage-3 graphed
    /// entry: lanes enabled, lens match, uniform `k`, `n*k` within the row
    /// budget, every lane's `bp + k` within `lane_ctx`, no decode-capture in
    /// progress, no f16 KV, no exotic attention knobs, journal enabled, no
    /// uncommitted previous chunk. Returns `(n, lane_ctx, k, p)`.
    fn lanes_verify_precheck(
        &self,
        tokens: &[Vec<u32>],
        base_positions: &[usize],
        op: &str,
    ) -> Result<(usize, usize, usize, usize), String> {
        let (n, lane_ctx) = {
            let Some(ls) = self.lanes.as_ref() else {
                return Err(format!("{op}: call enable_lanes() first"));
            };
            (ls.n, ls.lane_ctx)
        };
        if tokens.len() != n || base_positions.len() != n {
            return Err(format!(
                "{op}: tokens/positions must be len {n} (got {} / {})",
                tokens.len(),
                base_positions.len()
            ));
        }
        let k = tokens[0].len();
        if k == 0 || n * k > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "{op}: lanes*k = {n}*{k} must be in 1..={QWEN38_VERIFY_MAX_P}"
            ));
        }
        if tokens.iter().any(|t| t.len() != k) {
            return Err(format!("{op}: uniform k required (the row budget is packed)"));
        }
        for (l, &bp) in base_positions.iter().enumerate() {
            if bp + k > lane_ctx {
                return Err(format!(
                    "{op}: lane {l} pos {bp} + k {k} exceeds lane_ctx {lane_ctx}"
                ));
            }
        }
        if self.capturing {
            return Err(format!("{op}: eager-only (no graph capture)"));
        }
        if self.kv_f16 {
            return Err(format!(
                "{op}: the f16 KV hatch is unsupported in lane mode"
            ));
        }
        if self.verify_attn_mma || self.verify_attn_2p || self.verify_attn_dotma {
            return Err(format!(
                "{op}: exotic attention knobs (mma/2p/dotma) unsupported in lane mode"
            ));
        }
        if self.gdn_journal.is_none() {
            return Err(format!("{op}: call enable_gdn_journal() first"));
        }
        if self.lanes_commit_ready.is_some() {
            return Err(format!(
                "{op}: the previous chunk is uncommitted — call lanes_commit() first"
            ));
        }
        Ok((n, lane_ctx, k, n * k))
    }

    /// Stage 3 — the rows-arm-only capture guard. `verify_use_qg` resolves
    /// the qg arm when `(bp + k) · kvd · 8B ≥ 96 MB`; `bp + k` is bounded by
    /// `lane_ctx`, so a lane set under the threshold NEVER resolves qg and
    /// the capture can pin the rows arm for the set's whole lifetime. A set
    /// over the threshold (or lanes off) is a permanent eager fallback for
    /// the graphed entry.
    fn lanes_graphs_supported(&self) -> bool {
        let Some(ls) = self.lanes.as_ref() else {
            return false;
        };
        let kvd = self.cfg.n_kv_head * self.cfg.head_dim;
        ls.lane_ctx * kvd * 8 < 96 * 1024 * 1024
    }

    /// Stage 3 harness/diagnostic accessor: how many lane verify graphs are
    /// captured for the live lane set (`None` when lanes are off). The G1
    /// gate pins this at exactly ONE across a full mixed-acceptance run —
    /// replay-only afterwards, with per-lane positions advancing every
    /// cycle (the anti-vacuity proof that positions ride `pos_dev`, not a
    /// baked scalar, and that no cycle silently fell back to eager).
    pub fn lanes_graph_captured(&self) -> Option<usize> {
        self.lanes.as_ref().map(|ls| ls.graphs.len())
    }

    /// Plan 556 Stage 3 (mechanism F) — the GRAPHED lane verify chunk: the
    /// Stage-2 launch sequence captured ONCE per `(n, k)` shape and replayed
    /// as ONE graph launch per chunk (the ~2.4k-launch WDDM submit overhead
    /// collapses — the solo T9.12 result at lane shapes). Per-lane positions
    /// ride the lane set's `pos_dev` `[n] i32` buffer (uploaded per cycle,
    /// read at kernel runtime by the `_devpos` twins) so ONE capture replays
    /// at EVERY per-lane position mix; tokens ride `verify.tokens_dev`,
    /// uploaded before each launch.
    ///
    /// Rows-arm-only by construction: [`Self::lanes_graphs_supported`] (the
    /// `verify_use_qg` threshold bound) makes the qg arm unreachable
    /// naturally, and a per-cycle `verify_use_qg` re-check (which only fires
    /// under the `QWEN38_VERIFY_ATTN_QG` force env) forwards any
    /// qg-resolving cycle to the eager chunk — so the capture's pinned rows
    /// arm is ALWAYS the arm the cycle would have resolved eagerly.
    ///
    /// The GDN snapshot dtods stay OUTSIDE the graph (the capture contract
    /// is kernels-only): every cycle runs snapshot (eager dtod) → uploads →
    /// `graph.launch` → sync → argmax dtoh. [`Self::lanes_commit`] semantics
    /// are unchanged — the graph replaces ONLY the chunk's launch sequence.
    ///
    /// Graphs live INSIDE the lane set (they bake arena pointers): a
    /// re-`enable_lanes` or [`Self::disable_lanes`] drops them together with
    /// the pointers they bake — replay against freed pointers is
    /// structurally impossible. Capture failure marks the key failed →
    /// permanent eager for that `(n, k)` on this lane set.
    #[allow(clippy::too_many_lines)]
    pub fn forward_lanes_verify_graph(
        &mut self,
        tokens: &[Vec<u32>],
        base_positions: &[usize],
    ) -> Result<Vec<Vec<u32>>, String> {
        let (n, _lane_ctx, k, p) =
            self.lanes_verify_precheck(tokens, base_positions, "forward_lanes_verify_graph")?;
        // The rows-arm-only capture guard (see the doc above).
        if !self.lanes_graphs_supported() {
            return self.forward_lanes_verify(tokens, base_positions);
        }
        for &bp in base_positions.iter() {
            if self.verify_use_qg(bp, k) {
                return self.forward_lanes_verify(tokens, base_positions);
            }
        }
        let key = (n, k);
        if self.lanes.as_ref().unwrap().graph_failed.contains(&key) {
            return self.forward_lanes_verify(tokens, base_positions);
        }
        // ── G: the rollback medium predates every mutation (every cycle —
        // capture executes nothing; replay mutates) ──
        self.lanes_snapshot_gdn_inner()?;
        self.lanes_commit_ready = None;
        if !self.lanes.as_ref().unwrap().graphs.contains_key(&key) {
            let t0 = std::time::Instant::now();
            {
                let stream = &self.stream;
                // SAFETY: single-threaded and single-stream; the capture arm
                // performs no cross-stream ops (the T5 decode-graph
                // contract).
                unsafe { stream.context().disable_event_tracking(); }
                stream
                    .begin_capture(
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                    .map_err(|e| format!("lanes verify begin_capture: {e}"))?;
            }
            let run = self.lanes_verify_capture_body(p, k);
            let end = {
                let stream = &self.stream;
                stream.end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
            };
            match (run, end) {
                (Ok(()), Ok(Some(graph))) => {
                    graph
                        .upload()
                        .map_err(|e| format!("lanes verify graph upload: {e}"))?;
                    self.stream.synchronize().map_err(|e| e.to_string())?;
                    self.lanes.as_mut().unwrap().graphs.insert(key, graph);
                    eprintln!(
                        "[556-lg] lanes verify graph captured (n={n}, k={k}) in {:.1} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                (r, e) => {
                    let run_err = r.err().map(|e| e.to_string()).unwrap_or_default();
                    let end_err = e.err().map(|e| e.to_string()).unwrap_or_default();
                    eprintln!(
                        "[556-lg] lanes verify graph capture FAILED (n={n}, k={k}): run={run_err} end={end_err} - eager fallback"
                    );
                    self.lanes.as_mut().unwrap().graph_failed.insert(key);
                    return self.forward_lanes_verify(tokens, base_positions);
                }
            }
        }
        // Replay: eager snapshot (taken above), uploads, ONE graph launch.
        {
            let stream = &self.stream;
            // G4 (the chunk's rule): stack staging — p ≤ QWEN38_VERIFY_MAX_P.
            let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
            for (i, t) in tokens.iter().flatten().enumerate() {
                toks[i] = *t as i32;
            }
            let mut poss = [0i32; QWEN38_VERIFY_MAX_P];
            for (l, &bp) in base_positions.iter().enumerate() {
                poss[l] = bp as i32;
            }
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
            {
                let ls = self.lanes.as_mut().unwrap();
                stream
                    .memcpy_htod(&poss[..n], &mut ls.pos_dev)
                    .map_err(|e| e.to_string())?;
            }
            let graph = self
                .lanes
                .as_ref()
                .unwrap()
                .graphs
                .get(&key)
                .expect("graph captured above");
            graph
                .launch()
                .map_err(|e| format!("lanes verify graph launch: {e}"))?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let packed = stream
                .clone_dtoh(&self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            let out: Vec<u32> = packed[..p].iter().map(|&pk| !(pk as u32)).collect();
            self.lanes_commit_ready = Some(k);
            Ok((0..n)
                .map(|l| out[l * k..(l + 1) * k].to_vec())
                .collect())
        }
    }

    /// Stage 3 — the lane verify chunk's launch sequence for graph capture:
    /// [`Self::forward_lanes_verify`]'s kernel sequence VERBATIM except
    /// (a) the pos-dependent per-lane launches use the `_devpos` twins with
    /// `pos_dev.slice(l..l+1)` (`LaneVerifyPos::Dev` — a scalar position
    /// would bake into the graph), (b) the scores arm is PINNED to rows at
    /// the fixed max `attn_n_chunks` grid (the capture guard makes qg
    /// unreachable — dead chunks write neutral partials, the T9.6
    /// contract), and (c) the argmax zero rides a capturable device memset.
    /// No host memcpys, no syncs, no allocs inside (the capture SAFETY
    /// contract). The GDN layer fn is position-free and shared verbatim
    /// (its journal copies ride the `copy_f32` kernel — captured).
    #[allow(clippy::too_many_lines)]
    fn lanes_verify_capture_body(&mut self, p: usize, k: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd;
        // Embedding rows (the token VALUES ride tokens_dev — device-side
        // already, uploaded before each launch).
        {
            let w = &self.weights.token_embd;
            let v = &self.verify;
            // SAFETY: out covers p*n; tokens covers p.
            unsafe {
                self.dense.launch_dequant_q4k_rows(
                    &self.stream,
                    &w.dev,
                    &v.xb,
                    &v.tokens_dev,
                    n_embd,
                    w.blocks_per_row,
                    p,
                )
            }?;
        }
        // Dev pins every lane to the rows arm (the capture guard — see the
        // graphed entry's doc); use_qg is resolved only on the Live path.
        let no_qg = [false; QWEN38_VERIFY_MAX_P];
        let lanes = self.lanes.as_ref().expect("lanes enabled (prechecked)");
        let pos_dev = &lanes.pos_dev;
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 — the fused boundary chain, p = n*k (row-agnostic).
            // VERBATIM forward_lanes_verify's fused arm except
            // LaneVerifyPos::Dev (the capture contract: plain kernel
            // launches only, no host work).
            self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
            self.rmsnorm_quant_x_rows(
                &self.verify.xb,
                &self.weights.layers[0].input_norm.dev,
                n_embd,
                cfg.rms_norm_eps,
                p,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.lanes_verify_gdn_layer(lanes, i, gdn_idx, p, k)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.lanes_verify_attn_layer(
                            lanes,
                            i,
                            attn_idx,
                            LaneVerifyPos::Dev(pos_dev),
                            &no_qg,
                            p,
                            k,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x_rows(
                    &self.verify.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                    p,
                )?;
                self.verify_mlp(i, p)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x_rows(
                        &self.verify.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                        p,
                    )?;
                } else {
                    self.ew
                        .launch_residual_add(
                            &self.stream,
                            &self.verify.xb_res,
                            &self.verify.y,
                            &self.verify.xb,
                            n_embd * p,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.lanes_verify_gdn_layer(lanes, i, gdn_idx, p, k)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.lanes_verify_attn_layer(
                            lanes,
                            i,
                            attn_idx,
                            LaneVerifyPos::Dev(pos_dev),
                            &no_qg,
                            p,
                            k,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n_embd * p)?;
                self.verify_mlp(i, p)?;
            }
        }
        // Final norm + lm_head (verbatim the chunk tail).
        {
            let v = &self.verify;
            self.rmsnorm_quant_x_rows(
                &v.xb,
                &self.weights.output_norm.dev,
                n_embd,
                cfg.rms_norm_eps,
                p,
            )?;
            self.gemv_quant_rows(&self.weights.lm_head, &v.logits, p)?;
        }
        // Per-row argmax (zero via the capturable device memset — the eager
        // path's host memcpy would bake into the graph).
        {
            let stream = &self.stream;
            stream
                .memset_zeros(&mut self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            let v = &self.verify;
            // SAFETY: logits covers p*vocab; argmax_res covers p (zeroed).
            unsafe {
                self.dense
                    .launch_argmax_rows(stream, &v.logits, cfg.vocab_size, p, &v.argmax_res)
            }?;
        }
        Ok(())
    }

    /// The lane VERIFY GDN layer: packed row-agnostic ops at p = n*k rows
    /// (GEMVs, beta/decay, expand, zgate, ssm_out — ONE weight pass), the
    /// per-lane stateful ops (conv, recurrence) at k rows per lane on slice
    /// views of the lane arenas. Journal capture mirrors the solo layer:
    /// capture 1 (pre-conv qkv + beta/decay) covers ALL p rows BEFORE any
    /// per-lane conv overwrites them; capture 2 (post-expand) after the
    /// packed expand.
    #[allow(clippy::too_many_lines)]
    fn lanes_verify_gdn_layer(
        &self,
        lanes: &Qwen38LaneSet,
        layer_idx: usize,
        gdn_idx: usize,
        p: usize,
        k: usize,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let n_v = cfg.n_v_heads;
        let l_exp = 3 * n_v * cfg.head_k_dim;
        let v = &self.verify;
        let n_lanes = lanes.n;
        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.input_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        self.gemv_quant_rows(lw.qkv.as_ref().unwrap(), &v.qkv, p)?;
        self.gemv_quant_rows(lw.z.as_ref().unwrap(), &v.z, p)?;
        self.gemv_quant_rows(lw.alpha.as_ref().unwrap(), &v.a_raw, p)?;
        self.gemv_quant_rows(lw.beta.as_ref().unwrap(), &v.b_raw, p)?;
        self.dn
            .launch_beta_decay_rows(
                stream,
                &v.a_raw,
                &v.b_raw,
                &lw.a_log.as_ref().unwrap().dev,
                &lw.dt_bias.as_ref().unwrap().dev,
                &v.beta,
                &v.decay,
                n_v,
                p,
            )
            .map_err(|e| e.to_string())?;
        // Journal capture 1: ALL p rows BEFORE any per-lane conv.
        if let Some(jr) = self.gdn_journal.as_ref() {
            let conv_w = lw.conv1d.as_ref().unwrap();
            let cd = conv_w.len / cfg.conv_kernel;
            self.copy_f32(&v.qkv, &jr.qkv[gdn_idx], p * cd)?;
            self.copy_f32(&v.beta, &jr.beta[gdn_idx], p * n_v)?;
            self.copy_f32(&v.decay, &jr.decay[gdn_idx], p * n_v)?;
        }
        // Per-lane conv at k rows (in place over the lane's packed rows).
        let conv_w = lw.conv1d.as_ref().unwrap();
        let cd = conv_w.len / cfg.conv_kernel;
        for l in 0..n_lanes {
            let ioff = l * k * cd;
            let input = v.qkv.slice(ioff..ioff + k * cd);
            let soff = l * cd * cfg.conv_kernel;
            let state = lanes.conv[gdn_idx].slice(soff..soff + cd * cfg.conv_kernel);
            self.dn
                .launch_conv1d_rows(stream, &input, &conv_w.dev, &state, cd, cfg.conv_kernel, k)
                .map_err(|e| e.to_string())?;
        }
        // Packed expand (row-agnostic over the conv outputs).
        self.dn
            .launch_expand_l2_rows(
                stream,
                &v.qkv,
                &v.qkv_exp,
                cfg.n_k_heads,
                n_v,
                cfg.head_k_dim,
                p,
            )
            .map_err(|e| e.to_string())?;
        // Journal capture 2: post-expand recurrence inputs, all p rows.
        if let Some(jr) = self.gdn_journal.as_ref() {
            self.copy_f32(&v.qkv_exp, &jr.exp[gdn_idx], p * l_exp)?;
        }
        // Per-lane recurrence at k rows (the state slice IS the lane state).
        let rec_len = n_v * cfg.head_k_dim * cfg.head_v_dim;
        for l in 0..n_lanes {
            let eoff = l * k * l_exp;
            let qkv_in = v.qkv_exp.slice(eoff..eoff + k * l_exp);
            let boff = l * k * n_v;
            let beta = v.beta.slice(boff..boff + k * n_v);
            let decay = v.decay.slice(boff..boff + k * n_v);
            let soff = l * rec_len;
            let state = lanes.recurrent[gdn_idx].slice(soff..soff + rec_len);
            let ooff = l * k * cfg.d_inner;
            let out = v.rec_out.slice(ooff..ooff + k * cfg.d_inner);
            self.dn
                .launch_recurrence_fused_rows_hd128(
                    stream, &qkv_in, &beta, &decay, &state, &out, n_v, k,
                )
                .map_err(|e| e.to_string())?;
        }
        // SAFETY: rec_out/z cover p*d_inner; xq/xs/xsum cover p*(d_inner/16).
        unsafe {
            self.dense.launch_rmsnorm_zgate_quant_x_q8(
                stream,
                &v.rec_out,
                &v.z,
                &lw.ssm_norm.as_ref().unwrap().dev,
                &v.xq,
                &v.xs,
                &v.xsum,
                p * n_v,
                cfg.head_v_dim,
                cfg.rms_norm_eps,
            )
        }?;
        self.gemv_quant_rows(lw.ssm_out.as_ref().unwrap(), &v.y, p)?;
        if !self.res_nq_fused {
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The lane VERIFY attention layer: packed projections/norms at
    /// p = n*k rows, then PER-LANE rope → KV-append → split-GQA scores at
    /// k rows (each lane's position run and KV region are its own; the
    /// solo chunk's Live-arm semantics verbatim per lane — the non-qg arm
    /// pins the FIXED `attn_n_chunks` count per the Bench-759 lesson, the
    /// qg arm's live count matches the solo Live arm).
    ///
    /// Stage 3: `pos` selects the position source — `Live` is the eager
    /// Stage-2 sequence (scalar `bp` per lane launch); `Dev` is the captured-
    /// graph sequence (`pos_dev.slice(l..l+1)` per lane launch via the
    /// `_devpos` twins, scores pinned to the rows arm at the fixed max grid).
    /// The launch ORDER is identical in both modes (the capture-body
    /// verbatim contract); only the position SOURCE and the scores arm
    /// differ.
    #[allow(clippy::too_many_lines)]
    fn lanes_verify_attn_layer(
        &self,
        lanes: &Qwen38LaneSet,
        layer_idx: usize,
        attn_idx: usize,
        pos: LaneVerifyPos<'_>,
        use_qg: &[bool; QWEN38_VERIFY_MAX_P],
        p: usize,
        k: usize,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let lane_kv = lanes.lane_ctx * kvd;
        let v = &self.verify;
        let n_lanes = lanes.n;
        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.input_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        self.gemv_quant_rows(lw.wq.as_ref().unwrap(), &v.qg, p)?;
        self.attn
            .launch_split_qg_rows(stream, &v.qg, &v.q, &v.gate, cfg.head_dim, cfg.n_head, p)
            .map_err(|e| e.to_string())?;
        self.gemv_quant_rows(lw.wk.as_ref().unwrap(), &v.k, p)?;
        self.gemv_quant_rows(lw.wv.as_ref().unwrap(), &v.vv, p)?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &v.q,
                &lw.q_norm.as_ref().unwrap().dev,
                &v.q_normed,
                p * cfg.n_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &v.k,
                &lw.k_norm.as_ref().unwrap().dev,
                &v.k_normed,
                p * cfg.n_kv_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        // Per-lane rope FIRST (the cache must receive the ROPED k), then
        // per-lane KV append into the lane's cache region, then per-lane
        // scores — the solo chunk's site order per lane.
        match pos {
            LaneVerifyPos::Live(base_positions) => {
                for (l, &bp) in base_positions.iter().enumerate().take(n_lanes) {
                    let qoff = l * k * q_dim;
                    let qrow = v.q_normed.slice(qoff..qoff + k * q_dim);
                    let koff = l * k * kvd;
                    let krow = v.k_normed.slice(koff..koff + k * kvd);
                    self.attn
                        .launch_rope_rows(
                            stream,
                            &qrow,
                            &krow,
                            cfg.rotary_dim,
                            cfg.head_dim,
                            cfg.n_head,
                            cfg.n_kv_head,
                            bp,
                            k,
                            cfg.rope_theta,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
            LaneVerifyPos::Dev(pos_dev) => {
                for l in 0..n_lanes {
                    let qoff = l * k * q_dim;
                    let qrow = v.q_normed.slice(qoff..qoff + k * q_dim);
                    let koff = l * k * kvd;
                    let krow = v.k_normed.slice(koff..koff + k * kvd);
                    self.attn
                        .launch_rope_rows_devpos(
                            stream,
                            &qrow,
                            &krow,
                            cfg.rotary_dim,
                            cfg.head_dim,
                            cfg.n_head,
                            cfg.n_kv_head,
                            &pos_dev.slice(l..=l),
                            k,
                            cfg.rope_theta,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        match pos {
            LaneVerifyPos::Live(base_positions) => {
                for (l, &bp) in base_positions.iter().enumerate().take(n_lanes) {
                    let koff = l * k * kvd;
                    let krow = v.k_normed.slice(koff..koff + k * kvd);
                    let voff = l * k * kvd;
                    let vrow = v.vv.slice(voff..voff + k * kvd);
                    let coff = l * lane_kv;
                    let kc = lanes.keys[attn_idx].slice(coff..coff + lane_kv);
                    let vc = lanes.values[attn_idx].slice(coff..coff + lane_kv);
                    self.attn
                        .launch_kv_append_rows(stream, &krow, &vrow, &kc, &vc, kvd, bp, k)
                        .map_err(|e| e.to_string())?;
                }
            }
            LaneVerifyPos::Dev(pos_dev) => {
                for l in 0..n_lanes {
                    let koff = l * k * kvd;
                    let krow = v.k_normed.slice(koff..koff + k * kvd);
                    let voff = l * k * kvd;
                    let vrow = v.vv.slice(voff..voff + k * kvd);
                    let coff = l * lane_kv;
                    let kc = lanes.keys[attn_idx].slice(coff..coff + lane_kv);
                    let vc = lanes.values[attn_idx].slice(coff..coff + lane_kv);
                    self.attn
                        .launch_kv_append_rows_devpos(
                            stream,
                            &krow,
                            &vrow,
                            &kc,
                            &vc,
                            kvd,
                            &pos_dev.slice(l..=l),
                            k,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        // The scratch partials are sized [MAX_P][n_head][verify_attn_n_chunks]
        // (the max column count); lane l's band is its k row slots. Partials
        // are written AND consumed within each launch, so the band only needs
        // to cover k rows at the kernel's own (smaller-or-equal) stride.
        let pstride = cfg.n_head * self.verify_attn_n_chunks;
        let postride = pstride * cfg.head_dim;
        match pos {
            LaneVerifyPos::Live(base_positions) => {
                for (l, &bp) in base_positions.iter().enumerate().take(n_lanes) {
                    let coff = l * lane_kv;
                    let kc = lanes.keys[attn_idx].slice(coff..coff + lane_kv);
                    let vc = lanes.values[attn_idx].slice(coff..coff + lane_kv);
                    let qoff = l * k * q_dim;
                    let qrow = v.q_normed.slice(qoff..qoff + k * q_dim);
                    let pm = v.part_m.slice(l * k * pstride..(l + 1) * k * pstride);
                    let pl = v.part_l.slice(l * k * pstride..(l + 1) * k * pstride);
                    let po = v.part_out.slice(l * k * postride..(l + 1) * k * postride);
                    let aoff = l * k * q_dim;
                    let arow = v.attn_out.slice(aoff..aoff + k * q_dim);
                    let attn_res = if use_qg[l] {
                        let n_chunks = (bp + k).div_ceil(self.verify_attn_chunk_len).max(1);
                        self.attn.launch_attention_splitgqa_rows_qg(
                            stream, &qrow, &kc, &vc, &pm, &pl, &po, &arow, cfg.head_dim,
                            cfg.n_head, cfg.n_kv_head, self.verify_attn_chunk_len, n_chunks,
                            bp, k,
                        )
                    } else {
                        self.attn.launch_attention_splitgqa_rows(
                            stream, &qrow, &kc, &vc, &pm, &pl, &po, &arow, cfg.head_dim,
                            cfg.n_head, cfg.n_kv_head, self.attn_chunk_len, self.attn_n_chunks,
                            bp, k,
                        )
                    };
                    attn_res.map_err(|e| e.to_string())?;
                }
            }
            LaneVerifyPos::Dev(pos_dev) => {
                // The capture arm: rows arm PINNED at the fixed max grid
                // (dead chunks write neutral partials — the T9.6 contract).
                // The capture guard (lane_ctx·kvd·8B < 96 MB) makes the qg
                // arm unreachable for this lane set, so the pinned arm IS
                // the arm every replay resolves.
                for l in 0..n_lanes {
                    let coff = l * lane_kv;
                    let kc = lanes.keys[attn_idx].slice(coff..coff + lane_kv);
                    let vc = lanes.values[attn_idx].slice(coff..coff + lane_kv);
                    let qoff = l * k * q_dim;
                    let qrow = v.q_normed.slice(qoff..qoff + k * q_dim);
                    let pm = v.part_m.slice(l * k * pstride..(l + 1) * k * pstride);
                    let pl = v.part_l.slice(l * k * pstride..(l + 1) * k * pstride);
                    let po = v.part_out.slice(l * k * postride..(l + 1) * k * postride);
                    let aoff = l * k * q_dim;
                    let arow = v.attn_out.slice(aoff..aoff + k * q_dim);
                    self.attn
                        .launch_attention_splitgqa_rows_devpos(
                            stream, &qrow, &kc, &vc, &pm, &pl, &po, &arow, cfg.head_dim,
                            cfg.n_head, cfg.n_kv_head, self.attn_chunk_len, self.attn_n_chunks,
                            &pos_dev.slice(l..=l), k,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        self.attn
            .launch_output_gate(stream, &v.attn_out, &v.gate, q_dim * p)
            .map_err(|e| e.to_string())?;
        // SAFETY: flat buffers, q_dim*p multiple of 16.
        unsafe {
            self.dense
                .launch_quant_x_q8(stream, &v.attn_out, &v.xq, &v.xs, &v.xsum, q_dim * p)
        }?;
        self.gemv_quant_rows(lw.wo.as_ref().unwrap(), &v.y, p)?;
        if !self.res_nq_fused {
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Stage 2 — the fail-closed commit: lane `l` accepts its first
    /// `accepted[l]` rows (greedy compare, the caller computes it from the
    /// chunk's argmaxes). Semantics per lane:
    /// - `j == k` (full accept): NOTHING runs — the chunk already left the
    ///   GDN state exactly at base+k and every KV row is valid.
    /// - `0 < j < k`: the lane's GDN state is rolled back to the chunk's
    ///   base snapshot, then advanced by `j` journaled rows via the replay
    ///   kernels — NO weight-reading stack (mechanism B). KV rows beyond j
    ///   stay garbage-but-append-only (the Issue 746 immunity — the next
    ///   chunk's write-before-read repairs them).
    /// - `j == 0`: rollback only.
    ///
    /// Fail-closed (H): every check runs before the first mutation; the
    /// commit flag is consumed UP FRONT so a mid-commit error cannot be
    /// re-committed on torn state. When every lane fully accepted, the
    /// commit is FREE (the dominant case in the high-acceptance regime —
    /// the G2 measurement carries it honestly).
    pub fn lanes_commit(&mut self, accepted: &[usize]) -> Result<(), String> {
        // ── H: validate everything, consume the flag, THEN mutate ──
        let Some(k) = self.lanes_commit_ready else {
            return Err("lanes_commit: no verify chunk is pending".into());
        };
        let Some(ls) = self.lanes.as_ref() else {
            return Err("lanes_commit: lanes not enabled".into());
        };
        let n = ls.n;
        if accepted.len() != n {
            return Err(format!(
                "lanes_commit: accepted must be len {n} (got {})",
                accepted.len()
            ));
        }
        if accepted.iter().any(|&j| j > k) {
            return Err(format!(
                "lanes_commit: accepted count exceeds the chunk width k={k}"
            ));
        }
        if ls.snap_recurrent.is_none() || ls.snap_conv.is_none() {
            return Err("lanes_commit: snapshots not enabled".into());
        }
        if self.gdn_journal.is_none() {
            return Err("lanes_commit: journal not enabled".into());
        }
        self.lanes_commit_ready = None;
        let need: Vec<usize> = (0..n).filter(|&l| accepted[l] < k).collect();
        if need.is_empty() {
            // Every lane fully accepted: the chunk's own state evolution IS
            // the commit. Zero extra work.
            return Ok(());
        }
        let cfg = &self.cfg;
        let rec_len = cfg.n_v_heads * cfg.head_k_dim * cfg.head_v_dim;
        let l_qkv_out =
            2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
        let conv_len = l_qkv_out * cfg.conv_kernel;
        // ── rollback the lanes that need it (G) ──
        {
            let ls = self.lanes.as_mut().unwrap();
            let snap_r = ls.snap_recurrent.as_ref().unwrap();
            let snap_c = ls.snap_conv.as_ref().unwrap();
            if need.len() == n {
                // Whole-arena restore: 2 dtods per GDN layer instead of 2n.
                for (snap, live) in snap_r.iter().zip(ls.recurrent.iter_mut()) {
                    self.stream
                        .memcpy_dtod(snap, live)
                        .map_err(|e| format!("lane rollback rec dtod: {e}"))?;
                }
                for (snap, live) in snap_c.iter().zip(ls.conv.iter_mut()) {
                    self.stream
                        .memcpy_dtod(snap, live)
                        .map_err(|e| format!("lane rollback conv dtod: {e}"))?;
                }
            } else {
                for &l in &need {
                    let rsoff = l * rec_len;
                    let csoff = l * conv_len;
                    for (snap, live) in snap_r.iter().zip(ls.recurrent.iter_mut()) {
                        let src = snap.slice(rsoff..rsoff + rec_len);
                        let mut dst = live.slice_mut(rsoff..rsoff + rec_len);
                        self.stream
                            .memcpy_dtod(&src, &mut dst)
                            .map_err(|e| format!("lane rollback rec dtod: {e}"))?;
                    }
                    for (snap, live) in snap_c.iter().zip(ls.conv.iter_mut()) {
                        let src = snap.slice(csoff..csoff + conv_len);
                        let mut dst = live.slice_mut(csoff..csoff + conv_len);
                        self.stream
                            .memcpy_dtod(&src, &mut dst)
                            .map_err(|e| format!("lane rollback conv dtod: {e}"))?;
                    }
                }
            }
        }
        // ── advance the partially-accepted lanes by j journaled rows (B) ──
        // The staging copies land the lane's journaled rows [l*k .. l*k+j]
        // into the verify scratch rows [0..j] (the same staging the solo
        // advance uses — the scratch is free post-chunk), then the SAME
        // conv1d_rows + recurrence_fused_rows_hd128 kernels re-evolve the
        // lane's state by j tokens with NO weight-reading stack.
        let jr = self.gdn_journal.as_ref().unwrap();
        let n_v = cfg.n_v_heads;
        let l_exp = 3 * n_v * cfg.head_k_dim;
        for &l in &need {
            let j = accepted[l];
            if j == 0 {
                continue;
            }
            let mut gdn_idx = 0usize;
            for layer_idx in 0..cfg.n_layer {
                if !matches!(cfg.layer_types[layer_idx], Qwen38LayerType::Deltanet) {
                    continue;
                }
                let conv_w = self.weights.layers[layer_idx]
                    .conv1d
                    .as_ref()
                    .expect("gdn layer conv1d weight");
                let cd = conv_w.len / cfg.conv_kernel;
                {
                    let stream = &self.stream;
                    let cnt = j * cd;
                    let joff = l * k * cd;
                    let src = jr.qkv[gdn_idx].slice(joff..joff + cnt);
                    let mut dst = self.verify.qkv.slice_mut(0..cnt);
                    stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| format!("lane advance qkv dtod: {e}"))?;
                }
                {
                    let soff = l * cd * cfg.conv_kernel;
                    let state = self
                        .lanes
                        .as_ref()
                        .unwrap()
                        .conv[gdn_idx]
                        .slice(soff..soff + cd * cfg.conv_kernel);
                    self.dn
                        .launch_conv1d_rows(
                            &self.stream,
                            &self.verify.qkv,
                            &conv_w.dev,
                            &state,
                            cd,
                            cfg.conv_kernel,
                            j,
                        )
                        .map_err(|e| format!("lane advance conv: {e}"))?;
                }
                {
                    let stream = &self.stream;
                    let cnt = j * l_exp;
                    let joff = l * k * l_exp;
                    let src = jr.exp[gdn_idx].slice(joff..joff + cnt);
                    let mut dst = self.verify.qkv_exp.slice_mut(0..cnt);
                    stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| format!("lane advance exp dtod: {e}"))?;
                    let cnt = j * n_v;
                    let joff = l * k * n_v;
                    let src = jr.beta[gdn_idx].slice(joff..joff + cnt);
                    let mut dst = self.verify.beta.slice_mut(0..cnt);
                    stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| format!("lane advance beta dtod: {e}"))?;
                    let src = jr.decay[gdn_idx].slice(joff..joff + cnt);
                    let mut dst = self.verify.decay.slice_mut(0..cnt);
                    stream
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| format!("lane advance decay dtod: {e}"))?;
                }
                {
                    let soff = l * rec_len;
                    let state = self
                        .lanes
                        .as_ref()
                        .unwrap()
                        .recurrent[gdn_idx]
                        .slice(soff..soff + rec_len);
                    self.dn
                        .launch_recurrence_fused_rows_hd128(
                            &self.stream,
                            &self.verify.qkv_exp,
                            &self.verify.beta,
                            &self.verify.decay,
                            &state,
                            &self.verify.rec_out,
                            n_v,
                            j,
                        )
                        .map_err(|e| format!("lane advance recurrence: {e}"))?;
                }
                gdn_idx += 1;
            }
        }
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────
    // Issue 742 T4 — the single-stream prefix KV+GDN cache. The caller
    // owns position bookkeeping (the forward takes pos per call): insert
    // EXACTLY the tokens filled so far, in order; restore returns the
    // matched prefix length to continue filling from.
    // ─────────────────────────────────────────────────────────────────

    /// Snapshot the current GDN state as a prefix checkpoint for `tokens`
    /// (which must be exactly the tokens filled so far, in order). The KV
    /// rows below `tokens.len()` stay in the live buffers (the lineage
    /// rule keeps them byte-valid; captured graphs keep their addresses).
    pub fn prefix_cache_insert(&mut self, tokens: &[u32]) -> Result<(), String> {
        self.prefix_cache
            .insert(&self.stream, &self.state, tokens)
    }

    /// Longest-prefix match: on a hit, restore the GDN state (dtod,
    /// stream-ordered) and return the matched length — re-fill only
    /// `prompt[matched..]`. `Ok(None)` = no reusable prefix (full fill).
    pub fn prefix_cache_restore(&mut self, prompt: &[u32]) -> Result<Option<usize>, String> {
        self.prefix_cache
            .restore(&self.stream, &mut self.state, prompt)
    }

    /// Live prefix-cache entries (MRU-first; diagnostic).
    pub fn prefix_cache_len(&self) -> usize {
        self.prefix_cache.len()
    }

    /// Total lineage prunes (diagnostic — a prune is the KV-validity rule
    /// firing, not an error).
    pub fn prefix_cache_pruned(&self) -> usize {
        self.prefix_cache.pruned()
    }

    /// The p-row rmsnorm+quantize (rows twin of `rmsnorm_quant_x`).
    fn rmsnorm_quant_x_rows(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        n: usize,
        eps: f32,
        p: usize,
    ) -> Result<(), String> {
        let v = &self.verify;
        // Unconditional fused rows twin: it is bit-identical to the fused
        // single-row kernel, which is bit-identical to the unfused pair —
        // so the verify path matches every decode env configuration.
        // SAFETY: row-contiguous buffers cover p*n / p*(n/16); the dim%16
        // gate asserted in-launch; xq/xs/xsum sized for the max call dim
        // (mlp_hidden) which covers every site's n_embd.
        unsafe {
            self.dense.launch_rmsnorm_quant_x_q8_rows(
                &self.stream,
                x,
                gamma,
                &v.xq,
                &v.xs,
                &v.xsum,
                n,
                eps,
                p,
            )
        }
    }

    /// The p-row batched GEMV into `y` (output row-major [p, m]); the
    /// quantized input rows ride the shared verify scratch.
    ///
    /// Two arms (A/B, `QWEN38_VERIFY_GEMV`): `rows` (default) — the fused
    /// rows kernel, weights read once for all p rows; `loop` — p back-to-
    /// back launches of the PRODUCTION single-row GEMV over row-sliced
    /// views (the literal "batched wrapper over the existing kernel" arm —
    /// the p=1 kernel's proven 732 GB/s inner loop, with the p launches
    /// co-scheduled so L2 catches the weight re-reads).
    fn gemv_quant_rows(&self, w: &QuantW, y: &CudaSlice<f32>, p: usize) -> Result<(), String> {
        static VERIFY_GEMV_ARM: std::sync::OnceLock<String> = std::sync::OnceLock::new();

let v = &self.verify;
        let stream = &self.stream;
        // Issue 754 P4-b — the WIDE-INGEST arm: while the 64-row scratch is
        // swapped in, every projection site rides the T5 tolerance-class
        // GEMM (Bench 772 B1 winner, kernel gates in Bench 783). The strict
        // GEMV family below stays p <= 16-capped (Bench 753's walls 1+2);
        // this routing is unreachable on decode/verify paths (the flag is
        // set only inside `forward_ingest_chunk_wide` and never during
        // capture).
        if self.wide_ingest_active {
            // SAFETY: the wide scratch covers [64, max_dim] ⊇ [64, n] for
            // every site (the chunk entry pins p == QWEN38_INGEST_WIDE_P, so
            // all 64 rows were freshly written by this site's quantize); y
            // covers [p, m]; n % 128 == 0 and blocks_per_row == n / 256 are
            // asserted in-launch.
            unsafe {
                if w.q4 {
                    self.verify_mma.launch_t5_gemm_q4k_p64(
                        stream, &w.dev, &v.xq, &v.xs, &v.xsum, y, w.rows, w.n, w.blocks_per_row,
                        p,
                    )?;
                } else {
                    self.verify_mma.launch_t5_gemm_q6k_p64(
                        stream, &w.dev, &v.xq, &v.xs, &v.xsum, y, w.rows, w.n, w.blocks_per_row,
                        p,
                    )?;
                }
            }
            return Ok(());
        }
        let (m_i, n_i, bpr_i, p_i) = (
            w.rows as i32,
            w.n as i32,
            w.blocks_per_row as i32,
            p as i32,
        );
        let grid = w.rows.div_ceil(8).max(1) as u32;
        let arm = VERIFY_GEMV_ARM.get_or_init(|| std::env::var("QWEN38_VERIFY_GEMV").unwrap_or_default());
        if arm.as_str() == "mma" {
            // Issue 742 T9.10 — the tensor-core GEMM arm: each weight read
            // serves the full 8-feature x 8-token mma tile (the x-side L1
            // re-streaming wall of the GEMV family), bit-identical per
            // element to the strict rows kernels (the fold-order contract).
            // SAFETY: same buffers/contract as the strict arm; every
            // production n divides 1024 (Q4_K) / 512 (Q6_K) - asserted
            // in-launch.
            unsafe {
                if w.q4 {
                    self.verify_mma.launch_gemv_q4k_rows_mma(
                        stream, &w.dev, &v.xq, &v.xs, &v.xsum, y, w.rows, w.n,
                        w.blocks_per_row, p,
                    )?;
                } else {
                    self.verify_mma.launch_gemv_q6k_rows_mma(
                        stream, &w.dev, &v.xq, &v.xs, &v.xsum, y, w.rows, w.n,
                        w.blocks_per_row, p,
                    )?;
                }
            }
            return Ok(());
        }
        if arm.as_str() == "loop" {
            let groups = w.n / 16;
            for r in 0..p {
                // SAFETY: row-sliced views cover exactly n / groups / m.
                unsafe {
                    self.dense.launch_gemv_q8x_views(
                        stream,
                        &w.dev,
                        &v.xq.slice(r * w.n..(r + 1) * w.n),
                        &v.xs.slice(r * groups..(r + 1) * groups),
                        &v.xsum.slice(r * groups..(r + 1) * groups),
                        &y.slice(r * w.rows..(r + 1) * w.rows),
                        w.rows,
                        w.n,
                        w.blocks_per_row,
                        w.q4,
                    )?;
                }
            }
            return Ok(());
        }
        // SAFETY: the scratch covers p * max_dim (mlp_hidden); every GEMV
        // site's n divides 32 (Q4_K) / 16 (Q6_K); p <= 16 asserted at the
        // chunk entry. STRICT (bit-identical) is the default; shape B
        // (lane=(row,half), 8 rows/warp) measured 45-103 GB/s (broadcast
        // loads + serial row blocks — REFUTED) and stays behind
        // QWEN38_VERIFY_GEMV=shapeb as the artifact.
        if w.q4 && arm.as_str() == "shapeb" {
            return unsafe {
                self.dense.launch_gemv_q4k_rows(
                    stream, &w.dev, &v.xq, &v.xs, &v.xsum, y, w.rows, w.n, w.blocks_per_row, p,
                )
            };
        }
        if w.q4 {
            return unsafe {
                self.dense.launch_gemv_q4k_rows_strict(
                    stream, &w.dev, &v.xq, &v.xs, &v.xsum, y, w.rows, w.n, w.blocks_per_row, p,
                )
            };
        }
        unsafe {
            stream
                .launch_builder(&self.dense.q6k_q8x_rows)
                .arg(&w.dev)
                .arg(&v.xq)
                .arg(&v.xs)
                .arg(&v.xsum)
                .arg(y)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&bpr_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn verify_gdn_layer(&self, layer_idx: usize, gdn_idx: usize, p: usize) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let n_v = cfg.n_v_heads;
        let v = &self.verify;

        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.input_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        // fused path: norm done by the previous site's fused boundary
        // kernel (the loop's site B); residual folded into the next site's.
        self.gemv_quant_rows(lw.qkv.as_ref().unwrap(), &v.qkv, p)?;
        self.gemv_quant_rows(lw.z.as_ref().unwrap(), &v.z, p)?;
        self.gemv_quant_rows(lw.alpha.as_ref().unwrap(), &v.a_raw, p)?;
        self.gemv_quant_rows(lw.beta.as_ref().unwrap(), &v.b_raw, p)?;
        self.dn
            .launch_beta_decay_rows(
                stream,
                &v.a_raw,
                &v.b_raw,
                &lw.a_log.as_ref().unwrap().dev,
                &lw.dt_bias.as_ref().unwrap().dev,
                &v.beta,
                &v.decay,
                n_v,
                p,
            )
            .map_err(|e| e.to_string())?;
        // Issue 755 T4 — journal capture 1: the PRE-conv qkv rows (before the
        // in-place conv output overwrites them) + beta/decay rows. Kernel
        // copies (shared borrows — the &self receiver constraint; ~4 launches
        // ≈ 30 µs at p=8, stream-ordered under the chunk's own timing window).
        if let Some(jr) = self.gdn_journal.as_ref() {
            let conv_w = lw.conv1d.as_ref().unwrap();
            let conv_dim = conv_w.len / cfg.conv_kernel;
            self.copy_f32(&v.qkv, &jr.qkv[gdn_idx], p * conv_dim)?;
            self.copy_f32(&v.beta, &jr.beta[gdn_idx], p * n_v)?;
            self.copy_f32(&v.decay, &jr.decay[gdn_idx], p * n_v)?;
        }
        {
            let conv_w = lw.conv1d.as_ref().unwrap();
            let conv_state = &self.state.conv[gdn_idx];
            self.dn
                .launch_conv1d_rows(
                    stream,
                    &v.qkv,
                    &conv_w.dev,
                    conv_state,
                    conv_w.len / cfg.conv_kernel,
                    cfg.conv_kernel,
                    p,
                )
                .map_err(|e| e.to_string())?;
        }
        self.dn
            .launch_expand_l2_rows(
                stream,
                &v.qkv,
                &v.qkv_exp,
                cfg.n_k_heads,
                cfg.n_v_heads,
                cfg.head_k_dim,
                p,
            )
            .map_err(|e| e.to_string())?;
        // Issue 755 T4 — journal capture 2: the post-expand recurrence inputs.
        if let Some(jr) = self.gdn_journal.as_ref() {
            let l_exp = 3 * cfg.n_v_heads * cfg.head_k_dim;
            self.copy_f32(&v.qkv_exp, &jr.exp[gdn_idx], p * l_exp)?;
        }
        {
            let state = &self.state.recurrent[gdn_idx];
            self.dn
                .launch_recurrence_fused_rows_hd128(
                    stream,
                    &v.qkv_exp,
                    &v.beta,
                    &v.decay,
                    state,
                    &v.rec_out,
                    n_v,
                    p,
                )
                .map_err(|e| e.to_string())?;
        }
        // The fused zgate kernel works as-is with grid p*48 (its indexing is
        // linear in head_idx over the contiguous [p, 48*head_v_dim] rows).
        // SAFETY: rec_out/z cover p*d_inner; xq/xs/xsum cover p*(d_inner/16).
        unsafe {
            self.dense.launch_rmsnorm_zgate_quant_x_q8(
                stream,
                &v.rec_out,
                &v.z,
                &lw.ssm_norm.as_ref().unwrap().dev,
                &v.xq,
                &v.xs,
                &v.xsum,
                p * cfg.n_v_heads,
                cfg.head_v_dim,
                cfg.rms_norm_eps,
            )
        }?;
        self.gemv_quant_rows(lw.ssm_out.as_ref().unwrap(), &v.y, p)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel.
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 753 — `QWEN38_KV_DTYPE=f16` stores the KV cache as f16 halves
    /// (halved KV bytes: ~2× ctx headroom + the verify/C-axis lever measured
    /// in Bench 751). Default (and any other value) = f32, byte-identical to
    /// the pre-753 path. Resolved once per process.
    fn kv_dtype_f16() -> bool {
        static KV_F16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *KV_F16.get_or_init(|| {
            matches!(
                std::env::var("QWEN38_KV_DTYPE").as_deref(),
                Ok("f16") | Ok("half")
            )
        })
    }

    #[allow(clippy::too_many_lines)]
    /// The T9.11 adaptive q-group arm, resolved ONCE per chunk (the eager
    /// path and the graph-capture key share it - a captured graph bakes the
    /// arm, so the key must flip with it).
    fn verify_use_qg(&self, base_pos: usize, p: usize) -> bool {
        static VERIFY_ATTN_QG: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        let qg_force = *VERIFY_ATTN_QG.get_or_init(|| {
            match std::env::var("QWEN38_VERIFY_ATTN_QG").as_deref() {
                Ok("0") => Some(false),
                Ok("1") => Some(true),
                _ => None,
            }
        });
        // Issue 753 — K+V bytes per token-slot: 8 under f32, 4 under f16.
        let bytes_per_slot = if self.kv_f16 { 4 } else { 8 };
        let kv_bytes = (base_pos + p) * self.cfg.n_kv_head * self.cfg.head_dim * bytes_per_slot;
        qg_force.unwrap_or(kv_bytes >= 96 * 1024 * 1024) && self.cfg.head_dim == 256
    }

    fn verify_attn_layer(
        &self,
        layer_idx: usize,
        attn_idx: usize,
        pos: VerifyPos,
        use_qg: bool,
        p: usize,
    ) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let v = &self.verify;

        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.input_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        // fused path: norm done by the previous site's fused boundary
        // kernel; residual folded into the next site's.
        self.gemv_quant_rows(lw.wq.as_ref().unwrap(), &v.qg, p)?;
        self.attn
            .launch_split_qg_rows(stream, &v.qg, &v.q, &v.gate, cfg.head_dim, cfg.n_head, p)
            .map_err(|e| e.to_string())?;
        self.gemv_quant_rows(lw.wk.as_ref().unwrap(), &v.k, p)?;
        self.gemv_quant_rows(lw.wv.as_ref().unwrap(), &v.vv, p)?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &v.q,
                &lw.q_norm.as_ref().unwrap().dev,
                &v.q_normed,
                p * cfg.n_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        self.attn
            .launch_rmsnorm_batched(
                stream,
                &v.k,
                &lw.k_norm.as_ref().unwrap().dev,
                &v.k_normed,
                p * cfg.n_kv_head,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )
            .map_err(|e| e.to_string())?;
        match pos {
            VerifyPos::Live(bp) => {
                self.attn
                    .launch_rope_rows(
                        stream,
                        &v.q_normed,
                        &v.k_normed,
                        cfg.rotary_dim,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        bp,
                        p,
                        cfg.rope_theta,
                    )
                    .map_err(|e| e.to_string())?;
            }
            VerifyPos::Dev => {
                self.attn
                    .launch_rope_rows_devpos(
                        stream,
                        &v.q_normed,
                        &v.k_normed,
                        cfg.rotary_dim,
                        cfg.head_dim,
                        cfg.n_head,
                        cfg.n_kv_head,
                        &self.pos_dev,
                        p,
                        cfg.rope_theta,
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        {
            let kc = &self.state.keys[attn_idx];
            let vc = &self.state.values[attn_idx];
            match pos {
                VerifyPos::Live(bp) => {
                    self.attn
                        .launch_kv_append_rows(stream, &v.k_normed, &v.vv, kc, vc, kvd, bp, p)
                        .map_err(|e| e.to_string())?;
                }
                VerifyPos::Dev => {
                    self.attn
                        .launch_kv_append_rows_devpos(
                            stream, &v.k_normed, &v.vv, kc, vc, kvd, &self.pos_dev, p,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
            // Issue 742 T9.11 - the q-group arm is resolved ONCE per chunk
            // by the caller (`verify_use_qg`); a captured graph bakes it,
            // so the T9.12 graph key carries it too. `Dev` mode pins the
            // grid to the fixed max chunk count (dead chunks write neutral
            // partials - every partial slot rewritten on every launch).
            // Issue 742 T9.13 - the TILE-LEVEL SPLIT rides the qg arm only:
            // the attn-only sweep measured the win there (the long-ctx
            // regime: -19% @20K, -20% @12K at sub=2; chunk 160/128 the
            // sweet spot, sub=4 gives it back to partial/combine traffic).
            // The rows arm (short ctx) measured FLAT - its smem-limited
            // 2-blocks/SM occupancy is the binding constraint, not block
            // count - so it stays at the decode-proven `attn_chunk_len`.
            // Tolerance-class at kernel level (finer reassociation
            // boundaries, ~4e-6 max_rel measured); the model-level G1
            // (0/256 argmax + loop stream) is the gate.
            let n_chunks = match (use_qg, pos) {
                (true, VerifyPos::Live(bp)) => {
                    (bp + p).div_ceil(self.verify_attn_chunk_len).max(1)
                }
                (true, VerifyPos::Dev) => self.verify_attn_n_chunks,
                // Bench 759 G1 root-cause: the Live count MUST be the FIXED
                // decode count (`attn_n_chunks`), not a live div_ceil — decode
                // (graph-captured) bakes the fixed count, so a live count made
                // the chunk's attention partial association differ from decode
                // below the first 256-token boundary, flipping near-tie argmax
                // (first chat-lane near-tie at pos 96 → deterministic 287/384-
                // class G1 divergence; dead partials write neutral per the Dev
                // contract, so the fixed count is exact everywhere).
                (false, VerifyPos::Live(_)) => self.attn_n_chunks,
                (false, VerifyPos::Dev) => self.attn_n_chunks,
            };
            let attn_res = match (use_qg, pos) {
                // Plan 551 / Issue 773 T2 — the 3xtf32 mma score-phase arm
                // (opt-in QWEN38_VERIFY_ATTN_MMA=1; takes priority over the
                // 2p/dotma arms when resolved — same buffers/combine, the
                // devpos twin keeps the T9.12 graph contract).
                (true, VerifyPos::Live(bp)) if self.verify_attn_mma => {
                    let mma = self.attn_mma.as_ref().expect("mma knob without kernels");
                    mma.launch_splitgqa_rows_qg_mma(
                        self.verify_attn_mma_arm24,
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, n_chunks, bp, p,
                    )
                }
                (true, VerifyPos::Dev) if self.verify_attn_mma => {
                    let mma = self.attn_mma.as_ref().expect("mma knob without kernels");
                    mma.launch_splitgqa_rows_qg_mma_devpos(
                        self.verify_attn_mma_arm24,
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, self.verify_attn_n_chunks, &self.pos_dev, p,
                    )
                }
                // Issue 742 T9.14 - the two-pass arm (qg regime, default):
                // pass A tile-parallel stats -> merge -> pass B frozen-max
                // PV -> the unchanged combine. The grids/strides mirror
                // the qg arm's Live/Dev split (exact counts live; the
                // T9.12 pins devpos).
                (true, VerifyPos::Live(bp)) if self.verify_attn_2p => self
                    .attn
                    .launch_attention_verify2p(
                        stream, &v.q_normed, kc, vc,
                        &v.stat_m, &v.stat_l, &v.mrg_m, &v.mrg_l,
                        &v.part_m, &v.part_l, &v.part_out, &v.attn_out,
                        cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, n_chunks,
                        (bp + p).div_ceil(32).max(1), bp, p,
                    ),
                (true, VerifyPos::Dev) if self.verify_attn_2p => self
                    .attn
                    .launch_attention_verify2p_devpos(
                        stream, &v.q_normed, kc, vc,
                        &v.stat_m, &v.stat_l, &v.mrg_m, &v.mrg_l,
                        &v.part_m, &v.part_l, &v.part_out, &v.attn_out,
                        cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, self.verify_attn_n_chunks,
                        self.verify_stat_n_tiles, &self.pos_dev, p,
                    ),
                (true, VerifyPos::Live(bp)) if self.verify_attn_dotma => self
                    .attn
                    .launch_attention_splitgqa_rows_qgma(
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, n_chunks, bp, p,
                    ),
                (true, VerifyPos::Live(bp)) => self.attn.launch_attention_splitgqa_rows_qg(
                    stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                    &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                    self.verify_attn_chunk_len, n_chunks, bp, p,
                ),
                (true, VerifyPos::Dev) if self.verify_attn_dotma => self
                    .attn
                    .launch_attention_splitgqa_rows_qgma_devpos(
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, self.verify_attn_n_chunks, &self.pos_dev, p,
                    ),
                (true, VerifyPos::Dev) => self
                    .attn
                    .launch_attention_splitgqa_rows_qg_devpos(
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.verify_attn_chunk_len, self.verify_attn_n_chunks, &self.pos_dev, p,
                    ),
                (false, VerifyPos::Live(bp)) if self.verify_attn_dotma => self
                    .attn
                    .launch_attention_splitgqa_rows_ma(
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.attn_chunk_len, n_chunks, bp, p,
                    ),
                (false, VerifyPos::Live(bp)) => self.attn.launch_attention_splitgqa_rows(
                    stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                    &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                    self.attn_chunk_len, n_chunks, bp, p,
                ),
                (false, VerifyPos::Dev) if self.verify_attn_dotma => self
                    .attn
                    .launch_attention_splitgqa_rows_ma_devpos(
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.attn_chunk_len, self.attn_n_chunks, &self.pos_dev, p,
                    ),
                (false, VerifyPos::Dev) => self
                    .attn
                    .launch_attention_splitgqa_rows_devpos(
                        stream, &v.q_normed, kc, vc, &v.part_m, &v.part_l, &v.part_out,
                        &v.attn_out, cfg.head_dim, cfg.n_head, cfg.n_kv_head,
                        self.attn_chunk_len, self.attn_n_chunks, &self.pos_dev, p,
                    ),
            };
            attn_res.map_err(|e| e.to_string())?;
        }
        self.attn
            .launch_output_gate(stream, &v.attn_out, &v.gate, q_dim * p)
            .map_err(|e| e.to_string())?;
        // SAFETY: flat buffers, q_dim*p multiple of 16.
        unsafe {
            self.dense
                .launch_quant_x_q8(stream, &v.attn_out, &v.xq, &v.xs, &v.xsum, q_dim * p)
        }?;
        self.gemv_quant_rows(lw.wo.as_ref().unwrap(), &v.y, p)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel.
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn verify_mlp(&self, layer_idx: usize, p: usize) -> Result<(), String> {
        let stream = &self.stream;
        let cfg = &self.cfg;
        let lw = &self.weights.layers[layer_idx];
        let n = cfg.n_embd;
        let v = &self.verify;
        if !self.res_nq_fused {
            self.rmsnorm_quant_x_rows(&v.xb, &lw.post_attn_norm.dev, n, cfg.rms_norm_eps, p)?;
        }
        // fused path: norm done by the previous site's fused boundary
        // kernel (the loop's site A); residual folded into the next site's.
        self.gemv_quant_rows(&lw.ffn_gate, &v.mlp_gate, p)?;
        self.gemv_quant_rows(&lw.ffn_up, &v.mlp_up, p)?;
        self.ew
            .launch_swiglu(
                stream,
                &v.mlp_gate,
                &v.mlp_up,
                &v.mlp_hidden,
                cfg.mlp_hidden * p,
            )
            .map_err(|e| e.to_string())?;
        // SAFETY: flat buffers, mlp_hidden*p multiple of 16.
        unsafe {
            self.dense.launch_quant_x_q8(
                stream,
                &v.mlp_hidden,
                &v.xq,
                &v.xs,
                &v.xsum,
                cfg.mlp_hidden * p,
            )
        }?;
        self.gemv_quant_rows(&lw.ffn_down, &v.y, p)?;
        if !self.res_nq_fused {
            // fused path: residual folded into the next site's fused
            // boundary kernel (or the plain last-layer add).
            self.ew
                .launch_residual_add(stream, &v.xb_res, &v.y, &v.xb, n * p)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The verify chunk: feed `tokens` (p <= 16) at `base_pos`, return the
    /// greedy argmax after each position (am[i] = the model's greedy
    /// continuation after tokens[0..=i]). Mutates the KV/GDN state exactly
    /// as p sequential decode steps would (bit-identically per position);
    /// pair with [`Self::verify_snapshot_gdn`]/[`Self::verify_rollback_gdn`]
    /// for the loop's mismatch path.
    ///
    /// Issue 755 T2 — the eager body was factored so the tap variant
    /// ([`Self::forward_verify_chunk_taps`]) shares it VERBATIM (the tap
    /// copies are pure observers: they read `v.xb` at the post-layer
    /// boundary and never touch a compute buffer, so the argmax outputs are
    /// bit-identical with taps on).
    pub fn forward_verify_chunk(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<Vec<u32>, String> {
        self.verify_chunk_eager(tokens, base_pos, false)
    }

    /// Issue 755 T2 — allocate the `[16][25600]` verify-tap buffer. Call once
    /// before the first tap chunk (allocation outside any hot loop; eager
    /// path only, so no capture-region interaction). Idempotent.
    pub fn enable_verify_taps(&mut self) -> Result<(), String> {
        if self.verify_taps_dev.is_some() {
            return Ok(());
        }
        if self.cfg.n_layer <= QWEN38_DFLASH2_TAP_LAYERS[4] {
            return Err(format!(
                "enable_verify_taps: n_layer {} does not cover tap layer {}",
                self.cfg.n_layer,
                QWEN38_DFLASH2_TAP_LAYERS[4]
            ));
        }
        let len = QWEN38_VERIFY_MAX_P * QWEN38_DFLASH2_TAP_LAYERS.len() * self.cfg.n_embd;
        let buf = self
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| format!("verify taps alloc: {e}"))?;
        self.verify_taps_dev = Some(buf);
        Ok(())
    }

    /// Issue 755 T2 — the verify chunk WITH the DFlash2 feature taps: the
    /// eager chunk verbatim, plus a copy of the post-layer residual
    /// (`v.xb` = raw `xb_res + y`, the same value [`Self::forward_token_capture`]
    /// taps as `self.x`) into [`Self::verify_taps_dev`] at every
    /// [`QWEN38_DFLASH2_TAP_LAYERS`] boundary. Read the accepted prefix back
    /// with [`Self::verify_taps_download`].
    ///
    /// EAGER-only by design (the graph arm's captured sequence has no tap
    /// copies — a tap-carrying capture would need its own graph key; Issue
    /// 755 T2 runs the loop on the eager chunk and records the graph-eager
    /// delta instead). Tap rows ≥ the accepted prefix are STALE after a
    /// mismatch (the rejected rows wrote them) — the caller downloads only
    /// the accepted prefix, and the next chunk rewrites every row it uses.
    pub fn forward_verify_chunk_taps(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<Vec<u32>, String> {
        if self.verify_taps_dev.is_none() {
            return Err(
                "forward_verify_chunk_taps: call enable_verify_taps() first".to_string(),
            );
        }
        self.verify_chunk_eager(tokens, base_pos, true)
    }

    /// Issue 755 T2 — download the first `rows` tap rows (`[rows][25600]`
    /// host f32, ascending tap order — the exact `inject_positions` input).
    /// Call AFTER the chunk (and after any rollback/advance — neither
    /// touches the tap buffer; the advance's plain chunk runs taps-off).
    pub fn verify_taps_download(&self, rows: usize) -> Result<Vec<f32>, String> {
        let inp = QWEN38_DFLASH2_TAP_LAYERS.len() * self.cfg.n_embd;
        let Some(taps) = self.verify_taps_dev.as_ref() else {
            return Err("verify_taps_download: taps not enabled".to_string());
        };
        if rows == 0 || rows > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "verify_taps_download: rows must be in 1..={QWEN38_VERIFY_MAX_P} (got {rows})"
            ));
        }
        let view = taps.slice(0..rows * inp);
        let mut out = vec![0.0f32; rows * inp];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| format!("verify taps dtoh: {e}"))?;
        Ok(out)
    }

    /// Bench 759 G1 diag — dump one attention layer's KV row pair at
    /// `pos` (`[kvd] + [kvd]` concatenated). Diagnostic only (2 syncs + a
    /// small dtoh); compares chunk-written vs decode-written rows.
    pub fn dump_kv_row(
        &self,
        attn_idx: usize,
        pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let kvd = self.cfg.n_kv_head * self.cfg.head_dim;
        let k = &self.state.keys[attn_idx];
        let v = &self.state.values[attn_idx];
        if (pos + 1) * kvd > k.len() {
            return Err(format!("dump_kv_row: pos {pos} beyond cache"));
        }
        let mut ko = vec![0.0f32; kvd];
        let mut vo = vec![0.0f32; kvd];
        self.stream
            .memcpy_dtoh(&k.slice(pos * kvd..(pos + 1) * kvd), &mut ko)
            .map_err(|e| format!("dump k: {e}"))?;
        self.stream
            .memcpy_dtoh(&v.slice(pos * kvd..(pos + 1) * kvd), &mut vo)
            .map_err(|e| format!("dump v: {e}"))?;
        Ok((ko, vo))
    }

    /// Bench 759 G1 diag — dump one GDN layer's recurrent+conv state
    /// (concatenated). Diagnostic only.
    pub fn dump_gdn_state(&self, gdn_idx: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let r = &self.state.recurrent[gdn_idx];
        let c = &self.state.conv[gdn_idx];
        let mut ro = vec![0.0f32; r.len()];
        let mut co = vec![0.0f32; c.len()];
        self.stream
            .memcpy_dtoh(r, &mut ro)
            .map_err(|e| format!("dump gdn rec: {e}"))?;
        self.stream
            .memcpy_dtoh(c, &mut co)
            .map_err(|e| format!("dump gdn conv: {e}"))?;
        Ok((ro, co))
    }

    /// Copy one tap layer's post-layer residual into the taps buffer at slot
    /// `k` (row-major `[p][5*n_embd]`; 5·p strided dtod copies, 20 KB each —
    /// noise against the chunk's GEMV traffic). Static with explicitly-split
    /// borrows: the chunk body holds `&self.cfg`/`&self.verify` live across
    /// the layer loop, so a `&mut self` method would not borrow-check — the
    /// callers pass the three DISJOINT fields (`stream`, `verify.xb`,
    /// `verify_taps_dev`) directly.
    fn verify_tap_copy(
        stream: &Arc<CudaStream>,
        xb: &CudaSlice<f32>,
        taps: &mut CudaSlice<f32>,
        n_embd: usize,
        k: usize,
        p: usize,
    ) -> Result<(), String> {
        let inp = QWEN38_DFLASH2_TAP_LAYERS.len() * n_embd;
        for r in 0..p {
            let src = xb.slice(r * n_embd..(r + 1) * n_embd);
            let mut dst = taps.slice_mut(r * inp + k * n_embd..r * inp + (k + 1) * n_embd);
            stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| format!("verify tap dtod: {e}"))?;
        }
        Ok(())
    }

    /// The shared eager chunk body (`taps` gates the Issue-755 tap copies).
    #[allow(clippy::too_many_lines)]
    fn verify_chunk_eager(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        taps: bool,
    ) -> Result<Vec<u32>, String> {
        let p = tokens.len();
        if p == 0 || p > QWEN38_VERIFY_MAX_P {
            return Err(format!("verify chunk: p must be in 1..=16 (got {p})"));
        }
        if base_pos + p > self.ctx_len {
            return Err(format!(
                "verify chunk: base_pos {base_pos} + p {p} exceeds ctx_len {}",
                self.ctx_len
            ));
        }
        // T4 lineage rule: the chunk clobbers KV rows [base_pos, ..).
        self.prefix_cache.note_write(base_pos);
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        // G4 (T6): stack staging — p <= QWEN38_VERIFY_MAX_P (16), so the
        // i32 upload buffer never touches the allocator in the hot loop.
        let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
        for (i, &t) in tokens.iter().enumerate() {
            toks[i] = t as i32;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
        }
        // Embedding rows.
        {
            let w = &self.weights.token_embd;
            let v = &self.verify;
            // SAFETY: out covers p*n; tokens covers p.
            unsafe {
                self.dense.launch_dequant_q4k_rows(
                    &self.stream,
                    &w.dev,
                    &v.xb,
                    &v.tokens_dev,
                    n,
                    w.blocks_per_row,
                    p,
                )
            }?;
        }
        // T9.12: the q-group arm resolved ONCE per chunk (the captured
        // graph bakes it - the graph key carries it too).
        let use_qg = self.verify_use_qg(base_pos, p);
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 — the fused boundary chain (rows twin): prologue
            // snapshot + layer-0 input norm, then every boundary in ONE
            // kernel. The verify layer fns + verify_mlp skip their own
            // leading norm + trailing residual (the `!res_nq_fused`
            // conditionals inside them). VERBATIM the same sequence as
            // verify_capture_body's fused arm (the capture twin).
            self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
            self.rmsnorm_quant_x_rows(
                &self.verify.xb,
                &self.weights.layers[0].input_norm.dev,
                n,
                cfg.rms_norm_eps,
                p,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.verify_gdn_layer(i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.verify_attn_layer(
                            i,
                            attn_idx,
                            VerifyPos::Live(base_pos),
                            use_qg,
                            p,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x_rows(
                    &self.verify.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                    p,
                )?;
                self.verify_mlp(i, p)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x_rows(
                        &self.verify.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                        p,
                    )?;
                } else {
                    // Last layer: plain residual — the tail below owns
                    // output_norm + lm_head.
                    self.ew
                        .launch_residual_add(
                            &self.stream,
                            &self.verify.xb_res,
                            &self.verify.y,
                            &self.verify.xb,
                            n * p,
                        )
                        .map_err(|e| e.to_string())?;
                }
                if let Some(k) = dflash2_tap_slot(i).filter(|_| taps) {
                    let Some(taps_buf) = self.verify_taps_dev.as_mut() else {
                        return Err("verify tap copy: taps not enabled".to_string());
                    };
                    Self::verify_tap_copy(&self.stream, &self.verify.xb, taps_buf, n, k, p)?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.verify_gdn_layer(i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.verify_attn_layer(
                            i,
                            attn_idx,
                            VerifyPos::Live(base_pos),
                            use_qg,
                            p,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
                self.verify_mlp(i, p)?;
                // Unfused arm: `v.xb` holds the raw post-layer residual the
                // instant verify_mlp's trailing residual_add lands (the same
                // value the fused boundary materializes above).
                if let Some(k) = dflash2_tap_slot(i).filter(|_| taps) {
                    let Some(taps_buf) = self.verify_taps_dev.as_mut() else {
                        return Err("verify tap copy: taps not enabled".to_string());
                    };
                    Self::verify_tap_copy(&self.stream, &self.verify.xb, taps_buf, n, k, p)?;
                }
            }
        }
        // Final norm + lm_head + per-row argmax.
        {
            let v = &self.verify;
            self.rmsnorm_quant_x_rows(&v.xb, &self.weights.output_norm.dev, n, cfg.rms_norm_eps, p)?;
            self.gemv_quant_rows(&self.weights.lm_head, &v.logits, p)?;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(ZERO_U64_16.as_slice(), &mut self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            // SAFETY: logits covers p*vocab; argmax_res covers p (zeroed).
            unsafe {
                self.dense.launch_argmax_rows(
                    stream,
                    &self.verify.logits,
                    cfg.vocab_size,
                    p,
                    &self.verify.argmax_res,
                )
            }?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let packed = stream
                .clone_dtoh(&self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            Ok(packed[..p].iter().map(|&pk| !(pk as u32)).collect())
        }
    }
    /// Issue 742 T9.12 - the verify chunk's launch sequence for graph
    /// capture: [`Self::forward_verify_chunk`]'s kernel sequence VERBATIM
    /// except (a) the pos-dependent launches use the `_devpos` twins
    /// (`VerifyPos::Dev` - `base_pos` rides `pos_dev`), (b) the attention
    /// grid is the FIXED max [`Self::attn_n_chunks`] (dead chunks write
    /// neutral partials - bit-identical merge, the T9.6/T9.11 contract),
    /// and (c) the argmax zero rides a capturable device memset, and (d)
    /// `ingest` (Issue 754 T2) swaps the tail to lm_head + argmax for the
    /// LAST row only. No host memcpys, no syncs, no allocs inside (the
    /// capture SAFETY contract).
    fn verify_capture_body(&mut self, p: usize, use_qg: bool, ingest: bool) -> Result<(), String> {
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        // Embedding rows (the token VALUES ride `tokens_dev` - device-side
        // already, uploaded before each launch).
        {
            let w = &self.weights.token_embd;
            let v = &self.verify;
            // SAFETY: out covers p*n; tokens covers p.
            unsafe {
                self.dense.launch_dequant_q4k_rows(
                    &self.stream,
                    &w.dev,
                    &v.xb,
                    &v.tokens_dev,
                    n,
                    w.blocks_per_row,
                    p,
                )
            }?;
        }
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            // Issue 755 — the fused boundary chain (rows twin), VERBATIM
            // forward_verify_chunk's fused arm except VerifyPos::Dev (the
            // capture contract: plain kernel launches only, no host work —
            // the fused kernel is a single launch, trivially capturable).
            self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
            self.rmsnorm_quant_x_rows(
                &self.verify.xb,
                &self.weights.layers[0].input_norm.dev,
                n,
                cfg.rms_norm_eps,
                p,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.verify_gdn_layer(i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.verify_attn_layer(i, attn_idx, VerifyPos::Dev, use_qg, p)?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x_rows(
                    &self.verify.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                    p,
                )?;
                self.verify_mlp(i, p)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x_rows(
                        &self.verify.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                        p,
                    )?;
                } else {
                    // Last layer: plain residual — the tail below owns
                    // output_norm (+ the ingest lm_head arm).
                    self.ew
                        .launch_residual_add(
                            &self.stream,
                            &self.verify.xb_res,
                            &self.verify.y,
                            &self.verify.xb,
                            n * p,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.verify_gdn_layer(i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.verify_attn_layer(i, attn_idx, VerifyPos::Dev, use_qg, p)?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
                self.verify_mlp(i, p)?;
            }
        }
        // Final norm + lm_head.
        {
            let v = &self.verify;
            self.rmsnorm_quant_x_rows(
                &v.xb,
                &self.weights.output_norm.dev,
                n,
                cfg.rms_norm_eps,
                p,
            )?;
            if ingest {
                // Issue 754 T2 — the ingest tail: lm_head runs for the LAST
                // row only (every ingestion caller discards the other rows'
                // logits; at p=16 that skips 15/16 of one of the model's
                // largest matrix reads). The retained row rides the
                // production single-row GEMV over the same quantized views
                // the full tail reads — per-row bit-identity to the rows
                // kernel is the T9.9 fold-order contract (gated 0/512 +
                // continuation in bench_755).
                let w = &self.weights.lm_head;
                let groups = w.n / 16;
                let stream = &self.stream;
                // SAFETY: views cover exactly n / groups / rows elements.
                unsafe {
                    self.dense.launch_gemv_q8x_views(
                        stream,
                        &w.dev,
                        &v.xq.slice((p - 1) * w.n..p * w.n),
                        &v.xs.slice((p - 1) * groups..p * groups),
                        &v.xsum.slice((p - 1) * groups..p * groups),
                        &v.logits.slice(0..w.rows),
                        w.rows,
                        w.n,
                        w.blocks_per_row,
                        w.q4,
                    )?;
                }
            } else {
                self.gemv_quant_rows(&self.weights.lm_head, &v.logits, p)?;
            }
        }
        // Per-row argmax (zero via the capturable device memset - the
        // eager path's host memcpy would bake into the graph).
        {
            let stream = &self.stream;
            stream
                .memset_zeros(&mut self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            let v = &self.verify;
            // Ingest reads row 0 only (the retained row's logits land there).
            let rows = if ingest { 1 } else { p };
            // SAFETY: logits covers rows * vocab; argmax_res covers p (zeroed).
            unsafe {
                self.dense
                    .launch_argmax_rows(stream, &v.logits, cfg.vocab_size, rows, &v.argmax_res)
            }?;
        }
        Ok(())
    }
    /// Issue 742 T9.12 - the graphed verify chunk: capture the chunk's
    /// launch sequence ONCE per (p, use_qg) shape, then one `graph.launch()`
    /// per chunk (the ~1.8k-launch WDDM submit overhead collapses to a
    /// single launch). `base_pos` rides `pos_dev` and the attention grid is
    /// the fixed max chunk count, so ONE capture replays at EVERY context
    /// position - no per-position recapture (context grows under replay;
    /// dead chunk blocks early-exit writing neutral partials). Tokens ride
    /// `tokens_dev`, uploaded before each launch. Capture failure (or a
    /// previously failed key) falls back to the eager
    /// [`Self::forward_verify_chunk`].
    pub fn forward_verify_chunk_graph(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<Vec<u32>, String> {
        let p = tokens.len();
        if p == 0 || p > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "verify chunk graph: p must be in 1..=16 (got {p})"
            ));
        }
        if base_pos + p > self.ctx_len {
            return Err(format!(
                "verify chunk graph: base_pos {base_pos} + p {p} exceeds ctx_len {}",
                self.ctx_len
            ));
        }
        // T4 lineage rule (see forward_verify_chunk).
        self.prefix_cache.note_write(base_pos);
        let use_qg = self.verify_use_qg(base_pos, p);
        let key = (p, use_qg, false);
        if self.verify_graph_failed.contains(&key) {
            return self.forward_verify_chunk(tokens, base_pos);
        }
        if !self.verify_graphs.contains_key(&key) {
            let t0 = std::time::Instant::now();
            {
                let stream = &self.stream;
                // SAFETY: single-threaded and single-stream; the capture arm
                // performs no cross-stream ops (the T5 decode-graph
                // contract).
                unsafe { stream.context().disable_event_tracking(); }
                stream
                    .begin_capture(
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                    .map_err(|e| format!("verify begin_capture: {e}"))?;
            }
            let run = self.verify_capture_body(p, use_qg, false);
            let end = {
                let stream = &self.stream;
                stream.end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
            };
            match (run, end) {
                (Ok(()), Ok(Some(graph))) => {
                    graph
                        .upload()
                        .map_err(|e| format!("verify graph upload: {e}"))?;
                    self.stream.synchronize().map_err(|e| e.to_string())?;
                    self.verify_graphs.insert(key, graph);
                    eprintln!(
                        "[742-vg] verify graph captured (p={p}, qg={use_qg}) in {:.1} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                (r, e) => {
                    let run_err = r.err().map(|e| e.to_string()).unwrap_or_default();
                    let end_err = e.err().map(|e| e.to_string()).unwrap_or_default();
                    eprintln!(
                        "[742-vg] verify graph capture FAILED (p={p}, qg={use_qg}): run={run_err} end={end_err} - eager fallback"
                    );
                    self.verify_graph_failed.insert(key);
                    return self.forward_verify_chunk(tokens, base_pos);
                }
            }
        }
        // Replay: upload tokens + pos, launch, read back.
        {
            let stream = &self.stream;
            // G4 (T6): stack staging — same as the eager twin.
            let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
            for (i, &t) in tokens.iter().enumerate() {
                toks[i] = t as i32;
            }
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(&[base_pos as i32], &mut self.pos_dev)
                .map_err(|e| e.to_string())?;
            let graph = self.verify_graphs.get(&key).expect("graph captured above");
            graph
                .launch()
                .map_err(|e| format!("verify graph launch: {e}"))?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let packed = stream
                .clone_dtoh(&self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            Ok(packed[..p].iter().map(|&pk| !(pk as u32)).collect())
        }
    }

    /// Issue 754 T2 — the INGEST chunk (eager): [`Self::forward_verify_chunk`]'s
    /// launch sequence verbatim through the layer loop, but the lm_head GEMV +
    /// argmax run for the LAST row only — every ingestion caller (prefix-cache
    /// fill, cold-prompt prefill) discards the other rows' logits, and at p=16
    /// that skip saves 15/16 of one of the model's largest matrix reads per
    /// chunk. State-affecting launches (KV/GDN writes) are IDENTICAL to the
    /// verify chunk — only discarded logits work differs. The retained row
    /// rides the production single-row GEMV over the same quantized views the
    /// verify tail reads (per-row bit-identity: the T9.9 fold-order contract;
    /// gated 0/512 + continuation in bench_755).
    pub fn forward_ingest_chunk(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<u32, String> {
        Ok(self
            .ingest_chunk_eager_body(tokens, base_pos, false)?
            .pop()
            .expect("ingest tail produced exactly one argmax"))
    }

    /// The eager ingest chunk's launch sequence, shared verbatim by the
    /// p <= 16 strict path ([`Self::forward_ingest_chunk`]) and the P4-b
    /// wide path ([`Self::forward_ingest_chunk_wide`]). `all_argmax` swaps
    /// the T2 last-row-only tail for the diagnostic all-rows tail (strict
    /// 16-row GEMV slices + one batched argmax — the per-position greedy
    /// continuation the G1 flip-budget harness compares). The state-affecting
    /// launches are IDENTICAL under both tails.
    fn ingest_chunk_eager_body(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        all_argmax: bool,
    ) -> Result<Vec<u32>, String> {
        let p = tokens.len();
        let max_p = if self.wide_ingest_active {
            QWEN38_INGEST_WIDE_P
        } else {
            QWEN38_VERIFY_MAX_P
        };
        if p == 0 || p > max_p {
            return Err(format!(
                "ingest chunk: p must be in 1..={max_p} (got {p})"
            ));
        }
        if base_pos + p > self.ctx_len {
            return Err(format!(
                "ingest chunk: base_pos {base_pos} + p {p} exceeds ctx_len {}",
                self.ctx_len
            ));
        }
        // T4 lineage rule (see forward_verify_chunk).
        self.prefix_cache.note_write(base_pos);
        let cfg = &self.cfg;
        let n = cfg.n_embd;
        // G4 (T6): stack staging — sized for the WIDE path (the memcpy only
        // consumes ..p, and the p<=16 strict path stages the same way).
        let mut toks = [0i32; QWEN38_INGEST_WIDE_P];
        for (i, &t) in tokens.iter().enumerate() {
            toks[i] = t as i32;
        }
        {
            let stream = &self.stream;
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
        }
        // Embedding rows.
        {
            let w = &self.weights.token_embd;
            let v = &self.verify;
            // SAFETY: out covers p*n; tokens covers p.
            unsafe {
                self.dense.launch_dequant_q4k_rows(
                    &self.stream,
                    &w.dev,
                    &v.xb,
                    &v.tokens_dev,
                    n,
                    w.blocks_per_row,
                    p,
                )
            }?;
        }
        // The layer loop: IDENTICAL launches to the verify chunk (same
        // Issue-755 fused-boundary arm).
        let use_qg = self.verify_use_qg(base_pos, p);
        let mut gdn_idx = 0usize;
        let mut attn_idx = 0usize;
        if self.res_nq_fused {
            self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
            self.rmsnorm_quant_x_rows(
                &self.verify.xb,
                &self.weights.layers[0].input_norm.dev,
                n,
                cfg.rms_norm_eps,
                p,
            )?;
            for i in 0..cfg.n_layer {
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.verify_gdn_layer(i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.verify_attn_layer(
                            i,
                            attn_idx,
                            VerifyPos::Live(base_pos),
                            use_qg,
                            p,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.residual_norm_quant_x_rows(
                    &self.verify.y,
                    &self.weights.layers[i].post_attn_norm.dev,
                    p,
                )?;
                self.verify_mlp(i, p)?;
                if i + 1 < cfg.n_layer {
                    self.residual_norm_quant_x_rows(
                        &self.verify.y,
                        &self.weights.layers[i + 1].input_norm.dev,
                        p,
                    )?;
                } else {
                    // Last layer: plain residual — the ingest tail below
                    // owns output_norm + the last-row lm_head.
                    self.ew
                        .launch_residual_add(
                            &self.stream,
                            &self.verify.xb_res,
                            &self.verify.y,
                            &self.verify.xb,
                            n * p,
                        )
                        .map_err(|e| e.to_string())?;
                }
            }
        } else {
            for i in 0..cfg.n_layer {
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
                match cfg.layer_types[i] {
                    Qwen38LayerType::Deltanet => {
                        self.verify_gdn_layer(i, gdn_idx, p)?;
                        gdn_idx += 1;
                    }
                    Qwen38LayerType::Attention => {
                        self.verify_attn_layer(
                            i,
                            attn_idx,
                            VerifyPos::Live(base_pos),
                            use_qg,
                            p,
                        )?;
                        attn_idx += 1;
                    }
                }
                self.copy_f32(&self.verify.xb, &self.verify.xb_res, n * p)?;
                self.verify_mlp(i, p)?;
            }
        }
        // The ingest tail: norm all rows (cheap), then either the T2
        // last-row-only lm_head + argmax (production) or the P4-b diagnostic
        // all-rows tail (strict 16-row GEMV slices — the verify tail's
        // kernel — plus one batched argmax) for the G1 flip-budget harness.
        {
            let v = &self.verify;
            self.rmsnorm_quant_x_rows(
                &v.xb,
                &self.weights.output_norm.dev,
                n,
                cfg.rms_norm_eps,
                p,
            )?;
        }
        let w = &self.weights.lm_head;
        let groups = w.n / 16;
        let stream = &self.stream;
        let v = &self.verify;
        if all_argmax {
            // P4-b diagnostic tail: all p rows' logits + one batched argmax.
            // The lm_head rides the same wide routing (`gemv_quant_rows` ->
            // the T5 GEMM at m = vocab), then `launch_argmax_rows` covers
            // all p rows. Requires p % 16 == 0 for the argmax packing (the
            // wide path's p=64 pins that).
            assert!(p.is_multiple_of(16), "all-argmax tail requires p % 16 == 0");
            self.gemv_quant_rows(w, &v.logits, p)?;
        } else {
            // SAFETY: views cover exactly n / groups / rows elements.
            unsafe {
                self.dense.launch_gemv_q8x_views(
                    stream,
                    &w.dev,
                    &v.xq.slice((p - 1) * w.n..p * w.n),
                    &v.xs.slice((p - 1) * groups..p * groups),
                    &v.xsum.slice((p - 1) * groups..p * groups),
                    &v.logits.slice(0..w.rows),
                    w.rows,
                    w.n,
                    w.blocks_per_row,
                    w.q4,
                )?;
            }
        }
        let rows = if all_argmax { p } else { 1 };
        stream
            .memset_zeros(&mut self.verify.argmax_res)
            .map_err(|e| e.to_string())?;
        // SAFETY: logits covers rows * vocab; argmax_res covers p (zeroed).
        unsafe {
            self.dense.launch_argmax_rows(
                stream,
                &self.verify.logits,
                cfg.vocab_size,
                rows,
                &self.verify.argmax_res,
            )
        }?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let packed = stream
            .clone_dtoh(&self.verify.argmax_res)
            .map_err(|e| e.to_string())?;
        Ok(packed[..rows].iter().map(|&pk| !(pk as u32)).collect())
    }

    /// Issue 754 T2 — the graphed ingest chunk: [`Self::forward_ingest_chunk`]'s
    /// launch sequence captured once per `(p, use_qg, ingest=true)` shape (the
    /// same machinery as [`Self::forward_verify_chunk_graph`]; the tail axis
    /// keeps ingest and verify graphs distinct captures). `base_pos` rides
    /// `pos_dev` and the attention grid is the fixed max chunk count, so ONE
    /// capture replays at EVERY context position. Capture failure (or a
    /// previously failed key) falls back to the eager
    /// [`Self::forward_ingest_chunk`].
    pub fn forward_ingest_chunk_graph(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
    ) -> Result<u32, String> {
        let p = tokens.len();
        if p == 0 || p > QWEN38_VERIFY_MAX_P {
            return Err(format!(
                "ingest chunk graph: p must be in 1..=16 (got {p})"
            ));
        }
        if base_pos + p > self.ctx_len {
            return Err(format!(
                "ingest chunk graph: base_pos {base_pos} + p {p} exceeds ctx_len {}",
                self.ctx_len
            ));
        }
        // T4 lineage rule (see forward_verify_chunk).
        self.prefix_cache.note_write(base_pos);
        let use_qg = self.verify_use_qg(base_pos, p);
        let key = (p, use_qg, true);
        if self.verify_graph_failed.contains(&key) {
            return self.forward_ingest_chunk(tokens, base_pos);
        }
        if !self.verify_graphs.contains_key(&key) {
            let t0 = std::time::Instant::now();
            {
                let stream = &self.stream;
                // SAFETY: single-threaded and single-stream; the capture arm
                // performs no cross-stream ops (the T5 decode-graph
                // contract).
                unsafe { stream.context().disable_event_tracking(); }
                stream
                    .begin_capture(
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                    .map_err(|e| format!("ingest begin_capture: {e}"))?;
            }
            let run = self.verify_capture_body(p, use_qg, true);
            let end = {
                let stream = &self.stream;
                stream.end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
            };
            match (run, end) {
                (Ok(()), Ok(Some(graph))) => {
                    graph
                        .upload()
                        .map_err(|e| format!("ingest graph upload: {e}"))?;
                    self.stream.synchronize().map_err(|e| e.to_string())?;
                    self.verify_graphs.insert(key, graph);
                    eprintln!(
                        "[754-ig] ingest graph captured (p={p}, qg={use_qg}) in {:.1} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                (r, e) => {
                    let run_err = r.err().map(|e| e.to_string()).unwrap_or_default();
                    let end_err = e.err().map(|e| e.to_string()).unwrap_or_default();
                    eprintln!(
                        "[754-ig] ingest graph capture FAILED (p={p}, qg={use_qg}): run={run_err} end={end_err} - eager fallback"
                    );
                    self.verify_graph_failed.insert(key);
                    return self.forward_ingest_chunk(tokens, base_pos);
                }
            }
        }
        // Replay: upload tokens + pos, launch, read back.
        {
            let stream = &self.stream;
            // G4 (T6): stack staging — same as the eager twin.
            let mut toks = [0i32; QWEN38_VERIFY_MAX_P];
            for (i, &t) in tokens.iter().enumerate() {
                toks[i] = t as i32;
            }
            stream
                .memcpy_htod(&toks[..p], &mut self.verify.tokens_dev)
                .map_err(|e| e.to_string())?;
            stream
                .memcpy_htod(&[base_pos as i32], &mut self.pos_dev)
                .map_err(|e| e.to_string())?;
            let graph = self.verify_graphs.get(&key).expect("graph captured above");
            graph
                .launch()
                .map_err(|e| format!("ingest graph launch: {e}"))?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let packed = stream
                .clone_dtoh(&self.verify.argmax_res)
                .map_err(|e| e.to_string())?;
            Ok(!(packed[0] as u32))
        }
    }

    /// Issue 754 P4-b — the WIDE ingest chunk: [`Self::forward_ingest_chunk`]'s
    /// launch sequence at p = [`QWEN38_INGEST_WIDE_P`] exactly, with every
    /// projection GEMV site routed to the T5 tolerance-class GEMM (the Bench
    /// 772 B1 winner, kernel gates in Bench 783) and the T6-widened qg
    /// attention arm (p <= 64). The sequential GDN recurrence, norms,
    /// elementwise, and attention K/V writes are the SAME p-generic kernels
    /// the strict path runs — the only numerics delta is the T5 fold order,
    /// which is exactly what the P0 flip budgets gate (Plan 547: retained-row
    /// argmax <= 8/512, continuation <= 16/512).
    ///
    /// Mechanics: the 64-row scratch is allocated lazily on the first call
    /// (outside any capture region — the wide path is eager-only in P4-b;
    /// no new graph keys are introduced), then swapped with the p=16 scratch
    /// for the body and swapped back on every exit path. `wide_ingest_active`
    /// routes [`Self::gemv_quant_rows`] to the T5 GEMMs and is never true
    /// outside this method, so decode/verify/graph paths cannot reach it.
    ///
    /// `all_argmax = true` swaps the T2 last-row tail for the diagnostic
    /// all-rows tail (all p per-position greedy argmaxes — the G1 harness's
    /// comparison vector; production callers keep it false).
    pub fn forward_ingest_chunk_wide(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        all_argmax: bool,
    ) -> Result<Vec<u32>, String> {
        let p = tokens.len();
        if p != QWEN38_INGEST_WIDE_P {
            return Err(format!(
                "ingest wide chunk: p must be exactly {QWEN38_INGEST_WIDE_P} (got {p}) — ragged tails ride the p<=16 strict path"
            ));
        }
        if base_pos + p > self.ctx_len {
            return Err(format!(
                "ingest wide chunk: base_pos {base_pos} + p {p} exceeds ctx_len {}",
                self.ctx_len
            ));
        }
        if self.verify_attn_2p {
            return Err(
                "ingest wide chunk: QWEN38_VERIFY_ATTN_2P=1 is not widened to p>16 — run the wide path on the default qg arm".to_string(),
            );
        }
        // Lazy wide scratch (once). Same dims as the construction-time alloc;
        // the attention partials cover verify_attn_n_chunks (the qg Live
        // count at any depth: (bp + p).div_ceil(verify_attn_chunk_len) <=
        // ctx_len.div_ceil(verify_attn_chunk_len) since bp + p <= ctx_len).
        if self.wide_scratch.is_none() {
            let cfg = self.cfg.clone();
            let l_qkv_out =
                2 * cfg.n_k_heads * cfg.head_k_dim + cfg.n_v_heads * cfg.head_v_dim;
            let l_exp = 3 * cfg.n_v_heads * cfg.head_k_dim;
            let q_dim = cfg.n_head * cfg.head_dim;
            let kvd = cfg.n_kv_head * cfg.head_dim;
            self.wide_scratch = Some(alloc_verify_scratch(
                &self.stream,
                &cfg,
                l_qkv_out,
                l_exp,
                q_dim,
                kvd,
                self.verify_attn_n_chunks,
                self.verify_stat_n_tiles,
                QWEN38_INGEST_WIDE_P,
            )?);
        }
        // T4 lineage rule (same as the strict chunk).
        self.prefix_cache.note_write(base_pos);
        // Swap in, run, swap out on every path. The GDN snapshot/rollback
        // machinery MUST be used outside wide chunks (it reads whichever
        // scratch is live) — the P4-b harness snapshots before the wide
        // window and rolls back after it, both unswapped.
        std::mem::swap(
            &mut self.verify,
            self.wide_scratch.as_mut().expect("wide scratch allocated above"),
        );
        self.wide_ingest_active = true;
        let result = self.ingest_chunk_eager_body(tokens, base_pos, all_argmax);
        self.wide_ingest_active = false;
        std::mem::swap(
            &mut self.verify,
            self.wide_scratch.as_mut().expect("wide scratch allocated above"),
        );
        result
    }

    /// Issue 754 P4-c — the PRODUCTION fill loop (the "P=64 ingest production
    /// flip" the verify-C ledger's reopen condition (c) named): drains
    /// `tokens` through [`QWEN38_INGEST_WIDE_P`]-wide ingest chunks (the T5
    /// GEMMs, Bench 772 B1 winner; eager-only) with the ragged remainder on
    /// the p<=16 strict ingest path, and returns the FINAL row's greedy
    /// argmax — the decode seed a caller needs after a prompt fill. Interior
    /// chunks still pay one argmax each (the T2 finding: the strict rows
    /// kernel already amortizes the lm_head weight read across p, so the
    /// per-chunk argmax is FMAs + one read — noise next to the chunk).
    ///
    /// Measured (Bench 785): whole-fill 0→20K 326.6 tok/s = 1.29× the p=16
    /// verify-chunk fill (252.9, Bench 750), argmax flips 0/512, continuation
    /// 0/512. Callers that need the per-position argmax vector (G1 harnesses)
    /// keep the verify-chunk fill; production fill callers route here.
    ///
    /// The GDN snapshot/rollback machinery must stay OUTSIDE this loop (the
    /// wide-chunk scratch swap contract — see
    /// [`Self::forward_ingest_chunk_wide`]).
    pub fn ingest_fill(&mut self, tokens: &[u32], base_pos: usize) -> Result<u32, String> {
        let mut last = None;
        let mut c0 = 0usize;
        while c0 < tokens.len() {
            let rem = tokens.len() - c0;
            let pos = base_pos + c0;
            // Wide while a full 64 rows remain; the ragged remainder drains
            // in p<=16 strict chunks (the tail shapes the Bench 785 gates
            // exercised: 64+16+16+4-class splits, incl. the all-strict
            // short-prompt case).
            let take = if rem >= QWEN38_INGEST_WIDE_P {
                QWEN38_INGEST_WIDE_P
            } else {
                rem.min(QWEN38_VERIFY_MAX_P)
            };
            let chunk = &tokens[c0..c0 + take];
            last = Some(if take == QWEN38_INGEST_WIDE_P {
                let am = self.forward_ingest_chunk_wide(chunk, pos, false)?;
                am.into_iter().next().expect("ingest tail produced one argmax")
            } else {
                self.forward_ingest_chunk(chunk, pos)?
            });
            c0 += take;
        }
        last.ok_or_else(|| "ingest_fill: empty token slice".to_string())
    }
}
