//! Bench 903 — Issue 1127 T3 kernel probe: bit-identical geometry variants
//! for the fused `sign_fwht_quantize` decode family.
//!
//! The Bench-902 census measured the family at 8-95x over its traffic
//! roofline (`rmsnorm_sign_fwht_quantize` 5.04 µs for ~45 KB at n=5120) —
//! barrier-chain-latency-bound at the shipped geometry (dim/hblock blocks ×
//! 256 threads, ~20 `__syncthreads` per launch), NOT bandwidth-bound and NOT
//! launch-bound (the launches replay inside the captured decode graph where
//! node scheduling is ns-class).
//!
//! This probe prices the bit-identical fix space BEFORE any rung wiring (the
//! Bench-897 precedent: measure the kernel, then decide). Arms against the
//! shipped P0, each value-identical BY CONSTRUCTION:
//!
//! - **P0** — the shipped kernels, copied verbatim from
//!   `deltanet_rotation_cudarc.rs` `ROTATION_CUDA_SRC` (the control).
//! - **V1** — float4 vectorized GLOBAL loads in the fill only (the shipped
//!   strided fill is auto-vectorization-hostile; per-element math unchanged,
//!   smem writes stay scalar so no shared-mem alignment is introduced).
//! - **V2** — V1 + register-tiled butterfly stages 2 and 4 (thread-owned
//!   consecutive pairs — the SAME pairwise ops in the SAME stage order) +
//!   `shfl_down` reduction tail for K1 (levels 16..1 replicate the smem
//!   tree's exact adds within warp 0 — distance pairing with upper-lane
//!   results discarded; an xor tree would re-associate and drift bits).
//!
//! Gates executed here:
//! - **Bit-identity**: every variant's i8 + ascale outputs to_bits-equal
//!   P0's at every probed shape (the G1 pin law — a drift here kills the
//!   rung before it exists).
//! - **Interleaved burst A/B**: per-arm time = N back-to-back launches +
//!   one sync (the in-graph per-launch view the census saw); arms alternate
//!   per pair, median of per-pair ratios (two sequential arms measure the
//!   box — Issue 723's law).
//! - **A/A control** (P0 vs P0) prints the harness noise reference.
//!
//! Box-state law: the PROVENANCE line quotes nvidia-smi beside the numbers;
//! a co-resident riir/cargo/python/llama compute process REFUSES the run
//! (the identity gate is a correctness gate — the Bench-649 law). This is a
//! measurement probe, not a scored league cell.
//!
//! Run (4090 lane): `cargo test -p riir-infer-gpu --release --features
//! ternary_gemv_cuda_raw --test bench903_fwht_rotq_variant_probe -- --ignored
//! --nocapture`. Env: `B903_PAIRS` (default 24), `B903_BURST` (default 300).

#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use std::sync::Arc;
use std::time::Instant;

const SRC: &str = r#"
// ── shipped helpers (VERBATIM from ROTATION_CUDA_SRC) ───────────────────

__device__ __forceinline__ void rot_reduction_inv_rms_256(
    const float* __restrict__ x, int dim, float inv_dim, float eps,
    float* red, float* inv_rms_out)
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

// ── P0: the shipped K1/K4/K5 (VERBATIM bodies) ──────────────────────────

extern "C" __global__ void p0_rmsnorm_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
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

extern "C" __global__ void p0_gate_sigmoid_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
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

extern "C" __global__ void p0_swiglu_sign_fwht_quantize_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
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

// ── V1: float4 GLOBAL loads in the fill; scalar smem stores; the FULL
// shipped butterfly. Per-element arithmetic unchanged — the only difference
// is load width (the shipped stride-256 fill defeats auto-vectorization).
// Alignment: base = blockIdx.x * hblock with hblock % 4 == 0 and 256-byte
// device allocations ⇒ every float4 access is 16-byte aligned.

extern "C" __global__ void v1_rmsnorm_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    float inv_dim, float eps,
    int dim, int hblock)
{
    __shared__ float red[256];
    __shared__ float buf[1024];
    float inv_rms;
    rot_reduction_inv_rms_256(x, dim, inv_dim, eps, red, &inv_rms);
    const int base = blockIdx.x * hblock;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = threadIdx.x; i < hblock / 4; i += blockDim.x) {
        float4 a = x4[i];
        float4 g = g4[i];
        float4 s = s4[i];
        float v0 = (a.x * inv_rms * g.x) * s.x;
        float v1 = (a.y * inv_rms * g.y) * s.y;
        float v2 = (a.z * inv_rms * g.z) * s.z;
        float v3 = (a.w * inv_rms * g.w) * s.w;
        int b0 = i * 4;
        buf[b0] = v0; buf[b0 + 1] = v1; buf[b0 + 2] = v2; buf[b0 + 3] = v3;
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

extern "C" __global__ void v1_gate_sigmoid_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    int hblock)
{
    __shared__ float buf[1024];
    const int base = blockIdx.x * hblock;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* g4 = reinterpret_cast<const float4*>(gate + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = threadIdx.x; i < hblock / 4; i += blockDim.x) {
        float4 a = x4[i];
        float4 g = g4[i];
        float4 s = s4[i];
        float v0 = (a.x * (1.0f / (1.0f + expf(-g.x)))) * s.x;
        float v1 = (a.y * (1.0f / (1.0f + expf(-g.y)))) * s.y;
        float v2 = (a.z * (1.0f / (1.0f + expf(-g.z)))) * s.z;
        float v3 = (a.w * (1.0f / (1.0f + expf(-g.w)))) * s.w;
        int b0 = i * 4;
        buf[b0] = v0; buf[b0 + 1] = v1; buf[b0 + 2] = v2; buf[b0 + 3] = v3;
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

extern "C" __global__ void v1_swiglu_sign_fwht_quantize_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    int hblock)
{
    __shared__ float buf[1024];
    const int base = blockIdx.x * hblock;
    const float4* g4 = reinterpret_cast<const float4*>(gate + base);
    const float4* u4 = reinterpret_cast<const float4*>(up + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = threadIdx.x; i < hblock / 4; i += blockDim.x) {
        float4 g = g4[i];
        float4 u = u4[i];
        float4 s = s4[i];
        float v0 = ((g.x / (1.0f + expf(-g.x))) * u.x) * s.x;
        float v1 = ((g.y / (1.0f + expf(-g.y))) * u.y) * s.y;
        float v2 = ((g.z / (1.0f + expf(-g.z))) * u.z) * s.z;
        float v3 = ((g.w / (1.0f + expf(-g.w))) * u.w) * s.w;
        int b0 = i * 4;
        buf[b0] = v0; buf[b0 + 1] = v1; buf[b0 + 2] = v2; buf[b0 + 3] = v3;
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

// ── V2: V1 + register-tiled butterfly stages 2 and 4 (+ the shfl_down
// reduction tail for K1).
//
// Stage legality: at step s the butterfly's pair set {(j, j+s/2)} is fixed;
// each pair reads/writes only its own two slots, so thread assignment cannot
// change any operand or the stage order. Thread t owns the CONSECUTIVE four
// elements 4t..4t+3 ⇒ the step-2 pairs (4t,4t+1),(4t+2,4t+3) and the step-4
// pairs (4t,4t+2),(4t+1,4t+3) are entirely thread-local: two of the ten
// `__syncthreads` barriers vanish with zero value change.
//
// The K1 reduction tail: levels 128/64/32 stay in smem VERBATIM; levels
// 16..1 replicate the shipped adds (lane L<s computes red[L] += red[L+s])
// as `__shfl_down_sync` within warp 0 — the SAME two operands in the SAME
// order; upper lanes compute sums that are simply never read. An xor tree
// would re-associate and drift bits — deliberately not used.

__device__ __forceinline__ void v2_regtile_stage24_store(
    float* buf, int i, float v0, float v1, float v2, float v3)
{
    const float k = 0.7071067811865476f;
    float t0 = (v0 + v1) * k;
    float t1 = (v0 - v1) * k;
    float t2 = (v2 + v3) * k;
    float t3 = (v2 - v3) * k;
    float w0 = (t0 + t2) * k;
    float w2 = (t0 - t2) * k;
    float w1 = (t1 + t3) * k;
    float w3 = (t1 - t3) * k;
    int b0 = i * 4;
    buf[b0] = w0; buf[b0 + 1] = w1; buf[b0 + 2] = w2; buf[b0 + 3] = w3;
}

__device__ __forceinline__ void rot_butterfly_from_8(float* buf, int hblock)
{
    const float inv_sqrt2 = 0.7071067811865476f;
    for (int step = 8; step <= hblock; step <<= 1) {
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

extern "C" __global__ void v3_rmsnorm_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    float inv_dim, float eps,
    int dim, int hblock)
{
    // V3 = V1 + the shfl_down reduction tail ONLY (full 10-stage smem
    // butterfly) — the bisect arm that isolates the tail.
    __shared__ float red[256];
    __shared__ float buf[1024];
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
    float r = red[tid];
    r += __shfl_down_sync(0xFFFFFFFFu, r, 16);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 8);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 4);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 2);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 1);
    if (tid == 0) {
        float mean_sq = r * inv_dim;
        red[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float inv_rms = red[0];

    const int base = blockIdx.x * hblock;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = tid; i < hblock / 4; i += blockDim.x) {
        float4 a = x4[i];
        float4 g = g4[i];
        float4 s = s4[i];
        float v0 = (a.x * inv_rms * g.x) * s.x;
        float v1 = (a.y * inv_rms * g.y) * s.y;
        float v2 = (a.z * inv_rms * g.z) * s.z;
        float v3 = (a.w * inv_rms * g.w) * s.w;
        int b0 = i * 4;
        buf[b0] = v0; buf[b0 + 1] = v1; buf[b0 + 2] = v2; buf[b0 + 3] = v3;
    }
    __syncthreads();
    rot_butterfly_1024(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

extern "C" __global__ void v4_rmsnorm_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    float inv_dim, float eps,
    int dim, int hblock)
{
    // V4 = V1 + the register-tiled stages 2+4 ONLY (shipped full smem
    // reduction) — the bisect arm that isolates the reg-tile.
    __shared__ float red[256];
    __shared__ float buf[1024];
    float inv_rms;
    rot_reduction_inv_rms_256(x, dim, inv_dim, eps, red, &inv_rms);
    const int base = blockIdx.x * hblock;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = threadIdx.x; i < hblock / 4; i += blockDim.x) {
        float4 a = x4[i];
        float4 g = g4[i];
        float4 s = s4[i];
        float v0 = (a.x * inv_rms * g.x) * s.x;
        float v1 = (a.y * inv_rms * g.y) * s.y;
        float v2 = (a.z * inv_rms * g.z) * s.z;
        float v3 = (a.w * inv_rms * g.w) * s.w;
        v2_regtile_stage24_store(buf, i, v0, v1, v2, v3);
    }
    __syncthreads();
    rot_butterfly_from_8(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

extern "C" __global__ void v2_rmsnorm_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    float inv_dim, float eps,
    int dim, int hblock)
{
    __shared__ float red[256];
    __shared__ float buf[1024];
    // reduction: levels 128/64/32 in smem verbatim, tail (16..1) in-warp.
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
    float r = red[tid];
    r += __shfl_down_sync(0xFFFFFFFFu, r, 16);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 8);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 4);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 2);
    r += __shfl_down_sync(0xFFFFFFFFu, r, 1);
    if (tid == 0) {
        float mean_sq = r * inv_dim;
        red[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float inv_rms = red[0];

    const int base = blockIdx.x * hblock;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = tid; i < hblock / 4; i += blockDim.x) {
        float4 a = x4[i];
        float4 g = g4[i];
        float4 s = s4[i];
        float v0 = (a.x * inv_rms * g.x) * s.x;
        float v1 = (a.y * inv_rms * g.y) * s.y;
        float v2 = (a.z * inv_rms * g.z) * s.z;
        float v3 = (a.w * inv_rms * g.w) * s.w;
        v2_regtile_stage24_store(buf, i, v0, v1, v2, v3);
    }
    __syncthreads();
    rot_butterfly_from_8(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

extern "C" __global__ void v2_gate_sigmoid_sign_fwht_quantize_f32(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    int hblock)
{
    __shared__ float buf[1024];
    const int base = blockIdx.x * hblock;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* g4 = reinterpret_cast<const float4*>(gate + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = threadIdx.x; i < hblock / 4; i += blockDim.x) {
        float4 a = x4[i];
        float4 g = g4[i];
        float4 s = s4[i];
        float v0 = (a.x * (1.0f / (1.0f + expf(-g.x)))) * s.x;
        float v1 = (a.y * (1.0f / (1.0f + expf(-g.y)))) * s.y;
        float v2 = (a.z * (1.0f / (1.0f + expf(-g.z)))) * s.z;
        float v3 = (a.w * (1.0f / (1.0f + expf(-g.w)))) * s.w;
        v2_regtile_stage24_store(buf, i, v0, v1, v2, v3);
    }
    __syncthreads();
    rot_butterfly_from_8(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}

extern "C" __global__ void v2_swiglu_sign_fwht_quantize_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ signs,
    signed char* __restrict__ out,
    float* __restrict__ ascale,
    int hblock)
{
    __shared__ float buf[1024];
    const int base = blockIdx.x * hblock;
    const float4* g4 = reinterpret_cast<const float4*>(gate + base);
    const float4* u4 = reinterpret_cast<const float4*>(up + base);
    const float4* s4 = reinterpret_cast<const float4*>(signs + base);
    for (int i = threadIdx.x; i < hblock / 4; i += blockDim.x) {
        float4 g = g4[i];
        float4 u = u4[i];
        float4 s = s4[i];
        float v0 = ((g.x / (1.0f + expf(-g.x))) * u.x) * s.x;
        float v1 = ((g.y / (1.0f + expf(-g.y))) * u.y) * s.y;
        float v2 = ((g.z / (1.0f + expf(-g.z))) * u.z) * s.z;
        float v3 = ((g.w / (1.0f + expf(-g.w))) * u.w) * s.w;
        v2_regtile_stage24_store(buf, i, v0, v1, v2, v3);
    }
    __syncthreads();
    rot_butterfly_from_8(buf, hblock);
    rot_quantize_16(buf, base, hblock, out, ascale);
}
"#;

// (The V1 kernels inline their fills because the two-input expressions
// differ per kernel; the shared shape is the float4 global load + scalar
// smem store pair.)

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Rmsnorm,
    GateSigmoid,
    Swiglu,
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_arm(
    kind: Kind,
    f: &CudaFunction,
    stream: &CudaStream,
    in_a: &CudaSlice<f32>,
    in_b: &CudaSlice<f32>,
    signs: &CudaSlice<f32>,
    out: &CudaSlice<i8>,
    ascale: &CudaSlice<f32>,
    dim: usize,
    hblock: usize,
) {
    let grid = (dim / hblock) as u32;
    let hb = hblock as i32;
    // The builder borrows every arg until `.launch` — all scalar temporaries
    // live at fn scope so the borrow outlives the match arms (E0597).
    let inv_dim = 1.0f32 / dim as f32;
    let eps = 1e-6f32;
    let dim_i = dim as i32;
    unsafe {
        let mut b = stream.launch_builder(f);
        match kind {
            Kind::Rmsnorm => {
                b.arg(in_a)
                    .arg(in_b)
                    .arg(signs)
                    .arg(out)
                    .arg(ascale)
                    .arg(&inv_dim)
                    .arg(&eps)
                    .arg(&dim_i)
                    .arg(&hb);
            }
            Kind::GateSigmoid => {
                b.arg(in_a)
                    .arg(in_b)
                    .arg(signs)
                    .arg(out)
                    .arg(ascale)
                    .arg(&hb);
            }
            Kind::Swiglu => {
                b.arg(in_a)
                    .arg(in_b)
                    .arg(signs)
                    .arg(out)
                    .arg(ascale)
                    .arg(&hb);
            }
        }
        b.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
        .expect("probe launch");
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[test]
#[ignore = "requires a CUDA device (the 4090 lane); run --release -- --ignored --nocapture"]
fn bench903_fwht_rotq_variant_probe() {
    // ── box-state provenance + the exclusivity refusal ──
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader",
        ])
        .output()
    {
        eprintln!(
            "PROVENANCE: nvidia-smi [{}]",
            String::from_utf8_lossy(&out.stdout).trim()
        );
    }
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ])
        .output()
    {
        let busy: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| {
                let l = l.to_lowercase();
                l.contains("riir")
                    || l.contains("cargo")
                    || l.contains("python")
                    || l.contains("llama")
            })
            .map(str::to_string)
            .collect();
        assert!(
            busy.is_empty(),
            "GPU-EXCLUSIVITY: co-resident compute process(es) {busy:?} — the identity gate \
             is a correctness gate (the Bench-649 law), refusing"
        );
    }

    let ctx = CudaContext::new(0).expect("CUDA context (the 4090 lane)");
    let stream = ctx.new_stream().expect("stream");
    let ptx = cudarc::nvrtc::compile_ptx(SRC).expect("compile rotation probe");
    let module = ctx.load_module(ptx).expect("load module");
    let load = |name: &str| -> CudaFunction {
        module
            .load_function(name)
            .unwrap_or_else(|e| panic!("load {name}: {e}"))
    };
    let _keepalive: Arc<_> = module.clone();

    let pairs: usize = std::env::var("B903_PAIRS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let burst: usize = std::env::var("B903_BURST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    assert!(pairs >= 3, "need at least 3 pairs for a median");

    let shapes: [(usize, Kind, &str); 3] = [
        (
            5120,
            Kind::Rmsnorm,
            "rmsnorm(K1: 64 FFN + 16 attn inputs + final)/token=81",
        ),
        (
            5120,
            Kind::GateSigmoid,
            "gate_sigmoid(K4, attention wo input)/token=16",
        ),
        (17408, Kind::Swiglu, "swiglu(K5, FFN down input)/token=64"),
    ];
    let hblock = 1024usize;
    assert_eq!(hblock % 4, 0, "V1/V2 fill requires hblock % 4 == 0");

    // census per-token µs (Bench 902) for the projected-family summary —
    // printed FROM the measured ratios, never hand-projected.
    let census_us_per_token = |kind: Kind| match kind {
        Kind::Rmsnorm => 408.0f64,
        Kind::GateSigmoid => 77.0,
        Kind::Swiglu => 303.0,
    };

    let mut family_ratio_sum = 0.0f64;
    let mut family_census_sum = 0.0f64;
    let mut shipping_ratio = 1.0f64; // per-shape shipping-arm ratio (V3 else V1)

    for (si, &(dim, kind, label)) in shapes.iter().enumerate() {
        let mut rng = Lcg(0x9E3779B97F4A7C15 ^ (si as u64 + 1));
        let a_h: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        let b_h: Vec<f32> = (0..dim).map(|_| 0.5 + rng.next_f32()).collect();
        let s_h: Vec<f32> = (0..dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let a_d = stream.clone_htod(&a_h).expect("htod a");
        let b_d = stream.clone_htod(&b_h).expect("htod b");
        let s_d = stream.clone_htod(&s_h).expect("htod s");
        let out_d = stream.alloc_zeros::<i8>(dim).expect("out");
        let ascale_d = stream.alloc_zeros::<f32>(dim / 16).expect("ascale");

        // per-kind arg mapping: Rmsnorm(a=x, b=gamma) · GateSigmoid(a=x, b=gate)
        // · Swiglu(a=gate, b=up) — kernel signatures are (x|gate, gamma|gate|up,
        // signs, ...), so (a_d, b_d) map straight through for every kind.
        let p0 = load(match kind {
            Kind::Rmsnorm => "p0_rmsnorm_sign_fwht_quantize_f32",
            Kind::GateSigmoid => "p0_gate_sigmoid_sign_fwht_quantize_f32",
            Kind::Swiglu => "p0_swiglu_sign_fwht_quantize_f32",
        });
        let v1 = load(match kind {
            Kind::Rmsnorm => "v1_rmsnorm_sign_fwht_quantize_f32",
            Kind::GateSigmoid => "v1_gate_sigmoid_sign_fwht_quantize_f32",
            Kind::Swiglu => "v1_swiglu_sign_fwht_quantize_f32",
        });
        let v2 = load(match kind {
            Kind::Rmsnorm => "v2_rmsnorm_sign_fwht_quantize_f32",
            Kind::GateSigmoid => "v2_gate_sigmoid_sign_fwht_quantize_f32",
            Kind::Swiglu => "v2_swiglu_sign_fwht_quantize_f32",
        });
        // bisect arms (rmsnorm only — the shape that drifted first)
        let (v3, v4) = if kind == Kind::Rmsnorm {
            (
                Some(load("v3_rmsnorm_sign_fwht_quantize_f32")),
                Some(load("v4_rmsnorm_sign_fwht_quantize_f32")),
            )
        } else {
            (None, None)
        };

        let time_arm = |f: &CudaFunction| -> f64 {
            stream.synchronize().expect("sync");
            let t = Instant::now();
            for _ in 0..burst {
                launch_arm(
                    kind, f, &stream, &a_d, &b_d, &s_d, &out_d, &ascale_d, dim, hblock,
                );
            }
            stream.synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e6 / burst as f64
        };

        // ── bit-identity gate (with drift diagnostics) ──
        let snapshot = |f: &CudaFunction| -> (Vec<i8>, Vec<u32>) {
            launch_arm(
                kind, f, &stream, &a_d, &b_d, &s_d, &out_d, &ascale_d, dim, hblock,
            );
            stream.synchronize().expect("sync");
            let mut o = vec![0i8; dim];
            let mut a = vec![0f32; dim / 16];
            stream.memcpy_dtoh(&out_d, &mut o).expect("dtoh out");
            stream.memcpy_dtoh(&ascale_d, &mut a).expect("dtoh ascale");
            (o, a.iter().map(|v| v.to_bits()).collect())
        };
        let drift_report = |name: &str, s: &(Vec<i8>, Vec<u32>), r: &(Vec<i8>, Vec<u32>)| {
            let bad_o: Vec<usize> = (0..dim).filter(|&i| s.0[i] != r.0[i]).collect();
            let bad_a: Vec<usize> = (0..dim / 16).filter(|&i| s.1[i] != r.1[i]).collect();
            if bad_o.is_empty() && bad_a.is_empty() {
                eprintln!("[bench903]   {name}: OK (i8 eq + ascale to_bits eq)");
                true
            } else {
                let first = bad_o.first().copied().unwrap_or(bad_a[0] * 16);
                eprintln!(
                    "[bench903]   {name}: DRIFT — {} i8 + {} ascale mismatches; first i8 at {first}: p0={} var={}",
                    bad_o.len(),
                    bad_a.len(),
                    s.0[first],
                    r.0[first]
                );
                false
            }
        };
        let reference = snapshot(&p0);
        let v1_s = snapshot(&v1);
        let v2_s = snapshot(&v2);
        eprintln!("[bench903] dim={dim} {label}");
        let v1_ok = drift_report("V1 float4-fill", &reference, &v1_s);
        let v2_ok = drift_report("V2 +regtile+shfltail", &reference, &v2_s);
        // Bisect arms (rmsnorm only): the Bench-902-followup run isolated the
        // reg-tile as the drifter (V4 DRIFT with 0 i8 + 106 ascale = the
        // common-mode ±1-ULP signature — i8 is invariant to a group-wide
        // 1-ULP scale, the stored absmax is not). This is the recorded nvcc
        // contraction-pairing class: the same math, register-resident
        // operands, pairs differently than the smem-load form. Dead by the
        // FNV-pin law; V3 (shfl-tail only) is the shipping candidate.
        let mut v3_ok = false;
        let mut v4_ok = false;
        let (v3_s, v4_s) = match (&v3, &v4) {
            (Some(f3), Some(f4)) => {
                let s3 = snapshot(f3);
                let s4 = snapshot(f4);
                v3_ok = drift_report("V3 shfltail-only", &reference, &s3);
                v4_ok = drift_report("V4 regtile-only", &reference, &s4);
                (Some(s3), Some(s4))
            }
            _ => (None, None),
        };
        let _ = (v1_s, v2_s, v3_s, v4_s);
        // A drifted variant is EXCLUDED from the rung set, never a panic —
        // the probe prices the shipping space (the identity gate IS the
        // exclusion rule).
        if !v1_ok || !v2_ok || (kind == Kind::Rmsnorm && (!v3_ok || !v4_ok)) {
            eprintln!("[bench903]   (drifted arms excluded from timing)");
        }
        let _ = v2_ok;

        // ── interleaved burst A/B (shipping arms only) ──
        for _ in 0..60 {
            launch_arm(
                kind, &p0, &stream, &a_d, &b_d, &s_d, &out_d, &ascale_d, dim, hblock,
            );
            if v1_ok {
                launch_arm(
                    kind, &v1, &stream, &a_d, &b_d, &s_d, &out_d, &ascale_d, dim, hblock,
                );
            }
            if v3_ok {
                launch_arm(
                    kind,
                    v3.as_ref().unwrap(),
                    &stream,
                    &a_d,
                    &b_d,
                    &s_d,
                    &out_d,
                    &ascale_d,
                    dim,
                    hblock,
                );
            }
        }
        stream.synchronize().expect("sync");

        // A/A control on the first shape (harness noise reference).
        if si == 0 {
            let mut aa = Vec::with_capacity(pairs);
            for i in 0..pairs {
                let r = if i % 2 == 0 {
                    let x = time_arm(&p0);
                    x / time_arm(&p0)
                } else {
                    let y = time_arm(&p0);
                    time_arm(&p0) / y
                };
                aa.push(r);
            }
            eprintln!(
                "[bench903]   A/A    p0 vs p0           ratio median {:.4} (harness noise reference)",
                median(aa)
            );
        }

        for (name, f, ok) in [
            ("V1 float4-fill", Some(&v1), v1_ok),
            ("V3 shfltail", v3.as_ref(), v3_ok),
        ] {
            if !ok {
                eprintln!("[bench903]   {name:<20} DRIFTED — excluded from timing");
                continue;
            }
            let f = f.expect("shipping arm");
            let mut ratios = Vec::with_capacity(pairs);
            let mut base_us = Vec::with_capacity(pairs);
            let mut var_us = Vec::with_capacity(pairs);
            for i in 0..pairs {
                let (b, v) = if i % 2 == 0 {
                    let x = time_arm(&p0);
                    (x, time_arm(f))
                } else {
                    let y = time_arm(f);
                    (time_arm(&p0), y)
                };
                ratios.push(v / b);
                base_us.push(b);
                var_us.push(v);
            }
            let mr = median(ratios);
            eprintln!(
                "[bench903]   {name:<20} p0 {:>6.3}µs → var {:>6.3}µs  ratio median {mr:.4}  speedup {:.3}x  (burst {burst}, pairs {pairs})",
                median(base_us),
                median(var_us),
                1.0 / mr
            );
            // shipping ratio per kind — V3 lands last and wins when present
            // (more barriers saved); V1 is the fallback
            shipping_ratio = mr;
        }
        family_ratio_sum += census_us_per_token(kind) * shipping_ratio;
        family_census_sum += census_us_per_token(kind);
    }

    // ── projected family summary (from measured ratios × census weights) ──
    const TOKEN_US: f64 = 10_907.0; // 91.68 tok/s (Bench 1004 the RED baseline)
    let projected = family_ratio_sum;
    let census_total = family_census_sum;
    eprintln!("[bench903] ── family projection (shipping arms, census-weighted) ──");
    eprintln!(
        "[bench903]   family {:?} µs/token → projected {:.0} µs/token | saves {:.0} µs/token = {:.2}% of the {:.0} µs token (re-cross bar +2.71%)",
        census_total,
        projected,
        census_total - projected,
        (census_total - projected) / TOKEN_US * 100.0,
        TOKEN_US
    );
}
