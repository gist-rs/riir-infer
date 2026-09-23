//! Issue 755 T1 — the GPU (cudarc) port of the DFlash2 drafter forward.
//!
//! The CPU implementation ([`crate::qwen38_dflash2`]) is the ORACLE — this
//! module mirrors `draft_block_hidden` + `inject_position` op-for-op on the
//! GPU. The port exists because a CPU drafter forward can never pay in the
//! chat lane (Bench 746: the trained head is the only chat-capable drafter
//! at acceptance 0.49 vs lookup's 0.000; wall target ≤ 7 ms/cycle ≈ the
//! breakeven input at ~3.85 GB of BF16 weights).
//!
//! # What ships
//!
//! - `DFlash2GpuDrafter` — bf16-resident weights (~3.52 GB), per-layer K/V
//!   rings `[2048][8*128]` f32, host `ring_pos` occupancy (mirroring the
//!   CPU drafter's `VecDeque`), batched `inject_positions` + the
//!   `draft_block_hidden_gpu` noise-block forward.
//! - New CUDA kernels (one module, `sm_89`):
//!   `dflash2_gemv_bf16_rows` (the workhorse rows-GEMV: one block per
//!   output column per 8-row chunk, 256 threads, strided-K accumulate +
//!   smem tree — the `qwen38_rmsnorm_quant_x_q8` phase 1-2 reduction
//!   shape), `dflash2_rmsnorm_rows_f32` (plain f32, NO quantize),
//!   `dflash2_qk_norm_rope_neox` (per-head rmsnorm(128) + NEOX half-split
//!   RoPE fused), `dflash2_attention` (2-pass max/softmax over
//!   [ring ∪ block], non-causal per the TRAINED semantics — the checkpoint's
//!   config.json carries is_causal:false, honored by both z-lab refs and the
//!   llama.cpp fork (Issue 989 falsified the causal reading; knob
//!   `QWEN38_DFLASH2_CAUSAL=1`) — SWA floor, GQA 4, one block per
//!   (row, head)), `dflash2_conv` (the dynamic group conv, both sides),
//!   `dflash2_scatter_ring`, and the two 5-line elementwise twins
//!   (`dflash2_residual_add`, `dflash2_swiglu` — see the deviation note).
//! - Weight loading streams the safetensors tensor-by-tensor as RAW BF16
//!   (`u16`) — no f32 materialization of the 3.85 GB set (host peak = one
//!   tensor; the `fc` at 268 MB is the largest). The file's own minimal
//!   BF16 reader is local to this module (the CPU oracle's
//!   `SafetensorsFile` is private and returns f32 — different need, noted
//!   here for provenance).
//!
//! # Numerics contract (mirrored, op-for-op, per layer)
//!
//! 1. `h = rmsnorm(x, attn_norm)` (plain f32, no quantize)
//! 2. `dyn_coeff = attn_conv_proj @ h` (1280 out/row)
//! 3. conv side-0 on `h` → `hc` (`out[i][c] = Σ_t (dyn[i][g+n_groups·(t+kernel·side)]
//!    + base[side·(kernel·e)+t·e+c]) · src[i−t][c]`, zero-padded)
//! 4. Q/K/V GEMVs from `hc`
//! 5. per-head q_norm/k_norm rmsnorm + NEOX RoPE at `anchor_pos + i`
//! 6. attention over [ring ∪ block]: non-causal (the TRAINED semantics —
//!    config.json is_causal:false; Issue 989 falsified the causal reading,
//!    knob `QWEN38_DFLASH2_CAUSAL=1` arms the refuted arm), SWA
//!    `key_pos ≥ q_pos − (swa−1)`, ring entry at `anchor_pos` skipped (the
//!    block KV replaces it — the llama.cpp unified-cache semantics the CPU
//!    port pins), GQA group 4, scale `1/√128`, 2-pass online softmax (pass
//!    1 max over UNSCALED dots; pass 2 `w = exp((dot−max)·scale)`; divide
//!    by sum)
//! 7. o_proj → conv side-1 → `ffn_inp = attn_final + x`
//! 8. ffn_norm → ffn_conv_proj → conv side-0 → gate/up → `silu(g)*u` →
//!    down → conv side-1 → `x = ffn_final + ffn_inp`
//!
//! Final `output_norm` → the `[8][5120]` rows.
//!
//! Bit-identity vs the CPU is NOT expected (accumulation orders differ:
//! tree reductions vs serial sums; CUDA `expf/sinf/cosf` vs libm ulps) —
//! the gate is per-row cosine ≥ 0.99999 (the 749 T4 precedent). The GPU
//! forward itself is deterministic (fixed reduction trees, no atomics).
//!
//! # Deviations from the task sketch (documented)
//!
//! - `residual_add`/`swiglu` ship as twins inside this module instead of
//!   reusing `cudarc_kernels::ElementwiseKernels`: its constructor needs
//!   the `Arc<CudaContext>`, which this API surface deliberately does not
//!   carry (`new(stream, cfg, path)` — the stream owns its context). The
//!   twins are 5-line kernels with identical math.
//! - Ring geometry choice (the sketch offered two): the ring POSITION
//!   LIST is uploaded per cycle (`[≤2048] i32`, padded with `-1`, skipped
//!   in-kernel) — cheaper than a slot-visibility mask and it keeps the
//!   SWA floor + anchor-skip logic in one place that mirrors the CPU.
//! - The rows-GEMV tiles at 8 rows per block (`DFLASH2_TILE`): a batched
//!   inject of n positions is processed in 8-row chunks (the fc read
//!   amortizes 1× per chunk; n=8 — the verify-loop commit shape — reads
//!   it exactly once).
//! - `inject_positions` caps at `sliding_window` rows per call (mirrors
//!   the ring's capacity semantics).
//!
//! Non-goals (slice A): no chat-loop wiring, no `lattice_walk` port (the
//! selector stays CPU — cheap + pure), no CUDA-graph capture.
//!
//! Rides `ternary_gemv_cuda_raw` (dep:cudarc; CUDA-only, non-macOS).
#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use riir_infer_core::gguf_loader::{GgufFile, GgmlType};

use crate::qwen38_dflash2::DFlash2Config;

/// Rows per GEMV block chunk (block reads its weight column once per chunk).
const DFLASH2_TILE: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA source
// ─────────────────────────────────────────────────────────────────────────────

const DFLASH2_CUDA_SRC: &str = r#"
// bf16 -> f32: widen the 16 bits into the f32 high half (exact).
__device__ __forceinline__ float dfl_bf16_f32(unsigned short h) {
    return __int_as_float(((unsigned int)h) << 16);
}

// The workhorse rows-GEMV: out[r][o] = sum_i W[o][i] * x[r][i].
// W is bf16 [out_dim][in_dim] (row-major, dot of row · x — the CPU
// gemv_into convention); x is f32 [rows][in_dim]; y is f32 [rows][out_dim].
// One block per (output o, 8-row chunk); 256 threads stride in_dim in
// 4-ELEMENT chunks (uint2 w loads + float4 x loads — the vectorized shape;
// every in_dim in this model is a multiple of 256*4=1024, asserted
// host-side, so there is no tail path), accumulate acc[TILE] partial dots,
// then a per-row smem tree reduction (the qwen38_rmsnorm_quant_x_q8 phase
// 1-2 reduction shape). Weights are read exactly once per chunk; x
// re-reads ride L1/L2.
extern "C" __global__ void dflash2_gemv_bf16_rows(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int out_dim, const int in_dim, const int rows)
{
    const int o = blockIdx.x;
    const int r0 = blockIdx.y * 8;
    const int tid = threadIdx.x;
    const unsigned short* wrow = w + (size_t)o * (size_t)in_dim;
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.f;
    for (int i0 = tid * 4; i0 < in_dim; i0 += blockDim.x * 4) {
        const uint2 wraw = *reinterpret_cast<const uint2*>(wrow + i0);
        float wv0 = dfl_bf16_f32((unsigned short)(wraw.x & 0xFFFFu));
        float wv1 = dfl_bf16_f32((unsigned short)(wraw.x >> 16));
        float wv2 = dfl_bf16_f32((unsigned short)(wraw.y & 0xFFFFu));
        float wv3 = dfl_bf16_f32((unsigned short)(wraw.y >> 16));
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            const int r = r0 + t;
            if (r < rows) {
                const float4 xv =
                    *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + i0);
                acc[t] += wv0 * xv.x + wv1 * xv.y + wv2 * xv.z + wv3 * xv.w;
            }
        }
    }
    __shared__ float smem[8][256];
    #pragma unroll
    for (int t = 0; t < 8; t++) smem[t][tid] = acc[t];
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            #pragma unroll
            for (int t = 0; t < 8; t++) smem[t][tid] += smem[t][tid + off];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        const int r = r0 + t;
        if (tid == 0 && r < rows) y[(size_t)r * out_dim + o] = smem[t][0];
    }
}

// Plain rmsnorm over rows, f32 out (NO quantize) — mirrors the CPU
// rmsnorm_into: inv = 1 / sqrt(ss/n + eps); out = x * inv * gamma.
extern "C" __global__ void dflash2_rmsnorm_rows_f32(
    const float* __restrict__ input,  // [rows][dim]
    const float* __restrict__ gamma,  // [dim]
    float* __restrict__ output,       // [rows][dim]
    const float eps, const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const float* xrow = input + (size_t)row * dim;
    float partial = 0.f;
    for (int i = tid; i < dim; i += nthr) {
        const float v = xrow[i];
        partial += v * v;
    }
    __shared__ float smem[256];
    smem[tid] = partial;
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
    const float inv = 1.0f / sqrtf(smem[0] / (float)dim + eps);
    float* orow = output + (size_t)row * dim;
    for (int i = tid; i < dim; i += nthr) {
        orow[i] = xrow[i] * inv * gamma[i];
    }
}

// Per-head rmsnorm(head_dim) + NEOX half-split RoPE at pos = base_pos+row,
// IN PLACE. One block per (row, head), head_dim threads (head_dim == 128 —
// launcher contract). Mirrors the CPU: norm seg -> smem -> threads < half
// rotate the (i, i+half) pair by ang = pos * freq[i].
extern "C" __global__ void dflash2_qk_norm_rope_neox(
    float* __restrict__ qk,           // [rows][n_heads][head_dim]
    const float* __restrict__ gamma,  // [head_dim]
    const float* __restrict__ freqs,  // [head_dim/2]
    const int n_heads, const int head_dim, const int base_pos,
    const float eps)
{
    const int row = blockIdx.x / n_heads;
    const int head = blockIdx.x % n_heads;
    const int tid = threadIdx.x;
    const int pos = base_pos + row;
    float* seg = qk + ((size_t)row * n_heads + head) * head_dim;
    const float v = seg[tid];
    __shared__ float red[128];
    red[tid] = v * v;
    __syncthreads();
    if (tid < 64) red[tid] += red[tid + 64]; __syncthreads();
    if (tid < 32) red[tid] += red[tid + 32]; __syncthreads();
    if (tid < 16) red[tid] += red[tid + 16]; __syncthreads();
    if (tid < 8)  red[tid] += red[tid + 8];  __syncthreads();
    if (tid < 4)  red[tid] += red[tid + 4];  __syncthreads();
    if (tid < 2)  red[tid] += red[tid + 2];  __syncthreads();
    if (tid < 1)  red[0] += red[1];
    __syncthreads();
    const float inv = 1.0f / sqrtf(red[0] / (float)head_dim + eps);
    __shared__ float normed[128];
    normed[tid] = v * inv * gamma[tid];
    __syncthreads();
    const int half = head_dim >> 1;
    if (tid < half) {
        const float ang = (float)pos * freqs[tid];
        const float s = sinf(ang);
        const float c = cosf(ang);
        const float a = normed[tid];
        const float b = normed[tid + half];
        seg[tid] = a * c - b * s;
        seg[tid + half] = a * s + b * c;
    }
}

// Attention over [ring ∪ block KV], NON-CAUSAL (the TRAINED semantics —
// config.json is_causal:false, both z-lab refs honor it, Issue 989
// falsified the causal reading; causal=1 arms the refuted arm) with the
// SWA floor (key_pos >= q_pos - (swa-1)), ring entry at anchor_pos
// skipped, GQA
// head h -> kv head h/grp, scale 1/sqrt(head_dim). One block per
// (block-row i, head h), 128 threads = 4 warps, candidates STRIPED
// across the warps; each 128-dim dot is a per-lane 4-element product +
// a 5-step shuffle butterfly (no per-candidate barriers). Two passes
// mirroring the CPU semantics: pass A max over UNSCALED dots (warp
// local, then one cross-warp combine), pass B w = exp((dot-M)*scale),
// w*V accumulated per lane's 4 dims, one final cross-warp combine +
// element-wise divide (like the CPU). Summation order differs from the
// CPU's candidate-serial order — cosine-gated, not bit-gated.
extern "C" __global__ void dflash2_attention(
    const float* __restrict__ q,        // [bs][n_head][head_dim]
    const float* __restrict__ k_block,  // [bs][n_kv_head][head_dim]
    const float* __restrict__ v_block,  // [bs][n_kv_head][head_dim]
    const float* __restrict__ ring_k,   // [swa][n_kv_head*head_dim]
    const float* __restrict__ ring_v,
    const int* __restrict__ ring_pos,   // [n_ring] (-1 pad entries skipped)
    const int n_ring,
    const int anchor_pos, const int swa, const int grp,
    const int bs, const int n_head, const int n_kv_head, const int head_dim,
    const float scale,
    const int causal,                   // 1: mask block keys c > i (Issue 989)
    float* __restrict__ attn_out)       // [bs][n_head][head_dim]
{
    const int i = blockIdx.x / n_head;
    const int h = blockIdx.x % n_head;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kvh = h / grp;
    const int q_pos = anchor_pos + i;
    const int lo = q_pos - (swa - 1);
    const float* qseg = q + ((size_t)i * n_head + h) * head_dim;
    const float4 q4 = *reinterpret_cast<const float4*>(qseg + lane * 4);
    const int n_cand = bs + n_ring;

    __shared__ float sh_max[4];
    __shared__ float sh_sum[4];
    __shared__ float sh_acc[4][128];

    // ── Pass A: max unscaled score over visible keys (warp-striped) ──
    float m_w = -__int_as_float(0x7F800000u); // -inf (NVRTC has no INFINITY)
    for (int c = warp; c < n_cand; c += 4) {
        const float* kseg;
        if (c < bs) {
            if (causal && c > i) continue;
            if (anchor_pos + c < lo) continue;
            kseg = k_block + ((size_t)c * n_kv_head + kvh) * head_dim;
        } else {
            const int rp = ring_pos[c - bs];
            if (rp < 0 || rp == anchor_pos || rp < lo) continue;
            kseg = ring_k + ((size_t)(rp % swa) * n_kv_head + kvh) * head_dim;
        }
        const float4 k4 = *reinterpret_cast<const float4*>(kseg + lane * 4);
        float d = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
        m_w = fmaxf(m_w, d);
    }
    if (lane == 0) sh_max[warp] = m_w;
    __syncthreads();
    const float M =
        fmaxf(fmaxf(sh_max[0], sh_max[1]), fmaxf(sh_max[2], sh_max[3]));

    // ── Pass B: softmax weights + V accumulation (dots recomputed, like
    // the CPU; each lane owns dims [lane*4, lane*4+4)) ──
    float l_w = 0.f;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int c = warp; c < n_cand; c += 4) {
        const float* kseg;
        const float* vseg;
        if (c < bs) {
            if (causal && c > i) continue;
            if (anchor_pos + c < lo) continue;
            kseg = k_block + ((size_t)c * n_kv_head + kvh) * head_dim;
            vseg = v_block + ((size_t)c * n_kv_head + kvh) * head_dim;
        } else {
            const int rp = ring_pos[c - bs];
            if (rp < 0 || rp == anchor_pos || rp < lo) continue;
            const int slot = rp % swa;
            kseg = ring_k + ((size_t)slot * n_kv_head + kvh) * head_dim;
            vseg = ring_v + ((size_t)slot * n_kv_head + kvh) * head_dim;
        }
        const float4 k4 = *reinterpret_cast<const float4*>(kseg + lane * 4);
        float d = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
        const float wgt = expf((d - M) * scale);
        l_w += wgt;
        const float4 v4 = *reinterpret_cast<const float4*>(vseg + lane * 4);
        a0 += wgt * v4.x;
        a1 += wgt * v4.y;
        a2 += wgt * v4.z;
        a3 += wgt * v4.w;
    }
    sh_acc[warp][lane * 4 + 0] = a0;
    sh_acc[warp][lane * 4 + 1] = a1;
    sh_acc[warp][lane * 4 + 2] = a2;
    sh_acc[warp][lane * 4 + 3] = a3;
    if (lane == 0) sh_sum[warp] = l_w;
    __syncthreads();
    const float sum = sh_sum[0] + sh_sum[1] + sh_sum[2] + sh_sum[3];
    float* oseg = attn_out + ((size_t)i * n_head + h) * head_dim;
    oseg[lane * 4 + 0] = (sh_acc[0][lane * 4 + 0] + sh_acc[1][lane * 4 + 0]
        + sh_acc[2][lane * 4 + 0] + sh_acc[3][lane * 4 + 0]) / sum;
    oseg[lane * 4 + 1] = (sh_acc[0][lane * 4 + 1] + sh_acc[1][lane * 4 + 1]
        + sh_acc[2][lane * 4 + 1] + sh_acc[3][lane * 4 + 1]) / sum;
    oseg[lane * 4 + 2] = (sh_acc[0][lane * 4 + 2] + sh_acc[1][lane * 4 + 2]
        + sh_acc[2][lane * 4 + 2] + sh_acc[3][lane * 4 + 2]) / sum;
    oseg[lane * 4 + 3] = (sh_acc[0][lane * 4 + 3] + sh_acc[1][lane * 4 + 3]
        + sh_acc[2][lane * 4 + 3] + sh_acc[3][lane * 4 + 3]) / sum;
}

// The DFlash2 dynamic conv (both sides): per block token i, channel c
// (group g = c/gsz): out[i][c] = sum_t (dyn[i][g + n_groups*(t + kernel*side)]
// + base[side*(kernel*e) + t*e + c]) * src[(i-t)*e + c], zero-padded at
// block start — mirrors apply_conv exactly.
extern "C" __global__ void dflash2_conv(
    const float* __restrict__ dyn_coeff,  // [bs][2*kernel*n_groups]
    const float* __restrict__ base,       // [2][kernel][e]
    const float* __restrict__ src,        // [bs][e]
    float* __restrict__ dst,              // [bs][e]
    const int bs, const int e, const int side,
    const int n_groups, const int kernel, const int gsz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * e) return;
    const int i = idx / e;
    const int c = idx % e;
    const int g = c / gsz;
    const int stride = 2 * kernel * n_groups;
    float acc = 0.f;
    for (int t = 0; t < kernel; t++) {
        if (i >= t) {
            const int di = g + n_groups * (t + kernel * side);
            const int bi = side * (kernel * e) + t * e + c;
            acc += (dyn_coeff[(size_t)i * stride + di] + base[bi])
                 * src[(size_t)(i - t) * e + c];
        }
    }
    dst[idx] = acc;
}

// Scatter n injected K/V rows into their ring slots.
extern "C" __global__ void dflash2_scatter_ring(
    float* __restrict__ ring,        // [swa][n_kv]
    const float* __restrict__ src,   // [rows][n_kv]
    const int* __restrict__ slots,   // [rows]
    const int rows, const int n_kv)
{
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int slot = slots[r];
    const int tid = threadIdx.x;
    for (int j = tid; j < n_kv; j += blockDim.x) {
        ring[(size_t)slot * n_kv + j] = src[(size_t)r * n_kv + j];
    }
}

// The two elementwise twins (math identical to cudarc_kernels'
// residual_add_f32 / swiglu_f32 — instantiated here because
// ElementwiseKernels::new needs the Arc<CudaContext> this API surface
// does not carry; see the module doc's deviation note).
extern "C" __global__ void dflash2_residual_add(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

extern "C" __global__ void dflash2_swiglu(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ out, const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float g = gate[idx];
        out[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

// ── Issue 780 U3 wall-1 lever (a): the quant-resident rows-GEMV family ──
// The DFlash2 drafter loads the analogalok mixed-quant GGUF (Q2_K majority
// + Q3_K attn_output/ffn_down + Q4_K attn_v) — ~5.4× fewer weight bytes per
// cycle than BF16. Same outer geometry as dflash2_gemv_bf16_rows (one block
// per (output o, 8-row chunk), 256 threads); each thread owns ONE K-quant
// sub-block strided by blockDim.x and dequants in registers before the FMA.

// Exact IEEE f16 -> f32 (the qwen38_dense_cudarc helper verbatim — NVRTC
// has no cuda_fp16.h in its default include path).
__device__ __forceinline__ float dfl_f16_f32(unsigned short h) {
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

// The shared per-row smem tree reduction (verbatim the bf16 kernel's tail).
__device__ __forceinline__ void dfl_rows_reduce(
    float* __restrict__ y, const int out_dim, const int o, const int tid,
    const int r0, const int rows, const float acc[8])
{
    __shared__ float smem[8][256];
    #pragma unroll
    for (int t = 0; t < 8; t++) smem[t][tid] = acc[t];
    __syncthreads();
    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            #pragma unroll
            for (int t = 0; t < 8; t++) smem[t][tid] += smem[t][tid + off];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        const int r = r0 + t;
        if (tid == 0 && r < rows) y[(size_t)r * out_dim + o] = smem[t][0];
    }
}

// Q2_K: 84 B / 256 elems — layout per ggml block_q2_K: scales[16] @0,
// qs[64] @16, d @80, dmin @82. The qs lanes are INTERLEAVED: element l of
// sub-block (half, b, j) lives in BYTE qs[half*32 + b*16 + l] at bit-shift
// 2*j (one 2-bit code per byte position, four sub-blocks sharing each byte
// at shifts 0/2/4/6 — the reference's `q[l] >> shift`). scale/min = the
// 4-bit halves of scales[half*8 + 2*j + b]. 84 ≡ 0 (mod 4) keeps every u32
// load naturally aligned.
extern "C" __global__ void dflash2_gemv_q2k_rows(
    const unsigned char* __restrict__ w,   // [out_dim][bpr] * 84
    const float* __restrict__ x,
    float* __restrict__ y,
    const int out_dim, const int in_dim, const int rows)
{
    const int o = blockIdx.x;
    const int r0 = blockIdx.y * 8;
    const int tid = threadIdx.x;
    const int bpr = in_dim >> 8;
    const unsigned char* wrow = w + (size_t)o * (size_t)bpr * 84;
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.f;
    const int nsb = in_dim >> 4;
    for (int sb = tid; sb < nsb; sb += blockDim.x) {
        const int blk = sb >> 4;
        const int j16 = sb & 15;
        const unsigned char* b = wrow + (size_t)blk * 84;
        const float d = dfl_f16_f32(*reinterpret_cast<const unsigned short*>(b + 80));
        const float dmin = dfl_f16_f32(*reinterpret_cast<const unsigned short*>(b + 82));
        const unsigned char sc = b[j16];
        const float dl = d * (float)(sc & 0xF);
        const float ml = dmin * (float)(sc >> 4);
        const unsigned char* q = b + 16 + ((j16 >> 3) << 5) + ((j16 & 1) << 4);
        const int shift = j16 & 6;
        const unsigned int qw0 = *reinterpret_cast<const unsigned int*>(q);
        const unsigned int qw1 = *reinterpret_cast<const unsigned int*>(q + 4);
        const unsigned int qw2 = *reinterpret_cast<const unsigned int*>(q + 8);
        const unsigned int qw3 = *reinterpret_cast<const unsigned int*>(q + 12);
        float wv[16];
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            const unsigned int word =
                (l < 4) ? qw0 : (l < 8) ? qw1 : (l < 12) ? qw2 : qw3;
            // byte l at bit-shift `shift` (one code per byte position)
            wv[l] = dl * (float)((word >> (8 * (l & 3) + shift)) & 3) - ml;
        }
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            const int r = r0 + t;
            if (r < rows) {
                const float4 xa = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16);
                const float4 xb = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16 + 4);
                const float4 xc = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16 + 8);
                const float4 xd = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16 + 12);
                acc[t] += wv[0]*xa.x + wv[1]*xa.y + wv[2]*xa.z + wv[3]*xa.w
                        + wv[4]*xb.x + wv[5]*xb.y + wv[6]*xb.z + wv[7]*xb.w
                        + wv[8]*xc.x + wv[9]*xc.y + wv[10]*xc.z + wv[11]*xc.w
                        + wv[12]*xd.x + wv[13]*xd.y + wv[14]*xd.z + wv[15]*xd.w;
            }
        }
    }
    dfl_rows_reduce(y, out_dim, o, tid, r0, rows, acc);
}

// Q3_K: 110 B / 256 elems, uploaded PADDED to a 112-byte stride (2 zero
// bytes per block) so every u32 load is naturally aligned (110 ≡ 2 mod 4 —
// the pad is load-time only, never touches the artifact). Layout per ggml
// block_q3_K: hmask[32] @0, qs[64] @32, scales[12] @96, d @108. The 12
// scale bytes unpack into 16 6-bit values via the kmask u32 interleave
// (sc − 32 signed); the code is (qs&3) − (hbit ? 0 : 4) — 3-bit signed.
extern "C" __global__ void dflash2_gemv_q3k_rows(
    const unsigned char* __restrict__ w,   // [out_dim][bpr] * 112 (padded)
    const float* __restrict__ x,
    float* __restrict__ y,
    const int out_dim, const int in_dim, const int rows)
{
    const int o = blockIdx.x;
    const int r0 = blockIdx.y * 8;
    const int tid = threadIdx.x;
    const int bpr = in_dim >> 8;
    const unsigned char* wrow = w + (size_t)o * (size_t)bpr * 112;
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.f;
    const unsigned int kmask1 = 0x03030303u;
    const unsigned int kmask2 = 0x0f0f0f0fu;
    const int nsb = in_dim >> 4;
    for (int sb = tid; sb < nsb; sb += blockDim.x) {
        const int blk = sb >> 4;
        const int j16 = sb & 15;
        const unsigned char* b = wrow + (size_t)blk * 112;
        const float d_all = dfl_f16_f32(*reinterpret_cast<const unsigned short*>(b + 108));
        const unsigned int x0 = *reinterpret_cast<const unsigned int*>(b + 96);
        const unsigned int x1 = *reinterpret_cast<const unsigned int*>(b + 100);
        const unsigned int tmp = *reinterpret_cast<const unsigned int*>(b + 104);
        const unsigned int a2 = ((x0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        const unsigned int a3 = ((x1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        const unsigned int a0 = (x0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        const unsigned int a1 = (x1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const unsigned int sc_word =
            (j16 < 4) ? a0 : (j16 < 8) ? a1 : (j16 < 12) ? a2 : a3;
        const unsigned int sc_raw = (sc_word >> (8 * (j16 & 3))) & 0xFFu;
        const float dl = d_all * (float)((int)sc_raw - 32);
        const unsigned char* q = b + 32 + ((j16 >> 3) << 5) + ((j16 & 1) << 4);
        const unsigned char* hm = b + ((j16 & 1) << 4);
        const int hbit = ((j16 >> 3) << 2) + ((j16 & 7) >> 1);
        const int shift = j16 & 6;
        const unsigned int qw0 = *reinterpret_cast<const unsigned int*>(q);
        const unsigned int qw1 = *reinterpret_cast<const unsigned int*>(q + 4);
        const unsigned int qw2 = *reinterpret_cast<const unsigned int*>(q + 8);
        const unsigned int qw3 = *reinterpret_cast<const unsigned int*>(q + 12);
        float wv[16];
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            const unsigned int word =
                (l < 4) ? qw0 : (l < 8) ? qw1 : (l < 12) ? qw2 : qw3;
            const int code = (int)((word >> (8 * (l & 3) + shift)) & 3)
                           - (((hm[l] >> hbit) & 1) ? 0 : 4);
            wv[l] = dl * (float)code;
        }
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            const int r = r0 + t;
            if (r < rows) {
                const float4 xa = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16);
                const float4 xb = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16 + 4);
                const float4 xc = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16 + 8);
                const float4 xd = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 16 + 12);
                acc[t] += wv[0]*xa.x + wv[1]*xa.y + wv[2]*xa.z + wv[3]*xa.w
                        + wv[4]*xb.x + wv[5]*xb.y + wv[6]*xb.z + wv[7]*xb.w
                        + wv[8]*xc.x + wv[9]*xc.y + wv[10]*xc.z + wv[11]*xc.w
                        + wv[12]*xd.x + wv[13]*xd.y + wv[14]*xd.z + wv[15]*xd.w;
            }
        }
    }
    dfl_rows_reduce(y, out_dim, o, tid, r0, rows, acc);
}

// Q4_K: 144 B / 256 elems — layout per ggml block_q4_K (identical to the
// in-tree qwen38_dequant_q4k_row mapping): d @0, dmin @2, scales[12] @4
// (6-bit k4 packing), qs[128] @16. EIGHT 32-element sub-blocks per block:
// scale j from the 6-bit fields, nibbles lo (even j) / hi (odd j).
extern "C" __global__ void dflash2_gemv_q4k_rows(
    const unsigned char* __restrict__ w,   // [out_dim][bpr] * 144
    const float* __restrict__ x,
    float* __restrict__ y,
    const int out_dim, const int in_dim, const int rows)
{
    const int o = blockIdx.x;
    const int r0 = blockIdx.y * 8;
    const int tid = threadIdx.x;
    const int bpr = in_dim >> 8;
    const unsigned char* wrow = w + (size_t)o * (size_t)bpr * 144;
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.f;
    const int nsb = in_dim >> 5;   // 32-element sub-blocks (8 per block)
    for (int sb = tid; sb < nsb; sb += blockDim.x) {
        const int blk = sb >> 3;
        const int j = sb & 7;
        const unsigned char* b = wrow + (size_t)blk * 144;
        const float d = dfl_f16_f32(*reinterpret_cast<const unsigned short*>(b));
        const float dmin = dfl_f16_f32(*reinterpret_cast<const unsigned short*>(b + 2));
        const unsigned char* s = b + 4;
        float sc, mn;
        if (j < 4) {
            sc = (float)(s[j] & 63);
            mn = (float)(s[j + 4] & 63);
        } else {
            sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
            mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        }
        const float dl = d * sc;
        const float ml = dmin * mn;
        const unsigned char* q = b + 16 + ((j >> 1) << 5);   // 32 bytes
        const int nib_shift = (j & 1) ? 4 : 0;
        float wv[32];
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            wv[l] = dl * (float)((q[l] >> nib_shift) & 0x0F) - ml;
        }
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            const int r = r0 + t;
            if (r < rows) {
                const float4 xv0 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32);
                const float4 xv1 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 4);
                const float4 xv2 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 8);
                const float4 xv3 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 12);
                const float4 xv4 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 16);
                const float4 xv5 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 20);
                const float4 xv6 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 24);
                const float4 xv7 = *reinterpret_cast<const float4*>(x + (size_t)r * in_dim + sb * 32 + 28);
                acc[t] += wv[0]*xv0.x + wv[1]*xv0.y + wv[2]*xv0.z + wv[3]*xv0.w
                        + wv[4]*xv1.x + wv[5]*xv1.y + wv[6]*xv1.z + wv[7]*xv1.w
                        + wv[8]*xv2.x + wv[9]*xv2.y + wv[10]*xv2.z + wv[11]*xv2.w
                        + wv[12]*xv3.x + wv[13]*xv3.y + wv[14]*xv3.z + wv[15]*xv3.w
                        + wv[16]*xv4.x + wv[17]*xv4.y + wv[18]*xv4.z + wv[19]*xv4.w
                        + wv[20]*xv5.x + wv[21]*xv5.y + wv[22]*xv5.z + wv[23]*xv5.w
                        + wv[24]*xv6.x + wv[25]*xv6.y + wv[26]*xv6.z + wv[27]*xv6.w
                        + wv[28]*xv7.x + wv[29]*xv7.y + wv[30]*xv7.z + wv[31]*xv7.w;
            }
        }
    }
    dfl_rows_reduce(y, out_dim, o, tid, r0, rows, acc);
}
"#;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal raw-BF16 safetensors reader (streams one tensor at a time; the
// CPU oracle's reader is private + returns f32 — provenance note in the
// module doc).
// ─────────────────────────────────────────────────────────────────────────────

struct StRawFile {
    file: std::fs::File,
    index: std::collections::HashMap<String, (String, Vec<usize>, u64)>,
}

impl StRawFile {
    fn open(path: &Path) -> Result<Self, String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| format!("read header len: {e}"))?;
        let hlen = u64::from_le_bytes(len_buf) as usize;
        let mut hdr = vec![0u8; hlen];
        file.read_exact(&mut hdr)
            .map_err(|e| format!("read header: {e}"))?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&hdr).map_err(|e| format!("parse header json: {e}"))?;
        let obj = parsed
            .as_object()
            .ok_or_else(|| "header not an object".to_string())?;
        let data_base = 8 + hlen as u64;
        let mut index = std::collections::HashMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{name}: no dtype"))?
                .to_string();
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("{name}: no shape"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let beg = meta
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            index.insert(name.clone(), (dtype, shape, data_base + beg));
        }
        Ok(Self { file, index })
    }

    fn expect(&self, name: &str, want: &[usize]) -> Result<(), String> {
        let got = self
            .index
            .get(name)
            .map(|(_, s, _)| s.clone())
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        if got != want {
            return Err(format!("{name}: shape {got:?} != expected {want:?}"));
        }
        Ok(())
    }

    /// Read one tensor as raw bf16 bits (`u16`). F32 tensors convert via
    /// RNE (all weights in the z-lab artifact are BF16, where the
    /// round-trip is exact).
    fn read_bf16(&mut self, name: &str) -> Result<Vec<u16>, String> {
        use std::io::{Read, Seek, SeekFrom};
        let (dtype, shape, off) = self
            .index
            .get(name)
            .cloned()
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        let n: usize = shape.iter().product();
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek {name}: {e}"))?;
        match dtype.as_str() {
            "BF16" => {
                let mut raw = vec![0u8; n * 2];
                self.file
                    .read_exact(&mut raw)
                    .map_err(|e| format!("read {name}: {e}"))?;
                Ok(raw
                    .as_chunks::<2>().0.iter()
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect())
            }
            "F32" => {
                let mut raw = vec![0u8; n * 4];
                self.file
                    .read_exact(&mut raw)
                    .map_err(|e| format!("read {name}: {e}"))?;
                Ok(raw
                    .as_chunks::<4>().0.iter()
                    .map(|c| {
                        f32_to_bf16_bits(f32::from_bits(u32::from_le_bytes([
                            c[0], c[1], c[2], c[3],
                        ])))
                    })
                    .collect())
            }
            other => Err(format!("{name}: unsupported dtype {other}")),
        }
    }

    /// Read one tensor widened to f32 (bf16 widen is exact — this is what
    /// the CPU oracle's `read_f32` does).
    fn read_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        Ok(self
            .read_bf16(name)?
            .into_iter()
            .map(|h| f32::from_bits((h as u32) << 16))
            .collect())
    }
}

/// f32 -> bf16 bits, round-to-nearest-even. For values whose f32 bits came
/// FROM a bf16 widen (every weight here) the round-trip is exact: the low
/// 16 bits are zero so the rounding add cannot carry.
fn f32_to_bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return ((v.to_bits() >> 16) as u16) | 0x0040;
    }
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7FFF + lsb) >> 16) as u16
}

// ─────────────────────────────────────────────────────────────────────────────
// Device weights
// ─────────────────────────────────────────────────────────────────────────────

struct GpuLayer {
    attn_norm: CudaSlice<f32>,
    q_proj: QW,
    k_proj: QW,
    v_proj: QW,
    o_proj: QW,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    gate_proj: QW,
    up_proj: QW,
    down_proj: QW,
    attn_conv_base: CudaSlice<f32>,
    attn_conv_proj: QW,
    ffn_conv_base: CudaSlice<f32>,
    ffn_conv_proj: QW,
}

/// A 2D drafter weight matrix — BF16 (the z-lab safetensors path) or one of
/// the artifact's K-quants (the Issue-780-U3 quant-resident path: Q2_K
/// majority + Q3_K attn_output/ffn_down + Q4_K attn_v, all dequantized
/// in-kernel by the `dflash2_gemv_q{k}_rows` family).
enum QW {
    Bf16(CudaSlice<u16>),
    /// Q2_K raw blocks, 84-byte stride.
    Q2K(CudaSlice<u8>),
    /// Q3_K blocks PADDED to a 112-byte stride at upload (110 ≡ 2 mod 4 —
    /// the pad keeps the kernel's u32 loads naturally aligned; it never
    /// touches the artifact on disk).
    Q3K(CudaSlice<u8>),
    /// Q4_K raw blocks, 144-byte stride.
    Q4K(CudaSlice<u8>),
}

/// The GPU DFlash2 drafter — see the module doc.
pub struct DFlash2GpuDrafter {
    pub cfg: DFlash2Config,
    stream: Arc<CudaStream>,
    // kernels
    gemv: CudaFunction,
    gemv_q2k: CudaFunction,
    gemv_q3k: CudaFunction,
    gemv_q4k: CudaFunction,
    rmsnorm_rows: CudaFunction,
    qk_rope: CudaFunction,
    attention: CudaFunction,
    conv: CudaFunction,
    scatter: CudaFunction,
    residual_add: CudaFunction,
    swiglu: CudaFunction,
    _module: Arc<CudaModule>,
    // weights
    fc: QW,
    hidden_norm: CudaSlice<f32>,
    output_norm: CudaSlice<f32>,
    layers: Vec<GpuLayer>,
    // rings (device K/V + host occupancy, mirroring the CPU drafter)
    ring_k: Vec<CudaSlice<f32>>,
    ring_v: Vec<CudaSlice<f32>>,
    ring_pos: VecDeque<usize>,
    rope_freqs: CudaSlice<f32>,
    ring_pos_dev: CudaSlice<i32>,
    // draft scratch [8][dim] family
    x: CudaSlice<f32>,
    h: CudaSlice<f32>,
    hc: CudaSlice<f32>,
    dyn_coeff: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    ao: CudaSlice<f32>,
    attn_final: CudaSlice<f32>,
    ffn_inp: CudaSlice<f32>,
    hf: CudaSlice<f32>,
    dynf: CudaSlice<f32>,
    hfc: CudaSlice<f32>,
    gate_out: CudaSlice<f32>,
    up_out: CudaSlice<f32>,
    mid: CudaSlice<f32>,
    down_out: CudaSlice<f32>,
    ffn_final: CudaSlice<f32>,
    out_rows: CudaSlice<f32>,
    // inject scratch (8-row chunks)
    inj_x: CudaSlice<f32>,
    inj_g: CudaSlice<f32>,
    inj_gn: CudaSlice<f32>,
    inj_k: CudaSlice<f32>,
    inj_v: CudaSlice<f32>,
    inj_slots: CudaSlice<i32>,
    inj_x_host: Vec<f32>,
    inj_slots_host: Vec<i32>,
    ring_pos_host: Vec<i32>,
}

impl DFlash2GpuDrafter {
    /// Load the z-lab DFlash2 safetensors (BF16) onto the given stream,
    /// validate every shape against `cfg` (the CPU loader's subset — the
    /// selector tensors stay CPU), allocate rings + scratch. GPU-heavy
    /// (~3.6 GB upload); assumes an idle GPU.
    pub fn new(stream: Arc<CudaStream>, cfg: DFlash2Config, path: &str) -> Result<Self, String> {
        let mut st = StRawFile::open(Path::new(path))?;
        let e = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let inp = 5 * e;

        // Shape validation (the CPU loader's list minus the selector).
        st.expect("fc.weight", &[e, inp])?;
        st.expect("hidden_norm.weight", &[e])?;
        st.expect("norm.weight", &[e])?;
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            st.expect(&p("input_layernorm.weight"), &[e])?;
            st.expect(&p("self_attn.q_proj.weight"), &[qd, e])?;
            st.expect(&p("self_attn.k_proj.weight"), &[kvd, e])?;
            st.expect(&p("self_attn.v_proj.weight"), &[kvd, e])?;
            st.expect(&p("self_attn.o_proj.weight"), &[e, qd])?;
            st.expect(&p("self_attn.q_norm.weight"), &[cfg.head_dim])?;
            st.expect(&p("self_attn.k_norm.weight"), &[cfg.head_dim])?;
            st.expect(&p("post_attention_layernorm.weight"), &[e])?;
            st.expect(&p("mlp.gate_proj.weight"), &[ff, e])?;
            st.expect(&p("mlp.up_proj.weight"), &[ff, e])?;
            st.expect(&p("mlp.down_proj.weight"), &[e, ff])?;
            st.expect(&p("attention_conv.base_kernel"), &[2, 2, e])?;
            st.expect(&p("attention_conv.kernel_projection.weight"), &[1280, e])?;
            st.expect(&p("mlp_conv.base_kernel"), &[2, 2, e])?;
            st.expect(&p("mlp_conv.kernel_projection.weight"), &[1280, e])?;
        }

        // Stream tensors: read one -> upload -> drop the host copy.
        let up_u16 = |st: &mut StRawFile, name: &str| -> Result<QW, String> {
            let host = st.read_bf16(name)?;
            let mut dev = stream
                .alloc_zeros::<u16>(host.len())
                .map_err(|er| format!("alloc {name}: {er}"))?;
            stream
                .memcpy_htod(&host, &mut dev)
                .map_err(|er| format!("upload {name}: {er}"))?;
            Ok(QW::Bf16(dev))
        };
        let up_f32 = |st: &mut StRawFile, name: &str| -> Result<CudaSlice<f32>, String> {
            let host = st.read_f32(name)?;
            let mut dev = stream
                .alloc_zeros::<f32>(host.len())
                .map_err(|er| format!("alloc {name}: {er}"))?;
            stream
                .memcpy_htod(&host, &mut dev)
                .map_err(|er| format!("upload {name}: {er}"))?;
            Ok(dev)
        };

        let fc = up_u16(&mut st, "fc.weight")?;
        let hidden_norm = up_f32(&mut st, "hidden_norm.weight")?;
        let output_norm = up_f32(&mut st, "norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(GpuLayer {
                attn_norm: up_f32(&mut st, &p("input_layernorm.weight"))?,
                q_proj: up_u16(&mut st, &p("self_attn.q_proj.weight"))?,
                k_proj: up_u16(&mut st, &p("self_attn.k_proj.weight"))?,
                v_proj: up_u16(&mut st, &p("self_attn.v_proj.weight"))?,
                o_proj: up_u16(&mut st, &p("self_attn.o_proj.weight"))?,
                q_norm: up_f32(&mut st, &p("self_attn.q_norm.weight"))?,
                k_norm: up_f32(&mut st, &p("self_attn.k_norm.weight"))?,
                ffn_norm: up_f32(&mut st, &p("post_attention_layernorm.weight"))?,
                gate_proj: up_u16(&mut st, &p("mlp.gate_proj.weight"))?,
                up_proj: up_u16(&mut st, &p("mlp.up_proj.weight"))?,
                down_proj: up_u16(&mut st, &p("mlp.down_proj.weight"))?,
                attn_conv_base: up_f32(&mut st, &p("attention_conv.base_kernel"))?,
                attn_conv_proj: up_u16(&mut st, &p("attention_conv.kernel_projection.weight"))?,
                ffn_conv_base: up_f32(&mut st, &p("mlp_conv.base_kernel"))?,
                ffn_conv_proj: up_u16(&mut st, &p("mlp_conv.kernel_projection.weight"))?,
            });
        }
        Self::build_tail(stream, cfg, fc, hidden_norm, output_norm, layers)
    }

    /// Load the analogalok DFlash2 GGUF (the mixed-quant incumbent artifact)
    /// QUANT-RESIDENT — Issue 780 U3 wall-1 lever (a): ~0.65 GB of K-quant
    /// blocks on device vs 3.52 GB of BF16 (−2.87 GB VRAM, ~5.4× fewer
    /// weight bytes per draft cycle; the in-kernel dequant is the
    /// `dflash2_gemv_q{k}_rows` family, pinned against the host reference
    /// by the kernel-vs-host cosine gate). The artifact's per-tensor mix
    /// (Q2_K majority + Q3_K attn_output/ffn_down + Q4_K attn_v) is accepted
    /// as-found — each matrix dispatches on its own type. Norms/conv-bases
    /// are F32 in the artifact and upload as-is; the SELECTOR stays CPU
    /// (BF16 safetensors, the caller's `DFlash2Drafter` path — unchanged,
    /// isolating the quant effect to the block forward).
    ///
    /// Tensor names follow the fork's GGUF convention (`blk.N.attn_q.weight`
    /// etc.); shapes validate as GGUF ne `[in_dim, out_dim]` (ne[0] is the
    /// contiguous dim — row-major `out_dim` rows of `in_dim`, the same
    /// layout the Q4_K target path consumes).
    ///
    /// Two artifact name conventions exist in the wild (both measured on
    /// disk, Issue 780 U3 wall-1 (a)): the `hidden_norm.weight` convention
    /// AND the analogalok artifact's `enc.output_norm.weight` — the same
    /// encoder-exit norm under either name. Every other tensor name is
    /// identical across both artifacts; the selector planes the analogalok
    /// file carries (`selector_hidden/predecessor/successor.weight`) are
    /// deliberately ignored here (the selector stays CPU — the quant effect
    /// stays isolated to the block forward).
    pub fn new_from_gguf(
        stream: Arc<CudaStream>,
        cfg: DFlash2Config,
        path: &str,
    ) -> Result<Self, String> {
        let gguf =
            GgufFile::open(Path::new(path)).map_err(|e| format!("open {path}: {e}"))?;
        let e = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let inp = 5 * e;

        let up_q = |name: &str, out_d: usize, in_d: usize| -> Result<QW, String> {
            let info = gguf
                .tensor_info(name)
                .ok_or_else(|| format!("tensor '{name}' not found"))?;
            if info.shape.len() != 2 || info.shape[0] != in_d || info.shape[1] != out_d {
                return Err(format!(
                    "tensor '{name}' shape {:?} != GGUF ne [{in_d}(in), {out_d}(out)]",
                    info.shape
                ));
            }
            let bytes = gguf
                .tensor_slice(name)
                .ok_or_else(|| format!("tensor '{name}' has no data"))?;
            let bpr = in_d / 256;
            match info.ggml_type {
                GgmlType::Q2_K => {
                    let want = bpr * out_d * 84;
                    if info.byte_len != want as u64 {
                        return Err(format!(
                            "tensor '{name}' byte_len {} != {want} (Q2_K)",
                            info.byte_len
                        ));
                    }
                    let mut dev = stream
                        .alloc_zeros::<u8>(want)
                        .map_err(|er| format!("alloc {name}: {er}"))?;
                    stream
                        .memcpy_htod(bytes, &mut dev)
                        .map_err(|er| format!("upload {name}: {er}"))?;
                    Ok(QW::Q2K(dev))
                }
                GgmlType::Q4_K => {
                    let want = bpr * out_d * 144;
                    if info.byte_len != want as u64 {
                        return Err(format!(
                            "tensor '{name}' byte_len {} != {want} (Q4_K)",
                            info.byte_len
                        ));
                    }
                    let mut dev = stream
                        .alloc_zeros::<u8>(want)
                        .map_err(|er| format!("alloc {name}: {er}"))?;
                    stream
                        .memcpy_htod(bytes, &mut dev)
                        .map_err(|er| format!("upload {name}: {er}"))?;
                    Ok(QW::Q4K(dev))
                }
                GgmlType::Q3_K => {
                    let raw_want = bpr * out_d * 110;
                    if info.byte_len != raw_want as u64 {
                        return Err(format!(
                            "tensor '{name}' byte_len {} != {raw_want} (Q3_K)",
                            info.byte_len
                        ));
                    }
                    // Pad 110 -> 112 stride (u32 alignment — see QW::Q3K).
                    let n_blocks = bpr * out_d;
                    let mut padded = vec![0u8; n_blocks * 112];
                    for b in 0..n_blocks {
                        padded[b * 112..b * 112 + 110]
                            .copy_from_slice(&bytes[b * 110..b * 110 + 110]);
                    }
                    let mut dev = stream
                        .alloc_zeros::<u8>(padded.len())
                        .map_err(|er| format!("alloc {name}: {er}"))?;
                    stream
                        .memcpy_htod(&padded, &mut dev)
                        .map_err(|er| format!("upload {name}: {er}"))?;
                    Ok(QW::Q3K(dev))
                }
                t => Err(format!(
                    "tensor '{name}' is {t:?}, expected Q2_K/Q3_K/Q4_K"
                )),
            }
        };
        let up_f32g = |name: &str, len: usize| -> Result<CudaSlice<f32>, String> {
            let info = gguf
                .tensor_info(name)
                .ok_or_else(|| format!("tensor '{name}' not found"))?;
            if info.ggml_type != GgmlType::F32 {
                return Err(format!(
                    "tensor '{name}' is {:?}, expected F32",
                    info.ggml_type
                ));
            }
            if info.n_elements() != len {
                return Err(format!(
                    "tensor '{name}' n_elements {} != {len}",
                    info.n_elements()
                ));
            }
            let bytes = gguf
                .tensor_slice(name)
                .ok_or_else(|| format!("tensor '{name}' has no data"))?;
            let host: Vec<f32> = bytemuck::cast_slice(bytes).to_vec();
            let mut dev = stream
                .alloc_zeros::<f32>(host.len())
                .map_err(|er| format!("alloc {name}: {er}"))?;
            stream
                .memcpy_htod(&host, &mut dev)
                .map_err(|er| format!("upload {name}: {er}"))?;
            Ok(dev)
        };

        let fc = up_q("fc.weight", e, inp)?;
        // The encoder-exit norm under either artifact convention (see the
        // doc above): `hidden_norm.weight` OR `enc.output_norm.weight`.
        let hidden_norm_name = ["hidden_norm.weight", "enc.output_norm.weight"]
            .into_iter()
            .find(|n| gguf.tensor_info(n).is_some())
            .ok_or_else(|| {
                "tensor 'hidden_norm.weight' (or 'enc.output_norm.weight') not found".to_string()
            })?;
        let hidden_norm = up_f32g(hidden_norm_name, e)?;
        let output_norm = up_f32g("output_norm.weight", e)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            layers.push(GpuLayer {
                attn_norm: up_f32g(&p("attn_norm.weight"), e)?,
                q_proj: up_q(&p("attn_q.weight"), qd, e)?,
                k_proj: up_q(&p("attn_k.weight"), kvd, e)?,
                v_proj: up_q(&p("attn_v.weight"), kvd, e)?,
                o_proj: up_q(&p("attn_output.weight"), e, qd)?,
                q_norm: up_f32g(&p("attn_q_norm.weight"), cfg.head_dim)?,
                k_norm: up_f32g(&p("attn_k_norm.weight"), cfg.head_dim)?,
                ffn_norm: up_f32g(&p("ffn_norm.weight"), e)?,
                gate_proj: up_q(&p("ffn_gate.weight"), ff, e)?,
                up_proj: up_q(&p("ffn_up.weight"), ff, e)?,
                down_proj: up_q(&p("ffn_down.weight"), e, ff)?,
                attn_conv_base: up_f32g(&p("attn_conv_base"), 2 * 2 * e)?,
                attn_conv_proj: up_q(&p("attn_conv_proj.weight"), 1280, e)?,
                ffn_conv_base: up_f32g(&p("ffn_conv_base"), 2 * 2 * e)?,
                ffn_conv_proj: up_q(&p("ffn_conv_proj.weight"), 1280, e)?,
            });
        }
        Self::build_tail(stream, cfg, fc, hidden_norm, output_norm, layers)
    }

    /// Common constructor tail: module + kernels + rings + scratch.
    fn build_tail(
        stream: Arc<CudaStream>,
        cfg: DFlash2Config,
        fc: QW,
        hidden_norm: CudaSlice<f32>,
        output_norm: CudaSlice<f32>,
        layers: Vec<GpuLayer>,
    ) -> Result<Self, String> {
        // Context for module load: the stream's context (cudarc exposes it
        // via the module-load path through the stream's ctx handle).
        let ctx = stream_ctx(&stream)?;
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            DFLASH2_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let f = |name: &str| module.load_function(name).map_err(|e| format!("{name}: {e}"));

        let e = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let inp = 5 * e;

        // Rings + rope freqs.
        let swa = cfg.sliding_window;
        let n_kv = cfg.n_kv_head * cfg.head_dim;
        let mut ring_k = Vec::with_capacity(cfg.n_layer);
        let mut ring_v = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            ring_k.push(
                stream
                    .alloc_zeros::<f32>(swa * n_kv)
                    .map_err(|er| format!("ring_k alloc: {er}"))?,
            );
            ring_v.push(
                stream
                    .alloc_zeros::<f32>(swa * n_kv)
                    .map_err(|er| format!("ring_v alloc: {er}"))?,
            );
        }
        let mut rope_freqs_host = vec![0.0f32; cfg.head_dim / 2];
        for (i, fr) in rope_freqs_host.iter_mut().enumerate() {
            *fr = cfg.rope_theta.powf(-(2.0 * i as f32) / cfg.head_dim as f32);
        }
        let mut rope_freqs = stream
            .alloc_zeros::<f32>(rope_freqs_host.len())
            .map_err(|er| format!("rope alloc: {er}"))?;
        stream
            .memcpy_htod(&rope_freqs_host, &mut rope_freqs)
            .map_err(|er| format!("rope upload: {er}"))?;
        let mut ring_pos_dev = stream
            .alloc_zeros::<i32>(swa)
            .map_err(|er| format!("ring_pos alloc: {er}"))?;
        // -1 = empty (the kernel skips rp < 0).
        let pad = vec![-1i32; swa];
        stream
            .memcpy_htod(&pad, &mut ring_pos_dev)
            .map_err(|er| format!("ring_pos init: {er}"))?;

        // Scratch.
        let bs = cfg.block_size;
        let projected = 2 * cfg.conv_kernel * (e / cfg.conv_group);
        macro_rules! z {
            ($len:expr) => {
                stream
                    .alloc_zeros::<f32>($len)
                    .map_err(|er| format!("scratch alloc: {er}"))?
            };
        }
        let x = z!(bs * e);
        let h = z!(bs * e);
        let hc = z!(bs * e);
        let dyn_coeff = z!(bs * projected);
        let q = z!(bs * qd);
        let k = z!(bs * kvd);
        let v = z!(bs * kvd);
        let attn_out = z!(bs * qd);
        let ao = z!(bs * e);
        let attn_final = z!(bs * e);
        let ffn_inp = z!(bs * e);
        let hf = z!(bs * e);
        let dynf = z!(bs * projected);
        let hfc = z!(bs * e);
        let gate_out = z!(bs * ff);
        let up_out = z!(bs * ff);
        let mid = z!(bs * ff);
        let down_out = z!(bs * e);
        let ffn_final = z!(bs * e);
        let out_rows = z!(bs * e);
        let inj_x = z!(DFLASH2_TILE * inp);
        let inj_g = z!(DFLASH2_TILE * e);
        let inj_gn = z!(DFLASH2_TILE * e);
        let inj_k = z!(DFLASH2_TILE * kvd);
        let inj_v = z!(DFLASH2_TILE * kvd);
        let mut inj_slots = stream
            .alloc_zeros::<i32>(DFLASH2_TILE)
            .map_err(|er| format!("inj_slots alloc: {er}"))?;
        let inj_slots_host = vec![0i32; DFLASH2_TILE];
        stream
            .memcpy_htod(&inj_slots_host, &mut inj_slots)
            .map_err(|er| format!("inj_slots init: {er}"))?;

        Ok(Self {
            cfg,
            stream,
            gemv: f("dflash2_gemv_bf16_rows")?,
            gemv_q2k: f("dflash2_gemv_q2k_rows")?,
            gemv_q3k: f("dflash2_gemv_q3k_rows")?,
            gemv_q4k: f("dflash2_gemv_q4k_rows")?,
            rmsnorm_rows: f("dflash2_rmsnorm_rows_f32")?,
            qk_rope: f("dflash2_qk_norm_rope_neox")?,
            attention: f("dflash2_attention")?,
            conv: f("dflash2_conv")?,
            scatter: f("dflash2_scatter_ring")?,
            residual_add: f("dflash2_residual_add")?,
            swiglu: f("dflash2_swiglu")?,
            _module: module,
            fc,
            hidden_norm,
            output_norm,
            layers,
            ring_k,
            ring_v,
            ring_pos: VecDeque::with_capacity(swa + DFLASH2_TILE),
            rope_freqs,
            ring_pos_dev,
            x,
            h,
            hc,
            dyn_coeff,
            q,
            k,
            v,
            attn_out,
            ao,
            attn_final,
            ffn_inp,
            hf,
            dynf,
            hfc,
            gate_out,
            up_out,
            mid,
            down_out,
            ffn_final,
            out_rows,
            inj_x,
            inj_g,
            inj_gn,
            inj_k,
            inj_v,
            inj_slots,
            inj_x_host: vec![0.0f32; DFLASH2_TILE * inp],
            inj_slots_host,
            ring_pos_host: Vec::with_capacity(swa),
        })
    }

    /// Reset the ring for a fresh sequence (occupancy only — slots are
    /// rewritten on injection; mirrors the CPU `reset_ring`).
    pub fn reset_ring(&mut self) {
        self.ring_pos.clear();
    }

    /// Device bytes held by this drafter (weights + rings + scratch) —
    /// diagnostic/reporting helper.
    pub fn device_bytes(&self) -> usize {
        let e = self.cfg.n_embd;
        let ff = self.cfg.n_ff;
        let qd = self.cfg.n_head * self.cfg.head_dim;
        let kvd = self.cfg.n_kv_head * self.cfg.head_dim;
        let projected = 2 * self.cfg.conv_kernel * (e / self.cfg.conv_group);
        let bs = self.cfg.block_size;
        // Q2_K/Q3_K tensors are byte-counted directly (the Q3K upload pads
        // 110 -> 112 — the reported number includes that pad, honestly).
        let qw = |q: &QW| -> usize {
            match q {
                QW::Bf16(s) => s.len() * 2,
                QW::Q2K(s) | QW::Q3K(s) | QW::Q4K(s) => s.len(),
            }
        };
        let mut gemm = qw(&self.fc);
        gemm += self.hidden_norm.len() * 4 + self.output_norm.len() * 4;
        for l in &self.layers {
            gemm += qw(&l.q_proj)
                + qw(&l.k_proj)
                + qw(&l.v_proj)
                + qw(&l.o_proj)
                + qw(&l.gate_proj)
                + qw(&l.up_proj)
                + qw(&l.down_proj)
                + qw(&l.attn_conv_proj)
                + qw(&l.ffn_conv_proj)
                + l.attn_norm.len() * 4
                + l.q_norm.len() * 4
                + l.k_norm.len() * 4
                + l.ffn_norm.len() * 4
                + l.attn_conv_base.len() * 4
                + l.ffn_conv_base.len() * 4;
        }
        let rings = self.ring_k.len() * self.ring_k[0].len() * 4 * 2;
        let scratch = (self.x.len() + self.h.len() + self.hc.len() + self.dyn_coeff.len()
            + self.q.len() + self.k.len() + self.v.len() + self.attn_out.len()
            + self.ao.len() + self.attn_final.len() + self.ffn_inp.len() + self.hf.len()
            + self.dynf.len() + self.hfc.len() + self.gate_out.len() + self.up_out.len()
            + self.mid.len() + self.down_out.len() + self.ffn_final.len()
            + self.out_rows.len())
            * 4
            + (self.inj_x.len() + self.inj_g.len() + self.inj_gn.len() + self.inj_k.len()
                + self.inj_v.len())
                * 4
            + self.inj_slots.len() * 4
            + self.ring_pos_dev.len() * 4
            + self.rope_freqs.len() * 4;
        let _ = (ff, qd, projected, bs, kvd);
        gemm + rings + scratch
    }

    /// Download one layer's K/V ring (test/diagnostic accessor — the G1
    /// batch-vs-single parity gate reads the ring bits back).
    pub fn download_ring(&self, li: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        if li >= self.ring_k.len() {
            return Err(format!("ring layer {li} out of range"));
        }
        let mut k = vec![0.0f32; self.ring_k[li].len()];
        self.stream
            .memcpy_dtoh(&self.ring_k[li], &mut k)
            .map_err(|e| e.to_string())?;
        let mut v = vec![0.0f32; self.ring_v[li].len()];
        self.stream
            .memcpy_dtoh(&self.ring_v[li], &mut v)
            .map_err(|e| e.to_string())?;
        Ok((k, v))
    }

    /// Ring occupancy (host-side truth, mirroring the CPU drafter).
    pub fn ring_positions(&self) -> Vec<usize> {
        self.ring_pos.iter().copied().collect()
    }

    // ── launch helpers ──

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_gemv(
        &self,
        w: &QW,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
        rows: usize,
    ) -> Result<(), String> {
        let (od, id, rws) = (out_dim as i32, in_dim as i32, rows as i32);
        assert!(
            in_dim.is_multiple_of(256),
            "dflash2 rows-GEMV needs in_dim % 256 == 0 (got {in_dim})"
        );
        let grid_y = rows.div_ceil(DFLASH2_TILE) as u32;
        let lc = || LaunchConfig {
            grid_dim: (out_dim as u32, grid_y, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: caller guarantees extents cover the kernel's accesses
        // (weights cover out_dim × in_dim in the QW's block format; x covers
        // rows × in_dim; y covers rows × out_dim).
        unsafe {
            match w {
                QW::Bf16(s) => {
                    assert!(
                        in_dim.is_multiple_of(1024),
                        "bf16 vectorized loads need in_dim % 1024 == 0 (got {in_dim})"
                    );
                    self.stream
                        .launch_builder(&self.gemv)
                        .arg(s)
                        .arg(x)
                        .arg(y)
                        .arg(&od)
                        .arg(&id)
                        .arg(&rws)
                        .launch(lc())
                        .map_err(|e| e.to_string())?;
                }
                QW::Q2K(s) => {
                    self.stream
                        .launch_builder(&self.gemv_q2k)
                        .arg(s)
                        .arg(x)
                        .arg(y)
                        .arg(&od)
                        .arg(&id)
                        .arg(&rws)
                        .launch(lc())
                        .map_err(|e| e.to_string())?;
                }
                QW::Q3K(s) => {
                    self.stream
                        .launch_builder(&self.gemv_q3k)
                        .arg(s)
                        .arg(x)
                        .arg(y)
                        .arg(&od)
                        .arg(&id)
                        .arg(&rws)
                        .launch(lc())
                        .map_err(|e| e.to_string())?;
                }
                QW::Q4K(s) => {
                    self.stream
                        .launch_builder(&self.gemv_q4k)
                        .arg(s)
                        .arg(x)
                        .arg(y)
                        .arg(&od)
                        .arg(&id)
                        .arg(&rws)
                        .launch(lc())
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }

    unsafe fn launch_rmsnorm_rows(
        &self,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
    ) -> Result<(), String> {
        let (eps, dim_i) = (self.cfg.rms_eps, dim as i32);
        // SAFETY: extents cover rows*dim for each buffer, gamma covers dim.
        unsafe {
            self.stream
                .launch_builder(&self.rmsnorm_rows)
                .arg(input)
                .arg(gamma)
                .arg(output)
                .arg(&eps)
                .arg(&dim_i)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// In-place per-head rmsnorm + NEOX rope. `n_heads` selects the q (32)
    /// or k (8) interpretation; positions are `base_pos + row`. The kernel
    /// mutates `qk` through its raw pointer (standard cudarc practice —
    /// shared Rust borrows model device-side mutation).
    unsafe fn launch_qk_rope(
        &self,
        qk: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        rows: usize,
        n_heads: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        assert_eq!(
            self.cfg.head_dim, 128,
            "dflash2_qk_norm_rope_neox is specialized for head_dim == 128"
        );
        let (nh, hd, bp, eps) = (
            n_heads as i32,
            self.cfg.head_dim as i32,
            base_pos as i32,
            self.cfg.rms_eps,
        );
        // SAFETY: qk covers rows*n_heads*head_dim; gamma/freqs cover 128/64.
        unsafe {
            self.stream
                .launch_builder(&self.qk_rope)
                .arg(qk)
                .arg(gamma)
                .arg(&self.rope_freqs)
                .arg(&nh)
                .arg(&hd)
                .arg(&bp)
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: ((rows * n_heads) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_attention(
        &self,
        li: usize,
        anchor_pos: usize,
        n_ring: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let (bs, nh, kvh, hd) = (
            cfg.block_size as i32,
            cfg.n_head as i32,
            cfg.n_kv_head as i32,
            cfg.head_dim as i32,
        );
        let (swa, grp) = (cfg.sliding_window as i32, (cfg.n_head / cfg.n_kv_head) as i32);
        let (ap, nr) = (anchor_pos as i32, n_ring as i32);
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let causal = i32::from(crate::qwen38_dflash2::dflash2_causal_block());
        // SAFETY: all extents sized from cfg in `new`.
        unsafe {
            self.stream
                .launch_builder(&self.attention)
                .arg(&self.q)
                .arg(&self.k)
                .arg(&self.v)
                .arg(&self.ring_k[li])
                .arg(&self.ring_v[li])
                .arg(&self.ring_pos_dev)
                .arg(&nr)
                .arg(&ap)
                .arg(&swa)
                .arg(&grp)
                .arg(&bs)
                .arg(&nh)
                .arg(&kvh)
                .arg(&hd)
                .arg(&scale)
                .arg(&causal)
                .arg(&self.attn_out)
                .launch(LaunchConfig {
                    grid_dim: ((cfg.block_size * cfg.n_head) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_conv(
        &self,
        dyn_coeff: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        src: &CudaSlice<f32>,
        dst: &CudaSlice<f32>,
        side: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let bs = cfg.block_size;
        let n = bs * e;
        let (bs_i, e_i, side_i) = (bs as i32, e as i32, side as i32);
        let (ng, kern, gsz) = (
            (e / cfg.conv_group) as i32,
            cfg.conv_kernel as i32,
            cfg.conv_group as i32,
        );
        // SAFETY: extents cover the [bs][e] family + the [1280] dyn rows.
        unsafe {
            self.stream
                .launch_builder(&self.conv)
                .arg(dyn_coeff)
                .arg(base)
                .arg(src)
                .arg(dst)
                .arg(&bs_i)
                .arg(&e_i)
                .arg(&side_i)
                .arg(&ng)
                .arg(&kern)
                .arg(&gsz)
                .launch(LaunchConfig {
                    grid_dim: (n.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    unsafe fn launch_elementwise(
        &self,
        func: &CudaFunction,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let n_i = n as i32;
        // SAFETY: all three buffers cover n f32.
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(a)
                .arg(b)
                .arg(out)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (n.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    // ── the public forward paths ──

    /// Batched ring injection for committed positions: `features_rows` is
    /// `[n][25600]` f32 host (the 5×5120 target-tap concat per position,
    /// ascending contiguous from `base_pos`). Mirrors `inject_position`
    /// op-for-op, processed in 8-row chunks (the fc GEMV amortizes its
    /// 268 MB weight read once per chunk; n=8 reads it exactly once).
    pub fn inject_positions(&mut self, features_rows: &[f32], base_pos: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let inp = 5 * e;
        if features_rows.is_empty() {
            return Err("inject_positions: empty features".to_string());
        }
        if !features_rows.len().is_multiple_of(inp) {
            return Err(format!(
                "inject_positions: len {} not a multiple of {inp}",
                features_rows.len()
            ));
        }
        let n = features_rows.len() / inp;
        if n > cfg.sliding_window {
            return Err(format!(
                "inject_positions: {n} rows exceeds the ring capacity {}",
                cfg.sliding_window
            ));
        }
        if self.ring_pos.back() >= Some(&base_pos) {
            return Err(format!(
                "inject: pos {base_pos} not beyond ring back {:?} (must ascend)",
                self.ring_pos.back()
            ));
        }
        let kvd = cfg.n_kv_head * cfg.head_dim;

        for (ci, chunk) in features_rows
            .chunks(DFLASH2_TILE * inp)
            .enumerate()
            .map(|(i, c)| (i * DFLASH2_TILE, c))
        {
            let rows = chunk.len() / inp;
            let row_off = ci; // absolute row index of this chunk's first row
            // stage the chunk into the padded 8-row host buffer
            self.inj_x_host.iter_mut().for_each(|v| *v = 0.0);
            self.inj_x_host[..chunk.len()].copy_from_slice(chunk);
            self.stream
                .memcpy_htod(&self.inj_x_host, &mut self.inj_x)
                .map_err(|er| format!("inj upload: {er}"))?;

            // encoder fc + hidden_norm
            // SAFETY: extents sized in new.
            unsafe {
                self.launch_gemv(&self.fc, &self.inj_x, &self.inj_g, e, inp, rows)?;
                self.launch_rmsnorm_rows(&self.inj_g, &self.hidden_norm, &self.inj_gn, rows, e)?;
            }
            // per-layer k/v + rope + scatter
            for (li, layer) in self.layers.iter().enumerate() {
                // SAFETY: extents sized in new.
                unsafe {
                    self.launch_gemv(&layer.k_proj, &self.inj_gn, &self.inj_k, kvd, e, rows)?;
                    self.launch_gemv(&layer.v_proj, &self.inj_gn, &self.inj_v, kvd, e, rows)?;
                    // rope over the kv heads in place; positions base_pos + abs row
                    self.launch_qk_rope(
                        &self.inj_k,
                        &layer.k_norm,
                        rows,
                        cfg.n_kv_head,
                        base_pos + row_off,
                    )?;
                }
                // slots
                for r in 0..DFLASH2_TILE {
                    self.inj_slots_host[r] = ((base_pos + row_off + r) % cfg.sliding_window) as i32;
                }
                self.stream
                    .memcpy_htod(&self.inj_slots_host, &mut self.inj_slots)
                    .map_err(|er| format!("slots upload: {er}"))?;
                // scatter ONLY the real rows (grid = rows; the kernel guards
                // r >= rows — padded inj_k rows beyond `rows` are stale and
                // never scattered).
                let (rows_i, nkv_i) = (rows as i32, kvd as i32);
                // SAFETY: ring covers swa*kvd; src covers 8*kvd; slots valid.
                unsafe {
                    self.stream
                        .launch_builder(&self.scatter)
                        .arg(&self.ring_k[li])
                        .arg(&self.inj_k)
                        .arg(&self.inj_slots)
                        .arg(&rows_i)
                        .arg(&nkv_i)
                        .launch(LaunchConfig {
                            grid_dim: (DFLASH2_TILE as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|er| er.to_string())?;
                    self.stream
                        .launch_builder(&self.scatter)
                        .arg(&self.ring_v[li])
                        .arg(&self.inj_v)
                        .arg(&self.inj_slots)
                        .arg(&rows_i)
                        .arg(&nkv_i)
                        .launch(LaunchConfig {
                            grid_dim: (DFLASH2_TILE as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .map_err(|er| er.to_string())?;
                }
            }
        }

        // host occupancy (mirrors the CPU's per-position pop/push)
        for r in 0..n {
            if self.ring_pos.len() == cfg.sliding_window {
                self.ring_pos.pop_front();
            }
            self.ring_pos.push_back(base_pos + r);
        }
        Ok(())
    }

    /// The draft forward — returns the `[8][5120]` post-output_norm rows
    /// (host). `anchor_emb`/`mask_emb_row` are the TARGET's token
    /// embedding rows (host, `[5120]` each). Mirrors `draft_block_hidden`
    /// op-for-op; see the module doc's numerics contract.
    pub fn draft_block_hidden_gpu(
        &mut self,
        anchor_emb: &[f32],
        mask_emb_row: &[f32],
        anchor_pos: usize,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let hd = cfg.head_dim;
        let nh = cfg.n_head;
        let kvh = cfg.n_kv_head;
        let bs = cfg.block_size;
        let qd = nh * hd;
        let kvd = kvh * hd;
        let projected = 2 * cfg.conv_kernel * (e / cfg.conv_group);
        if anchor_emb.len() != e || mask_emb_row.len() != e {
            return Err(format!(
                "draft: embed rows must be [{e}] (got {} / {})",
                anchor_emb.len(),
                mask_emb_row.len()
            ));
        }

        // block input: anchor + 7 masks
        let mut x_host = vec![0.0f32; bs * e];
        x_host[..e].copy_from_slice(anchor_emb);
        for i in 1..bs {
            x_host[i * e..(i + 1) * e].copy_from_slice(mask_emb_row);
        }
        self.stream
            .memcpy_htod(&x_host, &mut self.x)
            .map_err(|er| format!("x upload: {er}"))?;

        // ring position list upload (pad -1 to the full slot count)
        self.ring_pos_host.clear();
        self.ring_pos_host.extend(self.ring_pos.iter().map(|&p| p as i32));
        self.ring_pos_host.resize(cfg.sliding_window, -1);
        let n_ring = self.ring_pos.len();
        self.stream
            .memcpy_htod(&self.ring_pos_host, &mut self.ring_pos_dev)
            .map_err(|er| format!("ring_pos upload: {er}"))?;

        for li in 0..cfg.n_layer {
            let layer = &self.layers[li];
            // SAFETY: every buffer below is sized for [bs][dim] in `new`.
            unsafe {
                // 1. h = rmsnorm(x, attn_norm); dyn_coeff
                self.launch_rmsnorm_rows(&self.x, &layer.attn_norm, &self.h, bs, e)?;
                self.launch_gemv(&layer.attn_conv_proj, &self.h, &self.dyn_coeff, projected, e, bs)?;
                // 2. conv side 0: h -> hc
                self.launch_conv(&self.dyn_coeff, &layer.attn_conv_base, &self.h, &self.hc, 0)?;
                // 3. q/k/v
                self.launch_gemv(&layer.q_proj, &self.hc, &self.q, qd, e, bs)?;
                self.launch_gemv(&layer.k_proj, &self.hc, &self.k, kvd, e, bs)?;
                self.launch_gemv(&layer.v_proj, &self.hc, &self.v, kvd, e, bs)?;
                // 4. per-head norms + rope at anchor_pos + i
                self.launch_qk_rope(&self.q, &layer.q_norm, bs, nh, anchor_pos)?;
                self.launch_qk_rope(&self.k, &layer.k_norm, bs, kvh, anchor_pos)?;
                // 5. attention
                self.launch_attention(li, anchor_pos, n_ring)?;
                // 6. o_proj + conv side 1 + residual
                self.launch_gemv(&layer.o_proj, &self.attn_out, &self.ao, e, qd, bs)?;
                self.launch_conv(&self.dyn_coeff, &layer.attn_conv_base, &self.ao, &self.attn_final, 1)?;
                self.launch_elementwise(&self.residual_add, &self.attn_final, &self.x, &self.ffn_inp, bs * e)?;
                // 7. ffn
                self.launch_rmsnorm_rows(&self.ffn_inp, &layer.ffn_norm, &self.hf, bs, e)?;
                self.launch_gemv(&layer.ffn_conv_proj, &self.hf, &self.dynf, projected, e, bs)?;
                self.launch_conv(&self.dynf, &layer.ffn_conv_base, &self.hf, &self.hfc, 0)?;
                self.launch_gemv(&layer.gate_proj, &self.hfc, &self.gate_out, cfg.n_ff, e, bs)?;
                self.launch_gemv(&layer.up_proj, &self.hfc, &self.up_out, cfg.n_ff, e, bs)?;
                self.launch_elementwise(&self.swiglu, &self.gate_out, &self.up_out, &self.mid, bs * cfg.n_ff)?;
                self.launch_gemv(&layer.down_proj, &self.mid, &self.down_out, e, cfg.n_ff, bs)?;
                self.launch_conv(&self.dynf, &layer.ffn_conv_base, &self.down_out, &self.ffn_final, 1)?;
                self.launch_elementwise(&self.residual_add, &self.ffn_final, &self.ffn_inp, &self.x, bs * e)?;
            }
        }

        // final norm -> [8][5120]
        // SAFETY: out_rows sized bs*e in new.
        unsafe {
            self.launch_rmsnorm_rows(&self.x, &self.output_norm, &self.out_rows, bs, e)?;
        }
        let mut out = vec![0.0f32; bs * e];
        self.stream
            .memcpy_dtoh(&self.out_rows, &mut out)
            .map_err(|er| format!("out download: {er}"))?;
        Ok(out)
    }
}



/// Recover the stream's context for module loading. cudarc 0.19 does not
/// expose `CudaStream::ctx()`, but the primary context is a process-wide
/// singleton — `CudaContext::new(0)` returns the SAME context the stream
/// was created on (device 0), which is what `load_module` needs.
fn stream_ctx(stream: &Arc<CudaStream>) -> Result<Arc<CudaContext>, String> {
    let _ = stream;
    CudaContext::new(0).map_err(|e| format!("cuda context: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip_exact_for_bf16_origin_values() {
        // every bf16 bit pattern widens then round-trips exactly
        for h in 0u32..0x1_0000 {
            let v = f32::from_bits(h << 16);
            if v.is_nan() {
                continue; // NaN payload handling is not needed here
            }
            assert_eq!(f32_to_bf16_bits(v) as u32, h, "h={h:04x}");
        }
    }
}

// ── Issue 780 U3 wall-1 (a): the quant-resident dequant-parity gate ──
//
// The GPU kernels dequant in-register; the HOST reference lives in
// riir-infer-core (dequant_tensor_row, the ggml-reference math). This gate
// runs BOTH over real artifact tensors and requires per-row cosine ≥ 0.99999
// (the Bench-758 accumulation-order precedent — tree vs serial sums differ
// in ulps, never in direction). A mapping bug (byte/shift/hbit) produces
// O(1) errors and fails hard.
#[cfg(test)]
mod q2k_parity_tests {
    use super::*;

    fn gguf_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("DFLASH2_Q2K_GGUF") {
            let pb = std::path::PathBuf::from(&p);
            if pb.exists() {
                return Some(pb);
            }
            eprintln!("DFLASH2_Q2K_GGUF={p:?} does not exist");
            return None;
        }
        let def = std::path::PathBuf::from("F:/models/Qwen3.8-27B-DFlash2-Q2_K.gguf");
        if def.exists() {
            return Some(def);
        }
        eprintln!("the Q2_K drafter artifact not found — skipping (set DFLASH2_Q2K_GGUF)");
        None
    }

    /// One probe: rows `[o0..o0+n_probe)` of `name`, random f32 x, kernel
    /// rows-GEMV vs host-dequant dots. Returns the worst cosine.
    fn probe_rows(
        drafter: &DFlash2GpuDrafter,
        gguf: &GgufFile,
        name: &str,
        w: &QW,
        out_dim: usize,
        in_dim: usize,
        o0: usize,
        n_probe: usize,
        x: &[f32],
    ) -> Result<f64, String> {
        // host expectations
        let mut expected = Vec::with_capacity(n_probe);
        for o in o0..o0 + n_probe {
            let row = gguf
                .dequant_tensor_row(name, o, in_dim)
                .map_err(|e| format!("host dequant {name}[{o}]: {e}"))?;
            let dot: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            expected.push(dot);
        }
        // device run (full rows=1 GEMV — the kernel computes every output
        // column; we read back just the probed rows)
        let stream = &drafter.stream;
        let mut x_dev = stream
            .alloc_zeros::<f32>(in_dim)
            .map_err(|e| e.to_string())?;
        // Copy ONLY the probe's in_dim prefix — x is sized for the largest
        // probe (inp); the fc probe's in_dim == x.len() masked this for every
        // other probe until the constructor made the gate runnable at all.
        stream
            .memcpy_htod(&x[..in_dim], &mut x_dev)
            .map_err(|e| e.to_string())?;
        let y_dev = stream
            .alloc_zeros::<f32>(out_dim)
            .map_err(|e| e.to_string())?;
        // SAFETY: extents sized directly above; w covers out_dim×in_dim.
        unsafe {
            drafter.launch_gemv(w, &x_dev, &y_dev, out_dim, in_dim, 1)?;
        }
        let mut got_full = vec![0.0f32; out_dim];
        stream
            .memcpy_dtoh(&y_dev, &mut got_full)
            .map_err(|e| e.to_string())?;
        let got = &got_full[o0..o0 + n_probe];
        let mut worst = 0.0f64;
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            // scalar dots — relative error, not cosine (per-row vectors would
            // be the 758 cosine form; a rows-GEMV output is one f32 per row).
            // Accumulation-order noise on 25600-term dots is ~1e-6; a mapping
            // bug is O(1). Bound 1e-3 sits far from both.
            let rel = ((g as f64 - e as f64) / e.abs().max(1e-9) as f64).abs();
            worst = worst.max(rel);
            assert!(
                rel <= 1e-3,
                "{name}[{}]: kernel {g} vs host {e} (rel {rel:.3e})",
                o0 + i
            );
        }
        Ok(worst)
    }

    #[test]
    #[ignore = "GPU gate — the analogalok artifact (~700 MB) + exclusive GPU (Issue 780 U3)"]
    fn dequant_gemv_matches_host_reference_per_format() {
        let Some(path) = gguf_path() else {
            eprintln!("artifact absent — gate not run (deliverable: code + this note)");
            return;
        };
        let ctx = CudaContext::new(0).expect("cuda ctx");
        let stream = ctx.new_stream().expect("stream");
        let cfg = DFlash2Config::default();
        let t0 = std::time::Instant::now();
        let drafter =
            DFlash2GpuDrafter::new_from_gguf(stream.clone(), cfg.clone(), path.to_str().unwrap())
                .expect("gguf drafter");
        eprintln!(
            "[load] gguf drafter in {:.1}s ({:.2} GiB device — the quant-resident figure)",
            t0.elapsed().as_secs_f64(),
            drafter.device_bytes() as f64 / (1 << 30) as f64
        );
        let gguf = GgufFile::open(&path).expect("reopen gguf for host reference");

        let e = cfg.n_embd;
        let ff = cfg.n_ff;
        let qd = cfg.n_head * cfg.head_dim;
        let kvd = cfg.n_kv_head * cfg.head_dim;
        let inp = 5 * e;

        // deterministic x (fixed-seed fastrand; values ~N(0,1) scale)
        let mut rng = fastrand::Rng::with_seed(0xDFA5_2026);
        let max_len = inp;
        let x: Vec<f32> = (0..max_len).map(|_| rng.f32() * 2.0 - 1.0).collect();

        // (name, weight field accessor, out_dim, in_dim, format label) — one
        // probe set per K-quant class + bf16-free (this artifact has none for
        // the block weights; norms are F32 and load as-is).
        let l = &drafter.layers[0];
        let probes: Vec<(&str, &QW, usize, usize, &str)> = vec![
            ("fc.weight", &drafter.fc, e, inp, "Q2K"),
            ("blk.0.attn_q.weight", &l.q_proj, qd, e, "Q2K"),
            ("blk.0.attn_output.weight", &l.o_proj, e, qd, "Q3K"),
            ("blk.0.ffn_down.weight", &l.down_proj, e, ff, "Q3K"),
            ("blk.0.attn_v.weight", &l.v_proj, kvd, e, "Q4K"),
        ];
        let n_probe = 8;
        for (name, w, out_dim, in_dim, fmt) in &probes {
            // first rows + a late row (row-order bugs hide at 0)
            let worst_first = probe_rows(&drafter, &gguf, name, w, *out_dim, *in_dim, 0, n_probe, &x)
                .unwrap_or_else(|err| panic!("{fmt} {name}: {err}"));
            let late = out_dim - 1;
            let worst_late =
                probe_rows(&drafter, &gguf, name, w, *out_dim, *in_dim, late, 1, &x)
                    .unwrap_or_else(|err| panic!("{fmt} {name}: {err}"));
            eprintln!(
                "[{fmt}] {name}: worst rel first-rows {worst_first:.3e} · last-row {worst_late:.3e} — PASS"
            );
        }
    }
}
