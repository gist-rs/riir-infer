//! Issue 615 T4 — Qwen attention decode CUDA kernels (cudarc + nvrtc).
//!
//! Ports the CubeCL Qwen3.5 attention path to raw CUDA:
//! - `rope_partial_f32` — partial RoPE (GPT-NeoX rotate-half convention)
//! - `split_qg_f32` — split interleaved [q, gate] per head
//! - `rmsnorm_batched_f32` — per-head RMSNorm for Q/K
//! - `kv_cache_append_f32` — append K/V to KV cache at position
//! - `attention_decode_f32` — flash attention with online softmax
//! - `attention_decode_split_partial_f32` (+ `_devpos` twin) — split-KV
//!   flash decode partial: one (head, chunk) block per position range
//!   (Issue 742 long-ctx parallelism fix)
//! - `attention_decode_splitgqa_partial_f32` (+ `_devpos` twin) — GQA-fused
//!   split partial: one block per (kv_head, chunk) serves the whole GQA group
//!   (Issue 742 headroom — coalesced Q·K + amortized V reads)
//! - `attention_decode_split_combine_f32` — flash-decode partial merge
//! - `output_gate_f32` — sigmoid output gating
//!
//! ## Why raw CUDA (not CubeCL)
//!
//! The CubeCL `qwen_attention_decode_f32` uses `Shared::<[f32]>::new_slice(128)`
//! — a compile-time constant that is **too small for Bonsai-27B** (head_dim=256).
//! The Issue 612 fix comment incorrectly claims "head_dim is 128 for all
//! Qwen/Bonsai models"; the real Bonsai-27B config has head_dim=256 (confirmed
//! via `gguf_loader.rs` line 1822: "rope.dimension_count = 64 for
//! Ternary-Bonsai-27B, head_dim=256"). These CUDA kernels use **dynamic shared
//! memory** (`extern __shared__`) so the smem allocation matches head_dim at
//! launch time — correct for any head_dim up to 1024.
//!
//! ## Dispatch summary
//!
//! | Kernel | Grid | Block | smem |
//! |---|---|---|---|
//! | `rope_partial_f32` | `ceil(n_head * rotary_pairs / 256)` | 256 | 0 |
//! | `split_qg_f32` | `ceil(n_head * head_dim / 256)` | 256 | 0 |
//! | `rmsnorm_batched_f32` | `n_head` or `n_kv_head` | 256 | 256 × 4 |
//! | `kv_cache_append_f32` | `ceil(kvd / 256)` | 256 | 0 |
//! | `attention_decode_f32` | `n_head` | `head_dim` | `head_dim × 4` |
//! | `attention_decode_split_partial_f32` | `(n_head, n_chunks)` | `head_dim` | `head_dim × 4` |
//! | `attention_decode_splitgqa_partial_f32` | `(n_kv_head, n_chunks)` | `256` | `~40 KB dynamic` |
//! | `attention_decode_split_combine_f32` | `n_head` | `head_dim` | 0 |
//! | `output_gate_f32` | `ceil(q_dim / 256)` | 256 | 0 |

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, DevicePtr, LaunchConfig};
use cudarc::driver::PushKernelArg;

/// Issue 754 T6 — the qg rows kernels' row cap. The kernel body owns rows
/// in 16-row grid.z slices (`row_off = blockIdx.z * 16`), so the launcher
/// maps any `p` up to this bound onto `grid.z = ceil(p/16)` blocks; the
/// per-block register/smem geometry is UNCHANGED from the p <= 16 kernel
/// (launch-config- and bit-identical at p <= 16 where grid.z == 1).
/// 64 = the P4 target (the T5 GEMM ingest win is measured at P=64,
/// Bench 772). Total K/V traffic at p=64 equals p<=16 semantics — the
/// z-blocks re-stage K from L2 exactly as 4 sequential p=16 chunks would
/// re-read it — while the T9.9 fallback (QPB=4, grid.z=16) pays 4x that.
pub const SPLITGQA_QG_ROWS_MAX_P: usize = 64;

/// Combined CUDA source for all attention-related kernels.
///
/// Compiled as a single PTX module — each kernel is a `__global__` function.
/// Uses `extern __shared__` for dynamic shared memory where the size depends
/// on a runtime dimension (head_dim).
pub const ATTENTION_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// Issue 753 — KV cache dtype hatch (`KV_F16`). The host compiles this source
// TWICE: the default build is f32 KV (byte-identical to the pre-753 kernels);
// `AttentionKernels::new_with_kv_dtype(.., true)` prepends `#define KV_F16 1`
// and the SAME kernel names then store K/V as IEEE f16 halves. Kernel names,
// signatures (pointer ABI), grids and launchers are IDENTICAL in both builds.
//
// The conversion happens at the global-memory boundary ONLY: K/V loads
// convert half→float at read, appends convert float→half (RN) at write.
// Everything downstream — smem staging (stays float), the score dot, online
// softmax, partials, the combine — is UNCHANGED f32 arithmetic, so the
// numerics delta vs the f32 build is exactly the f16 quantization of the
// K/V elements (element rel-err ≤ 2^-11 ≈ 4.9e-4). Tolerance-class change,
// NOT bit-identity — the Issue-753 gates are declared accordingly.
// ---------------------------------------------------------------------------
#ifdef KV_F16
// Manual f16↔f32 bit conversion (RN-even) — cudarc's nvrtc exposes NO
// include directories, so <cuda_fp16.h> is unavailable ("no directories in
// search list", pinned by the t916 compile-smoke gate). These are the
// textbook bit manipulations; kv_f32_to_f16 rounds to nearest-even exactly
// like __float2half_rn (subnormal carry into the min-normal encoding and
// overflow carry into Inf fall out of the +=1). A host-side twin lives in
// qwen38_dense_cudarc (kv_f16_bits) — the GPU parity gates pin the two
// implementations to agreement bit-for-bit.
typedef unsigned short kv_elt;
__device__ __forceinline__ float kv_f16_to_f32(unsigned short h)
{
    const unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    const unsigned int exp = ((unsigned int)h >> 10) & 0x1fu;
    unsigned int mant = (unsigned int)h & 0x3ffu;
    if (exp == 0x1fu) {
        // Inf / NaN (quiet the NaN)
        return __uint_as_float(sign | 0x7f800000u
                               | (mant ? (0x400000u | (mant << 13)) : 0u));
    }
    if (exp == 0) {
        if (mant == 0) return __uint_as_float(sign);       // ±0
        // Subnormal half: value = mant * 2^-24. Normalize: shift until the
        // leading 1 sits at bit 10, then the exponent is 113 - shifts.
        int s = 0;
        while (!(mant & 0x400u)) { mant <<= 1; s++; }
        return __uint_as_float(sign | ((unsigned int)(113 - s) << 23)
                               | ((mant & 0x3ffu) << 13));
    }
    return __uint_as_float(sign | ((exp + 112u) << 23) | (mant << 13));
}
__device__ __forceinline__ unsigned short kv_f32_to_f16(float f)
{
    const unsigned int x = __float_as_uint(f);
    const unsigned int sign = (x >> 16) & 0x8000u;
    const int e = (int)((x >> 23) & 0xffu);   // biased f32 exponent
    const unsigned int m = x & 0x007fffffu;   // 23-bit mantissa
    if (e == 0xffu) {
        // Inf / NaN
        return (unsigned short)(sign | 0x7c00u
                                | (m ? (0x0200u | (m >> 13)) : 0u));
    }
    if (e == 0u) {
        // f32 zero/subnormal: magnitude < 2^-126 is below the f16
        // subnormal rounding floor — rounds to signed zero.
        return (unsigned short)sign;
    }
    const int he = e - 112;                   // biased f16 exponent
    if (he >= 0x1f) {
        // |f| >= 2^16: certain overflow — Inf.
        return (unsigned short)(sign | 0x7c00u);
    }
    if (he <= 0) {
        if (he < -10) {
            // |f| < 2^-25 (strictly below the smallest subnormal's rounding
            // point): zero. Exactly 2^-25 is the tie -> rounds to even = 0.
            return (unsigned short)sign;
        }
        // f16 subnormal: n = M >> (14 - he) with M the 24-bit significand;
        // RN-even on the shifted-out bits. A carry to n == 1024 lands on
        // the min-NORMAL encoding (exp=1, mant=0) — falls out of sign | n.
        const unsigned int mf = m | 0x00800000u;
        const int sh = 14 - he;
        const unsigned int n = mf >> sh;
        const unsigned int rem = mf & ((1u << sh) - 1u);
        const unsigned int half = 1u << (sh - 1);
        const unsigned int nr = n + ((rem > half || (rem == half && (n & 1u))) ? 1u : 0u);
        return (unsigned short)(sign | nr);
    }
    // Normal: RN-even on the low 13 mantissa bits; a carry out of the
    // 10-bit mantissa increments the exponent (correct up to Inf at
    // exactly 65520 — the boundary value rounds to even = Inf).
    unsigned short h =
        (unsigned short)(sign | ((unsigned int)he << 10) | (m >> 13));
    const unsigned int rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;
    return h;
}
#define KV_RD(p) (kv_f16_to_f32(*(p)))
#define KV_WR(p, v) (*(p) = kv_f32_to_f16(v))
#else
typedef float kv_elt;
#define KV_RD(p) (*(p))
#define KV_WR(p, v) (*(p) = (v))
#endif

// ---------------------------------------------------------------------------
// Partial RoPE kernel (GPT-NeoX rotate-half convention)
// ---------------------------------------------------------------------------

extern "C" __global__ void rope_partial_f32(
    float* __restrict__ q,           // [n_head * head_dim] (in-place)
    float* __restrict__ k,           // [n_kv_head * head_dim] (in-place)
    const int rotary_dim,            // first rotary_dim elements are rotated
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int pos,
    const float theta_base)
{
    const int rotary_pairs = rotary_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q_pairs = n_head * rotary_pairs;

    if (idx >= total_q_pairs) return;

    const int head = idx / rotary_pairs;
    const int pair = idx % rotary_pairs;

    // inv_freq = 1 / theta_base^(2*pair / rotary_dim)
    const float exponent = 2.0f * (float)pair / (float)rotary_dim;
    const float inv_freq = powf(theta_base, -exponent);
    const float theta = (float)pos * inv_freq;
    const float cos_t = cosf(theta);
    const float sin_t = sinf(theta);

    // GPT-NeoX (rotate-half): pairs are (x[i0], x[i0 + rotary_pairs])
    // NOT interleaved (x[2*pair], x[2*pair+1]) — matches CPU apply_rope_heads_precomputed.
    const int q_head_off = head * head_dim;
    const int q_i0 = q_head_off + pair;
    const int q_i1 = q_i0 + rotary_pairs;
    const float q0 = q[q_i0];
    const float q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    // K rotation (only for heads in [0, n_kv_head))
    if (head < n_kv_head) {
        const int k_head_off = head * head_dim;
        const int k_i0 = k_head_off + pair;
        const int k_i1 = k_i0 + rotary_pairs;
        const float k0 = k[k_i0];
        const float k1 = k[k_i1];
        k[k_i0] = k0 * cos_t - k1 * sin_t;
        k[k_i1] = k0 * sin_t + k1 * cos_t;
    }
}

// ---------------------------------------------------------------------------
// Partial RoPE backward (inverse rotation: sin -> -sin).
//
// RoPE forward:  x' = x*cos - y*sin ; y' = x*sin + y*cos
// RoPE backward: x' = x*cos + y*sin ; y' = -x*sin + y*cos
// (equivalent to forward with sin negated)
//
// Applied in-place to gradient buffers grad_q and grad_k.
// ---------------------------------------------------------------------------

extern "C" __global__ void rope_partial_backward_f32(
    float* __restrict__ grad_q,      // [n_head * head_dim] (in-place)
    float* __restrict__ grad_k,      // [n_kv_head * head_dim] (in-place)
    const int rotary_dim,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int pos,
    const float theta_base)
{
    const int rotary_pairs = rotary_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q_pairs = n_head * rotary_pairs;

    if (idx >= total_q_pairs) return;

    const int head = idx / rotary_pairs;
    const int pair = idx % rotary_pairs;

    const float exponent = 2.0f * (float)pair / (float)rotary_dim;
    const float inv_freq = powf(theta_base, -exponent);
    const float theta = (float)pos * inv_freq;
    const float cos_t = cosf(theta);
    const float sin_t = -sinf(theta);  // NEGATED for backward

    const int q_head_off = head * head_dim;
    const int q_i0 = q_head_off + pair;
    const int q_i1 = q_i0 + rotary_pairs;
    const float q0 = grad_q[q_i0];
    const float q1 = grad_q[q_i1];
    grad_q[q_i0] = q0 * cos_t - q1 * sin_t;
    grad_q[q_i1] = q0 * sin_t + q1 * cos_t;

    if (head < n_kv_head) {
        const int k_head_off = head * head_dim;
        const int k_i0 = k_head_off + pair;
        const int k_i1 = k_i0 + rotary_pairs;
        const float k0 = grad_k[k_i0];
        const float k1 = grad_k[k_i1];
        grad_k[k_i0] = k0 * cos_t - k1 * sin_t;
        grad_k[k_i1] = k0 * sin_t + k1 * cos_t;
    }
}

// ---------------------------------------------------------------------------
// Split QG (gated Q projection output) into Q and gate
// ---------------------------------------------------------------------------

extern "C" __global__ void split_qg_f32(
    const float* __restrict__ qg,    // [n_head * 2 * head_dim] interleaved
    float* __restrict__ q,           // [n_head * head_dim]
    float* __restrict__ gate,        // [n_head * head_dim]
    const int head_dim,
    const int n_head)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_head * head_dim;
    if (idx >= total) return;

    const int head = idx / head_dim;
    const int dim = idx % head_dim;
    const int q_src = head * 2 * head_dim + dim;
    const int gate_src = q_src + head_dim;

    q[idx] = qg[q_src];
    gate[idx] = qg[gate_src];
}

// ---------------------------------------------------------------------------
// Per-head RMSNorm (batched)
// ---------------------------------------------------------------------------

/// One block per head. Each block normalizes head_dim elements starting at
/// `input[head_idx * head_dim]` using the shared gamma[head_dim].
/// Uses 256 threads with strided accumulation (supports head_dim > 256).
extern "C" __global__ void rmsnorm_batched_f32(
    const float* __restrict__ input,  // [n_batch * head_dim]
    const float* __restrict__ gamma,  // [head_dim]
    float* __restrict__ output,       // [n_batch * head_dim]
    const float inv_dim,
    const float eps,
    const int head_dim)
{
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // 256

    const int base = head_idx * head_dim;

    // Phase 1: strided sum of x²
    float partial_sq = 0.0f;
    for (int i = tid; i < head_dim; i += block_size) {
        float x = input[base + i];
        partial_sq += x * x;
    }

    // Phase 2: shared memory reduction
    __shared__ float smem[256];
    smem[tid] = partial_sq;
    __syncthreads();

    // Tree reduction (256 threads → 1)
    if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
    if (tid < 64)  smem[tid] += smem[tid + 64];  __syncthreads();
    if (tid < 32)  smem[tid] += smem[tid + 32];  __syncthreads();
    if (tid < 16)  smem[tid] += smem[tid + 16];  __syncthreads();
    if (tid < 8)   smem[tid] += smem[tid + 8];   __syncthreads();
    if (tid < 4)   smem[tid] += smem[tid + 4];   __syncthreads();
    if (tid < 2)   smem[tid] += smem[tid + 2];   __syncthreads();
    if (tid < 1)   smem[0] += smem[1];
    __syncthreads();

    // Phase 3: normalize
    float inv_rms;
    if (tid == 0) {
        float mean_sq = smem[0] * inv_dim;
        smem[0] = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();
    inv_rms = smem[0];

    for (int j = tid; j < head_dim; j += block_size) {
        output[base + j] = input[base + j] * inv_rms * gamma[j];
    }
}

// ---------------------------------------------------------------------------
// KV cache append kernel
// ---------------------------------------------------------------------------

extern "C" __global__ void kv_cache_append_f32(
    const float* __restrict__ k_vec,     // [kvd]
    const float* __restrict__ v_vec,     // [kvd]
    kv_elt* __restrict__ key_cache,      // [max_seq_len * kvd]
    kv_elt* __restrict__ value_cache,    // [max_seq_len * kvd]
    const int kvd,
    const int pos)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kvd) return;

    const int cache_off = pos * kvd + idx;
    KV_WR(&key_cache[cache_off], k_vec[idx]);
    KV_WR(&value_cache[cache_off], v_vec[idx]);
}

// ---------------------------------------------------------------------------
// Flash attention decode kernel (online softmax)
// ---------------------------------------------------------------------------

/// One block per query head. blockDim.x = head_dim threads.
/// Each thread:
///   Phase 1 — computes Q·K score for one position in the tile
///   Phase 2 — parallel max reduction over the tile
///   Phase 3 — computes exp(score - max) softmax weights
///   Phase 4 — accumulates weighted V for its output dimension + serial sum
///
/// Uses dynamic shared memory (`extern __shared__`) sized to head_dim × 4 bytes.
///
/// Issue 612 fix ported: smem sized to head_dim (not a fixed 128), reduction
/// starts at stride head_dim/2, serial sum replaces destructive parallel sum.
extern "C" __global__ void attention_decode_f32(
    const float* __restrict__ query,        // [n_head * head_dim]
    const kv_elt* __restrict__ key_cache,   // [n_positions * n_kv_head * head_dim]
    const kv_elt* __restrict__ value_cache, // [n_positions * n_kv_head * head_dim]
    float* __restrict__ attn_out,           // [n_head * head_dim]
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int n_positions)
{
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int cube_size = blockDim.x;  // = head_dim

    if (head_idx >= n_head) return;

    const int head_off = head_idx * head_dim;

    if (n_positions == 0) {
        if (tid < head_dim) {
            attn_out[head_off + tid] = 0.0f;
        }
        return;
    }

    const int kv_stride = n_kv_head * head_dim;
    const int kv_group = head_idx * n_kv_head / n_head;
    const int kv_off = kv_group * head_dim;
    const bool valid_dim = tid < head_dim;

    // Dynamic shared memory — sized at launch via shared_mem_bytes = head_dim * 4
    extern __shared__ float smem[];

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    const int n_tiles = (n_positions + cube_size - 1) / cube_size;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * cube_size;
        const int pos = tile_base + tid;
        const bool valid_pos = pos < n_positions;

        // Phase 1: Q·K score
        float my_score = -1e30f;
        if (valid_pos) {
            const int k_base = pos * kv_stride + kv_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += query[head_off + d] * KV_RD(&key_cache[k_base + d]);
            }
            my_score = dot * scale;
        }
        smem[tid] = my_score;
        __syncthreads();

        // Phase 2: parallel max reduction (cube_size → 1).
        // Standard tree reduction — works for any power-of-2 cube_size.
        // Loop reduces to 32 elements, then the warp handles the final 32→1.
        for (int stride = cube_size / 2; stride >= 32; stride >>= 1) {
            if (tid < stride && smem[tid + stride] > smem[tid]) {
                smem[tid] = smem[tid + stride];
            }
            __syncthreads();
        }
        // Final warp — no syncthreads needed (warp-synchronous)
        if (tid < 32) {
            volatile float* vsmem = smem;
            if (tid < 16 && vsmem[tid + 16] > vsmem[tid]) vsmem[tid] = vsmem[tid + 16];
            if (tid < 8  && vsmem[tid + 8]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 8];
            if (tid < 4  && vsmem[tid + 4]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 4];
            if (tid < 2  && vsmem[tid + 2]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 2];
            if (tid < 1  && vsmem[1] > vsmem[0]) vsmem[0] = vsmem[1];
        }
        __syncthreads();

        const float tile_max = smem[0];
        // Issue 715 race (a): every thread reads smem[0] here and Phase 3 below
        // has thread 0 overwrite that same slot with its weight. Without a
        // barrier a slower warp reads a weight as the tile max, the threads then
        // disagree on the softmax shift, and the tile's weights come out
        // inconsistently scaled. Length-independent - fires at any n_positions.
        __syncthreads();

        // Online softmax: rescale running state
        float new_max = (tile_max > running_max) ? tile_max : running_max;
        float exp_prev = expf(running_max - new_max);
        float exp_tile = expf(tile_max - new_max);

        running_sum *= exp_prev;
        running_out *= exp_prev;
        running_max = new_max;

        // Phase 3: compute weight = exp_tile * exp(my_score - tile_max)
        if (valid_pos) {
            smem[tid] = exp_tile * expf(my_score - tile_max);
        } else {
            smem[tid] = 0.0f;
        }
        __syncthreads();

        // Phase 4: weighted V accumulation + serial sum (Issue 612 — non-destructive).
        float tile_sum = 0.0f;
        float acc = 0.0f;
        for (int p = 0; p < cube_size; p++) {
            int kv_pos = tile_base + p;
            if (kv_pos < n_positions) {
                float weight = smem[p];
                tile_sum += weight;
                if (valid_dim) {
                    int v_idx = kv_pos * kv_stride + kv_off + tid;
                    acc += weight * KV_RD(&value_cache[v_idx]);
                }
            }
        }
        if (valid_dim) {
            running_out += acc;
        }
        running_sum += tile_sum;

        // Issue 715 race (b): Phase 4 above reads every smem[p] for this tile;
        // the next iteration's Phase 1 writes smem[tid]. With no barrier on the
        // back-edge a thread that finishes its read loop early clobbers a weight
        // another thread has not consumed. Only reachable when n_tiles >= 2,
        // i.e. n_positions > cube_size (= head_dim = 256 for Bonsai-27B).
        __syncthreads();
    }

    // Final normalization
    if (valid_dim) {
        float inv_sum = 1.0f / running_sum;
        attn_out[head_off + tid] = running_out * inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Output gating kernel
// ---------------------------------------------------------------------------

extern "C" __global__ void output_gate_f32(
    float* __restrict__ attn_out,    // [n] (in-place)
    const float* __restrict__ gate,  // [n]
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    attn_out[idx] *= sig;
}

// ===========================================================================
// Issue 618 — Device-pointer variants for CUDA Graph capture.
//
// Each variant reads `pos` from a `const int*` device pointer instead of a
// scalar `int` arg. The scalar would be baked into the captured graph; the
// device pointer is dereferenced at kernel runtime, so updating the device
// buffer before `graph.launch()` lets the graph process a different position
// each call. The caller writes the new pos value to a 4-byte device buffer
// via `memcpy_htod` on the same stream, ordered before `graph.launch()`.
// ===========================================================================

extern "C" __global__ void rope_partial_f32_devpos(
    float* __restrict__ q,
    float* __restrict__ k,
    const int rotary_dim,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int* __restrict__ pos_dev,   // device pointer to pos
    const float theta_base)
{
    const int pos = *pos_dev;
    const int rotary_pairs = rotary_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q_pairs = n_head * rotary_pairs;

    if (idx >= total_q_pairs) return;

    const int head = idx / rotary_pairs;
    const int pair = idx % rotary_pairs;

    const float exponent = 2.0f * (float)pair / (float)rotary_dim;
    const float inv_freq = powf(theta_base, -exponent);
    const float theta = (float)pos * inv_freq;
    const float cos_t = cosf(theta);
    const float sin_t = sinf(theta);

    const int q_head_off = head * head_dim;
    const int q_i0 = q_head_off + pair;
    const int q_i1 = q_i0 + rotary_pairs;
    const float q0 = q[q_i0];
    const float q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    if (head < n_kv_head) {
        const int k_head_off = head * head_dim;
        const int k_i0 = k_head_off + pair;
        const int k_i1 = k_i0 + rotary_pairs;
        const float k0 = k[k_i0];
        const float k1 = k[k_i1];
        k[k_i0] = k0 * cos_t - k1 * sin_t;
        k[k_i1] = k0 * sin_t + k1 * cos_t;
    }
}

extern "C" __global__ void kv_cache_append_f32_devpos(
    const float* __restrict__ k_vec,
    const float* __restrict__ v_vec,
    kv_elt* __restrict__ key_cache,
    kv_elt* __restrict__ value_cache,
    const int kvd,
    const int* __restrict__ pos_dev)
{
    const int pos = *pos_dev;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kvd) return;

    const int cache_off = pos * kvd + idx;
    KV_WR(&key_cache[cache_off], k_vec[idx]);
    KV_WR(&value_cache[cache_off], v_vec[idx]);
}

// attention_decode_f32_devpos: reads n_positions = *pos_dev + 1 internally.
// (Production invariant: n_positions = pos + 1 during decode.)
//
// Body is IDENTICAL to attention_decode_f32 above — only the n_positions
// source differs (device pointer dereference vs scalar arg). This keeps the
// two kernels provably equivalent.
extern "C" __global__ void attention_decode_f32_devpos(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ attn_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int* __restrict__ pos_dev)
{
    const int n_positions = *pos_dev + 1;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int cube_size = blockDim.x;  // = head_dim

    if (head_idx >= n_head) return;

    const int head_off = head_idx * head_dim;

    if (n_positions == 0) {
        if (tid < head_dim) {
            attn_out[head_off + tid] = 0.0f;
        }
        return;
    }

    const int kv_stride = n_kv_head * head_dim;
    const int kv_group = head_idx * n_kv_head / n_head;
    const int kv_off = kv_group * head_dim;
    const bool valid_dim = tid < head_dim;

    // Dynamic shared memory — sized at launch via shared_mem_bytes = head_dim * 4
    extern __shared__ float smem[];

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    const int n_tiles = (n_positions + cube_size - 1) / cube_size;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * cube_size;
        const int pos = tile_base + tid;
        const bool valid_pos = pos < n_positions;

        // Phase 1: Q·K score
        float my_score = -1e30f;
        if (valid_pos) {
            const int k_base = pos * kv_stride + kv_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += query[head_off + d] * KV_RD(&key_cache[k_base + d]);
            }
            my_score = dot * scale;
        }
        smem[tid] = my_score;
        __syncthreads();

        // Phase 2: parallel max reduction (cube_size → 1).
        for (int stride = cube_size / 2; stride >= 32; stride >>= 1) {
            if (tid < stride && smem[tid + stride] > smem[tid]) {
                smem[tid] = smem[tid + stride];
            }
            __syncthreads();
        }
        // Final warp — no syncthreads needed (warp-synchronous)
        if (tid < 32) {
            volatile float* vsmem = smem;
            if (tid < 16 && vsmem[tid + 16] > vsmem[tid]) vsmem[tid] = vsmem[tid + 16];
            if (tid < 8  && vsmem[tid + 8]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 8];
            if (tid < 4  && vsmem[tid + 4]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 4];
            if (tid < 2  && vsmem[tid + 2]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 2];
            if (tid < 1  && vsmem[1] > vsmem[0]) vsmem[0] = vsmem[1];
        }
        __syncthreads();

        const float tile_max = smem[0];
        // Issue 715 race (a): every thread reads smem[0] here and Phase 3 below
        // has thread 0 overwrite that same slot with its weight. Without a
        // barrier a slower warp reads a weight as the tile max, the threads then
        // disagree on the softmax shift, and the tile's weights come out
        // inconsistently scaled. Length-independent - fires at any n_positions.
        __syncthreads();

        // Online softmax: rescale running state
        float new_max = (tile_max > running_max) ? tile_max : running_max;
        float exp_prev = expf(running_max - new_max);
        float exp_tile = expf(tile_max - new_max);

        running_sum *= exp_prev;
        running_out *= exp_prev;
        running_max = new_max;

        // Phase 3: compute weight = exp_tile * exp(my_score - tile_max)
        if (valid_pos) {
            smem[tid] = exp_tile * expf(my_score - tile_max);
        } else {
            smem[tid] = 0.0f;
        }
        __syncthreads();

        // Phase 4: weighted V accumulation + serial sum (Issue 612 — non-destructive).
        float tile_sum = 0.0f;
        float acc = 0.0f;
        for (int p = 0; p < cube_size; p++) {
            int kv_pos = tile_base + p;
            if (kv_pos < n_positions) {
                float weight = smem[p];
                tile_sum += weight;
                if (valid_dim) {
                    int v_idx = kv_pos * kv_stride + kv_off + tid;
                    acc += weight * KV_RD(&value_cache[v_idx]);
                }
            }
        }
        if (valid_dim) {
            running_out += acc;
        }
        running_sum += tile_sum;

        // Issue 715 race (b): Phase 4 above reads every smem[p] for this tile;
        // the next iteration's Phase 1 writes smem[tid]. With no barrier on the
        // back-edge a thread that finishes its read loop early clobbers a weight
        // another thread has not consumed. Only reachable when n_tiles >= 2,
        // i.e. n_positions > cube_size (= head_dim = 256 for Bonsai-27B).
        __syncthreads();
    }

    // Final normalization
    if (valid_dim) {
        float inv_sum = 1.0f / running_sum;
        attn_out[head_off + tid] = running_out * inv_sum;
    }
}

// ===========================================================================
// Issue 742 - split-KV flash decode: the long-ctx parallelism fix.
//
// `attention_decode_f32` launches n_head blocks with a SERIAL tile loop over
// all positions - at 12K ctx that is 24 blocks on a 128-SM part (95% idle,
// ~21 GB/s effective vs the ~0.67 ms the bandwidth model predicts for the
// 537 MB KV read). The split parallelizes the position axis:
//
//   partial : grid (n_head, n_chunks) - each block online-softmaxes its
//             chunk of positions into an UNNORMALIZED partial
//             (m, l, out[head_dim]) in scratch.
//   combine : grid (n_head,) - flash-decode merge:
//             out = sum_c exp(m_c - M) * out_c / sum_c exp(m_c - M) * l_c.
//
// Numerics: the tile body is `attention_decode_f32` verbatim (both Issue-715
// barriers preserved); the split changes only float reassociation (different
// rescale chain + chunk-partitioned sums) - the tree-reduction class.
// Gate: max_rel <= 1e-5 vs the serial kernel.
//
// CUDA-graph contract (devpos twin): the grid is FIXED at
// (n_head, n_chunks_max) computed from ctx_len - chunk_len and the grid dims
// bake into the capture. Dead chunks (chunk_start >= n_positions, i.e.
// beyond the live pos) write the NEUTRAL partial (m = -1e30, l = 0, out = 0)
// and return - a uniform blockIdx-derived branch (no barrier crossed, the
// Issue-715 divergent-barrier hazard cannot arise), and every scratch slot
// is rewritten on every launch so a shrinking n_positions (speculative
// rollback) can never read a stale partial. The combine iterates dead chunks
// harmlessly: expf(-1e30 - M) underflows to exactly 0 and l = 0, so their
// contribution is exactly +0.0 - a live-grid combine and a max-grid combine
// are bit-identical.
// ===========================================================================

// Shared body of the two partial wrappers (scalar + devpos) - literally the
// same code, so the pair stays provably equivalent (the file's standing
// convention for _devpos twins).
__device__ __forceinline__ void attention_decode_split_partial_body(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,     // [n_head * gridDim.y]
    float* __restrict__ part_l,     // [n_head * gridDim.y]
    float* __restrict__ part_out,   // [n_head * gridDim.y * head_dim]
    float* __restrict__ smem,       // dynamic shared, head_dim floats
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int n_positions)
{
    const int head_idx = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int tid = threadIdx.x;
    const int cube_size = blockDim.x;  // = head_dim

    if (head_idx >= n_head) return;

    const int head_off = head_idx * head_dim;
    const int p_idx = head_idx * n_chunks + chunk_id;
    const int p_out_base = p_idx * head_dim;

    const int chunk_start = chunk_id * chunk_len;
    const int chunk_end_raw = chunk_start + chunk_len;
    const int chunk_end = (chunk_end_raw < n_positions) ? chunk_end_raw : n_positions;

    // Dead chunk - neutral partial (uniform early-exit: blockIdx-derived,
    // taken by the whole block before any barrier).
    if (chunk_start >= n_positions) {
        if (tid == 0) {
            part_m[p_idx] = -1e30f;
            part_l[p_idx] = 0.0f;
        }
        if (tid < head_dim) {
            part_out[p_out_base + tid] = 0.0f;
        }
        return;
    }

    const int kv_stride = n_kv_head * head_dim;
    const int kv_group = head_idx * n_kv_head / n_head;
    const int kv_off = kv_group * head_dim;
    const bool valid_dim = tid < head_dim;

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_out = 0.0f;

    // chunk_len is a multiple of cube_size (launcher contract), so tile
    // boundaries align with chunk boundaries: tiles [tile0, tile1) cover
    // exactly [chunk_start, chunk_end).
    const int tile0 = chunk_start / cube_size;
    const int tile1 = (chunk_end + cube_size - 1) / cube_size;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * cube_size;
        const int pos = tile_base + tid;
        const bool valid_pos = pos < chunk_end;

        // Phase 1: Q.K score
        float my_score = -1e30f;
        if (valid_pos) {
            const int k_base = pos * kv_stride + kv_off;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += query[head_off + d] * KV_RD(&key_cache[k_base + d]);
            }
            my_score = dot * scale;
        }
        smem[tid] = my_score;
        __syncthreads();

        // Phase 2: parallel max reduction (cube_size -> 1).
        for (int stride = cube_size / 2; stride >= 32; stride >>= 1) {
            if (tid < stride && smem[tid + stride] > smem[tid]) {
                smem[tid] = smem[tid + stride];
            }
            __syncthreads();
        }
        // Final warp - no syncthreads needed (warp-synchronous)
        if (tid < 32) {
            volatile float* vsmem = smem;
            if (tid < 16 && vsmem[tid + 16] > vsmem[tid]) vsmem[tid] = vsmem[tid + 16];
            if (tid < 8  && vsmem[tid + 8]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 8];
            if (tid < 4  && vsmem[tid + 4]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 4];
            if (tid < 2  && vsmem[tid + 2]  > vsmem[tid]) vsmem[tid] = vsmem[tid + 2];
            if (tid < 1  && vsmem[1] > vsmem[0]) vsmem[0] = vsmem[1];
        }
        __syncthreads();

        const float tile_max = smem[0];
        // Issue 715 race (a) - see attention_decode_f32.
        __syncthreads();

        // Online softmax: rescale running state
        float new_max = (tile_max > running_max) ? tile_max : running_max;
        float exp_prev = expf(running_max - new_max);
        float exp_tile = expf(tile_max - new_max);

        running_sum *= exp_prev;
        running_out *= exp_prev;
        running_max = new_max;

        // Phase 3: compute weight = exp_tile * exp(my_score - tile_max)
        if (valid_pos) {
            smem[tid] = exp_tile * expf(my_score - tile_max);
        } else {
            smem[tid] = 0.0f;
        }
        __syncthreads();

        // Phase 4: weighted V accumulation + serial sum (Issue 612).
        float tile_sum = 0.0f;
        float acc = 0.0f;
        for (int p = 0; p < cube_size; p++) {
            int kv_pos = tile_base + p;
            if (kv_pos < chunk_end) {
                float weight = smem[p];
                tile_sum += weight;
                if (valid_dim) {
                    int v_idx = kv_pos * kv_stride + kv_off + tid;
                    acc += weight * KV_RD(&value_cache[v_idx]);
                }
            }
        }
        if (valid_dim) {
            running_out += acc;
        }
        running_sum += tile_sum;

        // Issue 715 race (b) - back-edge barrier.
        __syncthreads();
    }

    // Partial is UNNORMALIZED (the combine owns the final 1/l).
    if (tid == 0) {
        part_m[p_idx] = running_max;
        part_l[p_idx] = running_sum;
    }
    if (valid_dim) {
        part_out[p_out_base + tid] = running_out;
    }
}

extern "C" __global__ void attention_decode_split_partial_f32(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int n_positions)
{
    extern __shared__ float smem[];
    attention_decode_split_partial_body(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, n_positions);
}

// Graph-capture twin: n_positions = *pos_dev + 1 at kernel runtime; the grid
// is FIXED at (n_head, n_chunks_max) at capture time.
extern "C" __global__ void attention_decode_split_partial_f32_devpos(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev)
{
    extern __shared__ float smem[];
    const int n_positions = *pos_dev + 1;
    attention_decode_split_partial_body(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, n_positions);
}


// ===========================================================================
// Issue 742 - GQA-fused split-KV flash decode partial (the Bench 732 §6
// headroom levers: coalesced Q.K dot + GQA read amplification).
//
// The per-head split partial launches (n_head, n_chunks) blocks: the g q-heads
// sharing one kv-head each read the SAME K/V chunk column (g-fold
// amplification - L2 absorbs most but not all at long ctx), and the Q.K dot
// has one thread walking one K ROW serially (each instruction hits 32
// different sectors - fully uncoalesced row walks).
//
// This variant fuses the whole GQA group into ONE block per (kv_head,
// chunk):
//   - K is staged to shared memory TRANSPOSED via coalesced row loads (one
//     1 KB transaction set per row; the +1 pad kills bank conflicts in both
//     the staged write and the score read),
//   - the dot runs warp-per-head with lane = position (broadcast q,
//     conflict-free transposed reads - the serial d-ascending FMA order is
//     unchanged, so each score is bit-identical to the per-head split),
//   - the P.V phase reads V coalesced (dim-major, same row for all threads)
//     ONCE and reuses it for all g heads (g FMAs per loaded value).
// Scratch layout is IDENTICAL to the per-head split (head-major partials) -
// the existing combine kernel merges it unchanged.
//
// Layout: block = 256 threads (8 warps), TILE = 32 positions. Warps [0, g)
// each own one q-head of the group; every thread owns output dim `tid`.
// Dynamic smem (floats): q[g][head_dim] | k[head_dim][TILE+1] | p[g][TILE] |
// st[g] (rescale factors). At qwen38 dims (g=6, head_dim=256) that is
// ~40.7 KB - under the 48 KB static limit, 2 blocks/SM.
//
// Numerics vs the per-head split: the dot is bit-identical (same values,
// same order); the online-softmax granularity changes (32-position warp
// tiles + shuffle reductions vs 256-position block tiles + smem tree) - the
// reassociation class. Gate: max_rel <= 1e-5 vs the serial kernel.
//
// CUDA-graph contract (devpos twin): identical to the per-head split - the
// grid is FIXED at (n_kv_head, n_chunks_max); dead chunks write the NEUTRAL
// partial for every head of the group (uniform whole-block early exit
// before any barrier) and every scratch slot is rewritten every launch.
//
// Launcher contracts: n_head % n_kv_head == 0, g <= 8 (warp budget),
// head_dim <= 256 (block budget), chunk_len % TILE == 0.
// ===========================================================================
#define ATT_SPLIT_GQA_TILE 32

__device__ __forceinline__ void attention_decode_splitgqa_partial_body(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,     // [n_head * gridDim.y]
    float* __restrict__ part_l,     // [n_head * gridDim.y]
    float* __restrict__ part_out,   // [n_head * gridDim.y * head_dim]
    float* __restrict__ smem,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int n_positions)
{
    const int kv_group = blockIdx.x;   // grid.x = n_kv_head
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = n_head / n_kv_head;

    // smem carve-up (launcher-sized):
    float* q_smem = smem;                                       // g*head_dim
    float* k_smem = q_smem + g * head_dim;                      // head_dim*(TILE+1)
    float* p_smem = k_smem + head_dim * (ATT_SPLIT_GQA_TILE + 1); // g*TILE
    float* st_smem = p_smem + g * ATT_SPLIT_GQA_TILE;           // g

    const int chunk_start = chunk_id * chunk_len;
    if (chunk_start >= n_positions) {
        // Dead chunk - neutral partials for every head of the group
        // (uniform early exit, before any barrier).
        for (int h = 0; h < g; h++) {
            const int p_idx = (kv_group * g + h) * n_chunks + chunk_id;
            if (tid == 0) {
                part_m[p_idx] = -1e30f;
                part_l[p_idx] = 0.0f;
            }
            if (tid < head_dim) {
                part_out[p_idx * head_dim + tid] = 0.0f;
            }
        }
        return;
    }
    const int chunk_end_raw = chunk_start + chunk_len;
    const int chunk_end = (chunk_end_raw < n_positions) ? chunk_end_raw : n_positions;

    const int kv_stride = n_kv_head * head_dim;
    const int kv_off = kv_group * head_dim;

    // Stage the g queries (coalesced, once per block).
    for (int i = tid; i < g * head_dim; i += blockDim.x) {
        const int h = i / head_dim;
        const int d = i % head_dim;
        q_smem[h * head_dim + d] = query[(kv_group * g + h) * head_dim + d];
    }
    __syncthreads();

    // Warp-local online-softmax state (warps [0, g) - one q-head each).
    float run_max = -1e30f;
    float run_sum = 0.0f;
    // Every thread owns output dim `tid` for ALL g heads.
    float acc[8];  // g <= 8 (launcher contract)
#pragma unroll
    for (int h = 0; h < 8; h++) acc[h] = 0.0f;

    const int tile0 = chunk_start / ATT_SPLIT_GQA_TILE;
    const int tile1 = (chunk_end + ATT_SPLIT_GQA_TILE - 1) / ATT_SPLIT_GQA_TILE;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * ATT_SPLIT_GQA_TILE;

        // (A) Stage K tile TRANSPOSED: k_smem[d][l]. Consecutive threads read
        // consecutive dims of one row (coalesced); the +1 pad makes the
        // writes warp-conflict-free and the score-phase reads conflict-free.
        for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
            const int r = i / head_dim;  // position within tile
            const int d = i % head_dim;
            k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + r] =
                KV_RD(&key_cache[(tile_base + r) * kv_stride + kv_off + d]);
        }
        __syncthreads();

        // (B) Score phase - warp h owns q-head h; lane = position.
        if (warp < g) {
            const float* qh = q_smem + warp * head_dim;
            const int pos = tile_base + lane;
            const bool valid = pos < chunk_end;
            float score = -1e30f;
            if (valid) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += qh[d] * k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + lane];
                }
                score = dot * scale;
            }
            // Warp max over lanes (every lane keeps the result).
            float m = score;
            for (int off = 16; off > 0; off >>= 1) {
                const float o = __shfl_down_sync(0xffffffffu, m, off);
                if (o > m) m = o;
            }
            m = __shfl_sync(0xffffffffu, m, 0);
            const float new_max = (m > run_max) ? m : run_max;
            const float exp_prev = expf(run_max - new_max);
            const float exp_tile = expf(m - new_max);
            float p = 0.0f;
            if (valid) p = exp_tile * expf(score - m);
            // Warp sum over lanes.
            float s = p;
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
            }
            s = __shfl_sync(0xffffffffu, s, 0);
            run_sum = run_sum * exp_prev + s;
            run_max = new_max;
            p_smem[warp * ATT_SPLIT_GQA_TILE + lane] = p;
            if (lane == 0) st_smem[warp] = exp_prev;
        }
        __syncthreads();

        // (C) P.V phase - every thread owns dim `tid`; V is read coalesced
        // (one position row, consecutive dims across threads) ONCE and
        // reused for all g heads.
        const int l_valid = (chunk_end - tile_base < ATT_SPLIT_GQA_TILE)
                                ? (chunk_end - tile_base)
                                : ATT_SPLIT_GQA_TILE;
#pragma unroll
        for (int h = 0; h < 8; h++) {
            if (h < g) acc[h] *= st_smem[h];
        }
        for (int l = 0; l < l_valid; l++) {
            const float v = KV_RD(&value_cache[(tile_base + l) * kv_stride + kv_off + tid]);
#pragma unroll
            for (int h = 0; h < 8; h++) {
                if (h < g) acc[h] += p_smem[h * ATT_SPLIT_GQA_TILE + l] * v;
            }
        }
        __syncthreads();
    }

    // Write the group's g partials (unnormalized - the combine owns 1/l).
    if (tid < head_dim) {
#pragma unroll
        for (int h = 0; h < 8; h++) {
            if (h < g) {
                const int p_idx = (kv_group * g + h) * n_chunks + chunk_id;
                part_out[p_idx * head_dim + tid] = acc[h];
            }
        }
    }
    if (warp < g && lane == 0) {
        const int p_idx = (kv_group * g + warp) * n_chunks + chunk_id;
        part_m[p_idx] = run_max;
        part_l[p_idx] = run_sum;
    }
}

extern "C" __global__ void attention_decode_splitgqa_partial_f32(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int n_positions)
{
    extern __shared__ float smem[];
    attention_decode_splitgqa_partial_body(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, n_positions);
}

// Graph-capture twin: n_positions = *pos_dev + 1 at kernel runtime; the grid
// is FIXED at (n_kv_head, n_chunks_max) at capture time.
extern "C" __global__ void attention_decode_splitgqa_partial_f32_devpos(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev)
{
    extern __shared__ float smem[];
    const int n_positions = *pos_dev + 1;
    attention_decode_splitgqa_partial_body(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, n_positions);
}

// Flash-decode combine: out[d] = sum_c w_c * out_c[d] / sum_c w_c * l_c,
// w_c = exp(m_c - M), M = max_c m_c. One block per head, one thread per dim;
// the per-chunk scalars are read redundantly by every thread (broadcast
// loads), so the merge is deterministic, barrier-free and race-free by
// construction - the Issue-715 class cannot arise. Dead chunks contribute
// exactly 0 (expf(-1e30 - M) = 0, l = 0). l_total = 0 (all chunks dead,
// i.e. n_positions = 0) writes 0 - matching the serial kernel's empty-cache
// semantics.
extern "C" __global__ void attention_decode_split_combine_f32(
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ part_out,
    float* __restrict__ attn_out,
    const int head_dim,
    const int n_chunks)
{
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const int base = head_idx * n_chunks;

    float m_max = -1e30f;
    for (int c = 0; c < n_chunks; c++) {
        if (part_m[base + c] > m_max) m_max = part_m[base + c];
    }

    float l_total = 0.0f;
    float out_acc = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        float w = expf(part_m[base + c] - m_max);
        l_total += part_l[base + c] * w;
        out_acc += w * part_out[(base + c) * head_dim + tid];
    }

    attn_out[head_idx * head_dim + tid] =
        (l_total == 0.0f) ? 0.0f : (out_acc / l_total);
}

// ===========================================================================
// Issue 742 T9.9 - the p-ROW batched verify family (the Q4_K speculative
// verify port). Every kernel reproduces its single-row parent's arithmetic
// VERBATIM per (row, element) - same expressions, same accumulation order -
// so a verify chunk's outputs are bit-identical per row to the decode path
// processing those positions sequentially.
// ===========================================================================

// Batched partial RoPE: row r rotates at absolute position base_pos + r.
// Per (row, head, pair) the math is VERBATIM rope_partial_f32.
extern "C" __global__ void rope_partial_rows_f32(
    float* __restrict__ q,           // [p, n_head * head_dim] (in-place)
    float* __restrict__ k,           // [p, n_kv_head * head_dim] (in-place)
    const int rotary_dim,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int base_pos,
    const int p,
    const float theta_base)
{
    const int rotary_pairs = rotary_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q_pairs = p * n_head * rotary_pairs;

    if (idx >= total_q_pairs) return;

    const int q_stride1 = n_head * rotary_pairs;
    const int row = idx / q_stride1;
    const int within = idx % q_stride1;
    const int head = within / rotary_pairs;
    const int pair = within % rotary_pairs;
    const int pos = base_pos + row;

    const float exponent = 2.0f * (float)pair / (float)rotary_dim;
    const float inv_freq = powf(theta_base, -exponent);
    const float theta = (float)pos * inv_freq;
    const float cos_t = cosf(theta);
    const float sin_t = sinf(theta);

    const int q_row_stride = n_head * head_dim;
    const int q_head_off = row * q_row_stride + head * head_dim;
    const int q_i0 = q_head_off + pair;
    const int q_i1 = q_i0 + rotary_pairs;
    const float q0 = q[q_i0];
    const float q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    if (head < n_kv_head) {
        const int k_row_stride = n_kv_head * head_dim;
        const int k_head_off = row * k_row_stride + head * head_dim;
        const int k_i0 = k_head_off + pair;
        const int k_i1 = k_i0 + rotary_pairs;
        const float k0 = k[k_i0];
        const float k1 = k[k_i1];
        k[k_i0] = k0 * cos_t - k1 * sin_t;
        k[k_i1] = k0 * sin_t + k1 * cos_t;
    }
}

// Issue 742 T9.12 - graph-capture twin of `rope_partial_rows_f32`: base_pos
// read from a device buffer at kernel runtime (the scalar arg would bake
// into the captured graph). Body identical - `base_pos = *pos_dev` then the
// scalar kernel's verbatim remainder.
extern "C" __global__ void rope_partial_rows_f32_devpos(
    float* __restrict__ q,           // [p, n_head * head_dim] (in-place)
    float* __restrict__ k,           // [p, n_kv_head * head_dim] (in-place)
    const int rotary_dim,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int* __restrict__ pos_dev,
    const int p,
    const float theta_base)
{
    const int base_pos = *pos_dev;
    const int rotary_pairs = rotary_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q_pairs = p * n_head * rotary_pairs;

    if (idx >= total_q_pairs) return;

    const int q_stride1 = n_head * rotary_pairs;
    const int row = idx / q_stride1;
    const int within = idx % q_stride1;
    const int head = within / rotary_pairs;
    const int pair = within % rotary_pairs;
    const int pos = base_pos + row;

    const float exponent = 2.0f * (float)pair / (float)rotary_dim;
    const float inv_freq = powf(theta_base, -exponent);
    const float theta = (float)pos * inv_freq;
    const float cos_t = cosf(theta);
    const float sin_t = sinf(theta);

    const int q_row_stride = n_head * head_dim;
    const int q_head_off = row * q_row_stride + head * head_dim;
    const int q_i0 = q_head_off + pair;
    const int q_i1 = q_i0 + rotary_pairs;
    const float q0 = q[q_i0];
    const float q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    if (head < n_kv_head) {
        const int k_row_stride = n_kv_head * head_dim;
        const int k_head_off = row * k_row_stride + head * head_dim;
        const int k_i0 = k_head_off + pair;
        const int k_i1 = k_i0 + rotary_pairs;
        const float k0 = k[k_i0];
        const float k1 = k[k_i1];
        k[k_i0] = k0 * cos_t - k1 * sin_t;
        k[k_i1] = k0 * sin_t + k1 * cos_t;
    }
}

// Batched KV append: row r writes cache row base_pos + r.
extern "C" __global__ void kv_cache_append_rows_f32(
    const float* __restrict__ k_vecs,   // [p, kvd]
    const float* __restrict__ v_vecs,   // [p, kvd]
    kv_elt* __restrict__ key_cache,      // [rows * kvd] absolute
    kv_elt* __restrict__ value_cache,
    const int kvd,
    const int base_pos,
    const int p)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * kvd;
    if (idx >= total) return;
    const int row = (int)(idx / kvd);
    const int col = (int)(idx % kvd);
    const long cache_off = (long)(base_pos + row) * kvd + col;
    KV_WR(&key_cache[cache_off], k_vecs[idx]);
    KV_WR(&value_cache[cache_off], v_vecs[idx]);
}

// Issue 742 T9.12 - graph-capture twin of `kv_cache_append_rows_f32`:
// base_pos read from a device buffer at kernel runtime. Body identical.
extern "C" __global__ void kv_cache_append_rows_f32_devpos(
    const float* __restrict__ k_vecs,   // [p, kvd]
    const float* __restrict__ v_vecs,   // [p, kvd]
    kv_elt* __restrict__ key_cache,      // [rows * kvd] absolute
    kv_elt* __restrict__ value_cache,
    const int kvd,
    const int* __restrict__ pos_dev,
    const int p)
{
    const int base_pos = *pos_dev;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * kvd;
    if (idx >= total) return;
    const int row = (int)(idx / kvd);
    const int col = (int)(idx % kvd);
    const long cache_off = (long)(base_pos + row) * kvd + col;
    KV_WR(&key_cache[cache_off], k_vecs[idx]);
    KV_WR(&value_cache[cache_off], v_vecs[idx]);
}

// Batched QG split - per element a verbatim copy of split_qg_f32 with the
// row offset added to the source index.
extern "C" __global__ void split_qg_rows_f32(
    const float* __restrict__ qg,   // [p, n_head * 2 * head_dim]
    float* __restrict__ q,          // [p, n_head * head_dim]
    float* __restrict__ gate,       // [p, n_head * head_dim]
    const int head_dim,
    const int n_head,
    const int p)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * n_head * head_dim;
    if (idx >= total) return;
    const int per_token = n_head * head_dim;
    const int row = (int)(idx / per_token);
    const int within = (int)(idx % per_token);
    const int head = within / head_dim;
    const int dim = within % head_dim;
    const long src = (long)row * n_head * 2 * head_dim + head * 2 * head_dim + dim;
    q[idx] = qg[src];
    gate[idx] = qg[src + head_dim];
}

// Issue 742 T9.9 - the multi-QUERY GQA split-KV partial: the decode
// splitgqa kernel's structure extended to ATT_ROWS_QPB consecutive query
// rows per block (grid.z = ceil(p / QPB)). K is staged TRANSPOSED once per
// tile and shared by all rows of the block; the P.V phase reads each V row
// ONCE (register) and applies it to every (row, head) pair. Per (query row,
// head) the arithmetic is VERBATIM the decode kernel's - warp h owns the
// head, lane = position, serial ascending-d dot from the transposed K tile,
// the same warp-shuffle max/sum, p = exp_tile * expf(score - m),
// run_sum = run_sum * exp_prev + s, acc *= st then ascending-l FMAs - so
// every partial is bit-identical to what the decode kernel produces for
// that query row alone, and the rows-combine reproduces the decode combine
// per row. Tiles beyond a shorter row's causal range are EXACT no-ops
// (all-masked scores keep p = 0; masked lanes contribute 0 to the warp
// sum), and a chunk entirely beyond a row's extent yields the neutral
// partial (-1e30, 0, 0) - the combine's dead-chunk semantics.
//
// Causal masking: position pos is valid for query row r iff
// pos <= base_pos + r (the decode kernel's n_positions = pos + 1 per row).
//
// Scratch layout: part_m/l [p, n_head, n_chunks], part_out
// [p, n_head, n_chunks, head_dim] - the decode layout with a row-major
// leading axis. Dynamic smem: k[hd][TILE+1] | p[QPB][g][TILE] | st[QPB][g].
#define ATT_ROWS_QPB 4

// Issue 742 T9.12 - the shared body: ONE source for the scalar-arg kernel
// and its devpos graph-capture twin (the in-file decode-splitgqa
// convention - keeps the twins provably equivalent).
template <int DOTMA>
static __device__ void attention_decode_splitgqa_rows_body_t(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    float* smem,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p,
    const int k4)
{
    const int n_positions = base_pos + p;
    const int kv_group = blockIdx.x;   // grid.x = n_kv_head
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int q_base = blockIdx.z * ATT_ROWS_QPB;
    const int q_count = (p - q_base) < ATT_ROWS_QPB ? (p - q_base) : ATT_ROWS_QPB;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = n_head / n_kv_head;

    float* k_smem = smem;                                            // hd*(TILE+1)
    float* p_smem = k_smem + head_dim * (ATT_SPLIT_GQA_TILE + 1);    // QPB*g*TILE
    float* st_smem = p_smem + ATT_ROWS_QPB * g * ATT_SPLIT_GQA_TILE; // QPB*g

    const int chunk_start = chunk_id * chunk_len;
    if (chunk_start >= n_positions) {
        // Dead chunk - neutral partials for this block's query rows
        // (uniform early exit, before any barrier).
        for (int ri = 0; ri < q_count; ri++) {
            const int r = q_base + ri;
            for (int h = 0; h < g; h++) {
                const int p_idx = (r * n_head + kv_group * g + h) * n_chunks + chunk_id;
                if (tid == 0) {
                    part_m[p_idx] = -1e30f;
                    part_l[p_idx] = 0.0f;
                }
                if (tid < head_dim) part_out[p_idx * head_dim + tid] = 0.0f;
            }
        }
        return;
    }
    const int chunk_end_raw = chunk_start + chunk_len;
    const int chunk_end = (chunk_end_raw < n_positions) ? chunk_end_raw : n_positions;

    const int kv_stride = n_kv_head * head_dim;
    const int kv_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;

    // Warp-local online-softmax state: warps [0, g) own one q-head each for
    // ALL QPB query rows of the block.
    float run_max[ATT_ROWS_QPB];
    float run_sum[ATT_ROWS_QPB];
#pragma unroll
    for (int ri = 0; ri < ATT_ROWS_QPB; ri++) {
        run_max[ri] = -1e30f;
        run_sum[ri] = 0.0f;
    }
    // Every thread owns output dim `tid` for all (ri, h) pairs. g <= 8
    // (launcher contract).
    float acc[ATT_ROWS_QPB][8];
#pragma unroll
    for (int ri = 0; ri < ATT_ROWS_QPB; ri++)
#pragma unroll
        for (int h = 0; h < 8; h++) acc[ri][h] = 0.0f;

    const int tile0 = chunk_start / ATT_SPLIT_GQA_TILE;
    const int tile1 = (chunk_end + ATT_SPLIT_GQA_TILE - 1) / ATT_SPLIT_GQA_TILE;
    // Issue 742 T9.16 - k4=1 stages K ROW-MAJOR (stride head_dim+4) so the
    // score dot reads each lane's K row CONTIGUOUSLY as float4 (LDS.128,
    // 1 load per 4 dims vs 4 LDS.32 - the T9.15 load-issue cut applied to
    // the K half of the dot). Values BIT-IDENTICAL to the transposed
    // layout: the same K[pos][d] elements feed the same FMAs in the same
    // fold order - only the smem layout and load width change. The
    // head_dim+4 stride keeps 16B alignment (head_dim % 4 == 0, launcher
    // contract) and conflict-free LDS.128 ((head_dim/4 + 1) % 8 == 1 for
    // every head_dim that is a multiple of 32 - 64/128/256 all satisfy
    // it). The row-major tile's footprint (32*(head_dim+4) floats incl.
    // pad) fits the transposed head_dim*(TILE+1) envelope only for
    // head_dim >= 128 (Issue 779: below that the staging overflows into
    // p_smem and phase B races its row-0 score writes against the
    // overflowed K reads). The LAUNCHER therefore passes k4=1 only when
    // the envelope fits (verify_kf4_for) - smaller geometries take the
    // bit-identical transposed arm and the p_smem/st_smem offsets stay
    // shared by both arms.
    const int kstride = head_dim + 4;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * ATT_SPLIT_GQA_TILE;

        // (A) Stage K tile - once per tile, shared by every query row of
        // the block (k4: row-major; else VERBATIM the decode kernel's
        // transposed stage).
        if (k4) {
            for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
                const int r_ = i / head_dim;
                const int d = i % head_dim;
                k_smem[r_ * kstride + d] =
                    KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
            }
        } else {
            for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
                const int r_ = i / head_dim;
                const int d = i % head_dim;
                k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + r_] =
                    KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
            }
        }
        __syncthreads();

        // (B) Score phase - per query row ri, VERBATIM the decode kernel's
        // warp arithmetic (the causal extent is per-row).
        if (warp < g) {
            const float* qh_base =
                query + (long)q_base * q_stride + (kv_group * g + warp) * head_dim;
#pragma unroll
            for (int ri = 0; ri < ATT_ROWS_QPB; ri++) {
                if (ri < q_count) {
                    const int r = q_base + ri;
                    const int pos = tile_base + lane;
                    const bool valid = (pos < chunk_end) && (pos <= base_pos + r);
                    float score = -1e30f;
                    if (valid) {
                        const float* qh = qh_base + (long)ri * q_stride;
                        float dot = 0.0f;
                        if (DOTMA) {
                            // Issue 742 T9.15 - the serial-FMA chain break:
                            // FOUR independent 64-deep partial accumulators,
                            // fixed ascending fold. head_dim == 256 is the
                            // launcher contract (compile-time bound, the
                            // Issue-706 rule). Tolerance class (reassociation
                            // - only the fold order differs from serial).
                            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                            if (k4) {
                                // T9.16 - float4 K loads from the row-major
                                // tile (the K-side twin of the qh float4);
                                // SAME fold order as the transposed form.
#pragma unroll
                                for (int d4 = 0; d4 < 64; d4++) {
                                    const float4 qv =
                                        *reinterpret_cast<const float4*>(qh + d4 * 4);
                                    const float4 kv =
                                        *reinterpret_cast<const float4*>(k_smem + lane * kstride + d4 * 4);
                                    p0 += qv.x * kv.x;
                                    p1 += qv.y * kv.y;
                                    p2 += qv.z * kv.z;
                                    p3 += qv.w * kv.w;
                                }
                            } else {
                                const int ks = ATT_SPLIT_GQA_TILE + 1;
#pragma unroll
                                for (int d4 = 0; d4 < 64; d4++) {
                                    // float4 qh loads: 1 LDG.128 per 4 dims vs 4
                                    // LDG.32 (the issue-count cut — the chain
                                    // break alone measured a wash; the phase is
                                    // load-issue-bound). qh is 16B-aligned (q
                                    // strides are head_dim multiples of 256).
                                    // SAME fold order as the scalar-load form —
                                    // bit-identical arithmetic, load width only.
                                    const float4 qv =
                                        *reinterpret_cast<const float4*>(qh + d4 * 4);
                                    p0 += qv.x * k_smem[(d4 * 4 + 0) * ks + lane];
                                    p1 += qv.y * k_smem[(d4 * 4 + 1) * ks + lane];
                                    p2 += qv.z * k_smem[(d4 * 4 + 2) * ks + lane];
                                    p3 += qv.w * k_smem[(d4 * 4 + 3) * ks + lane];
                                }
                            }
                            dot = ((p0 + p1) + p2) + p3;
                        } else if (k4) {
                            // T9.16 - serial fold, float4 loads: the adds stay
                            // STRICTLY sequential ascending-d on the single
                            // accumulator - bit-identical to the scalar form.
                            for (int d4 = 0; d4 < (head_dim >> 2); d4++) {
                                const float4 qv =
                                    *reinterpret_cast<const float4*>(qh + d4 * 4);
                                const float4 kv =
                                    *reinterpret_cast<const float4*>(k_smem + lane * kstride + d4 * 4);
                                dot += qv.x * kv.x;
                                dot += qv.y * kv.y;
                                dot += qv.z * kv.z;
                                dot += qv.w * kv.w;
                            }
                        } else {
                            for (int d = 0; d < head_dim; d++) {
                                dot += qh[d] * k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + lane];
                            }
                        }
                        score = dot * scale;
                    }
                    float m = score;
                    for (int off = 16; off > 0; off >>= 1) {
                        const float o = __shfl_down_sync(0xffffffffu, m, off);
                        if (o > m) m = o;
                    }
                    m = __shfl_sync(0xffffffffu, m, 0);
                    const float new_max = (m > run_max[ri]) ? m : run_max[ri];
                    const float exp_prev = expf(run_max[ri] - new_max);
                    const float exp_tile = expf(m - new_max);
                    float pv = 0.0f;
                    if (valid) pv = exp_tile * expf(score - m);
                    float s = pv;
                    for (int off = 16; off > 0; off >>= 1) {
                        s += __shfl_down_sync(0xffffffffu, s, off);
                    }
                    s = __shfl_sync(0xffffffffu, s, 0);
                    run_sum[ri] = run_sum[ri] * exp_prev + s;
                    run_max[ri] = new_max;
                    p_smem[(ri * g + warp) * ATT_SPLIT_GQA_TILE + lane] = pv;
                    if (lane == 0) st_smem[ri * g + warp] = exp_prev;
                }
            }
        }
        __syncthreads();

        // (C) P.V phase - every thread owns dim `tid`; the V row is loaded
        // coalesced ONCE per (tile, l) and reused for all (ri, h).
        const int l_valid = (chunk_end - tile_base < ATT_SPLIT_GQA_TILE)
                                ? (chunk_end - tile_base)
                                : ATT_SPLIT_GQA_TILE;
#pragma unroll
        for (int ri = 0; ri < ATT_ROWS_QPB; ri++) {
            if (ri < q_count) {
#pragma unroll
                for (int h = 0; h < 8; h++) {
                    if (h < g) acc[ri][h] *= st_smem[ri * g + h];
                }
            }
        }
        for (int l = 0; l < l_valid; l++) {
            // Issue 779 hardening: blockDim is 256 regardless of head_dim, so
            // at fixture geometries (head_dim < 256) threads tid >= head_dim
            // would read past this head's V row (past the cache allocation on
            // the last kv head). Their acc lanes are dead (the partial write
            // is tid < head_dim), so they contribute nothing - just don't
            // read. Production head_dim 256 keeps the load unconditional.
            const float v = (tid < head_dim)
                ? KV_RD(&value_cache[(tile_base + l) * kv_stride + kv_off + tid])
                : 0.0f;
#pragma unroll
            for (int ri = 0; ri < ATT_ROWS_QPB; ri++) {
                if (ri < q_count) {
                    const float* prow = p_smem + (ri * g) * ATT_SPLIT_GQA_TILE + l;
#pragma unroll
                    for (int h = 0; h < 8; h++) {
                        if (h < g) acc[ri][h] += prow[h * ATT_SPLIT_GQA_TILE] * v;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write the block's partials (unnormalized - the combine owns 1/l).
    if (tid < head_dim) {
#pragma unroll
        for (int ri = 0; ri < ATT_ROWS_QPB; ri++) {
            if (ri < q_count) {
                const int r = q_base + ri;
#pragma unroll
                for (int h = 0; h < 8; h++) {
                    if (h < g) {
                        const int p_idx =
                            (r * n_head + kv_group * g + h) * n_chunks + chunk_id;
                        part_out[p_idx * head_dim + tid] = acc[ri][h];
                    }
                }
            }
        }
    }
    if (warp < g && lane == 0) {
#pragma unroll
        for (int ri = 0; ri < ATT_ROWS_QPB; ri++) {
            if (ri < q_count) {
                const int r = q_base + ri;
                const int p_idx =
                    (r * n_head + kv_group * g + warp) * n_chunks + chunk_id;
                part_m[p_idx] = run_max[ri];
                part_l[p_idx] = run_sum[ri];
            }
        }
    }}

extern "C" __global__ void attention_decode_splitgqa_partial_rows_f32(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    attention_decode_splitgqa_rows_body_t<0>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.12 - graph-capture twin: base_pos read from a device buffer
// at kernel runtime; the grid is pinned to the FIXED max chunk count at
// capture time (dead chunks write neutral partials - the over-launch
// contract in the body; every partial slot is rewritten on every launch).
extern "C" __global__ void attention_decode_splitgqa_partial_rows_f32_devpos(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_decode_splitgqa_rows_body_t<0>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.15 - the multi-accumulator dot twins of the T9.9 rows
// kernel: the score-phase Q dot K single 256-deep serial-FMA accumulator
// becomes FOUR independent 64-deep partial chains (DOTMA=1 - see the
// templated body). The T9.14 re-diagnosis: the verify walk is bound by
// the per-lane serial-FMA dot chains, NOT the cross-tile running-max
// dependency. Tolerance class vs the serial kernels (fold-order
// reassociation ~1e-6, the T9.13 finer-chunk precedent); gated by
// `qwen38_verify_qgma_g1` + the model-level 0/256 argmax / loop-stream
// gates. Serial kernels stay the default unless the A/B wins
// (QWEN38_VERIFY_ATTN_DOTMA).
extern "C" __global__ void attention_decode_splitgqa_partial_rows_ma_f32(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    attention_decode_splitgqa_rows_body_t<1>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.12/T9.15 - graph-capture twin (base_pos via pos_dev; the
// pinned max chunk grid with neutral dead-chunk partials).
extern "C" __global__ void attention_decode_splitgqa_partial_rows_ma_f32_devpos(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_decode_splitgqa_rows_body_t<1>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}


// Issue 742 T9.9 - the rows combine: attention_decode_split_combine_f32's
// body per (row, head) with grid (p * n_head); part arrays are the rows
// layout [p, n_head, n_chunks] and the output is [p, n_head * head_dim].
extern "C" __global__ void attention_decode_split_combine_rows_f32(
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ part_out,
    float* __restrict__ attn_out,
    const int head_dim,
    const int n_head,
    const int n_chunks,
    const int p)
{
    const int row_head = blockIdx.x;
    const int r = row_head / n_head;
    const int head_idx = row_head % n_head;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const int base = row_head * n_chunks;

    float m_max = -1e30f;
    for (int c = 0; c < n_chunks; c++) {
        if (part_m[base + c] > m_max) m_max = part_m[base + c];
    }

    float l_total = 0.0f;
    float out_acc = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        float w = expf(part_m[base + c] - m_max);
        l_total += part_l[base + c] * w;
        out_acc += w * part_out[(base + c) * head_dim + tid];
    }

    attn_out[(long)r * n_head * head_dim + head_idx * head_dim + tid] =
        (l_total == 0.0f) ? 0.0f : (out_acc / l_total);
}

// Issue 742 T9.11 — the q-group-major rows partial: ONE block per
// (kv_head, chunk) serving ALL p <= 16 query rows. The T9.9 kernel's
// grid.z = ceil(p/4) re-staged K and re-read V once per 4-row group —
// at 20K positions that quadrupled K/V traffic measured ~2x off the
// K/V floor. Here 1024 threads = 4 row-groups of 256 (= head_dim);
// row-group w owns rows [w*4, w*4+4) in phase C, and phase B spreads
// the 16*g (row, head) score tasks over all 32 warps (task
// warp + u*32, u < 4 — compile-time slots, the Issue-706 rule). Per
// (query row, head, dim) the arithmetic and its order are VERBATIM the
// T9.9 kernel: lane-per-position serial ascending-d dot over the
// shared transposed K tile, the same warp-shuffle max/sum,
// p = exp_tile * expf(score - m), the same online update, acc *= st
// then ascending-l FMAs — so every partial is bit-identical and the
// unchanged rows combine reproduces the decode combine per row.
// Contract: head_dim == 256, g <= 8, p <= 16 (launcher validates;
// other geometries route to the T9.9 kernel). Scratch layout identical.
// Dynamic smem: k[hd][TILE+1] | p[16*g][TILE] | st[16*g].
// Issue 742 T9.12 - the shared body (ONE source for the scalar-arg kernel
// and its devpos graph-capture twin). Issue 742 T9.15 - templated on DOTMA
// (see the rows body note): <0> serial dot (the T9.11 default), <1> the
// 4-partial multi-accumulator dot (the serial-FMA chain break).
template <int DOTMA>
static __device__ void attention_decode_splitgqa_rows_qg_body_t(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    float* smem,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p,
    const int k4)
{
    const int n_positions = base_pos + p;
    const int kv_group = blockIdx.x;   // grid.x = n_kv_head
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp = tid >> 8;      // row group 0..4 (256 threads each)
    const int dim = tid & 255;     // output dim (head_dim == 256)
    const int g = n_head / n_kv_head;
    // Issue 754 T6 — grid.z row-groups: block z owns the 16-row slice
    // [z*16, z*16+16); the launcher sets grid.z = ceil(p/16). p <= 16 keeps
    // grid.z == 1 and row_off == 0 (launch-config- and bit-identical to the
    // pre-T6 kernel). row_lim is the RELATIVE row cap of this block
    // (= 16 on every non-last z; < 16 only on the ragged tail).
    const int row_off = blockIdx.z * 16;
    const int row_lim = p - row_off;

    float* k_smem = smem;                                          // hd*(TILE+1)
    float* p_smem = k_smem + head_dim * (ATT_SPLIT_GQA_TILE + 1);  // 16*g*TILE
    float* st_smem = p_smem + 16 * g * ATT_SPLIT_GQA_TILE;         // 16*g

    const int chunk_start = chunk_id * chunk_len;
    if (chunk_start >= n_positions) {
        // Dead chunk — neutral partials (uniform early exit, pre-barrier;
        // the exact-launch path never sees this, the over-launched
        // graph-style grid does — dead slots contribute exactly 0).
        // T6: each grid.z row-group zeroes ONLY its own 16-row slice —
        // the z-blocks write DISJOINT row ranges (no redundant cross-z
        // stores at p > 16).
        if (tid < 16 * g) {
            const int r = row_off + tid / g;
            const int h = tid % g;
            if (r < p) {
                const int p_idx = (r * n_head + kv_group * g + h) * n_chunks + chunk_id;
                part_m[p_idx] = -1e30f;
                part_l[p_idx] = 0.0f;
            }
        }
        for (int i = tid; i < 16 * g * head_dim; i += blockDim.x) {
            const int d_ = i % head_dim;
            const int t = i / head_dim;
            const int r = row_off + t / g;
            const int h = t % g;
            if (r < p) {
                part_out[((r * n_head + kv_group * g + h) * n_chunks + chunk_id) * head_dim
                         + d_] = 0.0f;
            }
        }
        return;
    }
    const int chunk_end_raw = chunk_start + chunk_len;
    const int chunk_end = (chunk_end_raw < n_positions) ? chunk_end_raw : n_positions;

    const int kv_stride = n_kv_head * head_dim;
    const int kv_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;

    // Warp-local online-softmax state: warp w owns score tasks w, w+32,
    // w+64, w+96 (compile-time slots — the Issue-706 local-memory rule).
    float run_max[4];
    float run_sum[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
        run_max[u] = -1e30f;
        run_sum[u] = 0.0f;
    }
    // Row-group accumulators: rows [grp*4, grp*4+4) x g heads (g <= 8).
    float acc[4][8];
#pragma unroll
    for (int u = 0; u < 4; u++)
#pragma unroll
        for (int h = 0; h < 8; h++) acc[u][h] = 0.0f;

    const int tile0 = chunk_start / ATT_SPLIT_GQA_TILE;
    const int tile1 = (chunk_end + ATT_SPLIT_GQA_TILE - 1) / ATT_SPLIT_GQA_TILE;
    // T9.16 k4: see the rows body note (row-major K tile, float4 score
    // loads; the envelope fits only at head_dim >= 128 — the qg contract
    // pins head_dim == 256, so the launcher's verify_kf4_for always
    // admits k4 here; shared p/st offsets).
    const int kstride = head_dim + 4;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * ATT_SPLIT_GQA_TILE;

        // (A) Stage K tile — once per tile for ALL 16 query rows (the
        // amortization this kernel exists for); k4: row-major, else the
        // transposed stage verbatim.
        if (k4) {
            for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
                const int r_ = i / head_dim;
                const int d = i % head_dim;
                k_smem[r_ * kstride + d] =
                    KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
            }
        } else {
            for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
                const int r_ = i / head_dim;
                const int d = i % head_dim;
                k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + r_] =
                    KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
            }
        }
        __syncthreads();

        // (B) Score phase — task t = (row, head) spread over all 32
        // warps; per task VERBATIM the decode kernel's warp arithmetic
        // (the causal extent is per-row). T6: t maps to the RELATIVE row
        // of this grid.z block; qh/causality use the ABSOLUTE row.
        {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int t = warp + u * 32;
                if (t < 16 * g) {
                    const int r = t / g;
                    const int h = t % g;
                    if (r < row_lim) {
                        const float* qh = query
                            + (long)(row_off + r) * q_stride
                            + (kv_group * g + h) * head_dim;
                        const int pos = tile_base + lane;
                        const bool valid =
                            (pos < chunk_end) && (pos <= base_pos + row_off + r);
                        float score = -1e30f;
                        if (valid) {
                            float dot = 0.0f;
                            if (DOTMA) {
                                // Issue 742 T9.15 - the serial-FMA chain break
                                // (see the rows body note): FOUR independent
                                // 64-deep partial accumulators, fixed ascending
                                // fold; head_dim == 256 is the launcher
                                // contract (compile-time bound, Issue-706
                                // rule). Same total FMA work - only the fold
                                // order differs (tolerance class).
                                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                                if (k4) {
                                    // T9.16 - float4 K loads (the K-side twin
                                    // of the qh float4); SAME fold order.
#pragma unroll
                                    for (int d4 = 0; d4 < 64; d4++) {
                                        const float4 qv =
                                            *reinterpret_cast<const float4*>(qh + d4 * 4);
                                        const float4 kv = *reinterpret_cast<const float4*>(
                                            k_smem + lane * kstride + d4 * 4);
                                        p0 += qv.x * kv.x;
                                        p1 += qv.y * kv.y;
                                        p2 += qv.z * kv.z;
                                        p3 += qv.w * kv.w;
                                    }
                                } else {
                                    const int ks = ATT_SPLIT_GQA_TILE + 1;
#pragma unroll
                                    for (int d4 = 0; d4 < 64; d4++) {
                                        // float4 qh loads (see the rows body
                                        // note): the issue-count cut; SAME fold
                                        // order as the scalar-load form.
                                        const float4 qv =
                                            *reinterpret_cast<const float4*>(qh + d4 * 4);
                                        p0 += qv.x * k_smem[(d4 * 4 + 0) * ks + lane];
                                        p1 += qv.y * k_smem[(d4 * 4 + 1) * ks + lane];
                                        p2 += qv.z * k_smem[(d4 * 4 + 2) * ks + lane];
                                        p3 += qv.w * k_smem[(d4 * 4 + 3) * ks + lane];
                                    }
                                }
                                dot = ((p0 + p1) + p2) + p3;
                            } else if (k4) {
                                // T9.16 - serial fold, float4 loads; strictly
                                // sequential ascending-d adds - bit-identical
                                // to the scalar form.
                                for (int d4 = 0; d4 < (head_dim >> 2); d4++) {
                                    const float4 qv =
                                        *reinterpret_cast<const float4*>(qh + d4 * 4);
                                    const float4 kv = *reinterpret_cast<const float4*>(
                                        k_smem + lane * kstride + d4 * 4);
                                    dot += qv.x * kv.x;
                                    dot += qv.y * kv.y;
                                    dot += qv.z * kv.z;
                                    dot += qv.w * kv.w;
                                }
                            } else {
                                for (int d = 0; d < head_dim; d++) {
                                    dot += qh[d] * k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + lane];
                                }
                            }
                            score = dot * scale;
                        }
                        float m = score;
                        for (int off = 16; off > 0; off >>= 1) {
                            const float o = __shfl_down_sync(0xffffffffu, m, off);
                            if (o > m) m = o;
                        }
                        m = __shfl_sync(0xffffffffu, m, 0);
                        const float new_max = (m > run_max[u]) ? m : run_max[u];
                        const float exp_prev = expf(run_max[u] - new_max);
                        const float exp_tile = expf(m - new_max);
                        float pv = 0.0f;
                        if (valid) pv = exp_tile * expf(score - m);
                        float s = pv;
                        for (int off = 16; off > 0; off >>= 1) {
                            s += __shfl_down_sync(0xffffffffu, s, off);
                        }
                        s = __shfl_sync(0xffffffffu, s, 0);
                        run_sum[u] = run_sum[u] * exp_prev + s;
                        run_max[u] = new_max;
                        p_smem[t * ATT_SPLIT_GQA_TILE + lane] = pv;
                        if (lane == 0) st_smem[t] = exp_prev;
                    }
                }
            }
        }
        __syncthreads();

        // (C) P.V phase — thread owns dim `dim`, its row-group's 4 rows x
        // g heads; the V row is loaded per (tile, l) (L1 broadcast across
        // the 4 row-groups), ascending-l FMAs per (row, head, dim).
        const int l_valid = (chunk_end - tile_base < ATT_SPLIT_GQA_TILE)
                                ? (chunk_end - tile_base)
                                : ATT_SPLIT_GQA_TILE;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int r = grp * 4 + u;   // RELATIVE row of this z-block
            if (r < row_lim) {
#pragma unroll
                for (int h = 0; h < 8; h++) {
                    if (h < g) acc[u][h] *= st_smem[r * g + h];
                }
            }
        }
        for (int l = 0; l < l_valid; l++) {
            const float v = KV_RD(&value_cache[(tile_base + l) * kv_stride + kv_off + dim]);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int r = grp * 4 + u;   // RELATIVE row of this z-block
                if (r < row_lim) {
#pragma unroll
                    for (int h = 0; h < 8; h++) {
                        if (h < g) {
                            acc[u][h] += p_smem[(r * g + h) * ATT_SPLIT_GQA_TILE + l] * v;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write the block's partials (unnormalized — the combine owns 1/l).
    // Row-groups write DISJOINT row ranges; every thread writes its dim.
    // T6: ABSOLUTE row (row_off + r) in the partials index.
    {
        const int r0 = grp * 4;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int r = r0 + u;
            if (r < row_lim) {
#pragma unroll
                for (int h = 0; h < 8; h++) {
                    if (h < g) {
                        const int p_idx =
                            ((row_off + r) * n_head + kv_group * g + h) * n_chunks + chunk_id;
                        part_out[p_idx * head_dim + dim] = acc[u][h];
                    }
                }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int t = warp + u * 32;
            if (t < 16 * g) {
                const int r = t / g;
                const int h = t % g;
                if (r < row_lim) {
                    const int p_idx =
                        ((row_off + r) * n_head + kv_group * g + h) * n_chunks + chunk_id;
                    part_m[p_idx] = run_max[u];
                    part_l[p_idx] = run_sum[u];
                }
            }
        }
    }}


extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qg_f32(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    attention_decode_splitgqa_rows_qg_body_t<0>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.12 - graph-capture twin: base_pos read from a device buffer
// at kernel runtime; the grid is pinned to the FIXED max chunk count at
// capture time (dead chunks write neutral partials). Same smem geometry as
// the scalar kernel (the launcher sets the opt-in attribute on BOTH).
extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qg_f32_devpos(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_decode_splitgqa_rows_qg_body_t<0>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.15 - the multi-accumulator dot twins of the T9.11 qg
// kernel (DOTMA=1 - see the templated qg body): the ONLY change vs the
// qg kernels is the score-dot fold order. Tolerance class (reassociation
// ~1e-6); the T9.12 graph machinery captures whichever arm the
// QWEN38_VERIFY_ATTN_DOTMA knob resolved at construction (the capture
// key (p, use_qg) is unchanged - the arm is process-fixed).
extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qgma_f32(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    attention_decode_splitgqa_rows_qg_body_t<1>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.12/T9.15 - graph-capture twin (same smem geometry - the
// launcher sets the opt-in attribute on BOTH).
extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qgma_f32_devpos(
    const float* __restrict__ query,      // [p, n_head * head_dim]
    const kv_elt* __restrict__ key_cache,  // [(base_pos+p), n_kv * head_dim]
    const kv_elt* __restrict__ value_cache,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev,
    const int p,
    const int k4)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_decode_splitgqa_rows_qg_body_t<1>(query, key_cache, value_cache,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p, k4);
}

// Issue 742 T9.14 — the TWO-PASS flash restructure of the verify-attention
// walk (the qg arm's serial running-max chain removed). Pass A
// (`_stats_`) computes per-(row, head) flash stats (m_tile, s_tile) for
// EVERY 32-position tile — fully tile-parallel, ONE block per
// (kv_head, tile), no cross-tile state. The merge kernel folds the tile
// stats into the frozen global (M, L) per (row, head) (ascending tile
// order — deterministic; dead tiles carry the neutral (-1e30, 0) partial
// so the pinned over-launched grid and the exact grid merge
// bit-identically). Pass B (`_pv_`) re-walks its chunk's tiles under the
// FROZEN M: p = expf(score - M) directly (no online max, no per-tile acc
// rescale — the serial dependency chain that pinned the qg kernel at
// ~100 GB/s is gone by construction) and writes UNNORMALIZED per-chunk
// partials with part_m = M so the UNCHANGED rows combine
// (`expf(M - M) = 1`) reduces to a plain ascending-c sum + 1/L divide.
// Tolerance-class vs the qg arm (frozen-max reassociation — a larger
// delta than T9.13's finer-chunk boundary move); gated by the dedicated
// unit test (max-rel bound + argmax identity on golden fixtures) + the
// model-level G1 (0/256 argmax). Contract: identical to the qg kernel
// (head_dim == 256, g <= 8, p <= 16); grids pinned for graph capture
// (dead tiles/chunks write neutral partials — every live slot rewritten
// on every launch, the T9.12 rollback-safety property).
static __device__ void attention_verify2p_stats_body(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    float* __restrict__ stat_m,   // [p*n_head * n_tiles]
    float* __restrict__ stat_l,
    float* smem,                  // k[hd][TILE+1]
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int n_tiles,            // stride == gridDim.y (pinned or exact)
    const int base_pos,
    const int p)
{
    const int n_positions = base_pos + p;
    const int kv_group = blockIdx.x;   // grid.x = n_kv_head
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = n_head / n_kv_head;

    float* k_smem = smem;
    const int tile_base = tile * ATT_SPLIT_GQA_TILE;
    if (tile_base >= n_positions) {
        // Dead tile (over-launched pinned grid) — neutral stats.
        if (tid < p * g) {
            const int r = tid / g;
            const int h = tid % g;
            const int idx = (r * n_head + kv_group * g + h) * n_tiles + tile;
            stat_m[idx] = -1e30f;
            stat_l[idx] = 0.0f;
        }
        return;
    }
    const int tile_end_raw = tile_base + ATT_SPLIT_GQA_TILE;
    const int tile_end = (tile_end_raw < n_positions) ? tile_end_raw : n_positions;

    const int kv_stride = n_kv_head * head_dim;
    const int q_stride = n_head * head_dim;

    // Stage K tile TRANSPOSED (verbatim qg phase A — once for all tasks).
    for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
        const int r_ = i / head_dim;
        const int d = i % head_dim;
        k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + r_] =
            KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_group * head_dim + d]);
    }
    __syncthreads();

    // Score phase — the qg phase-B warp arithmetic VERBATIM minus the
    // running state: per task the tile-local (m, s) flash stats.
#pragma unroll
    for (int u = 0; u < 4; u++) {
        const int t = warp + u * 32;
        if (t < 16 * g) {
            const int r = t / g;
            const int h = t % g;
            if (r < p) {
                const float* qh =
                    query + (long)r * q_stride + (kv_group * g + h) * head_dim;
                const int pos = tile_base + lane;
                const bool valid = (pos < tile_end) && (pos <= base_pos + r);
                float score = -1e30f;
                if (valid) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += qh[d] * k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + lane];
                    }
                    score = dot * scale;
                }
                float m = score;
                for (int off = 16; off > 0; off >>= 1) {
                    const float o = __shfl_down_sync(0xffffffffu, m, off);
                    if (o > m) m = o;
                }
                m = __shfl_sync(0xffffffffu, m, 0);
                float pv = 0.0f;
                if (valid) pv = expf(score - m);
                float s = pv;
                for (int off = 16; off > 0; off >>= 1) {
                    s += __shfl_down_sync(0xffffffffu, s, off);
                }
                s = __shfl_sync(0xffffffffu, s, 0);
                if (lane == 0) {
                    const int idx = (r * n_head + kv_group * g + h) * n_tiles + tile;
                    stat_m[idx] = m;
                    stat_l[idx] = s;
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(1024)
attention_verify2p_stats_f32(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    float* __restrict__ stat_m,
    float* __restrict__ stat_l,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int n_tiles,
    const int base_pos,
    const int p)
{
    extern __shared__ float smem[];
    attention_verify2p_stats_body(query, key_cache, stat_m, stat_l, smem,
        scale, head_dim, n_head, n_kv_head, n_tiles, base_pos, p);
}

// T9.14 graph-capture twin: base_pos from pos_dev; the grid (and the
// stat stride) pinned to the max tile count.
extern "C" __global__ void __launch_bounds__(1024)
attention_verify2p_stats_f32_devpos(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    float* __restrict__ stat_m,
    float* __restrict__ stat_l,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int n_tiles,
    const int* __restrict__ pos_dev,
    const int p)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_verify2p_stats_body(query, key_cache, stat_m, stat_l, smem,
        scale, head_dim, n_head, n_kv_head, n_tiles, base_pos, p);
}

// T9.14 pass A→B fold: per (kv_head block, task thread) reduce the tile
// stats into the frozen (M, L). No position dependence (dead tiles are
// neutral in the stats) — no devpos twin needed. Ascending-tile order —
// deterministic across replays.
extern "C" __global__ void attention_verify2p_merge_f32(
    const float* __restrict__ stat_m,
    const float* __restrict__ stat_l,
    float* __restrict__ mrg_m,    // [p*n_head]
    float* __restrict__ mrg_l,
    const int n_head,
    const int n_kv_head,
    const int n_tiles,
    const int p)
{
    const int kv_group = blockIdx.x;
    const int g = n_head / n_kv_head;
    const int t = threadIdx.x;
    if (t >= p * g) return;
    const int r = t / g;
    const int h = t % g;
    const int head = kv_group * g + h;
    const int base = (r * n_head + head) * n_tiles;
    float m = -1e30f;
    for (int i = 0; i < n_tiles; i++) {
        if (stat_m[base + i] > m) m = stat_m[base + i];
    }
    float l = 0.0f;
    for (int i = 0; i < n_tiles; i++) {
        l += stat_l[base + i] * expf(stat_m[base + i] - m);
    }
    mrg_m[r * n_head + head] = m;
    mrg_l[r * n_head + head] = l;
}

static __device__ void attention_verify2p_pv_body(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    const float* __restrict__ mrg_m,   // [p*n_head] frozen max
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    float* smem,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p)
{
    const int n_positions = base_pos + p;
    const int kv_group = blockIdx.x;   // grid.x = n_kv_head
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;    // stride (pinned or exact)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp = tid >> 8;      // row group 0..4 (256 threads each)
    const int dim = tid & 255;     // output dim (head_dim == 256)
    const int g = n_head / n_kv_head;

    float* k_smem = smem;                                          // hd*(TILE+1)
    float* p_smem = k_smem + head_dim * (ATT_SPLIT_GQA_TILE + 1);  // 16*g*TILE

    const int chunk_start = chunk_id * chunk_len;
    if (chunk_start >= n_positions) {
        // Dead chunk — neutral partials (verbatim qg dead branch).
        if (tid < p * g) {
            const int r = tid / g;
            const int h = tid % g;
            const int p_idx = (r * n_head + kv_group * g + h) * n_chunks + chunk_id;
            part_m[p_idx] = -1e30f;
            part_l[p_idx] = 0.0f;
        }
        for (int i = tid; i < p * g * head_dim; i += blockDim.x) {
            const int d_ = i % head_dim;
            const int t = i / head_dim;
            const int r = t / g;
            const int h = t % g;
            part_out[((r * n_head + kv_group * g + h) * n_chunks + chunk_id) * head_dim + d_] =
                0.0f;
        }
        return;
    }
    const int chunk_end_raw = chunk_start + chunk_len;
    const int chunk_end = (chunk_end_raw < n_positions) ? chunk_end_raw : n_positions;

    const int kv_stride = n_kv_head * head_dim;
    const int kv_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;

    // Frozen max per warp u-slot task (loaded once; L1-resident).
    float m_frozen[4];
    float s_acc[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
        m_frozen[u] = -1e30f;
        s_acc[u] = 0.0f;
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
        const int t = warp + u * 32;
        if (t < 16 * g) {
            const int r = t / g;
            const int h = t % g;
            if (r < p) m_frozen[u] = mrg_m[r * n_head + kv_group * g + h];
        }
    }
    // Row-group accumulators: rows [grp*4, grp*4+4) x g heads (g <= 8).
    float acc[4][8];
#pragma unroll
    for (int u = 0; u < 4; u++)
#pragma unroll
        for (int h = 0; h < 8; h++) acc[u][h] = 0.0f;

    const int tile0 = chunk_start / ATT_SPLIT_GQA_TILE;
    const int tile1 = (chunk_end + ATT_SPLIT_GQA_TILE - 1) / ATT_SPLIT_GQA_TILE;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * ATT_SPLIT_GQA_TILE;

        // (A) Stage K tile TRANSPOSED (verbatim qg phase A).
        for (int i = tid; i < ATT_SPLIT_GQA_TILE * head_dim; i += blockDim.x) {
            const int r_ = i / head_dim;
            const int d = i % head_dim;
            k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + r_] =
                KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
        }
        __syncthreads();

        // (B) Score phase under the FROZEN max: p = expf(score - M)
        // directly — no online max, no rescale state (the T9.14 core).
        {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int t = warp + u * 32;
                if (t < 16 * g) {
                    const int r = t / g;
                    const int h = t % g;
                    if (r < p) {
                        const float* qh =
                            query + (long)r * q_stride + (kv_group * g + h) * head_dim;
                        const int pos = tile_base + lane;
                        const bool valid = (pos < chunk_end) && (pos <= base_pos + r);
                        float score = -1e30f;
                        if (valid) {
                            float dot = 0.0f;
                            for (int d = 0; d < head_dim; d++) {
                                dot += qh[d] * k_smem[d * (ATT_SPLIT_GQA_TILE + 1) + lane];
                            }
                            score = dot * scale;
                        }
                        float pv = 0.0f;
                        if (valid) pv = expf(score - m_frozen[u]);
                        float s = pv;
                        for (int off = 16; off > 0; off >>= 1) {
                            s += __shfl_down_sync(0xffffffffu, s, off);
                        }
                        s = __shfl_sync(0xffffffffu, s, 0);
                        s_acc[u] += s;
                        p_smem[t * ATT_SPLIT_GQA_TILE + lane] = pv;
                    }
                }
            }
        }
        __syncthreads();

        // (C) P.V phase — ascending-l FMAs per (row, head, dim), NO
        // rescale (the frozen M already normalizes the exponent range).
        const int l_valid = (chunk_end - tile_base < ATT_SPLIT_GQA_TILE)
                                ? (chunk_end - tile_base)
                                : ATT_SPLIT_GQA_TILE;
        for (int l = 0; l < l_valid; l++) {
            const float v = KV_RD(&value_cache[(tile_base + l) * kv_stride + kv_off + dim]);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int r = grp * 4 + u;
                if (r < p) {
#pragma unroll
                    for (int h = 0; h < 8; h++) {
                        if (h < g) {
                            acc[u][h] += p_smem[(r * g + h) * ATT_SPLIT_GQA_TILE + l] * v;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Partials: part_m = the frozen M (the combine's expf(M - M) = 1),
    // part_l = the chunk's p-sum, part_out UNNORMALIZED.
    {
        const int r0 = grp * 4;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int r = r0 + u;
            if (r < p) {
#pragma unroll
                for (int h = 0; h < 8; h++) {
                    if (h < g) {
                        const int p_idx =
                            (r * n_head + kv_group * g + h) * n_chunks + chunk_id;
                        part_out[p_idx * head_dim + dim] = acc[u][h];
                    }
                }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int t = warp + u * 32;
            if (t < 16 * g) {
                const int r = t / g;
                const int h = t % g;
                if (r < p) {
                    const int p_idx =
                        (r * n_head + kv_group * g + h) * n_chunks + chunk_id;
                    part_m[p_idx] = m_frozen[u];
                    part_l[p_idx] = s_acc[u];
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(1024)
attention_verify2p_pv_f32(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    const float* __restrict__ mrg_m,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int base_pos,
    const int p)
{
    extern __shared__ float smem[];
    attention_verify2p_pv_body(query, key_cache, value_cache, mrg_m,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p);
}

// T9.14 graph-capture twin: base_pos from pos_dev; grid (and the
// partial stride) pinned to the max chunk count.
extern "C" __global__ void __launch_bounds__(1024)
attention_verify2p_pv_f32_devpos(
    const float* __restrict__ query,
    const kv_elt* __restrict__ key_cache,
    const kv_elt* __restrict__ value_cache,
    const float* __restrict__ mrg_m,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    const float scale,
    const int head_dim,
    const int n_head,
    const int n_kv_head,
    const int chunk_len,
    const int* __restrict__ pos_dev,
    const int p)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_verify2p_pv_body(query, key_cache, value_cache, mrg_m,
        part_m, part_l, part_out, smem, scale, head_dim, n_head, n_kv_head,
        chunk_len, base_pos, p);
}

"#;

/// Issue 742 — contracts of the GQA-fused split-KV partial kernel:
/// `n_head % n_kv_head == 0`, GQA group ≤ 8 (one score warp per head of the
/// group), `head_dim` ≤ 256 (the 256-thread block doubles as dim owner),
/// `chunk_len % 32 == 0` (32-position tile alignment).
pub(crate) fn validate_splitgqa(
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    chunk_len: usize,
) -> Result<(), super::CudarcKernelError> {
    if !n_head.is_multiple_of(n_kv_head) {
        return Err(super::CudarcKernelError::InvalidArg(format!(
            "splitgqa: n_head ({n_head}) not divisible by n_kv_head ({n_kv_head})"
        )));
    }
    let g = n_head / n_kv_head;
    if g == 0 || g > 8 {
        return Err(super::CudarcKernelError::InvalidArg(format!(
            "splitgqa: GQA group {g} outside [1, 8] (warp budget)"
        )));
    }
    if head_dim == 0 || head_dim > 256 {
        return Err(super::CudarcKernelError::InvalidArg(format!(
            "splitgqa: head_dim {head_dim} outside [1, 256] (block budget)"
        )));
    }
    if !chunk_len.is_multiple_of(32) {
        return Err(super::CudarcKernelError::InvalidArg(format!(
            "splitgqa: chunk_len {chunk_len} not a multiple of the 32-position tile"
        )));
    }
    Ok(())
}

/// Holds compiled CUDA kernels for Qwen3.5 attention decode.
///
/// Compiled once at construction; all kernels share the same CUDA module.
/// The caller provides the stream — kernels are launched on whatever stream
/// the caller specifies, allowing sharing with the dp4a GEMV + elementwise ops.
pub struct AttentionKernels {
    rope: CudaFunction,
    /// Issue 618 — device-pointer variant for CUDA Graph capture.
    rope_devpos: CudaFunction,
    /// Issue 641 T7.4 — RoPE backward (inverse rotation, sin negated).
    rope_backward: CudaFunction,
    split_qg: CudaFunction,
    rmsnorm_batched: CudaFunction,
    kv_cache_append: CudaFunction,
    /// Issue 618 — device-pointer variant.
    kv_cache_append_devpos: CudaFunction,
    attention_decode: CudaFunction,
    /// Issue 618 — device-pointer variant.
    attention_decode_devpos: CudaFunction,
    /// Issue 742 split-KV flash decode (partial + devpos twin + combine).
    attention_decode_split: CudaFunction,
    attention_decode_split_devpos: CudaFunction,
    attention_decode_split_combine: CudaFunction,
    /// Issue 742 — GQA-fused split-KV flash decode (partial + devpos twin;
    /// Bench 732 §6 headroom levers: coalesced Q·K + GQA-amortized V reads).
    /// Merges through the same `attention_decode_split_combine_f32`.
    attention_decode_splitgqa: CudaFunction,
    attention_decode_splitgqa_devpos: CudaFunction,
    output_gate: CudaFunction,
    /// Issue 742 T9.9 — the p-row batched verify family (the Q4_K verify
    /// port): the decode kernels' arithmetic VERBATIM per (row, element).
    rope_rows: CudaFunction,
    /// Issue 742 T9.12 - the devpos graph-capture twins.
    rope_rows_devpos: CudaFunction,
    kv_append_rows: CudaFunction,
    kv_append_rows_devpos: CudaFunction,
    split_qg_rows: CudaFunction,
    attention_splitgqa_rows: CudaFunction,
    attention_splitgqa_rows_devpos: CudaFunction,
    attention_split_combine_rows: CudaFunction,
    /// Issue 742 T9.11 — the q-group-major rows partial: one block per
    /// (kv_head, chunk) serving all p <= 16 query rows (the K/V
    /// amortization kernel; grid.z re-staging eliminated).
    /// Issue 754 T6 — widened to p <= [`SPLITGQA_QG_ROWS_MAX_P`] via
    /// grid.z = ceil(p/16) 16-row slices (the kernel body's `row_off`).
    attention_splitgqa_rows_qg: CudaFunction,
    attention_splitgqa_rows_qg_devpos: CudaFunction,
    /// Issue 742 T9.15 — the multi-accumulator dot variants (DOTMA=1
    /// instantiations of the shared bodies): the score-phase 256-deep
    /// serial-FMA chain broken into FOUR independent 64-deep partial
    /// accumulators (the T9.14 re-diagnosis lever). Tolerance class vs
    /// the serial kernels (fold-order reassociation); rows pair carries
    /// the short-ctx arm, qg pair the long-ctx arm.
    attention_splitgqa_rows_ma: CudaFunction,
    attention_splitgqa_rows_ma_devpos: CudaFunction,
    attention_splitgqa_rows_qgma: CudaFunction,
    attention_splitgqa_rows_qgma_devpos: CudaFunction,
    /// Issue 742 T9.14 — the two-pass flash restructure of the verify
    /// qg-attention walk: pass A (tile-parallel stats) + merge + pass B
    /// (frozen-max weighted V-accumulate) + the UNCHANGED rows combine.
    /// Live + devpos twins for the position-carrying passes; the merge
    /// is position-free (neutral dead-tile stats).
    attention_verify2p_stats: CudaFunction,
    attention_verify2p_stats_devpos: CudaFunction,
    attention_verify2p_merge: CudaFunction,
    attention_verify2p_pv: CudaFunction,
    attention_verify2p_pv_devpos: CudaFunction,
    _module: Arc<CudaModule>,
}

impl AttentionKernels {
    /// Issue 742 T9.16 — the K-float4 smem-layout arm for the verify rows/qg
    /// kernel family: k4=1 stages the K tile row-major (stride head_dim+4)
    /// and the score dot reads it as float4 (LDS.128 — the T9.15 load-issue
    /// cut applied to the K half; values bit-identical, load width only).
    /// Default ON; `QWEN38_VERIFY_ATTN_KF4=0` restores the transposed
    /// scalar-LDS layout (the A/B hatch). Resolved ONCE per process (the
    /// T9.12 graph capture bakes the arg — the arm must be process-fixed).
    fn verify_kf4() -> i32 {
        static KF4: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        *KF4.get_or_init(|| {
            if std::env::var("QWEN38_VERIFY_ATTN_KF4").is_ok_and(|v| v == "0") {
                0
            } else {
                1
            }
        })
    }

    /// Issue 779 — the geometry half of the k4 arm selector. The row-major
    /// staging writes a `head_dim+4`-strided 32-row tile: its footprint
    /// (`32*(head_dim+4)` floats incl. pad) fits the transposed envelope the
    /// launchers allocate (`head_dim*(TILE+1) = head_dim*33`) only for
    /// head_dim >= 128. Below that the staging overflows into `p_smem` and
    /// phase B races its own row-0 score writes against the overflowed K
    /// reads (lane 31's float4 tail) with no barrier between — measured as
    /// run-to-run nondeterministic row-0 corruption at head_dim 64
    /// (Bench 802, Issue 779 findings 1+2; finding 4's masked clobber is
    /// the same overflow). Production geometry (head_dim 256) is unaffected;
    /// smaller fixtures silently take the bit-identical transposed arm, so
    /// no gate expectation changes. Deterministic in head_dim — capture-safe
    /// like `verify_kf4` (a graph's head_dim never changes across replays).
    fn verify_kf4_for(head_dim: usize) -> i32 {
        if 32 * (head_dim + 4) <= head_dim * 33 {
            Self::verify_kf4()
        } else {
            0
        }
    }

    /// Compile all attention kernels via nvrtc.
    ///
    /// Uses `sm_89` (Ada Lovelace / RTX 4090). The kernels use basic CUDA
    /// features (shared memory, `__syncthreads`, dynamic `extern __shared__`).
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, super::CudarcKernelError> {
        Self::new_with_kv_dtype(ctx, false)
    }

    /// Issue 753 — the KV cache dtype hatch. `kv_f16 = true` prepends
    /// `#define KV_F16 1` to the SAME source, so kernel names, signatures
    /// (pointer ABI), grids and launchers are IDENTICAL in both builds and
    /// the f32 module is byte-identical to the pre-753 one (the default
    /// G3 arm). The f16 kernels convert half↔float at the global-memory
    /// boundary ONLY (see the source header) — smem staging, dots, online
    /// softmax, partials and the combine stay f32, so the numerics delta is
    /// exactly the f16 quantization of the K/V elements (tolerance class,
    /// not bit-identity; gates declared in Issue 753). The dtype is
    /// process-fixed: it selects the PTX module at load time and sizes the
    /// host-side KV buffers — it must not vary between instances feeding
    /// the same cache buffers.
    pub fn new_with_kv_dtype(
        ctx: Arc<CudaContext>,
        kv_f16: bool,
    ) -> Result<Self, super::CudarcKernelError> {
        let src = if kv_f16 {
            format!("#define KV_F16 1\n{ATTENTION_CUDA_SRC}")
        } else {
            ATTENTION_CUDA_SRC.to_string()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            &src,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| super::CudarcKernelError::Compile(e.to_string()))?;

        let rope = module
            .load_function("rope_partial_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let rope_devpos = module
            .load_function("rope_partial_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let rope_backward = module
            .load_function("rope_partial_backward_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let split_qg = module
            .load_function("split_qg_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let rmsnorm_batched = module
            .load_function("rmsnorm_batched_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let kv_cache_append = module
            .load_function("kv_cache_append_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let kv_cache_append_devpos = module
            .load_function("kv_cache_append_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode = module
            .load_function("attention_decode_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode_devpos = module
            .load_function("attention_decode_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode_split = module
            .load_function("attention_decode_split_partial_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode_split_devpos = module
            .load_function("attention_decode_split_partial_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode_split_combine = module
            .load_function("attention_decode_split_combine_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode_splitgqa = module
            .load_function("attention_decode_splitgqa_partial_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_decode_splitgqa_devpos = module
            .load_function("attention_decode_splitgqa_partial_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let rope_rows = module
            .load_function("rope_partial_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let rope_rows_devpos = module
            .load_function("rope_partial_rows_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let kv_append_rows = module
            .load_function("kv_cache_append_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let kv_append_rows_devpos = module
            .load_function("kv_cache_append_rows_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let split_qg_rows = module
            .load_function("split_qg_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_splitgqa_rows = module
            .load_function("attention_decode_splitgqa_partial_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_splitgqa_rows_devpos = module
            .load_function("attention_decode_splitgqa_partial_rows_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_split_combine_rows = module
            .load_function("attention_decode_split_combine_rows_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_splitgqa_rows_qg = module
            .load_function("attention_decode_splitgqa_partial_rows_qg_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_splitgqa_rows_qg_devpos = module
            .load_function("attention_decode_splitgqa_partial_rows_qg_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        // qg rows smem opt-in: k[256][33] + p[16*g][32] + st[16*g] reaches
        // 50,688 B at g=8 — over the 48 KB default combined budget (under
        // the 99 KB sm_89 per-block max). Same pattern as the prefill
        // mq8s kernels.
        let qg_smem = (256 * 33 + 16 * 8 * 32 + 16 * 8) * core::mem::size_of::<f32>() as i32;
        attention_splitgqa_rows_qg
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                qg_smem,
            )
            .map_err(|e| {
                super::CudarcKernelError::Launch(format!("qg rows smem opt-in: {e}"))
            })?;
        // T9.12: the devpos twin shares the qg smem geometry - same opt-in.
        attention_splitgqa_rows_qg_devpos
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                qg_smem,
            )
            .map_err(|e| {
                super::CudarcKernelError::Launch(format!("qg rows devpos smem opt-in: {e}"))
            })?;
        // Issue 742 T9.15 - the multi-accumulator dot kernels. The rows
        // pair needs no opt-in (rows smem stays under the 48 KB default);
        // the qgma pair shares the qg smem geometry (same opt-in value).
        let attention_splitgqa_rows_ma = module
            .load_function("attention_decode_splitgqa_partial_rows_ma_f32")
            .map_err(|e| super::CudarcKernelError::Compile(e.to_string()))?;
        let attention_splitgqa_rows_ma_devpos = module
            .load_function("attention_decode_splitgqa_partial_rows_ma_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(e.to_string()))?;
        let attention_splitgqa_rows_qgma = module
            .load_function("attention_decode_splitgqa_partial_rows_qgma_f32")
            .map_err(|e| super::CudarcKernelError::Compile(e.to_string()))?;
        let attention_splitgqa_rows_qgma_devpos = module
            .load_function("attention_decode_splitgqa_partial_rows_qgma_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(e.to_string()))?;
        for f in [&attention_splitgqa_rows_qgma, &attention_splitgqa_rows_qgma_devpos] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                qg_smem,
            )
            .map_err(|e| {
                super::CudarcKernelError::Launch(format!("qgma rows smem opt-in: {e}"))
            })?;
        }
        let attention_verify2p_stats = module
            .load_function("attention_verify2p_stats_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_verify2p_stats_devpos = module
            .load_function("attention_verify2p_stats_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_verify2p_merge = module
            .load_function("attention_verify2p_merge_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_verify2p_pv = module
            .load_function("attention_verify2p_pv_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        let attention_verify2p_pv_devpos = module
            .load_function("attention_verify2p_pv_f32_devpos")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;
        // T9.14 pv smem opt-in: k[256][33] + p[16*g][32] reaches 50,176 B
        // at g=8 — over the 48 KB default combined budget (the merge/stats
        // kernels stay under; stats is k-only 33,792 B). Same pattern as
        // the qg opt-in above.
        let pv2p_smem = (256 * 33 + 16 * 8 * 32) * core::mem::size_of::<f32>() as i32;
        attention_verify2p_pv
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                pv2p_smem,
            )
            .map_err(|e| {
                super::CudarcKernelError::Launch(format!("verify2p pv smem opt-in: {e}"))
            })?;
        attention_verify2p_pv_devpos
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                pv2p_smem,
            )
            .map_err(|e| {
                super::CudarcKernelError::Launch(format!(
                    "verify2p pv devpos smem opt-in: {e}"
                ))
            })?;
        let output_gate = module
            .load_function("output_gate_f32")
            .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))?;

        Ok(Self {
            rope,
            rope_devpos,
            rope_backward,
            split_qg,
            rmsnorm_batched,
            kv_cache_append,
            kv_cache_append_devpos,
            attention_decode,
            attention_decode_devpos,
            attention_decode_split,
            attention_decode_split_devpos,
            attention_decode_split_combine,
            attention_decode_splitgqa,
            attention_decode_splitgqa_devpos,
            output_gate,
            rope_rows,
            rope_rows_devpos,
            kv_append_rows,
            kv_append_rows_devpos,
            split_qg_rows,
            attention_splitgqa_rows,
            attention_splitgqa_rows_devpos,
            attention_split_combine_rows,
            attention_splitgqa_rows_qg,
            attention_splitgqa_rows_qg_devpos,
            attention_splitgqa_rows_ma,
            attention_splitgqa_rows_ma_devpos,
            attention_splitgqa_rows_qgma,
            attention_splitgqa_rows_qgma_devpos,
            attention_verify2p_stats,
            attention_verify2p_stats_devpos,
            attention_verify2p_merge,
            attention_verify2p_pv,
            attention_verify2p_pv_devpos,
            _module: module,
        })
    }

    /// Apply partial RoPE to Q and K in-place (GPT-NeoX rotate-half convention).
    ///
    /// - `q`: `[n_head * head_dim]` f32 device slice
    /// - `k`: `[n_kv_head * head_dim]` f32 device slice
    /// - `rotary_dim`: only the first `rotary_dim` elements are rotated
    pub fn launch_rope(
        &self,
        stream: &CudaStream,
        q: &cudarc::driver::safe::CudaSlice<f32>,
        k: &cudarc::driver::safe::CudaSlice<f32>,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos: usize,
        theta_base: f32,
    ) -> Result<(), super::CudarcKernelError> {
        let rotary_pairs = rotary_dim / 2;
        let total_pairs = n_head * rotary_pairs;
        let grid_x = (total_pairs as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rotary_dim_i32 = rotary_dim as i32;
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let pos_i32 = pos as i32;
        unsafe {
            stream
                .launch_builder(&self.rope)
                .arg(q)
                .arg(k)
                .arg(&rotary_dim_i32)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&pos_i32)
                .arg(&theta_base)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 641 T7.4 — Apply partial RoPE backward (inverse rotation) to grad_q
    /// and grad_k in-place.
    ///
    /// The backward rotation is the transpose of the forward rotation matrix,
    /// which is equivalent to the forward with `sin -> -sin`.
    ///
    /// - `grad_q`: `[n_head * head_dim]` f32 device slice (in-place)
    /// - `grad_k`: `[n_kv_head * head_dim]` f32 device slice (in-place)
    /// - `rotary_dim`: only the first `rotary_dim` elements are rotated
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_backward(
        &self,
        stream: &CudaStream,
        grad_q: &cudarc::driver::safe::CudaSlice<f32>,
        grad_k: &cudarc::driver::safe::CudaSlice<f32>,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos: usize,
        theta_base: f32,
    ) -> Result<(), super::CudarcKernelError> {
        let rotary_pairs = rotary_dim / 2;
        let total_pairs = n_head * rotary_pairs;
        let grid_x = (total_pairs as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rotary_dim_i32 = rotary_dim as i32;
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let pos_i32 = pos as i32;
        unsafe {
            stream
                .launch_builder(&self.rope_backward)
                .arg(grad_q)
                .arg(grad_k)
                .arg(&rotary_dim_i32)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&pos_i32)
                .arg(&theta_base)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 618 — Device-pointer variant of `launch_rope`. Reads `pos` from
    /// `pos_dev` (1-element device buffer) at kernel runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_devpos(
        &self,
        stream: &CudaStream,
        q: &cudarc::driver::safe::CudaSlice<f32>,
        k: &cudarc::driver::safe::CudaSlice<f32>,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
        theta_base: f32,
    ) -> Result<(), super::CudarcKernelError> {
        let rotary_pairs = rotary_dim / 2;
        let total_pairs = n_head * rotary_pairs;
        let grid_x = (total_pairs as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rotary_dim_i32 = rotary_dim as i32;
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        unsafe {
            stream
                .launch_builder(&self.rope_devpos)
                .arg(q)
                .arg(k)
                .arg(&rotary_dim_i32)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(pos_dev)
                .arg(&theta_base)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Split interleaved QG buffer `[n_head * 2 * head_dim]` into separate
    /// Q and gate buffers, each `[n_head * head_dim]`.
    pub fn launch_split_qg(
        &self,
        stream: &CudaStream,
        qg: &cudarc::driver::safe::CudaSlice<f32>,
        q: &cudarc::driver::safe::CudaSlice<f32>,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let total = n_head * head_dim;
        let grid_x = (total as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        unsafe {
            stream
                .launch_builder(&self.split_qg)
                .arg(qg)
                .arg(q)
                .arg(gate)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Per-head RMSNorm: `output[b*hd + i] = input[b*hd + i] * inv_rms * gamma[i]`.
    ///
    /// One block per head (`n_batch` blocks). Gamma is shared across all heads.
    pub fn launch_rmsnorm_batched(
        &self,
        stream: &CudaStream,
        input: &cudarc::driver::safe::CudaSlice<f32>,
        gamma: &cudarc::driver::safe::CudaSlice<f32>,
        output: &cudarc::driver::safe::CudaSlice<f32>,
        n_batch: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(), super::CudarcKernelError> {
        let inv_dim = 1.0f32 / head_dim as f32;
        let head_dim_i32 = head_dim as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_batch as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4, // 256 floats
        };
        unsafe {
            stream
                .launch_builder(&self.rmsnorm_batched)
                .arg(input)
                .arg(gamma)
                .arg(output)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&head_dim_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Append K and V vectors to the KV cache at position `pos`.
    pub fn launch_kv_cache_append(
        &self,
        stream: &CudaStream,
        k_vec: &cudarc::driver::safe::CudaSlice<f32>,
        v_vec: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        kvd: usize,
        pos: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let grid_x = (kvd as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let kvd_i32 = kvd as i32;
        let pos_i32 = pos as i32;
        unsafe {
            stream
                .launch_builder(&self.kv_cache_append)
                .arg(k_vec)
                .arg(v_vec)
                .arg(key_cache)
                .arg(value_cache)
                .arg(&kvd_i32)
                .arg(&pos_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 618 — Device-pointer variant of `launch_kv_cache_append`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kv_cache_append_devpos(
        &self,
        stream: &CudaStream,
        k_vec: &cudarc::driver::safe::CudaSlice<f32>,
        v_vec: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        kvd: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
    ) -> Result<(), super::CudarcKernelError> {
        let grid_x = (kvd as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let kvd_i32 = kvd as i32;
        unsafe {
            stream
                .launch_builder(&self.kv_cache_append_devpos)
                .arg(k_vec)
                .arg(v_vec)
                .arg(key_cache)
                .arg(value_cache)
                .arg(&kvd_i32)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Flash attention decode: single-token attention over the full KV cache.
    ///
    /// - `query`: `[n_head * head_dim]`
    /// - `key_cache`, `value_cache`: `[n_positions * n_kv_head * head_dim]`
    /// - `attn_out`: `[n_head * head_dim]`
    ///
    /// Dispatch: `n_head` blocks × `head_dim` threads. Dynamic shared memory
    /// = `head_dim * 4` bytes.
    pub fn launch_attention_decode(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        n_positions: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let n_positions_i32 = n_positions as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim * 4) as u32, // dynamic smem = head_dim floats
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(attn_out)
                .arg(&scale)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&n_positions_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 618 — Device-pointer variant of `launch_attention_decode`.
    /// Reads `n_positions = *pos_dev + 1` at kernel runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_decode_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
    ) -> Result<(), super::CudarcKernelError> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim * 4) as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(attn_out)
                .arg(&scale)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 — split-KV flash attention decode (eager form).
    ///
    /// The position axis is the parallelism axis: the partial kernel launches
    /// `(n_head, ceil(n_positions / chunk_len))` blocks — at 12K ctx that is
    /// 24 × 6 blocks vs the serial kernel's 24 (the long-ctx latency wall:
    /// ~21 GB/s effective, 95% of SMs idle). Each block online-softmaxes its
    /// chunk into an UNNORMALIZED partial `(m, l, out[head_dim])`; the
    /// combine kernel merges them (the standard flash-decode split).
    ///
    /// `chunk_len` must be a multiple of `head_dim` (tile boundaries align
    /// with chunk boundaries — the launcher contract). The caller's scratch
    /// must be sized for `ceil(n_positions / chunk_len)` chunk-columns.
    ///
    /// Numerics: the tile body is `attention_decode_f32` verbatim — the split
    /// reassociates the accumulation only (gate: max_rel ≤ 1e-5 vs serial).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_decode_split(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_positions: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let chunk_len_i32 = chunk_len as i32;
        let n_positions_i32 = n_positions as i32;
        let n_chunks = n_positions.div_ceil(chunk_len).max(1);
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, n_chunks as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim * 4) as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode_split)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&chunk_len_i32)
                .arg(&n_positions_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        self.launch_split_combine(
            stream, part_m, part_l, part_out, attn_out, head_dim, n_head, n_chunks,
        )
    }

    /// Issue 742 — split-KV flash decode, CUDA-graph form: `n_positions` is
    /// `*pos_dev + 1` at kernel runtime and the grid is FIXED at
    /// `(n_head, n_chunks_max)` (grid dims bake into the capture). Chunks
    /// beyond the live position write the NEUTRAL partial (m = −1e30, l = 0,
    /// out = 0) — every scratch slot is rewritten every launch, so the fixed
    /// grid is rollback-safe (a shrinking n_positions can never read stale
    /// partials) and dead chunks contribute exactly 0 to the combine.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_decode_split_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks_max: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
    ) -> Result<(), super::CudarcKernelError> {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let chunk_len_i32 = chunk_len as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, n_chunks_max as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim * 4) as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode_split_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&chunk_len_i32)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        self.launch_split_combine(
            stream, part_m, part_l, part_out, attn_out, head_dim, n_head, n_chunks_max,
        )
    }

    /// Issue 742 — GQA-fused split-KV flash decode. One block per
    /// `(kv_head, chunk)` serves the whole GQA group: K staged to shared
    /// memory transposed via coalesced row loads, warp-per-head Q·K dots
    /// (lane = position, conflict-free reads), P·V reads V coalesced once and
    /// reuses it for all `g = n_head / n_kv_head` heads. Scratch layout is
    /// identical to the per-head split — the same combine merges it.
    ///
    /// Contracts (checked): `n_head % n_kv_head == 0`, group ≤ 8 (warp
    /// budget), `head_dim` ≤ 256 (block budget), `chunk_len % 32 == 0`
    /// (tile alignment).
    ///
    /// Numerics vs the per-head split: each score is bit-identical (same
    /// serial FMA order); the online-softmax granularity changes
    /// (32-position warp tiles) — the reassociation class, gate max_rel ≤
    /// 1e-5 vs the serial kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_decode_split_gqa(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_positions: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        let g = n_head / n_kv_head;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let chunk_len_i32 = chunk_len as i32;
        let n_positions_i32 = n_positions as i32;
        let n_chunks = n_positions.div_ceil(chunk_len).max(1);
        // smem (floats): q[g*head_dim] | k[head_dim*33] | p[g*32] | st[g]
        let smem_floats = g * head_dim + head_dim * 33 + g * 32 + g;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode_splitgqa)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&chunk_len_i32)
                .arg(&n_positions_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        self.launch_split_combine(
            stream, part_m, part_l, part_out, attn_out, head_dim, n_head, n_chunks,
        )
    }

    /// Issue 742 — GQA-fused split-KV flash decode, CUDA-graph form:
    /// `n_positions` is `*pos_dev + 1` at kernel runtime and the grid is
    /// FIXED at `(n_kv_head, n_chunks_max)`. Dead chunks write the neutral
    /// partial for every head of the group — rollback-safe exactly like the
    /// per-head split devpos twin.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_decode_split_gqa_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks_max: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        let g = n_head / n_kv_head;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let head_dim_i32 = head_dim as i32;
        let n_head_i32 = n_head as i32;
        let n_kv_head_i32 = n_kv_head as i32;
        let chunk_len_i32 = chunk_len as i32;
        let smem_floats = g * head_dim + head_dim * 33 + g * 32 + g;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks_max as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode_splitgqa_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&head_dim_i32)
                .arg(&n_head_i32)
                .arg(&n_kv_head_i32)
                .arg(&chunk_len_i32)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        self.launch_split_combine(
            stream, part_m, part_l, part_out, attn_out, head_dim, n_head, n_chunks_max,
        )
    }

    /// Flash-decode partial merge: one block per head, one thread per dim.
    /// Dead chunks contribute exactly 0 (`expf(−1e30 − M)` underflows to 0,
    /// l = 0), so a live-grid combine and a max-grid combine are
    /// bit-identical — the property that makes the fixed graph grid safe.
    fn launch_split_combine(
        &self,
        stream: &CudaStream,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_chunks: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let head_dim_i32 = head_dim as i32;
        let n_chunks_i32 = n_chunks as i32;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_decode_split_combine)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&head_dim_i32)
                .arg(&n_chunks_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Apply sigmoid output gating in-place: `attn_out[i] *= sigmoid(gate[i])`.
    pub fn launch_output_gate(
        &self,
        stream: &CudaStream,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
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
                .launch_builder(&self.output_gate)
                .arg(attn_out)
                .arg(gate)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    // ── Issue 742 T9.9: the p-row batched verify launchers ────────────────

    /// Batched partial RoPE for `p` rows (row r rotates at `base_pos + r`),
    /// in-place on the row-major q/k buffers. Per element VERBATIM
    /// [`Self::launch_rope`]'s kernel.
    ///
    /// # Safety
    ///
    /// Caller guarantees `q` covers `p * n_head * head_dim`, `k` covers
    /// `p * n_kv_head * head_dim`, and `rotary_pairs * 2 <= head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_rows(
        &self,
        stream: &CudaStream,
        q: &impl DevicePtr<f32>,
        k: &impl DevicePtr<f32>,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        base_pos: usize,
        p: usize,
        theta_base: f32,
    ) -> Result<(), super::CudarcKernelError> {
        let rotary_pairs = rotary_dim / 2;
        let total = (p * n_head * rotary_pairs) as u32;
        let grid_x = total.div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rd, hd, nh, nk, bp, pi) = (
            rotary_dim as i32,
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            base_pos as i32,
            p as i32,
        );
        // DevicePtr-widened — the lane path passes per-lane views (Plan 556).
        let (q_ptr, _sync_q) = q.device_ptr(stream);
        let (k_ptr, _sync_k) = k.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.rope_rows)
                .arg(&q_ptr)
                .arg(&k_ptr)
                .arg(&rd)
                .arg(&hd)
                .arg(&nh)
                .arg(&nk)
                .arg(&bp)
                .arg(&pi)
                .arg(&theta_base)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.12 - graph-capture twin of [`Self::launch_rope_rows`]:
    /// `base_pos` read from `pos_dev` at kernel runtime (the scalar would
    /// bake into the captured graph). Identical grid/geometry.
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_rope_rows`]; `pos_dev` covers 1.
    /// Plan 556 Stage 3: DevicePtr-widened — the lane graph path passes
    /// `pos_dev.slice(l..l+1)` views (one per lane).
    pub fn launch_rope_rows_devpos(
        &self,
        stream: &CudaStream,
        q: &impl DevicePtr<f32>,
        k: &impl DevicePtr<f32>,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos_dev: &impl DevicePtr<i32>,
        p: usize,
        theta_base: f32,
    ) -> Result<(), super::CudarcKernelError> {
        let rotary_pairs = rotary_dim / 2;
        let total = (p * n_head * rotary_pairs) as u32;
        let grid_x = total.div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (rd, hd, nh, nk, pi) = (
            rotary_dim as i32,
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            p as i32,
        );
        // DevicePtr-widened — the lane graph path passes per-lane views (Plan 556).
        let (q_ptr, _sync_q) = q.device_ptr(stream);
        let (k_ptr, _sync_k) = k.device_ptr(stream);
        let (pos_ptr, _sync_pos) = pos_dev.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.rope_rows_devpos)
                .arg(&q_ptr)
                .arg(&k_ptr)
                .arg(&rd)
                .arg(&hd)
                .arg(&nh)
                .arg(&nk)
                .arg(&pos_ptr)
                .arg(&pi)
                .arg(&theta_base)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched KV append: row r writes cache row `base_pos + r`.
    ///
    /// # Safety
    ///
    /// Caller guarantees `k_vecs`/`v_vecs` cover `p * kvd` and the caches
    /// cover `(base_pos + p) * kvd`.
    pub fn launch_kv_append_rows(
        &self,
        stream: &CudaStream,
        k_vecs: &impl DevicePtr<f32>,
        v_vecs: &impl DevicePtr<f32>,
        key_cache: &impl DevicePtr<f32>,
        value_cache: &impl DevicePtr<f32>,
        kvd: usize,
        base_pos: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let total = (p * kvd) as u32;
        let grid_x = total.div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kvd_i, bp_i, p_i) = (kvd as i32, base_pos as i32, p as i32);
        // DevicePtr-widened — the lane path passes per-lane views (Plan 556).
        let (k_ptr, _sync_k) = k_vecs.device_ptr(stream);
        let (v_ptr, _sync_v) = v_vecs.device_ptr(stream);
        let (kc_ptr, _sync_kc) = key_cache.device_ptr(stream);
        let (vc_ptr, _sync_vc) = value_cache.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.kv_append_rows)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&kc_ptr)
                .arg(&vc_ptr)
                .arg(&kvd_i)
                .arg(&bp_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.12 - graph-capture twin of [`Self::launch_kv_append_rows`]:
    /// `base_pos` read from `pos_dev` at kernel runtime. Identical
    /// grid/geometry.
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_kv_append_rows`]; `pos_dev` covers 1.
    /// Plan 556 Stage 3: DevicePtr-widened — the lane graph path passes
    /// `pos_dev.slice(l..l+1)` views (one per lane).
    pub fn launch_kv_append_rows_devpos(
        &self,
        stream: &CudaStream,
        k_vecs: &impl DevicePtr<f32>,
        v_vecs: &impl DevicePtr<f32>,
        key_cache: &impl DevicePtr<f32>,
        value_cache: &impl DevicePtr<f32>,
        kvd: usize,
        pos_dev: &impl DevicePtr<i32>,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let total = (p * kvd) as u32;
        let grid_x = total.div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (kvd_i, p_i) = (kvd as i32, p as i32);
        // DevicePtr-widened — the lane graph path passes per-lane views (Plan 556).
        let (k_ptr, _sync_k) = k_vecs.device_ptr(stream);
        let (v_ptr, _sync_v) = v_vecs.device_ptr(stream);
        let (kc_ptr, _sync_kc) = key_cache.device_ptr(stream);
        let (vc_ptr, _sync_vc) = value_cache.device_ptr(stream);
        let (pos_ptr, _sync_pos) = pos_dev.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.kv_append_rows_devpos)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&kc_ptr)
                .arg(&vc_ptr)
                .arg(&kvd_i)
                .arg(&pos_ptr)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Batched QG split on row-major buffers (per element VERBATIM
    /// [`Self::launch_split_qg`]'s kernel).
    ///
    /// # Safety
    ///
    /// Caller guarantees `qg` covers `p * n_head * 2 * head_dim` and
    /// `q`/`gate` cover `p * n_head * head_dim`.
    pub fn launch_split_qg_rows(
        &self,
        stream: &CudaStream,
        qg: &cudarc::driver::safe::CudaSlice<f32>,
        q: &cudarc::driver::safe::CudaSlice<f32>,
        gate: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let total = (p * n_head * head_dim) as u32;
        let grid_x = total.div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (hd, nh, pi) = (head_dim as i32, n_head as i32, p as i32);
        unsafe {
            stream
                .launch_builder(&self.split_qg_rows)
                .arg(qg)
                .arg(q)
                .arg(gate)
                .arg(&hd)
                .arg(&nh)
                .arg(&pi)
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.9 — the multi-query GQA split-KV partial + rows combine:
    /// computes attention for `p` query rows against the shared K/V cache in
    /// one partial pass (K/V read once per (kv_head, chunk); QPB=4 query rows
    /// per block) and merges per (row, head). Per (row, head) bit-identical
    /// to the decode splitgqa path at position `base_pos + r`.
    ///
    /// Partials layout: `part_m`/`part_l` `[p, n_head, n_chunks]`,
    /// `part_out` `[p, n_head, n_chunks, head_dim]`; output
    /// `attn_out` `[p, n_head * head_dim]`.
    ///
    /// # Safety
    ///
    /// Same shape contracts as [`Self::launch_attention_decode_split_gqa`]
    /// plus `query` covering `p * n_head * head_dim`, `attn_out` covering
    /// `p * n_head * head_dim`, and the partial buffers covering
    /// `p * n_head * n_chunks` (`* head_dim` for `part_out`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows(
        &self,
        stream: &CudaStream,
        query: &impl DevicePtr<f32>,
        key_cache: &impl DevicePtr<f32>,
        value_cache: &impl DevicePtr<f32>,
        part_m: &impl DevicePtr<f32>,
        part_l: &impl DevicePtr<f32>,
        part_out: &impl DevicePtr<f32>,
        attn_out: &impl DevicePtr<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        base_pos: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        let g = n_head / n_kv_head;
        // dynamic smem: k[hd][TILE+1] | p[QPB][g][TILE] | st[QPB][g]
        let smem_floats = head_dim * 33 + 4 * g * 32 + 4 * g;
        let q_blocks = p.div_ceil(4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, q_blocks),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, bp_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            base_pos as i32,
            p as i32,
        );
        // DevicePtr-widened — the lane path passes per-lane views (Plan 556).
        let (q_ptr, _sync_q) = query.device_ptr(stream);
        let (kc_ptr, _sync_kc) = key_cache.device_ptr(stream);
        let (vc_ptr, _sync_vc) = value_cache.device_ptr(stream);
        let (pm_ptr, _sync_pm) = part_m.device_ptr(stream);
        let (pl_ptr, _sync_pl) = part_l.device_ptr(stream);
        let (po_ptr, _sync_po) = part_out.device_ptr(stream);
        let (ao_ptr, _sync_ao) = attn_out.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows)
                .arg(&q_ptr)
                .arg(&kc_ptr)
                .arg(&vc_ptr)
                .arg(&pm_ptr)
                .arg(&pl_ptr)
                .arg(&po_ptr)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&bp_i)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // rows combine: grid (p * n_head), one thread per dim
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(&pm_ptr)
                .arg(&pl_ptr)
                .arg(&po_ptr)
                .arg(&ao_ptr)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.12 - graph-capture twin of
    /// [`Self::launch_attention_splitgqa_rows`]: `base_pos` read from
    /// `pos_dev` at kernel runtime; the grid is pinned to the FIXED max
    /// chunk count (dead chunks write neutral partials - every partial
    /// slot rewritten on every launch, the T9.6 over-launch contract).
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_attention_splitgqa_rows`];
    /// `pos_dev` covers 1 and `n_chunks` must stay fixed across replays.
    /// Plan 556 Stage 3: DevicePtr-widened — the lane graph path passes
    /// `pos_dev.slice(l..l+1)` views (one per lane).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_devpos(
        &self,
        stream: &CudaStream,
        query: &impl DevicePtr<f32>,
        key_cache: &impl DevicePtr<f32>,
        value_cache: &impl DevicePtr<f32>,
        part_m: &impl DevicePtr<f32>,
        part_l: &impl DevicePtr<f32>,
        part_out: &impl DevicePtr<f32>,
        attn_out: &impl DevicePtr<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        pos_dev: &impl DevicePtr<i32>,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        let g = n_head / n_kv_head;
        let smem_floats = head_dim * 33 + 4 * g * 32 + 4 * g;
        let q_blocks = p.div_ceil(4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, q_blocks),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            p as i32,
        );
        // DevicePtr-widened — the lane graph path passes per-lane views (Plan 556).
        let (q_ptr, _sync_q) = query.device_ptr(stream);
        let (kc_ptr, _sync_kc) = key_cache.device_ptr(stream);
        let (vc_ptr, _sync_vc) = value_cache.device_ptr(stream);
        let (pm_ptr, _sync_pm) = part_m.device_ptr(stream);
        let (pl_ptr, _sync_pl) = part_l.device_ptr(stream);
        let (po_ptr, _sync_po) = part_out.device_ptr(stream);
        let (ao_ptr, _sync_ao) = attn_out.device_ptr(stream);
        let (pos_ptr, _sync_pos) = pos_dev.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_devpos)
                .arg(&q_ptr)
                .arg(&kc_ptr)
                .arg(&vc_ptr)
                .arg(&pm_ptr)
                .arg(&pl_ptr)
                .arg(&po_ptr)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&pos_ptr)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(&pm_ptr)
                .arg(&pl_ptr)
                .arg(&po_ptr)
                .arg(&ao_ptr)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.11 — the q-group-major rows partial: ONE block per
    /// (kv_head, chunk, 16-row slice) — 1024 threads = 4 row-groups x
    /// head_dim. K/V read once per (kv_head, chunk) per slice. Per (row,
    /// head, dim) bit-identical to [`Self::launch_attention_splitgqa_rows`]
    /// (and thus to the decode splitgqa per row).
    ///
    /// Issue 754 T6 — `p` widened to [`SPLITGQA_QG_ROWS_MAX_P`] via
    /// `grid.z = ceil(p/16)`: each z-block keeps the original 16-row
    /// register/smem geometry (unchanged occupancy); total K/V traffic at
    /// p=64 equals 4 sequential p=16 chunks (NOT the T9.9 fallback's 16x).
    /// At p <= 16, grid.z == 1 and the launch is config- and
    /// bit-identical to the pre-T6 kernel.
    ///
    /// Partials layout + combine: identical to
    /// [`Self::launch_attention_splitgqa_rows`].
    ///
    /// # Safety
    ///
    /// Same shape contracts as [`Self::launch_attention_splitgqa_rows`]
    /// plus `head_dim == 256` and `p <= SPLITGQA_QG_ROWS_MAX_P` (the
    /// kernel's compile-time row-group geometry).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_qg(
        &self,
        stream: &CudaStream,
        query: &impl DevicePtr<f32>,
        key_cache: &impl DevicePtr<f32>,
        value_cache: &impl DevicePtr<f32>,
        part_m: &impl DevicePtr<f32>,
        part_l: &impl DevicePtr<f32>,
        part_out: &impl DevicePtr<f32>,
        attn_out: &impl DevicePtr<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        base_pos: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qg: head_dim must be 256 (got {head_dim})"
            )));
        }
        if p == 0 || p > SPLITGQA_QG_ROWS_MAX_P {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qg: p must be in 1..={SPLITGQA_QG_ROWS_MAX_P} (got {p})"
            )));
        }
        let g = n_head / n_kv_head;
        // dynamic smem: k[hd][TILE+1] | p[16*g][TILE] | st[16*g]
        let smem_floats = head_dim * 33 + 16 * g * 32 + 16 * g;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, p.div_ceil(16) as u32),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, bp_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            base_pos as i32,
            p as i32,
        );
        // DevicePtr-widened — the lane path passes per-lane views (Plan 556).
        let (q_ptr, _sync_q) = query.device_ptr(stream);
        let (kc_ptr, _sync_kc) = key_cache.device_ptr(stream);
        let (vc_ptr, _sync_vc) = value_cache.device_ptr(stream);
        let (pm_ptr, _sync_pm) = part_m.device_ptr(stream);
        let (pl_ptr, _sync_pl) = part_l.device_ptr(stream);
        let (po_ptr, _sync_po) = part_out.device_ptr(stream);
        let (ao_ptr, _sync_ao) = attn_out.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_qg)
                .arg(&q_ptr)
                .arg(&kc_ptr)
                .arg(&vc_ptr)
                .arg(&pm_ptr)
                .arg(&pl_ptr)
                .arg(&po_ptr)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&bp_i)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // rows combine: grid (p * n_head), one thread per dim — identical
        // to the T9.9 path (shared scratch layout).
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(&pm_ptr)
                .arg(&pl_ptr)
                .arg(&po_ptr)
                .arg(&ao_ptr)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.12 - graph-capture twin of
    /// [`Self::launch_attention_splitgqa_rows_qg`]: `base_pos` read from
    /// `pos_dev` at kernel runtime; the grid is pinned to the FIXED max
    /// chunk count (dead chunks write neutral partials). The smem opt-in
    /// attribute is set on this twin at load time.
    /// Issue 754 T6 — widened to p <= [`SPLITGQA_QG_ROWS_MAX_P`] via
    /// `grid.z = ceil(p/16)` (the twin shares the widened body; the p <= 16
    /// launch remains config- and bit-identical to the pre-T6 kernel).
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_attention_splitgqa_rows_qg`];
    /// `pos_dev` covers 1 and `n_chunks` must stay fixed across replays.
    /// Plan 556 Stage 3: DevicePtr-widened (API-consistency with the rows
    /// twin; the lane path pins rows and never launches this arm).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_qg_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        pos_dev: &impl DevicePtr<i32>,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qg devpos: head_dim must be 256 (got {head_dim})"
            )));
        }
        if p == 0 || p > SPLITGQA_QG_ROWS_MAX_P {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qg devpos: p must be in 1..={SPLITGQA_QG_ROWS_MAX_P} (got {p})"
            )));
        }
        let g = n_head / n_kv_head;
        let smem_floats = head_dim * 33 + 16 * g * 32 + 16 * g;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, p.div_ceil(16) as u32),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            p as i32,
        );
        let (pos_ptr, _sync_pos) = pos_dev.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_qg_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&pos_ptr)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.15 — the multi-accumulator dot variant of the T9.9
    /// rows launcher (DOTMA=1 kernels: the score dot's 256-deep
    /// serial-FMA chain broken into FOUR independent 64-deep partials).
    /// head_dim MUST be 256 (the compile-time dot bound). Tolerance
    /// class vs the serial launcher (fold-order reassociation).
    /// Issue 742 T9.9 — the multi-query GQA split-KV partial + rows combine:
    /// computes attention for `p` query rows against the shared K/V cache in
    /// one partial pass (K/V read once per (kv_head, chunk); QPB=4 query rows
    /// per block) and merges per (row, head). Per (row, head) bit-identical
    /// to the decode splitgqa path at position `base_pos + r`.
    ///
    /// Partials layout: `part_m`/`part_l` `[p, n_head, n_chunks]`,
    /// `part_out` `[p, n_head, n_chunks, head_dim]`; output
    /// `attn_out` `[p, n_head * head_dim]`.
    ///
    /// # Safety
    ///
    /// Same shape contracts as [`Self::launch_attention_decode_split_gqa`]
    /// plus `query` covering `p * n_head * head_dim`, `attn_out` covering
    /// `p * n_head * head_dim`, and the partial buffers covering
    /// `p * n_head * n_chunks` (`* head_dim` for `part_out`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_ma(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        base_pos: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows ma: head_dim must be 256 (got {head_dim})"
            )));
        }
        let g = n_head / n_kv_head;
        // dynamic smem: k[hd][TILE+1] | p[QPB][g][TILE] | st[QPB][g]
        let smem_floats = head_dim * 33 + 4 * g * 32 + 4 * g;
        let q_blocks = p.div_ceil(4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, q_blocks),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, bp_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            base_pos as i32,
            p as i32,
        );
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_ma)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&bp_i)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // rows combine: grid (p * n_head), one thread per dim
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.12 - graph-capture twin of
    /// [`Self::launch_attention_splitgqa_rows_ma`]: `base_pos` read from
    /// `pos_dev` at kernel runtime; the grid is pinned to the FIXED max
    /// chunk count (dead chunks write neutral partials - every partial
    /// slot rewritten on every launch, the T9.6 over-launch contract).
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_attention_splitgqa_rows_ma`];
    /// `pos_dev` covers 1 and `n_chunks` must stay fixed across replays.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_ma_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows ma: head_dim must be 256 (got {head_dim})"
            )));
        }
        let g = n_head / n_kv_head;
        let smem_floats = head_dim * 33 + 4 * g * 32 + 4 * g;
        let q_blocks = p.div_ceil(4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, q_blocks),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            p as i32,
        );
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_ma_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(pos_dev)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.15 — the multi-accumulator dot variant of the T9.11
    /// qg launcher (DOTMA=1 kernels; same smem opt-in at load time).
    /// Tolerance class vs the serial qg launcher (fold-order
    /// reassociation ~1e-6).
    /// Issue 742 T9.11 — the q-group-major rows partial: ONE block per
    /// (kv_head, chunk) serving all `p <= 16` query rows (1024 threads =
    /// 4 row-groups x head_dim). K/V read ONCE per (kv_head, chunk) —
    /// the T9.9 kernel's grid.z = ceil(p/4) re-staged K per 4-row group,
    /// ~2x off the K/V floor at 20K positions. Per (row, head, dim)
    /// bit-identical to [`Self::launch_attention_splitgqa_rows`] (and
    /// thus to the decode splitgqa per row).
    ///
    /// Partials layout + combine: identical to
    /// [`Self::launch_attention_splitgqa_rows`].
    ///
    /// # Safety
    ///
    /// Same shape contracts as [`Self::launch_attention_splitgqa_rows`]
    /// plus `head_dim == 256` and `p <= 16` (the kernel's compile-time
    /// row-group geometry).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_qgma(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        base_pos: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qgma: head_dim must be 256 (got {head_dim})"
            )));
        }
        if p == 0 || p > SPLITGQA_QG_ROWS_MAX_P {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qgma: p must be in 1..={SPLITGQA_QG_ROWS_MAX_P} (got {p})"
            )));
        }
        let g = n_head / n_kv_head;
        // dynamic smem: k[hd][TILE+1] | p[16*g][TILE] | st[16*g]
        let smem_floats = head_dim * 33 + 16 * g * 32 + 16 * g;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, p.div_ceil(16) as u32),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, bp_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            base_pos as i32,
            p as i32,
        );
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_qgma)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&bp_i)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // rows combine: grid (p * n_head), one thread per dim — identical
        // to the T9.9 path (shared scratch layout).
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.12 - graph-capture twin of
    /// [`Self::launch_attention_splitgqa_rows_qgma`]: `base_pos` read from
    /// `pos_dev` at kernel runtime; the grid is pinned to the FIXED max
    /// chunk count (dead chunks write neutral partials). The smem opt-in
    /// attribute is set on this twin at load time.
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_attention_splitgqa_rows_qgma`];
    /// `pos_dev` covers 1 and `n_chunks` must stay fixed across replays.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_splitgqa_rows_qgma_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qgma devpos: head_dim must be 256 (got {head_dim})"
            )));
        }
        if p == 0 || p > SPLITGQA_QG_ROWS_MAX_P {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "splitgqa rows qgma devpos: p must be in 1..={SPLITGQA_QG_ROWS_MAX_P} (got {p})"
            )));
        }
        let g = n_head / n_kv_head;
        let smem_floats = head_dim * 33 + 16 * g * 32 + 16 * g;
        let cfg = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, p.div_ceil(16) as u32),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: (smem_floats * 4) as u32,
        };
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            p as i32,
        );
        unsafe {
            stream
                .launch_builder(&self.attention_splitgqa_rows_qgma_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(pos_dev)
                .arg(&p_i)
                .arg(&Self::verify_kf4_for(head_dim))
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let nc_i = n_chunks as i32;
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.14 — the two-pass flash restructure of the verify
    /// qg-attention walk, LIVE (scalar `base_pos`) variant:
    /// (1) `attention_verify2p_stats_f32` — pass A, grid
    ///     `(n_kv_head, n_stat_tiles)` with `n_stat_tiles` the EXACT
    ///     live tile count (`ceil((base_pos+p)/32)`), which is also the
    ///     stat stride;
    /// (2) `attention_verify2p_merge_f32` — the (M, L) fold, grid
    ///     `(n_kv_head)`, 128 threads;
    /// (3) `attention_verify2p_pv_f32` — pass B, grid
    ///     `(n_kv_head, n_chunks)` at the caller's `chunk_len`
    ///     (T9.13's `verify_attn_chunk_len`);
    /// (4) the UNCHANGED `attention_split_combine_rows` over the chunk
    ///     partials (pass B writes `part_m = M`, so the combine reduces
    ///     to the plain ascending sum + 1/L divide).
    ///
    /// Same contract as [`Self::launch_attention_splitgqa_rows_qg`]:
    /// head_dim == 256, g <= 8, p in 1..=16, chunk_len % 32 == 0.
    ///
    /// # Safety
    ///
    /// Scratch contracts: `stat_m`/`stat_l` cover
    /// `p * n_head * n_stat_tiles`; `mrg_m`/`mrg_l` cover `p * n_head`;
    /// `part_m`/`part_l` cover `p * n_head * n_chunks`; `part_out` covers
    /// `p * n_head * n_chunks * head_dim`; key/value caches cover
    /// `(base_pos + p) * n_kv_head * head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_verify2p(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        stat_m: &cudarc::driver::safe::CudaSlice<f32>,
        stat_l: &cudarc::driver::safe::CudaSlice<f32>,
        mrg_m: &cudarc::driver::safe::CudaSlice<f32>,
        mrg_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        n_stat_tiles: usize,
        base_pos: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "verify2p: head_dim must be 256 (got {head_dim})"
            )));
        }
        if p == 0 || p > 16 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "verify2p: p must be in 1..=16 (got {p})"
            )));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, nc_i, nt_i, bp_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            n_chunks as i32,
            n_stat_tiles as i32,
            base_pos as i32,
            p as i32,
        );
        // (1) pass A — tile-parallel stats. smem: k[hd][TILE+1].
        let stats_smem = (head_dim * 33 * core::mem::size_of::<f32>()) as u32;
        let cfg_stats = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_stat_tiles as u32, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: stats_smem,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_verify2p_stats)
                .arg(query)
                .arg(key_cache)
                .arg(stat_m)
                .arg(stat_l)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&nt_i)
                .arg(&bp_i)
                .arg(&p_i)
                .launch(cfg_stats)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // (2) merge — the frozen (M, L) fold.
        let cfg_merge = LaunchConfig {
            grid_dim: (n_kv_head as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_verify2p_merge)
                .arg(stat_m)
                .arg(stat_l)
                .arg(mrg_m)
                .arg(mrg_l)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&nt_i)
                .arg(&p_i)
                .launch(cfg_merge)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // (3) pass B — frozen-max weighted V-accumulate. smem:
        // k[hd][TILE+1] | p[16*g][TILE].
        let g = n_head / n_kv_head;
        let pv_smem = ((head_dim * 33 + 16 * g * 32) * core::mem::size_of::<f32>()) as u32;
        let cfg_pv = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: pv_smem,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_verify2p_pv)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(mrg_m)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(&bp_i)
                .arg(&p_i)
                .launch(cfg_pv)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        // (4) the UNCHANGED rows combine (part_m = M ⇒ plain sum + 1/L).
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 742 T9.14 — graph-capture twin of
    /// [`Self::launch_attention_verify2p`]: `base_pos` read from
    /// `pos_dev` at kernel runtime; BOTH grids pinned to the fixed max
    /// counts (`n_stat_tiles` / `n_chunks`) with dead tiles/chunks
    /// writing neutral partials — live-grid ≡ max-grid bit-identically
    /// (the T9.12 property; unit-gated).
    ///
    /// # Safety
    ///
    /// Same contracts as [`Self::launch_attention_verify2p`]; `pos_dev`
    /// covers 1, and `n_stat_tiles`/`n_chunks` must stay fixed across
    /// replays.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_attention_verify2p_devpos(
        &self,
        stream: &CudaStream,
        query: &cudarc::driver::safe::CudaSlice<f32>,
        key_cache: &cudarc::driver::safe::CudaSlice<f32>,
        value_cache: &cudarc::driver::safe::CudaSlice<f32>,
        stat_m: &cudarc::driver::safe::CudaSlice<f32>,
        stat_l: &cudarc::driver::safe::CudaSlice<f32>,
        mrg_m: &cudarc::driver::safe::CudaSlice<f32>,
        mrg_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        chunk_len: usize,
        n_chunks: usize,
        n_stat_tiles: usize,
        pos_dev: &cudarc::driver::safe::CudaSlice<i32>,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        validate_splitgqa(head_dim, n_head, n_kv_head, chunk_len)?;
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "verify2p devpos: head_dim must be 256 (got {head_dim})"
            )));
        }
        if p == 0 || p > 16 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "verify2p devpos: p must be in 1..=16 (got {p})"
            )));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (hd_i, nh_i, nk_i, ch_i, nc_i, nt_i, p_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            chunk_len as i32,
            n_chunks as i32,
            n_stat_tiles as i32,
            p as i32,
        );
        let stats_smem = (head_dim * 33 * core::mem::size_of::<f32>()) as u32;
        let cfg_stats = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_stat_tiles as u32, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: stats_smem,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_verify2p_stats_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(stat_m)
                .arg(stat_l)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&nt_i)
                .arg(pos_dev)
                .arg(&p_i)
                .launch(cfg_stats)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let cfg_merge = LaunchConfig {
            grid_dim: (n_kv_head as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_verify2p_merge)
                .arg(stat_m)
                .arg(stat_l)
                .arg(mrg_m)
                .arg(mrg_l)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&nt_i)
                .arg(&p_i)
                .launch(cfg_merge)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let g = n_head / n_kv_head;
        let pv_smem = ((head_dim * 33 + 16 * g * 32) * core::mem::size_of::<f32>()) as u32;
        let cfg_pv = LaunchConfig {
            grid_dim: (n_kv_head as u32, n_chunks as u32, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: pv_smem,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_verify2p_pv_devpos)
                .arg(query)
                .arg(key_cache)
                .arg(value_cache)
                .arg(mrg_m)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(&scale)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&ch_i)
                .arg(pos_dev)
                .arg(&p_i)
                .launch(cfg_pv)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_split_combine_rows)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(cfg2)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }
}

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

    // ── CPU reference implementations ──

    fn cpu_partial_rope(
        q: &mut [f32],
        k: &mut [f32],
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos: usize,
        theta_base: f32,
    ) {
        let rotary_pairs = rotary_dim / 2;
        for head in 0..n_head {
            for pair in 0..rotary_pairs {
                let exponent = 2.0 * pair as f32 / rotary_dim as f32;
                let inv_freq = theta_base.powf(-exponent);
                let theta = pos as f32 * inv_freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                let q_off = head * head_dim;
                let q_i0 = q_off + pair;
                let q_i1 = q_i0 + rotary_pairs;
                let q0 = q[q_i0];
                let q1 = q[q_i1];
                q[q_i0] = q0 * cos_t - q1 * sin_t;
                q[q_i1] = q0 * sin_t + q1 * cos_t;

                if head < n_kv_head {
                    let k_off = head * head_dim;
                    let k_i0 = k_off + pair;
                    let k_i1 = k_i0 + rotary_pairs;
                    let k0 = k[k_i0];
                    let k1 = k[k_i1];
                    k[k_i0] = k0 * cos_t - k1 * sin_t;
                    k[k_i1] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
    }

    fn cpu_attention_decode(
        query: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_positions: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = n_kv_head * head_dim;
        let mut out = vec![0.0f32; n_head * head_dim];

        for h in 0..n_head {
            let head_off = h * head_dim;
            let kv_group = h * n_kv_head / n_head;
            let kv_off = kv_group * head_dim;

            let mut scores = vec![0.0f32; n_positions];
            let mut max_score = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
                let k_base = j * kv_stride + kv_off;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[head_off + d] * key_cache[k_base + d];
                }
                *score = dot * scale;
                if *score > max_score {
                    max_score = *score;
                }
            }

            let mut sum = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum += *score;
            }

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &score) in scores.iter().enumerate() {
                    let v_idx = j * kv_stride + kv_off + d;
                    acc += score * value_cache[v_idx];
                }
                out[head_off + d] = acc / sum;
            }
        }
        out
    }

    fn cpu_rmsnorm_batched(
        input: &[f32],
        gamma: &[f32],
        n_batch: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let inv_dim = 1.0 / head_dim as f32;
        let mut out = vec![0.0f32; n_batch * head_dim];
        for b in 0..n_batch {
            let base = b * head_dim;
            let mean_sq: f32 = (0..head_dim).map(|i| input[base + i].powi(2)).sum::<f32>() * inv_dim;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..head_dim {
                out[base + i] = input[base + i] * inv_rms * gamma[i];
            }
        }
        out
    }

    // ── Tests ──

    #[test]
    fn test_rope_partial_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        // Use head_dim=256 (Bonsai-27B), rotary_dim=64, n_head=4, n_kv_head=2.
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let n_head = 4usize;
        let n_kv_head = 2usize;
        let pos = 5usize;
        let theta_base = 1e7f32;

        let q_len = n_head * head_dim;
        let k_len = n_kv_head * head_dim;

        let mut q: Vec<f32> = (0..q_len).map(|i| ((i as f32) / 1000.0) - 0.5).collect();
        let mut k: Vec<f32> = (0..k_len).map(|i| ((i as f32) / 1000.0) - 0.3).collect();

        // CPU reference
        let mut q_cpu = q.clone();
        let mut k_cpu = k.clone();
        cpu_partial_rope(
            &mut q_cpu,
            &mut k_cpu,
            rotary_dim,
            head_dim,
            n_head,
            n_kv_head,
            pos,
            theta_base,
        );

        // GPU
        let q_dev = stream.clone_htod(&q).unwrap();
        let k_dev = stream.clone_htod(&k).unwrap();

        kernels
            .launch_rope(
                &stream,
                &q_dev,
                &k_dev,
                rotary_dim,
                head_dim,
                n_head,
                n_kv_head,
                pos,
                theta_base,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        stream.memcpy_dtoh(&q_dev, &mut q).unwrap();
        stream.memcpy_dtoh(&k_dev, &mut k).unwrap();

        let mut max_rel = 0f32;
        for i in 0..q_len {
            let denom = q_cpu[i].abs().max(1e-6);
            let rel = (q[i] - q_cpu[i]).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        for i in 0..k_len {
            let denom = k_cpu[i].abs().max(1e-6);
            let rel = (k[i] - k_cpu[i]).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!("[rope] head_dim={head_dim}, rotary_dim={rotary_dim}: max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-3, "RoPE max_rel {max_rel:.4e} > 1e-3");
    }

    /// Issue 618 — verify rope_partial_f32_devpos produces the same output as
    /// rope_partial_f32 (scalar pos).
    #[test]
    fn test_rope_devpos_matches_scalar() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let n_head = 4usize;
        let n_kv_head = 2usize;
        let pos = 5usize;
        let theta_base = 1e7f32;

        let q_len = n_head * head_dim;
        let k_len = n_kv_head * head_dim;

        let q_init: Vec<f32> = (0..q_len).map(|i| ((i as f32) / 1000.0) - 0.5).collect();
        let k_init: Vec<f32> = (0..k_len).map(|i| ((i as f32) / 1000.0) - 0.3).collect();

        // Scalar variant.
        let q_scalar = stream.clone_htod(&q_init).unwrap();
        let k_scalar = stream.clone_htod(&k_init).unwrap();
        kernels
            .launch_rope(&stream, &q_scalar, &k_scalar,
                rotary_dim, head_dim, n_head, n_kv_head, pos, theta_base)
            .expect("scalar launch");

        // Devpos variant — write pos to a 1-element device buffer.
        let pos_dev = stream.clone_htod(&[pos as i32]).unwrap();
        let q_devpos = stream.clone_htod(&q_init).unwrap();
        let k_devpos = stream.clone_htod(&k_init).unwrap();
        kernels
            .launch_rope_devpos(&stream, &q_devpos, &k_devpos,
                rotary_dim, head_dim, n_head, n_kv_head, &pos_dev, theta_base)
            .expect("devpos launch");
        stream.synchronize().expect("sync");

        let mut q_scalar_out = q_init.clone();
        let mut k_scalar_out = k_init.clone();
        let mut q_devpos_out = q_init.clone();
        let mut k_devpos_out = k_init.clone();
        stream.memcpy_dtoh(&q_scalar, &mut q_scalar_out).unwrap();
        stream.memcpy_dtoh(&k_scalar, &mut k_scalar_out).unwrap();
        stream.memcpy_dtoh(&q_devpos, &mut q_devpos_out).unwrap();
        stream.memcpy_dtoh(&k_devpos, &mut k_devpos_out).unwrap();

        let mut max_diff = 0f32;
        for i in 0..q_len {
            max_diff = max_diff.max((q_scalar_out[i] - q_devpos_out[i]).abs());
        }
        for i in 0..k_len {
            max_diff = max_diff.max((k_scalar_out[i] - k_devpos_out[i]).abs());
        }
        eprintln!("[rope_devpos_vs_scalar] max_diff={max_diff:.4e}");
        assert!(max_diff < 1e-6, "rope devpos max_diff {max_diff:.4e}");
    }

    #[test]
    fn test_split_qg_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let head_dim = 256usize;
        let n_head = 4usize;
        let qg_len = n_head * 2 * head_dim;
        let out_len = n_head * head_dim;

        // qg[h] = [q_values..., gate_values...]
        let qg: Vec<f32> = (0..qg_len).map(|i| i as f32 * 0.01).collect();

        let qg_dev = stream.clone_htod(&qg).unwrap();
        let q_dev = stream.alloc_zeros::<f32>(out_len).unwrap();
        let gate_dev = stream.alloc_zeros::<f32>(out_len).unwrap();

        kernels
            .launch_split_qg(&stream, &qg_dev, &q_dev, &gate_dev, head_dim, n_head)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut q = vec![0f32; out_len];
        let mut gate = vec![0f32; out_len];
        stream.memcpy_dtoh(&q_dev, &mut q).unwrap();
        stream.memcpy_dtoh(&gate_dev, &mut gate).unwrap();

        for h in 0..n_head {
            for d in 0..head_dim {
                let idx = h * head_dim + d;
                let q_src = h * 2 * head_dim + d;
                let gate_src = q_src + head_dim;
                assert!(
                    (q[idx] - qg[q_src]).abs() < 1e-6,
                    "q mismatch at h={h}, d={d}: got {}, expected {}",
                    q[idx],
                    qg[q_src]
                );
                assert!(
                    (gate[idx] - qg[gate_src]).abs() < 1e-6,
                    "gate mismatch at h={h}, d={d}: got {}, expected {}",
                    gate[idx],
                    qg[gate_src]
                );
            }
        }
        eprintln!("[split_qg] head_dim={head_dim}, n_head={n_head}: all match");
    }

    #[test]
    fn test_rmsnorm_batched_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let head_dim = 256usize;
        let n_batch = 4usize;
        let eps = 1e-6f32;

        let input: Vec<f32> = (0..n_batch * head_dim)
            .map(|i| (i as f32) / 100.0 - 5.0)
            .collect();
        let gamma: Vec<f32> = (0..head_dim).map(|i| 1.0 + (i as f32) * 0.001).collect();

        let cpu_out = cpu_rmsnorm_batched(&input, &gamma, n_batch, head_dim, eps);

        let input_dev = stream.clone_htod(&input).unwrap();
        let gamma_dev = stream.clone_htod(&gamma).unwrap();
        let out_dev = stream.alloc_zeros::<f32>(n_batch * head_dim).unwrap();

        kernels
            .launch_rmsnorm_batched(
                &stream,
                &input_dev,
                &gamma_dev,
                &out_dev,
                n_batch,
                head_dim,
                eps,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; n_batch * head_dim];
        stream.memcpy_dtoh(&out_dev, &mut gpu_out).unwrap();

        let mut max_rel = 0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            let denom = a.abs().max(1e-6);
            let rel = (a - b).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!("[rmsnorm_batched] head_dim={head_dim}, n_batch={n_batch}: max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-4, "RMSNorm batched max_rel {max_rel:.4e} > 1e-4");
    }

    #[test]
    fn test_kv_cache_append_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let kvd = 1024usize; // n_kv_head(4) * head_dim(256) for Bonsai-27B
        let max_seq = 16usize;
        let pos = 5usize;

        let k_vec: Vec<f32> = (0..kvd).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let v_vec: Vec<f32> = (0..kvd).map(|i| (i as f32) * 0.02 + 1.0).collect();
        let mut key_cache = vec![-99.0f32; max_seq * kvd];
        let mut value_cache = vec![-99.0f32; max_seq * kvd];

        let k_dev = stream.clone_htod(&k_vec).unwrap();
        let v_dev = stream.clone_htod(&v_vec).unwrap();
        let key_cache_dev = stream.clone_htod(&key_cache).unwrap();
        let value_cache_dev = stream.clone_htod(&value_cache).unwrap();

        kernels
            .launch_kv_cache_append(
                &stream,
                &k_dev,
                &v_dev,
                &key_cache_dev,
                &value_cache_dev,
                kvd,
                pos,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        // CPU reference: only modify position `pos`
        for i in 0..kvd {
            key_cache[pos * kvd + i] = k_vec[i];
            value_cache[pos * kvd + i] = v_vec[i];
        }

        stream
            .memcpy_dtoh(&key_cache_dev, &mut vec![0f32; max_seq * kvd])
            .ok();
        // Read back properly
        let mut key_cache_gpu = vec![0f32; max_seq * kvd];
        let mut value_cache_gpu = vec![0f32; max_seq * kvd];
        stream.memcpy_dtoh(&key_cache_dev, &mut key_cache_gpu).unwrap();
        stream.memcpy_dtoh(&value_cache_dev, &mut value_cache_gpu).unwrap();

        // Verify only position `pos` was written
        for j in 0..max_seq {
            for i in 0..kvd {
                let expected_k = if j == pos { k_vec[i] } else { -99.0 };
                let expected_v = if j == pos { v_vec[i] } else { -99.0 };
                assert!(
                    (key_cache_gpu[j * kvd + i] - expected_k).abs() < 1e-6,
                    "key_cache[{j}][{i}] = {}, expected {expected_k}",
                    key_cache_gpu[j * kvd + i]
                );
                assert!(
                    (value_cache_gpu[j * kvd + i] - expected_v).abs() < 1e-6,
                    "value_cache[{j}][{i}] = {}, expected {expected_v}",
                    value_cache_gpu[j * kvd + i]
                );
            }
        }
        eprintln!("[kv_cache_append] kvd={kvd}, pos={pos}: all match");
    }

    /// Issue 618 — verify kv_cache_append_f32_devpos matches scalar variant.
    #[test]
    fn test_kv_cache_append_devpos_matches_scalar() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let kvd = 1024usize;
        let max_seq = 16usize;
        let pos = 5usize;

        let k_vec: Vec<f32> = (0..kvd).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let v_vec: Vec<f32> = (0..kvd).map(|i| (i as f32) * 0.02 + 1.0).collect();
        let cache_init = vec![-99.0f32; max_seq * kvd];

        // Scalar variant.
        let k_dev = stream.clone_htod(&k_vec).unwrap();
        let v_dev = stream.clone_htod(&v_vec).unwrap();
        let key_s = stream.clone_htod(&cache_init).unwrap();
        let value_s = stream.clone_htod(&cache_init).unwrap();
        kernels
            .launch_kv_cache_append(&stream, &k_dev, &v_dev, &key_s, &value_s, kvd, pos)
            .expect("scalar");

        // Devpos variant.
        let pos_dev = stream.clone_htod(&[pos as i32]).unwrap();
        let key_d = stream.clone_htod(&cache_init).unwrap();
        let value_d = stream.clone_htod(&cache_init).unwrap();
        kernels
            .launch_kv_cache_append_devpos(&stream, &k_dev, &v_dev, &key_d, &value_d, kvd, &pos_dev)
            .expect("devpos");
        stream.synchronize().expect("sync");

        let mut key_s_out = vec![0f32; max_seq * kvd];
        let mut value_s_out = vec![0f32; max_seq * kvd];
        let mut key_d_out = vec![0f32; max_seq * kvd];
        let mut value_d_out = vec![0f32; max_seq * kvd];
        stream.memcpy_dtoh(&key_s, &mut key_s_out).unwrap();
        stream.memcpy_dtoh(&value_s, &mut value_s_out).unwrap();
        stream.memcpy_dtoh(&key_d, &mut key_d_out).unwrap();
        stream.memcpy_dtoh(&value_d, &mut value_d_out).unwrap();

        let mut max_diff = 0f32;
        for i in 0..max_seq * kvd {
            max_diff = max_diff.max((key_s_out[i] - key_d_out[i]).abs());
            max_diff = max_diff.max((value_s_out[i] - value_d_out[i]).abs());
        }
        eprintln!("[kv_cache_append_devpos_vs_scalar] max_diff={max_diff:.4e}");
        assert!(max_diff < 1e-6, "kv_cache_append devpos max_diff {max_diff:.4e}");
    }

    #[test]
    fn test_attention_decode_head_dim_256_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        // Bonsai-27B shape: head_dim=256, n_head=4 (subset), n_kv_head=2 (GQA 2:1)
        let n_head = 4usize;
        let n_kv_head = 2usize;
        let head_dim = 256usize;
        let n_positions = 3usize; // multi-position to exercise online softmax

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        // Build deterministic test data with distinct values per head/position.
        let mut query = vec![0.0f32; q_len];
        let mut key_cache = vec![0.0f32; kv_len];
        let mut value_cache = vec![0.0f32; kv_len];

        for h in 0..n_head {
            for d in 0..head_dim {
                query[h * head_dim + d] = 0.01 * ((h + 1) as f32) + 0.001 * (d as f32);
            }
        }
        for j in 0..n_positions {
            for kvh in 0..n_kv_head {
                for d in 0..head_dim {
                    let base = j * n_kv_head * head_dim + kvh * head_dim;
                    key_cache[base + d] = 0.01 * ((j + 1) as f32) + 0.001 * (d as f32);
                    value_cache[base + d] = 0.02 * ((j + 1) as f32) + 0.001 * (d as f32);
                }
            }
        }

        let q_dev = stream.clone_htod(&query).unwrap();
        let k_dev = stream.clone_htod(&key_cache).unwrap();
        let v_dev = stream.clone_htod(&value_cache).unwrap();
        let out_dev = stream.alloc_zeros::<f32>(q_len).unwrap();

        kernels
            .launch_attention_decode(
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                &out_dev,
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; q_len];
        stream.memcpy_dtoh(&out_dev, &mut gpu_out).unwrap();

        let cpu_out = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );

        let mut max_diff = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..q_len {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
        }
        eprintln!(
            "[attention_decode] head_dim={head_dim}, n_pos={n_positions}: max_diff={max_diff:.6e}"
        );
        assert!(
            max_diff <= TOL,
            "Attention decode mismatch (head_dim={head_dim}): max_diff={max_diff:.6} at idx={worst_idx} \
             (gpu={:.6}, cpu={:.6})",
            gpu_out[worst_idx],
            cpu_out[worst_idx]
        );
    }

    #[test]
    fn test_attention_decode_head_dim_128_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        // Also test head_dim=128 (used by many Qwen variants)
        let n_head = 4usize;
        let n_kv_head = 2usize;
        let head_dim = 128usize;
        let n_positions = 5usize;

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        let mut query = vec![0.0f32; q_len];
        let mut key_cache = vec![0.0f32; kv_len];
        let mut value_cache = vec![0.0f32; kv_len];

        for h in 0..n_head {
            for d in 0..head_dim {
                query[h * head_dim + d] = 0.01 * ((h + 1) as f32) + 0.001 * (d as f32);
            }
        }
        for j in 0..n_positions {
            for kvh in 0..n_kv_head {
                for d in 0..head_dim {
                    let base = j * n_kv_head * head_dim + kvh * head_dim;
                    key_cache[base + d] = 0.01 * ((j + 1) as f32) + 0.001 * (d as f32);
                    value_cache[base + d] = 0.02 * ((j + 1) as f32) + 0.001 * (d as f32);
                }
            }
        }

        let q_dev = stream.clone_htod(&query).unwrap();
        let k_dev = stream.clone_htod(&key_cache).unwrap();
        let v_dev = stream.clone_htod(&value_cache).unwrap();
        let out_dev = stream.alloc_zeros::<f32>(q_len).unwrap();

        kernels
            .launch_attention_decode(
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                &out_dev,
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; q_len];
        stream.memcpy_dtoh(&out_dev, &mut gpu_out).unwrap();

        let cpu_out = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );

        let mut max_diff = 0.0f32;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
        eprintln!(
            "[attention_decode] head_dim={head_dim}, n_pos={n_positions}: max_diff={max_diff:.6e}"
        );
        assert!(max_diff <= TOL, "head_dim=128 attention max_diff={max_diff:.6}");
    }

    /// Issue 618 — verify attention_decode_f32_devpos matches scalar variant.
    /// The devpos kernel computes n_positions = *pos_dev + 1 internally.
    #[test]
    fn test_attention_decode_devpos_matches_scalar() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let n_head = 4usize;
        let n_kv_head = 2usize;
        let head_dim = 128usize;
        let n_positions = 5usize;
        let pos = n_positions - 1; // n_positions = pos + 1

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        let mut query = vec![0.0f32; q_len];
        let mut key_cache = vec![0.0f32; kv_len];
        let mut value_cache = vec![0.0f32; kv_len];

        for h in 0..n_head {
            for d in 0..head_dim {
                query[h * head_dim + d] = 0.01 * ((h + 1) as f32) + 0.001 * (d as f32);
            }
        }
        for j in 0..n_positions {
            for kvh in 0..n_kv_head {
                for d in 0..head_dim {
                    let base = j * n_kv_head * head_dim + kvh * head_dim;
                    key_cache[base + d] = 0.01 * ((j + 1) as f32) + 0.001 * (d as f32);
                    value_cache[base + d] = 0.02 * ((j + 1) as f32) + 0.001 * (d as f32);
                }
            }
        }

        let q_dev = stream.clone_htod(&query).unwrap();
        let k_dev = stream.clone_htod(&key_cache).unwrap();
        let v_dev = stream.clone_htod(&value_cache).unwrap();

        // Scalar variant.
        let out_s = stream.alloc_zeros::<f32>(q_len).unwrap();
        kernels
            .launch_attention_decode(&stream, &q_dev, &k_dev, &v_dev, &out_s,
                head_dim, n_head, n_kv_head, n_positions)
            .expect("scalar");

        // Devpos variant — pos_dev = pos (n_positions = pos + 1).
        let pos_dev = stream.clone_htod(&[pos as i32]).unwrap();
        let out_d = stream.alloc_zeros::<f32>(q_len).unwrap();
        kernels
            .launch_attention_decode_devpos(&stream, &q_dev, &k_dev, &v_dev, &out_d,
                head_dim, n_head, n_kv_head, &pos_dev)
            .expect("devpos");
        stream.synchronize().expect("sync");

        let mut out_s_cpu = vec![0f32; q_len];
        let mut out_d_cpu = vec![0f32; q_len];
        stream.memcpy_dtoh(&out_s, &mut out_s_cpu).unwrap();
        stream.memcpy_dtoh(&out_d, &mut out_d_cpu).unwrap();

        let mut max_diff = 0f32;
        for i in 0..q_len {
            max_diff = max_diff.max((out_s_cpu[i] - out_d_cpu[i]).abs());
        }
        eprintln!("[attention_decode_devpos_vs_scalar] max_diff={max_diff:.4e}");
        assert!(max_diff < 1e-4, "attention_decode devpos max_diff {max_diff:.4e}");
    }

    #[test]
    fn test_output_gate_correctness() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let n = 1024usize; // q_dim = n_head * head_dim
        let attn: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005).collect();

        // CPU reference: attn[i] * sigmoid(gate[i])
        let cpu_out: Vec<f32> = (0..n)
            .map(|i| {
                let g = gate[i];
                let sig = 1.0 / (1.0 + (-g).exp());
                attn[i] * sig
            })
            .collect();

        let attn_dev = stream.clone_htod(&attn).unwrap();
        let gate_dev = stream.clone_htod(&gate).unwrap();

        kernels
            .launch_output_gate(&stream, &attn_dev, &gate_dev, n)
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; n];
        stream.memcpy_dtoh(&attn_dev, &mut gpu_out).unwrap();

        let mut max_rel = 0f32;
        for i in 0..n {
            let denom = cpu_out[i].abs().max(1e-6);
            let rel = (gpu_out[i] - cpu_out[i]).abs() / denom;
            max_rel = max_rel.max(rel);
        }
        eprintln!("[output_gate] n={n}: max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-5, "output gate max_rel {max_rel:.4e} > 1e-5");
    }

    // ── CPU reference: RoPE backward (inverse rotation, sin negated) ──

    fn cpu_partial_rope_backward(
        q: &mut [f32],
        k: &mut [f32],
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        pos: usize,
        theta_base: f32,
    ) {
        let rotary_pairs = rotary_dim / 2;
        for head in 0..n_head {
            for pair in 0..rotary_pairs {
                let exponent = 2.0 * pair as f32 / rotary_dim as f32;
                let inv_freq = theta_base.powf(-exponent);
                let theta = pos as f32 * inv_freq;
                let cos_t = theta.cos();
                let sin_t = -theta.sin(); // NEGATED for backward

                let q_off = head * head_dim;
                let q_i0 = q_off + pair;
                let q_i1 = q_i0 + rotary_pairs;
                let q0 = q[q_i0];
                let q1 = q[q_i1];
                q[q_i0] = q0 * cos_t - q1 * sin_t;
                q[q_i1] = q0 * sin_t + q1 * cos_t;

                if head < n_kv_head {
                    let k_off = head * head_dim;
                    let k_i0 = k_off + pair;
                    let k_i1 = k_i0 + rotary_pairs;
                    let k0 = k[k_i0];
                    let k1 = k[k_i1];
                    k[k_i0] = k0 * cos_t - k1 * sin_t;
                    k[k_i1] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
    }

    /// Issue 641 T7.4 — verify rope_partial_backward_f32 matches the CPU inverse
    /// rotation, AND that forward+backward = identity (round-trip test).
    #[test]
    fn test_rope_backward_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = AttentionKernels::new(ctx).expect("compile");

        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let n_head = 4usize;
        let n_kv_head = 2usize;
        let pos = 7usize;
        let theta_base = 1e7f32;

        let q_len = n_head * head_dim;
        let k_len = n_kv_head * head_dim;

        // Start with random gradient values.
        let q_orig: Vec<f32> = (0..q_len).map(|i| ((i as f32) / 1000.0) - 0.5).collect();
        let k_orig: Vec<f32> = (0..k_len).map(|i| ((i as f32) / 1000.0) - 0.3).collect();

        // CPU reference: apply backward directly.
        let mut q_cpu = q_orig.clone();
        let mut k_cpu = k_orig.clone();
        cpu_partial_rope_backward(
            &mut q_cpu, &mut k_cpu, rotary_dim, head_dim, n_head, n_kv_head, pos, theta_base,
        );

        // GPU backward.
        let q_dev = stream.clone_htod(&q_orig).unwrap();
        let k_dev = stream.clone_htod(&k_orig).unwrap();
        kernels
            .launch_rope_backward(
                &stream, &q_dev, &k_dev, rotary_dim, head_dim, n_head, n_kv_head,
                pos, theta_base,
            )
            .expect("launch backward");
        stream.synchronize().expect("sync");

        let mut q_gpu = vec![0f32; q_len];
        let mut k_gpu = vec![0f32; k_len];
        stream.memcpy_dtoh(&q_dev, &mut q_gpu).unwrap();
        stream.memcpy_dtoh(&k_dev, &mut k_gpu).unwrap();

        // Check backward matches CPU.
        let mut max_rel = 0f32;
        for i in 0..q_len {
            let denom = q_cpu[i].abs().max(1e-6);
            max_rel = max_rel.max((q_gpu[i] - q_cpu[i]).abs() / denom);
        }
        for i in 0..k_len {
            let denom = k_cpu[i].abs().max(1e-6);
            max_rel = max_rel.max((k_gpu[i] - k_cpu[i]).abs() / denom);
        }
        eprintln!("[rope_backward] max_rel={max_rel:.4e}");
        assert!(max_rel < 1e-3, "RoPE backward max_rel {max_rel:.4e} > 1e-3");

        // Round-trip test: forward then backward should return to original.
        // Apply forward RoPE on the backward output.
        kernels
            .launch_rope(
                &stream, &q_dev, &k_dev, rotary_dim, head_dim, n_head, n_kv_head,
                pos, theta_base,
            )
            .expect("launch forward");
        stream.synchronize().expect("sync");

        let mut q_rt = vec![0f32; q_len];
        let mut k_rt = vec![0f32; k_len];
        stream.memcpy_dtoh(&q_dev, &mut q_rt).unwrap();
        stream.memcpy_dtoh(&k_dev, &mut k_rt).unwrap();

        let mut rt_err = 0f32;
        for i in 0..q_len {
            rt_err = rt_err.max((q_rt[i] - q_orig[i]).abs());
        }
        for i in 0..k_len {
            rt_err = rt_err.max((k_rt[i] - k_orig[i]).abs());
        }
        eprintln!("[rope_roundtrip] max_abs_err={rt_err:.4e}");
        assert!(rt_err < 1e-3, "RoPE round-trip err {rt_err:.4e} > 1e-3");
    }
}
