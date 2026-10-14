//! Issue 1066 (riir-ai) — the **GDN chunked recurrence, MMA class**: a port
//! of llama.cpp's merged `ggml/src/ggml-cuda/gated-delta-net-mma.cu`
//! (PR #29353 "cuda, hip: GDN chunked kernel", merge `62df1815`, 2026-10-14,
//! MIT, by am17an) into the cudarc prefill stack — the Plan-605-T5 defense
//! lane fired by the ai-971 merge-moment protocol.
//!
//! # What this replaces / composes
//!
//! The Arm-12 `gdn_chunked` slot previously ran the v3 four-kernel CS=64
//! scalar pipeline ([`crate::prefill_cuda_gdn_chunked`], zero tensor-core
//! ops). This module lands the **MMA class** INTO the same slot: ONE
//! persistent kernel per layer-pass, CS=16, all five products
//! `mma.sync.m16n8k16` on double-BF16 hi/lo residual pairs (3 mma per 16-k
//! step, lo·lo dropped — the near-lossless numerics mechanism, KL 4.63e-4 /
//! top-1 98.82% source-reported), FP32 state in per-warp register tiles
//! (load once / store once), in-kernel WY solve (warp-0 shfl-scan gate
//! prefix → Gram with intra-chunk decay → in-register (I+L)⁻¹
//! one-row-per-warp forward substitution → solved → out), union smem scratch
//! (~36 KB / ~85 KB, smpbo-gated), SM-occupancy-keyed V∈{32,128} ladder,
//! benchmark-gated CC admission with incumbent fallback.
//!
//! The v3 path stays the in-slot fallback (it handles partial last chunks
//! and any p); the rowpar incumbent stays below it (bit-identical ROWPAR).
//!
//! # Numerics contract (ai Issue 1066 / Issue 734's owner-approved class)
//!
//! NOT bit-identical to the rowpar kernel (chunked batched-matmul reduction
//! order + the double-BF16 product). Gates on **argmax-stability** +
//! distributional agreement — the same class as v3, one rounding step wider
//! (the bf16 residual pair keeps ~16 mantissa bits per product element vs
//! f32's 24). DO NOT replace the 3-mma residual-pair product with a plain
//! bf16 product — the residual pair IS the numerics contract.
//!
//! Deviations from the reference (all fidelity-preserving, documented here):
//! - CUDA-only (no HIP/MUSA paths); compiled for sm_89 via NVRTC.
//! - Single-sequence prefill (no `n_seqs`/`rq3` head replication — our
//!   expand stage materializes q/k per v-head, so `h % H_k == h`).
//! - `g` input is the per-token decay ALPHA (we take `__logf` at the
//!   prefix-scan load) with the v3-proven −60 cumulative floor; the
//!   reference sums a precomputed log-decay without a floor.
//! - `scale = rsqrtf(128)` computed in-kernel and applied at q staging (the
//!   v3/rowpar math applies the same scale at the output — identical up to
//!   fp reassociation; the reference's form is kept).
//! - bf16 via inline PTX `cvt` (NVRTC standalone has no `cuda_bf16.h`);
//!   tile/ldmatrix/mma helpers are a minimal transcription of the CUDA path
//!   of llama.cpp `mma.cuh` (fragment accessors + swizzle + ldmatrix
//!   swizzled/trans loads + `mma.sync.aligned.m16n8k16...bf16.f32`).
//!
//! # Admission (the reference's ladder, single-seq form)
//!
//! `H ∈ {16,32,48,64}` + `n_tokens ≥ 64` (kernel-level; the SEAM adds the
//! scope guard `p ≥ 128 && p % 16 == 0` — partial last chunks route to v3),
//! CC-gated sm_86/89/120/121 (we run sm_89), and the occupancy-keyed value
//! split: `H*2 ≤ nsm` → V=32 (one-block if `H*4 ≤ nsm`, else two-block
//! launch_bounds), else V=128 whole-head. 4090 (nsm=128, Bonsai H=48):
//! `96 ≤ 128 && 192 > 128` → V=32 two-blocks, grid `4*H = 192` blocks.
//!
//! # G4
//!
//! Zero device scratch buffers and zero host allocations on the launch path:
//! all intermediates live in the dynamic-smem union + registers; the smem
//! size is a compile-time constant per instantiation (`static_assert`-pinned
//! against the host constant).

use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};

/// Dynamic smem for the V=32 instantiation — MUST equal
/// `sizeof(gdn_shared<16, 32>)` (static_assert-pinned in the CUDA source).
pub const GDN_MMA_SMEM_V32: u32 = 38_016;
/// Dynamic smem for the V=128 instantiation — MUST equal
/// `sizeof(gdn_shared<16, 128>)`.
pub const GDN_MMA_SMEM_V128: u32 = 93_312;

pub(crate) const GDN_MMA_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// Minimal transcription of the CUDA path of llama.cpp mma.cuh (MIT) — the
// fragment accessors, the XOR swizzle, the ldmatrix loads, and the bf16
// m16n8k16 mma. threadIdx.x IS the lane for every warp (block is dim3(32,8)).
// ---------------------------------------------------------------------------
typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_f32(bf16_t v)
{
    float r;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(r) : "h"(v));
    return r;
}
__device__ __forceinline__ bf16_t f32_to_bf16_rn(float v)
{
    bf16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v));
    return r;
}

// XOR swizzle in bf16x2 (4-byte) units — mma.cuh swizzle<stride, bf16x2>:
// stride_bytes = stride*4; shift = 2.
__device__ __forceinline__ int swizzle_bf162(int offset, int i, int stride)
{
    const int stride_bytes = stride * 4;
    if (stride_bytes % 32 != 0) return offset;
    if (stride_bytes % 64 != 0) return offset ^ (((i >> 2) & 1) << 2);
    if (stride_bytes % 128 != 0) return offset ^ (((i >> 1) & 3) << 2);
    return offset ^ ((i & 7) << 2);
}

// A-fragment tile<16,8,bf16x2> accumulator accessors: acc_get_i/get_j are
// the f32 OUTPUT (tile<16,8,float>) positions.
__device__ __forceinline__ int acc_get_i(int l)
{
    return ((l >> 1) << 3) + ((int)threadIdx.x >> 2);
}
__device__ __forceinline__ int acc_get_j(int l)
{
    return (((int)threadIdx.x & 3) << 1) + (l & 1);
}

// ldmatrix loads — origin in bf16x2 units, stride = matrix row length / 2.
__device__ __forceinline__ void ldmatrix_a16_sw(
    unsigned (&x)[4], const bf16_t* __restrict__ data, int row, int col, int stride)
{
    const int i = (int)threadIdx.x & 15;
    const int j = ((int)threadIdx.x >> 4) << 2; // {0,4}
    int o = row * stride + (col >> 1) + i * stride + j;
    o = swizzle_bf162(o, i, stride);
    const unsigned* p = reinterpret_cast<const unsigned*>(data) + o;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]) : "l"(p));
}
__device__ __forceinline__ void ldmatrix_b8_sw(
    unsigned (&x)[2], const bf16_t* __restrict__ data, int row, int col, int stride)
{
    const int i = (int)threadIdx.x & 7;
    const int j = ((int)threadIdx.x >> 3 & 1) << 2; // {0,4,0,4}
    int o = row * stride + (col >> 1) + i * stride + j;
    o = swizzle_bf162(o, i, stride);
    const unsigned* p = reinterpret_cast<const unsigned*>(data) + o;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
        : "=r"(x[0]), "=r"(x[1]) : "l"(p));
}
__device__ __forceinline__ void ldmatrix_a16_trans_sw(
    unsigned (&x)[4], const bf16_t* __restrict__ data, int row, int col, int stride)
{
    // mma.cuh load_ldmatrix_trans_swizzled, I=16: i = lane%16, j = (lane/16)*4,
    // origin = col*stride + row/2, x4.trans with the {x0,x2,x1,x3} swap.
    const int i = (int)threadIdx.x & 15;
    const int j = ((int)threadIdx.x >> 4) << 2;
    int o = col * stride + (row >> 1) + i * stride + j;
    o = swizzle_bf162(o, i, stride);
    const unsigned* p = reinterpret_cast<const unsigned*>(data) + o;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%2,%1,%3}, [%4];"
        : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]) : "l"(p));
}
__device__ __forceinline__ void ldmatrix_b8_trans_sw(
    unsigned (&x)[2], const bf16_t* __restrict__ data, int row, int col, int stride)
{
    // mma.cuh load_ldmatrix_trans_swizzled, I=8: i = lane%16, j = 0, x2.trans.
    const int i = (int)threadIdx.x & 15;
    int o = col * stride + (row >> 1) + i * stride;
    o = swizzle_bf162(o, i, stride);
    const unsigned* p = reinterpret_cast<const unsigned*>(data) + o;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
        : "=r"(x[0]), "=r"(x[1]) : "l"(p));
}

__device__ __forceinline__ void mma_bf16(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2])
{
    // NOT volatile: the "+f"/"r" constraints carry the data dependencies.
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// The hi/lo residual-pair matrix (the reference's matrix<R,C> verbatim, bf16
// via PTX cvt). index() is the ldmatrix-contract swizzled element index.
// ---------------------------------------------------------------------------
template <int R, int C> struct matrix {
    bf16_t hi[R * C], lo[R * C];

    static __device__ __forceinline__ int index(int r, int c)
    {
        return 2 * swizzle_bf162(r * (C / 2) + (c >> 1), r, C / 2) + (c & 1);
    }
    __device__ __forceinline__ void store2(int r, int c, float2 x)
    {
        const int i = index(r, c);
        const bf16_t hx = f32_to_bf16_rn(x.x);
        const bf16_t hy = f32_to_bf16_rn(x.y);
        const unsigned hxy = ((unsigned)hy << 16) | (unsigned)hx;
        const unsigned lxy =
            ((unsigned)f32_to_bf16_rn(x.x - bf16_to_f32(hx)) << 16)
            | (unsigned)f32_to_bf16_rn(x.y - bf16_to_f32(hy));
        *reinterpret_cast<unsigned*>(hi + i) = hxy;
        *reinterpret_cast<unsigned*>(lo + i) = lxy;
    }
    __device__ __forceinline__ void store(int r, int c, float x)
    {
        const int i = index(r, c);
        const bf16_t h = f32_to_bf16_rn(x);
        hi[i] = h;
        lo[i] = f32_to_bf16_rn(x - bf16_to_f32(h));
    }
};

template <int C, int V> struct alignas(128) gdn_shared {
    matrix<C, 128> q, k;
    matrix<C, C> p, inverse;
    matrix<C, V> delta;
    union {
        matrix<128, V> state;
        matrix<C, V> solved;
    } scratch;
    float lower[C * C], prefix[C], beta[C];
};

static_assert(sizeof(gdn_shared<16, 32>) == 38016, "smem drift vs host GDN_MMA_SMEM_V32");
static_assert(sizeof(gdn_shared<16, 128>) == 93312, "smem drift vs host GDN_MMA_SMEM_V128");

// store an acc fragment into a matrix (the reference's store(), i+=2 pairs)
template <int R, int C>
__device__ __forceinline__ void store_acc(matrix<R, C>& m, const float (&x)[4], int r, int c)
{
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        m.store2(r + acc_get_i(i), c + acc_get_j(i), make_float2(x[i], x[i + 1]));
    }
}

// The product: C_tile[row][col] += sum_k A[row][k]*B[col][k] over double-BF16
// residual pairs — 3 mma per 16-k step, lo*lo dropped (the reference's
// product verbatim). `c` MUST be zero-initialized by the caller (or carry a
// live accumulator, as the state update does).
template <int K, bool TA = false, bool TB = false, int AR, int AC, int BR, int BC>
__device__ __forceinline__ void product(
    float (&c)[4], const matrix<AR, AC>& a, const matrix<BR, BC>& b, int row, int col)
{
#pragma unroll
    for (int k = 0; k < K; k += 16) {
        unsigned ah[4], al[4], bh[2], bl[2];
        if (TA) { ldmatrix_a16_trans_sw(ah, a.hi, row, k, AC / 2);
                  ldmatrix_a16_trans_sw(al, a.lo, row, k, AC / 2); }
        else    { ldmatrix_a16_sw(ah, a.hi, row, k, AC / 2);
                  ldmatrix_a16_sw(al, a.lo, row, k, AC / 2); }
        if (TB) { ldmatrix_b8_trans_sw(bh, b.hi, col, k, BC / 2);
                  ldmatrix_b8_trans_sw(bl, b.lo, col, k, BC / 2); }
        else    { ldmatrix_b8_sw(bh, b.hi, col, k, BC / 2);
                  ldmatrix_b8_sw(bl, b.lo, col, k, BC / 2); }
        mma_bf16(c, al, bh);
        mma_bf16(c, ah, bl);
        mma_bf16(c, ah, bh);
    }
}

// ---------------------------------------------------------------------------
// The kernel body — gdn_single<C, V> from the reference, single-sequence
// form (blockIdx.x = (slice, h)), qkvx layout [p, 3*v_dim] (q | k | v, each
// head-major over v_dim = n_v*128), beta/decay(alpha) [p, n_v] token-major,
// state [n_v, 128, 128] in/out (state[h*16384 + v*128 + d]), out [p, v_dim].
// ---------------------------------------------------------------------------
template <int C, int V>
static __device__ __forceinline__ void gdn_mma_body(
    const float* __restrict__ qkvx,
    const float* __restrict__ beta_in,
    const float* __restrict__ decay_in,
    float* __restrict__ state,
    float* __restrict__ dst,
    int n_tokens,
    int n_v)
{
    constexpr int D = 128;
    constexpr int WARPS = 8;
    constexpr int N = 8;
    static_assert(C == 16 && (C & (C - 1)) == 0 && V % N == 0 && D % V == 0,
                  "invalid GDN tile");
    const float scale = rsqrtf((float)D);
    extern __shared__ __align__(128) unsigned char gdn_bytes[];
    gdn_shared<C, V>& s = *reinterpret_cast<gdn_shared<C, V>*>(gdn_bytes);
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int slice = blockIdx.x % (D / V);
    const int h = blockIdx.x / (D / V);
    const int v0 = slice * V;
    const int soff = h * (D * D);
    const int v_dim = n_v * D;
    const int row3 = 3 * v_dim;
    const int qbase = h * D, kbase = v_dim + h * D, vbase = 2 * v_dim + h * D;
    float* out = dst + (size_t)h * D + v0;
    constexpr int STATE_TILES = (D / 16) * (V / N);
    constexpr int PER_WARP = STATE_TILES / WARPS;
    static_assert(STATE_TILES % WARPS == 0, "state must divide across warps");
    float state_acc[PER_WARP][4];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int tile_index = warp + j * WARPS;
        const int r = (tile_index / (V / N)) * 16, c = (tile_index % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            state_acc[j][i] =
                state[soff + (size_t)(v0 + c + acc_get_j(i)) * D + r + acc_get_i(i)];
        }
    }
    auto read_qk = [&](int t, float4& q, float4& k) {
        t += warp;
        q = k = make_float4(0, 0, 0, 0);
        if (t < n_tokens) {
            const size_t off = (size_t)t * row3 + 4 * lane;
            if ((((unsigned long long)(qkvx + qbase + off)
                  | (unsigned long long)(qkvx + kbase + off)) & 15) == 0) {
                q = *reinterpret_cast<const float4*>(qkvx + qbase + off);
                k = *reinterpret_cast<const float4*>(qkvx + kbase + off);
            } else {
                q = make_float4(qkvx[qbase + off], qkvx[qbase + off + 1],
                                qkvx[qbase + off + 2], qkvx[qbase + off + 3]);
                k = make_float4(qkvx[kbase + off], qkvx[kbase + off + 1],
                                qkvx[kbase + off + 2], qkvx[kbase + off + 3]);
            }
        }
    };
    // Carry the first eight input rows across chunks to hide global-load
    // latency (the reference's PREFETCH — V32 shapes only).
    const bool PREFETCH = (C == 16 && V == 32);
    float4 next_q, next_k;
    if (PREFETCH) {
        read_qk(0, next_q, next_k);
    }
    for (int t0 = 0; t0 < n_tokens; t0 += C) {
        const long long rem = (long long)n_tokens - t0;
        const int valid = rem >= (long long)C ? C : (int)rem;
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            float4 q, k;
            if (PREFETCH && j == 0) {
                q = next_q;
                k = next_k;
            } else {
                read_qk(t0 + 8 * j, q, k);
            }
            s.q.store2(t, d, make_float2(q.x * scale, q.y * scale));
            s.q.store2(t, d + 2, make_float2(q.z * scale, q.w * scale));
            s.k.store2(t, d, make_float2(k.x, k.y));
            s.k.store2(t, d + 2, make_float2(k.z, k.w));
        }
        // Gates in parallel before the FP32 prefix scan. Our input is decay
        // ALPHA: take __logf at load (v3's form) + the -60 cumulative floor
        // (the v3-proven class; keeps alpha==0 from poisoning the ratios).
        if (warp == 0) {
            float carry = 0.f;
            for (int base = 0; base < C; base += 32) {
                const int t = base + lane;
                float g = t < valid ? __logf(decay_in[(size_t)(t0 + t) * n_v + h]) : 0.f;
                const float beta = t < valid ? beta_in[(size_t)(t0 + t) * n_v + h] : 0.f;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const float previous = __shfl_up_sync(0xffffffffu, g, offset, 32);
                    if (lane >= offset) {
                        g += previous;
                    }
                }
                g += carry;
                if (g < -60.0f) g = -60.0f;
                if (t < C) {
                    s.prefix[t] = g;
                    s.beta[t] = beta;
                }
                carry = __shfl_sync(0xffffffffu, g, 31, 32);
            }
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS;
            store_acc(s.scratch.state, state_acc[j], (ti / (V / N)) * 16,
                      (ti % (V / N)) * N);
        }
        __syncthreads();
        if (PREFETCH) {
            if (t0 + C < n_tokens) {
                read_qk(t0 + C, next_q, next_k);
            }
        }
        // V32 overlaps Gram work with one output tile per projection warp.
        const bool SPLIT_PROJECTION = (C == 16 && V == 32);
        float base_output[((C / 16) * (V / N) + WARPS - 1) / WARPS][4];
#pragma unroll
        for (int bi = 0; bi < (int)((C / 16) * (V / N) + WARPS - 1) / WARPS; ++bi) {
#pragma unroll
            for (int i = 0; i < 4; ++i) base_output[bi][i] = 0.f;
        }
        // delta = beta * (v - decay * K S), base output = decay * Q S
        auto project = [&]() {
            if (V == 32) {
                if (warp < 4) return;
            }
            for (int ti = (V == 32 ? warp - 4 : warp); ti < (C / 16) * (V / N);
                 ti += (V == 32 ? 4 : WARPS)) {
                const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
                float qs[4] = {0.f, 0.f, 0.f, 0.f};
                float ks[4] = {0.f, 0.f, 0.f, 0.f};
                product<D, false, true>(qs, s.q, s.scratch.state, r, c);
                product<D, false, true>(ks, s.k, s.scratch.state, r, c);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int t = r + acc_get_i(i), v = c + acc_get_j(i);
                    const float decay = expf(s.prefix[t]);
                    const float vv =
                        t < valid ? qkvx[(size_t)(t0 + t) * row3 + vbase + v0 + v] : 0.f;
                    s.delta.store(t, v, s.beta[t] * (vv - decay * ks[i]));
                    base_output[ti / WARPS][i] = decay * qs[i];
                }
            }
        };
        // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
        for (int ti = warp; ti < 2 * (C / 16) * (C / N) && (V != 32 || warp < 4);
             ti += (V == 32 ? 4 : WARPS)) {
            const bool gram = ti < (C / 16) * (C / N);
            const int t = ti % ((C / 16) * (C / N));
            const int r = (t / (C / N)) * 16, c = (t % (C / N)) * N;
            float x[4] = {0.f, 0.f, 0.f, 0.f};
            if (gram) {
                product<D>(x, s.k, s.k, r, c);
            } else {
                product<D>(x, s.q, s.k, r, c);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rr = r + acc_get_i(i), cc = c + acc_get_j(i);
                const float v =
                    rr < valid && cc <= rr ? x[i] * expf(s.prefix[rr] - s.prefix[cc]) : 0.f;
                if (gram) {
                    if (cc < rr) {
                        s.lower[rr * C + cc] = v * s.beta[rr];
                    }
                } else {
                    s.p.store(rr, cc, v);
                }
            }
        }
        if (V == 32) {
            project();
        }
        __syncthreads();
        // inverse = (I + A)^-1 by forward substitution
        for (int row = warp; row < C; row += WARPS) {
            float x[(C + 31) / 32];
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                const int col = lane + 32 * j;
                x[j] = col == row ? 1.f : (col < row ? -s.lower[row * C + col] : 0.f);
            }
            for (int pivot = row - 1; pivot > 0; --pivot) {
                const float xp = __shfl_sync(0xffffffffu, x[pivot / 32], pivot % 32, 32);
#pragma unroll
                for (int j = 0; j < (C + 31) / 32; ++j) {
                    const int col = lane + 32 * j;
                    if (col < pivot) {
                        x[j] -= xp * s.lower[pivot * C + col];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                if (lane + 32 * j < C) {
                    s.inverse.store(row, lane + 32 * j, x[j]);
                }
            }
        }
        if (V != 32) {
            project();
        }
        __syncthreads();
        // solved = inverse * delta
        for (int ti = warp; ti < (C / 16) * (V / N); ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            float x[4] = {0.f, 0.f, 0.f, 0.f};
            product<C, false, true>(x, s.inverse, s.delta, r, c);
            store_acc(s.scratch.solved, x, r, c);
        }
        __syncthreads();
        // out = base output + P * solved
        for (int ti = (SPLIT_PROJECTION ? warp - 4 : warp);
             ti < (C / 16) * (V / N) && (!SPLIT_PROJECTION || warp >= 4); ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            float x[4] = {0.f, 0.f, 0.f, 0.f};
            product<C, false, true>(x, s.p, s.scratch.solved, r, c);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = r + acc_get_i(i), v = c + acc_get_j(i);
                if (t < valid) {
                    out[(size_t)(t0 + t) * v_dim + v] = base_output[ti / WARPS][i] + x[i];
                }
            }
        }
        // S = decay(chunk) * S + K^T * (solved * decay to chunk end)
        for (int i = tid; i < C * V; i += 256) {
            const int t = i / V, v = i % V;
            const int j = s.scratch.solved.index(t, v);
            const float x =
                bf16_to_f32(s.scratch.solved.hi[j]) + bf16_to_f32(s.scratch.solved.lo[j]);
            s.delta.store(t, v, x * expf(s.prefix[valid - 1] - s.prefix[t]));
        }
        __syncthreads();
        const float decay = expf(s.prefix[valid - 1]);
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16,
                      c = (ti % (V / N)) * N;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                state_acc[j][i] *= decay;
            }
            product<C, true, true>(state_acc[j], s.k, s.delta, r, c);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16,
                  c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            state[soff + (size_t)(v0 + c + acc_get_j(i)) * D + r + acc_get_i(i)] =
                state_acc[j][i];
        }
    }
}

// Entry points — one per instantiation (launch_bounds carries the reference's
// BLOCKS occupancy hint).
extern "C" __global__ void __launch_bounds__(256, 1) gdn_mma_v32_b1(
    const float* __restrict__ qkvx, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ out, int n_tokens, int n_v)
{
    gdn_mma_body<16, 32>(qkvx, beta, decay, state, out, n_tokens, n_v);
}
extern "C" __global__ void __launch_bounds__(256, 2) gdn_mma_v32_b2(
    const float* __restrict__ qkvx, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ out, int n_tokens, int n_v)
{
    gdn_mma_body<16, 32>(qkvx, beta, decay, state, out, n_tokens, n_v);
}
extern "C" __global__ void __launch_bounds__(256, 1) gdn_mma_v128(
    const float* __restrict__ qkvx, const float* __restrict__ beta,
    const float* __restrict__ decay, float* __restrict__ state,
    float* __restrict__ out, int n_tokens, int n_v)
{
    gdn_mma_body<16, 128>(qkvx, beta, decay, state, out, n_tokens, n_v);
}
"#;

/// The selected MMA instantiation (the reference's `gdn_path`, minus the
/// `ar` fallback — `None` plays that role).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnMmaPath {
    /// `gdn_single<16, 32, 1>` — under-filled grids, one block resident.
    Slice32OneBlock,
    /// `gdn_single<16, 32, 2>` — V-split with two blocks resident.
    Slice32TwoBlocks,
    /// `gdn_single<16, 128>` — whole-head value columns.
    WholeHead,
}

/// The cudarc-side GDN MMA kernels (Issue 1066 — the #29353 port).
pub struct CudaGdnMmaKernels {
    v32_b1: Option<CudaFunction>,
    v32_b2: Option<CudaFunction>,
    v128: Option<CudaFunction>,
    nsm: i32,
    smpbo: i32,
    _module: Arc<CudaModule>,
}

impl CudaGdnMmaKernels {
    /// Own context + stream (the `CudaGdnChunkedKernels::new_standalone`
    /// precedent) — integration tests construct without importing cudarc.
    pub fn new_standalone() -> Result<(Self, Arc<CudaStream>), String> {
        let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let kernels = Self::new(ctx)?;
        Ok((kernels, stream))
    }

    /// Compile (NVRTC, sm_89) + load; per-arm availability = the smem opt-in
    /// succeeded (the reference's `init<C, V, BLOCKS>` device-gated form).
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            GDN_MMA_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let nsm = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| format!("nsm query: {e}"))?;
        let smpbo = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| format!("smpbo query: {e}"))?;
        let opt_in = |name: &str, smem: u32| -> Result<Option<CudaFunction>, String> {
            if smem as i32 > smpbo {
                return Ok(None); // arm doesn't fit this device's opt-in ceiling
            }
            let f = module.load_function(name).map_err(|e| format!("{name}: {e}"))?;
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )
            .map_err(|e| format!("{name} smem opt-in: {e}"))?;
            Ok(Some(f))
        };
        let v32_b1 = opt_in("gdn_mma_v32_b1", GDN_MMA_SMEM_V32)?;
        let v32_b2 = opt_in("gdn_mma_v32_b2", GDN_MMA_SMEM_V32)?;
        let v128 = opt_in("gdn_mma_v128", GDN_MMA_SMEM_V128)?;
        if v32_b1.is_none() && v32_b2.is_none() && v128.is_none() {
            return Err("no GDN MMA arm available (smem opt-in refused on all)".into());
        }
        Ok(Self {
            v32_b1,
            v32_b2,
            v128,
            nsm,
            smpbo,
            _module: module,
        })
}

    /// The reference's `select_gdn_mma_path` ladder (single-seq form):
    /// `H ∈ {16,32,48,64}` + `n_tokens ≥ 64` + the occupancy-keyed value
    /// split. `None` → the caller falls back (v3 chunked → rowpar).
    pub fn select_path(&self, n_tokens: usize, n_head: usize) -> Option<GdnMmaPath> {
        if !matches!(n_head, 16 | 32 | 48 | 64) || n_tokens < 64 {
            return None;
        }
        let h = n_head as i32;
        if h * 2 <= self.nsm {
            // A grid that already fits in one wave benefits from a larger
            // register budget.
            if h * 4 <= self.nsm {
                if self.v32_b1.is_some() {
                    return Some(GdnMmaPath::Slice32OneBlock);
                }
            } else if self.v32_b2.is_some() {
                return Some(GdnMmaPath::Slice32TwoBlocks);
            }
            // fall through to whole-head if the slice arms are unavailable
            return self.v128.as_ref().map(|_| GdnMmaPath::WholeHead);
        }
        self.v128.as_ref().map(|_| GdnMmaPath::WholeHead)
    }

    /// Launch the whole chunked MMA recurrence for one layer (ONE kernel —
    /// no host crossings, no device scratch).
    ///
    /// # Safety
    ///
    /// Caller guarantees `qkvx` covers `p * 3 * v_dim` f32 (q at
    /// `[t, 0..v_dim)`, k at `[t, v_dim..2*v_dim)`, v at
    /// `[t, 2*v_dim..3*v_dim)`, head-major over `v_dim = n_head * 128`),
    /// `beta`/`decay` cover `p * n_head` (token-major, decay = ALPHA in
    /// (0,1]), `state` covers `n_head * 128 * 128` (in/out, layout
    /// `[h, v, d]`), `output` covers `p * v_dim`, and
    /// [`Self::select_path`] returned `Some` for `(p, n_head)`.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        qkvx: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
        state: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n_head: usize,
        p: usize,
    ) -> Result<(), String> {
        let path = self
            .select_path(p, n_head)
            .ok_or("launch called without an admitted path")?;
        let (f, grid, smem): (&CudaFunction, u32, u32) = match path {
            GdnMmaPath::Slice32OneBlock => (
                self.v32_b1.as_ref().expect("checked in select_path"),
                (128 / 32 * n_head) as u32,
                GDN_MMA_SMEM_V32,
            ),
            GdnMmaPath::Slice32TwoBlocks => (
                self.v32_b2.as_ref().expect("checked in select_path"),
                (128 / 32 * n_head) as u32,
                GDN_MMA_SMEM_V32,
            ),
            GdnMmaPath::WholeHead => (
                self.v128.as_ref().expect("checked in select_path"),
                n_head as u32,
                GDN_MMA_SMEM_V128,
            ),
        };
        let (p_i, nv_i) = (p as i32, n_head as i32);
        unsafe {
            stream
                .launch_builder(f)
                .arg(qkvx)
                .arg(beta)
                .arg(decay)
                .arg(state)
                .arg(output)
                .arg(&p_i)
                .arg(&nv_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: smem,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Device facts (disclosure for traces/tests).
    pub fn device_facts(&self) -> (i32, i32) {
        (self.nsm, self.smpbo)
    }
}
