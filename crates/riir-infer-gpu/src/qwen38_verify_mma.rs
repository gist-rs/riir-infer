//! Issue 742 T9.10 — the Q4_K/Q6_K tensor-core mma verify GEMM.
//!
//! The T9.9 `forward_verify_chunk` port proved the composition but hit the
//! p-row GEMV family's structural wall: every output row re-streams the
//! 16-row quantized-x set through L1 (the p-sweep 1559→492 GB/s from p=1→16
//! pins that family's floor above the whole-chunk budget). The fix is the
//! T1.3 `tm32sp` class ported to Q4_K/Q6_K: an `mma.sync.m8n8k16.s8` GEMM
//! where each weight read serves a full 8-feature × 8-token output tile —
//! the x-side amplification vanishes by construction (each x byte is read
//! once per feature-block instead of once per output row).
//!
//! # Bit-identity (the T9.9 crown preserved BY CONSTRUCTION)
//!
//! The reference is `qwen38_gemv_q4k_q8x_rows_strict` / `qwen38_gemv_q6k_q8x_rows`
//! (the decode-path-verbatim arms — the reason the T9.9 loop stream EQUALS
//! sequential greedy decode). Three properties make the mma twins
//! bit-identical to them:
//!
//! 1. **The integer dots are exact and order-free.** The mma's s8×s8→s32
//!    products are the same `(nibble-8)·xq` products the dp4a chain sums;
//!    integer addition is associative, so the s32 group dot is IDENTICAL
//!    regardless of the mma's internal summation order.
//! 2. **The f32 fold statements are textually identical** to the reference
//!    kernels (same parenthesization, same `acc += xs * (...)` shape — the
//!    NVRTC fmad-contraction decisions apply to the same expression trees).
//! 3. **The f32 accumulation ORDER is replicated exactly.** The reference
//!    kernels accumulate per GEMV lane: lane `l` serially over sub-blocks
//!    `l, l+32, l+64, …` (each iteration adding the PAIRED half-folds
//!    `(0.f + h0) + h1` for Q4_K), then a `__shfl_down` butterfly
//!    (offsets 16,8,4,2,1). The mma twins keep 32 per-cell "bucket"
//!    accumulators indexed `sb & 31` (Q4_K: sub-block mod 32; Q6_K: group
//!    mod 32), folded in the same ascending chunk order, and finish with
//!    the sequential butterfly tree `for span in 16..1: v[l] += v[l+span]`
//!    — which is bit-for-bit the shfl_down tree at lane 0.
//!
//! The buckets are statically indexed: the k-loop runs in unrolled chunks
//! of 64 groups (Q4_K: 32 sub-blocks = 1024 elements) / 32 groups (Q6_K:
//! 512 elements) — every production n (5120, 6144, 10240, 17408) divides
//! evenly — so the per-thread `float[32]` bucket arrays stay in REGISTERS
//! (the Issue-706 runtime-index lesson; `n` is a kernel arg, so the inner
//! index must be compile-time).
//!
//! # Geometry
//!
//! `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` — A = 8 rows × 16 k
//! (one u32/thread), B = 16 k × 8 tokens (one u32/thread), C = 8×8 (two
//! s32/thread). The k=16 tile is exactly ONE x-quant group — the mma C
//! fragment IS the per-group integer dot the fold consumes (the k=32
//! shapes would merge two groups with different `xs` scales and are
//! unusable here). Block = 128 threads = 4 warps covering 16 feature rows
//! × all p≤16 tokens; grid = `ceil(m/16)` — every weight byte is read by
//! exactly one block (the weights-are-streamed-once floor).
//!
//! # G2 verdict (measured, 2026-08-23 — the A/B artifact record)
//!
//! **The mma GEMMs are 1.3-1.7x SLOWER than the strict rows GEMVs at
//! every production shape, and the model-scale loop follows (C 2.58 →
//! 4.27, 230.5 → 137.1 tok/s @8K; both arms 0/256 + 0/512
//! bit-identical).** Five variants were measured (bare register loads
//! 0.46-0.77x; hoisted sub-block-invariant loads 0.58-0.82x — the best,
//! shipped here; cp.async depth-1 0.58-0.74x; depth-3 0.34-0.72x; and
//! ablations). Root cause (ablation-pinned): the WEIGHT side alone
//! reaches 500 GB/s (B/scalar loads disabled), but the x-side loads
//! thrash L1 — the per-SM x working set at 4 blocks (64 tokens x ~7.7 KB
//! = ~490 KB) exceeds the ~112 KB L1, so every B-word and scalar-pair
//! load is an L2 roundtrip serialized into the fold chain. The
//! exact-fold-order constraint itself is the structural cost: the 32
//! per-cell lane-buckets pin 64 f32 registers/thread (109 regs, 33%
//! occupancy), and the per-group per-cell scalar traffic cannot amortize
//! the way the GEMV's per-row-pair loop does. A tolerance-class mma
//! (4-accumulator fold, the T1.3 tm32sp shape — mma↔mma identity only)
//! would remove the register wall but forfeit chunk≡decode bit-identity
//! (the T1.8 re-specification class; owner call, not taken here).
//! Opt-in via `QWEN38_VERIFY_GEMV=mma`; the strict GEMV stays the
//! default.
//!
//! # Issue 754 T5 — the wide-P tolerance-class GEMMs (this module, 2026-08-28)
//!
//! The owner call above WAS taken (T4, 2026-08-27): flip-rate tolerance
//! ACCEPTED scoped to the INGEST path only — decode/verify keep the strict
//! bit-identical arms. The T5 kernels (`qwen38_t5_gemm_{q4k,q6k}_tw1_p64`)
//! are the [Bench 772](../.benchmarks/772_issue754_t5_gemm_ingest_ab.md) B1
//! winner ported VERBATIM: smem-staged x tiles (the wall-2 fix),
//! 8-register tolerance-class accumulators (wall 1 never paid),
//! `mma.sync.m8n8k16.s8` per x-quant group. Measured at the real 449-call
//! inventory: **1.47× vs the p=16 strict-GEMV inventory at P=64** (weight
//! stream 4×14.96 GB → 1×, 82 → 120 GB/s); loses at P=32 — the P4
//! integration runs p=64 chunks only. Kernel-level P0 gate: normalized
//! max error <= 1e-4 (measured 7.2e-7..1.25e-6, 80-140× margin) +
//! run-twice determinism.
//!
//! Rides `ternary_gemv_cuda_raw` (dep:cudarc; CUDA-only, non-macOS).
#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use cudarc::driver::{LaunchConfig, PushKernelArg};

const QWEN38_VERIFY_MMA_CUDA_SRC: &str = r#"
// Exact IEEE f16 -> f32 conversion (NVRTC has no cuda_fp16.h in its default
// include path). Case-based so subnormals are exact (Issue 593 class).
// Verbatim the qwen38_dense_cudarc.rs helper.
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

// One m8n8k16 s8 mma: D += A @ B (accumulating).
__device__ __forceinline__ void mma_s8_m8n8k16(
    int& d0, int& d1,
    unsigned int a0, unsigned int b0)
{
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(b0));
}

// ---------------------------------------------------------------------------
// m8n8k16 fragment-layout probe — one warp, linear row-major A (8x16 s8)
// and col-major B (16x8 s8), writes D (8x8 i32). The unit test compares
// against a CPU reference to pin the PTX fragment mapping (the
// mma_layout_probe_m16n8k32 pattern).
// ---------------------------------------------------------------------------
extern "C" __global__ void mma_layout_probe_m8n8k16(
    const signed char* __restrict__ a_lin, // [8*16] row-major
    const signed char* __restrict__ b_lin, // [16*8] col-major (b[k*8+c])
    int* __restrict__ d_out)               // [8*8] row-major
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    unsigned int a0 = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        a0 |= ((unsigned int)(unsigned char)a_lin[(g) * 16 + t * 4 + j]) << (8 * j);
    unsigned int b0 = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        b0 |= ((unsigned int)(unsigned char)b_lin[(t * 4 + j) * 8 + g]) << (8 * j);
    int d0 = 0, d1 = 0;
    mma_s8_m8n8k16(d0, d1, a0, b0);
    d_out[(g) * 8 + t * 2    ] = d0;
    d_out[(g) * 8 + t * 2 + 1] = d1;
}

// The shfl_down butterfly tree (offsets 16,8,4,2,1) fully manual — the
// loop form is NOT unrolled by nvcc (measured: dynamic bk[] indexing ->
// the whole bucket array demotes to local memory, the Issue-706 class).
#define MMA_BFLY(bk)     bk[0]+=bk[16]; bk[1]+=bk[17]; bk[2]+=bk[18]; bk[3]+=bk[19]; bk[4]+=bk[20]; bk[5]+=bk[21]; bk[6]+=bk[22]; bk[7]+=bk[23];     bk[8]+=bk[24]; bk[9]+=bk[25]; bk[10]+=bk[26]; bk[11]+=bk[27]; bk[12]+=bk[28]; bk[13]+=bk[29]; bk[14]+=bk[30]; bk[15]+=bk[31];     bk[0]+=bk[8]; bk[1]+=bk[9]; bk[2]+=bk[10]; bk[3]+=bk[11]; bk[4]+=bk[12]; bk[5]+=bk[13]; bk[6]+=bk[14]; bk[7]+=bk[15];     bk[0]+=bk[4]; bk[1]+=bk[5]; bk[2]+=bk[6]; bk[3]+=bk[7];     bk[0]+=bk[2]; bk[1]+=bk[3];     bk[0]+=bk[1];

// ---------------------------------------------------------------------------
// The Q4_K p-row mma GEMM. Bit-identical per (token, row) output element to
// qwen38_gemv_q4k_q8x_rows_strict (see the module doc: exact integer dots +
// textually-identical fold statements + the replicated lane-strided
// serial-pair + butterfly accumulation order).
//
// Block 128 threads = 4 warps: warp w covers rows [16*bx + 8*(w&1), +8) x
// tokens [8*(w>>1), +8). grid.x = ceil(m/16).
// ---------------------------------------------------------------------------
extern "C" __global__ void qwen38_gemv_q4k_q8x_rows_mma(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16]
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g_id = lane >> 2;   // A row (of the warp's 8) == C row
    const int t_id = lane & 3;
    const int row_base = (blockIdx.x << 4) + ((warp & 1) << 3);
    const int tok_base = (warp >> 1) << 3;
    const int row = (row_base + g_id) < m ? (row_base + g_id) : (m - 1);
    const int groups = n >> 4;
    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 144;

    // The C-cell tokens this lane folds (C: row g_id, cols 2*t_id + {0,1}).
    const int tokA = tok_base + (t_id << 1);
    const int tokB = tokA + 1;
    const int tokA_c = tokA < p ? tokA : (p - 1);
    const int tokB_c = tokB < p ? tokB : (p - 1);

    float bk0[32];
    float bk1[32];
    #pragma unroll
    for (int l = 0; l < 32; ++l) { bk0[l] = 0.f; bk1[l] = 0.f; }
    float pend0 = 0.f;
    float pend1 = 0.f;

    // Chunk = 64 groups = 32 sub-blocks = 1024 elements (bucket-aligned).
    // #pragma unroll N (not the bare hint): the bucket index must be
    // COMPILE-TIME or the arrays demote to local memory (Issue-706 class —
    // the bare hint over 64 heavy iterations measured 256 B local spills).
    for (int cbase = 0; cbase < groups; cbase += 64) {
        #pragma unroll 8
        for (int o8 = 0; o8 < 8; ++o8) {
        #pragma unroll 8
        for (int i8 = 0; i8 < 8; ++i8) {
            const int gg = o8 * 8 + i8;
            const int g = cbase + gg;
            const int sb = g >> 1;
            const int j = sb & 7;
            const int blk = sb >> 3;
            const unsigned int nib_shift = (j & 1) ? 4 : 0;
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
            // A fragment: this row's (nibble - 8) s8 quad at the half's
            // element offset 4*t_id — the strict kernel's word unpack.
            const unsigned char* q = b + 16 + ((j >> 1) << 5) + ((g & 1) << 4);
            const unsigned int aw = __vsub4(
                (*(const unsigned int*)(q + (t_id << 2)) >> nib_shift) & 0x0F0F0F0Fu,
                0x08080808u);
            // B fragment: token (tok_base + g_id)'s x quad at this group.
            const int tokL = tok_base + g_id;
            const int tokL_c = tokL < p ? tokL : (p - 1);
            const unsigned int bw =
                *(const unsigned int*)(xq + (size_t)tokL_c * n + (g << 4) + (t_id << 2));
            int d0 = 0, d1 = 0;
            mma_s8_m8n8k16(d0, d1, aw, bw);
            // The cells' xs/isum (garbage for dead tokens; never written).
            const float xsA = xs[(size_t)tokA_c * groups + g];
            const float xsB = xs[(size_t)tokB_c * groups + g];
            const int isA = xsum[(size_t)tokA_c * groups + g];
            const int isB = xsum[(size_t)tokB_c * groups + g];
            if ((gg & 1) == 0) {
                pend0 = xsA * ((d * sc) * (float)(d0 + 8 * isA) - (dmin * mn) * (float)isA);
                pend1 = xsB * ((d * sc) * (float)(d1 + 8 * isB) - (dmin * mn) * (float)isB);
            } else {
                // The strict kernel's pair accumulation:
                // sub = 0.f; sub += h0; sub += h1; acc += sub.
                float sub0 = 0.f;
                float sub1 = 0.f;
                sub0 += pend0;
                sub1 += pend1;
                sub0 += xsA * ((d * sc) * (float)(d0 + 8 * isA) - (dmin * mn) * (float)isA);
                sub1 += xsB * ((d * sc) * (float)(d1 + 8 * isB) - (dmin * mn) * (float)isB);
                bk0[gg >> 1] += sub0;
                bk1[gg >> 1] += sub1;
            }
        }
        }
    }
    // The butterfly: the shfl_down tree (offsets 16,8,4,2,1) at lane 0,
    // replicated as the sequential span loop (bit-for-bit the same order).
    MMA_BFLY(bk0);
    MMA_BFLY(bk1);
    if ((row_base + g_id) < m) {
        if (tokA < p) y[(size_t)tokA * m + row_base + g_id] = bk0[0];
        if (tokB < p) y[(size_t)tokB * m + row_base + g_id] = bk1[0];
    }
}

// ---------------------------------------------------------------------------
// The Q6_K p-row mma GEMM — bit-identical per element to
// qwen38_gemv_q6k_q8x_rows. Same geometry; the Q6_K group IS the x-quant
// group (16 elements, one sc/d per group) so the fold is per group with no
// pairing; the bucket index is (g & 31) and the chunk is 32 groups (512
// elements).
// ---------------------------------------------------------------------------
extern "C" __global__ void qwen38_gemv_q6k_q8x_rows_mma(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16]
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g_id = lane >> 2;
    const int t_id = lane & 3;
    const int row_base = (blockIdx.x << 4) + ((warp & 1) << 3);
    const int tok_base = (warp >> 1) << 3;
    const int row = (row_base + g_id) < m ? (row_base + g_id) : (m - 1);
    const int groups = n >> 4;
    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 210;

    const int tokA = tok_base + (t_id << 1);
    const int tokB = tokA + 1;
    const int tokA_c = tokA < p ? tokA : (p - 1);
    const int tokB_c = tokB < p ? tokB : (p - 1);

    float bk0[32];
    float bk1[32];
    #pragma unroll
    for (int l = 0; l < 32; ++l) { bk0[l] = 0.f; bk1[l] = 0.f; }

    // Chunk = 32 groups = 512 elements (bucket = g & 31). #pragma unroll N
    // for compile-time bucket indices (the Issue-706 local-memory lesson).
    for (int cbase = 0; cbase < groups; cbase += 32) {
        #pragma unroll 4
        for (int o8 = 0; o8 < 4; ++o8) {
        #pragma unroll 8
        for (int i8 = 0; i8 < 8; ++i8) {
            const int gg = o8 * 8 + i8;
            const int g = cbase + gg;
            const int blk = g >> 4;
            const int h = (g >> 3) & 1;
            const int slot = (g >> 1) & 3;
            const int l0 = (g & 1) << 4;
            const unsigned char* b = wrow + (size_t)blk * 210;
            const unsigned char* qlp = b + ((h << 6) + ((slot & 1) << 5));
            const unsigned char* qhp = b + 128 + (h << 5);
            const float sc = (float)(signed char)b[192 + (h << 3) + (g & 1) + (slot << 1)];
            const float d = f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 208));
            const unsigned int nib_shift = (slot < 2) ? 0 : 4;
            const unsigned int qh_shift = (unsigned int)(slot << 1);
            // A fragment: this row's 6-bit quad at the group's element
            // offset 4*t_id — the rows kernel's Q6_U32 assembly.
#define Q6_U32_MMA(p_) (((unsigned int)*(const unsigned short*)(p_)) | \
                        ((unsigned int)*(const unsigned short*)((p_) + 2) << 16))
            const unsigned int qlw = Q6_U32_MMA(qlp + l0 + (t_id << 2));
            const unsigned int qhw = Q6_U32_MMA(qhp + l0 + (t_id << 2));
#undef Q6_U32_MMA
            const unsigned int as_ =
                ((qlw >> nib_shift) & 0x0F0F0F0Fu) | (((qhw >> qh_shift) & 0x03030303u) << 4);
            const unsigned int aw = __vsub4(as_, 0x20202020u);
            const int tokL = tok_base + g_id;
            const int tokL_c = tokL < p ? tokL : (p - 1);
            const unsigned int bw =
                *(const unsigned int*)(xq + (size_t)tokL_c * n + (g << 4) + (t_id << 2));
            int d0 = 0, d1 = 0;
            mma_s8_m8n8k16(d0, d1, aw, bw);
            const float xsA = xs[(size_t)tokA_c * groups + g];
            const float xsB = xs[(size_t)tokB_c * groups + g];
            // The rows kernel's per-group fold (no pairing, no isum term).
            bk0[gg] += xsA * ((d * sc) * (float)d0);
            bk1[gg] += xsB * ((d * sc) * (float)d1);
        }
        }
    }
    MMA_BFLY(bk0);
    MMA_BFLY(bk1);
    if ((row_base + g_id) < m) {
        if (tokA < p) y[(size_t)tokA * m + row_base + g_id] = bk0[0];
        if (tokB < p) y[(size_t)tokB * m + row_base + g_id] = bk1[0];
    }
}

// ---------------------------------------------------------------------------
// Issue 754 T5 — the wide-P tolerance-class GEMM (Bench 772 B1 winner, tw1
// @ P=64, ported VERBATIM from the bench harness templates). Block = 256
// threads = 8 warps as 4 row-warps x 2 token-warps (M_TILE=32). The x rows
// are staged in SHARED memory per 128-element k-tile (sxq[P][144] with the
// zero-bank-conflict 36-word stride) — the wall-2 fix: one global read per
// x byte per BLOCK, the 8 warps then read smem. Per-lane f32 accumulators
// acc[TOK_TILES=4][2] = 8 regs — the wall-1 register pressure of the exact
// fold is NOT paid. Tolerance class vs the strict GEMV (fold-order
// reassociation; the P0 budget is norm <= 1e-4, measured 7.2e-7..1.25e-6
// with 80-140x margin). Writes guarded by `tok < p` — a partial tail chunk
// runs the P=64 instantiation over zero-filled rows (buffers MUST be sized
// [64, n]; dead columns accumulate 0 from zeroed xs and are never written).
// ---------------------------------------------------------------------------
extern "C" __global__ void qwen38_t5_gemm_q4k_tw1_p64(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16]
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int MTILE = 32;
    const int TOK_TILES = 4;   // __P__ >> 4 at P=64
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g_id = lane >> 2;
    const int t_id = lane & 3;
    const int row_base = blockIdx.x * MTILE + ((warp & 3) << 3);
    const int row = (row_base + g_id) < m ? (row_base + g_id) : (m - 1);
    const int warp_tok_base = (warp >> 2) * 32;   // (__P__ >> 1) at P=64
    const int groups = n >> 4;

    __shared__ __align__(16) unsigned char sxq[64][144];
    __shared__ float sxs[64][9];
    __shared__ int sxsum[64][9];

    float acc[TOK_TILES][2];
#pragma unroll
    for (int tt = 0; tt < TOK_TILES; ++tt) { acc[tt][0] = 0.f; acc[tt][1] = 0.f; }

    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 144;

    for (int ktile = 0; ktile < n; ktile += 128) {
        {
            const int words = 64 * 32;
            for (int iw = threadIdx.x; iw < words; iw += (int)blockDim.x) {
                const int tok = iw >> 5;
                const int w32 = iw & 31;
                *reinterpret_cast<unsigned int*>(&sxq[tok][w32 << 2]) =
                    *reinterpret_cast<const unsigned int*>(
                        xq + (size_t)tok * n + ktile + (w32 << 2));
            }
            const int tile_g0 = ktile >> 4;
            for (int isc = threadIdx.x; isc < 64 * 8; isc += (int)blockDim.x) {
                const int tok = isc >> 3;
                const int j8 = isc & 7;
                sxs[tok][j8] = xs[(size_t)tok * groups + tile_g0 + j8];
                sxsum[tok][j8] = xsum[(size_t)tok * groups + tile_g0 + j8];
            }
        }
        __syncthreads();
#pragma unroll
        for (int gl = 0; gl < 8; ++gl) {
            const int g = (ktile >> 4) + gl;
            const int sb = g >> 1;
            const int j = sb & 7;
            const int blk = sb >> 3;
            const unsigned char* b = wrow + (size_t)blk * 144;
            const float d =
                f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b));
            const float dmin =
                f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 2));
            const unsigned char* s = b + 4;
            float sc, mn;
            if (j < 4) {
                sc = (float)(s[j] & 63);
                mn = (float)(s[j + 4] & 63);
            } else {
                sc = (float)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
                mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
            }
            const unsigned char* q = b + 16 + ((j >> 1) << 5) + ((g & 1) << 4);
            const unsigned int nib_shift = (j & 1) ? 4 : 0;
            const unsigned int aw = __vsub4(
                (__ldcs(reinterpret_cast<const unsigned int*>(q + (t_id << 2)))
                    >> nib_shift) & 0x0F0F0F0Fu,
                0x08080808u);
#pragma unroll
            for (int tt = 0; tt < TOK_TILES; ++tt) {
                const int tb = warp_tok_base + (tt << 3);
                const unsigned int bw = *reinterpret_cast<const unsigned int*>(
                    &sxq[tb + g_id][(gl << 4) + (t_id << 2)]);
                int d0 = 0, d1 = 0;
                mma_s8_m8n8k16(d0, d1, aw, bw);
                const int tokA = tb + (t_id << 1);
                const int tokB = tokA + 1;
                const float xsA = sxs[tokA][gl];
                const float xsB = sxs[tokB][gl];
                const int isA = sxsum[tokA][gl];
                const int isB = sxsum[tokB][gl];
                acc[tt][0] += xsA * ((d * sc) * (float)(d0 + 8 * isA)
                                     - (dmin * mn) * (float)isA);
                acc[tt][1] += xsB * ((d * sc) * (float)(d1 + 8 * isB)
                                     - (dmin * mn) * (float)isB);
            }
        }
        __syncthreads();
    }
    if ((row_base + g_id) < m) {
#pragma unroll
        for (int tt = 0; tt < TOK_TILES; ++tt) {
            const int tokA = warp_tok_base + (tt << 3) + (t_id << 1);
            const int tokB = tokA + 1;
            if (tokA < p) y[(size_t)tokA * m + row_base + g_id] = acc[tt][0];
            if (tokB < p) y[(size_t)tokB * m + row_base + g_id] = acc[tt][1];
        }
    }
}

// The Q6_K twin (same geometry; no xsum/dmin terms — the Q6_K group IS the
// x-quant group). Ported VERBATIM from the bench_772 Q6 template at tw1/64.
extern "C" __global__ void qwen38_t5_gemm_q6k_tw1_p64(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,   // [p, n]
    const float* __restrict__ xs,         // [p, n/16]
    const int* __restrict__ xsum,         // [p, n/16] (unused — uniform signature)
    float* __restrict__ y,                // [p, m]
    const int m, const int n, const int blocks_per_row, const int p)
{
    const int MTILE = 32;
    const int TOK_TILES = 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g_id = lane >> 2;
    const int t_id = lane & 3;
    const int row_base = blockIdx.x * MTILE + ((warp & 3) << 3);
    const int row = (row_base + g_id) < m ? (row_base + g_id) : (m - 1);
    const int warp_tok_base = (warp >> 2) * 32;
    const int groups = n >> 4;

    __shared__ __align__(16) unsigned char sxq[64][144];
    __shared__ float sxs[64][9];

    float acc[TOK_TILES][2];
#pragma unroll
    for (int tt = 0; tt < TOK_TILES; ++tt) { acc[tt][0] = 0.f; acc[tt][1] = 0.f; }

    const unsigned char* wrow = w + (size_t)row * (size_t)blocks_per_row * 210;

    for (int ktile = 0; ktile < n; ktile += 128) {
        {
            const int words = 64 * 32;
            for (int iw = threadIdx.x; iw < words; iw += (int)blockDim.x) {
                const int tok = iw >> 5;
                const int w32 = iw & 31;
                *reinterpret_cast<unsigned int*>(&sxq[tok][w32 << 2]) =
                    *reinterpret_cast<const unsigned int*>(
                        xq + (size_t)tok * n + ktile + (w32 << 2));
            }
            const int tile_g0 = ktile >> 4;
            for (int isc = threadIdx.x; isc < 64 * 8; isc += (int)blockDim.x) {
                const int tok = isc >> 3;
                const int j8 = isc & 7;
                sxs[tok][j8] = xs[(size_t)tok * groups + tile_g0 + j8];
            }
        }
        __syncthreads();
#pragma unroll
        for (int gl = 0; gl < 8; ++gl) {
            const int g = (ktile >> 4) + gl;
            const int blk = g >> 4;
            const int h = (g >> 3) & 1;
            const int slot = (g >> 1) & 3;
            const int l0 = (g & 1) << 4;
            const unsigned char* b = wrow + (size_t)blk * 210;
            const unsigned char* qlp = b + ((h << 6) + ((slot & 1) << 5));
            const unsigned char* qhp = b + 128 + (h << 5);
            const float sc =
                (float)(signed char)b[192 + (h << 3) + (g & 1) + (slot << 1)];
            const float d =
                f16_to_f32_exact(*reinterpret_cast<const unsigned short*>(b + 208));
            const unsigned int nib_shift = (slot < 2) ? 0 : 4;
            const unsigned int qh_shift = (unsigned int)(slot << 1);
#define Q6_U32_T5(p_) (((unsigned int)__ldcs((const unsigned short*)(p_))) | \
                       ((unsigned int)__ldcs((const unsigned short*)((p_) + 2)) << 16))
            const unsigned int qlw = Q6_U32_T5(qlp + l0 + (t_id << 2));
            const unsigned int qhw = Q6_U32_T5(qhp + l0 + (t_id << 2));
#undef Q6_U32_T5
            const unsigned int as_ =
                ((qlw >> nib_shift) & 0x0F0F0F0Fu)
                | (((qhw >> qh_shift) & 0x03030303u) << 4);
            const unsigned int aw = __vsub4(as_, 0x20202020u);
#pragma unroll
            for (int tt = 0; tt < TOK_TILES; ++tt) {
                const int tb = warp_tok_base + (tt << 3);
                const unsigned int bw = *reinterpret_cast<const unsigned int*>(
                    &sxq[tb + g_id][(gl << 4) + (t_id << 2)]);
                int d0 = 0, d1 = 0;
                mma_s8_m8n8k16(d0, d1, aw, bw);
                const int tokA = tb + (t_id << 1);
                const int tokB = tokA + 1;
                const float xsA = sxs[tokA][gl];
                const float xsB = sxs[tokB][gl];
                acc[tt][0] += xsA * ((d * sc) * (float)d0);
                acc[tt][1] += xsB * ((d * sc) * (float)d1);
            }
        }
        __syncthreads();
    }
    if ((row_base + g_id) < m) {
#pragma unroll
        for (int tt = 0; tt < TOK_TILES; ++tt) {
            const int tokA = warp_tok_base + (tt << 3) + (t_id << 1);
            const int tokB = tokA + 1;
            if (tokA < p) y[(size_t)tokA * m + row_base + g_id] = acc[tt][0];
            if (tokB < p) y[(size_t)tokB * m + row_base + g_id] = acc[tt][1];
        }
    }
}
"#;

/// The m8n8k16-s8 mma verify-GEMM kernel set (Issue 742 T9.10).
pub struct VerifyMmaKernels {
    q4k_mma: CudaFunction,
    q6k_mma: CudaFunction,
    layout_probe: CudaFunction,
    /// Issue 754 T5 — the wide-P tolerance-class GEMMs (Bench 772 B1
    /// winner: tw1 @ P=64). Tolerance class vs the strict GEMV (P0 budget
    /// norm <= 1e-4, measured 80-140x margin); the INGEST path's arm
    /// (decode/verify keep the strict GEMV — the T4 scope rule).
    t5_q4k_p64: CudaFunction,
    t5_q6k_p64: CudaFunction,
    _module: Arc<CudaModule>,
}

impl VerifyMmaKernels {
    /// Register/local-memory diagnostics (the Issue-706 class check:
    /// local_size_bytes > 0 means the bucket arrays spilled).
    pub fn regs_debug(&self) -> Result<[(i32, i32); 2], String> {
        let q4 = self
            .q4k_mma
            .num_regs()
            .map(|r| (r, self.q4k_mma.local_size_bytes().unwrap_or(-1)))
            .map_err(|e| e.to_string())?;
        let q6 = self
            .q6k_mma
            .num_regs()
            .map(|r| (r, self.q6k_mma.local_size_bytes().unwrap_or(-1)))
            .map_err(|e| e.to_string())?;
        Ok([q4, q6])
    }

    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            QWEN38_VERIFY_MMA_CUDA_SRC,
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
            q4k_mma: f("qwen38_gemv_q4k_q8x_rows_mma")?,
            q6k_mma: f("qwen38_gemv_q6k_q8x_rows_mma")?,
            layout_probe: f("mma_layout_probe_m8n8k16")?,
            t5_q4k_p64: f("qwen38_t5_gemm_q4k_tw1_p64")?,
            t5_q6k_p64: f("qwen38_t5_gemm_q6k_tw1_p64")?,
            _module: module,
        })
    }

    /// The p-row batched Q4_K mma GEMM — bit-identical per element to
    /// `qwen38_gemv_q4k_q8x_rows_strict` (see the kernel doc).
    ///
    /// # Safety
    ///
    /// Caller guarantees `w` covers `m * blocks_per_row * 144` bytes,
    /// the quantized buffers cover `[p, n]` / `[p, n/16]`, `y` covers
    /// `[p, m]`, `p <= 16`, `n % 1024 == 0` (chunk alignment), and
    /// `blocks_per_row == n / 256`.
    pub unsafe fn launch_gemv_q4k_rows_mma(
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
        assert!(p <= 16 && n.is_multiple_of(1024), "q4k mma: p<=16, n%1024==0");
        let (m_i, n_i, bpr_i, p_i) = (m as i32, n as i32, blocks_per_row as i32, p as i32);
        let grid = m.div_ceil(16).max(1) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.q4k_mma)
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
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The p-row batched Q6_K mma GEMM — bit-identical per element to
    /// `qwen38_gemv_q6k_q8x_rows` (see the kernel doc).
    ///
    /// # Safety
    ///
    /// Caller guarantees `w` covers `m * blocks_per_row * 210` bytes,
    /// the quantized buffers cover `[p, n]` / `[p, n/16]`, `y` covers
    /// `[p, m]`, `p <= 16`, `n % 512 == 0` (chunk alignment), and
    /// `blocks_per_row == n / 256`.
    pub unsafe fn launch_gemv_q6k_rows_mma(
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
        assert!(p <= 16 && n.is_multiple_of(512), "q6k mma: p<=16, n%512==0");
        let (m_i, n_i, bpr_i, p_i) = (m as i32, n as i32, blocks_per_row as i32, p as i32);
        let grid = m.div_ceil(16).max(1) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.q6k_mma)
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
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 754 T5 — the wide-P Q4_K GEMM (tw1 @ P=64, the Bench 772 B1
    /// winner). **Tolerance class** vs the strict GEMV (fold-order
    /// reassociation; the P0 budget is normalized max error <= 1e-4,
    /// measured 7.2e-7..1.25e-6). INGEST-path arm only (the T4 scope rule —
    /// decode/verify keep the strict bit-identical GEMVs).
    ///
    /// # Safety
    ///
    /// Caller guarantees `w` covers `m * blocks_per_row * 144` bytes, the
    /// quantized buffers cover `[64, n]` / `[64, n/16]` (rows `>= p` valid
    /// — the kernel stages ALL 64 rows from global; a partial tail chunk
    /// requires zero-filled dead rows), `y` covers `[p, m]`, `p <= 64`,
    /// `n % 128 == 0` (the k-tile step), and `blocks_per_row == n / 256`.
    pub unsafe fn launch_t5_gemm_q4k_p64(
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
        assert!(p <= 64 && n.is_multiple_of(128), "t5 q4k p64: p<=64, n%128==0");
        let (m_i, n_i, bpr_i, p_i) = (m as i32, n as i32, blocks_per_row as i32, p as i32);
        let grid = m.div_ceil(32).max(1) as u32;   // M_TILE=32 (tw1)
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.t5_q4k_p64)
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

    /// Issue 754 T5 — the wide-P Q6_K GEMM twin (tw1 @ P=64). Same
    /// tolerance-class contract as [`Self::launch_t5_gemm_q4k_p64`].
    ///
    /// # Safety
    ///
    /// Caller guarantees `w` covers `m * blocks_per_row * 210` bytes, the
    /// quantized buffers cover `[64, n]` / `[64, n/16]`, `y` covers
    /// `[p, m]`, `p <= 64`, `n % 128 == 0`, and
    /// `blocks_per_row == n / 256`.
    pub unsafe fn launch_t5_gemm_q6k_p64(
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
        assert!(p <= 64 && n.is_multiple_of(128), "t5 q6k p64: p<=64, n%128==0");
        let (m_i, n_i, bpr_i, p_i) = (m as i32, n as i32, blocks_per_row as i32, p as i32);
        let grid = m.div_ceil(32).max(1) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.t5_q6k_p64)
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

    /// The m8n8k16 fragment-layout probe (test-only; the CPU reference in
    /// `qwen38_verify_rows_g1` pins the A/B/C mapping).
    ///
    /// # Safety
    ///
    /// Caller guarantees `a_lin` covers 128 bytes, `b_lin` 128,
    /// `d_out` 64 i32; one warp block.
    pub unsafe fn launch_layout_probe(
        &self,
        stream: &CudaStream,
        a_lin: &CudaSlice<i8>,
        b_lin: &CudaSlice<i8>,
        d_out: &mut CudaSlice<i32>,
    ) -> Result<(), String> {
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.layout_probe)
                .arg(a_lin)
                .arg(b_lin)
                .arg(d_out)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
