//! Issue 734 Arm 6 (Bench 719) — raw-CUDA int8 tensor-core ternary GEMM
//! (`mma.sync.m16n8k32.s8`), the structural answer to the ~33-40 TFLOPS
//! cubecl-spirv staging wall.
//!
//! ## Why this kernel exists
//!
//! Every shrink axis INSIDE the CubeCL i8 kernel is measured closed
//! (Bench 715 ALU, 711 B-layout, 712 A-traffic, 716 pipelining, 717
//! store/load counts, 718 occupancy +4%): the wall is the substrate's
//! from_slice→cmma::store→smem-reduction structure. The same substrate's
//! raw load+mma probe hit **764.8 TOPS** (Bench 709) — the gap is
//! structure, not silicon. Raw CUDA can build the structure cubecl
//! cannot express on SPIR-V:
//!
//! - **Register accumulators + per-group fold** — the smem
//!   partial-store/reduce round trip (18% of the wall-probe budget) is
//!   gone entirely; hi/lo i32 partials live in mma D registers.
//! - **B straight from global into fragment registers** — the PTX
//!   m16n8k32 B-fragment layout is "4 consecutive k of one token" =
//!   EXACTLY one packed `q_hi`/`q_lo` u32 word. No B staging, no B smem.
//! - **A staged once per block per 128-k group** (bitplane→i8 unpack,
//!   2 barriers per GROUP, not per k-step) — each row's unpack work is
//!   shared by the 4 N-direction warps instead of duplicated 4×.
//!
//! ## Numerics contract (bit-identity by construction)
//!
//! Replicates the shipping `GemmTernaryCmmaI8CubeCL` arithmetic exactly:
//! per-token abs-max hi/lo quantization (`x ≈ s·(qh + ql/128)`), exact
//! i32 integer products (order-free), and the per-group f32 fold
//! `o += sw·((f32)hi + (f32)lo·(1/128))` in strictly ascending group
//! order, epilogue `× s_t[tok]`. Measured cross-backend facts (Bench 719,
//! element-level dumps on this box):
//!
//! - **The outer fold is FMA-contracted by the Vulkan driver** (naga emits
//!   no `NoContraction`): `FoldMode::Fused` (`__fmaf_rn`) is bit-identical
//!   to shipping; `Strict` differs on 1248/6240 small-fixture slots.
//! - **The NVIDIA SPIR-V consumer's OpFDiv is NOT div.rn** — it matches PTX
//!   `div.full.f32` / `div.approx.f32` bit-for-bit on every tested input
//!   (4160-word sweep: Full 0, Approx 0, rn 62 diffs vs shipping), while
//!   IEEE-correctly-rounded division differs. Quotients landing within
//!   ~1 ulp of a half-integer flip `rint` and shift `ql` by ±1 — the
//!   `QuantDiv` enum selects the form; `Full` is the default choice.
//! - `cvt.rni.f32.f32` (ties-to-even) matches WGSL `round`; plain C casts
//!   are trunc, matching `i32::cast_from`.
//!
//! **Weight invariant:** the unpack computes `byte = pos_bit − neg_bit`
//! per byte (general — overlapping bitplanes yield 0, matching the
//! shipping kernel's arithmetic even on pathological inputs; pinned by
//! the non-disjoint fixture gate).
//!
//! ## Measured (Bench 719, 4090, ffn_gate m=17408 n=5120)
//!
//! | arm | p=2048 | p=4096 |
//! |---|---|---|
//! | psplit (cubecl, shipping) | 42.2 TF-equiv · 8.64 ms | 43.7 · 16.72 |
//! | mma tm128 (raw cuda) | 84.4 · 4.32 ms (1.998×) | 82.1 · 8.89 (1.881×) |
//! | mma tm64 (raw cuda) | **105.1 · 3.47 ms (2.487×)** | **107.1 · 6.82 (2.453×)** |
//!
//! **Kill gate (≥2×) PASSED** — tm64 is the production candidate. G1:
//! 0/35,651,584 bit-diffs vs shipping at p=2048 on BOTH tiles; 0/6240 on
//! the ragged small fixture (m=96, n=256, p=65), the degenerate-quantize
//! fixture, and the non-disjoint-bitplane fixture.
//!
//! ## Status
//!
//! POC behind `ternary_gemv_cuda_raw` — NOT wired into any dispatch. The
//! e2e wiring (weights on the cudarc stack, `prefill_project` arm, the
//! Bench-710 FNV full-model gates) is the follow-up unit; this POC's
//! 2.45× kernel win plus bit-identity at production shapes arms it.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

/// Padded A-stage row stride (bytes). 132 keeps the 128 payload bytes
/// 4-byte aligned while breaking the every-row-same-bank pattern a 128
/// stride would create (row r's quad q maps to bank (r·32 + q) mod 32 = q
/// for stride 128 — an 8-way conflict on every fragment load).
const ASM_ROW_STRIDE: usize = 132;

pub(crate) const GEMM_MMA_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// round-to-nearest-EVEN (WGSL `round` lowers to RoundEven; cubecl-cpp uses
// rint). Plain roundf() is ties-AWAY — wrong.
__device__ __forceinline__ float rne_f32(float x)
{
    float r;
    asm("cvt.rni.f32.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// One m16n8k32 s8 mma: D += A @ B (accumulating).
__device__ __forceinline__ void mma_s8_m16n8k32(
    int& d0, int& d1, int& d2, int& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Unpack one u32 of 4 int8 weight signs from a pos/neg bitplane word pair.
// byte j = bit(pw, k0+j) - bit(nw, k0+j)  (general: overlap -> 0, matching
// the shipping kernel's pos_bit - neg_bit arithmetic).
__device__ __forceinline__ unsigned int sign_word(unsigned int pw, unsigned int nw, int k0)
{
    unsigned int r = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int b = (int)((pw >> (k0 + j)) & 1u) - (int)((nw >> (k0 + j)) & 1u);
        r |= ((unsigned int)b & 0xFFu) << (8 * j);
    }
    return r;
}

// Division variants for the quantize — the NVIDIA SPIR-V consumer's
// OpFDiv lowering is NOT div.rn (measured: quotients 1 ulp off correctly-
// rounded flip rint near half-integer boundaries; the Bench-719 element-
// level dump). div.full / div.approx replicate the fast forms.
__device__ __forceinline__ float div_rn(float a, float b)
{
    return a / b;
}
__device__ __forceinline__ float div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float div_approx(float a, float b)
{
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#define QBODY(NAME, DIV)                                                       \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ input,   /* [p * n] */                            \
    unsigned int* __restrict__ q_hi_w, /* [p * (n/4)] */                         \
    unsigned int* __restrict__ q_lo_w, /* [p * (n/4)] */                         \
    float* __restrict__ s_out,         /* [p] */                                \
    int n)                                                                      \
{                                                                              \
    __shared__ float red[128];                                                 \
    const int row = blockIdx.x;                                                \
    const int tid = threadIdx.x;                                               \
    const long base = (long)row * n;                                           \
                                                                               \
    /* Pass 1: row max |x| (max is order-free — deterministic). */             \
    float mx = 0.0f;                                                           \
    for (int c = tid; c < n; c += 128) {                                       \
        float v = input[base + c];                                             \
        float a = v < 0.0f ? -v : v;                                           \
        mx = a > mx ? a : mx;                                                  \
    }                                                                          \
    red[tid] = mx;                                                             \
    __syncthreads();                                                           \
    float m = 0.0f;                                                            \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 128; ++i) {                                            \
        m = red[i] > m ? red[i] : m;                                           \
    }                                                                          \
    const float s = m > 0.0f ? DIV(m, 127.0f) : 1.0f;                          \
    if (tid == 0) s_out[row] = s;                                              \
                                                                               \
    /* Pass 2: packed words, 4 consecutive k per word, byte j at shift j*8. */ \
    const int words = n / 4;                                                   \
    for (int w = tid; w < words; w += 128) {                                   \
        unsigned int wh = 0u, wl = 0u;                                         \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 4; ++j) {                                          \
            float x = input[base + (long)w * 4 + j];                           \
            float xf = DIV(x, s);                                              \
            float qh = rne_f32(xf);                                            \
            qh = qh > 127.0f ? 127.0f : (qh < -127.0f ? -127.0f : qh);         \
            float rem = xf - qh;                                               \
            float ql = rem * 128.0f;                                           \
            ql = ql > 64.0f ? 64.0f : (ql < -64.0f ? -64.0f : ql);             \
            wh |= ((unsigned int)(int)qh & 0xFFu) << (8 * j);                  \
            wl |= ((unsigned int)(int)ql & 0xFFu) << (8 * j);                  \
        }                                                                      \
        q_hi_w[(long)row * words + w] = wh;                                    \
        q_lo_w[(long)row * words + w] = wl;                                    \
    }                                                                          \
}

QBODY(quantize_rows_i8_hilo_cuda_rn, div_rn)
QBODY(quantize_rows_i8_hilo_cuda_full, div_full)
QBODY(quantize_rows_i8_hilo_cuda_approx, div_approx)

// ---------------------------------------------------------------------------
// mma fragment-layout probe — one warp, linear row-major A (16x32 s8) and
// col-major B (32x8 s8) inputs, writes D (16x8 i32). The unit test compares
// against a CPU reference to pin the PTX fragment mapping.
// ---------------------------------------------------------------------------

extern "C" __global__ void mma_layout_probe_m16n8k32(
    const signed char* __restrict__ a_lin, // [16*32] row-major
    const signed char* __restrict__ b_lin, // [32*8] col-major (b[k*8+c])
    int* __restrict__ d_out)               // [16*8] row-major
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;      // groupID
    const int t = lane & 3;       // thread-in-group

    // A fragment (row-major m16 x k32): a0 row g k t*4.., a1 row g+8,
    // a2/a3 = +16 k. Assemble 4 bytes from linear memory.
    unsigned int a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        a0 |= ((unsigned int)(unsigned char)a_lin[(g) * 32 + t * 4 + j]     ) << (8 * j);
        a1 |= ((unsigned int)(unsigned char)a_lin[(g + 8) * 32 + t * 4 + j] ) << (8 * j);
        a2 |= ((unsigned int)(unsigned char)a_lin[(g) * 32 + t * 4 + 16 + j]) << (8 * j);
        a3 |= ((unsigned int)(unsigned char)a_lin[(g + 8) * 32 + t * 4 + 16 + j]) << (8 * j);
    }
    // B fragment (col-major k32 x n8): b0 = 4 consecutive k of column g at
    // k = t*4, b1 at k = t*4 + 16.
    unsigned int b0 = 0u, b1 = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        b0 |= ((unsigned int)(unsigned char)b_lin[(t * 4 + j) * 8 + g]     ) << (8 * j);
        b1 |= ((unsigned int)(unsigned char)b_lin[(t * 4 + 16 + j) * 8 + g]) << (8 * j);
    }
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_s8_m16n8k32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);

    // D (16x8 i32): c0 row g col t*2, c1 col t*2+1, c2/c3 row g+8.
    d_out[(g) * 8 + t * 2    ] = d0;
    d_out[(g) * 8 + t * 2 + 1] = d1;
    d_out[(g + 8) * 8 + t * 2    ] = d2;
    d_out[(g + 8) * 8 + t * 2 + 1] = d3;
}

// Warp mapping (256 threads = 8 warps for TM=128; 128 threads = 4 warps for
// TM=64): warp grid (TM/64) x 4, warp tile 64 rows x 16 tokens
// (4 m16 frags x 2 n8 frags). WARP_ROWS and THREADS derive from TM.

#define GEMM_BODY(NAME, TM, FUSED)                                            \
extern "C" __global__ void NAME(                                              \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */    \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */           \
    const unsigned int* __restrict__ q_hi_w,      /* [p * (n/4)] */           \
    const unsigned int* __restrict__ q_lo_w,                                    \
    const float* __restrict__ s_t,                /* [p] */                   \
    float* __restrict__ out,                      /* [p * m] */               \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char smem[];                      \
    unsigned char* a_stage = smem;                    /* [TM][132] */          \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int WARP_ROWS = TM / 64;                                             \
    const int THREADS = WARP_ROWS * 128;                                       \
    const int warp_row = (wid & (WARP_ROWS - 1)) * 64;                         \
    const int warp_n = wid / WARP_ROWS;            /* 0..3, x 16 tokens */     \
    const int tok_base = blockIdx.y * 64 + warp_n * 16;                        \
    const int row_base = blockIdx.x * TM + warp_row;                           \
    const int blk_rows = blockIdx.x * TM;   /* staging covers ALL TM rows */   \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* accumulators: 4 mfrag x 2 nfrag x 4 regs, hi + lo */                    \
    int dh[4][2][4];                                                           \
    int dl[4][2][4];                                                           \
    float o[4][2][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 4; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 2; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }                \
        }                                                                      \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        /* -- stage A: unpack this group's sign bytes into smem -- */          \
        if (grp != 0) __syncthreads();  /* all warps done consuming grp-1 */  \
        const int wpr = words_per_row;                                         \
        /* per group each row consumes words [grp*4 .. grp*4+4); 32 output   */\
        /* u32 per row (4 signs each) cover the group's 128 k.                */\
        for (int idx = tid; idx < TM * 32; idx += THREADS) {                   \
            const int r = idx >> 5;                                            \
            const int wq = idx & 31;                                           \
            const int row_c = blk_rows + r;                                    \
            const int rc = row_c < m ? row_c : m - 1;                          \
            const unsigned int pw = pos_bits[(long)rc * wpr + grp * 4 + (wq >> 3)];\
            const unsigned int nw = neg_bits[(long)rc * wpr + grp * 4 + (wq >> 3)];\
            const int k0 = (wq & 7) * 4;                                       \
            *(unsigned int*)(a_stage + r * 132 + wq * 4) = sign_word(pw, nw, k0);\
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 4; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; dl[mf][nf][c] = 0; }\
            }                                                                  \
        }                                                                      \
                                                                               \
        /* -- k loop: 4 steps of 32 k, mma accumulate -- */                    \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            /* A words: mfrag mf, rows g and g+8, quads t and t+4 (k+16).      \
               NOTE: the smem row slot is 128 bytes REUSED each group          \
               (staged per group), so the byte offset is group-RELATIVE. */  \
            unsigned int aw[4][4];                                             \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 4; ++mf) {                                   \
                const int r0 = warp_row + mf * 16 + g_id;                      \
                const unsigned char* p0 = a_stage + r0 * 132 + (ks * 32 + t_id * 4);\
                const unsigned char* p8 = p0 + 8 * 132;                        \
                aw[mf][0] = *(const unsigned int*)p0;                          \
                aw[mf][1] = *(const unsigned int*)p8;                          \
                aw[mf][2] = *(const unsigned int*)(p0 + 16);                   \
                aw[mf][3] = *(const unsigned int*)(p8 + 16);                   \
            }                                                                  \
            /* B words: nfrag nf, token warp_n*16 + nf*8 + g, words kb+t (+4).*/\
            unsigned int bw[2][2][2];                                          \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                int tok = tok_base + nf * 8 + g_id;                            \
                tok = tok < p ? tok : p - 1;                                   \
                const long wb = (long)tok * qwpr + grp * 32 + ks * 8 + t_id;   \
                bw[nf][0][0] = q_hi_w[wb];      bw[nf][0][1] = q_hi_w[wb + 4]; \
                bw[nf][1][0] = q_lo_w[wb];      bw[nf][1][1] = q_lo_w[wb + 4]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 4; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1], dh[mf][nf][2], dh[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][0][0], bw[nf][0][1]);                           \
                    mma_s8_m16n8k32(dl[mf][nf][0], dl[mf][nf][1], dl[mf][nf][2], dl[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][1][0], bw[nf][1][1]);                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* -- per-group fold: o += sw * ((f32)hi + (f32)lo / 128) -- */        \
        const float one128 = 0.0078125f;                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 4; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0 = r0 < m ? r0 : m - 1;                               \
            const int rc8 = (r0 + 8) < m ? (r0 + 8) : m - 1;                   \
            const float sw0 = group_scale[(long)rc0 * groups_per_row + grp];   \
            const float sw8 = group_scale[(long)rc8 * groups_per_row + grp];   \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) {                                  \
                    const float hi = (float)dh[mf][nf][c];                     \
                    const float lo = (float)dl[mf][nf][c];                     \
                    const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));     \
                    if (FUSED) {                                               \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[mf][nf][c]);\
                    } else {                                                   \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);    \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* -- epilogue: stage through smem for coalesced writes. Chunks of 32    */\
    /* rows; a warp's 16-row mfrag blocks never straddle a chunk.           */\
    float* stg = (float*)smem;                       /* [32][64] reused */    \
    const int tok_blk = blockIdx.y * 64;                                       \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < TM / 32; ++ch) {                                     \
        __syncthreads();                                                       \
        /* warps whose warp_row == ch*32's owning 64-row band (TM=64: the    */\
        /* single band; TM=128: bands 0/64 alternating chunks). Per mfrag:   */\
        /* write only when its 16-row block falls in this chunk.             */\
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 4; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    const int tk0 = warp_n * 16 + nf * 8 + t_id * 2;           \
                    stg[r_in * 64 + tk0]     = o[mf][nf][0];                   \
                    stg[r_in * 64 + tk0 + 1] = o[mf][nf][1];                   \
                    stg[(r_in + 8) * 64 + tk0]     = o[mf][nf][2];             \
                    stg[(r_in + 8) * 64 + tk0 + 1] = o[mf][nf][3];             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* cooperative coalesced copy: 32 rows x 64 toks out */                \
        const int row0 = blockIdx.x * TM + ch * 32;                            \
        for (int idx = tid; idx < 32 * 64; idx += THREADS) {                   \
            const int tk = idx >> 5;                                           \
            const int r = idx & 31;                                            \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * 64 + tk] * s_t[tok];   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BODY(gemm_i8_mma_tm128_fused, 128, 1)
GEMM_BODY(gemm_i8_mma_tm128_strict, 128, 0)
GEMM_BODY(gemm_i8_mma_tm64_fused, 64, 1)
GEMM_BODY(gemm_i8_mma_tm64_strict, 64, 0)

// ---------------------------------------------------------------------------
// v2 (Issue 734 Arm 10): SWAR sign unpack + input-word-major staging +
// double-buffered group pipeline. Bit-identical to v1 BY CONSTRUCTION —
// same sign bytes (the SWAR form equals per-bit pos−neg extraction for all
// four (pos,neg) bit combos: t=pw&~nw and u=nw&~pw are disjoint by
// construction, spread bytes ∈ {0,1}, smear 0x01→0xFF, OR of disjoint
// per-byte values), identical mma fragment sequence, identical fold order
// and epilogue. The changes are pure instruction economics + scheduling:
//   - staging per thread per group: 32 broadcast LDG + ~512 ALU + 16 STS
//     → 4 coalesced LDG + ~250 ALU + 16 STS (each input word feeds 8 output
//     words; the multiply-spread `(x*0x00204081)&0x01010101` puts nibble b
// at byte b, exact).
//   - group pipeline: unpack(g+1) interleaved into k-steps 0/1 (writes the
//     OTHER a_stage buffer — race-free: every warp passed the loop-top
//     barrier of g, hence finished g−1 entirely, before any unpack(g+1)
//     touches the buffer g−1 read); fetch(g+2) LDGs issue at k-step 2
//     (latency hidden behind the remaining mma batch + fold + barrier);
//     ONE barrier per group instead of two.
// Staging store banks at stride 132: (33r + 8w4 + q) mod 32 = r + 8w4 + q,
// distinct across the warp per instruction — conflict-free, same property
// the v1 stride was chosen for.
// ---------------------------------------------------------------------------

#define GEMM_BODY_V2(NAME, TM, FUSED, LB)                                      \
extern "C" __global__ void LB NAME(                                           \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */    \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */           \
    const unsigned int* __restrict__ q_hi_w,      /* [p * (n/4)] */           \
    const unsigned int* __restrict__ q_lo_w,                                    \
    const float* __restrict__ s_t,                /* [p] */                   \
    float* __restrict__ out,                      /* [p * m] */               \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char smem[];                      \
    unsigned char* const a_buf0 = smem;                                        \
    unsigned char* const a_buf1 = smem + (TM * 132);                           \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int WARP_ROWS = TM / 64;                                             \
    const int THREADS = WARP_ROWS * 128;                                       \
    const int warp_row = (wid & (WARP_ROWS - 1)) * 64;                         \
    const int warp_n = wid / WARP_ROWS;            /* 0..3, x 16 tokens */     \
    const int tok_base = blockIdx.y * 64 + warp_n * 16;                        \
    const int row_base = blockIdx.x * TM + warp_row;                           \
    const int blk_rows = blockIdx.x * TM;   /* staging covers ALL TM rows */   \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* staging ownership: 2 (row, input-word) slots per thread; input word    */\
    /* w4 covers output words w4*8..w4*8+8 (nibble q of the word = 4 signs).  */\
    const int s_w4 = tid & 3;                                                  \
    const int s_r0 = tid >> 2;                                                 \
    const int s_r1 = s_r0 + (THREADS >> 2);                                    \
    int rc0 = blk_rows + s_r0; rc0 = rc0 < m ? rc0 : m - 1;                    \
    int rc1 = blk_rows + s_r1; rc1 = rc1 < m ? rc1 : m - 1;                    \
    const long a_off0 = (long)rc0 * words_per_row + s_w4;                      \
    const long a_off1 = (long)rc1 * words_per_row + s_w4;                      \
    unsigned int pf_pw0, pf_nw0, pf_pw1, pf_nw1;                               \
                                                                               \
    /* accumulators: 4 mfrag x 2 nfrag x 4 regs, hi + lo */                    \
    int dh[4][2][4];                                                           \
    int dl[4][2][4];                                                           \
    float o[4][2][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 4; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 2; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }                \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: stage group 0 into buf0, fetch group 1. */                    \
    pf_pw0 = pos_bits[a_off0];                                                 \
    pf_nw0 = neg_bits[a_off0];                                                 \
    pf_pw1 = pos_bits[a_off1];                                                 \
    pf_nw1 = neg_bits[a_off1];                                                 \
    {                                                                          \
        const unsigned int t0 = pf_pw0 & ~pf_nw0;                              \
        const unsigned int u0 = pf_nw0 & ~pf_pw0;                              \
        _Pragma("unroll")                                                      \
        for (int q = 0; q < 8; ++q) {                                          \
            unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                       \
            *(unsigned int*)(a_buf0 + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
        }                                                                      \
        const unsigned int t1 = pf_pw1 & ~pf_nw1;                              \
        const unsigned int u1 = pf_nw1 & ~pf_pw1;                              \
        _Pragma("unroll")                                                      \
        for (int q = 0; q < 8; ++q) {                                          \
            unsigned int tp = ((t1 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u1 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                       \
            *(unsigned int*)(a_buf0 + s_r1 * 132 + s_w4 * 32 + q * 4) = tp | up;\
        }                                                                      \
    }                                                                          \
    if (1 < groups_per_row) {                                                  \
        const long g4 = 4;                                                     \
        pf_pw0 = pos_bits[a_off0 + g4];                                        \
        pf_nw0 = neg_bits[a_off0 + g4];                                        \
        pf_pw1 = pos_bits[a_off1 + g4];                                        \
        pf_nw1 = neg_bits[a_off1 + g4];                                        \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        __syncthreads();   /* unpack(grp) visible to all warps; all g−1 done */\
        unsigned char* buf = (grp & 1) ? a_buf1 : a_buf0;                      \
        unsigned char* nbuf = (grp & 1) ? a_buf0 : a_buf1;                     \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 4; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; dl[mf][nf][c] = 0; }\
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[4][4];                                             \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 4; ++mf) {                                   \
                const int r0 = warp_row + mf * 16 + g_id;                      \
                const unsigned char* p0 = buf + r0 * 132 + (ks * 32 + t_id * 4);\
                const unsigned char* p8 = p0 + 8 * 132;                         \
                aw[mf][0] = *(const unsigned int*)p0;                           \
                aw[mf][1] = *(const unsigned int*)p8;                           \
                aw[mf][2] = *(const unsigned int*)(p0 + 16);                   \
                aw[mf][3] = *(const unsigned int*)(p8 + 16);                   \
            }                                                                  \
            unsigned int bw[2][2][2];                                          \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                int tok = tok_base + nf * 8 + g_id;                            \
                tok = tok < p ? tok : p - 1;                                   \
                const long wb = (long)tok * qwpr + grp * 32 + ks * 8 + t_id;   \
                bw[nf][0][0] = q_hi_w[wb];      bw[nf][0][1] = q_hi_w[wb + 4]; \
                bw[nf][1][0] = q_lo_w[wb];      bw[nf][1][1] = q_lo_w[wb + 4]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 4; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1], dh[mf][nf][2], dh[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][0][0], bw[nf][0][1]);                           \
                    mma_s8_m16n8k32(dl[mf][nf][0], dl[mf][nf][1], dl[mf][nf][2], dl[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][1][0], bw[nf][1][1]);                           \
                }                                                              \
            }                                                                  \
            /* interleaved staging for grp+1 (writes nbuf — the buffer this  */\
            /* warp's g−1 iteration read; every warp finished g−1 before the */\
            /* loop-top barrier of g, so the write is race-free) + prefetch  */\
            /* for grp+2 at ks==2 (LDG latency hidden behind the remaining   */\
            /* mma batch, fold, and next barrier).                            */\
            if (has_next) {                                                    \
                if (ks == 0) {                                                 \
                    const unsigned int t0 = pf_pw0 & ~pf_nw0;                  \
                    const unsigned int u0 = pf_nw0 & ~pf_pw0;                  \
                    _Pragma("unroll")                                          \
                    for (int q = 0; q < 8; ++q) {                              \
                        unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;           \
                        *(unsigned int*)(nbuf + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                          \
                } else if (ks == 1) {                                          \
                    const unsigned int t1 = pf_pw1 & ~pf_nw1;                  \
                    const unsigned int u1 = pf_nw1 & ~pf_pw1;                  \
                    _Pragma("unroll")                                          \
                    for (int q = 0; q < 8; ++q) {                              \
                        unsigned int tp = ((t1 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u1 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;           \
                        *(unsigned int*)(nbuf + s_r1 * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                          \
                } else if (ks == 2 && (grp + 2) < groups_per_row) {            \
                    const long g4 = (long)(grp + 2) * 4;                       \
                    pf_pw0 = pos_bits[a_off0 + g4];                            \
                    pf_nw0 = neg_bits[a_off0 + g4];                            \
                    pf_pw1 = pos_bits[a_off1 + g4];                            \
                    pf_nw1 = neg_bits[a_off1 + g4];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const float one128 = 0.0078125f;                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 4; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) {                                  \
                    const float hi = (float)dh[mf][nf][c];                     \
                    const float lo = (float)dl[mf][nf][c];                     \
                    const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));     \
                    if (FUSED) {                                               \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[mf][nf][c]);\
                    } else {                                                   \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);    \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: identical to v1. */                                           \
    float* stg = (float*)smem;                       /* [32][64] reused */    \
    const int tok_blk = blockIdx.y * 64;                                       \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < TM / 32; ++ch) {                                     \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 4; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    const int tk0 = warp_n * 16 + nf * 8 + t_id * 2;           \
                    stg[r_in * 64 + tk0]     = o[mf][nf][0];                   \
                    stg[r_in * 64 + tk0 + 1] = o[mf][nf][1];                   \
                    stg[(r_in + 8) * 64 + tk0]     = o[mf][nf][2];             \
                    stg[(r_in + 8) * 64 + tk0 + 1] = o[mf][nf][3];             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * TM + ch * 32;                            \
        for (int idx = tid; idx < 32 * 64; idx += THREADS) {                   \
            const int tk = idx >> 5;                                           \
            const int r = idx & 31;                                            \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * 64 + tk] * s_t[tok];   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BODY_V2(gemm_i8_mma_tm128v2_fused, 128, 1, )
GEMM_BODY_V2(gemm_i8_mma_tm128v2_strict, 128, 0, )
GEMM_BODY_V2(gemm_i8_mma_tm64v2_fused, 64, 1, )
GEMM_BODY_V2(gemm_i8_mma_tm64v2_strict, 64, 0, )
// Occupancy-pinned probes (Arm 10 G2): trade registers for blocks/SM.
GEMM_BODY_V2(gemm_i8_mma_tm128v2b2_fused, 128, 1, __launch_bounds__(256, 2))
GEMM_BODY_V2(gemm_i8_mma_tm64v2b4_fused, 64, 1, __launch_bounds__(128, 4))

// ---------------------------------------------------------------------------
// v3 (Arm 10): 512-thread TM=128 shape with 32-row warp tiles. The v1/v2
// warp tile (64r x 16t) needs 96 accumulator registers, capping TM=128 at
// 8 warps/SM — while BOTH tiles run at the same effective B L2 traffic
// (warp-rows re-read the same tokens' q words; the measured plateau
// ~105-116 TF is that traffic at the ~1.7 TB/s effective L2 ceiling).
// v3 halves the warp tile (mf=2: 48 acc regs, ~110 total) so 16 warps fit
// one block (512 thr x ~110 regs = 56K <= 64K): the 4 row-bands' duplicate
// B reads then hit L1 (same SM, concurrent k-loops) — L2 B traffic halves
// (~2.85 GB at the ffn_gate shape) AND 16 warps cover latency.
// Bit-identity: same sign bytes (the v2 SWAR unpack), same mma fragment
// sequence per output, same fold order — only the warp/tile decomposition
// changes. Requires TM=128 exactly (4 row-bands x 4 token-warps).
// ---------------------------------------------------------------------------

#define GEMM_BODY_V3(NAME, FUSED)                                              \
extern "C" __global__ void __launch_bounds__(512, 1) NAME(                    \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */    \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */           \
    const unsigned int* __restrict__ q_hi_w,      /* [p * (n/4)] */           \
    const unsigned int* __restrict__ q_lo_w,                                    \
    const float* __restrict__ s_t,                /* [p] */                   \
    float* __restrict__ out,                      /* [p * m] */               \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char smem[];                      \
    unsigned char* const a_buf0 = smem;                                        \
    unsigned char* const a_buf1 = smem + (128 * 132);                          \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                    /* 0..15 */                   \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid >> 2) * 32;         /* 4 row-bands of 32 */      \
    const int warp_n = wid & 3;                   /* 4 token-warps x 16 */     \
    const int tok_base = blockIdx.y * 64 + warp_n * 16;                        \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* staging ownership: ONE (row, input-word) slot per thread. */            \
    const int s_w4 = tid & 3;                                                  \
    const int s_r0 = tid >> 2;               /* 0..127 */                      \
    int rc0 = blk_rows + s_r0; rc0 = rc0 < m ? rc0 : m - 1;                    \
    const long a_off0 = (long)rc0 * words_per_row + s_w4;                      \
    unsigned int pf_pw0, pf_nw0;                                               \
                                                                               \
    /* accumulators: 2 mfrag x 2 nfrag x 4 regs, hi + lo */                    \
    int dh[2][2][4];                                                           \
    int dl[2][2][4];                                                           \
    float o[2][2][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 2; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }                \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: stage group 0 into buf0, fetch group 1. */                    \
    pf_pw0 = pos_bits[a_off0];                                                 \
    pf_nw0 = neg_bits[a_off0];                                                 \
    {                                                                          \
        const unsigned int t0 = pf_pw0 & ~pf_nw0;                              \
        const unsigned int u0 = pf_nw0 & ~pf_pw0;                              \
        _Pragma("unroll")                                                      \
        for (int q = 0; q < 8; ++q) {                                          \
            unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                       \
            *(unsigned int*)(a_buf0 + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
        }                                                                      \
    }                                                                          \
    if (1 < groups_per_row) {                                                  \
        const long g4 = 4;                                                     \
        pf_pw0 = pos_bits[a_off0 + g4];                                        \
        pf_nw0 = neg_bits[a_off0 + g4];                                        \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        __syncthreads();   /* unpack(grp) visible to all warps; all g-1 done */\
        unsigned char* buf = (grp & 1) ? a_buf1 : a_buf0;                      \
        unsigned char* nbuf = (grp & 1) ? a_buf0 : a_buf1;                     \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; dl[mf][nf][c] = 0; }\
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[2][4];                                             \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                const int r0 = warp_row + mf * 16 + g_id;                      \
                const unsigned char* p0 = buf + r0 * 132 + (ks * 32 + t_id * 4);\
                const unsigned char* p8 = p0 + 8 * 132;                         \
                aw[mf][0] = *(const unsigned int*)p0;                           \
                aw[mf][1] = *(const unsigned int*)p8;                           \
                aw[mf][2] = *(const unsigned int*)(p0 + 16);                   \
                aw[mf][3] = *(const unsigned int*)(p8 + 16);                   \
            }                                                                  \
            unsigned int bw[2][2][2];                                          \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                int tok = tok_base + nf * 8 + g_id;                            \
                tok = tok < p ? tok : p - 1;                                   \
                const long wb = (long)tok * qwpr + grp * 32 + ks * 8 + t_id;   \
                bw[nf][0][0] = q_hi_w[wb];      bw[nf][0][1] = q_hi_w[wb + 4]; \
                bw[nf][1][0] = q_lo_w[wb];      bw[nf][1][1] = q_lo_w[wb + 4]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1], dh[mf][nf][2], dh[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][0][0], bw[nf][0][1]);                           \
                    mma_s8_m16n8k32(dl[mf][nf][0], dl[mf][nf][1], dl[mf][nf][2], dl[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][1][0], bw[nf][1][1]);                           \
                }                                                              \
            }                                                                  \
            /* interleaved staging for grp+1 (single slot — one unpack) +    */\
            /* prefetch for grp+2 at ks==2.                                   */\
            if (has_next) {                                                    \
                if (ks == 0) {                                                 \
                    const unsigned int t0 = pf_pw0 & ~pf_nw0;                  \
                    const unsigned int u0 = pf_nw0 & ~pf_pw0;                  \
                    _Pragma("unroll")                                          \
                    for (int q = 0; q < 8; ++q) {                              \
                        unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;           \
                        *(unsigned int*)(nbuf + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                          \
                } else if (ks == 2 && (grp + 2) < groups_per_row) {            \
                    const long g4 = (long)(grp + 2) * 4;                       \
                    pf_pw0 = pos_bits[a_off0 + g4];                            \
                    pf_nw0 = neg_bits[a_off0 + g4];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const float one128 = 0.0078125f;                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) {                                  \
                    const float hi = (float)dh[mf][nf][c];                     \
                    const float lo = (float)dl[mf][nf][c];                     \
                    const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));     \
                    if (FUSED) {                                               \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[mf][nf][c]);\
                    } else {                                                   \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);    \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: 32-row chunks; warp owns rows only in its chunk. */           \
    float* stg = (float*)smem;                       /* [32][64] reused */    \
    const int tok_blk = blockIdx.y * 64;                                       \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    const int tk0 = warp_n * 16 + nf * 8 + t_id * 2;           \
                    stg[r_in * 64 + tk0]     = o[mf][nf][0];                   \
                    stg[r_in * 64 + tk0 + 1] = o[mf][nf][1];                   \
                    stg[(r_in + 8) * 64 + tk0]     = o[mf][nf][2];             \
                    stg[(r_in + 8) * 64 + tk0 + 1] = o[mf][nf][3];             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * 64; idx += 512) {                       \
            const int tk = idx >> 5;                                           \
            const int r = idx & 31;                                            \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * 64 + tk] * s_t[tok];   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BODY_V3(gemm_i8_mma_tm128v3_fused, 1)
GEMM_BODY_V3(gemm_i8_mma_tm128v3_strict, 0)

// ---------------------------------------------------------------------------
// v4 (Arm 10): v3's 512-thread TM=128 shape + B smem staging. The measured
// model (Bench 724): every v1/v2/v3 variant's time == its effective B-side
// L2 traffic / ~1.65-1.7 TB/s — the kernel is B-L2-bandwidth-bound, and the
// warp-row structure re-reads the same tokens' q words per row-band (L1
// absorbs only ~55-70%). v4 stages the block's 64-token × 128-k B slab
// ONCE per group in shared memory (shared across the 4 row-bands):
// B L2 traffic 5.7 GB -> 2.85 GB at the ffn_gate shape (halved), fragment
// reads become LDS. B-slab stride 36 words: 16-byte aligned AND bank
// conflict-free ((4·tok + w) mod 32 distinct across the warp).
// Bit-identity: identical bytes staged (plain copies), identical fragment
// words, identical mma sequence + fold — only the B data path moves.
// ---------------------------------------------------------------------------

#define GEMM_BODY_V4(NAME, FUSED)                                              \
extern "C" __global__ void __launch_bounds__(512, 1) NAME(                    \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */    \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */           \
    const unsigned int* __restrict__ q_hi_w,      /* [p * (n/4)] */           \
    const unsigned int* __restrict__ q_lo_w,                                    \
    const float* __restrict__ s_t,                /* [p] */                   \
    float* __restrict__ out,                      /* [p * m] */               \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char smem[];                      \
    unsigned char* const a_buf0 = smem;                                        \
    unsigned char* const a_buf1 = smem + (128 * 132);                          \
    /* B slabs: [2 planes][64 toks][36 words] u32, double-buffered. */         \
    unsigned int* const b_buf0 = (unsigned int*)(smem + 2 * (128 * 132));      \
    unsigned int* const b_buf1 = b_buf0 + (2 * 64 * 36);                       \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                    /* 0..15 */                   \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid >> 2) * 32;         /* 4 row-bands of 32 */      \
    const int warp_n = wid & 3;                   /* 4 token-warps x 16 */     \
    const int tok_base = blockIdx.y * 64 + warp_n * 16;                        \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* A staging ownership: ONE (row, input-word) slot per thread. */          \
    const int s_w4 = tid & 3;                                                  \
    const int s_r0 = tid >> 2;               /* 0..127 */                      \
    int rc0 = blk_rows + s_r0; rc0 = rc0 < m ? rc0 : m - 1;                    \
    const long a_off0 = (long)rc0 * words_per_row + s_w4;                      \
    unsigned int pf_pw0, pf_nw0;                                               \
    /* B staging ownership: each thread copies ONE 16B quad per plane per      */\
    /* group (tid -> w4 = 4*(tid&7), tok = tid>>3): 1 LDG.128 + 1 STS.128 per   */\
    /* plane, 16B-aligned on both sides (qwpr and grp*32 words are 32B+ aligned,*/\
    /* the slab stride 36 words = 144B).                                        */\
    const int bs_w4 = (tid & 7) << 2;                                           \
    const int bs_t0 = tid >> 3;               /* 0..63 */                       \
    int bs_tr = (int)(blockIdx.y * 64) + bs_t0; bs_tr = bs_tr < p ? bs_tr : p - 1;\
    const long bs_g = (long)bs_tr * qwpr + bs_w4;                              \
    const unsigned int bs_sofs = (unsigned int)(bs_t0 * 36) + (unsigned int)bs_w4;\
                                                                               \
    /* accumulators: 2 mfrag x 2 nfrag x 4 regs, hi + lo */                    \
    int dh[2][2][4];                                                           \
    int dl[2][2][4];                                                           \
    float o[2][2][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 2; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }                \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: stage group 0 (A + B) into buf0, fetch group 1's A. */        \
    pf_pw0 = pos_bits[a_off0];                                                 \
    pf_nw0 = neg_bits[a_off0];                                                 \
    {                                                                          \
        const unsigned int t0 = pf_pw0 & ~pf_nw0;                              \
        const unsigned int u0 = pf_nw0 & ~pf_pw0;                              \
        _Pragma("unroll")                                                      \
        for (int q = 0; q < 8; ++q) {                                          \
            unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                       \
            *(unsigned int*)(a_buf0 + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int pl = 0; pl < 2; ++pl) {                                           \
        const unsigned int* const qp = pl ? q_lo_w : q_hi_w;                   \
        const unsigned int* const g4p = (const unsigned int*)(qp + bs_g);      \
        *(uint4*)(b_buf0 + (unsigned int)(pl * 2304) + bs_sofs) =              \
            *(const uint4*)g4p;                                                \
    }                                                                          \
    if (1 < groups_per_row) {                                                  \
        pf_pw0 = pos_bits[a_off0 + 4];                                         \
        pf_nw0 = neg_bits[a_off0 + 4];                                         \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        __syncthreads();   /* A(grp) + B(grp) staged & g-1 fully done */        \
        unsigned char* const buf = (grp & 1) ? a_buf1 : a_buf0;               \
        unsigned char* const nbuf = (grp & 1) ? a_buf0 : a_buf1;               \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        const unsigned int lo_plane = 64u * 36u;                               \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; dl[mf][nf][c] = 0; }\
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[2][4];                                             \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                const int r0 = warp_row + mf * 16 + g_id;                      \
                const unsigned char* p0 = buf + r0 * 132 + (ks * 32 + t_id * 4);\
                const unsigned char* p8 = p0 + 8 * 132;                         \
                aw[mf][0] = *(const unsigned int*)p0;                           \
                aw[mf][1] = *(const unsigned int*)p8;                           \
                aw[mf][2] = *(const unsigned int*)(p0 + 16);                   \
                aw[mf][3] = *(const unsigned int*)(p8 + 16);                   \
            }                                                                  \
            unsigned int bw[2][2][2];                                          \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                /* slab slots are BLOCK-RELATIVE tokens (0..63); the staging */\
                /* already clamped the source row to p-1. */                   \
                const unsigned int bb = (unsigned int)(warp_n * 16 + nf * 8 + g_id) * 36u\
                    + (unsigned int)(ks * 8 + t_id);                           \
                bw[nf][0][0] = bbuf[bb];          bw[nf][0][1] = bbuf[bb + 4u]; \
                bw[nf][1][0] = bbuf[lo_plane + bb]; bw[nf][1][1] = bbuf[lo_plane + bb + 4u];\
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1], dh[mf][nf][2], dh[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][0][0], bw[nf][0][1]);                           \
                    mma_s8_m16n8k32(dl[mf][nf][0], dl[mf][nf][1], dl[mf][nf][2], dl[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][1][0], bw[nf][1][1]);                           \
                }                                                              \
            }                                                                  \
            if (has_next) {                                                    \
                if (ks == 0) {                                                 \
                    const unsigned int t0 = pf_pw0 & ~pf_nw0;                  \
                    const unsigned int u0 = pf_nw0 & ~pf_pw0;                  \
                    _Pragma("unroll")                                          \
                    for (int q = 0; q < 8; ++q) {                              \
                        unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;           \
                        *(unsigned int*)(nbuf + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                          \
                }                                                              \
                /* B stage for grp+1: one 16B quad per plane at ks1/ks2 (writes  */\
                /* nbbuf, the slab grp-1 read — every warp finished grp-1 at the  */\
                /* loop-top barrier; load latency covered by the remaining mma    */\
                /* batch + fold + barrier before consumption).                    */\
                if (ks == 1) {                                                 \
                    const unsigned int* const qp = q_hi_w;                     \
                    *(uint4*)(nbbuf + bs_sofs) =                               \
                        *(const uint4*)(qp + bs_g + (long)((grp + 1) << 5));    \
                } else if (ks == 2) {                                          \
                    const unsigned int* const qp = q_lo_w;                     \
                    *(uint4*)(nbbuf + 2304u + bs_sofs) =                       \
                        *(const uint4*)(qp + bs_g + (long)((grp + 1) << 5));    \
                }                                                              \
                if (ks == 2 && (grp + 2) < groups_per_row) {                   \
                    const long g4 = (long)(grp + 2) * 4;                       \
                    pf_pw0 = pos_bits[a_off0 + g4];                            \
                    pf_nw0 = neg_bits[a_off0 + g4];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const float one128 = 0.0078125f;                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) {                                  \
                    const float hi = (float)dh[mf][nf][c];                     \
                    const float lo = (float)dl[mf][nf][c];                     \
                    const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));     \
                    if (FUSED) {                                               \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[mf][nf][c]);\
                    } else {                                                   \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);    \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: identical to v3. */                                           \
    float* stg = (float*)smem;                       /* [32][64] reused */    \
    const int tok_blk = blockIdx.y * 64;                                       \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    const int tk0 = warp_n * 16 + nf * 8 + t_id * 2;           \
                    stg[r_in * 64 + tk0]     = o[mf][nf][0];                   \
                    stg[r_in * 64 + tk0 + 1] = o[mf][nf][1];                   \
                    stg[(r_in + 8) * 64 + tk0]     = o[mf][nf][2];             \
                    stg[(r_in + 8) * 64 + tk0 + 1] = o[mf][nf][3];             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * 64; idx += 512) {                       \
            const int tk = idx >> 5;                                           \
            const int r = idx & 31;                                            \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * 64 + tk] * s_t[tok];   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BODY_V4(gemm_i8_mma_tm128v4_fused, 1)
GEMM_BODY_V4(gemm_i8_mma_tm128v4_strict, 0)

// ---------------------------------------------------------------------------
// v4-sp (Issue 742 T1): the small-p variant of v4 — 128 threads, TM=32 rows
// x TOKS=16, warp tile 16r x 8t (1 mfrag x 1 nfrag). Numerics are BYTE-
// IDENTICAL to v4: same quantize scratch, same A sign bytes (identical SWAR
// staging at the same 132-byte row stride), same B slab words at the same
// 36-word stride, same mma fragment mapping (a0 row g_id / a1 row g_id+8 /
// a2,a3 = +16 k; b0/b1 = the token's q words at ks*8+t_id and +4), same
// per-group fmaf fold in ascending group order, same s_t epilogue — ONLY
// the block/warp geometry changes (which rows/tokens a block covers never
// affects per-element arithmetic).
//
// Why (Bench 729): at p <= 16 v4's grid is (m/128, 1) — 136 blocks on 128
// SMs with 70.7 KB smem = 1 block/SM; the lone block's barrier + epilogue
// bubbles are never hidden (measured ~110 GB/s effective vs the ~700 floor
// — the token axis IS v4's parallelism axis and p=16 leaves it empty). The
// sp shape cuts smem to 17.7 KB (5 blocks/SM) and quarters the row tile
// (grid m/32 — 544 blocks at ffn_gate), restoring the stall interleaving
// that saturates the memory system.
// ---------------------------------------------------------------------------

#define GEMM_BODY_V4_SP(NAME, FUSED)                                              \
extern "C" __global__ void __launch_bounds__(128) NAME(                    \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */    \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */           \
    const unsigned int* __restrict__ q_hi_w,      /* [p * (n/4)] */           \
    const unsigned int* __restrict__ q_lo_w,                                    \
    const float* __restrict__ s_t,                /* [p] */                   \
    float* __restrict__ out,                      /* [p * m] */               \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char smem[];                      \
    unsigned char* const a_buf0 = smem;                                        \
    unsigned char* const a_buf1 = smem + (32 * 132);                           \
    /* B slabs: [2 planes][16 toks][36 words] u32, double-buffered. */         \
    unsigned int* const b_buf0 = (unsigned int*)(smem + 2 * (32 * 132));      \
    unsigned int* const b_buf1 = b_buf0 + (2 * 16 * 36);                       \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                    /* 0..3 */                    \
    const int g_id = lane >> 2;                                                 \
    const int t_id = lane & 3;                                                  \
    const int warp_row = (wid >> 1) * 16;         /* 2 row-halves of 16 */     \
    const int warp_n = wid & 1;                   /* 2 token-warps x 8 */      \
    const int tok_base = warp_n * 8;               /* p <= 16: blockIdx.y==0 */\
    const int row_base = blockIdx.x * 32 + warp_row;                          \
    const int blk_rows = blockIdx.x * 32;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* A staging ownership: ONE (row, input-word) slot per thread. */          \
    const int s_w4 = tid & 3;                                                  \
    const int s_r0 = tid >> 2;               /* 0..31 */                       \
    int rc0 = blk_rows + s_r0; rc0 = rc0 < m ? rc0 : m - 1;                    \
    const long a_off0 = (long)rc0 * words_per_row + s_w4;                      \
    unsigned int pf_pw0, pf_nw0;                                               \
    /* B staging ownership: each thread copies ONE 16B quad per plane per      */\
    /* group (tid -> w4 = 4*(tid&7), tok = tid>>3): 16 toks x 32 words = 128.   */\
    const int bs_w4 = (tid & 7) << 2;                                           \
    const int bs_t0 = tid >> 3;               /* 0..15 */                       \
    int bs_tr = bs_t0; bs_tr = bs_tr < p ? bs_tr : p - 1;                      \
    const long bs_g = (long)bs_tr * qwpr + bs_w4;                              \
    const unsigned int bs_sofs = (unsigned int)(bs_t0 * 36) + (unsigned int)bs_w4;\
    const unsigned int lo_plane = 16u * 36u;                                   \
                                                                               \
    /* accumulators: 1 mfrag x 1 nfrag x 4 regs, hi + lo. */                   \
    int dh[1][1][4];                                                           \
    int dl[1][1][4];                                                           \
    float o[1][1][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 4; ++c) { o[0][0][c] = 0.0f; }                          \
                                                                               \
    /* prologue: stage group 0 (A + B) into buf0, fetch group 1's A. */        \
    pf_pw0 = pos_bits[a_off0];                                                 \
    pf_nw0 = neg_bits[a_off0];                                                 \
    {                                                                          \
        const unsigned int t0 = pf_pw0 & ~pf_nw0;                              \
        const unsigned int u0 = pf_nw0 & ~pf_pw0;                              \
        _Pragma("unroll")                                                      \
        for (int q = 0; q < 8; ++q) {                                          \
            unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                       \
            *(unsigned int*)(a_buf0 + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int pl = 0; pl < 2; ++pl) {                                           \
        const unsigned int* const qp = pl ? q_lo_w : q_hi_w;                   \
        const unsigned int* const g4p = (const unsigned int*)(qp + bs_g);      \
        *(uint4*)(b_buf0 + (unsigned int)(pl * 576) + bs_sofs) =               \
            *(const uint4*)g4p;                                                \
    }                                                                          \
    if (1 < groups_per_row) {                                                  \
        pf_pw0 = pos_bits[a_off0 + 4];                                         \
        pf_nw0 = neg_bits[a_off0 + 4];                                         \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        __syncthreads();   /* A(grp) + B(grp) staged & g-1 fully done */        \
        unsigned char* const buf = (grp & 1) ? a_buf1 : a_buf0;               \
        unsigned char* const nbuf = (grp & 1) ? a_buf0 : a_buf1;               \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;                \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < 4; ++c) { dh[0][0][c] = 0; dl[0][0][c] = 0; }      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[4];                                                \
            const int r0 = warp_row + g_id;                                    \
            const unsigned char* p0 = buf + r0 * 132 + (ks * 32 + t_id * 4);   \
            const unsigned char* p8 = p0 + 8 * 132;                            \
            aw[0] = *(const unsigned int*)p0;                                  \
            aw[1] = *(const unsigned int*)p8;                                  \
            aw[2] = *(const unsigned int*)(p0 + 16);                           \
            aw[3] = *(const unsigned int*)(p8 + 16);                           \
            unsigned int bw[2][2];                                             \
            const unsigned int bb = (unsigned int)(warp_n * 8 + g_id) * 36u    \
                + (unsigned int)(ks * 8 + t_id);                              \
            bw[0][0] = bbuf[bb];            bw[0][1] = bbuf[bb + 4u];           \
            bw[1][0] = bbuf[lo_plane + bb];  bw[1][1] = bbuf[lo_plane + bb + 4u];\
            mma_s8_m16n8k32(dh[0][0][0], dh[0][0][1], dh[0][0][2], dh[0][0][3],\
                aw[0], aw[1], aw[2], aw[3], bw[0][0], bw[0][1]);               \
            mma_s8_m16n8k32(dl[0][0][0], dl[0][0][1], dl[0][0][2], dl[0][0][3],\
                aw[0], aw[1], aw[2], aw[3], bw[1][0], bw[1][1]);               \
            if (has_next) {                                                    \
                if (ks == 0) {                                                 \
                    const unsigned int t0 = pf_pw0 & ~pf_nw0;                  \
                    const unsigned int u0 = pf_nw0 & ~pf_pw0;                  \
                    _Pragma("unroll")                                          \
                    for (int q = 0; q < 8; ++q) {                              \
                        unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;           \
                        *(unsigned int*)(nbuf + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                          \
                }                                                              \
                if (ks == 1) {                                                 \
                    const unsigned int* const qp = q_hi_w;                     \
                    *(uint4*)(nbbuf + bs_sofs) =                               \
                        *(const uint4*)(qp + bs_g + (long)((grp + 1) << 5));    \
                } else if (ks == 2) {                                          \
                    const unsigned int* const qp = q_lo_w;                     \
                    *(uint4*)(nbbuf + lo_plane + bs_sofs) =                    \
                        *(const uint4*)(qp + bs_g + (long)((grp + 1) << 5));    \
                }                                                              \
                if (ks == 2 && (grp + 2) < groups_per_row) {                   \
                    const long g4 = (long)(grp + 2) * 4;                       \
                    pf_pw0 = pos_bits[a_off0 + g4];                            \
                    pf_nw0 = neg_bits[a_off0 + g4];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const float one128 = 0.0078125f;                                       \
        const int r0 = row_base + g_id;                                        \
        const int rc0f = r0 < m ? r0 : m - 1;                                  \
        const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                      \
        const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];      \
        const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];      \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < 4; ++c) {                                          \
            const float hi = (float)dh[0][0][c];                               \
            const float lo = (float)dl[0][0][c];                               \
            const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));             \
            if (FUSED) {                                                       \
                o[0][0][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[0][0][c]);    \
            } else {                                                           \
                const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);            \
                o[0][0][c] = __fadd_rn(o[0][0][c], t3);                         \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: staged transpose then coalesced stores — same formula as     */\
    /* v4 (out[tok * m + row] = o * s_t[tok]) over the block's 32r x 16t tile. */\
    float* stg = (float*)smem;                       /* [32][16] reused */    \
    __syncthreads();                                                           \
    const int r_in = warp_row + g_id;                                          \
    const int tk0 = tok_base + t_id * 2;                                       \
    stg[r_in * 16 + tk0]         = o[0][0][0];                                  \
    stg[r_in * 16 + tk0 + 1]     = o[0][0][1];                                  \
    stg[(r_in + 8) * 16 + tk0]     = o[0][0][2];                                \
    stg[(r_in + 8) * 16 + tk0 + 1] = o[0][0][3];                                \
    __syncthreads();                                                           \
    const int row0 = blockIdx.x * 32;                                          \
    for (int idx = tid; idx < 32 * 16; idx += 128) {                           \
        const int r = idx & 31;                                                \
        const int tk = idx >> 5;                                               \
        if (tk < p && row0 + r < m) {                                          \
            out[(long)tk * m + row0 + r] = stg[r * 16 + tk] * s_t[tk];         \
        }                                                                      \
    }                                                                          \
}

GEMM_BODY_V4_SP(gemm_i8_mma_tm32sp_fused, 1)
GEMM_BODY_V4_SP(gemm_i8_mma_tm32sp_strict, 0)

// ---------------------------------------------------------------------------
// Issue 742 T1.4 — the y-split small-p GEMM (32r x 8t, 64 threads).
// MEASURED NEGATIVE (2026-08-24, GPU-exclusive): x0.63-0.71 vs TM32sp at
// every shape — both x-blocks of a row tile stream the SAME A rows (2x
// weight DRAM traffic; down-proj: 44.6 MB at ~274 GB/s eff = 162.8 us vs
// 103.4), swamping the wave-balance gain. The discrete-wave imbalance
// model was also too crude: TM32sp's 5-blocks/SM occupancy already
// softens the ragged tail. Kept bit-identical (G1-gated at every shape)
// behind the opt-in RIIR_PREFILL_SMALLP_Y8=1 knob as the reproducible
// negative-result artifact.
// ---------------------------------------------------------------------------

#define GEMM_BODY_V4_SP8Y(NAME, FUSED)                                            \
extern "C" __global__ void __launch_bounds__(64) NAME(                           \
    const unsigned int* __restrict__ pos_bits,                                    \
    const unsigned int* __restrict__ neg_bits,                                    \
    const float* __restrict__ group_scale,                                        \
    const unsigned int* __restrict__ q_hi_w,                                      \
    const unsigned int* __restrict__ q_lo_w,                                      \
    const float* __restrict__ s_t,                                                \
    float* __restrict__ out,                                                      \
    int m, int n, int p,                                                          \
    int words_per_row, int groups_per_row)                                        \
{                                                                                 \
    extern __shared__ __align__(16) unsigned char smem[];                         \
    unsigned char* const a_buf0 = smem;                                           \
    unsigned char* const a_buf1 = smem + (32 * 132);                              \
    /* B slabs: [2 planes][8 toks][36 words] u32, double-buffered. */             \
    unsigned int* const b_buf0 = (unsigned int*)(smem + 2 * (32 * 132));          \
    unsigned int* const b_buf1 = b_buf0 + (2 * 8 * 36);                           \
                                                                                  \
    const int tid = threadIdx.x;                                                  \
    const int lane = tid & 31;                                                    \
    const int wid = tid >> 5;                    /* 0..1 */                       \
    const int g_id = lane >> 2;                                                   \
    const int t_id = lane & 3;                                                    \
    const int warp_row = wid * 16;                /* 2 row-halves of 16 */        \
    const int tok_base = blockIdx.x * 8;          /* grid.x = token slice */     \
    const int row_base = blockIdx.y * 32;                                         \
    const int blk_rows = blockIdx.y * 32;                                         \
    const int qwpr = n >> 2;                                                      \
                                                                                  \
    /* A staging: 128 (row, input-word) slots / 64 threads = 2 each —            \
       slot a: rows 0..15, slot b: rows 16..31 (same word column). */            \
    const int s_w4 = tid & 3;                                                     \
    const int s_ra = tid >> 2;               /* 0..15 */                          \
    const int s_rb = s_ra + 16;              /* 16..31 */                         \
    int rca = blk_rows + s_ra; rca = rca < m ? rca : m - 1;                        \
    int rcb = blk_rows + s_rb; rcb = rcb < m ? rcb : m - 1;                        \
    const long a_offa = (long)rca * words_per_row + s_w4;                          \
    const long a_offb = (long)rcb * words_per_row + s_w4;                          \
    unsigned int pf_pwa, pf_nwa, pf_pwb, pf_nwb;                                  \
    /* B staging: 8 toks x 8 quad-cols = 64 quads, ONE per thread.                \
       Source token = the block's slice + local token, clamped (p < 16). */      \
    const int bs_w4 = (tid & 7) << 2;                                              \
    const int bs_tl = tid >> 3;               /* local token 0..7 */              \
    int bs_tg = tok_base + bs_tl; bs_tg = bs_tg < p ? bs_tg : p - 1;              \
    const long bs_g = (long)bs_tg * qwpr + bs_w4;                                  \
    const unsigned int bs_sofs = (unsigned int)(bs_tl * 36) + (unsigned int)bs_w4;\
    const unsigned int lo_plane = 8u * 36u;                                        \
                                                                                  \
    /* accumulators: 1 mfrag x 1 nfrag x 4 regs, hi + lo. */                      \
    int dh[1][1][4];                                                              \
    int dl[1][1][4];                                                              \
    float o[1][1][4];                                                             \
    _Pragma("unroll")                                                             \
    for (int c = 0; c < 4; ++c) { o[0][0][c] = 0.0f; }                             \
                                                                                  \
    /* prologue: stage group 0 (A + B) into buf0, fetch group 1's A. */           \
    pf_pwa = pos_bits[a_offa];                                                    \
    pf_nwa = neg_bits[a_offa];                                                    \
    pf_pwb = pos_bits[a_offb];                                                    \
    pf_nwb = neg_bits[a_offb];                                                    \
    {                                                                             \
        const unsigned int t0a = pf_pwa & ~pf_nwa;                                \
        const unsigned int u0a = pf_nwa & ~pf_pwa;                                \
        _Pragma("unroll")                                                        \
        for (int q = 0; q < 8; ++q) {                                             \
            unsigned int tp = ((t0a >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0a >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                           \
            *(unsigned int*)(a_buf0 + s_ra * 132 + s_w4 * 32 + q * 4) = tp | up;  \
        }                                                                         \
        const unsigned int t0b = pf_pwb & ~pf_nwb;                                \
        const unsigned int u0b = pf_nwb & ~pf_pwb;                                \
        _Pragma("unroll")                                                        \
        for (int q = 0; q < 8; ++q) {                                             \
            unsigned int tp = ((t0b >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0b >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                           \
            *(unsigned int*)(a_buf0 + s_rb * 132 + s_w4 * 32 + q * 4) = tp | up;  \
        }                                                                         \
    }                                                                             \
    _Pragma("unroll")                                                            \
    for (int pl = 0; pl < 2; ++pl) {                                              \
        const unsigned int* const qp = pl ? q_lo_w : q_hi_w;                       \
        const unsigned int* const g4p = (const unsigned int*)(qp + bs_g);          \
        *(uint4*)(b_buf0 + (unsigned int)(pl * 288) + bs_sofs) =                   \
            *(const uint4*)g4p;                                                   \
    }                                                                             \
    if (1 < groups_per_row) {                                                     \
        pf_pwa = pos_bits[a_offa + 4];                                            \
        pf_nwa = neg_bits[a_offa + 4];                                            \
        pf_pwb = pos_bits[a_offb + 4];                                            \
        pf_nwb = neg_bits[a_offb + 4];                                            \
    }                                                                             \
                                                                                  \
    for (int grp = 0; grp < groups_per_row; ++grp) {                              \
        __syncthreads();   /* A(grp) + B(grp) staged & g-1 fully done */           \
        unsigned char* const buf = (grp & 1) ? a_buf1 : a_buf0;                  \
        unsigned char* const nbuf = (grp & 1) ? a_buf0 : a_buf1;                  \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                   \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;                   \
        const bool has_next = (grp + 1) < groups_per_row;                         \
        _Pragma("unroll")                                                        \
        for (int c = 0; c < 4; ++c) { dh[0][0][c] = 0; dl[0][0][c] = 0; }         \
        _Pragma("unroll")                                                        \
        for (int ks = 0; ks < 4; ++ks) {                                          \
            unsigned int aw[4];                                                   \
            const int r0 = warp_row + g_id;                                       \
            const unsigned char* p0 = buf + r0 * 132 + (ks * 32 + t_id * 4);      \
            const unsigned char* p8 = p0 + 8 * 132;                               \
            aw[0] = *(const unsigned int*)p0;                                      \
            aw[1] = *(const unsigned int*)p8;                                      \
            aw[2] = *(const unsigned int*)(p0 + 16);                               \
            aw[3] = *(const unsigned int*)(p8 + 16);                               \
            unsigned int bw[2][2];                                                \
            const unsigned int bb = (unsigned int)(g_id * 36)                      \
                + (unsigned int)(ks * 8 + t_id);                                  \
            bw[0][0] = bbuf[bb];            bw[0][1] = bbuf[bb + 4u];               \
            bw[1][0] = bbuf[lo_plane + bb];  bw[1][1] = bbuf[lo_plane + bb + 4u];  \
            mma_s8_m16n8k32(dh[0][0][0], dh[0][0][1], dh[0][0][2], dh[0][0][3],    \
                aw[0], aw[1], aw[2], aw[3], bw[0][0], bw[0][1]);                  \
            mma_s8_m16n8k32(dl[0][0][0], dl[0][0][1], dl[0][0][2], dl[0][0][3],    \
                aw[0], aw[1], aw[2], aw[3], bw[1][0], bw[1][1]);                  \
            if (has_next) {                                                       \
                if (ks == 0) {                                                    \
                    const unsigned int t0a = pf_pwa & ~pf_nwa;                    \
                    const unsigned int u0a = pf_nwa & ~pf_pwa;                    \
                    _Pragma("unroll")                                            \
                    for (int q = 0; q < 8; ++q) {                                 \
                        unsigned int tp = ((t0a >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0a >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;               \
                        *(unsigned int*)(nbuf + s_ra * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                             \
                    const unsigned int t0b = pf_pwb & ~pf_nwb;                    \
                    const unsigned int u0b = pf_nwb & ~pf_pwb;                    \
                    _Pragma("unroll")                                            \
                    for (int q = 0; q < 8; ++q) {                                 \
                        unsigned int tp = ((t0b >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0b >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;               \
                        *(unsigned int*)(nbuf + s_rb * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                             \
                }                                                                 \
                if (ks == 1) {                                                    \
                    const unsigned int* const qp = q_hi_w;                         \
                    *(uint4*)(nbbuf + bs_sofs) =                                  \
                        *(const uint4*)(qp + bs_g + (long)((grp + 1) << 5));       \
                } else if (ks == 2) {                                             \
                    const unsigned int* const qp = q_lo_w;                         \
                    *(uint4*)(nbbuf + lo_plane + bs_sofs) =                        \
                        *(const uint4*)(qp + bs_g + (long)((grp + 1) << 5));       \
                }                                                                 \
                if (ks == 2 && (grp + 2) < groups_per_row) {                      \
                    const long g4 = (long)(grp + 2) * 4;                           \
                    pf_pwa = pos_bits[a_offa + g4];                                \
                    pf_nwa = neg_bits[a_offa + g4];                                \
                    pf_pwb = pos_bits[a_offb + g4];                                \
                    pf_nwb = neg_bits[a_offb + g4];                                \
                }                                                                 \
            }                                                                     \
        }                                                                         \
        const float one128 = 0.0078125f;                                          \
        const int r0 = row_base + warp_row + g_id;                                 \
        const int rc0f = r0 < m ? r0 : m - 1;                                      \
        const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                          \
        const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];          \
        const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];          \
        _Pragma("unroll")                                                        \
        for (int c = 0; c < 4; ++c) {                                             \
            const float hi = (float)dh[0][0][c];                                   \
            const float lo = (float)dl[0][0][c];                                   \
            const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));                 \
            if (FUSED) {                                                          \
                o[0][0][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[0][0][c]);        \
            } else {                                                              \
                const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);                \
                o[0][0][c] = __fadd_rn(o[0][0][c], t3);                             \
            }                                                                     \
        }                                                                         \
    }                                                                             \
                                                                                  \
    /* epilogue: staged transpose then coalesced stores — same formula as        \
       v4 (out[tok * m + row] = o * s_t[tok]) over the block's 32r x 8t tile.  */ \
    float* stg = (float*)smem;                       /* [32][8] reused */        \
    __syncthreads();                                                              \
    const int r_in = warp_row + g_id;                                             \
    const int tk0 = t_id * 2;                   /* local, within the slice */     \
    stg[r_in * 8 + tk0]           = o[0][0][0];                                    \
    stg[r_in * 8 + tk0 + 1]       = o[0][0][1];                                    \
    stg[(r_in + 8) * 8 + tk0]     = o[0][0][2];                                    \
    stg[(r_in + 8) * 8 + tk0 + 1] = o[0][0][3];                                    \
    __syncthreads();                                                              \
    const int row0 = blockIdx.y * 32;                                             \
    for (int idx = tid; idx < 32 * 8; idx += 64) {                                 \
        const int r = idx & 31;                                                   \
        const int tkl = idx >> 5;               /* local token 0..7 */            \
        const int tk = tok_base + tkl;                                            \
        if (tk < p && row0 + r < m) {                                              \
            out[(long)tk * m + row0 + r] = stg[r * 8 + tkl] * s_t[tk];             \
        }                                                                         \
    }                                                                             \
}

GEMM_BODY_V4_SP8Y(gemm_i8_mma_tm32sp8y_fused, 1)
GEMM_BODY_V4_SP8Y(gemm_i8_mma_tm32sp8y_strict, 0)

// ---------------------------------------------------------------------------
// v5 (Arm 11): the traffic-halving generation — TWO shapes from one macro:
//   tm256v5  = ROWS=256, TOKS=64  — halves the B stream: grid.x halves, so
//              the per-M-block re-staging of the same tokens' q words halves
//              (the B side is the byte-dominant stream, 8-16x the A side at
//              the production shapes).
//   tm128v5t = ROWS=128, TOKS=128 — halves the A stream: grid.y halves, so
//              the weight bits are re-read half as often (the armed Arm-10
//              next step; ships bench-only via launch_gemm_gen).
// Structure: v4's B-slab staging + SINGLE-buffered A (the A double-buffer
// is traded for the wider tile — smem 70.7 KB / 90.6 KB). The A unpack for
// grp+1 moves AFTER a post-mma barrier: all warps' last A read is the ks=3
// aw load, so one extra __syncthreads() per group orders those reads before
// the grp+1 store (the fold below is register+global only and overlaps the
// staging stores). The pf(grp+2) prefetch moved with it — at v4's ks==2 it
// would clobber pf(grp+1) BEFORE the post-loop store consumes it.
// Warp tile is 32 rows x 32 tokens in BOTH shapes (mf=2, nf=4) — 2x the v4
// accumulators (dh/dl/o [2][4][4]); per-warp mma work doubles, blocks halve
// (tm256) or token coverage doubles (v5t). Bit-identity: identical fragment
// words, identical mma sequence + fold — only WHO loads WHAT changes.
// ---------------------------------------------------------------------------

#define GEMM_BODY_V5(NAME, FUSED, ROWS, TOKS)                                   \
extern "C" __global__ void __launch_bounds__(512, 1) NAME(                     \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */     \
    const unsigned int* __restrict__ neg_bits,                                  \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_hi_w,      /* [p * (n/4)] */            \
    const unsigned int* __restrict__ q_lo_w,                                     \
    const float* __restrict__ s_t,                /* [p] */                    \
    float* __restrict__ out,                      /* [p * m] */                \
    int m, int n, int p,                                                          \
    int words_per_row, int groups_per_row)                                        \
{                                                                                \
    static_assert(TOKS / ((512 / ROWS) * 8) == 4, "v5 warp tile must be 32x32"); \
    extern __shared__ __align__(16) unsigned char smem[];                        \
    unsigned char* const a_buf = smem;                    /* [ROWS][132] single */\
    /* B slabs: [2 planes][TOKS toks][36 words] u32, double-buffered. */          \
    unsigned int* const b_buf0 = (unsigned int*)(smem + ROWS * 132);             \
    unsigned int* const b_buf1 = b_buf0 + (2 * TOKS * 36);                       \
                                                                                 \
    const int tid = threadIdx.x;                                                 \
    const int lane = tid & 31;                                                   \
    const int wid = tid >> 5;                    /* 0..15 */                     \
    const int g_id = lane >> 2;                                                  \
    const int t_id = lane & 3;                                                   \
    const int tok_warps = 512 / ROWS;            /* 4 (ROWS=128) / 2 (256) */   \
    const int tpw = TOKS / tok_warps;            /* tokens per warp = 32 */     \
    const int warp_row = (wid / tok_warps) * 32;                                 \
    const int warp_n = wid % tok_warps;                                          \
    const int row_base = blockIdx.x * ROWS + warp_row;                           \
    const int blk_rows = blockIdx.x * ROWS;                                      \
    const int qwpr = n >> 2;                                                     \
                                                                                 \
    /* A staging ownership: (ROWS/128) x one (row, input-word) slot per thread. */\
    const int s_w4 = tid & 3;                                                    \
    const int s_r0 = tid >> 2;               /* 0..127 */                        \
    long a_off[ROWS / 128];                                                       \
    _Pragma("unroll")                                                            \
    for (int ap = 0; ap < ROWS / 128; ++ap) {                                    \
        int rc = blk_rows + ap * 128 + s_r0; rc = rc < m ? rc : m - 1;           \
        a_off[ap] = (long)rc * words_per_row + s_w4;                             \
    }                                                                            \
    unsigned int pf_pw[ROWS / 128];                                              \
    unsigned int pf_nw[ROWS / 128];                                              \
    /* B staging ownership: each thread copies one 16B quad per plane per        */\
    /* 64-token sub-block per group (sub-block base offset bp * 2304 words). */    \
    const int bs_w4 = (tid & 7) << 2;                                            \
    const int bs_t0 = tid >> 3;               /* 0..63 */                        \
    const unsigned int bs_sofs = (unsigned int)(bs_t0 * 36) + (unsigned int)bs_w4;\
    long bs_g[TOKS / 64];                                                         \
    _Pragma("unroll")                                                            \
    for (int bp = 0; bp < TOKS / 64; ++bp) {                                     \
        int tr = (int)(blockIdx.y * TOKS) + bp * 64 + bs_t0;                     \
        tr = tr < p ? tr : p - 1;                                                \
        bs_g[bp] = (long)tr * qwpr + bs_w4;                                      \
    }                                                                            \
                                                                                 \
    /* accumulators: 2 mfrag x 4 nfrag x 4 regs, hi + lo. */                     \
    int dh[2][4][4];                                                             \
    int dl[2][4][4];                                                             \
    float o[2][4][4];                                                            \
    _Pragma("unroll")                                                            \
    for (int mf = 0; mf < 2; ++mf) {                                             \
        _Pragma("unroll")                                                        \
        for (int nf = 0; nf < 4; ++nf) {                                         \
            _Pragma("unroll")                                                    \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }                  \
        }                                                                        \
    }                                                                            \
                                                                                 \
    /* prologue: stage group 0 (A + B) into the buf0 slabs, prefetch grp 1's A. */\
    _Pragma("unroll")                                                            \
    for (int ap = 0; ap < ROWS / 128; ++ap) {                                    \
        pf_pw[ap] = pos_bits[a_off[ap]];                                         \
        pf_nw[ap] = neg_bits[a_off[ap]];                                         \
    }                                                                            \
    _Pragma("unroll")                                                            \
    for (int ap = 0; ap < ROWS / 128; ++ap) {                                    \
        const unsigned int t0 = pf_pw[ap] & ~pf_nw[ap];                          \
        const unsigned int u0 = pf_nw[ap] & ~pf_pw[ap];                          \
        const int st_r = ap * 128 + s_r0;                                        \
        _Pragma("unroll")                                                        \
        for (int q = 0; q < 8; ++q) {                                            \
            unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                         \
            *(unsigned int*)(a_buf + st_r * 132 + s_w4 * 32 + q * 4) = tp | up;  \
        }                                                                        \
    }                                                                            \
    _Pragma("unroll")                                                            \
    for (int bp = 0; bp < TOKS / 64; ++bp) {                                     \
        _Pragma("unroll")                                                        \
        for (int pl = 0; pl < 2; ++pl) {                                         \
            const unsigned int* const qp = pl ? q_lo_w : q_hi_w;                 \
            *(uint4*)(b_buf0 + (unsigned int)(pl * TOKS * 36)                    \
                      + (unsigned int)(bp * 2304) + bs_sofs)                     \
                = *(const uint4*)(qp + bs_g[bp]);                                \
        }                                                                        \
    }                                                                            \
    if (1 < groups_per_row) {                                                    \
        _Pragma("unroll")                                                        \
        for (int ap = 0; ap < ROWS / 128; ++ap) {                                \
            pf_pw[ap] = pos_bits[a_off[ap] + 4];                                 \
            pf_nw[ap] = neg_bits[a_off[ap] + 4];                                 \
        }                                                                        \
    }                                                                            \
                                                                                 \
    for (int grp = 0; grp < groups_per_row; ++grp) {                             \
        __syncthreads();   /* A(grp) + B(grp) staged & g-1 fully done */           \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                  \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;                 \
        const bool has_next = (grp + 1) < groups_per_row;                        \
        const unsigned int lo_plane = (unsigned int)(TOKS * 36);                 \
        _Pragma("unroll")                                                        \
        for (int mf = 0; mf < 2; ++mf) {                                         \
            _Pragma("unroll")                                                    \
            for (int nf = 0; nf < 4; ++nf) {                                     \
                _Pragma("unroll")                                                \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; dl[mf][nf][c] = 0; }\
            }                                                                    \
        }                                                                        \
        _Pragma("unroll")                                                        \
        for (int ks = 0; ks < 4; ++ks) {                                         \
            unsigned int aw[2][4];                                               \
            _Pragma("unroll")                                                    \
            for (int mf = 0; mf < 2; ++mf) {                                     \
                const int r0 = warp_row + mf * 16 + g_id;                        \
                const unsigned char* p0 = a_buf + r0 * 132 + (ks * 32 + t_id * 4);\
                const unsigned char* p8 = p0 + 8 * 132;                          \
                aw[mf][0] = *(const unsigned int*)p0;                            \
                aw[mf][1] = *(const unsigned int*)p8;                            \
                aw[mf][2] = *(const unsigned int*)(p0 + 16);                     \
                aw[mf][3] = *(const unsigned int*)(p8 + 16);                     \
            }                                                                    \
            unsigned int bw[4][2][2];                                            \
            _Pragma("unroll")                                                    \
            for (int nf = 0; nf < 4; ++nf) {                                     \
                /* slab slots are BLOCK-RELATIVE tokens; the staging already    */\
                /* clamped the source row to p-1. */                             \
                const unsigned int bb = (unsigned int)(warp_n * tpw + nf * 8 + g_id) * 36u\
                    + (unsigned int)(ks * 8 + t_id);                             \
                bw[nf][0][0] = bbuf[bb];          bw[nf][0][1] = bbuf[bb + 4u];  \
                bw[nf][1][0] = bbuf[lo_plane + bb]; bw[nf][1][1] = bbuf[lo_plane + bb + 4u];\
            }                                                                    \
            _Pragma("unroll")                                                    \
            for (int mf = 0; mf < 2; ++mf) {                                     \
                _Pragma("unroll")                                                \
                for (int nf = 0; nf < 4; ++nf) {                                 \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1], dh[mf][nf][2], dh[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],              \
                        bw[nf][0][0], bw[nf][0][1]);                             \
                    mma_s8_m16n8k32(dl[mf][nf][0], dl[mf][nf][1], dl[mf][nf][2], dl[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],              \
                        bw[nf][1][0], bw[nf][1][1]);                             \
                }                                                                \
            }                                                                    \
            if (has_next) {                                                      \
                /* B stage for grp+1: one 16B quad per plane per sub-block at    */\
                /* ks1/ks2 (writes nbbuf, the slab grp-1 read — every warp        */\
                /* finished grp-1 at the loop-top barrier). */                    \
                if (ks == 1) {                                                   \
                    _Pragma("unroll")                                            \
                    for (int bp = 0; bp < TOKS / 64; ++bp) {                     \
                        *(uint4*)(nbbuf + (unsigned int)(bp * 2304) + bs_sofs) = \
                            *(const uint4*)(q_hi_w + bs_g[bp] + (long)((grp + 1) << 5));\
                    }                                                            \
                } else if (ks == 2) {                                            \
                    _Pragma("unroll")                                            \
                    for (int bp = 0; bp < TOKS / 64; ++bp) {                     \
                        *(uint4*)(nbbuf + lo_plane + (unsigned int)(bp * 2304) + bs_sofs) =\
                            *(const uint4*)(q_lo_w + bs_g[bp] + (long)((grp + 1) << 5));\
                    }                                                            \
                }                                                                \
            }                                                                    \
        }                                                                        \
        /* A is single-buffered: all warps' last A read was the ks=3 aw load;   */\
        /* one extra barrier orders those reads before the grp+1 store. The      */\
        /* fold below is register+global only and overlaps the staging stores.   */\
        __syncthreads();                                                         \
        if (has_next) {                                                          \
            _Pragma("unroll")                                                    \
            for (int ap = 0; ap < ROWS / 128; ++ap) {                            \
                const unsigned int t0 = pf_pw[ap] & ~pf_nw[ap];                  \
                const unsigned int u0 = pf_nw[ap] & ~pf_pw[ap];                  \
                const int st_r = ap * 128 + s_r0;                                \
                _Pragma("unroll")                                                \
                for (int q = 0; q < 8; ++q) {                                    \
                    unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                    unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                    up |= up << 1; up |= up << 2; up |= up << 4;                 \
                    *(unsigned int*)(a_buf + st_r * 132 + s_w4 * 32 + q * 4) = tp | up;\
                }                                                                \
            }                                                                    \
            if (grp + 2 < groups_per_row) {                                      \
                _Pragma("unroll")                                                \
                for (int ap = 0; ap < ROWS / 128; ++ap) {                        \
                    const long g4 = (long)(grp + 2) * 4;                         \
                    pf_pw[ap] = pos_bits[a_off[ap] + g4];                        \
                    pf_nw[ap] = neg_bits[a_off[ap] + g4];                        \
                }                                                                \
            }                                                                    \
        }                                                                        \
        const float one128 = 0.0078125f;                                         \
        _Pragma("unroll")                                                        \
        for (int mf = 0; mf < 2; ++mf) {                                         \
            const int r0 = row_base + mf * 16 + g_id;                            \
            const int rc0f = r0 < m ? r0 : m - 1;                                \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                    \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];    \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];    \
            _Pragma("unroll")                                                    \
            for (int nf = 0; nf < 4; ++nf) {                                     \
                _Pragma("unroll")                                                \
                for (int c = 0; c < 4; ++c) {                                    \
                    const float hi = (float)dh[mf][nf][c];                       \
                    const float lo = (float)dl[mf][nf][c];                       \
                    const float t2 = __fadd_rn(hi, __fmul_rn(lo, one128));       \
                    if (FUSED) {                                                 \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, t2, o[mf][nf][c]);\
                    } else {                                                     \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, t2);      \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);              \
                    }                                                            \
                }                                                                \
            }                                                                    \
        }                                                                        \
    }                                                                            \
                                                                                 \
    /* epilogue: 32-row chunks; warp owns rows only in its chunk. */             \
    float* stg = (float*)smem;                        /* [32][TOKS] reused */   \
    const int tok_blk = blockIdx.y * TOKS;                                       \
    _Pragma("unroll")                                                            \
    for (int ch = 0; ch < ROWS / 32; ++ch) {                                     \
        __syncthreads();                                                         \
        _Pragma("unroll")                                                        \
        for (int mf = 0; mf < 2; ++mf) {                                         \
            if ((warp_row + mf * 16) / 32 == ch) {                               \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;            \
                _Pragma("unroll")                                                \
                for (int nf = 0; nf < 4; ++nf) {                                 \
                    const int tk0 = warp_n * tpw + nf * 8 + t_id * 2;            \
                    stg[r_in * TOKS + tk0]         = o[mf][nf][0];               \
                    stg[r_in * TOKS + tk0 + 1]     = o[mf][nf][1];               \
                    stg[(r_in + 8) * TOKS + tk0]     = o[mf][nf][2];             \
                    stg[(r_in + 8) * TOKS + tk0 + 1] = o[mf][nf][3];             \
                }                                                                \
            }                                                                    \
        }                                                                        \
        __syncthreads();                                                         \
        const int row0 = blockIdx.x * ROWS + ch * 32;                            \
        for (int idx = tid; idx < 32 * TOKS; idx += 512) {                       \
            const int r = idx & 31;                                              \
            const int tk = idx >> 5;                                             \
            const int tok = tok_blk + tk;                                        \
            if (tok < p && row0 + r < m) {                                       \
                out[(long)tok * m + row0 + r] = stg[r * TOKS + tk] * s_t[tok];   \
            }                                                                    \
        }                                                                        \
    }                                                                            \
}

GEMM_BODY_V5(gemm_i8_mma_tm256v5_fused, 1, 256, 64)
GEMM_BODY_V5(gemm_i8_mma_tm256v5_strict, 0, 256, 64)
GEMM_BODY_V5(gemm_i8_mma_tm128v5t_fused, 1, 128, 128)
GEMM_BODY_V5(gemm_i8_mma_tm128v5t_strict, 0, 128, 128)
"#;

/// Issue 884 T2a — the SINGLE-PLANE activation-quant module (opt-in
/// `prefill_q8_act`). Self-contained (its own helpers) so it compiles as a
/// second NVRTC module next to [`GEMM_MMA_CUDA_SRC`].
///
/// ## What changes vs the hi/lo lane
///
/// The shipping quantize decomposes each activation as
/// `x ≈ s·(qh + ql/128)` (per-token abs-max scale, hi ∈ [-127,127],
/// lo ∈ [-64,64]) and the GEMM executes TWO `mma.m16n8k32.s8` passes per
/// k-step — one per plane — folding `o += sw·((f32)hi + (f32)lo/128)`.
/// B877/T1 located the 2.2-2.4× prefill GEMM gap vs the pinned opponent
/// (llama.cpp PQ2_0 MMQ) in exactly that 2× mma work, not kernel
/// efficiency. These kernels drop the lo plane: `x ≈ s·qh` (the fork's
/// Q8_1-class single plane), ONE mma pass, fold `o += sw·(f32)hi`.
///
/// The scale pipeline is deliberately IDENTICAL to the hi/lo lane
/// (absmax/127, same div form, same rne rounding, same clamp) — qh equals
/// the hi plane bit-for-bit, so an A/B isolates the plane-2 removal alone.
/// This is a NUMERICS REVISION (lossy surface, Issue 750 T3 rule): G1 pins
/// move; promotion requires the NLL-by-position + greedy-drift gates
/// (Issue 884 T2a) before any re-baseline.
#[cfg(feature = "prefill_q8_act")]
pub(crate) const GEMM_Q8_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// helpers (the GEMM_MMA_CUDA_SRC set, minus sign_word — the q8 module never
// unpacks weight bitplanes into i8; it reads the same SWAR-staged smem bytes)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float rne_f32(float x)
{
    float r;
    asm("cvt.rni.f32.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_s8_m16n8k32(
    int& d0, int& d1, int& d2, int& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float div_rn(float a, float b) { return a / b; }
__device__ __forceinline__ float div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float div_approx(float a, float b)
{
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Single-plane quantize: the QBODY pass-1 absmax + scale VERBATIM (s_out is
// bit-identical to the hi/lo kernel at the same div form), pass 2 packing
// only the rne(x/s) clamp [-127,127] words. q_lo is never computed.
#define QBODY_Q8(NAME, DIV)                                                    \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ input,   /* [p * n] */                            \
    unsigned int* __restrict__ q_w,    /* [p * (n/4)] */                        \
    float* __restrict__ s_out,         /* [p] */                                \
    int n)                                                                      \
{                                                                              \
    __shared__ float red[128];                                                 \
    const int row = blockIdx.x;                                                \
    const int tid = threadIdx.x;                                               \
    const long base = (long)row * n;                                           \
                                                                               \
    float mx = 0.0f;                                                           \
    for (int c = tid; c < n; c += 128) {                                       \
        float v = input[base + c];                                             \
        float a = v < 0.0f ? -v : v;                                           \
        mx = a > mx ? a : mx;                                                  \
    }                                                                          \
    red[tid] = mx;                                                             \
    __syncthreads();                                                           \
    float m = 0.0f;                                                            \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < 128; ++i) {                                            \
        m = red[i] > m ? red[i] : m;                                           \
    }                                                                          \
    const float s = m > 0.0f ? DIV(m, 127.0f) : 1.0f;                          \
    if (tid == 0) s_out[row] = s;                                              \
                                                                               \
    const int words = n / 4;                                                   \
    for (int w = tid; w < words; w += 128) {                                   \
        unsigned int wh = 0u;                                                  \
        _Pragma("unroll")                                                      \
        for (int j = 0; j < 4; ++j) {                                          \
            float x = input[base + (long)w * 4 + j];                           \
            float xf = DIV(x, s);                                              \
            float qh = rne_f32(xf);                                            \
            qh = qh > 127.0f ? 127.0f : (qh < -127.0f ? -127.0f : qh);         \
            wh |= ((unsigned int)(int)qh & 0xFFu) << (8 * j);                  \
        }                                                                      \
        q_w[(long)row * words + w] = wh;                                       \
    }                                                                          \
}

QBODY_Q8(quantize_rows_i8_q8_cuda_rn, div_rn)
QBODY_Q8(quantize_rows_i8_q8_cuda_full, div_full)
QBODY_Q8(quantize_rows_i8_q8_cuda_approx, div_approx)

// ---------------------------------------------------------------------------
// The single-plane v4 GEMM (the v4 B-stage generation, lo pass dropped).
// Identical to GEMM_BODY_V4 except:
//   - no q_lo_w param; the B slab is ONE plane ([64 toks][36 words],
//     double-buffered) — smem halves on the B side (52224 vs 70656 B),
//   - accumulators dh only; ONE mma per (mf, nf, ks),
//   - fold `o += sw * (f32)hi` (Fused fmaf / Strict mul+add — the same
//     FoldMode selector; the hi/lo +lo/128 term is gone with the plane).
// Everything else — A SWAR staging, fragment mappings, group order, s_t
// epilogue — is the v4 text verbatim.
// ---------------------------------------------------------------------------

#define GEMM_BODY_V4_Q8(NAME, FUSED)                                           \
extern "C" __global__ void __launch_bounds__(512, 1) NAME(                    \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */    \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */           \
    const unsigned int* __restrict__ q_w,         /* [p * (n/4)] */           \
    const float* __restrict__ s_t,                /* [p] */                   \
    float* __restrict__ out,                      /* [p * m] */               \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char smem[];                      \
    unsigned char* const a_buf0 = smem;                                        \
    unsigned char* const a_buf1 = smem + (128 * 132);                          \
    unsigned int* const b_buf0 = (unsigned int*)(smem + 2 * (128 * 132));      \
    unsigned int* const b_buf1 = b_buf0 + (64 * 36);                           \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                    /* 0..15 */                   \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid >> 2) * 32;         /* 4 row-bands of 32 */      \
    const int warp_n = wid & 3;                   /* 4 token-warps x 16 */     \
    const int tok_base = blockIdx.y * 64 + warp_n * 16;                        \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
    (void)tok_base;                                                            \
                                                                               \
    const int s_w4 = tid & 3;                                                  \
    const int s_r0 = tid >> 2;               /* 0..127 */                      \
    int rc0 = blk_rows + s_r0; rc0 = rc0 < m ? rc0 : m - 1;                    \
    const long a_off0 = (long)rc0 * words_per_row + s_w4;                      \
    unsigned int pf_pw0, pf_nw0;                                               \
    const int bs_w4 = (tid & 7) << 2;                                           \
    const int bs_t0 = tid >> 3;               /* 0..63 */                       \
    int bs_tr = (int)(blockIdx.y * 64) + bs_t0; bs_tr = bs_tr < p ? bs_tr : p - 1;\
    const long bs_g = (long)bs_tr * qwpr + bs_w4;                              \
    const unsigned int bs_sofs = (unsigned int)(bs_t0 * 36) + (unsigned int)bs_w4;\
                                                                               \
    /* accumulators: 2 mfrag x 2 nfrag x 4 regs (single plane) */              \
    int dh[2][2][4];                                                           \
    float o[2][2][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 2; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }                \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: stage group 0 (A + B) into buf0, fetch group 1's A. */        \
    pf_pw0 = pos_bits[a_off0];                                                 \
    pf_nw0 = neg_bits[a_off0];                                                 \
    {                                                                          \
        const unsigned int t0 = pf_pw0 & ~pf_nw0;                              \
        const unsigned int u0 = pf_nw0 & ~pf_pw0;                              \
        _Pragma("unroll")                                                      \
        for (int q = 0; q < 8; ++q) {                                          \
            unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
            up |= up << 1; up |= up << 2; up |= up << 4;                       \
            *(unsigned int*)(a_buf0 + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
        }                                                                      \
    }                                                                          \
    {                                                                          \
        const unsigned int* const g4p = (const unsigned int*)(q_w + bs_g);     \
        *(uint4*)(b_buf0 + bs_sofs) = *(const uint4*)g4p;                      \
    }                                                                          \
    if (1 < groups_per_row) {                                                  \
        pf_pw0 = pos_bits[a_off0 + 4];                                         \
        pf_nw0 = neg_bits[a_off0 + 4];                                         \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        __syncthreads();   /* A(grp) + B(grp) staged & g-1 fully done */        \
        unsigned char* const buf = (grp & 1) ? a_buf1 : a_buf0;               \
        unsigned char* const nbuf = (grp & 1) ? a_buf0 : a_buf1;              \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[2][4];                                             \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                const int r0 = warp_row + mf * 16 + g_id;                      \
                const unsigned char* p0 = buf + r0 * 132 + (ks * 32 + t_id * 4);\
                const unsigned char* p8 = p0 + 8 * 132;                        \
                aw[mf][0] = *(const unsigned int*)p0;                           \
                aw[mf][1] = *(const unsigned int*)p8;                           \
                aw[mf][2] = *(const unsigned int*)(p0 + 16);                   \
                aw[mf][3] = *(const unsigned int*)(p8 + 16);                   \
            }                                                                  \
            unsigned int bw[2][2];                                             \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                const unsigned int bb = (unsigned int)(warp_n * 16 + nf * 8 + g_id) * 36u\
                    + (unsigned int)(ks * 8 + t_id);                           \
                bw[nf][0] = bbuf[bb];          bw[nf][1] = bbuf[bb + 4u];      \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1], dh[mf][nf][2], dh[mf][nf][3],\
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][0], bw[nf][1]);                                 \
                }                                                              \
            }                                                                  \
            if (has_next) {                                                    \
                if (ks == 0) {                                                 \
                    const unsigned int t0 = pf_pw0 & ~pf_nw0;                  \
                    const unsigned int u0 = pf_nw0 & ~pf_pw0;                  \
                    _Pragma("unroll")                                          \
                    for (int q = 0; q < 8; ++q) {                              \
                        unsigned int tp = ((t0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        unsigned int up = ((u0 >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
                        up |= up << 1; up |= up << 2; up |= up << 4;           \
                        *(unsigned int*)(nbuf + s_r0 * 132 + s_w4 * 32 + q * 4) = tp | up;\
                    }                                                          \
                }                                                              \
                /* B stage for grp+1: one 16B quad at ks1 (single plane). */    \
                if (ks == 1) {                                                 \
                    *(uint4*)(nbbuf + bs_sofs) =                               \
                        *(const uint4*)(q_w + bs_g + (long)((grp + 1) << 5));   \
                }                                                              \
                if (ks == 2 && (grp + 2) < groups_per_row) {                   \
                    const long g4 = (long)(grp + 2) * 4;                       \
                    pf_pw0 = pos_bits[a_off0 + g4];                            \
                    pf_nw0 = neg_bits[a_off0 + g4];                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) {                                  \
                    const float hi = (float)dh[mf][nf][c];                     \
                    if (FUSED) {                                               \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, hi, o[mf][nf][c]);\
                    } else {                                                   \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, hi);    \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: identical to v4. */                                           \
    float* stg = (float*)smem;                       /* [32][64] reused */    \
    const int tok_blk = blockIdx.y * 64;                                       \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    const int tk0 = warp_n * 16 + nf * 8 + t_id * 2;           \
                    stg[r_in * 64 + tk0]     = o[mf][nf][0];                   \
                    stg[r_in * 64 + tk0 + 1] = o[mf][nf][1];                   \
                    stg[(r_in + 8) * 64 + tk0]     = o[mf][nf][2];             \
                    stg[(r_in + 8) * 64 + tk0 + 1] = o[mf][nf][3];             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * 64; idx += 512) {                       \
            const int tk = idx >> 5;                                           \
            const int r = idx & 31;                                            \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * 64 + tk] * s_t[tok];   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

GEMM_BODY_V4_Q8(gemm_i8_mma_tm128v4_q8_fused, 1)
GEMM_BODY_V4_Q8(gemm_i8_mma_tm128v4_q8_strict, 0)
"#;

/// Error type for the raw-CUDA mma GEMM.
#[derive(Debug)]
pub enum GemmI8MmaError {
    Compile(String),
    Load(String),
    Launch(String),
}

impl fmt::Display for GemmI8MmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmI8MmaError::Compile(s) => write!(f, "nvrtc compile failed: {s}"),
            GemmI8MmaError::Load(s) => write!(f, "module/function load failed: {s}"),
            GemmI8MmaError::Launch(s) => write!(f, "kernel launch failed: {s}"),
        }
    }
}

impl Error for GemmI8MmaError {}

/// Quantized-activation scratch (allocated once, reused across launches —
/// mirrors the CubeCL launcher's persistent handles; allocation is NOT
/// graph-capture safe, deliberately: this POC never runs under capture).
pub struct GemmI8MmaScratch {
    /// `[p * (n/4)]` packed hi words.
    pub q_hi_w: CudaSlice<u32>,
    /// `[p * (n/4)]` packed lo words.
    pub q_lo_w: CudaSlice<u32>,
    /// `[p]` per-token scales.
    pub s_t: CudaSlice<f32>,
}

/// The raw-CUDA int8 tensor-core ternary GEMM (Arm 6 POC).
pub struct GemmTernaryI8MmaCuda {
    quantize_rn: CudaFunction,
    quantize_full: CudaFunction,
    quantize_approx: CudaFunction,
    /// Selected division form for `launch_quantize` (the default `Rn` can
    /// be flipped to match the shipping GPU's OpFDiv lowering — see module
    /// doc; atomic so the struct stays Sync behind the Arc-shared ctx).
    quant_div: std::sync::atomic::AtomicU8,
    gemm_tm128_fused: CudaFunction,
    gemm_tm128_strict: CudaFunction,
    gemm_tm64_fused: CudaFunction,
    gemm_tm64_strict: CudaFunction,
    gemm_tm128v2_fused: CudaFunction,
    gemm_tm128v2_strict: CudaFunction,
    gemm_tm64v2_fused: CudaFunction,
    gemm_tm64v2_strict: CudaFunction,
    /// Occupancy-pinned probes (Arm 10 G2; not dispatched by the knob).
    probe_tm128v2b2: CudaFunction,
    probe_tm64v2b4: CudaFunction,
    gemm_tm128v3_fused: CudaFunction,
    gemm_tm128v3_strict: CudaFunction,
    gemm_tm128v4_fused: CudaFunction,
    gemm_tm128v4_strict: CudaFunction,
    /// Issue 742 T1 — the small-p (p ≤ 16) v4 twin: 128 threads, TM=32 ×
    /// TOKS=16, 17.7 KB smem (5 blocks/SM vs v4's 1) — fixes the small-p
    /// parallelism collapse (Bench 729). Bit-identical to v4 by construction.
    gemm_tm32sp_fused: CudaFunction,
    gemm_tm32sp_strict: CudaFunction,
    /// Issue 742 T1.4 — the y-split small-p GEMM (32r x 8t, 64 threads,
    /// grid (p/8, m/32)). **MEASURED NEGATIVE** — ×0.63-0.71 vs TM32sp at
    /// every shape (2× A weight traffic; see [`smallp_y8_enabled`]). Kept
    /// opt-in as the reproducible negative-result artifact.
    gemm_tm32sp8y_fused: CudaFunction,
    gemm_tm32sp8y_strict: CudaFunction,
    gemm_tm256v5_fused: CudaFunction,
    gemm_tm256v5_strict: CudaFunction,
    gemm_tm128v5t_fused: CudaFunction,
    gemm_tm128v5t_strict: CudaFunction,
    /// Issue 884 T2a — the single-plane activation-quant kernels (the
    /// [`GEMM_Q8_CUDA_SRC`] module; opt-in `prefill_q8_act`). `launch_gemm_q8`
    /// reads `scratch.q_hi_w` only; the q8 quantize never writes `q_lo_w`.
    #[cfg(feature = "prefill_q8_act")]
    quantize_q8_rn: CudaFunction,
    #[cfg(feature = "prefill_q8_act")]
    quantize_q8_full: CudaFunction,
    #[cfg(feature = "prefill_q8_act")]
    quantize_q8_approx: CudaFunction,
    #[cfg(feature = "prefill_q8_act")]
    gemm_tm128v4_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_q8_act")]
    gemm_tm128v4_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_q8_act")]
    _q8_module: Arc<CudaModule>,
    /// Issue 884 T2b — the fork-config v6-q8 GEMM (the `GEMM_MMQ_V2_CUDA_SRC`
    /// module; opt-in `prefill_mmq_v2`). Scheduling-only vs v4-q8: the unit
    /// gate pins the two generations bit-identical.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6_q8_strict: CudaFunction,
    /// The TOKS=128 tile (v6t): 512 threads, 16 warps, B slab 128 toks.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6t_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6t_q8_strict: CudaFunction,
    /// The TOKS=128 + double-A tile (v6d): v4's 1-barrier pipeline.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6d_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6d_q8_strict: CudaFunction,
    /// The ldmatrix twins (T2b continuation, Bench 880): v6t's tile with
    /// `ldmatrix.m8n8.x4` A-fragment loads (v6tl; 32 LDS.32/group/warp → 8
    /// ldmatrix) and + `ldmatrix.m8n8.x2` B loads (v6tb). Same arithmetic
    /// sequence — the unit gate pins them bit-identical to v4-q8.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6tl_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6tl_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6tb_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v6tb_q8_strict: CudaFunction,
    /// The fork-style global-A twins (T2b rung 3, Bench 881): NO A smem
    /// stage — mask words LDG'd per fragment, SWAR-expanded in registers
    /// (v7: TOKS=64/256thr/2 blocks, v7t: TOKS=128/512thr/1 block). Same
    /// arithmetic sequence — the unit gate pins them bit-identical to v4-q8.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v7_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v7_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v7t_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v7t_q8_strict: CudaFunction,
    /// The format-rung twins (T2b rung 4, Plan 572): v7's global-A structure
    /// over PACKED-2-BIT Q2_0-code weights + the PRMT-table fragment decode
    /// (8 ALU-pipe ops per 8 weights post-903-T2a — the fork's decode class;
    /// v8: TOKS=64/256thr/2 blocks, v8t: TOKS=128/512thr/1 block). Same arithmetic
    /// sequence — the unit gate pins them bit-identical to v4-q8.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v8_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v8_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v8t_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v8t_q8_strict: CudaFunction,
    /// The staged-packed twins (T2b rung 5, Plan 573): the v6tb staged
    /// structure over the packed codes — A cp.async'd into a double-buffered
    /// code stage (once per block; no token-warp re-read) + per-use
    /// `v8_decode_pair` (v9: TOKS=64/256thr/2 blocks, v9t: TOKS=128/512thr/1
    /// block). Same arithmetic sequence — the unit gate pins them
    /// bit-identical to v4-q8.
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v9_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v9_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v9t_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v9t_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v10_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v10_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v10t_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v10t_q8_strict: CudaFunction,
    /// Plan 597 rung A — the v10 body at MINB=3 (24 warps/SM, the
    /// `__launch_bounds__(256, 3)` 85-reg cap). Kernel-level probe; no
    /// production dispatch (the S1.e gate decides whether a tile-shrunk
    /// production variant follows). Read only by the rung-A probe test.
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    gemm_tm128v10o3_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    gemm_tm128v10o3_q8_strict: CudaFunction,
    /// Plan 597 ceiling probe — the MINB=3 body with decode stripped
    /// (garbage values, timing-only; the Issue-903 nodecode convention).
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) gemm_tm128v10o3_q8_nodecode: CudaFunction,
    /// Plan 597 v12 — the tile-shrunk body (64-row blocks, 16-tok warps,
    /// 24 warps/SM at the 85-reg cap). Probe arms pending the S1.e gate;
    /// no production dispatch until it clears.
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    gemm_tm64v12_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    gemm_tm64v12_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) gemm_tm64v12_q8_nodecode: CudaFunction,
    /// Issue 902 T1 — the fused gate+up pair kernels (v11gu: TOKS=64 /
    /// 256 thr / 2 blocks/SM; v11gut: TOKS=128 / 512 thr / 1 block/SM).
    /// One launch computes both same-shape FFN projections from one B
    /// (activation) tile stage; bit-identical per-slab to v10t by
    /// construction (same per-output op order).
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gu_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gu_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gut_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gut_q8_strict: CudaFunction,
    /// Bench 895 repair rungs: v11gs (sequential shared-dh, 2 blocks/SM),
    /// v11gq (interleaved at the 255-reg budget, 1 block/SM).
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gs_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gs_q8_strict: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gq_q8_fused: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    gemm_tm128v11gq_q8_strict: CudaFunction,
    /// Issue 903 T1 — the nodecode TIMING PROBE (v10t with the decode ALU
    /// ops stripped; VALUES GARBAGE BY DESIGN). Prices the decode's
    /// issue-slot share; never dispatched by any production path — read only
    /// by the `q8_gemm_v11_pair_perf_ab_ffn` test probe, hence lib-dead.
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) gemm_tm128v10t_q8_nodecode: CudaFunction,
    #[cfg(feature = "prefill_mmq_v2")]
    _mmq_v2_module: Arc<CudaModule>,
    _module: Arc<CudaModule>,
}

/// Division form for the quantize pre-pass (bit-identity selector — the
/// NVIDIA SPIR-V consumer's OpFDiv lowering is not div.rn; see module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantDiv {
    /// Plain IEEE `/` (div.rn.f32).
    Rn,
    /// PTX `div.full.f32` (the fast full-range form).
    Full,
    /// PTX `div.approx.f32`.
    Approx,
}

/// Tile variant for the GEMM dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaTile {
    /// 128 rows x 64 tokens (2 warp-rows).
    Tm128,
    /// 64 rows x 64 tokens (1 warp-row).
    Tm64,
    /// 256 rows x 64 tokens — the v5 B-traffic-halving tile (v5 only).
    Tm256,
}

/// Outer-fold FMA variant (bit-identity selector; see module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldMode {
    /// `o = __fmaf_rn(sw, t2, o)` — matches a Vulkan driver that contracted.
    Fused,
    /// Strict `__fmul_rn` + `__fadd_rn` — matches an uncontracted driver.
    Strict,
}

/// GEMM kernel generation selector (explicit form for A/B benches).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaGen {
    /// Arm-6 kernels (broadcast-load per-bit staging, 2 barriers/group).
    V1,
    /// Arm-10 kernels (SWAR staging, double-buffered group pipeline).
    V2,
    /// Arm-10 v3 kernels (512-thread TM=128, 32-row warp tiles; tm128 only).
    V3,
    /// Arm-10 v4 kernels (v3 shape + B smem staging; tm128 only).
    V4,
    /// Arm-11 v5 kernels — the traffic-halving generation: Tm256 = 256x64
    /// (halves the B stream), Tm128 = 128x128 `v5t` (halves the A stream).
    V5,
}

/// v2+ GEMM kernel selection (Issue 734 Arm 10, DEFAULT ON): the modern
/// mma kernels — v4 by default (512-thread TM=128, 32-row warp tiles, SWAR
/// A staging, double-buffered group pipeline, and the B slab staged once per
/// (block, group) in shared memory — the measured fix for the
/// B-L2-bandwidth wall: every v1/v2/v3 variant ran at its effective B
/// L2 traffic / ~1.65-1.7 TB/s; staging the 64-token slab shared across
/// the row-bands halves that traffic). Kernel 1.42× at ffn_gate
/// (150 vs 106 TFLOPS, Bench 724). With [`mma_v5_enabled`] (Arm 11) the
/// m ≥ 2048 shapes further select the v5 TM=256 kernel. `RIIR_GEMM_MMA_V2=0|false|off` selects
/// the Arm-6 v1 kernels at the caller's tile — an A/B knob only: BOTH
/// kernel generations are bit-identical to the same reference (0 bit-diffs
/// on ragged + non-disjoint + production fixtures ×2 folds), so the knob
/// changes instruction economics and scheduling, never numerics. The knob
/// rewrites Tm64 requests to Tm128 when enabled (v4 is the 512-thread
/// TM=128 shape; the off-state reproduces the exact prior shipping path).
pub fn mma_v2_enabled() -> bool {
    static V2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V2.get_or_init(|| {
        std::env::var("RIIR_GEMM_MMA_V2").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// v5 GEMM kernel selection (Issue 734 Arm 11, DEFAULT OFF — the traffic-
/// halving tile growth was REFUTED, Bench 725): the 512-thread register
/// cap (64K regs/SM → 128 regs/thread at 16 warps) forces spills on any
/// tile ≥2x v4's (the 96 accumulator regs alone exceed the budget):
/// tm256 measured 0.796x and tm128v5t 0.954x vs v4 at ffn_gate, both
/// bit-identical (G1 0/35.6M diffs). v4's 128x64 tile sits exactly AT the
/// register cap — tile growth is structurally closed; kept as the A/B
/// apparatus + negative-result artifact. When enabled AND `m >= 2048` the
/// dispatch selects `gemm_i8_mma_tm256v5_*` (TM=256, halves the B-side L2
/// stream); the 128-token twin `gemm_i8_mma_tm128v5t_*` halves the A stream.
/// Both are gate-proven bit-identical to the same v1 reference — the knob
/// changes instruction economics, never numerics.
pub fn mma_v5_enabled() -> bool {
    static V5: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V5.get_or_init(|| {
        std::env::var("RIIR_GEMM_MMA_V5").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    })
}

/// Smallest `m` that dispatches to the v5 TM=256 kernel (below it the extra
/// rows would waste more compute than the halved B traffic saves; the small
/// projections stay on v4).
const MMA_V5_MIN_M: usize = 2048;

/// Small-p GEMM kernel selection (Issue 742 T1, DEFAULT ON): `p <= 16`
/// dispatches to the v4-sp kernel (128 threads, TM=32 × TOKS=16, 17.7 KB
/// smem → 5 blocks/SM, grid m/32) instead of v4's 512-thread TM=128 shape
/// (70.7 KB smem → 1 block/SM, grid m/128). At p=16 v4's grid is 136 blocks
/// on 128 SMs and the lone resident block's barrier + epilogue bubbles are
/// never hidden — measured ~110 GB/s effective vs the ~700 GB/s floor
/// (Bench 729, the C_verify wall). v4-sp restores the stall interleaving.
/// Bit-identical to v4 by construction (same quantize scratch, same sign
/// bytes, same fragment mapping, same per-group fold order) — the knob
/// changes only scheduling, never numerics. `RIIR_PREFILL_SMALLP_GEMM=0|false|off`
/// selects v4 at every p (the A/B apparatus).
pub fn smallp_gemm_enabled() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("RIIR_PREFILL_SMALLP_GEMM").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// Issue 742 T1.4 — the y-split small-p GEMM knob. **MEASURED NEGATIVE
/// (2026-08-24, GPU-exclusive): y8 loses to TM32sp at every shape —
/// ×0.63-0.71** (down 103.4→162.8 µs, out_proj 33.8→47.9, ffn_gate
/// 63.7→101.3, qkv 39.4→55.1). Root cause: both x-blocks of a row tile
/// stream the SAME A rows — the weight reads DOUBLE (down: 2× 22.3 MB ≈
/// 274 GB/s effective on 44.6 MB, matching the measured 163 µs) and the
/// co-resident L2 dedup does not materialize; the 2× A cost swamps the
/// wave-balance gain (~1.33× — and the discrete-wave model itself was too
/// crude: TM32sp's 5-blocks/SM occupancy already softens the ragged tail).
/// DEFAULT OFF; `RIIR_PREFILL_SMALLP_Y8=1|true|on` opts in (the A/B
/// apparatus — kept with the bit-identity gates as the reproducible
/// negative-result artifact).
pub fn smallp_y8_enabled() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("RIIR_PREFILL_SMALLP_Y8").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    })
}

/// The routing predicate (only consulted when `RIIR_PREFILL_SMALLP_Y8=1`):
/// y8 iff the TM32sp grid `m/32` would be in 129..=192 AND p > 8. MEASURED
/// NEGATIVE at these exact shapes (see [`smallp_y8_enabled`]) — the window
/// is kept for the opt-in A/B apparatus only.
fn smallp_y8_routes(m: usize, p: usize) -> bool {
    let g32 = m.div_ceil(32);
    p > 8 && g32 > 128 && g32 <= 192
}

/// Issue 884 T2a — the SINGLE-PLANE activation-quant arm for the prefill
/// GEMM surface (**DEFAULT ON since Issue 917**, the owner-authorized
/// default-path promotion 2026-09-11; feature `prefill_q8_act` compiles the
/// surface, the env is the KILL-SWITCH: `RIIR_PREFILL_Q8_ACT=0|false|off`
/// disables — every other value, including unset, keeps the q8 route on.
/// Pre-917 the polarity was inverted: unset = off, `1|true|on` = on. When
/// on, the arm-resolving launchers
/// [`GemmTernaryI8MmaCuda::launch_prefill_quantize`] /
/// [`GemmTernaryI8MmaCuda::launch_prefill_gemm`] dispatch the q8 kernels
/// (one mma pass per k-step instead of the hi/lo pair) — a NUMERICS
/// REVISION, not a scheduling knob: the G1 argmax/FNV pins move, and the
/// accuracy record is the Issue-884 T2a NLL-by-position + greedy-drift
/// gates + the route bit-identity cells (the lossy-surface rule, Issue 750
/// T3; Benches 878/883/884).
/// The q8 GEMM is the v4 shape at every p (no smallp twin) — the A/B arms
/// must run under the DEFAULT kernel-generation knobs
/// (`mma_v2_enabled()` on, [`mma_v5_enabled`] off) so the anchor resolves
/// to v4/v4-sp exactly.
#[cfg(feature = "prefill_q8_act")]
pub fn q8_act_enabled() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        !std::env::var("RIIR_PREFILL_Q8_ACT").is_ok_and(|s| matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// Issue 884 T2b — the fork-config UTILIZATION arm for the prefill GEMM
/// surface (**DEFAULT ON since Issue 917**; feature `prefill_mmq_v2`
/// compiles the surface, the env is the KILL-SWITCH:
/// `RIIR_PREFILL_MMQ_V2=0|false|off` disables — every other value,
/// including unset, keeps the arm on). Purely a SCHEDULING knob — unlike
/// the q8 arm it moves no numerics: the v6 kernel (`gemm_i8_mma_tm128v6_q8_*`)
/// is the v4-q8 arithmetic sequence under the opponent's config class (warp
/// tile 32x32 so each A smem fragment load feeds 8 mma per k-step instead
/// of 4; XOR-perm swizzled A staging killing the ~3.2-way `g+t`
/// bank-conflict anti-diagonal; the B slab staged by `cp.async.cg` 16 B;
/// single-buffered A + double-buffered B at 34.8 KB, 2 blocks/SM via
/// `__launch_bounds__(256, 2)`). Requires the q8 arm (it is a q8-surface
/// kernel): the v6 dispatch fires only when BOTH [`q8_act_enabled`] and
/// this knob are on; either off means the exact prior path. G1 discipline:
/// with both knobs on, the greedy FNV / argmax equal the q8-alone run
/// (bit-identity, unit-gated); the anchor pins are untouched (the arm is
/// env-gated).
#[cfg(feature = "prefill_mmq_v2")]
pub fn mmq_v2_enabled() -> bool {
    static K: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        !std::env::var("RIIR_PREFILL_MMQ_V2").is_ok_and(|s| matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// Issue 884 T2b rung 4 (Plan 572) — the format-rung arm:
/// `RIIR_PREFILL_MMQ_FMT=1|2|3` selects the packed-A structure
/// (**DEFAULT 3 since Issue 917** — the v10t L2-traffic twin, the measured
/// winner: `0|false|off` disables the rung, every other value keeps the
/// default). Requires `q8_act_enabled() && mmq_v2_enabled()` to take
/// effect at dispatch. Moves no numerics (the v8 kernel is bit-identical
/// to v4-q8 by construction); it swaps the A weight surface to the
/// packed-2-bit Q2_0-code mirror ([`
/// crate::prefill_cuda_mma::pack_bitplanes_to_q2`]) + the PRMT-decode kernel.
///
/// Rung 5 (Plan 573) added the mode axis: `RIIR_PREFILL_MMQ_FMT=2` selects
/// the STAGED-PACKED structure (v9/v9t — A codes cp.async'd into a
/// double-buffered smem stage, per-use decode) instead of v8/v8t's global-A.
/// Rung 6 (Plan 574) added mode 3: the L2-traffic-repair twin (v10/v10t —
/// smem-staged group scales + transposed coalesced epilogue, ncu-guided).
/// All nonzero modes arm the same packed-only mirror policy
/// ([`mmq_fmt_route_enabled`]).
#[cfg(feature = "prefill_mmq_v2")]
pub fn mmq_fmt_mode() -> u8 {
    static K: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        match std::env::var("RIIR_PREFILL_MMQ_FMT")
            .unwrap_or_default()
            .trim()
        {
            "0" | "false" | "off" => 0,
            "1" | "true" | "on" => 1,
            "2" => 2,
            "3" => 3,
            // Issue 917: unset is the promoted default (v10t). An unknown
            // non-empty value is ALSO the default — kill-switch polarity
            // means only the explicit off-values disable.
            _ => 3,
        }
    })
}

/// The format-rung arm is engaged (mode 1 OR 2).
#[cfg(feature = "prefill_mmq_v2")]
pub fn mmq_fmt_enabled() -> bool {
    mmq_fmt_mode() != 0
}

/// The full format-rung route predicate (Plan 572): all three knobs engaged.
/// Decides the weight-mirror build policy — on the route, GEMM mirrors carry
/// packed codes ONLY (the pair upload is skipped).
#[cfg(feature = "prefill_mmq_v2")]
pub fn mmq_fmt_route_enabled() -> bool {
    q8_act_enabled() && mmq_v2_enabled() && mmq_fmt_enabled()
}

/// Issue 902 T1 — the fused gate+up pair arm: `RIIR_PREFILL_MMQ_GU`.
/// `1` v11gu (TOKS=64/2-block, interleaved — refuted), `2` v11gut
/// (TOKS=128/1-block, interleaved — refuted), `3` v11gs (TOKS=64/2-block,
/// sequential shared-dh), `4` v11gq (TOKS=64/1-block at the 255-reg budget,
/// interleaved). `0` (default) keeps the two-launch path. Requires the full
/// format-rung route ([`mmq_fmt_route_enabled`] — the pair kernel consumes
/// packed codes) and takes effect only at the FFN gate+up pair dispatch;
/// every other GEMM (qkv/z/a/b, down, wq/wkv, wte) is unchanged. Moves no
/// numerics: each output slab's op sequence is v10t verbatim (unit-gated
/// bit-identity), so the route pins are untouched while the knob is off —
/// and unchanged when it is on (the anchor pins are VALUE pins over
/// identical arithmetic).
#[cfg(feature = "prefill_mmq_v2")]
pub fn mmq_gu_mode() -> u8 {
    static K: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        match std::env::var("RIIR_PREFILL_MMQ_GU")
            .unwrap_or_default()
            .trim()
        {
            "4" => 4,
            "3" => 3,
            "2" => 2,
            "1" => 1,
            _ => 0,
        }
    })
}

impl GemmTernaryI8MmaCuda {
    /// Own context + stream (the `TernaryGemmCudaRaw::new()` precedent) —
    /// lets integration tests construct the stack without importing cudarc
    /// types (cudarc is not a dev-dependency).
    pub fn new_standalone() -> Result<(Self, std::sync::Arc<CudaStream>), GemmI8MmaError> {
        let ctx = CudaContext::new(0).map_err(|e| GemmI8MmaError::Load(e.to_string()))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| GemmI8MmaError::Load(e.to_string()))?;
        let kernels = Self::new(ctx)?;
        Ok((kernels, stream))
    }

    /// Compile (NVRTC, sm_89) + load all kernels against a shared context.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, GemmI8MmaError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            GEMM_MMA_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| GemmI8MmaError::Compile(format!("{e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| GemmI8MmaError::Load(e.to_string()))?;
        let load = |n: &str| -> Result<CudaFunction, GemmI8MmaError> {
            module
                .load_function(n)
                .map_err(|e| GemmI8MmaError::Load(format!("{n}: {e}")))
        };
        // Issue 884 T2a — the q8 module (opt-in `prefill_q8_act`): a second
        // NVRTC compile + load; the v4-q8 smem opt-in covers 2×128×132 A
        // staging + 2×64×36×4 single-plane B slabs = 52224 B (> the 48 KB
        // default, like v4's 70.7 KB).
        #[cfg(feature = "prefill_q8_act")]
        let (quantize_q8_rn, quantize_q8_full, quantize_q8_approx, gemm_tm128v4_q8_fused, gemm_tm128v4_q8_strict, _q8_module) = {
            let q8_ptx = cudarc::nvrtc::compile_ptx_with_opts(
                GEMM_Q8_CUDA_SRC,
                cudarc::nvrtc::CompileOptions {
                    arch: Some("sm_89"),
                    ..Default::default()
                },
            )
            .map_err(|e| GemmI8MmaError::Compile(format!("q8: {e}")))?;
            let q8_module = ctx
                .load_module(q8_ptx)
                .map_err(|e| GemmI8MmaError::Load(format!("q8: {e}")))?;
            let load_q8 = |n: &str| -> Result<CudaFunction, GemmI8MmaError> {
                q8_module
                    .load_function(n)
                    .map_err(|e| GemmI8MmaError::Load(format!("{n}: {e}")))
            };
            let (q8_fused, q8_strict) = (
                load_q8("gemm_i8_mma_tm128v4_q8_fused")?,
                load_q8("gemm_i8_mma_tm128v4_q8_strict")?,
            );
            for f in [&q8_fused, &q8_strict] {
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    128 * 132 * 2 + 2 * 64 * 36 * 4,
                )
                .map_err(|e| GemmI8MmaError::Load(format!("q8 smem opt-in: {e}")))?;
            }
            (
                load_q8("quantize_rows_i8_q8_cuda_rn")?,
                load_q8("quantize_rows_i8_q8_cuda_full")?,
                load_q8("quantize_rows_i8_q8_cuda_approx")?,
                q8_fused,
                q8_strict,
                q8_module,
            )
        };
        // Issue 884 T2b — the v6 module (opt-in `prefill_mmq_v2`): a third
        // NVRTC compile + load. 34,816 B dynamic smem (single 128x32-word A
        // stage + double 64x36-word B slabs) is under the 48 KB default —
        // no per-function smem opt-in (unlike v4/v5/q8).
        #[cfg(feature = "prefill_mmq_v2")]
        let (gemm_tm128v6_q8_fused, gemm_tm128v6_q8_strict, gemm_tm128v6t_q8_fused, gemm_tm128v6t_q8_strict, gemm_tm128v6d_q8_fused, gemm_tm128v6d_q8_strict, gemm_tm128v6tl_q8_fused, gemm_tm128v6tl_q8_strict, gemm_tm128v6tb_q8_fused, gemm_tm128v6tb_q8_strict, gemm_tm128v7_q8_fused, gemm_tm128v7_q8_strict, gemm_tm128v7t_q8_fused, gemm_tm128v7t_q8_strict, gemm_tm128v8_q8_fused, gemm_tm128v8_q8_strict, gemm_tm128v8t_q8_fused, gemm_tm128v8t_q8_strict, gemm_tm128v9_q8_fused, gemm_tm128v9_q8_strict, gemm_tm128v9t_q8_fused, gemm_tm128v9t_q8_strict, gemm_tm128v10_q8_fused, gemm_tm128v10_q8_strict, gemm_tm128v10t_q8_fused, gemm_tm128v10t_q8_strict, gemm_tm128v10o3_q8_fused, gemm_tm128v10o3_q8_strict, gemm_tm128v10o3_q8_nodecode, gemm_tm64v12_q8_fused, gemm_tm64v12_q8_strict, gemm_tm64v12_q8_nodecode, gemm_tm128v11gu_q8_fused, gemm_tm128v11gu_q8_strict, gemm_tm128v11gut_q8_fused, gemm_tm128v11gut_q8_strict, gemm_tm128v11gs_q8_fused, gemm_tm128v11gs_q8_strict, gemm_tm128v11gq_q8_fused, gemm_tm128v11gq_q8_strict, gemm_tm128v10t_q8_nodecode, _mmq_v2_module) = {
            let v6_ptx = cudarc::nvrtc::compile_ptx_with_opts(
                crate::gemm_ternary_i8_mma_v6_src::GEMM_MMQ_V2_CUDA_SRC,
                cudarc::nvrtc::CompileOptions {
                    arch: Some("sm_89"),
                    ..Default::default()
                },
            )
            .map_err(|e| GemmI8MmaError::Compile(format!("mmq_v2: {e}")))?;
            let v6_module = ctx
                .load_module(v6_ptx)
                .map_err(|e| GemmI8MmaError::Load(format!("mmq_v2: {e}")))?;
            let load_v6 = |n: &str| -> Result<CudaFunction, GemmI8MmaError> {
                v6_module
                    .load_function(n)
                    .map_err(|e| GemmI8MmaError::Load(format!("{n}: {e}")))
            };
            let (v6_fused, v6_strict) = (
                load_v6("gemm_i8_mma_tm128v6_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v6_q8_strict")?,
            );
            let (v6t_fused, v6t_strict) = (
                load_v6("gemm_i8_mma_tm128v6t_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v6t_q8_strict")?,
            );
            let (v6d_fused, v6d_strict) = (
                load_v6("gemm_i8_mma_tm128v6d_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v6d_q8_strict")?,
            );
            let (v6tl_fused, v6tl_strict) = (
                load_v6("gemm_i8_mma_tm128v6tl_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v6tl_q8_strict")?,
            );
            let (v6tb_fused, v6tb_strict) = (
                load_v6("gemm_i8_mma_tm128v6tb_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v6tb_q8_strict")?,
            );
            let (v7_fused, v7_strict) = (
                load_v6("gemm_i8_mma_tm128v7_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v7_q8_strict")?,
            );
            let (v7t_fused, v7t_strict) = (
                load_v6("gemm_i8_mma_tm128v7t_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v7t_q8_strict")?,
            );
            let (v8_fused, v8_strict) = (
                load_v6("gemm_i8_mma_tm128v8_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v8_q8_strict")?,
            );
            let (v8t_fused, v8t_strict) = (
                load_v6("gemm_i8_mma_tm128v8t_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v8t_q8_strict")?,
            );
            let (v9_fused, v9_strict) = (
                load_v6("gemm_i8_mma_tm128v9_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v9_q8_strict")?,
            );
            let (v9t_fused, v9t_strict) = (
                load_v6("gemm_i8_mma_tm128v9t_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v9t_q8_strict")?,
            );
            let (v10_fused, v10_strict) = (
                load_v6("gemm_i8_mma_tm128v10_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v10_q8_strict")?,
            );
            let (v10t_fused, v10t_strict) = (
                load_v6("gemm_i8_mma_tm128v10t_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v10t_q8_strict")?,
            );
            let (v10o3_fused, v10o3_strict) = (
                load_v6("gemm_i8_mma_tm128v10o3_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v10o3_q8_strict")?,
            );
            let v10o3_nodecode = load_v6("gemm_i8_mma_tm128v10o3_q8_nodecode")?;
            let (v12_fused, v12_strict) = (
                load_v6("gemm_i8_mma_tm64v12_q8_fused")?,
                load_v6("gemm_i8_mma_tm64v12_q8_strict")?,
            );
            let v12_nodecode = load_v6("gemm_i8_mma_tm64v12_q8_nodecode")?;
            let (v11gu_fused, v11gu_strict) = (
                load_v6("gemm_i8_mma_tm128v11gu_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v11gu_q8_strict")?,
            );
            let (v11gut_fused, v11gut_strict) = (
                load_v6("gemm_i8_mma_tm128v11gut_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v11gut_q8_strict")?,
            );
            let (v11gs_fused, v11gs_strict) = (
                load_v6("gemm_i8_mma_tm128v11gs_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v11gs_q8_strict")?,
            );
            let (v11gq_fused, v11gq_strict) = (
                load_v6("gemm_i8_mma_tm128v11gq_q8_fused")?,
                load_v6("gemm_i8_mma_tm128v11gq_q8_strict")?,
            );
            let v10t_nodecode = load_v6("gemm_i8_mma_tm128v10t_q8_nodecode")?;
            // v6t's B slab is 128 toks: 16384 + 2*128*36*4 = 53,248 B, and
            // v6d adds the second A stage: 69,632 B — both over the 48 KB
            // default (v6's 34,816 B is not). The ldmatrix twins share v6t's
            // tile (53,248 B).
            for (f, bytes) in [
                (&v6t_fused, 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6t_strict, 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6d_fused, 2 * 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6d_strict, 2 * 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6tl_fused, 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6tl_strict, 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6tb_fused, 128 * 32 * 4 + 2 * 128 * 36 * 4),
                (&v6tb_strict, 128 * 32 * 4 + 2 * 128 * 36 * 4),
                // v9t's code stage is 2 x 128 x 12 u32 (double-buffered,
                // 12-word rows): 49,152 B total with the B slabs — over the
                // 48 KB default (v9's 30,720 B is not). v10t adds the
                // double-buffered 128-float scale stage: 50,176 B.
                (&v9t_fused, 2 * 128 * 12 * 4 + 2 * 128 * 36 * 4),
                (&v9t_strict, 2 * 128 * 12 * 4 + 2 * 128 * 36 * 4),
                (&v10t_fused, 2 * 128 * 12 * 4 + 2 * 128 * 36 * 4 + 2 * 128 * 4),
                (&v10t_strict, 2 * 128 * 12 * 4 + 2 * 128 * 36 * 4 + 2 * 128 * 4),
                // v11gu/v11gut (Issue 902): 4*128*12*4 A codes (2 mats x 2
                // bufs) + 2*TOKS*36*4 B + 4*128*4 scales — 45,056 B at TOKS=64
                // (under the 48 KB default but set for uniformity) and 63,488 B
                // at TOKS=128 (the opt-in case).
                (&v11gu_fused, 4 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 4 * 128 * 4),
                (&v11gu_strict, 4 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 4 * 128 * 4),
                (&v11gut_fused, 4 * 128 * 12 * 4 + 2 * 128 * 36 * 4 + 4 * 128 * 4),
                (&v11gut_strict, 4 * 128 * 12 * 4 + 2 * 128 * 36 * 4 + 4 * 128 * 4),
                (&v11gs_fused, 4 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 4 * 128 * 4),
                (&v11gs_strict, 4 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 4 * 128 * 4),
                (&v11gq_fused, 4 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 4 * 128 * 4),
                (&v11gq_strict, 4 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 4 * 128 * 4),
                (&v10t_nodecode, 2 * 128 * 12 * 4 + 2 * 128 * 36 * 4 + 2 * 128 * 4),
            ] {
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes,
                )
                .map_err(|e| GemmI8MmaError::Load(format!("v6 smem opt-in: {e}")))?;
            }
            (v6_fused, v6_strict, v6t_fused, v6t_strict, v6d_fused, v6d_strict,
             v6tl_fused, v6tl_strict, v6tb_fused, v6tb_strict,
             v7_fused, v7_strict, v7t_fused, v7t_strict,
             v8_fused, v8_strict, v8t_fused, v8t_strict,
             v9_fused, v9_strict, v9t_fused, v9t_strict,
             v10_fused, v10_strict, v10t_fused, v10t_strict,
             v10o3_fused, v10o3_strict, v10o3_nodecode,
             v12_fused, v12_strict, v12_nodecode,
             v11gu_fused, v11gu_strict, v11gut_fused, v11gut_strict,
             v11gs_fused, v11gs_strict, v11gq_fused, v11gq_strict,
             v10t_nodecode, v6_module)
        };
        let ret = Self {
            quantize_rn: load("quantize_rows_i8_hilo_cuda_rn")?,
            quantize_full: load("quantize_rows_i8_hilo_cuda_full")?,
            quantize_approx: load("quantize_rows_i8_hilo_cuda_approx")?,
            quant_div: std::sync::atomic::AtomicU8::new(0),
            gemm_tm128_fused: load("gemm_i8_mma_tm128_fused")?,
            gemm_tm128_strict: load("gemm_i8_mma_tm128_strict")?,
            gemm_tm64_fused: load("gemm_i8_mma_tm64_fused")?,
            gemm_tm64_strict: load("gemm_i8_mma_tm64_strict")?,
            gemm_tm128v2_fused: load("gemm_i8_mma_tm128v2_fused")?,
            gemm_tm128v2_strict: load("gemm_i8_mma_tm128v2_strict")?,
            gemm_tm64v2_fused: load("gemm_i8_mma_tm64v2_fused")?,
            gemm_tm64v2_strict: load("gemm_i8_mma_tm64v2_strict")?,
            probe_tm128v2b2: load("gemm_i8_mma_tm128v2b2_fused")?,
            probe_tm64v2b4: load("gemm_i8_mma_tm64v2b4_fused")?,
            gemm_tm128v3_fused: load("gemm_i8_mma_tm128v3_fused")?,
            gemm_tm128v3_strict: load("gemm_i8_mma_tm128v3_strict")?,
            gemm_tm128v4_fused: load("gemm_i8_mma_tm128v4_fused")?,
            gemm_tm128v4_strict: load("gemm_i8_mma_tm128v4_strict")?,
            gemm_tm32sp_fused: load("gemm_i8_mma_tm32sp_fused")?,
            gemm_tm32sp_strict: load("gemm_i8_mma_tm32sp_strict")?,
            gemm_tm32sp8y_fused: load("gemm_i8_mma_tm32sp8y_fused")?,
            gemm_tm32sp8y_strict: load("gemm_i8_mma_tm32sp8y_strict")?,
            gemm_tm256v5_fused: load("gemm_i8_mma_tm256v5_fused")?,
            gemm_tm256v5_strict: load("gemm_i8_mma_tm256v5_strict")?,
            gemm_tm128v5t_fused: load("gemm_i8_mma_tm128v5t_fused")?,
            gemm_tm128v5t_strict: load("gemm_i8_mma_tm128v5t_strict")?,
            #[cfg(feature = "prefill_q8_act")]
            quantize_q8_rn,
            #[cfg(feature = "prefill_q8_act")]
            quantize_q8_full,
            #[cfg(feature = "prefill_q8_act")]
            quantize_q8_approx,
            #[cfg(feature = "prefill_q8_act")]
            gemm_tm128v4_q8_fused,
            #[cfg(feature = "prefill_q8_act")]
            gemm_tm128v4_q8_strict,
            #[cfg(feature = "prefill_q8_act")]
            _q8_module,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6t_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6t_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6d_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6d_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6tl_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6tl_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6tb_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v6tb_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v7_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v7_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v7t_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v7t_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v8_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v8_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v8t_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v8t_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v9_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v9_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v9t_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v9t_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10t_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10t_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10o3_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10o3_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10o3_q8_nodecode,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm64v12_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm64v12_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm64v12_q8_nodecode,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gu_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gu_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gut_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gut_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gs_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gs_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gq_q8_fused,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v11gq_q8_strict,
            #[cfg(feature = "prefill_mmq_v2")]
            gemm_tm128v10t_q8_nodecode,
            #[cfg(feature = "prefill_mmq_v2")]
            _mmq_v2_module,
            _module: module,
        };
        // v4 uses 70.7 KB dynamic smem (> the 48 KB default) — opt in.
        for f in [&ret.gemm_tm128v4_fused, &ret.gemm_tm128v4_strict] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                128 * 132 * 2 + 2 * 2 * 64 * 36 * 4,
            )
            .map_err(|e| GemmI8MmaError::Load(format!("smem opt-in: {e}")))?;
        }
        // v5 tm256: 256*132 (single A buf) + 2 * 2*64*36*4 (B slabs) = 70656 B;
        // v5t:     128*132 + 2 * 2*128*36*4 = 90624 B.
        for (f, bytes) in [
            (&ret.gemm_tm256v5_fused, 256 * 132 + 2 * 2 * 64 * 36 * 4),
            (&ret.gemm_tm256v5_strict, 256 * 132 + 2 * 2 * 64 * 36 * 4),
            (&ret.gemm_tm128v5t_fused, 128 * 132 + 2 * 2 * 128 * 36 * 4),
            (&ret.gemm_tm128v5t_strict, 128 * 132 + 2 * 2 * 128 * 36 * 4),
        ] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
            .map_err(|e| GemmI8MmaError::Load(format!("smem opt-in: {e}")))?;
        }
        Ok(ret)
    }

    /// Select the division form used by [`Self::launch_quantize`].
    pub fn set_quant_div(&self, d: QuantDiv) {
        self.quant_div.store(d as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Occupancy probe for the A/B benches: (name, regs/thread, local bytes,
    /// threads) per kernel variant.
    pub fn debug_kernel_attrs(&self) -> Vec<(&'static str, i32, i32, u32)> {
        let rows = [
            ("tm128 v1", &self.gemm_tm128_fused, 256u32),
            ("tm64 v1", &self.gemm_tm64_fused, 128),
            ("tm128 v2", &self.gemm_tm128v2_fused, 256),
            ("tm64 v2", &self.gemm_tm64v2_fused, 128),
            ("tm128 v2 lb(256,2)", &self.probe_tm128v2b2, 256),
            ("tm64 v2 lb(128,4)", &self.probe_tm64v2b4, 128),
            ("tm128 v3 (512thr)", &self.gemm_tm128v3_fused, 512),
            ("tm128 v4 bstage", &self.gemm_tm128v4_fused, 512),
            ("tm32 sp (16tok)", &self.gemm_tm32sp_fused, 128),
            ("tm32 sp8y (8tok)", &self.gemm_tm32sp8y_fused, 64),
            ("tm256 v5", &self.gemm_tm256v5_fused, 512),
            ("tm128 v5t (128tok)", &self.gemm_tm128v5t_fused, 512),
        ];
        rows.iter()
            .map(|(n, f, t)| {
                (
                    *n,
                    f.num_regs().unwrap_or(-1),
                    f.local_size_bytes().unwrap_or(-1),
                    *t,
                )
            })
            .collect()
    }

    fn current_quant_div(&self) -> QuantDiv {
        match self.quant_div.load(std::sync::atomic::Ordering::Relaxed) {
            1 => QuantDiv::Full,
            2 => QuantDiv::Approx,
            _ => QuantDiv::Rn,
        }
    }

    /// Allocate the quantize scratch for a given `(n, p)`.
    pub fn alloc_scratch(
        &self,
        stream: &Arc<CudaStream>,
        n: usize,
        p: usize,
    ) -> Result<GemmI8MmaScratch, GemmI8MmaError> {
        let words = p * (n / 4);
        Ok(GemmI8MmaScratch {
            q_hi_w: stream
                .alloc_zeros::<u32>(words)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?,
            q_lo_w: stream
                .alloc_zeros::<u32>(words)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?,
            s_t: stream
                .alloc_zeros::<f32>(p)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?,
        })
    }

    /// Run the quantize pre-pass with the selected division form.
    pub fn launch_quantize(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        n: usize,
        p: usize,
    ) -> Result<(), GemmI8MmaError> {
        let div = self.current_quant_div();
        self.launch_quantize_div(stream, input, scratch, n, p, div)
    }

    /// Run the quantize pre-pass with an explicit division form.
    pub fn launch_quantize_div(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        n: usize,
        p: usize,
        div: QuantDiv,
    ) -> Result<(), GemmI8MmaError> {
        let func = match div {
            QuantDiv::Rn => &self.quantize_rn,
            QuantDiv::Full => &self.quantize_full,
            QuantDiv::Approx => &self.quantize_approx,
        };
        let (n_i, p_u) = (n as i32, p as u32);
        let cfg = LaunchConfig {
            grid_dim: (p_u, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(input)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.q_lo_w)
                .arg(&scratch.s_t)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Run the SINGLE-PLANE quantize pre-pass (Issue 884 T2a; opt-in
    /// `prefill_q8_act`) with an explicit division form. Writes
    /// `scratch.q_hi_w` + `scratch.s_t` — bit-identical to the hi/lo
    /// kernel's hi plane + scale at the same `div`; `q_lo_w` is never
    /// written (the q8 GEMM never reads it).
    #[cfg(feature = "prefill_q8_act")]
    pub fn launch_quantize_q8_div(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        n: usize,
        p: usize,
        div: QuantDiv,
    ) -> Result<(), GemmI8MmaError> {
        let func = match div {
            QuantDiv::Rn => &self.quantize_q8_rn,
            QuantDiv::Full => &self.quantize_q8_full,
            QuantDiv::Approx => &self.quantize_q8_approx,
        };
        let (n_i, p_u) = (n as i32, p as u32);
        let cfg = LaunchConfig {
            grid_dim: (p_u, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(input)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the single-plane v4-q8 GEMM (Issue 884 T2a; opt-in
    /// `prefill_q8_act`): ONE `mma.m16n8k32.s8` pass per k-step over the
    /// hi plane, fold `o += sw · (f32)hi`, identical A staging / group
    /// order / s_t epilogue. Reads `scratch.q_hi_w` only. The v4 shape at
    /// every `p` (no smallp twin — the prefill lane's p ≫ 16).
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_q8_act")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v4_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v4_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // 2 x (128 x 132) A stage + 2 x (64 x 36 x 4) single-plane B slabs
        // = 52224 B (the smem opt-in is set in [`Self::new`]).
        let smem = (128 * ASM_ROW_STRIDE * 2 + 2 * 64 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the fork-config v6-q8 GEMM (Issue 884 T2b; opt-in
    /// `prefill_mmq_v2`): the v4-q8 arithmetic under the opponent's config
    /// class — warp tile 32×32 (8 mma per A fragment load), XOR-perm swizzled
    /// conflict-free A reads, `cp.async.cg` B stage, single-buffered A +
    /// double-buffered B (34,816 B smem, 2 blocks/SM). Bit-identical to
    /// [`Self::launch_gemm_q8`] at every shape by construction — the unit
    /// gate asserts 0 bit-diffs; only scheduling/layout change. Same
    /// block/grid shape as v4-q8: grid `(m/128, p/64)`, one mma pass per
    /// k-step over the hi plane, reads `scratch.q_hi_w` only.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v6(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v6: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v6: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v6_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v6_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // Single 128x32-word A stage + double 64x36-word single-plane B
        // slabs = 16384 + 18432 = 34816 B — under the 48 KB default.
        let smem = (128 * 32 * 4 + 2 * 64 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the TOKS=128 v6t twin (Issue 884 T2b; opt-in
    /// `prefill_mmq_v2`): the v6 body at 512 threads / 16 warps covering 128
    /// tokens per block — half the blocks of v6, halving the per-token A
    /// global-staging traffic and epilogue passes at the same warp tile
    /// (32x32) and A-dup. 1 block/SM (16 warps — v6 reaches the same
    /// occupancy with 2 blocks of 8). Bit-identical to v4-q8 by the same
    /// construction as [`Self::launch_gemm_q8_v6`].
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v6t(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v6t: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v6t: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v6t_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v6t_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // Single 128x32-word A stage + double 128x36-word B slabs = 16384 +
        // 36864 = 53248 B (the >48 KB opt-in is set in [`Self::new`]).
        let smem = (128 * 32 * 4 + 2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the double-A v6d twin (Issue 884 T2b; opt-in `prefill_mmq_v2`):
    /// v6t's tile with v4's 1-barrier pipeline — A(g+1) lands in the OTHER
    /// buffer during g's compute (no mid-group A barrier), so staging is
    /// fully overlapped. 2 blocks/SM is impossible at 69,632 B smem; 1 block
    /// × 16 warps matches v4's occupancy.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v6d(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v6d: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v6d: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v6d_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v6d_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // Double 128x32-word A stages + double 128x36-word B slabs = 32768 +
        // 36864 = 69632 B (the >48 KB opt-in is set in [`Self::new`]).
        let smem = (2 * 128 * 32 * 4 + 2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the ldmatrix-A twin (T2b continuation, Bench 880; opt-in
    /// `prefill_mmq_v2`): v6t's tile with each A fragment loaded by ONE
    /// `ldmatrix.m8n8.x4.shared.b16` (32 LDS.32/group/warp → 8 ldmatrix; the
    /// fork pairs its swizzle with ldmatrix). Same swizzle — `V6_A_WORD` at a
    /// 16-B-aligned base is 4 consecutive physical words, the layout contract
    /// ldmatrix needs. Bit-identical to v4-q8 by the same construction as
    /// [`Self::launch_gemm_q8_v6`].
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v6tl(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v6tl: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v6tl: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v6tl_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v6tl_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // v6t's tile: single 128x32-word A stage + double 128x36-word B slabs.
        let smem = (128 * 32 * 4 + 2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the ldmatrix-A+B twin (T2b continuation, Bench 880; opt-in
    /// `prefill_mmq_v2`): [`Self::launch_gemm_q8_v6tl`] plus one
    /// `ldmatrix.m8n8.x2.shared.b16` per B fragment (2 LDS → 1 per (nf, ks)).
    /// Bit-identical to v4-q8 by the same construction.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v6tb(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v6tb: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v6tb: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v6tb_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v6tb_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (128 * 32 * 4 + 2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the fork-style global-A twin (T2b rung 3, Bench 881; opt-in
    /// `prefill_mmq_v2`): v6-class geometry at TOKS=64 / 256 threads /
    /// 2 blocks/SM with NO A smem stage — mask words are LDG'd per fragment
    /// and SWAR-expanded in registers (the pinned opponent's A-path).
    /// 18,432 B smem (B only, under the default). Bit-identical to v4-q8 by
    /// the same construction as [`Self::launch_gemm_q8_v6`].
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v7(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v7: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v7: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v7_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v7_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 64 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the fork-style global-A twin at TOKS=128 (T2b rung 3, Bench
    /// 881; opt-in `prefill_mmq_v2`): 512 threads / 16 warps / 1 block/SM,
    /// 36,864 B smem (under the 48 KB default). Bit-identical to v4-q8.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v7t(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v7t: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v7t: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v7t_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v7t_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the format-rung twin (T2b rung 4, Plan 572; opt-in
    /// `prefill_mmq_v2`): v7's global-A structure over PACKED-2-BIT Q2_0-code
    /// weights + the PRMT-table fragment decode (8 ALU-pipe ops per 8 weights
    /// post-903-T2a — the fork's decode class). `packed` is the
    /// [`crate::prefill_cuda_mma::pack_bitplanes_to_q2`] output (or a Q2_0
    /// GGUF passthrough): `u32[m * n/16]`, k-contiguous LSB-first codes
    /// (00=-1, 01=0, 10=+1; code 3 never emitted). TOKS=64 / 256 threads /
    /// 2 blocks/SM, 18,432 B smem (under the default). Bit-identical to
    /// v4-q8 by the same construction as [`Self::launch_gemm_q8_v6`] — the
    /// unit gate pins it (incl. the non-disjoint-plane fold).
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v8(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v8: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v8: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v8_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v8_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 64 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the format-rung twin at TOKS=128 (T2b rung 4, Plan 572; the
    /// primary arm — v7t's geometry): 512 threads / 16 warps / 1 block/SM,
    /// 36,864 B smem (under the 48 KB default). Bit-identical to v4-q8.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v8t(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v8t: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v8t: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v8t_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v8t_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the staged-packed twin (T2b rung 5, Plan 573; opt-in
    /// `prefill_mmq_v2`): the v6tb staged structure over the packed codes —
    /// A cp.async'd into a double-buffered 128x12-word code stage (256 x 16-B
    /// chunks per k-group, once per block: no per-lane global A stream, no
    /// token-warp re-read) + per-use `v8_decode_pair` fragment decode from
    /// quad-broadcast LDS reads. `packed` is the
    /// [`crate::prefill_cuda_mma::pack_bitplanes_to_q2`] output: `u32[m *
    /// n/16]`, k-contiguous LSB-first codes. TOKS=64 / 256 threads /
    /// 2 blocks/SM, 30,720 B smem (under the default). Bit-identical to
    /// v4-q8 by the same construction as [`Self::launch_gemm_q8_v8`] — the
    /// unit gate pins it (incl. the non-disjoint-plane fold).
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v9(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v9: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v9: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v9_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v9_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 12 * 4 + 2 * 64 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the staged-packed twin at TOKS=128 (T2b rung 5, Plan 573; the
    /// primary arm — v8t/v6tb's geometry): 512 threads / 16 warps /
    /// 1 block/SM, 49,152 B smem (Rust-side opt-in). Bit-identical to v4-q8.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v9t(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v9t: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v9t: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v9t_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v9t_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 12 * 4 + 2 * 128 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the L2-traffic-repair twin (T4 rung, Plan 574; opt-in
    /// `prefill_mmq_v2`): v9's staged-packed body + (a) the group scales
    /// cp.async'd once per group per block into a double-buffered 128-float
    /// stage (kills the ~16x scattered scale-load amplification) + (b) a
    /// transposed [TOKS][33] epilogue stage whose readers store 128-B
    /// contiguous per warp (kills the 8.0x strided out-write amplification).
    /// Same ops, same order — bit-identical to v4-q8 by construction, pinned
    /// by the unit gate. TOKS=64 / 256 threads / 2 blocks/SM, 31,744 B smem
    /// (under the default).
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v10(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v10: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v10: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v10_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v10_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 2 * 128 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the L2-traffic-repair twin at TOKS=128 (T4 rung, Plan 574; the
    /// primary arm — v9t's geometry + the scale stage): 512 threads / 16
    /// warps / 1 block/SM, 50,176 B smem (Rust-side opt-in). Bit-identical
    /// to v4-q8.
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v10t(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v10t: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v10t: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v10t_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v10t_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 12 * 4 + 2 * 128 * 36 * 4 + 2 * 128 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Plan 597 rung A — launch the v10 body at MINB=3 (24 warps/SM, the
    /// 85-reg `__launch_bounds__(256, 3)` cap): same geometry as v10
    /// (TOKS=64, 256 threads, 31,744 B smem — under the 48 KB default, no
    /// opt-in; 3 blocks × 31,744 = 95,232 B fits the SM budget). Kernel-level
    /// probe arm for the S1.e gate — no production dispatch. Bit-identical
    /// to v10t by construction (same per-output op sequence; the geometry
    /// change touches no arithmetic).
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v10o3(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v10o3: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v10o3: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm128v10o3_q8_fused,
            FoldMode::Strict => &self.gemm_tm128v10o3_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 2 * 128 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Plan 597 v12 — launch the tile-shrunk body (64-row blocks, 16-tok
    /// warps, TOKS=64: 256 threads / 8 warps / 3 blocks/SM = 24 warps at the
    /// 85-reg `__launch_bounds__(256, 3)` cap). 25,088 B smem (under the
    /// 48 KB default, no opt-in; 3 × 25,088 = 75,264 B fits the SM budget).
    /// Probe arm for the S1.e gate — no production dispatch until it clears
    /// (+10% kernel at G1 bit-identity, spill-free). Bit-identical to v10t by
    /// construction (same per-output op sequence; geometry-only change).
    ///
    /// # Errors
    /// `n % 128 != 0` or `p == 0` → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v12(
        &self,
        stream: &CudaStream,
        packed: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v12: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v12: p must be >= 1".into()));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm64v12_q8_fused,
            FoldMode::Strict => &self.gemm_tm64v12_q8_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (2 * 64 * 12 * 4 + 2 * 64 * 36 * 4 + 2 * 64 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(64) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the fused gate+up pair at TOKS=64 (Issue 902 T1 — the
    /// occupancy arm): 256 threads / 8 warps / 2 blocks per SM, 45,056 B
    /// smem. `out0`/`out1` receive the two [p * m] slabs; bit-identical per
    /// slab to two `launch_gemm_q8_v10t` calls on the respective weights.
    ///
    /// # Errors
    /// `n % 128 != 0`, `p == 0`, or shape mismatch between the two weight
    /// sets → [`GemmI8MmaError::Launch`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v11gu_pair(
        &self,
        stream: &CudaStream,
        packed0: &CudaSlice<u32>,
        scale0: &CudaSlice<f32>,
        packed1: &CudaSlice<u32>,
        scale1: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out0: &CudaSlice<f32>,
        out1: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        self.launch_gemm_q8_v11_pair_geom(
            stream, packed0, scale0, packed1, scale1, scratch, out0, out1, m, n, p, fold,
            &self.gemm_tm128v11gu_q8_fused,
            &self.gemm_tm128v11gu_q8_strict,
            64,
            256,
            45_056,
        )
    }

    /// Launch the fused gate+up pair at TOKS=128 (Issue 902 T1 — v10t's
    /// geometry): 512 threads / 16 warps / 1 block per SM, 63,488 B smem.
    /// Same contract as [`launch_gemm_q8_v11gu_pair`].
    ///
    /// # Errors
    /// Same as [`launch_gemm_q8_v11gu_pair`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v11gut_pair(
        &self,
        stream: &CudaStream,
        packed0: &CudaSlice<u32>,
        scale0: &CudaSlice<f32>,
        packed1: &CudaSlice<u32>,
        scale1: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out0: &CudaSlice<f32>,
        out1: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        self.launch_gemm_q8_v11_pair_geom(
            stream, packed0, scale0, packed1, scale1, scratch, out0, out1, m, n, p, fold,
            &self.gemm_tm128v11gut_q8_fused,
            &self.gemm_tm128v11gut_q8_strict,
            128,
            512,
            63_488,
        )
    }

    /// Launch the fused gate+up pair, sequential shared-dh body at
    /// TOKS=64/2 blocks per SM (Bench 895 repair rung 1): 16 warps/SM kept,
    /// v10t's per-mat ldmatrix count, the B cp.async stage shared.
    ///
    /// # Errors
    /// Same as [`launch_gemm_q8_v11gu_pair`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v11gs_pair(
        &self,
        stream: &CudaStream,
        packed0: &CudaSlice<u32>,
        scale0: &CudaSlice<f32>,
        packed1: &CudaSlice<u32>,
        scale1: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out0: &CudaSlice<f32>,
        out1: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        self.launch_gemm_q8_v11_pair_geom(
            stream, packed0, scale0, packed1, scale1, scratch, out0, out1, m, n, p, fold,
            &self.gemm_tm128v11gs_q8_fused,
            &self.gemm_tm128v11gs_q8_strict,
            64,
            256,
            45_056,
        )
    }

    /// Launch the fused gate+up pair, interleaved body at the 255-register
    /// budget (Bench 895 repair rung 2): TOKS=64, 1 block per SM (8 warps),
    /// both dh sets + shared B-fragment loads, no spill.
    ///
    /// # Errors
    /// Same as [`launch_gemm_q8_v11gu_pair`].
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_q8_v11gq_pair(
        &self,
        stream: &CudaStream,
        packed0: &CudaSlice<u32>,
        scale0: &CudaSlice<f32>,
        packed1: &CudaSlice<u32>,
        scale1: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out0: &CudaSlice<f32>,
        out1: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        self.launch_gemm_q8_v11_pair_geom(
            stream, packed0, scale0, packed1, scale1, scratch, out0, out1, m, n, p, fold,
            &self.gemm_tm128v11gq_q8_fused,
            &self.gemm_tm128v11gq_q8_strict,
            64,
            256,
            45_056,
        )
    }

    /// The shared v11 pair-launch body (geometry-parameterized; the two
    /// public twins above pin the geometry constants).
    #[cfg(feature = "prefill_mmq_v2")]
    #[allow(clippy::too_many_arguments)]
    fn launch_gemm_q8_v11_pair_geom(
        &self,
        stream: &CudaStream,
        packed0: &CudaSlice<u32>,
        scale0: &CudaSlice<f32>,
        packed1: &CudaSlice<u32>,
        scale1: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out0: &CudaSlice<f32>,
        out1: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
        fused: &CudaFunction,
        strict: &CudaFunction,
        toks: u32,
        threads: u32,
        smem: u32,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "q8_v11gu: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 {
            return Err(GemmI8MmaError::Launch("q8_v11gu: p must be >= 1".into()));
        }
        let shape_err = |which: &str| {
            GemmI8MmaError::Launch(format!(
                "q8_v11gu: both weight sets must share m/n/p ({which} mismatch)"
            ))
        };
        if packed0.len() != packed1.len() {
            return Err(shape_err("packed words"));
        }
        if scale0.len() != scale1.len() {
            return Err(shape_err("group scales"));
        }
        if out0.len() != out1.len() {
            return Err(shape_err("out slabs"));
        }
        let func = match fold {
            FoldMode::Fused => fused,
            FoldMode::Strict => strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 16) as i32;
        let groups_i = (n / 128) as i32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(128) as u32, p.div_ceil(toks as usize) as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(packed0)
                .arg(scale0)
                .arg(packed1)
                .arg(scale1)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.s_t)
                .arg(out0)
                .arg(out1)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Run the GEMM (`out[tok * m + row] = dequant(W) @ x^T`, `out` is
    /// `[p * m]` f32). Requires `n % 128 == 0` (same GROUP_COLS contract as
    /// the CubeCL i8 kernel) and `p >= 1`.
    #[allow(clippy::cast_possible_truncation)]
    /// Launch the small-p (p ≤ 16) v4-sp GEMM (Issue 742 T1). Bit-identical
    /// to v4 at every shape — same arithmetic, different block/warp geometry
    /// (see [`smallp_gemm_enabled`]).
    ///
    /// # Panics / errors
    /// `n % 128 != 0` or `p` outside `1..=16` → [`GemmI8MmaError::Launch`].
    #[allow(clippy::cast_possible_truncation)]
    pub fn launch_gemm_smallp(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "smallp: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 || p > 16 {
            return Err(GemmI8MmaError::Launch(format!(
                "smallp: p must be in 1..=16, got {p}"
            )));
        }
        // Issue 742 T1.4 — opt-in y-split routing (MEASURED NEGATIVE, default
        // off — see smallp_y8_enabled). Call sites wanting a specific tile
        // at any m/p use `launch_gemm_smallp_tm32` / `launch_gemm_smallp_y8`.
        if smallp_y8_enabled() && smallp_y8_routes(m, p) {
            return self.launch_gemm_smallp_y8(
                stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, fold,
            );
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm32sp_fused,
            FoldMode::Strict => &self.gemm_tm32sp_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // 2 x (32 x 132) A stage + 2 x (2 x 16 x 36 x 4) B slabs = 17664 B.
        let smem = (2 * 32 * ASM_ROW_STRIDE + 2 * 2 * 16 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(32) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.q_lo_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the TM32sp small-p GEMM at any dispatchable shape (bypasses the
    /// y-split routing — the explicit A/B arm for the T1.4 benches).
    #[allow(clippy::cast_possible_truncation)]
    pub fn launch_gemm_smallp_tm32(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "smallp: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 || p > 16 {
            return Err(GemmI8MmaError::Launch(format!(
                "smallp: p must be in 1..=16, got {p}"
            )));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm32sp_fused,
            FoldMode::Strict => &self.gemm_tm32sp_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // 2 x (32 x 132) A stage + 2 x (2 x 16 x 36 x 4) B slabs = 17664 B.
        let smem = (2 * 32 * ASM_ROW_STRIDE + 2 * 2 * 16 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(32) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.q_lo_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch the y-split small-p GEMM (32r x 8t tile, 64 threads, grid
    /// `(p/8, m/32)`) — the T1.4 shape. Bit-identical to v4 at every
    /// dispatchable shape (G1-gated) but **MEASURED SLOWER than TM32sp at
    /// every shape** (×0.63-0.71 — 2× A weight traffic); opt-in only.
    ///
    /// # Panics / errors
    /// `n % 128 != 0` or `p` outside `1..=16` → [`GemmI8MmaError::Launch`].
    #[allow(clippy::cast_possible_truncation)]
    pub fn launch_gemm_smallp_y8(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        if !n.is_multiple_of(128) {
            return Err(GemmI8MmaError::Launch(
                "smallp-y8: n must be a multiple of 128 (GROUP_COLS)".into(),
            ));
        }
        if p == 0 || p > 16 {
            return Err(GemmI8MmaError::Launch(format!(
                "smallp-y8: p must be in 1..=16, got {p}"
            )));
        }
        let func = match fold {
            FoldMode::Fused => &self.gemm_tm32sp8y_fused,
            FoldMode::Strict => &self.gemm_tm32sp8y_strict,
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // 2 x (32 x 132) A stage + 2 x (2 x 8 x 36 x 4) B slabs = 13056 B.
        let smem = (2 * 32 * ASM_ROW_STRIDE + 2 * 2 * 8 * 36 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (p.div_ceil(8) as u32, m.div_ceil(32) as u32, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.q_lo_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    pub fn launch_gemm(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        tile: MmaTile,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        // Issue 742 T1 — small-p chunks dispatch to the v4-sp kernel BEFORE
        // the v4/v5 resolution (v5's TM=256 would cut the grid further at
        // p=16; v4-sp's m/32 grid + 5 blocks/SM is the small-p shape).
        if smallp_gemm_enabled() && (1..=16).contains(&p) && n.is_multiple_of(128) {
            return self.launch_gemm_smallp(
                stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, fold,
            );
        }
        let kernel_gen = if mma_v2_enabled() {
            if mma_v5_enabled() && m >= MMA_V5_MIN_M {
                MmaGen::V5
            } else {
                MmaGen::V4
            }
        } else {
            MmaGen::V1
        };
        // v4 is the 512-thread TM=128 shape — the caller's Tm64 is the
        // pre-v4 size hint; rewrite it so the off-state ("0") reproduces
        // the exact prior shipping path (v1 + Tm64). v5 rewrites to its own
        // TM=256 shape (m >= MMA_V5_MIN_M only — small projections stay v4).
        let tile = match kernel_gen {
            MmaGen::V4 => MmaTile::Tm128,
            MmaGen::V5 => MmaTile::Tm256,
            _ => tile,
        };
        self.launch_gemm_gen(
            stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, tile, fold, kernel_gen,
        )
    }

    /// Launch an occupancy-pinned probe kernel (Arm 10 G2 ladder; Fused fold
    /// only — these are perf probes, bit-identity is pinned by the unbounded
    /// v2 twins).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_probe(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        tile: MmaTile,
        b2_or_b4: u8,
    ) -> Result<(), GemmI8MmaError> {
        assert!(n.is_multiple_of(128));
        let (func, tm, threads) = match (tile, b2_or_b4) {
            (MmaTile::Tm128, 2) => (&self.probe_tm128v2b2, 128usize, 256u32),
            (MmaTile::Tm64, 4) => (&self.probe_tm64v2b4, 64, 128),
            _ => return Err(GemmI8MmaError::Launch("unknown probe".into())),
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        let smem = (tm * ASM_ROW_STRIDE * 2) as u32;
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(tm) as u32, p.div_ceil(64) as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.q_lo_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Explicit-generation GEMM launch (the A/B bench form; [`Self::launch_gemm`]
    /// resolves the generation from the [`MmaGen`] knob).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gemm_gen(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        tile: MmaTile,
        fold: FoldMode,
        kernel_gen: MmaGen,
    ) -> Result<(), GemmI8MmaError> {
        assert!(n.is_multiple_of(128), "n must be a multiple of 128 (GROUP_COLS)");
        assert!(p >= 1);
        let func = match (tile, fold, kernel_gen) {
            (MmaTile::Tm128, FoldMode::Fused, MmaGen::V1) => &self.gemm_tm128_fused,
            (MmaTile::Tm128, FoldMode::Strict, MmaGen::V1) => &self.gemm_tm128_strict,
            (MmaTile::Tm64, FoldMode::Fused, MmaGen::V1) => &self.gemm_tm64_fused,
            (MmaTile::Tm64, FoldMode::Strict, MmaGen::V1) => &self.gemm_tm64_strict,
            (MmaTile::Tm128, FoldMode::Fused, MmaGen::V2) => &self.gemm_tm128v2_fused,
            (MmaTile::Tm128, FoldMode::Strict, MmaGen::V2) => &self.gemm_tm128v2_strict,
            (MmaTile::Tm64, FoldMode::Fused, MmaGen::V2) => &self.gemm_tm64v2_fused,
            (MmaTile::Tm64, FoldMode::Strict, MmaGen::V2) => &self.gemm_tm64v2_strict,
            (MmaTile::Tm128, FoldMode::Fused, MmaGen::V3) => &self.gemm_tm128v3_fused,
            (MmaTile::Tm128, FoldMode::Strict, MmaGen::V3) => &self.gemm_tm128v3_strict,
            (MmaTile::Tm128, FoldMode::Fused, MmaGen::V4) => &self.gemm_tm128v4_fused,
            (MmaTile::Tm128, FoldMode::Strict, MmaGen::V4) => &self.gemm_tm128v4_strict,
            (MmaTile::Tm256, FoldMode::Fused, MmaGen::V5) => &self.gemm_tm256v5_fused,
            (MmaTile::Tm256, FoldMode::Strict, MmaGen::V5) => &self.gemm_tm256v5_strict,
            (MmaTile::Tm128, FoldMode::Fused, MmaGen::V5) => &self.gemm_tm128v5t_fused,
            (MmaTile::Tm128, FoldMode::Strict, MmaGen::V5) => &self.gemm_tm128v5t_strict,
            (MmaTile::Tm64, _, MmaGen::V3)
            | (MmaTile::Tm64, _, MmaGen::V4)
            | (MmaTile::Tm64, _, MmaGen::V5)
            | (MmaTile::Tm256, _, MmaGen::V1)
            | (MmaTile::Tm256, _, MmaGen::V2)
            | (MmaTile::Tm256, _, MmaGen::V3)
            | (MmaTile::Tm256, _, MmaGen::V4) => {
                return Err(GemmI8MmaError::Launch(
                    "tile/generation combo unavailable (v3/v4 are tm128-only; v5 is tm256 + tm128-v5t only)"
                        .into(),
                ))
            }
        };
        let tm = match tile {
            MmaTile::Tm128 => 128usize,
            MmaTile::Tm64 => 64usize,
            MmaTile::Tm256 => 256usize,
        };
        // v3/v4/v5 = 512 threads (16 warps); v2/v1 = TM/64 * 128.
        let threads = match kernel_gen {
            MmaGen::V3 | MmaGen::V4 | MmaGen::V5 => 512u32,
            _ => match tile {
                MmaTile::Tm128 => 256u32,
                MmaTile::Tm64 => 128u32,
                MmaTile::Tm256 => unreachable!("Tm256 is v5-only"),
            },
        };
        let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
        let wpr_i = (n / 32) as i32;
        let groups_i = (n / 128) as i32;
        // v2/v3 double-buffer the A stage (2 x TM x 132 bytes); v4 adds the
        // double-buffered B slabs (2 x 2 x 64 x 36 x 4 bytes); v5 single-
        // buffers A and widens the tile (tm256: 256*132 + B slabs = 70656;
        // v5t: 128*132 + 2 x 2 x 128 x 36 x 4 = 90624).
        let smem = match (kernel_gen, tile) {
            (MmaGen::V1, _) => (tm * ASM_ROW_STRIDE) as u32,
            (MmaGen::V4, _) => (tm * ASM_ROW_STRIDE * 2 + 2 * 2 * 64 * 36 * 4) as u32,
            (MmaGen::V5, MmaTile::Tm256) => (256 * ASM_ROW_STRIDE + 2 * 2 * 64 * 36 * 4) as u32,
            (MmaGen::V5, _) => (128 * ASM_ROW_STRIDE + 2 * 2 * 128 * 36 * 4) as u32,
            _ => (tm * ASM_ROW_STRIDE * 2) as u32,
        };
        // Token block: 64 for every generation except v5t (128 tokens).
        let tok_block = if kernel_gen == MmaGen::V5 && tile == MmaTile::Tm128 {
            128usize
        } else {
            64usize
        };
        let cfg = LaunchConfig {
            grid_dim: (m.div_ceil(tm) as u32, p.div_ceil(tok_block) as u32, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(&scratch.q_hi_w)
                .arg(&scratch.q_lo_w)
                .arg(&scratch.s_t)
                .arg(out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&p_i)
                .arg(&wpr_i)
                .arg(&groups_i)
                .launch(cfg)
                .map_err(|e| GemmI8MmaError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Quantize + GEMM in one call (the POC A/B arm — includes the quantize
    /// cost, matching the shipping launcher's accounting).
    pub fn launch(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
        tile: MmaTile,
        fold: FoldMode,
    ) -> Result<(), GemmI8MmaError> {
        self.launch_quantize(stream, input, scratch, n, p)?;
        self.launch_gemm(
            stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, tile, fold,
        )
    }

    /// The prefill QUANTIZE with the activation-quant arm resolved
    /// (Issue 884 T2a). The canonical prefill div form (`QuantDiv::Full`,
    /// the Bench-719 bit-identity selector) in BOTH arms; the q8 arm
    /// (opt-in `prefill_q8_act` + [`q8_act_enabled`]) computes the
    /// single plane only. Knob off → EXACTLY the shipping
    /// [`Self::launch_quantize_div`] call — bit-preserving by
    /// construction.
    pub fn launch_prefill_quantize(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        n: usize,
        p: usize,
    ) -> Result<(), GemmI8MmaError> {
        #[cfg(feature = "prefill_q8_act")]
        if q8_act_enabled() {
            return self.launch_quantize_q8_div(stream, input, scratch, n, p, QuantDiv::Full);
        }
        self.launch_quantize_div(stream, input, scratch, n, p, QuantDiv::Full)
    }

    /// The prefill GEMM with the activation-quant arm resolved (the tile /
    /// fold / generation resolution of the shipping prefill call sites:
    /// `MmaTile::Tm64` + [`FoldMode::Fused`] through [`Self::launch_gemm`],
    /// whose smallp / v4 / v5 routing is the anchor semantics). The q8 arm
    /// (opt-in `prefill_q8_act` + [`q8_act_enabled`]) dispatches the
    /// single-plane v4-q8 kernel instead; with `prefill_mmq_v2` +
    /// [`mmq_v2_enabled`] also on, the fork-config v6-q8 kernel dispatches
    /// (bit-identical to v4-q8 — a scheduling knob, not a numerics one).
    /// Knobs off → EXACTLY the shipping call — bit-preserving by
    /// construction.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_prefill_gemm(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        scratch: &GemmI8MmaScratch,
        out: &CudaSlice<f32>,
        m: usize,
        n: usize,
        p: usize,
    ) -> Result<(), GemmI8MmaError> {
        #[cfg(feature = "prefill_mmq_v2")]
        if q8_act_enabled() && mmq_v2_enabled() {
            // The measured winner of the T2b A/B (Bench 880): v6t's tile with
            // ldmatrix A+B fragment loads (1.156x over v4-q8 vs v6t's
            // 1.107x). The v6/v6d/v6tl variants stay launchable via their
            // direct launchers as A/B artifacts.
            return self.launch_gemm_q8_v6tb(
                stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, FoldMode::Fused,
            );
        }
        #[cfg(feature = "prefill_q8_act")]
        if q8_act_enabled() {
            return self.launch_gemm_q8(stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, FoldMode::Fused);
        }
        self.launch_gemm(
            stream, pos_bits, neg_bits, group_scale, scratch, out, m, n, p, MmaTile::Tm64,
            FoldMode::Fused,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    /// The PTX fragment mapping is load-bearing and derived from the ISA
    /// doc from memory — pinned here against a CPU reference on an
    /// asymmetric fixture (Bench 709's transpose-asymmetric lesson).
    #[test]
    fn mma_layout_probe_matches_cpu() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        // Asymmetric A (row ramp) and B (k ramp) — exposes any row/k/byte
        // permutation in the mapping.
        let a_lin: Vec<i8> = (0..16 * 32)
            .map(|i| (((i * 37) % 251) - 125) as i8)
            .collect();
        let b_lin: Vec<i8> = (0..32 * 8)
            .map(|i| (((i * 89) % 241) - 120) as i8)
            .collect();

        // CPU reference: D[r][c] = sum_k A[r][k] * B[k][c].
        let mut expect = vec![0i32; 16 * 8];
        for r in 0..16 {
            for c in 0..8 {
                let mut acc = 0i32;
                for k in 0..32 {
                    acc += i32::from(a_lin[r * 32 + k]) * i32::from(b_lin[k * 8 + c]);
                }
                expect[r * 8 + c] = acc;
            }
        }

        let a_dev = stream.clone_htod(&a_lin).unwrap();
        let b_dev = stream.clone_htod(&b_lin).unwrap();
        let mut d_host = vec![0i32; 16 * 8];
        let d_dev = stream.alloc_zeros::<i32>(16 * 8).unwrap();

        // Launch the probe kernel directly (bypass the placeholder helper).
        let module = &kernels._module;
        let f = module.load_function("mma_layout_probe_m16n8k32").unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&f)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&d_dev)
                .launch(cfg)
                .unwrap();
        }
        stream.memcpy_dtoh(&d_dev, &mut d_host).unwrap();
        assert_eq!(d_host, expect, "PTX m16n8k32 fragment mapping is wrong");
    }

    // ── Issue 884 T2a — the single-plane (q8) kernels ──────────────────────

    /// The q8 quantize kernel's CPU reference: per-row absmax, `s = m/127`
    /// (rn), `q = round-ties-even(x/s)` clamped to [-127, 127], packed 4
    /// bytes per word (byte j at shift 8j) — the hi plane of the hi/lo
    /// kernel, bit-for-bit.
    #[cfg(feature = "prefill_q8_act")]
    fn q8_quantize_cpu(row: &[f32]) -> (Vec<u32>, f32) {
        let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let s = if m > 0.0 { m / 127.0 } else { 1.0 };
        let words = row.len() / 4;
        let mut out = Vec::with_capacity(words);
        for w in 0..words {
            let mut acc = 0u32;
            for j in 0..4 {
                let xf = row[w * 4 + j] / s;
                let q = (xf.round_ties_even() as i32).clamp(-127, 127);
                acc |= ((q as u32) & 0xFF) << (8 * j);
            }
            out.push(acc);
        }
        (out, s)
    }

    /// The q8 quantize kernel matches the CPU reference (rn form) on a
    /// ragged multi-row fixture, and leaves `q_lo_w` untouched.
    #[cfg(feature = "prefill_q8_act")]
    #[test]
    fn q8_quantize_matches_cpu() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let n = 256usize;
        let p = 7usize;
        let input: Vec<f32> = (0..p * n)
            .map(|i| {
                let v = ((i * 37) % 251) as f32 - 120.0;
                if i % 13 == 0 { 0.0 } else { v * 0.031 }
            })
            .collect();
        let in_dev = stream.clone_htod(&input).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        kernels
            .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Rn)
            .unwrap();
        let mut q_host = vec![0u32; p * (n / 4)];
        let mut s_host = vec![0f32; p];
        stream.memcpy_dtoh(&scratch.q_hi_w, &mut q_host).unwrap();
        stream.memcpy_dtoh(&scratch.s_t, &mut s_host).unwrap();
        let lo_host = stream.clone_dtoh(&scratch.q_lo_w).unwrap();

        for r in 0..p {
            let (q_exp, s_exp) = q8_quantize_cpu(&input[r * n..(r + 1) * n]);
            assert_eq!(s_host[r], s_exp, "row {r}: scale");
            assert_eq!(&q_host[r * (n / 4)..(r + 1) * (n / 4)], &q_exp[..], "row {r}: q words");
        }
        assert!(lo_host.iter().all(|&w| w == 0), "q_lo_w must stay zeroed");
    }

    /// Bit-identity differential: with `q_lo_w` ZEROED, the anchor v4 GEMM
    /// restricted to the hi plane is arithmetically the q8 GEMM (the lo
    /// term contributes exactly 0 to every fold). Both kernels must agree
    /// BIT-exactly on the same scratch — Fused and Strict folds.
    #[cfg(feature = "prefill_q8_act")]
    #[test]
    fn q8_gemm_matches_anchor_with_zeroed_lo_plane() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n, p) = (256usize, 256usize, 130usize);
        let wpr = n / 32;
        let groups = n / 128;
        // Non-disjoint bitplanes on purpose (the G1 fixture class).
        // wrapping_mul: debug-profile-safe (the v6 fixture's convention —
        // release-mode wrap == wrapping_mul).
        let pos_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) as u32).collect();
        let neg_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0xBF58476D1CE4E5B9) as u32).collect();
        let group_scale: Vec<f32> = (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let input: Vec<f32> = (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();

        let pos_dev = stream.clone_htod(&pos_bits).unwrap();
        let neg_dev = stream.clone_htod(&neg_bits).unwrap();
        let gs_dev = stream.clone_htod(&group_scale).unwrap();
        let in_dev = stream.clone_htod(&input).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        kernels
            .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Rn)
            .unwrap();

        for fold in [FoldMode::Fused, FoldMode::Strict] {
            let out_anchor = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_q8 = stream.alloc_zeros::<f32>(p * m).unwrap();
            kernels
                .launch_gemm_gen(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_anchor, m, n, p,
                    MmaTile::Tm128, fold, MmaGen::V4,
                )
                .unwrap();
            kernels
                .launch_gemm_q8(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_q8, m, n, p, fold)
                .unwrap();
            let a = stream.clone_dtoh(&out_anchor).unwrap();
            let b = stream.clone_dtoh(&out_q8).unwrap();
            let diffs = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs, 0, "{fold:?}: q8 GEMM must be bit-identical to the \
                 zero-lo anchor (the lo term is exactly 0)");
        }

        // Absolute grounding: one CPU reference element (row 0, tok 0) —
        // exact i32 group products, f32 Fused fold in ascending group
        // order, s_t epilogue.
        let q_words = stream.clone_dtoh(&scratch.q_hi_w).unwrap();
        let s_t = stream.clone_dtoh(&scratch.s_t).unwrap();
        let qh = |tok: usize, k: usize| -> i32 {
            let w = q_words[tok * (n / 4) + k / 4];
            ((w >> (8 * (k % 4))) & 0xFF) as i8 as i32
        };
        let wbyte = |row: usize, k: usize| -> i32 {
            let pw = pos_bits[row * wpr + k / 32];
            let nw = neg_bits[row * wpr + k / 32];
            (((pw >> (k % 32)) & 1) as i32) - (((nw >> (k % 32)) & 1) as i32)
        };
        let mut o = 0f32;
        let wrow = 0usize; // the CPU-reference row (mirrors the kernel's row 0)
        for g in 0..groups {
            let mut acc = 0i32;
            for k in g * 128..(g + 1) * 128 {
                acc += wbyte(wrow, k) * qh(0, k);
            }
            o = group_scale[wrow * groups + g].mul_add(acc as f32, o);
        }
        let expected = o * s_t[0];
        let out = {
            let o = stream.alloc_zeros::<f32>(p * m).unwrap();
            kernels
                .launch_gemm_q8(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &o, m, n, p, FoldMode::Fused)
                .unwrap();
            stream.clone_dtoh(&o).unwrap()
        };
        assert_eq!(out[0], expected, "q8 GEMM CPU reference (row 0, tok 0)");
    }

    /// Issue 884 T2a — kernel-level A/B at the B801/B803 ffn_gate slab
    /// (m=17408, n=5120, p=2048): the anchor v4 (hi/lo, 2 mma passes per
    /// k-step) vs the single-plane v4-q8 (1 pass). Includes the quantize
    /// pre-pass in both arms (the production accounting). NOT the formal
    /// G2 gate (that is T4's full-slab suite) — this is the T2a POC's
    /// "did the dropped plane buy the projected R1 rung" datum.
    ///
    /// Run GPU-exclusive (the Issue 834 rule — verify no co-resident
    /// compute first); release profile only.
    #[cfg(feature = "prefill_q8_act")]
    #[test]
    #[ignore = "GPU perf probe: 7.1 GB-class slab allocations + exclusive GPU; run with --release --ignored --nocapture"]
    fn q8_gemm_v4_perf_ab_ffn_gate() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n, p) = (17408usize, 5120usize, 2048usize);
        let wpr = n / 32;
        let groups = n / 128;
        let pos_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0x9E3779B97F4A7C15) as u32).collect();
        let neg_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0xBF58476D1CE4E5B9) as u32).collect();
        let group_scale: Vec<f32> =
            (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let input: Vec<f32> =
            (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();
        let pos_dev = stream.clone_htod(&pos_bits).unwrap();
        let neg_dev = stream.clone_htod(&neg_bits).unwrap();
        let gs_dev = stream.clone_htod(&group_scale).unwrap();
        let in_dev = stream.clone_htod(&input).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        let out = stream.alloc_zeros::<f32>(p * m).unwrap();
        stream.synchronize().unwrap();

        let nominal_tf = 2.0 * m as f64 * n as f64 * p as f64 / 1e12;
        let _ = nominal_tf;
        let time = |run: &dyn Fn()| -> f64 {
            for _ in 0..2 {
                run();
            }
            stream.synchronize().unwrap();
            let mut xs = Vec::with_capacity(5);
            for _ in 0..5 {
                let t = std::time::Instant::now();
                run();
                stream.synchronize().unwrap();
                xs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            xs[2]
        };

        let anchor = time(&|| {
            kernels
                .launch_quantize_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Full)
                .unwrap();
            kernels
                .launch_gemm_gen(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    MmaTile::Tm128, FoldMode::Fused, MmaGen::V4,
                )
                .unwrap();
        });
        let q8 = time(&|| {
            kernels
                .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Full)
                .unwrap();
            kernels
                .launch_gemm_q8(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        });
        eprintln!(
            "[i884-t2a] ffn_gate m={m} n={n} p={p}: \
             anchor v4 {anchor:.3} ms ({:.1} TF nominal / {:.1} executed) \
             | q8 v4 {q8:.3} ms ({:.1} TF nominal / {:.1} executed) \
             | wall speedup {:.3}x (2.0x = the executed-work ceiling: the anchor doubles every MAC)",
            2.0 * m as f64 * n as f64 * p as f64 / 1e12 / (anchor * 1e-3),
            4.0 * m as f64 * n as f64 * p as f64 / 1e12 / (anchor * 1e-3),
            2.0 * m as f64 * n as f64 * p as f64 / 1e12 / (q8 * 1e-3),
            2.0 * m as f64 * n as f64 * p as f64 / 1e12 / (q8 * 1e-3),
            anchor / q8,
        );
    }

    // ── Issue 884 T2b — the fork-config (v6) kernels ───────────────────

    /// Bit-identity differential: the v6 fork-config kernel must equal the
    /// v4-q8 kernel BIT-exactly at the same scratch (same mma k-order, same
    /// per-group fold sequence — only scheduling/layout differ). Ragged p
    /// (130 % 64 != 0) exercises both tail paths.
    #[cfg(feature = "prefill_mmq_v2")]
    #[test]
    fn v6_gemm_matches_v4_q8_bitwise() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n, p) = (256usize, 256usize, 130usize);
        let wpr = n / 32;
        let groups = n / 128;
        // Non-disjoint bitplanes on purpose (the G1 fixture class).
        // wrapping_mul: the T2a fixture values verbatim (release-mode wrap ==
        // wrapping_mul), debug-profile-safe.
        let pos_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) as u32).collect();
        let neg_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0xBF58476D1CE4E5B9) as u32).collect();
        let group_scale: Vec<f32> = (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let input: Vec<f32> = (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();

        let pos_dev = stream.clone_htod(&pos_bits).unwrap();
        let neg_dev = stream.clone_htod(&neg_bits).unwrap();
        let gs_dev = stream.clone_htod(&group_scale).unwrap();
        let in_dev = stream.clone_htod(&input).unwrap();
        // The format-rung mirror (v6_gemm_matches_v4_q8_bitwise): the same
        // fixture through the ingest transform (exercises the (1,1)-fold and
        // the word-pair packing the PRMT decode consumes).
        let packed = crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos_bits, &neg_bits, m, n);
        let packed_dev = stream.clone_htod(&packed).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        kernels
            .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Rn)
            .unwrap();

        for fold in [FoldMode::Fused, FoldMode::Strict] {
            let out_v4 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v6 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v6t = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v6d = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v6tl = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v6tb = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v7 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v7t = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v8 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v8t = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v9 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v9t = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v10 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out_v10t = stream.alloc_zeros::<f32>(p * m).unwrap();
            kernels
                .launch_gemm_q8(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v4, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v6(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v6t(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6t, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v6d(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6d, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v6tl(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6tl, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v6tb(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6tb, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v7(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v7, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v7t(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v7t, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v8(&stream, &packed_dev, &gs_dev, &scratch, &out_v8, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v8t(&stream, &packed_dev, &gs_dev, &scratch, &out_v8t, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v9(&stream, &packed_dev, &gs_dev, &scratch, &out_v9, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v9t(&stream, &packed_dev, &gs_dev, &scratch, &out_v9t, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v10(&stream, &packed_dev, &gs_dev, &scratch, &out_v10, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v10t(&stream, &packed_dev, &gs_dev, &scratch, &out_v10t, m, n, p, fold)
                .unwrap();
            let a = stream.clone_dtoh(&out_v4).unwrap();
            let b = stream.clone_dtoh(&out_v6).unwrap();
            let t = stream.clone_dtoh(&out_v6t).unwrap();
            let d = stream.clone_dtoh(&out_v6d).unwrap();
            let l = stream.clone_dtoh(&out_v6tl).unwrap();
            let bb = stream.clone_dtoh(&out_v6tb).unwrap();
            let s7 = stream.clone_dtoh(&out_v7).unwrap();
            let s7t = stream.clone_dtoh(&out_v7t).unwrap();
            let s8 = stream.clone_dtoh(&out_v8).unwrap();
            let s8t = stream.clone_dtoh(&out_v8t).unwrap();
            let s9 = stream.clone_dtoh(&out_v9).unwrap();
            let s9t = stream.clone_dtoh(&out_v9t).unwrap();
            let s10 = stream.clone_dtoh(&out_v10).unwrap();
            let s10t = stream.clone_dtoh(&out_v10t).unwrap();
            let diffs = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs, 0, "{fold:?}: the v6 fork-config kernel is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_t = a.iter().zip(&t).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_t, 0, "{fold:?}: the v6t TOKS=128 tile is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_d = a.iter().zip(&d).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_d, 0, "{fold:?}: the v6d double-A tile is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_l = a.iter().zip(&l).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_l, 0, "{fold:?}: the v6tl ldmatrix-A twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_b = a.iter().zip(&bb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_b, 0, "{fold:?}: the v6tb ldmatrix-A+B twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_7 = a.iter().zip(&s7).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_7, 0, "{fold:?}: the v7 fork-style global-A twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_7t = a.iter().zip(&s7t).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_7t, 0, "{fold:?}: the v7t TOKS=128 global-A twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_8 = a.iter().zip(&s8).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_8, 0, "{fold:?}: the v8 format-rung twin (packed \
                 Q2_0 codes + PRMT decode) moves no numerics — it must be \
                 bit-identical to v4-q8 (incl. the non-disjoint-plane fold)");
            let diffs_8t = a.iter().zip(&s8t).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_8t, 0, "{fold:?}: the v8t format-rung twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_9 = a.iter().zip(&s9).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_9, 0, "{fold:?}: the v9 staged-packed twin (cp.async \
                 code stage + per-use decode) moves no numerics — it must be \
                 bit-identical to v4-q8 (incl. the non-disjoint-plane fold)");
            let diffs_9t = a.iter().zip(&s9t).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_9t, 0, "{fold:?}: the v9t staged-packed twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
            let diffs_10 = a.iter().zip(&s10).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_10, 0, "{fold:?}: the v10 L2-traffic twin (smem \
                 scale stage + transposed epilogue) moves no numerics — it \
                 must be bit-identical to v4-q8");
            let diffs_10t = a.iter().zip(&s10t).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(diffs_10t, 0, "{fold:?}: the v10t L2-traffic twin is a \
                 scheduling change only — it must be bit-identical to v4-q8");
        }

        // Absolute grounding: the same CPU reference element the T2a gate
        // pins (row 0, tok 0) — v6 must reproduce it exactly.
        let q_words = stream.clone_dtoh(&scratch.q_hi_w).unwrap();
        let s_t = stream.clone_dtoh(&scratch.s_t).unwrap();
        let qh = |tok: usize, k: usize| -> i32 {
            let w = q_words[tok * (n / 4) + k / 4];
            ((w >> (8 * (k % 4))) & 0xFF) as i8 as i32
        };
        let wbyte = |row: usize, k: usize| -> i32 {
            let pw = pos_bits[row * wpr + k / 32];
            let nw = neg_bits[row * wpr + k / 32];
            (((pw >> (k % 32)) & 1) as i32) - (((nw >> (k % 32)) & 1) as i32)
        };
        let mut o = 0f32;
        let wrow = 0usize; // the CPU-reference row (mirrors the kernel's row 0)
        for g in 0..groups {
            let mut acc = 0i32;
            for k in g * 128..(g + 1) * 128 {
                acc += wbyte(wrow, k) * qh(0, k);
            }
            o = group_scale[wrow * groups + g].mul_add(acc as f32, o);
        }
        let expected = o * s_t[0];
        for (name, launch) in [
            ("v6", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v6(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
            ("v6tl", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v6tl(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
            ("v6tb", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v6tb(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
            ("v7", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v7(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
            ("v7t", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v7t(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
            ("v9t", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v9t(&stream, &packed_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
            ("v10t", &(|o: &CudaSlice<f32>|
                kernels.launch_gemm_q8_v10t(&stream, &packed_dev, &gs_dev, &scratch, o, m, n, p, FoldMode::Fused))
                as &dyn Fn(&CudaSlice<f32>) -> Result<(), GemmI8MmaError>),
        ] {
            let o = stream.alloc_zeros::<f32>(p * m).unwrap();
            launch(&o).unwrap();
            let out = stream.clone_dtoh(&o).unwrap();
            assert_eq!(out[0], expected, "{name} GEMM CPU reference (row 0, tok 0)");
        }
    }

    /// Issue 902 T1 — the fused gate+up pair differential: v11gu/v11gut
    /// (one launch, two slabs) must equal TWO v10t launches on the
    /// respective weight sets BIT-exactly per slab (the B884 epilogue
    /// discipline: same VALUES, same per-output op order). Two DISTINCT
    /// weight fixtures prove the slabs did not cross-contaminate; ragged
    /// p=130 exercises both TOKS tail paths.
    #[cfg(feature = "prefill_mmq_v2")]
    #[test]
    fn v11_gu_pair_matches_two_v10t_bitwise() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n, p) = (256usize, 256usize, 130usize);
        let wpr = n / 32;
        let groups = n / 128;
        // mat0: the T2a fixture constants; mat1: shifted multipliers (distinct
        // weights AND distinct scales — a slab swap would fail loudly).
        let pos0: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15) as u32).collect();
        let neg0: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0xBF58476D1CE4E5B9) as u32).collect();
        let sc0: Vec<f32> =
            (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let pos1: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0x94D049BB133111EB) as u32).collect();
        let neg1: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64).wrapping_mul(0x8D2E4FB5A5B77735) as u32).collect();
        let sc1: Vec<f32> =
            (0..m * groups).map(|i| 0.002 + ((i * 29) % 100) as f32 * 0.0002).collect();
        let input: Vec<f32> =
            (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();

        let _p0_dev = stream.clone_htod(&pos0).unwrap();
        let _n0_dev = stream.clone_htod(&neg0).unwrap();
        let s0_dev = stream.clone_htod(&sc0).unwrap();
        let _p1_dev = stream.clone_htod(&pos1).unwrap();
        let _n1_dev = stream.clone_htod(&neg1).unwrap();
        let s1_dev = stream.clone_htod(&sc1).unwrap();
        let in_dev = stream.clone_htod(&input).unwrap();
        let packed0 = crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos0, &neg0, m, n);
        let packed1 = crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos1, &neg1, m, n);
        let pk0_dev = stream.clone_htod(&packed0).unwrap();
        let pk1_dev = stream.clone_htod(&packed1).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        kernels
            .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Rn)
            .unwrap();

        for fold in [FoldMode::Fused, FoldMode::Strict] {
            let ref0 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let ref1 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out0 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out1 = stream.alloc_zeros::<f32>(p * m).unwrap();
            kernels
                .launch_gemm_q8_v10t(&stream, &pk0_dev, &s0_dev, &scratch, &ref0, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v10t(&stream, &pk1_dev, &s1_dev, &scratch, &ref1, m, n, p, fold)
                .unwrap();
            kernels
                .launch_gemm_q8_v11gu_pair(
                    &stream, &pk0_dev, &s0_dev, &pk1_dev, &s1_dev, &scratch, &out0, &out1, m, n,
                    p, fold,
                )
                .unwrap();
            let (r0, r1) = (
                stream.clone_dtoh(&ref0).unwrap(),
                stream.clone_dtoh(&ref1).unwrap(),
            );
            let (o0, o1) = (
                stream.clone_dtoh(&out0).unwrap(),
                stream.clone_dtoh(&out1).unwrap(),
            );
            let d0 = r0.iter().zip(&o0).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(d0, 0, "{fold:?}: v11gu slab0 is the pair fusion of v10t — \
                 per-output op order is unchanged; it must be bit-identical");
            let d1 = r1.iter().zip(&o1).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(d1, 0, "{fold:?}: v11gu slab1 is the pair fusion of v10t — \
                 per-output op order is unchanged; it must be bit-identical");
            // v11gut (TOKS=128 geometry) + the Bench-895 repair rungs
            // (v11gs sequential, v11gq interleaved@255regs) — same refs.
            for (name, launch) in [
                (
                    "v11gut",
                    &(|o0: &CudaSlice<f32>, o1: &CudaSlice<f32>| {
                        kernels.launch_gemm_q8_v11gut_pair(
                            &stream, &pk0_dev, &s0_dev, &pk1_dev, &s1_dev, &scratch, o0, o1, m,
                            n, p, fold,
                        )
                    })
                        as &dyn Fn(&CudaSlice<f32>, &CudaSlice<f32>) -> Result<(), GemmI8MmaError>,
                ),
                (
                    "v11gs",
                    &(|o0: &CudaSlice<f32>, o1: &CudaSlice<f32>| {
                        kernels.launch_gemm_q8_v11gs_pair(
                            &stream, &pk0_dev, &s0_dev, &pk1_dev, &s1_dev, &scratch, o0, o1, m,
                            n, p, fold,
                        )
                    })
                        as &dyn Fn(&CudaSlice<f32>, &CudaSlice<f32>) -> Result<(), GemmI8MmaError>,
                ),
                (
                    "v11gq",
                    &(|o0: &CudaSlice<f32>, o1: &CudaSlice<f32>| {
                        kernels.launch_gemm_q8_v11gq_pair(
                            &stream, &pk0_dev, &s0_dev, &pk1_dev, &s1_dev, &scratch, o0, o1, m,
                            n, p, fold,
                        )
                    })
                        as &dyn Fn(&CudaSlice<f32>, &CudaSlice<f32>) -> Result<(), GemmI8MmaError>,
                ),
            ] {
                launch(&out0, &out1).unwrap();
                let (o0, o1) = (
                    stream.clone_dtoh(&out0).unwrap(),
                    stream.clone_dtoh(&out1).unwrap(),
                );
                let d0 = r0.iter().zip(&o0).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                assert_eq!(d0, 0, "{fold:?}: {name} slab0 must be bit-identical to v10t");
                let d1 = r1.iter().zip(&o1).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                assert_eq!(d1, 0, "{fold:?}: {name} slab1 must be bit-identical to v10t");
            }
        }
    }

    /// Issue 902 T1 G2 — kernel-level pair A/B at the league FFN shapes
    /// (m=17408, n=5120, p ∈ {2048, 4096}): the two-launch v10t baseline vs
    /// the fused v11gu (TOKS=64, 2 blocks/SM) and v11gut (TOKS=128, 1
    /// block/SM) pair arms. Interleaved ×9, medians; prints the fused
    /// kernels' register counts + local (spill) bytes — a local-bytes > 0
    /// re-reading is required (the launch-bounds occupancy claim spilled).
    ///
    /// Run GPU-exclusive (the Issue 834 rule); release profile only.
    #[cfg(feature = "prefill_mmq_v2")]
    #[test]
    #[ignore = "GPU perf probe: GB-class slab allocations + exclusive GPU; run with --release --ignored --nocapture"]
    fn q8_gemm_v11_pair_perf_ab_ffn() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n) = (17408usize, 5120usize);
        let wpr = n / 32;
        let groups = n / 128;
        let pos_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0x9E3779B97F4A7C15) as u32).collect();
        let neg_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0xBF58476D1CE4E5B9) as u32).collect();
        let group_scale: Vec<f32> =
            (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let pos1_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0x94D049BB133111EB) as u32).collect();
        let neg1_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0x8D2E4FB5A5B77735) as u32).collect();
        let group1_scale: Vec<f32> =
            (0..m * groups).map(|i| 0.002 + ((i * 29) % 100) as f32 * 0.0002).collect();
        let packed0 = crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos_bits, &neg_bits, m, n);
        let packed1 =
            crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos1_bits, &neg1_bits, m, n);
        let pk0_dev = stream.clone_htod(&packed0).unwrap();
        let pk1_dev = stream.clone_htod(&packed1).unwrap();
        let s0_dev = stream.clone_htod(&group_scale).unwrap();
        let s1_dev = stream.clone_htod(&group1_scale).unwrap();

        for f in [
            (&kernels.gemm_tm128v10t_q8_fused, "v10t"),
            (&kernels.gemm_tm128v11gu_q8_fused, "v11gu"),
            (&kernels.gemm_tm128v11gut_q8_fused, "v11gut"),
            (&kernels.gemm_tm128v11gs_q8_fused, "v11gs"),
            (&kernels.gemm_tm128v11gq_q8_fused, "v11gq"),
            (&kernels.gemm_tm128v10t_q8_nodecode, "v10t-nodecode"),
        ] {
            eprintln!(
                "[902-ab] {}: regs={} local_bytes={}",
                f.1,
                f.0.num_regs().unwrap_or(-1),
                f.0.local_size_bytes().unwrap_or(-1),
            );
        }

        type BenchArm<'a> = (&'a str, Box<dyn Fn() + 'a>);
        for p in [2048usize, 4096usize] {
            let input: Vec<f32> =
                (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();
            let _in_dev = stream.clone_htod(&input).unwrap();
            let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
            let out0 = stream.alloc_zeros::<f32>(p * m).unwrap();
            let out1 = stream.alloc_zeros::<f32>(p * m).unwrap();
            stream.synchronize().unwrap();

            let arms: Vec<BenchArm> = vec![
                (
                    "2x v10t",
                    Box::new({
                        let stream = &stream;
                        let kernels = &kernels;
                        let (pk0_dev, s0_dev, pk1_dev, s1_dev) =
                            (&pk0_dev, &s0_dev, &pk1_dev, &s1_dev);
                        let (scratch, out0, out1) = (&scratch, &out0, &out1);
                        move || {
                            kernels
                                .launch_gemm_q8_v10t(
                                    stream, pk0_dev, s0_dev, scratch, out0, m, n, p,
                                    FoldMode::Fused,
                                )
                                .unwrap();
                            kernels
                                .launch_gemm_q8_v10t(
                                    stream, pk1_dev, s1_dev, scratch, out1, m, n, p,
                                    FoldMode::Fused,
                                )
                                .unwrap();
                        }
                    }),
                ),
                (
                    "v11gu pair",
                    Box::new({
                        let stream = &stream;
                        let kernels = &kernels;
                        let (pk0_dev, s0_dev, pk1_dev, s1_dev) =
                            (&pk0_dev, &s0_dev, &pk1_dev, &s1_dev);
                        let (scratch, out0, out1) = (&scratch, &out0, &out1);
                        move || {
                            kernels
                                .launch_gemm_q8_v11gu_pair(
                                    stream, pk0_dev, s0_dev, pk1_dev, s1_dev, scratch, out0,
                                    out1, m, n, p, FoldMode::Fused,
                                )
                                .unwrap();
                        }
                    }),
                ),
                (
                    "v11gut pair",
                    Box::new({
                        let stream = &stream;
                        let kernels = &kernels;
                        let (pk0_dev, s0_dev, pk1_dev, s1_dev) =
                            (&pk0_dev, &s0_dev, &pk1_dev, &s1_dev);
                        let (scratch, out0, out1) = (&scratch, &out0, &out1);
                        move || {
                            kernels
                                .launch_gemm_q8_v11gut_pair(
                                    stream, pk0_dev, s0_dev, pk1_dev, s1_dev, scratch, out0,
                                    out1, m, n, p, FoldMode::Fused,
                                )
                                .unwrap();
                        }
                    }),
                ),
                (
                    "v11gs pair",
                    Box::new({
                        let stream = &stream;
                        let kernels = &kernels;
                        let (pk0_dev, s0_dev, pk1_dev, s1_dev) =
                            (&pk0_dev, &s0_dev, &pk1_dev, &s1_dev);
                        let (scratch, out0, out1) = (&scratch, &out0, &out1);
                        move || {
                            kernels
                                .launch_gemm_q8_v11gs_pair(
                                    stream, pk0_dev, s0_dev, pk1_dev, s1_dev, scratch, out0,
                                    out1, m, n, p, FoldMode::Fused,
                                )
                                .unwrap();
                        }
                    }),
                ),
                (
                    "v11gq pair",
                    Box::new({
                        let stream = &stream;
                        let kernels = &kernels;
                        let (pk0_dev, s0_dev, pk1_dev, s1_dev) =
                            (&pk0_dev, &s0_dev, &pk1_dev, &s1_dev);
                        let (scratch, out0, out1) = (&scratch, &out0, &out1);
                        move || {
                            kernels
                                .launch_gemm_q8_v11gq_pair(
                                    stream, pk0_dev, s0_dev, pk1_dev, s1_dev, scratch, out0,
                                    out1, m, n, p, FoldMode::Fused,
                                )
                                .unwrap();
                        }
                    }),
                ),
                (
                    "v10t-nodecode",
                    Box::new({
                        let stream = &stream;
                        let kernels = &kernels;
                        let (pk0_dev, s0_dev) = (&pk0_dev, &s0_dev);
                        let (scratch, out0) = (&scratch, &out0);
                        move || {
                            // Issue 903 T1: the decode-free timing probe — same
                            // geometry/loads/MMA count as v10t, VALUES GARBAGE
                            // (raw code words feed the mma). time(v10t) -
                            // time(nodecode) = the decode's issue-slot share.
                            let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
                            let cfg = LaunchConfig {
                                grid_dim: (m.div_ceil(128) as u32, p.div_ceil(128) as u32, 1),
                                block_dim: (512, 1, 1),
                                shared_mem_bytes: 50_176,
                            };
                            unsafe {
                                stream
                                    .launch_builder(&kernels.gemm_tm128v10t_q8_nodecode)
                                    .arg(pk0_dev)
                                    .arg(s0_dev)
                                    .arg(&scratch.q_hi_w)
                                    .arg(&scratch.s_t)
                                    .arg(out0)
                                    .arg(&m_i)
                                    .arg(&n_i)
                                    .arg(&p_i)
                                    .arg(&((n / 16) as i32))
                                    .arg(&((n / 128) as i32))
                                    .launch(cfg)
                                    .unwrap();
                            }
                        }
                    }),
                ),
            ];

            // Warmup + boost-clock burn (the fairness protocol).
            for _ in 0..20 {
                for (_, run) in arms.iter() {
                    run();
                }
            }
            stream.synchronize().unwrap();
            let mut xs = vec![Vec::new(); arms.len()];
            for _ in 0..9 {
                for (i, (_, run)) in arms.iter().enumerate() {
                    let t = std::time::Instant::now();
                    run();
                    stream.synchronize().unwrap();
                    xs[i].push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            let med: Vec<f64> = xs
                .iter()
                .map(|v| {
                    let mut v = v.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                })
                .collect();
            let base = med[0];
            eprintln!(
                "[902-ab] m={m} n={n} p={p}: {}",
                arms.iter()
                    .zip(&med)
                    .map(|((name, _), ms)| format!("{name}={ms:.3}ms ({:.3}x)", base / ms))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }

    /// Issue 884 T2b — kernel-level A/B at the B801/B803 ffn_gate slab
    /// (m=17408, n=5120, p=2048): the single-plane v4-q8 (T2a) vs the
    /// fork-config v6-q8 (T2b). Both arms include the quantize pre-pass and
    /// share it (identical kernels — the diff is the GEMM only). Also prints
    /// the v6 register count: `__launch_bounds__(256, 2)` caps at 128
    /// regs/thread; a value > 128 with nonzero local bytes means the 2-block
    /// occupancy claim spilled and the result needs re-reading.
    ///
    /// Run GPU-exclusive (the Issue 834 rule); release profile only.
    #[cfg(feature = "prefill_mmq_v2")]
    #[test]
    #[ignore = "GPU perf probe: 7.1 GB-class slab allocations + exclusive GPU; run with --release --ignored --nocapture"]
    fn q8_gemm_v6_perf_ab_ffn_gate() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n, p) = (17408usize, 5120usize, 2048usize);
        let wpr = n / 32;
        let groups = n / 128;
        let pos_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0x9E3779B97F4A7C15) as u32).collect();
        let neg_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0xBF58476D1CE4E5B9) as u32).collect();
        let group_scale: Vec<f32> =
            (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let input: Vec<f32> =
            (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();
        let pos_dev = stream.clone_htod(&pos_bits).unwrap();
        let neg_dev = stream.clone_htod(&neg_bits).unwrap();
        let gs_dev = stream.clone_htod(&group_scale).unwrap();
        let in_dev = stream.clone_htod(&input).unwrap();
        let packed = crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos_bits, &neg_bits, m, n);
        let packed_dev = stream.clone_htod(&packed).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        let out = stream.alloc_zeros::<f32>(p * m).unwrap();
        stream.synchronize().unwrap();

        let v6_regs = kernels
            .gemm_tm128v6_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v6_local = kernels
            .gemm_tm128v6_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);

        let time = |runs: Vec<&dyn Fn()>| -> Vec<f64> {
            // Warmup + burn: get the boost clocks up BEFORE sampling (the
            // GPU idles at 255 MHz; a short non-interleaved bench measures
            // the clock ramp, not the kernel).
            for _ in 0..20 {
                for run in &runs {
                    run();
                }
            }
            stream.synchronize().unwrap();
            // INTERLEAVED sampling: one timed sample per arm per round — a
            // clock drift hits every arm equally (the fairness protocol).
            let mut xs = vec![Vec::new(); runs.len()];
            for _ in 0..9 {
                for (i, run) in runs.iter().enumerate() {
                    let t = std::time::Instant::now();
                    run();
                    stream.synchronize().unwrap();
                    xs[i].push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            xs.iter()
                .map(|v| {
                    let mut v = v.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                })
                .collect()
        };

        let quantize = || {
            kernels
                .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Full)
                .unwrap();
        };
        let run_v4 = || {
            quantize();
            kernels
                .launch_gemm_q8(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v6 = || {
            quantize();
            kernels
                .launch_gemm_q8_v6(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v6t = || {
            quantize();
            kernels
                .launch_gemm_q8_v6t(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v6d = || {
            quantize();
            kernels
                .launch_gemm_q8_v6d(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v6tl = || {
            quantize();
            kernels
                .launch_gemm_q8_v6tl(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v6tb = || {
            quantize();
            kernels
                .launch_gemm_q8_v6tb(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v7 = || {
            quantize();
            kernels
                .launch_gemm_q8_v7(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v7t = || {
            quantize();
            kernels
                .launch_gemm_q8_v7t(
                    &stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v8 = || {
            quantize();
            kernels
                .launch_gemm_q8_v8(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v8t = || {
            quantize();
            kernels
                .launch_gemm_q8_v8t(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v9 = || {
            quantize();
            kernels
                .launch_gemm_q8_v9(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v9t = || {
            quantize();
            kernels
                .launch_gemm_q8_v9t(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v10 = || {
            quantize();
            kernels
                .launch_gemm_q8_v10(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v10t = || {
            quantize();
            kernels
                .launch_gemm_q8_v10t(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p,
                    FoldMode::Fused,
                )
                .unwrap();
        };
        let results = time(vec![
            &run_v4, &run_v6, &run_v6t, &run_v6d, &run_v6tl, &run_v6tb, &run_v7, &run_v7t,
            &run_v8, &run_v8t, &run_v9, &run_v9t, &run_v10, &run_v10t,
        ]);
        let (q8_v4, q8_v6, q8_v6t, q8_v6d, q8_v6tl, q8_v6tb, q8_v7, q8_v7t, q8_v8, q8_v8t, q8_v9, q8_v9t, q8_v10, q8_v10t) =
            (results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9], results[10], results[11], results[12], results[13]);
        let v6d_regs = kernels
            .gemm_tm128v6d_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v6d_local = kernels
            .gemm_tm128v6d_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v6t_regs = kernels
            .gemm_tm128v6t_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v6t_local = kernels
            .gemm_tm128v6t_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v6tl_regs = kernels
            .gemm_tm128v6tl_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v6tl_local = kernels
            .gemm_tm128v6tl_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v6tb_regs = kernels
            .gemm_tm128v6tb_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v6tb_local = kernels
            .gemm_tm128v6tb_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v7_regs = kernels
            .gemm_tm128v7_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v7_local = kernels
            .gemm_tm128v7_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v7t_regs = kernels
            .gemm_tm128v7t_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v7t_local = kernels
            .gemm_tm128v7t_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v8_regs = kernels
            .gemm_tm128v8_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v8_local = kernels
            .gemm_tm128v8_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v8t_regs = kernels
            .gemm_tm128v8t_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v8t_local = kernels
            .gemm_tm128v8t_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v9_regs = kernels
            .gemm_tm128v9_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v9_local = kernels
            .gemm_tm128v9_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v9t_regs = kernels
            .gemm_tm128v9t_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v9t_local = kernels
            .gemm_tm128v9t_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v10_regs = kernels
            .gemm_tm128v10_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v10_local = kernels
            .gemm_tm128v10_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        let v10t_regs = kernels
            .gemm_tm128v10t_q8_fused
            .num_regs()
            .unwrap_or(-1);
        let v10t_local = kernels
            .gemm_tm128v10t_q8_fused
            .local_size_bytes()
            .unwrap_or(-1);
        // Bit-identity witness on the bench fixture itself (one shot, every
        // v6-family arm vs v4-q8).
        let out_v4 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v6 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v6t = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v6tl = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v6tb = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v7 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v7t = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v8 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v8t = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v9 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v9t = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v10 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v10t = stream.alloc_zeros::<f32>(p * m).unwrap();
        kernels
            .launch_gemm_q8(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v4, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v6(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v6t(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6t, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v6tl(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6tl, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v6tb(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v6tb, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v7(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v7, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v7t(&stream, &pos_dev, &neg_dev, &gs_dev, &scratch, &out_v7t, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v8(&stream, &packed_dev, &gs_dev, &scratch, &out_v8, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v8t(&stream, &packed_dev, &gs_dev, &scratch, &out_v8t, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v9(&stream, &packed_dev, &gs_dev, &scratch, &out_v9, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v9t(&stream, &packed_dev, &gs_dev, &scratch, &out_v9t, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v10(&stream, &packed_dev, &gs_dev, &scratch, &out_v10, m, n, p, FoldMode::Fused)
            .unwrap();
        kernels
            .launch_gemm_q8_v10t(&stream, &packed_dev, &gs_dev, &scratch, &out_v10t, m, n, p, FoldMode::Fused)
            .unwrap();
        let a = stream.clone_dtoh(&out_v4).unwrap();
        let b = stream.clone_dtoh(&out_v6).unwrap();
        let bt = stream.clone_dtoh(&out_v6t).unwrap();
        let bl = stream.clone_dtoh(&out_v6tl).unwrap();
        let bb = stream.clone_dtoh(&out_v6tb).unwrap();
        let b7 = stream.clone_dtoh(&out_v7).unwrap();
        let b7t = stream.clone_dtoh(&out_v7t).unwrap();
        let b8 = stream.clone_dtoh(&out_v8).unwrap();
        let b8t = stream.clone_dtoh(&out_v8t).unwrap();
        let b9 = stream.clone_dtoh(&out_v9).unwrap();
        let b9t = stream.clone_dtoh(&out_v9t).unwrap();
        let b10 = stream.clone_dtoh(&out_v10).unwrap();
        let b10t = stream.clone_dtoh(&out_v10t).unwrap();
        let diffs = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&bt).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&bl).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&bb).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b7).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b7t).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b8).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b8t).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b9).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b9t).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b10).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.iter().zip(&b10t).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let nominal_tf = 2.0 * m as f64 * n as f64 * p as f64 / 1e12;
        eprintln!(
            "[i884-t2b] ffn_gate m={m} n={n} p={p}: \
             v6 regs {v6_regs}/{v6t_regs}/{v6d_regs}/{v6tl_regs}/{v6tb_regs} \
             local {v6_local}/{v6t_local}/{v6d_local}/{v6tl_local}/{v6tb_local} \
             | v7 regs {v7_regs}/{v7t_regs} local {v7_local}/{v7t_local} B \
             | v8 regs {v8_regs}/{v8t_regs} local {v8_local}/{v8t_local} B \
             | v9 regs {v9_regs}/{v9t_regs} local {v9_local}/{v9t_local} B \
             | v10 regs {v10_regs}/{v10t_regs} local {v10_local}/{v10t_local} B \
             | q8 v4 {q8_v4:.3} ms ({:.1} TF executed, {:.1}% of 660.6) \
             | q8 v6 {q8_v6:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v6t {q8_v6t:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v6d {q8_v6d:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v6tl {q8_v6tl:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v6tb {q8_v6tb:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v7 {q8_v7:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v7t {q8_v7t:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v8 {q8_v8:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v8t {q8_v8t:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v9 {q8_v9:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v9t {q8_v9t:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v10 {q8_v10:.3} ms ({:.1} TF, {:.1}%) \
             | q8 v10t {q8_v10t:.3} ms ({:.1} TF, {:.1}%) \
             | v6 {:.3}x | v6t {:.3}x | v6d {:.3}x | v6tl {:.3}x | v6tb {:.3}x | v7 {:.3}x | v7t {:.3}x | v8 {:.3}x | v8t {:.3}x | v9 {:.3}x | v9t {:.3}x | v10 {:.3}x | v10t {:.3}x | bitdiffs {diffs}",
            nominal_tf / (q8_v4 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v4 * 1e-3),
            nominal_tf / (q8_v6 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v6 * 1e-3),
            nominal_tf / (q8_v6t * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v6t * 1e-3),
            nominal_tf / (q8_v6d * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v6d * 1e-3),
            nominal_tf / (q8_v6tl * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v6tl * 1e-3),
            nominal_tf / (q8_v6tb * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v6tb * 1e-3),
            nominal_tf / (q8_v7 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v7 * 1e-3),
            nominal_tf / (q8_v7t * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v7t * 1e-3),
            nominal_tf / (q8_v8 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v8 * 1e-3),
            nominal_tf / (q8_v8 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v8 * 1e-3),
            nominal_tf / (q8_v9 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v9 * 1e-3),
            nominal_tf / (q8_v9t * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v9t * 1e-3),
            nominal_tf / (q8_v10 * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v10 * 1e-3),
            nominal_tf / (q8_v10t * 1e-3),
            100.0 * nominal_tf / 660.6 / (q8_v10t * 1e-3),
            q8_v4 / q8_v6,
            q8_v4 / q8_v6t,
            q8_v4 / q8_v6d,
            q8_v4 / q8_v6tl,
            q8_v4 / q8_v6tb,
            q8_v4 / q8_v7,
            q8_v4 / q8_v7t,
            q8_v4 / q8_v8,
            q8_v4 / q8_v8t,
            q8_v4 / q8_v9,
            q8_v4 / q8_v9t,
            q8_v4 / q8_v10,
            q8_v4 / q8_v10t,
        );
        assert_eq!(diffs, 0, "bench-fixture bit-identity v6/v7/v8/v9/v10 family vs v4-q8");
    }

    /// Plan 597 S1 rung-A occupancy probe (Issue 918, the owner-GO'd lever's
    /// cheapest first arm — addendum-2's same-silicon occupancy recipe class):
    /// the v10 body at MINB=3 → 24 warps/SM under the `__launch_bounds__(256, 3)`
    /// 85-reg cap, vs the production v10t (16 warps/SM) and v10 (16 warps/SM,
    /// TOKS=64). Registers the spill verdict (local bytes MUST be 0 — the
    /// B895 fragility class: 104 B spill → 0.55×), the median-of-9 interleaved
    /// cells at the production ffn_gate shape, and the G1 bit-identity witness
    /// vs v10t (the geometry change is load-path-only — identity holds by
    /// construction, a diff means arithmetic leaked).
    ///
    /// Run GPU-exclusive (the Issue 834 rule); release profile only.
    #[cfg(feature = "prefill_mmq_v2")]
    #[test]
    #[ignore = "GPU perf probe: 7.1 GB-class slab allocations + exclusive GPU; run with --release --ignored --nocapture"]
    fn q8_gemm_597_rung_a_occupancy_probe() {
        let Some(()) = cuda_or_skip() else {
            eprintln!("SKIP — no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = GemmTernaryI8MmaCuda::new(ctx).expect("compile");

        let (m, n, p) = (17408usize, 5120usize, 2048usize);
        let wpr = n / 32;
        let groups = n / 128;
        let pos_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0x9E3779B97F4A7C15) as u32).collect();
        let neg_bits: Vec<u32> =
            (0..m * wpr).map(|i| (i as u64 * 0xBF58476D1CE4E5B9) as u32).collect();
        let group_scale: Vec<f32> =
            (0..m * groups).map(|i| 0.001 + ((i * 17) % 100) as f32 * 0.0001).collect();
        let input: Vec<f32> =
            (0..p * n).map(|i| (((i * 37) % 251) as f32 - 120.0) * 0.02).collect();
        let gs_dev = stream.clone_htod(&group_scale).unwrap();
        let in_dev = stream.clone_htod(&input).unwrap();
        let packed = crate::prefill_cuda_mma::pack_bitplanes_to_q2(&pos_bits, &neg_bits, m, n);
        let packed_dev = stream.clone_htod(&packed).unwrap();
        let scratch = kernels.alloc_scratch(&stream, n, p).unwrap();
        let out = stream.alloc_zeros::<f32>(p * m).unwrap();
        stream.synchronize().unwrap();

        let v10_regs = kernels.gemm_tm128v10_q8_fused.num_regs().unwrap_or(-1);
        let v10_local = kernels.gemm_tm128v10_q8_fused.local_size_bytes().unwrap_or(-1);
        let v10t_regs = kernels.gemm_tm128v10t_q8_fused.num_regs().unwrap_or(-1);
        let v10t_local = kernels.gemm_tm128v10t_q8_fused.local_size_bytes().unwrap_or(-1);
        let o3_regs = kernels.gemm_tm128v10o3_q8_fused.num_regs().unwrap_or(-1);
        let o3_local = kernels.gemm_tm128v10o3_q8_fused.local_size_bytes().unwrap_or(-1);
        let o3nd_regs = kernels.gemm_tm128v10o3_q8_nodecode.num_regs().unwrap_or(-1);
        let o3nd_local = kernels.gemm_tm128v10o3_q8_nodecode.local_size_bytes().unwrap_or(-1);
        let v12_regs = kernels.gemm_tm64v12_q8_fused.num_regs().unwrap_or(-1);
        let v12_local = kernels.gemm_tm64v12_q8_fused.local_size_bytes().unwrap_or(-1);
        let v12nd_regs = kernels.gemm_tm64v12_q8_nodecode.num_regs().unwrap_or(-1);
        let v12nd_local = kernels.gemm_tm64v12_q8_nodecode.local_size_bytes().unwrap_or(-1);

        let time = |runs: Vec<&dyn Fn()>| -> Vec<f64> {
            for _ in 0..20 {
                for run in &runs {
                    run();
                }
            }
            stream.synchronize().unwrap();
            let mut xs = vec![Vec::new(); runs.len()];
            for _ in 0..9 {
                for (i, run) in runs.iter().enumerate() {
                    let t = std::time::Instant::now();
                    run();
                    stream.synchronize().unwrap();
                    xs[i].push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            xs.iter()
                .map(|v| {
                    let mut v = v.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                })
                .collect()
        };

        let quantize = || {
            kernels
                .launch_quantize_q8_div(&stream, &in_dev, &scratch, n, p, QuantDiv::Full)
                .unwrap();
        };
        let run_v10 = || {
            quantize();
            kernels
                .launch_gemm_q8_v10(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p, FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v10t = || {
            quantize();
            kernels
                .launch_gemm_q8_v10t(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p, FoldMode::Fused,
                )
                .unwrap();
        };
        let run_o3 = || {
            quantize();
            kernels
                .launch_gemm_q8_v10o3(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p, FoldMode::Fused,
                )
                .unwrap();
        };
        // Ceiling arm (timing-only, garbage values — the nodecode convention):
        // the 24-warp geometry with the decode stripped. If spill-free, its time
        // is the measured ceiling of the whole 24-warp class on this body.
        let run_o3nd = || {
            quantize();
            let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
            let cfg = LaunchConfig {
                grid_dim: (m.div_ceil(128) as u32, p.div_ceil(64) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 2 * 128 * 12 * 4 + 2 * 64 * 36 * 4 + 2 * 128 * 4,
            };
            unsafe {
                stream
                    .launch_builder(&kernels.gemm_tm128v10o3_q8_nodecode)
                    .arg(&packed_dev)
                    .arg(&gs_dev)
                    .arg(&scratch.q_hi_w)
                    .arg(&scratch.s_t)
                    .arg(&out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&p_i)
                    .arg(&((n / 16) as i32))
                    .arg(&((n / 128) as i32))
                    .launch(cfg)
                    .unwrap();
            }
        };
        // v12 — the tile-shrunk arm (Bench 933 §5/§6's one live route).
        let run_v12 = || {
            quantize();
            kernels
                .launch_gemm_q8_v12(
                    &stream, &packed_dev, &gs_dev, &scratch, &out, m, n, p, FoldMode::Fused,
                )
                .unwrap();
        };
        let run_v12nd = || {
            quantize();
            let (m_i, n_i, p_i) = (m as i32, n as i32, p as i32);
            let cfg = LaunchConfig {
                grid_dim: (m.div_ceil(64) as u32, p.div_ceil(64) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 2 * 64 * 12 * 4 + 2 * 64 * 36 * 4 + 2 * 64 * 4,
            };
            unsafe {
                stream
                    .launch_builder(&kernels.gemm_tm64v12_q8_nodecode)
                    .arg(&packed_dev)
                    .arg(&gs_dev)
                    .arg(&scratch.q_hi_w)
                    .arg(&scratch.s_t)
                    .arg(&out)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&p_i)
                    .arg(&((n / 16) as i32))
                    .arg(&((n / 128) as i32))
                    .launch(cfg)
                    .unwrap();
            }
        };
        let results = time(vec![&run_v10, &run_v10t, &run_o3, &run_o3nd, &run_v12, &run_v12nd]);
        let (t_v10, t_v10t, t_o3, t_o3nd, t_v12, t_v12nd) = (
            results[0], results[1], results[2], results[3], results[4], results[5],
        );

        // G1 witness: o3 + v12 vs v10t on the bench fixture (fresh outputs).
        let out_ref = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_o3 = stream.alloc_zeros::<f32>(p * m).unwrap();
        let out_v12 = stream.alloc_zeros::<f32>(p * m).unwrap();
        quantize();
        kernels
            .launch_gemm_q8_v10t(
                &stream, &packed_dev, &gs_dev, &scratch, &out_ref, m, n, p, FoldMode::Fused,
            )
            .unwrap();
        kernels
            .launch_gemm_q8_v10o3(
                &stream, &packed_dev, &gs_dev, &scratch, &out_o3, m, n, p, FoldMode::Fused,
            )
            .unwrap();
        kernels
            .launch_gemm_q8_v12(
                &stream, &packed_dev, &gs_dev, &scratch, &out_v12, m, n, p, FoldMode::Fused,
            )
            .unwrap();
        let a = stream.clone_dtoh(&out_ref).unwrap();
        let b = stream.clone_dtoh(&out_o3).unwrap();
        let v = stream.clone_dtoh(&out_v12).unwrap();
        let diffs = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let v12_diffs = a
            .iter()
            .zip(&v)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();

        let nominal_tf = 2.0 * m as f64 * n as f64 * p as f64 / 1e12;
        eprintln!(
            "[p597-rungA] ffn_gate m={m} n={n} p={p}: \
             v10 regs {v10_regs}/{v10t_regs}/{o3_regs}/{o3nd_regs}/{v12_regs}/{v12nd_regs} \
             local {v10_local}/{v10t_local}/{o3_local}/{o3nd_local}/{v12_local}/{v12nd_local} B \
             | v10 {t_v10:.3} ms ({:.1} TF, {:.1}% of 660.6) \
             | v10t {t_v10t:.3} ms ({:.1} TF, {:.1}%) \
             | v10o3 {t_o3:.3} ms ({:.1} TF, {:.1}%) \
             | v10o3-nodecode {t_o3nd:.3} ms ({:.1} TF, {:.1}%) \
             | v12 {t_v12:.3} ms ({:.1} TF, {:.1}%) \
             | v12-nodecode {t_v12nd:.3} ms ({:.1} TF, {:.1}%) \
             | o3/v10t {:.3}x | nodecode-o3/v10t {:.3}x | v12/v10t {:.3}x | v12nd/v10t {:.3}x \
             | bitdiffs o3 {diffs} v12 {v12_diffs}",
            nominal_tf / (t_v10 * 1e-3),
            100.0 * nominal_tf / 660.6 / (t_v10 * 1e-3),
            nominal_tf / (t_v10t * 1e-3),
            100.0 * nominal_tf / 660.6 / (t_v10t * 1e-3),
            nominal_tf / (t_o3 * 1e-3),
            100.0 * nominal_tf / 660.6 / (t_o3 * 1e-3),
            nominal_tf / (t_o3nd * 1e-3),
            100.0 * nominal_tf / 660.6 / (t_o3nd * 1e-3),
            nominal_tf / (t_v12 * 1e-3),
            100.0 * nominal_tf / 660.6 / (t_v12 * 1e-3),
            nominal_tf / (t_v12nd * 1e-3),
            100.0 * nominal_tf / 660.6 / (t_v12nd * 1e-3),
            t_v10t / t_o3,
            t_v10t / t_o3nd,
            t_v10t / t_v12,
            t_v10t / t_v12nd,
        );
        assert_eq!(diffs, 0, "rung-A v10o3 bit-identity vs v10t failed — arithmetic leaked");
        assert_eq!(v12_diffs, 0, "v12 tile-shrink bit-identity vs v10t failed — arithmetic leaked");
    }
}
