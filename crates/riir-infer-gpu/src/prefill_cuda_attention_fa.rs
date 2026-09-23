//! Plan 605 T1 — the FA-class (flash-attention, tensor-core) prefill
//! attention kernel: `att_pf_fa[_dp]` + the f32→f16 KV conversion pass
//! `kv_f32_to_f16[_dp]`.
//!
//! The incumbent family (`att_pf_mq8*`, Issues 742/898/899) is a serial
//! KV-scan with per-row scalar fma chains — DRAM/L2-latency-bound, the
//! measured pp4096 quadratic term (Issue 988 T0: 24.7% of the p=4096 pass;
//! ~10x kernel-class gap vs FA-class). This module ports the opponent's
//! production kernel class (`fattn-mma-f16`, llama.cpp @ 7dffb158d — the
//! pinned fork; the 2026-09-20 `.distill/001` intake row) to our lane:
//!
//! - **Geometry** (Issue 988 T0 / Plan 605 T0): DKQ=DV=256, n_head=24 /
//!   n_kv=4 (GQA exactly 6:1), 16 attn layers.
//! - **Config** (the reference's hd-256 `ncols=32` row): ncols1=4 q
//!   positions/block, ncols2=8 packed head lanes (6 real + 2 zero pad —
//!   the reference's GQA packing templates are {2,4,8,16}; ratio 6 pads
//!   to 8 exactly like their ratio-6 case), 128 threads = 4 warps, np=2
//!   parallel warps per Q-column group, nbatch_fa=32 KV rows per softmax
//!   rescale, full 256-dim K/V tiles, nstages=2 (cp.async pipeline),
//!   Q in registers.
//! - **Numerics class = the opponent's production path**: f16 Q/K/V/P
//!   operands (Q scaled at load; K/V via the up-front conversion pass —
//!   their "quantized caches pay a to_fp16 conversion" posture), **f32
//!   KQ accumulators**, f32 softmax, **f16 VKQ accumulators** with the
//!   two production traps vendored verbatim (`FA_KQ_MAX_OFFSET` =
//!   3.0*0.6931 — the f16-accumulator range guard, their #18606 fix;
//!   `FA_FTZ_THRESHOLD` = -20.0 — branchless flush-to-zero on the
//!   rescale). NOT bit-identical to the incumbent (mma accumulation
//!   order) — tolerance-class per the 742-splitkv re-pin precedent
//!   (Plan 605 T3).
//! - **Causal, no mask tensor**: the limit per q position is analytic
//!   (`kv < q_offset + q_pos + 1`); per-element masking to -1e30f
//!   replaces the reference's mask-tensor add (both underflow expf to
//!   exactly 0). With ncols2=8 packing the warp's 16 Q columns span
//!   exactly 2 positions and `KQ_idx` (the C-fragment i-half) IS the
//!   position index — the limits are 2 scalars per thread. Each warp's
//!   KQ_C covers its own 16-row KV half: `kv = k_VKQ_0 + (y%np)*16 +
//!   get_j(l)`.
//! - **Causal tile skip, in-kernel** (no mask-scan pre-kernel): each
//!   block stops at `ceil((q_offset + (jt+1)*ncols1)/nbatch_fa)` — a
//!   runtime loop bound, GRID unchanged → CUDA-graphs-safe (the devpos
//!   twin reads `q_offset` from the 1-element device pos buffer, the
//!   family's `*_dp` contract; the eager twin takes `base_pos`).
//! - **agate**: the per-element sigmoid output gate
//!   `out = (VKQ/rowsum) * sigmoid(gate)` in the epilogue write (the
//!   incumbent's final phase, verbatim semantics).
//! - **No stream-K / no parallel_blocks** (v1): grid = ceil(p/4) x n_kv
//!   (1024x4 at p=4096) — enough blocks to fill 128 SMs; the causal
//!   load-imbalance tail is the measured T2 question (the fixup/combine
//!   ladder is a later rung if the gate misses).
//!
//! ## The smem staging protocol (the reference's np>1 epilogue)
//!
//! The two smem regions (tile_Q + tile_V, 32x132 u32 each = 33,792 B —
//! under the 48 KB default budget, no attribute opt-in) become a
//! `[64][132]` staging area at the epilogue: warp `y` stages its VKQ
//! rows (Q columns `g*16 + m`, g = y/np) at STAGED rows `[y*16,
//! y*16+16)` — parallel warps of the same group land 16 rows apart. The
//! per-row padding (u32 128..131) carries each warp's (KQ_max,
//! KQ_rowsum) meta; the group's warp `g*np` LSE-merges the two parallel
//! metas (max via shfl_xor 16, scale = exp(max_p - max_all), combined
//! rowsum), and the final write remaps output column `jc` through
//! `jc_tile_K = (jc/16)*32 + jc%16` summing the two partials weighted by
//! their scales, dividing by the combined rowsum.
//!
//! ## The KV conversion pass
//!
//! K/V enter the kernel as f16 (the reference's contract; cp.async +
//! half the L2 bytes vs f32 tiles — the traffic-bound path's 2x). Our
//! cache is f32, so each attention call converts the live range
//! `[0, q_offset+p)` into a padded f16 scratch (`ceil(kv_len/32)*32`
//! rows, zero tail — the causal mask makes the zeros inert, so the
//! cp.async tile loads never read OOB).
//!
//! ## Provenance + license
//!
//! The fragment layer (`namespace fa_mma`) and the kernel-body structure
//! are ported from llama.cpp `fattn-mma-f16.cuh` + `mma.cuh` +
//! `cp-async.cuh` @ 7dffb158d (MIT). The fragment algebra (get_i/get_j,
//! ldmatrix addressing, mma register wiring, the trans-load destination
//! order) is copied VERBATIM — it encodes the PTX register-to-element
//! correspondence and has no independent derivation; any "simplification"
//! is a bug. The reference copy lives at `.raw/fattn/` (gitignored).
//!
//! G1: `tests/bench_946_attn_fa_g1.rs` — tolerance-class vs the serial
//! incumbent (per-element max_rel bar + causal-edge fixtures + pad-lane
//! neutrality + run-twice determinism), NOT to_bits (mma class).
//!
//! DEFAULT-ON since Plan 605 T4 (2026-09-20): the arm engages at
//! `p ≥ 128` (the engagement predicate — bench_946 crossover); the
//! KILL-SWITCH `QWEN38_PF_ATTN_FA=0|off|false` restores the incumbent
//! ladder (promotion evidence: bench_946 G1 + bench_948 the Issue-750-T3
//! per-family retention walk — 0 argmax flips × 6 families × 36 items).

#![allow(clippy::too_many_arguments)]

/// Dynamic smem: the two staging regions (tile_Q recycled as tile_K after
/// the Q registers are loaded, + tile_V) — 2 x 32 x 132 x 4 B = 33,792 B.
/// Under the 48 KB default budget — no CU_FUNC_ATTRIBUTE opt-in needed
/// (unlike the gang family's 49,440 B).
pub(crate) const ATTENTION_FA_SMEM_BYTES: usize = 2 * 32 * 132 * 4;

/// ceil to the nbatch_fa=32 row grid — the f16 scratch padding contract.
pub fn fa_scratch_rows(kv_len: usize) -> usize {
    kv_len.div_ceil(32) * 32
}

pub(crate) const ATTENTION_PREFILL_FA_CUDA_SRC: &str = r#"
// ===========================================================================
// Plan 605 T1 — att_pf_fa: the FA-class mma prefill attention kernel.
// Port of llama.cpp fattn-mma-f16 @ 7dffb158d (MIT), the wide
// cols_per_warp=16 path, fixed geometry DKQ=DV=256, ncols1=4, ncols2=8,
// nwarps=4, nbatch_fa=32. Fragment algebra VERBATIM from mma.cuh (the
// Turing branch) — see the Rust module doc. Numerics class = the
// opponent's production f16 path.
// ===========================================================================
#define FA_DKQ 256
#define FA_DV 256
#define FA_NCOLS1 4
#define FA_NCOLS2 8
#define FA_NCOLS (FA_NCOLS1 * FA_NCOLS2) /* 32 */
#define FA_NWARPS 4
#define FA_WARP_SIZE 32
#define FA_NTHREADS (FA_NWARPS * FA_WARP_SIZE) /* 128 */
#define FA_NBATCH_FA 32
#define FA_NBATCH_K2 (FA_DKQ / 2) /* 128 u32 — full 256-dim row */
#define FA_NBATCH_V2 (FA_DV / 2) /* 128 u32 — full row */
#define FA_STRIDE_TILE (FA_NBATCH_K2 + 4) /* 132 u32 — +4 bank pad */
#define FA_NBATCH_COMBINE FA_NBATCH_V2

// The two production softmax traps, verbatim (fattn-common.cuh L11/L19).
#define FA_FTZ_THRESHOLD -20.0f
#define FA_KQ_MAX_OFFSET (3.0f * 0.6931f)

namespace fa_mma {

__device__ __forceinline__ unsigned int h2_pack(unsigned short lo, unsigned short hi) {
    return (unsigned int)lo | ((unsigned int)hi << 16);
}

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

__device__ __forceinline__ unsigned short f32_to_f16_rn(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

__device__ __forceinline__ unsigned int h2_mul(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
    DATA_LAYOUT_J_MAJOR = 10,
};

struct f16x2 {};

template <int I_, int J_, typename T, data_layout ds_ = DATA_LAYOUT_I_MAJOR>
struct tile {};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
    static constexpr int         I  = I_;
    static constexpr int         J  = J_;
    static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;
    static constexpr int ne = I * J / 32;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        if (I == 8 && J == 4) {
            return threadIdx.x / 4;
        } else if (I == 8 && J == 8) {
            return threadIdx.x / 4;
        } else if (I == 16 && J == 8) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        } else if (I == 16 && J == 16) {
            return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
        } else {
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if (I == 8 && J == 4) {
            return threadIdx.x % 4;
        } else if (I == 8 && J == 8) {
            return (l * 4) + (threadIdx.x % 4);
        } else if (I == 16 && J == 8) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        } else if (I == 16 && J == 16) {
            return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
        } else {
            return -1;
        }
    }
};

// Packed-f16-pair tile, I major (J counts 32-bit pairs; ne = I*J/32).
template <int I_, int J_>
struct tile<I_, J_, f16x2, DATA_LAYOUT_I_MAJOR> {
    static constexpr int         I  = I_;
    static constexpr int         J  = J_;
    static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;
    static constexpr int ne = I * J / 32;
    unsigned int x[ne] = {0u};

    static __device__ __forceinline__ int get_i(const int l) {
        if (I == 8 && J == 8) {
            return threadIdx.x / 4;
        } else if (I == 16 && J == 4) {
            return (l * 8) + (threadIdx.x / 4);
        } else if (I == 16 && J == 8) {
            return ((l % 2) * 8) + (threadIdx.x / 4);
        } else {
            return -1;
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        if (I == 8 && J == 8) {
            return (l * 4) + (threadIdx.x % 4);
        } else if (I == 16 && J == 4) {
            return threadIdx.x % 4;
        } else if (I == 16 && J == 8) {
            return ((l / 2) * 4) + (threadIdx.x % 4);
        } else {
            return -1;
        }
    }
};

template <int I_, int J_, typename T>
struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
    static constexpr int         I  = I_;
    static constexpr int         J  = J_;
    static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;
    static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
    T x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
    }
    static __device__ __forceinline__ int get_j(const int l) {
        return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
    }
};

template <int I, int J>
static __device__ __forceinline__ tile<I, J / 2, f16x2> get_half2(const tile<I, J, float> & tile_float) {
    tile<I, J / 2, f16x2> ret;
#pragma unroll
    for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
        ret.x[l0 / 2] = h2_pack(f32_to_f16_rn(tile_float.x[l0 + 0]), f32_to_f16_rn(tile_float.x[l0 + 1]));
    }
    return ret;
}

template <data_layout dl>
static __device__ __forceinline__ void load_ldmatrix(
        tile<16, 8, f16x2, dl> & t, const unsigned int * __restrict__ xs0, const int stride) {
    unsigned int * xi = (unsigned int *) t.x;
    const unsigned int * xs = (const unsigned int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
}

// The destination order {xi[0], xi[2], xi[1], xi[3]} is VERBATIM from the
// source — do not "fix" it.
template <int I, data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_trans(
        tile<I, 8, f16x2, dl> & t, const unsigned int * __restrict__ xs0, const int stride) {
    unsigned int * xi = (unsigned int *) t.x;
    const unsigned int * xs = (const unsigned int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(xs));
}

template <data_layout dl_ab, data_layout dl_d>
static __device__ __forceinline__ void mma(
        tile<16, 16, float, dl_d> & D, const tile<16, 8, f16x2, dl_ab> & A, const tile<16, 8, f16x2, dl_ab> & B) {
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
}

static __device__ __forceinline__ void mma(
        tile<16, 8, f16x2> & D, const tile<16, 8, f16x2> & A, const tile<16, 8, f16x2> & B) {
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int       * Dxi = (int       *) D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[0]), "+r"(Dxi[1])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
}

__device__ __forceinline__ unsigned int cvta_generic_to_shared(void * generic_ptr) {
    return __cvta_generic_to_shared(generic_ptr);
}

__device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
        : : "r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

} // namespace fa_mma

using namespace fa_mma;

// ---------------------------------------------------------------------------
// The KV f32 -> f16 conversion pass over the PADDED scratch: converts rows
// [0, kv_len), zeroes the pad tail (the causal mask makes the zeros inert;
// the kernel's cp.async tile loads stay in bounds).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void kv_f32_to_f16_body(
    const float* __restrict__ k, const float* __restrict__ v,
    unsigned short* __restrict__ kh, unsigned short* __restrict__ vh,
    long kv_len, long per_tensor, long kv_elems)
{
    const long total = 2 * per_tensor;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const bool is_k = i < per_tensor;
        const long idx = is_k ? i : i - per_tensor;
        const unsigned short h16 = idx < kv_elems
            ? f32_to_f16_rn(is_k ? k[idx] : v[idx])
            : (unsigned short)0;
        if (is_k) { kh[idx] = h16; } else { vh[idx] = h16; }
    }
}

extern "C" __global__ void kv_f32_to_f16(
    const float* __restrict__ k, const float* __restrict__ v,
    unsigned short* __restrict__ kh, unsigned short* __restrict__ vh,
    int kv_len, int padded_rows, int n_kv)
{
    kv_f32_to_f16_body(k, v, kh, vh, (long)kv_len,
                       (long)padded_rows * n_kv * 256,
                       (long)kv_len * n_kv * 256);
}

extern "C" __global__ void kv_f32_to_f16_dp(
    const float* __restrict__ k, const float* __restrict__ v,
    unsigned short* __restrict__ kh, unsigned short* __restrict__ vh,
    int p, int padded_rows, int n_kv, const int* __restrict__ pos_dev)
{
    const int kv_len = *pos_dev + p;
    kv_f32_to_f16_body(k, v, kh, vh, (long)kv_len,
                       (long)padded_rows * n_kv * 256,
                       (long)kv_len * n_kv * 256);
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int fa_tid_flat() {
    return threadIdx.x + threadIdx.y * FA_WARP_SIZE;
}

// cp.async 16B-chunk tile load: 32 rows x 256 halves (= 32 x 32 chunks).
__device__ __forceinline__ void fa_load_tile(
    const unsigned short* __restrict__ kv, unsigned int* __restrict__ tile,
    int row0, long stride_kv_elems)
{
    const unsigned int tile32 = cvta_generic_to_shared(tile);
#pragma unroll
    for (int c = fa_tid_flat(); c < FA_NBATCH_FA * 32; c += FA_NTHREADS) {
        const int i = c / 32;
        const int ck = c % 32;
        cp_async_cg_16(
            tile32 + (unsigned int)(i * FA_STRIDE_TILE) * 4u + (unsigned int)(ck * 16),
            (const void*)(kv + ((long)(row0 + i)) * stride_kv_elems + (long)(ck * 8)));
    }
}

// ---------------------------------------------------------------------------
// The kernel body — the reference's process_tile + iter (wide path),
// inlined, our tensors + analytic causal + sigmoid gate epilogue.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void att_pf_fa_body(
    const float* __restrict__ query,        /* [p, n_head, 256] */
    const unsigned short* __restrict__ key_h, /* [padded, n_kv, 256] f16 */
    const unsigned short* __restrict__ val_h, /* [padded, n_kv, 256] f16 */
    const float* __restrict__ gate,         /* [p, n_head, 256] */
    float* __restrict__ attn_out,           /* [p, n_head, 256] */
    int n_head, int n_kv, int p, float scale, int q_offset)
{
    typedef fa_mma::tile<16, 8, fa_mma::f16x2> T_AB;        /* KQ/VKQ A+B, VKQ C */
    typedef fa_mma::tile<16, 16, float> T_C_KQ;             /* KQ f32 accumulator */

    constexpr int cols_per_warp = 16;
    constexpr int cols_per_thread = 2;
    constexpr int np = FA_NWARPS * cols_per_warp / FA_NCOLS; /* 2 */

    const int jt = blockIdx.x;           /* q tile: positions [jt*4, jt*4+4) */
    const int kv_group = blockIdx.y;     /* [0, n_kv) */
    const int gqa_ratio = n_head / n_kv; /* 6 */

    /* Our cache layout is [seq, n_kv, 256] — this group's K/V slice lives
     * at +kv_group*256 within every row (stride n_kv*256 between rows). */
    key_h += (long)kv_group * 256;
    val_h += (long)kv_group * 256;

    extern __shared__ unsigned int fa_smem[];
    unsigned int* tile_Q = fa_smem;                          /* also tile_K */
    unsigned int* tile_V = fa_smem + FA_NCOLS * FA_STRIDE_TILE;

    const int stride_kv_elems = n_kv * 256; /* f16 elems per kv row */

    // Per-thread softmax state (per POSITION: KQ_idx == the C i-half).
    float KQ_rowsum[cols_per_thread] = {0.0f, 0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -3.402823466e38f / 2.0f;
    }

    // ---- Stage Q: f32 pairs x scale -> f16 pairs into tile_Q. jc = j*8 + c
    // (j = position-in-tile, c = head lane); lanes >= gqa_ratio and
    // positions >= p are zero-filled (the reference's guard semantics).
    {
        const float2* q2 = (const float2*)query;
#pragma unroll 4
        for (int idx = fa_tid_flat(); idx < FA_NCOLS * 128; idx += FA_NTHREADS) {
            const int jc = idx / 128;
            const int k2 = idx % 128;
            const int j = jc / FA_NCOLS2;
            const int c = jc % FA_NCOLS2;
            unsigned int packed = 0u;
            if (c < gqa_ratio && jt * FA_NCOLS1 + j < p) {
                const long o = (long)(jt * FA_NCOLS1 + j) * ((long)n_head * 128)
                             + (long)(kv_group * gqa_ratio + c) * 128 + k2;
                const float2 qq = q2[o];
                packed = fa_mma::h2_pack(fa_mma::f32_to_f16_rn(qq.x * scale),
                                         fa_mma::f32_to_f16_rn(qq.y * scale));
            }
            tile_Q[jc * FA_STRIDE_TILE + k2] = packed;
        }
        __syncthreads();
    }

    // ---- Q into registers (Q_in_reg: ldmatrix x4 per 8-pair k chunk).
    T_AB Q_B[16];
    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;
#pragma unroll
        for (int k0 = 0; k0 < 128; k0 += 8) {
            fa_mma::load_ldmatrix(Q_B[k0 / 8], tile_Q + j0 * FA_STRIDE_TILE + k0,
                                  FA_STRIDE_TILE);
        }
        __syncthreads(); /* tile_Q region becomes tile_K below */
    }

    // ---- Causal bounds (runtime; graphs-safe — grid unchanged).
    const int kv_len = q_offset + p;
    const int ntiles_kv = (kv_len + FA_NBATCH_FA - 1) / FA_NBATCH_FA;
    const int kb_causal = (q_offset + (jt + 1) * FA_NCOLS1 + FA_NBATCH_FA - 1)
                        / FA_NBATCH_FA;
    const int kb0_stop = kb_causal < ntiles_kv ? kb_causal : ntiles_kv;

    // This warp's two positions (the 16 Q columns = 2 positions x 8 lanes)
    // and its KV-half base within each 32-row tile.
    const int j0 = (threadIdx.y / np) * cols_per_warp;
    const int q_pos0 = jt * FA_NCOLS1 + j0 / FA_NCOLS2;     /* KQ_idx 0 */
    const int q_pos1 = q_pos0 + 1;                          /* KQ_idx 1 */
    const int kv_half = (threadIdx.y % np) * 16;            /* this warp's rows */

    // ---- Preload K tile 0 (cp.async into the tile_K region).
    fa_load_tile(key_h, tile_Q, 0, stride_kv_elems);

    T_AB VKQ_C[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        VKQ_C[i].x[0] = 0u; VKQ_C[i].x[1] = 0u;
        VKQ_C[i].x[2] = 0u; VKQ_C[i].x[3] = 0u;
    }

    // ---- The KV loop (the reference's iter, nstages=2 discipline).
    for (int kb0 = 0; kb0 < kb0_stop; ++kb0) {
        const int k_VKQ_0 = kb0 * FA_NBATCH_FA;
        T_C_KQ KQ_C;

        // Wait K(kb0); issue V(kb0).
        fa_mma::cp_async_wait_all();
        __syncthreads();
        fa_load_tile(val_h, tile_V, k_VKQ_0, stride_kv_elems);

        // KQ = Q @ K^T over this warp's 16 KV rows (f32 accumulators).
#pragma unroll
        for (int l = 0; l < 8; ++l) { KQ_C.x[l] = 0.0f; }
        {
            const int i_KQ_0 = kv_half;
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < 128; k_KQ_0 += 8) {
                T_AB K_A;
                fa_mma::load_ldmatrix(K_A, tile_Q + i_KQ_0 * FA_STRIDE_TILE + k_KQ_0,
                                      FA_STRIDE_TILE);
                fa_mma::mma(KQ_C, Q_B[k_KQ_0 / 8], K_A);
            }
        }

        // Causal mask (replaces the mask-tensor add): element (m, n) with
        // n = this warp's kv rows: kv = k_VKQ_0 + kv_half + get_j(l); the
        // position = the i-half (l/2)%2. -1e30f underflows expf to 0.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int kv = k_VKQ_0 + kv_half + KQ_C.get_j(l);
            const int qi = (l / 2) % 2;
            const int lim = q_offset + (qi ? q_pos1 : q_pos0) + 1;
            if (kv >= lim) KQ_C.x[l] = -1e30f;
        }

        // Softmax: new max (the f16-accumulator range offset, verbatim).
        float KQ_max_new[cols_per_thread];
        KQ_max_new[0] = KQ_max[0];
        KQ_max_new[1] = KQ_max[1];
        float KQ_rowsum_add[cols_per_thread] = {0.0f, 0.0f};
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int qi = (l / 2) % 2;
            KQ_max_new[qi] = fmaxf(KQ_max_new[qi], KQ_C.x[l] + FA_KQ_MAX_OFFSET);
        }
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max_new[col] = fmaxf(KQ_max_new[col],
                __shfl_xor_sync(0xFFFFFFFFu, KQ_max_new[col], 2));
            KQ_max_new[col] = fmaxf(KQ_max_new[col],
                __shfl_xor_sync(0xFFFFFFFFu, KQ_max_new[col], 1));
        }

        // exp + rowsum (masked elements underflow to exactly 0).
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int qi = (l / 2) % 2;
            KQ_C.x[l] = expf(KQ_C.x[l] - KQ_max_new[qi]);
            KQ_rowsum_add[qi] += KQ_C.x[l];
        }

        // Rescale (the FTZ trap, verbatim) + the running merge.
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];
            *((unsigned int*)&KQ_max_scale[col]) *=
                (KQ_max_diff >= FA_FTZ_THRESHOLD) ? 1u : 0u;
            KQ_rowsum[col] = KQ_max_scale[col] * KQ_rowsum[col] + KQ_rowsum_add[col];
        }
        // Rescale the f16 VKQ accumulators (per position = the element
        // i-half; both dim-halves l0 — the reference's exact indexing).
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const unsigned int h2s = fa_mma::h2_pack(
                fa_mma::f32_to_f16_rn(KQ_max_scale[col]),
                fa_mma::f32_to_f16_rn(KQ_max_scale[col]));
#pragma unroll
            for (int i = 0; i < 16; ++i) {
#pragma unroll
                for (int l0 = 0; l0 < 4; l0 += 2) {
                    VKQ_C[i].x[l0 + col] = fa_mma::h2_mul(VKQ_C[i].x[l0 + col], h2s);
                }
            }
        }

        // P -> f16 B fragments (this warp's 16 kv rows).
        T_AB B;
        B = fa_mma::get_half2(KQ_C);

        // Wait V(kb0); issue K(kb0+1) into the tile_K region (dead now).
        fa_mma::cp_async_wait_all();
        __syncthreads();
        if (kb0 + 1 < kb0_stop) {
            fa_load_tile(key_h, tile_Q, (kb0 + 1) * FA_NBATCH_FA, stride_kv_elems);
        }

        // VKQ += P @ V — the swapped wide mma: A = P, B = V rows via the
        // TRANS ldmatrix (the reference's exact offset algebra).
        {
            const int k00 = kv_half / 2; /* (y%np)*T_A_VKQ::J */
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < FA_DV; i_VKQ_0 += 16) {
                T_AB A;
                fa_mma::load_ldmatrix_trans(A, tile_V + 2 * k00 * FA_STRIDE_TILE
                                                 + (i_VKQ_0 / 2), FA_STRIDE_TILE);
                fa_mma::mma(VKQ_C[i_VKQ_0 / 16], B, A);
            }
        }
    }

    // ---- Final rowsum reduce (offsets {2,1}: the 4 threads per position).
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFFu, KQ_rowsum[col], 2);
        KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFFu, KQ_rowsum[col], 1);
    }

    // The nstages>1 epilogue race guard (the reference's condition
    // nwarps*cols_per_warp > nbatch_fa: 64 > 32 — required).
    __syncthreads();

    // ---- Stage VKQ into the [64][132] area: warp y writes STAGED rows
    // [y*16, y*16+16) (its Q cols are (y/np)*16 + m — remapped at read).
#pragma unroll
    for (int i = 0; i < 16; ++i) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int j = threadIdx.y * cols_per_warp + T_AB::get_i(l);
            const int k = i * 8 + T_AB::get_j(l);
            tile_Q[j * FA_STRIDE_TILE + k] = VKQ_C[i].x[l];
        }
    }

    // Per-row meta (KQ_max, KQ_rowsum) into the row padding — two floats
    // per staged row, from the 16 threads with threadIdx.x%4 < 2.
    {
        const int jc_cwm = threadIdx.y * cols_per_warp
                         + (((threadIdx.x % 4) % 2) * 8) + threadIdx.x / 4;
        if (threadIdx.x % 4 < cols_per_thread) {
            float* m = (float*)(tile_Q + jc_cwm * FA_STRIDE_TILE + FA_NBATCH_COMBINE);
            m[0] = KQ_max[threadIdx.x % 2];
            m[1] = KQ_rowsum[threadIdx.x % 2];
        }
    }
    __syncthreads();

    // ---- The np>1 LSE combine (warps y%np==0; the others just sync).
    if (threadIdx.y % np == 0) {
        const int jc_meta = threadIdx.y * cols_per_warp + threadIdx.x;
        float* meta_ptr = (float*)(tile_Q + jc_meta * FA_STRIDE_TILE + FA_NBATCH_COMBINE);
        const float meta_max = meta_ptr[0];
        const float meta_sum = meta_ptr[1];

        float KQ_cmn = meta_max;
        KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFFu, KQ_cmn, 16));

        const float KQ_cms = expf(meta_max - KQ_cmn);
        float KQ_crs = KQ_cms * meta_sum;
        KQ_crs += __shfl_xor_sync(0xFFFFFFFFu, KQ_crs, 16);

        __syncthreads();
        meta_ptr[0] = KQ_cms;
        meta_ptr[1] = KQ_crs;
    } else {
        __syncthreads();
    }
    __syncthreads();

    // ---- Final write (warps y%np==0): remap output col jc through the
    // staged layout, merge the two partials weighted by their scales,
    // divide by the combined rowsum, apply the sigmoid gate.
    if (threadIdx.y % np == 0) {
#pragma unroll 4
        for (int jc0 = 0; jc0 < FA_NCOLS; jc0 += FA_NWARPS / np) {
            const int jc_dst = jc0 + (threadIdx.y / np);
            const int j_dst = jc_dst / FA_NCOLS2;
            const int c_dst = jc_dst % FA_NCOLS2;
            if (jt * FA_NCOLS1 + j_dst >= p || c_dst >= gqa_ratio) continue;

            const int jc_tile_K = (jc_dst / cols_per_warp) * (np * cols_per_warp)
                                + jc_dst % cols_per_warp;
            const float* meta_j = (const float*)(tile_Q + jc_tile_K * FA_STRIDE_TILE
                                                 + FA_NBATCH_COMBINE);
            for (int k = threadIdx.x; k < 128; k += FA_WARP_SIZE) {
                float accx = 0.0f;
                float accy = 0.0f;
#pragma unroll
                for (int ip = 0; ip < np; ++ip) {
                    const float scale_p = meta_j[ip * cols_per_warp * FA_STRIDE_TILE + 0];
                    const unsigned int h2 = tile_Q[(jc_tile_K + ip * cols_per_warp)
                                                   * FA_STRIDE_TILE + k];
                    accx += fa_mma::f16_to_f32_exact((unsigned short)(h2 & 0xFFFFu)) * scale_p;
                    accy += fa_mma::f16_to_f32_exact((unsigned short)(h2 >> 16)) * scale_p;
                }
                const float rowsum_j = meta_j[1];
                const long off = ((long)(jt * FA_NCOLS1 + j_dst) * n_head
                                 + (kv_group * gqa_ratio + c_dst)) * 256 + 2 * k;
                const float sigx = 1.0f / (1.0f + expf(0.0f - gate[off]));
                const float sigy = 1.0f / (1.0f + expf(0.0f - gate[off + 1]));
                attn_out[off] = (accx / rowsum_j) * sigx;
                attn_out[off + 1] = (accy / rowsum_j) * sigy;
            }
        }
    }
}

// The eager entry (host-provided base_pos).
extern "C" __global__ void __launch_bounds__(128, 2) att_pf_fa(
    const float* __restrict__ query,
    const unsigned short* __restrict__ key_h,
    const unsigned short* __restrict__ val_h,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int n_head, int n_kv, int p, float scale, int q_offset)
{
    att_pf_fa_body(query, key_h, val_h, gate, attn_out, n_head, n_kv, p,
                   scale, q_offset);
}

// The CUDA-graph twin: q_offset from the 1-element device pos buffer.
extern "C" __global__ void __launch_bounds__(128, 2) att_pf_fa_dp(
    const float* __restrict__ query,
    const unsigned short* __restrict__ key_h,
    const unsigned short* __restrict__ val_h,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int n_head, int n_kv, int p, float scale,
    const int* __restrict__ pos_dev)
{
    const int q_offset = *pos_dev;
    att_pf_fa_body(query, key_h, val_h, gate, attn_out, n_head, n_kv, p,
                   scale, q_offset);
}
"#;
